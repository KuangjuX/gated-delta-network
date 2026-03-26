/*
 * CUDA GDN Decode Kernel - gdn_decode_qk4_v8_d128_k_last
 *
 * Gated Delta Net decode: single-token, GVA config, k-last state.
 * Each block handles one (batch, v_head, v_tile) combination.
 * Each warp processes BLOCK_V/NUM_WARPS V-rows.
 * 32 threads per warp cover K=128 (4 elements/thread).
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define K_DIM 128
#define VEC_SIZE 4    // K_DIM / 32 threads
#define NUM_WARPS 4
#define BLOCK_THREADS (NUM_WARPS * 32)

template<int BLOCK_V>
__global__ void gdn_decode_kernel(
    const __nv_bfloat16* __restrict__ q_ptr,    // [B, H, K]
    const __nv_bfloat16* __restrict__ k_ptr,    // [B, H, K]
    const __nv_bfloat16* __restrict__ v_ptr,    // [B, HV, V]
    const float* __restrict__ state_ptr,         // [B, HV, V, K]
    __nv_bfloat16* __restrict__ output_ptr,     // [B, HV, V]
    float* __restrict__ new_state_ptr,           // [B, HV, V, K]
    const float* __restrict__ A_log_ptr,         // [HV]
    const __nv_bfloat16* __restrict__ a_ptr,    // [B, HV]
    const float* __restrict__ dt_bias_ptr,       // [HV]
    const __nv_bfloat16* __restrict__ b_ptr,    // [B, HV]
    float scale,
    int B, int H, int HV, int V
) {
    const int tidx = threadIdx.x;
    const int lane_id = tidx & 31;
    const int warp_id = tidx >> 5;

    const int pid = blockIdx.x;
    const int num_v_tiles = V / BLOCK_V;
    const int i_v = pid % num_v_tiles;
    const int tmp = pid / num_v_tiles;
    const int i_hv = tmp % HV;
    const int i_n = tmp / HV;
    const int i_h = i_hv * H / HV;

    // Load gating parameters
    float A_log_val = A_log_ptr[i_hv];
    float dt_bias_val = dt_bias_ptr[i_hv];
    float a_val = __bfloat162float(a_ptr[i_n * HV + i_hv]);
    float b_val = __bfloat162float(b_ptr[i_n * HV + i_hv]);

    float x = a_val + dt_bias_val;
    float softplus_x = (x <= 20.0f) ? logf(1.0f + expf(x)) : x;
    float g = -expf(A_log_val) * softplus_x;
    float decay = expf(g);
    float beta_val = 1.0f / (1.0f + expf(-b_val));

    // Load q, k vectors (4 elements per thread, coalesced)
    float q_reg[VEC_SIZE], k_reg[VEC_SIZE];
    const int qk_base = i_n * H * K_DIM + i_h * K_DIM + lane_id * VEC_SIZE;

    #pragma unroll
    for (int i = 0; i < VEC_SIZE; i++) {
        q_reg[i] = __bfloat162float(q_ptr[qk_base + i]);
        k_reg[i] = __bfloat162float(k_ptr[qk_base + i]);
    }

    // L2 normalize q (with scale) and k
    float q_sq = 0.0f, k_sq = 0.0f;
    #pragma unroll
    for (int i = 0; i < VEC_SIZE; i++) {
        q_sq += q_reg[i] * q_reg[i];
        k_sq += k_reg[i] * k_reg[i];
    }
    #pragma unroll
    for (int offset = 16; offset >= 1; offset >>= 1) {
        q_sq += __shfl_xor_sync(0xffffffff, q_sq, offset);
        k_sq += __shfl_xor_sync(0xffffffff, k_sq, offset);
    }
    float q_scale = rsqrtf(q_sq + 1e-6f) * scale;
    float k_scale = rsqrtf(k_sq + 1e-6f);
    #pragma unroll
    for (int i = 0; i < VEC_SIZE; i++) {
        q_reg[i] *= q_scale;
        k_reg[i] *= k_scale;
    }

    // Each warp processes BLOCK_V/NUM_WARPS V-rows
    const int rows_per_warp = BLOCK_V / NUM_WARPS;
    const long long s_base = (long long)(i_n * HV + i_hv) * V * K_DIM;

    #pragma unroll
    for (int r = 0; r < rows_per_warp; r++) {
        int v_idx = i_v * BLOCK_V + warp_id * rows_per_warp + r;
        if (v_idx >= V) continue;

        // Load state row [K] with vectorized access
        float h_reg[VEC_SIZE];
        long long s_offset = s_base + (long long)v_idx * K_DIM + lane_id * VEC_SIZE;
        #pragma unroll
        for (int i = 0; i < VEC_SIZE; i++) {
            h_reg[i] = state_ptr[s_offset + i];
        }

        // Step 1: Decay
        #pragma unroll
        for (int i = 0; i < VEC_SIZE; i++) {
            h_reg[i] *= decay;
        }

        // Step 2: Prediction sum(h * k)
        float sum_hk = 0.0f;
        #pragma unroll
        for (int i = 0; i < VEC_SIZE; i++) {
            sum_hk += h_reg[i] * k_reg[i];
        }
        #pragma unroll
        for (int offset = 16; offset >= 1; offset >>= 1) {
            sum_hk += __shfl_xor_sync(0xffffffff, sum_hk, offset);
        }

        // Step 3: Delta + gate
        float v_val = __bfloat162float(v_ptr[i_n * HV * V + i_hv * V + v_idx]);
        float v_new = (v_val - sum_hk) * beta_val;

        // Step 4: Rank-1 update
        #pragma unroll
        for (int i = 0; i < VEC_SIZE; i++) {
            h_reg[i] += k_reg[i] * v_new;
        }

        // Step 5: Output sum(h * q)
        float sum_hq = 0.0f;
        #pragma unroll
        for (int i = 0; i < VEC_SIZE; i++) {
            sum_hq += h_reg[i] * q_reg[i];
        }
        #pragma unroll
        for (int offset = 16; offset >= 1; offset >>= 1) {
            sum_hq += __shfl_xor_sync(0xffffffff, sum_hq, offset);
        }

        // Store output
        if (lane_id == 0) {
            output_ptr[i_n * HV * V + i_hv * V + v_idx] = __float2bfloat16(sum_hq);
        }

        // Store new state
        long long ns_offset = (long long)(i_n * HV + i_hv) * V * K_DIM + (long long)v_idx * K_DIM + lane_id * VEC_SIZE;
        #pragma unroll
        for (int i = 0; i < VEC_SIZE; i++) {
            new_state_ptr[ns_offset + i] = h_reg[i];
        }
    }
}

// Exposed entry points for different BLOCK_V configs
extern "C" {
    __global__ void gdn_decode_bv8(
        const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
        const float* state, __nv_bfloat16* output, float* new_state,
        const float* A_log, const __nv_bfloat16* a, const float* dt_bias, const __nv_bfloat16* b,
        float scale, int B, int H, int HV, int V
    ) { gdn_decode_kernel<8>(q, k, v, state, output, new_state, A_log, a, dt_bias, b, scale, B, H, HV, V); }

    __global__ void gdn_decode_bv16(
        const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
        const float* state, __nv_bfloat16* output, float* new_state,
        const float* A_log, const __nv_bfloat16* a, const float* dt_bias, const __nv_bfloat16* b,
        float scale, int B, int H, int HV, int V
    ) { gdn_decode_kernel<16>(q, k, v, state, output, new_state, A_log, a, dt_bias, b, scale, B, H, HV, V); }

    __global__ void gdn_decode_bv32(
        const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
        const float* state, __nv_bfloat16* output, float* new_state,
        const float* A_log, const __nv_bfloat16* a, const float* dt_bias, const __nv_bfloat16* b,
        float scale, int B, int H, int HV, int V
    ) { gdn_decode_kernel<32>(q, k, v, state, output, new_state, A_log, a, dt_bias, b, scale, B, H, HV, V); }
}
