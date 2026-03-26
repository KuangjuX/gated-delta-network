"""
PyTorch binding for CUDA GDN Decode kernel.
Uses torch.utils.cpp_extension to compile the CUDA kernel at import time.
"""

import os
import torch
from torch.utils.cpp_extension import load_inline

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define K_DIM 128
#define VEC_SIZE 4
#define NUM_WARPS 4
#define BLOCK_THREADS (NUM_WARPS * 32)

template<int BLOCK_V>
__global__ void gdn_decode_kernel(
    const __nv_bfloat16* __restrict__ q_ptr,
    const __nv_bfloat16* __restrict__ k_ptr,
    const __nv_bfloat16* __restrict__ v_ptr,
    const float* __restrict__ state_ptr,
    __nv_bfloat16* __restrict__ output_ptr,
    float* __restrict__ new_state_ptr,
    const float* __restrict__ A_log_ptr,
    const __nv_bfloat16* __restrict__ a_ptr,
    const float* __restrict__ dt_bias_ptr,
    const __nv_bfloat16* __restrict__ b_ptr,
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

    float A_log_val = A_log_ptr[i_hv];
    float dt_bias_val = dt_bias_ptr[i_hv];
    float a_val = __bfloat162float(a_ptr[i_n * HV + i_hv]);
    float b_val = __bfloat162float(b_ptr[i_n * HV + i_hv]);

    float x = a_val + dt_bias_val;
    float softplus_x = (x <= 20.0f) ? logf(1.0f + expf(x)) : x;
    float g = -expf(A_log_val) * softplus_x;
    float decay = expf(g);
    float beta_val = 1.0f / (1.0f + expf(-b_val));

    float q_reg[VEC_SIZE], k_reg[VEC_SIZE];
    const int qk_base = i_n * H * K_DIM + i_h * K_DIM + lane_id * VEC_SIZE;

    #pragma unroll
    for (int i = 0; i < VEC_SIZE; i++) {
        q_reg[i] = __bfloat162float(q_ptr[qk_base + i]);
        k_reg[i] = __bfloat162float(k_ptr[qk_base + i]);
    }

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

    const int rows_per_warp = BLOCK_V / NUM_WARPS;
    const long long s_base = (long long)(i_n * HV + i_hv) * V * K_DIM;

    #pragma unroll
    for (int r = 0; r < rows_per_warp; r++) {
        int v_idx = i_v * BLOCK_V + warp_id * rows_per_warp + r;
        if (v_idx >= V) continue;

        float h_reg[VEC_SIZE];
        long long s_offset = s_base + (long long)v_idx * K_DIM + lane_id * VEC_SIZE;
        #pragma unroll
        for (int i = 0; i < VEC_SIZE; i++) {
            h_reg[i] = state_ptr[s_offset + i];
        }

        #pragma unroll
        for (int i = 0; i < VEC_SIZE; i++) {
            h_reg[i] *= decay;
        }

        float sum_hk = 0.0f;
        #pragma unroll
        for (int i = 0; i < VEC_SIZE; i++) {
            sum_hk += h_reg[i] * k_reg[i];
        }
        #pragma unroll
        for (int offset = 16; offset >= 1; offset >>= 1) {
            sum_hk += __shfl_xor_sync(0xffffffff, sum_hk, offset);
        }

        float v_val = __bfloat162float(v_ptr[i_n * HV * V + i_hv * V + v_idx]);
        float v_new = (v_val - sum_hk) * beta_val;

        #pragma unroll
        for (int i = 0; i < VEC_SIZE; i++) {
            h_reg[i] += k_reg[i] * v_new;
        }

        float sum_hq = 0.0f;
        #pragma unroll
        for (int i = 0; i < VEC_SIZE; i++) {
            sum_hq += h_reg[i] * q_reg[i];
        }
        #pragma unroll
        for (int offset = 16; offset >= 1; offset >>= 1) {
            sum_hq += __shfl_xor_sync(0xffffffff, sum_hq, offset);
        }

        if (lane_id == 0) {
            output_ptr[i_n * HV * V + i_hv * V + v_idx] = __float2bfloat16(sum_hq);
        }

        long long ns_offset = (long long)(i_n * HV + i_hv) * V * K_DIM + (long long)v_idx * K_DIM + lane_id * VEC_SIZE;
        #pragma unroll
        for (int i = 0; i < VEC_SIZE; i++) {
            new_state_ptr[ns_offset + i] = h_reg[i];
        }
    }
}

void gdn_decode_cuda(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor state,
    torch::Tensor output, torch::Tensor new_state,
    torch::Tensor A_log, torch::Tensor a, torch::Tensor dt_bias, torch::Tensor b,
    float scale, int block_v
) {
    int B = q.size(0);
    int H = q.size(1);
    int K = q.size(2);
    int HV = v.size(1);
    int V = v.size(2);

    int num_v_tiles = V / block_v;
    int grid_size = B * HV * num_v_tiles;
    dim3 grid(grid_size);
    dim3 block(128);

    auto q_ptr = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr());
    auto k_ptr = reinterpret_cast<const __nv_bfloat16*>(k.data_ptr());
    auto v_ptr = reinterpret_cast<const __nv_bfloat16*>(v.data_ptr());
    auto state_ptr = state.data_ptr<float>();
    auto output_ptr = reinterpret_cast<__nv_bfloat16*>(output.data_ptr());
    auto new_state_ptr = new_state.data_ptr<float>();
    auto A_log_p = A_log.data_ptr<float>();
    auto a_ptr = reinterpret_cast<const __nv_bfloat16*>(a.data_ptr());
    auto dt_bias_p = dt_bias.data_ptr<float>();
    auto b_ptr = reinterpret_cast<const __nv_bfloat16*>(b.data_ptr());

    if (block_v == 8) {
        gdn_decode_kernel<8><<<grid, block>>>(
            q_ptr, k_ptr, v_ptr, state_ptr, output_ptr, new_state_ptr,
            A_log_p, a_ptr, dt_bias_p, b_ptr, scale, B, H, HV, V);
    } else if (block_v == 16) {
        gdn_decode_kernel<16><<<grid, block>>>(
            q_ptr, k_ptr, v_ptr, state_ptr, output_ptr, new_state_ptr,
            A_log_p, a_ptr, dt_bias_p, b_ptr, scale, B, H, HV, V);
    } else {
        gdn_decode_kernel<32><<<grid, block>>>(
            q_ptr, k_ptr, v_ptr, state_ptr, output_ptr, new_state_ptr,
            A_log_p, a_ptr, dt_bias_p, b_ptr, scale, B, H, HV, V);
    }
}
"""

_CPP_SRC = r"""
void gdn_decode_cuda(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor state,
    torch::Tensor output, torch::Tensor new_state,
    torch::Tensor A_log, torch::Tensor a, torch::Tensor dt_bias, torch::Tensor b,
    float scale, int block_v);
"""

_module = None

def _get_module():
    global _module
    if _module is None:
        _module = load_inline(
            name="gdn_decode_cuda",
            cpp_sources=[_CPP_SRC],
            cuda_sources=[_CUDA_SRC],
            functions=["gdn_decode_cuda"],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
    return _module


def kernel(q, k, v, state, A_log, a, dt_bias, b, scale, output, new_state):
    """GDN decode entry point (DPS style) using CUDA kernel."""
    B_size = q.shape[0]
    H = q.shape[2]
    K = q.shape[3]
    HV = v.shape[2]
    V = v.shape[3]

    if isinstance(scale, torch.Tensor):
        scale = scale.item()

    if B_size <= 64:
        block_v = 16
    elif B_size <= 256:
        block_v = 16
    else:
        block_v = 8

    mod = _get_module()

    q_c = q.reshape(B_size, H, K).contiguous()
    k_c = k.reshape(B_size, H, K).contiguous()
    v_c = v.reshape(B_size, HV, V).contiguous()
    a_c = a.reshape(B_size, HV).contiguous()
    b_c = b.reshape(B_size, HV).contiguous()

    mod.gdn_decode_cuda(
        q_c, k_c, v_c, state,
        output.view(B_size, HV, V), new_state,
        A_log, a_c, dt_bias, b_c,
        scale, block_v,
    )
