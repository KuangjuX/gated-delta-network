"""Sweep BLOCK_V and num_warps configurations for GDN decode."""

import torch
import triton
import triton.language as tl


@triton.jit
def _gdn_decode_kernel(
    q_ptr, k_ptr, v_ptr, state_ptr,
    output_ptr, new_state_ptr,
    A_log_ptr, a_ptr, dt_bias_ptr, b_ptr,
    scale,
    B, H: tl.constexpr, HV: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    num_v_tiles = V // BLOCK_V
    i_v = pid % num_v_tiles
    tmp = pid // num_v_tiles
    i_hv = tmp % HV
    i_n = tmp // HV

    i_h = i_hv * H // HV

    A_log_val = tl.load(A_log_ptr + i_hv)
    dt_bias_val = tl.load(dt_bias_ptr + i_hv)
    a_val = tl.load(a_ptr + i_n * HV + i_hv).to(tl.float32)
    b_val = tl.load(b_ptr + i_n * HV + i_hv).to(tl.float32)

    x = a_val + dt_bias_val
    softplus_x = tl.where(x <= 20.0, tl.math.log(1.0 + tl.math.exp(x)), x)
    g = -tl.math.exp(A_log_val) * softplus_x
    decay = tl.math.exp(g)
    beta = 1.0 / (1.0 + tl.math.exp(-b_val))

    k_offs = tl.arange(0, K)
    qk_base = i_n * H * K + i_h * K
    q_raw = tl.load(q_ptr + qk_base + k_offs).to(tl.float32)
    k_raw = tl.load(k_ptr + qk_base + k_offs).to(tl.float32)

    q_sq = tl.sum(q_raw * q_raw)
    k_sq = tl.sum(k_raw * k_raw)
    q_vec = q_raw * (tl.math.rsqrt(q_sq + 1e-6) * scale)
    k_vec = k_raw * tl.math.rsqrt(k_sq + 1e-6)

    v_range = i_v * BLOCK_V + tl.arange(0, BLOCK_V)
    v_base = i_n * HV * V + i_hv * V
    v_vals = tl.load(v_ptr + v_base + v_range).to(tl.float32)

    s_base = (i_n * HV + i_hv) * V * K
    s_offs = v_range[:, None] * K + k_offs[None, :]
    h = tl.load(state_ptr + s_base + s_offs)

    h = h * decay
    pred = tl.sum(h * k_vec[None, :], axis=1)
    v_new = (v_vals - pred) * beta
    h = h + v_new[:, None] * k_vec[None, :]
    out = tl.sum(h * q_vec[None, :], axis=1)

    o_base = (i_n * HV + i_hv) * V
    tl.store(output_ptr + o_base + v_range, out.to(tl.bfloat16))
    ns_base = (i_n * HV + i_hv) * V * K
    tl.store(new_state_ptr + ns_base + s_offs, h)


def run_bench(batch_size, block_v, num_warps, num_warmup=20, num_iters=200):
    H, HV, K, V = 4, 8, 128, 128
    device = "cuda"

    q = torch.randn(batch_size, H, K, dtype=torch.bfloat16, device=device)
    k = torch.randn(batch_size, H, K, dtype=torch.bfloat16, device=device)
    v = torch.randn(batch_size, HV, V, dtype=torch.bfloat16, device=device)
    state = torch.randn(batch_size, HV, V, K, dtype=torch.float32, device=device) * 0.01
    A_log = torch.randn(HV, dtype=torch.float32, device=device) * 0.1
    dt_bias = torch.randn(HV, dtype=torch.float32, device=device) * 0.1
    a = torch.randn(batch_size, HV, dtype=torch.bfloat16, device=device) * 0.1
    b_tensor = torch.randn(batch_size, HV, dtype=torch.bfloat16, device=device)
    output = torch.empty(batch_size, HV, V, dtype=torch.bfloat16, device=device)
    new_state = torch.empty_like(state)

    num_v_tiles = V // block_v
    grid = (batch_size * HV * num_v_tiles,)

    for _ in range(num_warmup):
        _gdn_decode_kernel[grid](
            q, k, v, state, output, new_state,
            A_log, a, dt_bias, b_tensor, 1.0,
            batch_size, H, HV, K, V, block_v,
            num_warps=num_warps,
        )
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(num_iters):
        _gdn_decode_kernel[grid](
            q, k, v, state, output, new_state,
            A_log, a, dt_bias, b_tensor, 1.0,
            batch_size, H, HV, K, V, block_v,
            num_warps=num_warps,
        )
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / num_iters


if __name__ == "__main__":
    batch_sizes = [1, 4, 8, 16, 32, 64, 128, 256, 512]
    block_vs = [4, 8, 16, 32, 64, 128]
    num_warps_list = [2, 4, 8]

    print(f"{'BS':>4s} | {'BV':>4s} | {'NW':>2s} | {'ms':>8s} | {'blocks':>6s}")
    print("-" * 40)

    for bs in batch_sizes:
        best_ms = float('inf')
        best_config = None
        for bv in block_vs:
            for nw in num_warps_list:
                if 128 % bv != 0:
                    continue
                num_blocks = bs * 8 * (128 // bv)
                try:
                    ms = run_bench(bs, bv, nw)
                    if ms < best_ms:
                        best_ms = ms
                        best_config = (bv, nw, num_blocks)
                except Exception:
                    pass
        bv, nw, nb = best_config
        print(f"{bs:>4d} | {bv:>4d} | {nw:>2d} | {best_ms:>8.4f} | {nb:>6d}")
