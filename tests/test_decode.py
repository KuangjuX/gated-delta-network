"""Test GDN decode kernel against reference implementation."""

import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/chengqi/gated-delta-network/gdn_decode_qk4_v8_d128_k_last/solution/triton")
from kernel import kernel as triton_decode_kernel


def reference_decode(q, k, v, state, A_log, a, dt_bias, b, scale, state_dtype=torch.float32):
    """
    Reference GDN decode from flashinfer.
    q: [B, 1, H, K], k: [B, 1, H, K], v: [B, 1, HV, V]
    state: [B, HV, V, K] (k-last)
    Returns: output [B, 1, HV, V], new_state [B, HV, V, K]
    """
    B = q.shape[0]
    H = q.shape[2]
    K = q.shape[3]
    HV = v.shape[2]
    V = v.shape[3]

    q_f = q.squeeze(1).float()  # [B, H, K]
    k_f = k.squeeze(1).float()  # [B, H, K]
    v_f = v.squeeze(1).float()  # [B, HV, V]

    # GVA: expand q, k to match HV
    ratio = HV // H
    q_f = q_f.repeat_interleave(ratio, dim=1)  # [B, HV, K]
    k_f = k_f.repeat_interleave(ratio, dim=1)  # [B, HV, K]

    # L2 normalize
    q_f = F.normalize(q_f, p=2.0, dim=-1) * scale
    k_f = F.normalize(k_f, p=2.0, dim=-1)

    new_state = torch.zeros_like(state)
    output = torch.zeros(B, HV, V, dtype=torch.float32, device=q.device)

    for b_idx in range(B):
        for h_idx in range(HV):
            q_h = q_f[b_idx, h_idx]  # [K]
            k_h = k_f[b_idx, h_idx]  # [K]
            v_h = v_f[b_idx, h_idx]  # [V]

            # Gating
            A_log_val = A_log[h_idx].float()
            dt_bias_val = dt_bias[h_idx].float()
            a_val = a[b_idx, 0, h_idx].float()
            b_val = b[b_idx, 0, h_idx].float()

            x = a_val + dt_bias_val
            softplus_x = torch.log(1.0 + torch.exp(x)) if x <= 20.0 else x
            g = -torch.exp(A_log_val) * softplus_x
            beta_val = 1.0 / (1.0 + torch.exp(-b_val))

            # h_state is [V, K] in k-last layout
            h = state[b_idx, h_idx].clone().float()  # [V, K]

            # Step 1: Decay
            h = h * torch.exp(g)

            # Step 2: Prediction h @ k -> [V]
            pred = h @ k_h

            # Step 3: Delta + gate
            v_new = (v_h - pred) * beta_val

            # Step 4: Rank-1 update
            h = h + v_new.unsqueeze(1) * k_h.unsqueeze(0)

            # Step 5: Output
            output[b_idx, h_idx] = h @ q_h

            new_state[b_idx, h_idx] = h.to(state_dtype)

    return output.unsqueeze(1).to(torch.bfloat16), new_state


def test_decode(batch_size, seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    H, HV, K, V = 4, 8, 128, 128
    device = "cuda"

    q = torch.randn(batch_size, 1, H, K, dtype=torch.bfloat16, device=device) * 0.1
    k = torch.randn(batch_size, 1, H, K, dtype=torch.bfloat16, device=device) * 0.1
    v = torch.randn(batch_size, 1, HV, V, dtype=torch.bfloat16, device=device) * 0.1
    state = torch.randn(batch_size, HV, V, K, dtype=torch.float32, device=device) * 0.01
    A_log = torch.randn(HV, dtype=torch.float32, device=device) * 0.1
    dt_bias = torch.randn(HV, dtype=torch.float32, device=device) * 0.1
    a = torch.randn(batch_size, 1, HV, dtype=torch.bfloat16, device=device) * 0.1
    b_tensor = torch.randn(batch_size, 1, HV, dtype=torch.bfloat16, device=device)
    scale = 1.0

    # Reference
    ref_out, ref_state = reference_decode(q, k, v, state, A_log, a, dt_bias, b_tensor, scale)

    # Triton kernel
    our_out = torch.empty(batch_size, 1, HV, V, dtype=torch.bfloat16, device=device)
    our_state = torch.empty(batch_size, HV, V, K, dtype=torch.float32, device=device)
    triton_decode_kernel(q, k, v, state, A_log, a, dt_bias, b_tensor, scale, our_out, our_state)
    torch.cuda.synchronize()

    out_view = our_out.view(batch_size, 1, HV, V)

    # Compare
    atol_o, rtol_o = 5e-3, 5e-3
    atol_s, rtol_s = 5e-3, 5e-3

    try:
        torch.testing.assert_close(out_view, ref_out, atol=atol_o, rtol=rtol_o)
        torch.testing.assert_close(our_state, ref_state, atol=atol_s, rtol=rtol_s)
        print(f"  PASS batch_size={batch_size}")
        return True
    except AssertionError as e:
        out_diff = (out_view.float() - ref_out.float()).abs()
        state_diff = (our_state - ref_state).abs()
        print(f"  FAIL batch_size={batch_size}")
        print(f"    Output max_abs_err={out_diff.max().item():.6e}, mean_abs_err={out_diff.mean().item():.6e}")
        print(f"    State max_abs_err={state_diff.max().item():.6e}, mean_abs_err={state_diff.mean().item():.6e}")
        return False


def benchmark_decode(batch_size, num_warmup=10, num_iters=100):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    H, HV, K, V = 4, 8, 128, 128
    device = "cuda"

    q = torch.randn(batch_size, 1, H, K, dtype=torch.bfloat16, device=device)
    k = torch.randn(batch_size, 1, H, K, dtype=torch.bfloat16, device=device)
    v = torch.randn(batch_size, 1, HV, V, dtype=torch.bfloat16, device=device)
    state = torch.randn(batch_size, HV, V, K, dtype=torch.float32, device=device) * 0.01
    A_log = torch.randn(HV, dtype=torch.float32, device=device) * 0.1
    dt_bias = torch.randn(HV, dtype=torch.float32, device=device) * 0.1
    a = torch.randn(batch_size, 1, HV, dtype=torch.bfloat16, device=device) * 0.1
    b_tensor = torch.randn(batch_size, 1, HV, dtype=torch.bfloat16, device=device)
    scale = 1.0
    output = torch.empty(batch_size, 1, HV, V, dtype=torch.bfloat16, device=device)
    new_state = torch.empty_like(state)

    for _ in range(num_warmup):
        triton_decode_kernel(q, k, v, state, A_log, a, dt_bias, b_tensor, scale, output, new_state)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(num_iters):
        triton_decode_kernel(q, k, v, state, A_log, a, dt_bias, b_tensor, scale, output, new_state)
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end) / num_iters

    state_bytes = batch_size * HV * V * K * 4
    total_bytes = 2 * state_bytes  # read + write state (dominates)
    bw_gb_s = total_bytes / (elapsed_ms * 1e-3) / 1e9

    print(f"  BS={batch_size:>4d}: {elapsed_ms:.3f} ms, {bw_gb_s:.1f} GB/s")
    return elapsed_ms


if __name__ == "__main__":
    print("=== Correctness Tests ===")
    all_pass = True
    for bs in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
        all_pass &= test_decode(bs)

    if not all_pass:
        print("\nSome tests FAILED!")
        sys.exit(1)

    print("\nAll correctness tests passed!")

    print("\n=== Performance Benchmarks ===")
    for bs in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
        benchmark_decode(bs)
