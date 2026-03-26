"""Test GDN prefill kernel against sequential reference implementation."""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "solutions" / "gdn_prefill_qk4_v8_d128_k_last" / "solution" / "triton"))
from kernel import kernel as triton_prefill_kernel


def reference_prefill_sequential(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
    """
    Sequential reference: process one token at a time using decode logic.
    This is the most straightforward reference and avoids any blockwise complexity.

    q: [total_seq_len, H, K], k: [total_seq_len, H, K], v: [total_seq_len, HV, V]
    state: [num_seqs, HV, V, K] or None
    Returns: output [total_seq_len, HV, V], new_state [num_seqs, HV, V, K]
    """
    total_len = q.shape[0]
    H = q.shape[1]
    K = q.shape[2]
    HV = v.shape[1]
    V = v.shape[2]
    num_seqs = cu_seqlens.shape[0] - 1
    device = q.device

    ratio = HV // H
    q_exp = q.float().repeat_interleave(ratio, dim=1)
    k_exp = k.float().repeat_interleave(ratio, dim=1)
    v_f = v.float()

    q_exp = F.normalize(q_exp, p=2.0, dim=-1) * scale
    k_exp = F.normalize(k_exp, p=2.0, dim=-1)

    output = torch.zeros(total_len, HV, V, dtype=torch.float32, device=device)
    new_state = torch.zeros(num_seqs, HV, V, K, dtype=torch.float32, device=device)

    for seq_idx in range(num_seqs):
        seq_start = cu_seqlens[seq_idx].item()
        seq_end = cu_seqlens[seq_idx + 1].item()

        if state is not None:
            h = state[seq_idx].clone().float()  # [HV, V, K]
        else:
            h = torch.zeros(HV, V, K, dtype=torch.float32, device=device)

        for t in range(seq_start, seq_end):
            for hv in range(HV):
                q_h = q_exp[t, hv]  # [K]
                k_h = k_exp[t, hv]  # [K]
                v_h = v_f[t, hv]    # [V]

                a_val = a[t, hv].float()
                b_val = b[t, hv].float()
                A_log_val = A_log[hv].float()
                dt_bias_val = dt_bias[hv].float()

                x = a_val + dt_bias_val
                if x <= 20.0:
                    softplus_x = torch.log(1.0 + torch.exp(x))
                else:
                    softplus_x = x
                g_val = -torch.exp(A_log_val) * softplus_x
                decay = torch.exp(g_val)
                beta_val = torch.sigmoid(b_val)

                h_hv = h[hv]  # [V, K]

                # Decay
                h_hv = h_hv * decay

                # Prediction: h @ k -> [V]
                pred = h_hv @ k_h

                # Delta + gate
                v_new = (v_h - pred) * beta_val

                # Rank-1 update
                h_hv = h_hv + v_new.unsqueeze(1) * k_h.unsqueeze(0)

                # Output
                output[t, hv] = h_hv @ q_h

                h[hv] = h_hv

        new_state[seq_idx] = h

    return output.to(torch.bfloat16), new_state


def test_prefill(seq_lens, seed=42, with_initial_state=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    H, HV, K, V = 4, 8, 128, 128
    device = "cuda"
    num_seqs = len(seq_lens)
    total_seq_len = sum(seq_lens)

    cu_seqlens = torch.zeros(num_seqs + 1, dtype=torch.int64, device=device)
    for i, sl in enumerate(seq_lens):
        cu_seqlens[i + 1] = cu_seqlens[i] + sl

    q = torch.randn(total_seq_len, H, K, dtype=torch.bfloat16, device=device) * 0.1
    k = torch.randn(total_seq_len, H, K, dtype=torch.bfloat16, device=device) * 0.1
    v = torch.randn(total_seq_len, HV, V, dtype=torch.bfloat16, device=device) * 0.1
    A_log = torch.randn(HV, dtype=torch.float32, device=device) * 0.1
    dt_bias = torch.randn(HV, dtype=torch.float32, device=device) * 0.1
    a = torch.randn(total_seq_len, HV, dtype=torch.bfloat16, device=device) * 0.1
    b_tensor = torch.randn(total_seq_len, HV, dtype=torch.bfloat16, device=device) * 0.1
    scale = 1.0

    if with_initial_state:
        state = torch.randn(num_seqs, HV, V, K, dtype=torch.float32, device=device) * 0.01
    else:
        state = torch.zeros(num_seqs, HV, V, K, dtype=torch.float32, device=device)

    # Reference
    ref_out, ref_state = reference_prefill_sequential(
        q, k, v, state, A_log, a, dt_bias, b_tensor, cu_seqlens, scale
    )

    # Triton kernel
    our_out = torch.empty(total_seq_len, HV, V, dtype=torch.bfloat16, device=device)
    our_state = torch.empty(num_seqs, HV, V, K, dtype=torch.float32, device=device)
    triton_prefill_kernel(q, k, v, state, A_log, a, dt_bias, b_tensor, cu_seqlens, scale, our_out, our_state)
    torch.cuda.synchronize()

    atol_o, rtol_o = 2e-2, 2e-2
    atol_s, rtol_s = 1e-2, 1e-2

    try:
        torch.testing.assert_close(our_out, ref_out, atol=atol_o, rtol=rtol_o)
        torch.testing.assert_close(our_state, ref_state, atol=atol_s, rtol=rtol_s)
        desc = f"seq_lens={seq_lens}" + (" (with init state)" if with_initial_state else "")
        print(f"  PASS {desc}")
        return True
    except AssertionError as e:
        out_diff = (our_out.float() - ref_out.float()).abs()
        state_diff = (our_state - ref_state).abs()
        desc = f"seq_lens={seq_lens}" + (" (with init state)" if with_initial_state else "")
        print(f"  FAIL {desc}")
        print(f"    Output max_abs_err={out_diff.max().item():.6e}, mean_abs_err={out_diff.mean().item():.6e}")
        print(f"    State max_abs_err={state_diff.max().item():.6e}, mean_abs_err={state_diff.mean().item():.6e}")
        return False


def benchmark_prefill(seq_lens, num_warmup=5, num_iters=20):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    H, HV, K, V = 4, 8, 128, 128
    device = "cuda"
    num_seqs = len(seq_lens)
    total_seq_len = sum(seq_lens)

    cu_seqlens = torch.zeros(num_seqs + 1, dtype=torch.int64, device=device)
    for i, sl in enumerate(seq_lens):
        cu_seqlens[i + 1] = cu_seqlens[i] + sl

    q = torch.randn(total_seq_len, H, K, dtype=torch.bfloat16, device=device)
    k = torch.randn(total_seq_len, H, K, dtype=torch.bfloat16, device=device)
    v = torch.randn(total_seq_len, HV, V, dtype=torch.bfloat16, device=device)
    state = torch.zeros(num_seqs, HV, V, K, dtype=torch.float32, device=device)
    A_log = torch.randn(HV, dtype=torch.float32, device=device) * 0.1
    dt_bias = torch.randn(HV, dtype=torch.float32, device=device) * 0.1
    a = torch.randn(total_seq_len, HV, dtype=torch.bfloat16, device=device)
    b_tensor = torch.randn(total_seq_len, HV, dtype=torch.bfloat16, device=device)
    scale = 1.0
    output = torch.empty(total_seq_len, HV, V, dtype=torch.bfloat16, device=device)
    new_state = torch.empty_like(state)

    for _ in range(num_warmup):
        triton_prefill_kernel(q, k, v, state, A_log, a, dt_bias, b_tensor, cu_seqlens, scale, output, new_state)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(num_iters):
        triton_prefill_kernel(q, k, v, state, A_log, a, dt_bias, b_tensor, cu_seqlens, scale, output, new_state)
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end) / num_iters
    print(f"  seq_lens={seq_lens}: {elapsed_ms:.3f} ms")
    return elapsed_ms


if __name__ == "__main__":
    print("=== Correctness Tests ===")
    all_pass = True

    # Short sequences (within one block)
    all_pass &= test_prefill([32])
    all_pass &= test_prefill([64])

    # Multi-block sequences
    all_pass &= test_prefill([128])
    all_pass &= test_prefill([256])

    # Non-aligned lengths (tail block)
    all_pass &= test_prefill([31])
    all_pass &= test_prefill([65])
    all_pass &= test_prefill([127])

    # Multiple sequences
    all_pass &= test_prefill([64, 64])
    all_pass &= test_prefill([32, 96])
    all_pass &= test_prefill([64, 128])

    # With initial state
    all_pass &= test_prefill([64], with_initial_state=True)
    all_pass &= test_prefill([128], with_initial_state=True)
    all_pass &= test_prefill([64, 128], with_initial_state=True)

    if not all_pass:
        print("\nSome tests FAILED!")
        sys.exit(1)

    print("\nAll correctness tests passed!")

    print("\n=== Performance Benchmarks ===")
    benchmark_prefill([256])
    benchmark_prefill([512])
    benchmark_prefill([1024])
    benchmark_prefill([256, 256])
    benchmark_prefill([512, 512])
