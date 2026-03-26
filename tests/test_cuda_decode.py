"""Test CUDA GDN decode kernel against reference and Triton."""
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/chengqi/gated-delta-network/gdn_decode_qk4_v8_d128_k_last/solution/cuda")
from binding import kernel as cuda_decode_kernel

sys.path.insert(0, "/home/chengqi/gated-delta-network/gdn_decode_qk4_v8_d128_k_last/solution/triton")
from kernel import kernel as triton_decode_kernel


def reference_decode(q, k, v, state, A_log, a, dt_bias, b, scale):
    B = q.shape[0]; H = q.shape[2]; K = q.shape[3]; HV = v.shape[2]; V = v.shape[3]
    ratio = HV // H
    q_f = F.normalize(q.squeeze(1).float().repeat_interleave(ratio, dim=1), p=2.0, dim=-1) * scale
    k_f = F.normalize(k.squeeze(1).float().repeat_interleave(ratio, dim=1), p=2.0, dim=-1)
    v_f = v.squeeze(1).float()
    new_state = torch.zeros_like(state); output = torch.zeros(B, HV, V, dtype=torch.float32, device=q.device)
    for b_idx in range(B):
        for hv in range(HV):
            h = state[b_idx, hv].clone().float()
            x = a[b_idx, 0, hv].float() + dt_bias[hv].float()
            sp = torch.log(1.0 + torch.exp(x)) if x <= 20 else x
            decay = torch.exp(-torch.exp(A_log[hv].float()) * sp)
            beta = torch.sigmoid(b[b_idx, 0, hv].float())
            h = h * decay
            pred = h @ k_f[b_idx, hv]
            v_new = (v_f[b_idx, hv] - pred) * beta
            h = h + v_new.unsqueeze(1) * k_f[b_idx, hv].unsqueeze(0)
            output[b_idx, hv] = h @ q_f[b_idx, hv]
            new_state[b_idx, hv] = h
    return output.unsqueeze(1).to(torch.bfloat16), new_state


def test_cuda_decode(batch_size, seed=42):
    torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    H, HV, K, V = 4, 8, 128, 128; device = "cuda"
    q = torch.randn(batch_size, 1, H, K, dtype=torch.bfloat16, device=device) * 0.1
    k = torch.randn(batch_size, 1, H, K, dtype=torch.bfloat16, device=device) * 0.1
    v = torch.randn(batch_size, 1, HV, V, dtype=torch.bfloat16, device=device) * 0.1
    state = torch.randn(batch_size, HV, V, K, dtype=torch.float32, device=device) * 0.01
    A_log = torch.randn(HV, dtype=torch.float32, device=device) * 0.1
    dt_bias = torch.randn(HV, dtype=torch.float32, device=device) * 0.1
    a = torch.randn(batch_size, 1, HV, dtype=torch.bfloat16, device=device) * 0.1
    b_t = torch.randn(batch_size, 1, HV, dtype=torch.bfloat16, device=device)
    scale = 1.0

    ref_out, ref_state = reference_decode(q, k, v, state, A_log, a, dt_bias, b_t, scale)

    cuda_out = torch.empty(batch_size, 1, HV, V, dtype=torch.bfloat16, device=device)
    cuda_state = torch.empty_like(state)
    cuda_decode_kernel(q, k, v, state, A_log, a, dt_bias, b_t, scale, cuda_out, cuda_state)
    torch.cuda.synchronize()

    try:
        torch.testing.assert_close(cuda_out, ref_out, atol=5e-3, rtol=5e-3)
        torch.testing.assert_close(cuda_state, ref_state, atol=5e-3, rtol=5e-3)
        print(f"  PASS BS={batch_size}")
        return True
    except AssertionError:
        diff_o = (cuda_out.float() - ref_out.float()).abs()
        diff_s = (cuda_state - ref_state).abs()
        print(f"  FAIL BS={batch_size}: out_err={diff_o.max():.6e}, state_err={diff_s.max():.6e}")
        return False


def benchmark_both(batch_size, num_warmup=20, num_iters=200):
    torch.manual_seed(42); torch.cuda.manual_seed(42)
    H, HV, K, V = 4, 8, 128, 128; device = "cuda"
    q = torch.randn(batch_size, 1, H, K, dtype=torch.bfloat16, device=device)
    k = torch.randn(batch_size, 1, H, K, dtype=torch.bfloat16, device=device)
    v = torch.randn(batch_size, 1, HV, V, dtype=torch.bfloat16, device=device)
    state = torch.randn(batch_size, HV, V, K, dtype=torch.float32, device=device) * 0.01
    A_log = torch.randn(HV, dtype=torch.float32, device=device) * 0.1
    dt_bias = torch.randn(HV, dtype=torch.float32, device=device) * 0.1
    a = torch.randn(batch_size, 1, HV, dtype=torch.bfloat16, device=device) * 0.1
    b_t = torch.randn(batch_size, 1, HV, dtype=torch.bfloat16, device=device)
    out = torch.empty(batch_size, 1, HV, V, dtype=torch.bfloat16, device=device)
    ns = torch.empty_like(state)

    # CUDA
    for _ in range(num_warmup):
        cuda_decode_kernel(q, k, v, state, A_log, a, dt_bias, b_t, 1.0, out, ns)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(num_iters):
        cuda_decode_kernel(q, k, v, state, A_log, a, dt_bias, b_t, 1.0, out, ns)
    e.record(); torch.cuda.synchronize()
    cuda_ms = s.elapsed_time(e) / num_iters

    # Triton
    for _ in range(num_warmup):
        triton_decode_kernel(q, k, v, state, A_log, a, dt_bias, b_t, 1.0, out, ns)
    torch.cuda.synchronize()
    s2 = torch.cuda.Event(enable_timing=True); e2 = torch.cuda.Event(enable_timing=True)
    s2.record()
    for _ in range(num_iters):
        triton_decode_kernel(q, k, v, state, A_log, a, dt_bias, b_t, 1.0, out, ns)
    e2.record(); torch.cuda.synchronize()
    triton_ms = s2.elapsed_time(e2) / num_iters

    print(f"  BS={batch_size:>4d}: CUDA={cuda_ms:.4f}ms, Triton={triton_ms:.4f}ms, speedup={triton_ms/cuda_ms:.2f}x")


if __name__ == "__main__":
    print("=== CUDA Decode Correctness ===")
    all_pass = True
    for bs in [1, 4, 8, 16, 64, 128, 256, 512]:
        all_pass &= test_cuda_decode(bs)
    if not all_pass:
        print("Some tests FAILED!"); sys.exit(1)
    print("All tests passed!\n")

    print("=== CUDA vs Triton Decode Performance ===")
    for bs in [1, 4, 8, 16, 32, 64, 128, 256, 512]:
        benchmark_both(bs)
