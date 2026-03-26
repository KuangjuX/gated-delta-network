"""
Benchmark our kernels against the FlashInfer-Bench reference implementations.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/benchmark.py decode
    CUDA_VISIBLE_DEVICES=0 python scripts/benchmark.py prefill
    CUDA_VISIBLE_DEVICES=0 python scripts/benchmark.py all
"""

import sys
import time
import math
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "reference"))


def benchmark_fn(fn, args, warmup=10, iters=100):
    """Benchmark a function, return median latency in ms."""
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    times = []
    for _ in range(3):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn(*args)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / iters)
    return sorted(times)[1]  # median


def check_correctness(our, ref, name, atol=5e-3, rtol=5e-3):
    """Check numerical correctness, return (pass, max_abs_err, max_rel_err)."""
    diff = (our.float() - ref.float()).abs()
    max_abs = diff.max().item()
    ref_abs = ref.float().abs().clamp(min=1e-8)
    max_rel = (diff / ref_abs).max().item()
    passed = max_abs < atol or max_rel < rtol
    return passed, max_abs, max_rel


def bench_decode():
    """Benchmark decode kernel vs reference."""
    from gdn_decode_ref import run as ref_decode

    sys.path.insert(0, str(PROJECT_ROOT / "solutions" / "gdn_decode_qk4_v8_d128_k_last" / "solution" / "triton"))
    from kernel import kernel as triton_decode

    H, HV, K, V = 4, 8, 128, 128
    device = "cuda"

    batch_sizes = [1, 2, 4, 8, 16, 32, 48, 64]

    print(f"{'BS':>4s} | {'Correct':>8s} | {'Ours(ms)':>9s} | {'Ref(ms)':>9s} | {'Speedup':>8s} | {'MaxAbsErr':>10s} | {'MaxRelErr':>10s}")
    print("-" * 80)

    all_pass = True
    speedups = []

    for bs in batch_sizes:
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        q = torch.randn(bs, 1, H, K, dtype=torch.bfloat16, device=device) * 0.1
        k = torch.randn(bs, 1, H, K, dtype=torch.bfloat16, device=device) * 0.1
        v = torch.randn(bs, 1, HV, V, dtype=torch.bfloat16, device=device) * 0.1
        state = torch.randn(bs, HV, V, K, dtype=torch.float32, device=device) * 0.01
        A_log = torch.randn(HV, dtype=torch.float32, device=device) * 0.1
        dt_bias = torch.randn(HV, dtype=torch.float32, device=device) * 0.1
        a = torch.randn(bs, 1, HV, dtype=torch.bfloat16, device=device) * 0.1
        b_t = torch.randn(bs, 1, HV, dtype=torch.bfloat16, device=device)
        scale = 1.0 / math.sqrt(K)

        # Reference
        ref_out, ref_state = ref_decode(q, k, v, state, A_log, a, dt_bias, b_t, scale)

        # Ours
        our_out = torch.empty(bs, 1, HV, V, dtype=torch.bfloat16, device=device)
        our_state = torch.empty_like(state)
        triton_decode(q, k, v, state, A_log, a, dt_bias, b_t, scale, our_out, our_state)
        torch.cuda.synchronize()

        p_o, ae_o, re_o = check_correctness(our_out, ref_out, "output")
        p_s, ae_s, re_s = check_correctness(our_state, ref_state, "state")
        passed = p_o and p_s
        all_pass &= passed

        our_ms = benchmark_fn(
            lambda *a: triton_decode(*a),
            (q, k, v, state, A_log, a, dt_bias, b_t, scale, our_out, our_state),
        )

        iters_ref = max(1, min(20, int(500 / max(bs, 1))))
        ref_ms = benchmark_fn(
            lambda *a: ref_decode(*a),
            (q, k, v, state, A_log, a, dt_bias, b_t, scale),
            warmup=2, iters=iters_ref,
        )

        spd = ref_ms / our_ms
        speedups.append(spd)

        status = "PASS" if passed else "FAIL"
        print(f"{bs:>4d} | {status:>8s} | {our_ms:>9.4f} | {ref_ms:>9.3f} | {spd:>7.1f}x | {max(ae_o,ae_s):>10.2e} | {max(re_o,re_s):>10.2e}")

    print(f"\nAll correct: {all_pass}")
    print(f"Speedup range: {min(speedups):.1f}x - {max(speedups):.1f}x (avg {sum(speedups)/len(speedups):.1f}x)")
    return all_pass


def bench_prefill():
    """Benchmark prefill kernel vs reference."""
    from gdn_prefill_ref import run as ref_prefill

    sys.path.insert(0, str(PROJECT_ROOT / "solutions" / "gdn_prefill_qk4_v8_d128_k_last" / "solution" / "triton"))
    from kernel import kernel as triton_prefill

    H, HV, K, V = 4, 8, 128, 128
    device = "cuda"

    configs = [
        ([32], "1x32"),
        ([64], "1x64"),
        ([128], "1x128"),
        ([256], "1x256"),
        ([512], "1x512"),
        ([1024], "1x1024"),
        ([64, 64], "2x64"),
        ([128, 128], "2x128"),
        ([256, 256], "2x256"),
        ([32, 64, 128, 256], "4xmixed"),
    ]

    print(f"{'Config':>12s} | {'Correct':>8s} | {'Ours(ms)':>9s} | {'Ref(ms)':>9s} | {'Speedup':>8s} | {'MaxAbsErr':>10s} | {'MaxRelErr':>10s}")
    print("-" * 90)

    all_pass = True
    speedups = []

    for seq_lens, label in configs:
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        num_seqs = len(seq_lens)
        total_seq_len = sum(seq_lens)

        cu_seqlens = torch.zeros(num_seqs + 1, dtype=torch.int64, device=device)
        for i, sl in enumerate(seq_lens):
            cu_seqlens[i + 1] = cu_seqlens[i] + sl

        q = torch.randn(total_seq_len, H, K, dtype=torch.bfloat16, device=device) * 0.1
        k = torch.randn(total_seq_len, H, K, dtype=torch.bfloat16, device=device) * 0.1
        v = torch.randn(total_seq_len, HV, V, dtype=torch.bfloat16, device=device) * 0.1
        state = torch.randn(num_seqs, HV, V, K, dtype=torch.float32, device=device) * 0.01
        A_log = torch.randn(HV, dtype=torch.float32, device=device) * 0.1
        dt_bias = torch.randn(HV, dtype=torch.float32, device=device) * 0.1
        a = torch.randn(total_seq_len, HV, dtype=torch.bfloat16, device=device) * 0.1
        b_t = torch.randn(total_seq_len, HV, dtype=torch.bfloat16, device=device)
        scale = 1.0 / math.sqrt(K)

        # Reference
        ref_out, ref_state = ref_prefill(q, k, v, state, A_log, a, dt_bias, b_t, cu_seqlens, scale)

        # Ours
        our_out = torch.empty(total_seq_len, HV, V, dtype=torch.bfloat16, device=device)
        our_state = torch.empty(num_seqs, HV, V, K, dtype=torch.float32, device=device)
        triton_prefill(q, k, v, state, A_log, a, dt_bias, b_t, cu_seqlens, scale, our_out, our_state)
        torch.cuda.synchronize()

        p_o, ae_o, re_o = check_correctness(our_out, ref_out, "output", atol=2e-2, rtol=2e-2)
        p_s, ae_s, re_s = check_correctness(our_state, ref_state, "state", atol=1e-2, rtol=1e-2)
        passed = p_o and p_s
        all_pass &= passed

        iters_ours = max(5, min(50, int(200 / max(total_seq_len / 64, 1))))
        our_ms = benchmark_fn(
            lambda *a: triton_prefill(*a),
            (q, k, v, state, A_log, a, dt_bias, b_t, cu_seqlens, scale, our_out, our_state),
            warmup=3, iters=iters_ours,
        )

        iters_ref = max(1, min(5, int(50 / max(total_seq_len / 64, 1))))
        ref_ms = benchmark_fn(
            lambda *a: ref_prefill(*a),
            (q, k, v, state, A_log, a, dt_bias, b_t, cu_seqlens, scale),
            warmup=1, iters=iters_ref,
        )

        spd = ref_ms / our_ms
        speedups.append(spd)

        status = "PASS" if passed else "FAIL"
        print(f"{label:>12s} | {status:>8s} | {our_ms:>9.3f} | {ref_ms:>9.3f} | {spd:>7.1f}x | {max(ae_o,ae_s):>10.2e} | {max(re_o,re_s):>10.2e}")

    print(f"\nAll correct: {all_pass}")
    print(f"Speedup range: {min(speedups):.1f}x - {max(speedups):.1f}x (avg {sum(speedups)/len(speedups):.1f}x)")
    return all_pass


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("target", choices=["decode", "prefill", "all"], default="all", nargs="?")
    args = parser.parse_args()

    if args.target in ("decode", "all"):
        print("=" * 80)
        print("  GDN Decode Benchmark (vs FlashInfer-Bench Reference)")
        print("=" * 80)
        bench_decode()
        print()

    if args.target in ("prefill", "all"):
        print("=" * 80)
        print("  GDN Prefill Benchmark (vs FlashInfer-Bench Reference)")
        print("=" * 80)
        bench_prefill()


if __name__ == "__main__":
    main()
