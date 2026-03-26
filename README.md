# Gated Delta Network Kernels

High-performance Triton/CUDA kernels for [Gated Delta Net](https://bench.flashinfer.ai/kernels/gdn_decode_qk4_v8_d128_k_last) decode and [prefill](https://bench.flashinfer.ai/kernels/gdn_prefill_qk4_v8_d128_k_last) operations, targeting NVIDIA Blackwell GPUs. Built for the [FlashInfer AI Kernel Generation Contest @ MLSys 2026](http://mlsys26.flashinfer.ai/).

---

<p align="center">
  <a href="https://www.nvidia.com"><img src="images/nvidia-logo.svg" alt="NVIDIA" height="50"/></a>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://modal.com"><img src="images/modal-logo.png" alt="Modal" height="50"/></a>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://mlsys.org"><img src="images/mlsys-logo.svg" alt="MLSys" height="50"/></a>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://github.com/flashinfer-ai/flashinfer"><img src="images/flashinfer-logo.png" alt="FlashInfer" height="50"/></a>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://github.com/flashinfer-ai/flashinfer-bench"><img src="images/fib_logo.png" alt="FlashInfer-Bench" height="50"/></a>
</p>

---

## Kernel Specifications

| Parameter | Value |
|---|---|
| Model | Qwen3 Next (linear attention layers, TP=4) |
| num_q_heads | 4 |
| num_k_heads | 4 |
| num_v_heads | 8 (GVA: more value heads than query heads) |
| head_size | 128 |
| State layout | k-last `[B, HV, V, K]` |

## Performance Summary

All benchmarks on **NVIDIA B200** (SM100, 183 GB HBM3e).

### Decode: vs FlashInfer-Bench Reference

Baseline: [FlashInfer-Bench reference](https://bench.flashinfer.ai/kernels/gdn_decode_qk4_v8_d128_k_last) — sequential PyTorch loop over batch and heads.

| Batch Size | Ours | Reference | Speedup |
|:---:|:---:|:---:|:---:|
| 1 | 0.026 ms | 1.261 ms | **48x** |
| 2 | 0.024 ms | 2.465 ms | **105x** |
| 4 | 0.023 ms | 4.587 ms | **197x** |
| 8 | 0.023 ms | 8.856 ms | **379x** |
| 16 | 0.023 ms | 17.691 ms | **765x** |
| 32 | 0.023 ms | 36.652 ms | **1585x** |
| 48 | 0.023 ms | 53.626 ms | **2327x** |
| 64 | 0.023 ms | 71.081 ms | **3061x** |

**FlashInfer-Bench full evaluation: 54/54 workloads PASSED** (avg speedup 529x, max 1469x).

### Decode: vs FlashInfer-Bench Reference (per batch_size group)

Aggregate results from FlashInfer-Bench with all 54 official workloads:

| Batch Size | Workloads | Avg Latency | Avg Ref Latency | Avg Speedup | Correctness |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 10 | 0.048 ms | 1.252 ms | 26.8x | 10/10 PASS |
| 4 | 8 | 0.047 ms | 4.392 ms | 94.5x | 8/8 PASS |
| 8 | 7 | 0.049 ms | 9.001 ms | 183.8x | 7/7 PASS |
| 16 | 7 | 0.049 ms | 18.108 ms | 370.5x | 7/7 PASS |
| 32 | 7 | 0.050 ms | 35.634 ms | 713.3x | 7/7 PASS |
| 48 | 7 | 0.051 ms | 53.281 ms | 1050.9x | 7/7 PASS |
| 64 | 8 | 0.050 ms | 71.417 ms | 1414.6x | 8/8 PASS |

### Prefill: vs FlashInfer-Bench Reference

Baseline: [FlashInfer-Bench reference](https://bench.flashinfer.ai/kernels/gdn_prefill_qk4_v8_d128_k_last) — sequential PyTorch loop over sequences and tokens.

| Config | Ours | Reference | Speedup |
|:---:|:---:|:---:|:---:|
| 1x32 (32 tokens) | 0.335 ms | 5.194 ms | **15.5x** |
| 1x64 | 0.339 ms | 10.703 ms | **31.6x** |
| 1x128 | 0.340 ms | 20.101 ms | **59.1x** |
| 1x256 | 0.347 ms | 40.545 ms | **117x** |
| 1x512 | 0.348 ms | 80.208 ms | **231x** |
| 1x1024 | 0.347 ms | 160.928 ms | **464x** |
| 2x64 (2 seqs) | 0.337 ms | 20.208 ms | **59.9x** |
| 2x128 | 0.340 ms | 40.235 ms | **118x** |
| 2x256 | 0.341 ms | 81.139 ms | **238x** |
| 4xmixed (32+64+128+256) | 0.350 ms | 75.899 ms | **217x** |

**All 10 configurations PASSED correctness verification.**

### Correctness

| Kernel | Max Absolute Error | Max Relative Error |
|---|---|---|
| Decode (all batch sizes) | < 7.63e-06 | < 3.52e-01 |
| Prefill (all configs) | < 2.26e-03 | tolerance-passing |

## Algorithm

### Gated Delta Net Decode

Single-token recurrent state update per (batch, v_head):

```
g = exp(-exp(A_log) * softplus(a + dt_bias))    -- decay factor
beta = sigmoid(b)                                -- update gate
state = g * state                                -- decay state [V, K]
pred = state @ k                                 -- prediction [V]
v_new = beta * (v - pred)                        -- gated delta
state = state + outer(v_new, k)                  -- rank-1 update
output = scale * state @ q                       -- readout [V]
```

### Gated Delta Net Prefill

Processes variable-length sequences using chunked delta rule (chunk_size=64), parallelizing across chunks with the WY representation:

```
For each chunk of C tokens:
  1. Compute gating: g, beta from (A_log, a, dt_bias, b)
  2. GVA head expansion: q[H,K], k[H,K] -> q[HV,K], k[HV,K]
  3. Intra-chunk: parallel matrix operations with tensor cores
  4. Inter-chunk: state propagation across chunks
  5. Output: combine intra-chunk and inter-chunk contributions
```

## Getting Started

### Install Dependencies

```bash
conda create -n fi-bench python=3.12
conda activate fi-bench
pip install flashinfer-bench modal flash-linear-attention
```

### Download the Dataset

```bash
git lfs install
git clone https://huggingface.co/datasets/flashinfer-ai/mlsys26-contest
export FIB_DATASET_PATH=/path/to/mlsys26-contest
```

### Run Benchmarks

**Against reference implementations:**

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/benchmark.py all
CUDA_VISIBLE_DEVICES=0 python scripts/benchmark.py decode
CUDA_VISIBLE_DEVICES=0 python scripts/benchmark.py prefill
```

**FlashInfer-Bench official evaluation:**

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_local.py gdn_decode_qk4_v8_d128_k_last
CUDA_VISIBLE_DEVICES=0 python scripts/run_local.py gdn_prefill_qk4_v8_d128_k_last
```

**Quick evaluation (subset of workloads):**

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/quick_eval.py gdn_decode_qk4_v8_d128_k_last --n 5
CUDA_VISIBLE_DEVICES=0 python scripts/quick_eval.py gdn_prefill_qk4_v8_d128_k_last --n 5
```

### Run Correctness Tests

```bash
CUDA_VISIBLE_DEVICES=0 python tests/test_decode.py
CUDA_VISIBLE_DEVICES=0 python tests/test_prefill.py
CUDA_VISIBLE_DEVICES=0 python tests/test_cuda_decode.py
```

### Pack Solution for Submission

```bash
python scripts/pack_solution.py gdn_decode_qk4_v8_d128_k_last
python scripts/pack_solution.py gdn_prefill_qk4_v8_d128_k_last
```

## Project Structure

```
gated-delta-network/
├── solutions/                                    # All kernel implementations
│   ├── gdn_decode_qk4_v8_d128_k_last/
│   │   ├── config.toml                           # Solution metadata
│   │   └── solution/
│   │       ├── triton/kernel.py                  # Triton decode kernel (BLOCK_V=8, nw=1)
│   │       └── cuda/
│   │           ├── kernel.cu                     # CUDA decode kernel (float4 + warp reduce)
│   │           └── binding.py                    # PyTorch binding via load_inline
│   └── gdn_prefill_qk4_v8_d128_k_last/
│       ├── config.toml
│       └── solution/
│           ├── triton/kernel.py                  # FLA chunk + fused precompute
│           └── cuda/kernel.py                    # Blockwise PyTorch + Triton gating
├── reference/                                    # FlashInfer-Bench official reference impls
│   ├── gdn_decode_ref.py                         # Sequential decode reference (PyTorch)
│   └── gdn_prefill_ref.py                        # Sequential prefill reference (PyTorch)
├── scripts/
│   ├── benchmark.py                              # Direct benchmark vs reference
│   ├── quick_eval.py                             # FlashInfer-Bench subset evaluation
│   ├── run_local.py                              # FlashInfer-Bench full evaluation
│   ├── run_modal.py                              # Modal cloud benchmark runner
│   └── pack_solution.py                          # Pack solution into solution.json
├── tests/
│   ├── test_decode.py                            # Decode correctness + perf tests
│   ├── test_cuda_decode.py                       # CUDA vs Triton comparison
│   ├── test_prefill.py                           # Prefill correctness + perf tests
│   └── bench_decode_sweep.py                     # BLOCK_V / num_warps sweep
├── config.toml                                   # Project-level config
├── OPTIMIZATION_LOG.md                           # Detailed optimization history
└── README.md
```

## Implementation Details

### Decode Kernel (Triton)

- **Grid:** `(B * HV * V/BLOCK_V,)` — 1D, one program per V-tile
- **BLOCK_V=8, num_warps=1** — maximizes occupancy with small blocks, more programs in flight for better memory-level parallelism
- **In-kernel gating:** fuses softplus, exp, sigmoid — no separate kernel launches
- **Memory-bound:** arithmetic intensity = 0.5 FLOP/byte (state I/O dominates)
- **Latency:** constant ~0.023 ms across all batch sizes (B=1..64)

### Prefill Kernel (FLA Chunk + Fused Precompute)

- **Chunked delta rule** via `fla.ops.gated_delta_rule.chunk.chunk_gated_delta_rule_fwd`
- **Fused precompute kernel:** single Triton kernel handles GVA head expansion (H=4 -> HV=8) and gating (A_log, a, dt_bias, b -> g, beta) for all tokens
- **Constant-time behavior:** ~0.34 ms regardless of sequence length (chunk parallelism)
- **Dependencies:** `flash-linear-attention >= 0.4.2`

## Notes

### Destination Passing Style (DPS)

Both kernels use DPS — inputs and pre-allocated outputs are passed as function parameters. This avoids measuring tensor allocation overhead in benchmarks.

### CUDA Kernel Bindings

The CUDA decode kernel uses `torch.utils.cpp_extension.load_inline` for JIT compilation. Set `binding: "torch"` in the solution spec for FlashInfer-Bench compatibility.

## References

- [FlashInfer-Bench](https://github.com/flashinfer-ai/flashinfer-bench) — official evaluation framework
- [FlashInfer-Bench Kernel Definitions](https://bench.flashinfer.ai/kernels/gdn_decode_qk4_v8_d128_k_last) — decode definition and reference
- [FlashInfer-Bench Kernel Definitions](https://bench.flashinfer.ai/kernels/gdn_prefill_qk4_v8_d128_k_last) — prefill definition and reference
- [Competition Dataset](https://huggingface.co/datasets/flashinfer-ai/mlsys26-contest) — workloads and definitions on HuggingFace
- [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention) — chunk_gated_delta_rule implementation
