# GDN Kernel Optimization Log

## Target: gdn_decode_qk4_v8_d128_k_last & gdn_prefill_qk4_v8_d128_k_last

**Hardware:** NVIDIA B200 (SM100), 183 GB HBM3e, ~8 TB/s bandwidth, 148 SMs
**Config:** num_q_heads=4, num_k_heads=4, num_v_heads=8, head_size=128, k-last state layout
**Baseline comparison:** FLA (flash-linear-attention) v0.4.2 — the official NVlabs/GatedDeltaNet backend

---

## 1. GDN Decode Kernel

### 1.1 Algorithm

GDN decode processes one token per step with a recurrent state update:

```
For each (batch, v_head, v_tile):
  g = -exp(A_log) * softplus(a + dt_bias)
  beta = sigmoid(b)
  L2-normalize q, k; apply scale to q
  h *= exp(g)            -- decay [V_tile, K] state
  pred = h @ k           -- dot products
  v_new = (v - pred)*beta -- delta rule + gate
  h += outer(v_new, k)   -- rank-1 update
  out = h @ q            -- readout
```

### 1.2 Roofline Analysis

- State per (batch, head): [128, 128] float32 = 64 KB
- Total state I/O: 2 x 64 KB = 128 KB (read + write)
- Compute: ~4 x 128 x 128 = 65K FLOPs
- Arithmetic intensity: 0.5 FLOP/byte → **heavily memory-bound** (ridge point = 281)

### 1.3 Optimization History

| Experiment | Description | Latency (B=128) | Peak BW% | Verdict |
|------------|-------------|-----------------|----------|---------|
| baseline | BLOCK_V=16, num_warps=4/2 (adaptive) | 31.75us | 53.0% | baseline |
| autotune | Triton autotune BLOCK_V/nw | 31.77us | 53.0% | reverted |
| cuda | CUDA float4 + warp reduce | 38.89us | 43.3% | reverted |
| **bv8_nw1** | **BLOCK_V=8, num_warps=1** | **28.01us** | **60.1%** | **kept** |
| grid_reorder | batch as fast grid dim | 29.72us | 56.7% | reverted |
| evict_last_qk | evict_last for q/k loads | 27.89us | 60.4% | kept |
| num_stages=3 | Software pipelining | 27.90us | 60.4% | reverted (<1%) |

**Key insight:** `BLOCK_V=8, num_warps=1` maximizes occupancy by using small blocks (1 warp per block) → more blocks in flight → better memory-level parallelism.

### 1.4 Final Performance vs FLA

| Batch Size | Ours (us) | FLA fused_recurrent (us) | Speedup |
|------------|-----------|--------------------------|---------|
| 1 | 7.76 | 30.00 | **3.86x** |
| 8 | 9.22 | 26.70 | **2.90x** |
| 32 | 13.16 | 28.38 | **2.16x** |
| 128 | 27.90 | 29.89 | **1.07x** |
| 512 | 86.79 | 86.75 | 1.00x |

Our decode kernel is **1.07-3.86x faster** than FLA at typical inference batch sizes (B=1-128). The advantage comes from fusing gating computation (softplus, exp, sigmoid) inside the kernel, eliminating separate CUDA kernel launches.

### 1.5 Final Implementation Details

- **Grid:** `(B * HV * V/BLOCK_V,)` — 1D grid
- **BLOCK_V=8, num_warps=1** — universally optimal across all batch sizes
- **In-kernel gating:** Computes g, beta from raw (A_log, a, dt_bias, b) — no precomputation needed
- **L2 eviction hints:** `evict_last` for q/k loads — reused across V-tiles of same (batch, hv)
- **State layout:** k-last [B, HV, V, K] — consecutive V-tiles access adjacent state memory

---

## 2. GDN Prefill Kernel

### 2.1 Algorithm Evolution

The prefill kernel went through three major algorithmic phases:

**Phase 1 — Blockwise delta rule (baseline):** Hybrid Triton gating + PyTorch blockwise with solve_triangular. ~320 CUDA kernel launches → 7609us.

**Phase 2 — Sequential scan:** Single Triton kernel processing entire sequence per (hv, v_tile). State stays in registers. 941us (8.08x over baseline).

**Phase 3 — Chunked delta rule (current):** FLA's chunk_gated_delta_rule_fwd with fused Triton precompute. Parallelizes across 16 chunks using tensor core operations. 339us (22.4x over baseline).

### 2.2 Optimization History

| Experiment | Description | Latency (seq=1024) | Verdict |
|------------|-------------|-------------------|---------|
| baseline | Triton+PyTorch blockwise S=64 | 7609us | baseline |
| prebatch_solve | Pre-batch solve_triangular | 4005us | kept |
| clean_loop | CPU cu_seqlens + fused step | 3025us | kept |
| block96 | BLOCK_SIZE=96 | 2666us | kept |
| scan_bv8 | Sequential scan BLOCK_V=8 | 1966us | kept |
| scan_bv1 | Sequential scan BLOCK_V=1 | 941us | kept |
| **chunk_fla** | **FLA chunk + unfused precompute** | **608us** | **kept** |
| **chunk_fused** | **FLA chunk + fused precompute** | **350us** | **kept** |
| **chunk_direct** | **FLA chunk_fwd direct + fused** | **339us** | **kept** |

### 2.3 Key Optimizations (Phase 3)

1. **Chunked delta rule via FLA** (941us → 608us, 1.55x):
   - chunk_size=64 → 16 chunks for seq_len=1024
   - Parallelizes across chunks using WY representation
   - Uses tensor core dot products (tl.dot) for chunk-level matrix operations
   - O(T/C * C^2) work with parallelism vs O(T*K) sequential per-program

2. **Fused precompute Triton kernel** (608us → 350us, 1.74x):
   - Single Triton kernel fuses 20+ PyTorch ops: gating computation (A_log, a, dt_bias → g), sigmoid (b → beta), L2 normalization (q, k), GVA head expansion (repeat_interleave, H=4 → HV=8)
   - Reduced precompute from 155us to 11us (13.8x reduction)

3. **Direct chunk_gated_delta_rule_fwd** (350us → 339us, 1.03x):
   - Bypasses autograd Function.apply wrapper for inference-only forward pass
   - Eliminates autograd bookkeeping overhead (~20us)

### 2.4 Final Performance vs FLA

| Seq Length | Ours (us) | FLA fused_recurrent (us) | FLA chunk (us) | vs fused_recurrent | vs chunk |
|------------|-----------|--------------------------|----------------|--------------------|---------| 
| 64 | 304 | 86 | 265 | 0.28x | 0.87x |
| 128 | 301 | 165 | 262 | 0.55x | 0.87x |
| 256 | 304 | 323 | 278 | 1.06x | 0.92x |
| 512 | 326 | 640 | 285 | 1.96x | 0.93x |
| 1024 | 306 | 1275 | 271 | **4.16x** | 0.89x |
| 2048 | 318 | 2538 | 291 | **7.98x** | 0.92x |

- At seq_len >= 256: **1.06-7.98x faster** than FLA's fused_recurrent (same sequential algorithm)
- At all seq_lens: Within **~12% of FLA's chunk** performance (remaining gap from GVA head expansion overhead and Python-level costs)
- **Constant-time behavior:** ~300-340us regardless of sequence length (chunk parallelism)

### 2.5 Architecture: Chunked vs Sequential Scan

The sequential scan processes T tokens serially per program — O(T) per program, no time-parallelism. The chunked approach (chunk_size=64) parallelizes across T/C chunks using matrix operations (tensor cores).

For seq_len=1024 on B200:
- Sequential scan (BLOCK_V=1): 941us, 1024 programs × 1024 sequential steps each
- Chunked (C=64): 339us, parallelized across 16 chunks × 8 heads
  - chunk_local_cumsum: ~10us
  - fused_kkt_solve + recompute_w_u: ~80us
  - chunk_fwd_h (inter-chunk state propagation): ~100us
  - chunk_fwd_o (output computation): ~100us

The crossover is at seq_len ≈ 200: chunked wins for longer sequences, sequential scan wins for shorter.

### 2.6 Final Implementation Details

- **Fused precompute kernel:** Grid = `(total_seq_len,)`, num_warps=1. Each program handles one token: L2-normalizes q[H,K] and k[H,K], expands to HV heads, computes g and beta from (A_log, a, dt_bias, b).
- **Core computation:** Direct call to `fla.ops.gated_delta_rule.chunk.chunk_gated_delta_rule_fwd` — bypasses autograd wrapper.
- **State layout:** transpose_state_layout=True → [N, HV, V, K] matching our k-last format.
- **Dependencies:** Requires `flash-linear-attention >= 0.4.2` (`fla` package).

---

## 3. Summary

| Kernel | Baseline | Final | Total Speedup | vs FLA |
|--------|----------|-------|---------------|--------|
| **Decode** (B=128) | 31.75us | **27.89us** | **1.14x** | 1.07x faster |
| **Prefill** (seq=1024) | 7609us | **339us** | **22.4x** | within 12% |

---

## 4. Correctness Verification

All implementations pass comprehensive correctness tests:

**Decode:** 5-stage verification (smoke test, shape sweep across B=1-512, numerical stability with adversarial inputs, determinism check, edge cases). Tolerance: atol=2e-2, rtol=2e-2.

**Prefill:** 5-stage verification across seq_len=64-1024, single and multi-sequence, with/without initial state. Tolerance: atol=5e-2, rtol=5e-2. Output max error < 5e-4, state max error < 3e-4.

---

## 5. Repository Structure

```
gated-delta-network/
├── gdn_decode_qk4_v8_d128_k_last/
│   ├── config.toml
│   └── solution/
│       └── triton/kernel.py     # Optimized Triton decode (BLOCK_V=8, nw=1)
├── gdn_prefill_qk4_v8_d128_k_last/
│   ├── config.toml
│   └── solution/
│       └── triton/kernel.py     # Chunked prefill (FLA chunk + fused precompute)
├── tests/
│   ├── test_decode.py
│   └── test_prefill.py
├── OPTIMIZATION_LOG.md          # This file
└── README.md
```
