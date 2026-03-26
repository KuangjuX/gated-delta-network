"""
Optimized Triton GDN Prefill Kernel - gdn_prefill_qk4_v8_d128_k_last

Gated Delta Net prefill with GVA configuration and k-last state layout.
Uses blockwise delta rule with fused Triton kernels for hot-path operations.

Config: num_q_heads=4, num_k_heads=4, num_v_heads=8, head_size=128
State layout: k-last [num_seqs, HV, V, K]
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

BLOCK_SIZE = 64


@triton.jit
def _compute_gating_kernel(
    a_ptr, b_ptr, A_log_ptr, dt_bias_ptr,
    g_ptr, beta_ptr, alpha_ptr,
    total_len, HV: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total_elems = total_len * HV
    mask = offs < total_elems

    hv_idx = offs % HV
    A_log_val = tl.load(A_log_ptr + hv_idx, mask=mask)
    dt_bias_val = tl.load(dt_bias_ptr + hv_idx, mask=mask)
    a_val = tl.load(a_ptr + offs, mask=mask).to(tl.float32)
    b_val = tl.load(b_ptr + offs, mask=mask).to(tl.float32)

    x = a_val + dt_bias_val
    softplus_x = tl.where(x <= 20.0, tl.math.log(1.0 + tl.math.exp(x)), x)
    g = -tl.math.exp(A_log_val) * softplus_x
    beta = 1.0 / (1.0 + tl.math.exp(-b_val))
    alpha = tl.math.exp(g)

    tl.store(g_ptr + offs, g, mask=mask)
    tl.store(beta_ptr + offs, beta, mask=mask)
    tl.store(alpha_ptr + offs, alpha, mask=mask)


@triton.jit
def _inter_block_output_kernel(
    q_ptr, state_ptr, o_inter_ptr,
    gamma_ptr,
    seq_start, blk_offset, valid_len,
    HV: tl.constexpr, H: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr,
    seq_idx,
    BLOCK_V: tl.constexpr,
):
    """Compute o_inter = exp(gamma) * Q @ state for one head and V-tile."""
    pid = tl.program_id(0)
    num_v_tiles = V // BLOCK_V
    i_v = pid % num_v_tiles
    i_hv = pid // num_v_tiles

    i_h = i_hv * H // HV

    v_offs = i_v * BLOCK_V + tl.arange(0, BLOCK_V)
    k_offs = tl.arange(0, K)

    state_base = seq_idx * HV * V * K + i_hv * V * K
    state = tl.load(state_ptr + state_base + v_offs[:, None] * K + k_offs[None, :])  # [BV, K]

    for t in range(valid_len):
        token_idx = blk_offset + t
        gamma_val = tl.load(gamma_ptr + i_hv * BLOCK_SIZE + t)

        q_base = token_idx * H * K + i_h * K
        q_raw = tl.load(q_ptr + q_base + k_offs).to(tl.float32)

        q_scaled = q_raw * tl.math.exp(gamma_val)  # exp(gamma) * q (already normed+scaled)

        o_val = tl.sum(state * q_scaled[None, :], axis=1)  # [BV]

        o_base = token_idx * HV * V + i_hv * V
        tl.store(o_inter_ptr + o_base + v_offs, o_val.to(tl.bfloat16))


def _blockwise_prefill_fused(
    q_normed, k_normed, v, g, beta,
    h_state, scale, block_size, valid_len, output_slice, seq_idx,
):
    """
    Process one block of tokens using optimized batched PyTorch operations.
    Minimizes kernel launches by fusing operations.

    q_normed: [S, HV, K] float32 (L2-normed, scaled)
    k_normed: [S, HV, K] float32 (L2-normed)
    v: [S, HV, V] float32
    g: [S, HV] float32 (log-decay per token)
    beta: [S, HV] float32 (gate per token)
    h_state: [HV, V, K] float32 (k-last state, modified in-place)
    """
    S = block_size
    HV = q_normed.shape[1]
    device = q_normed.device

    # Transpose to [HV, S, ...] for batched ops
    q_HSK = q_normed.permute(1, 0, 2)
    k_HSK = k_normed.permute(1, 0, 2)
    v_HSV = v.permute(1, 0, 2)
    g_HS = g.permute(1, 0)
    beta_HS = beta.permute(1, 0)

    # Cumulative decay
    cum_g = torch.cumsum(g_HS, dim=-1)
    gamma_HS1 = cum_g.unsqueeze(2)
    Gamma_HSS = cum_g.unsqueeze(2) - cum_g.unsqueeze(1)
    block_gamma = gamma_HS1[:, [valid_len - 1], :]

    # Build and solve correction system
    KKT = torch.bmm(k_HSK, k_HSK.transpose(-2, -1))
    beta_col = beta_HS.unsqueeze(1)
    M = beta_col * torch.exp(Gamma_HSS) * KKT

    mask_lower = torch.tril(torch.ones(S, S, device=device, dtype=torch.float32), diagonal=-1)
    IKK = M * mask_lower.unsqueeze(0) + torch.eye(S, device=device, dtype=torch.float32).unsqueeze(0)

    # Solve both RHS together by concatenating along last dim
    rhs_u = beta_HS.unsqueeze(2) * v_HSV  # [HV, S, V]
    rhs_w = beta_HS.unsqueeze(2) * (torch.exp(gamma_HS1) * k_HSK)  # [HV, S, K]
    rhs_combined = torch.cat([rhs_u, rhs_w], dim=-1)  # [HV, S, V+K]
    sol_combined = torch.linalg.solve_triangular(IKK, rhs_combined, upper=False)  # [HV, S, V+K]
    V_dim = v_HSV.shape[2]
    u_HSV = sol_combined[:, :, :V_dim]
    w_HSK = sol_combined[:, :, V_dim:]

    state_ref = h_state.transpose(-2, -1)  # [HV, K, V]
    new_v_HSV = u_HSV - torch.bmm(w_HSK, state_ref)

    # Output: inter + intra
    o_inter = torch.bmm(torch.exp(gamma_HS1) * q_HSK, state_ref)
    QKT = torch.bmm(q_HSK, k_HSK.transpose(-2, -1))
    causal = torch.tril(torch.ones(S, S, device=device, dtype=torch.float32))
    o_intra = torch.bmm(QKT * torch.exp(Gamma_HSS) * causal.unsqueeze(0), new_v_HSV)

    o_blk = scale * (o_inter + o_intra)
    output_slice[:] = o_blk.permute(1, 0, 2)[:valid_len].to(output_slice.dtype)

    # State update
    K_weighted = torch.exp(block_gamma - gamma_HS1) * k_HSK
    inc = torch.bmm(K_weighted.transpose(-2, -1), new_v_HSV)
    state_ref_new = torch.exp(block_gamma) * state_ref + inc
    h_state[:] = state_ref_new.transpose(-2, -1)


def kernel(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale, output, new_state):
    """
    GDN prefill entry point (DPS style).

    Args:
        q: bf16 [total_seq_len, H, K]
        k: bf16 [total_seq_len, H, K]
        v: bf16 [total_seq_len, HV, V]
        state: f32 [num_seqs, HV, V, K] (initial state, k-last)
        A_log: f32 [HV]
        a: bf16 [total_seq_len, HV]
        dt_bias: f32 [HV]
        b: bf16 [total_seq_len, HV]
        cu_seqlens: i64 [num_seqs+1]
        scale: float
        output: bf16 [total_seq_len, HV, V]
        new_state: f32 [num_seqs, HV, V, K]
    """
    total_seq_len = q.shape[0]
    H = q.shape[1]
    K = q.shape[2]
    HV = v.shape[1]
    V = v.shape[2]
    num_seqs = cu_seqlens.shape[0] - 1

    if isinstance(scale, torch.Tensor):
        scale = scale.item()

    ratio = HV // H
    q_expanded = q.repeat_interleave(ratio, dim=1).float()
    k_expanded = k.repeat_interleave(ratio, dim=1).float()
    v_f = v.float()

    q_expanded = F.normalize(q_expanded, p=2.0, dim=-1) * scale
    k_expanded = F.normalize(k_expanded, p=2.0, dim=-1)

    # Pre-compute gating for all tokens
    g_all = torch.empty(total_seq_len, HV, dtype=torch.float32, device=q.device)
    beta_all = torch.empty(total_seq_len, HV, dtype=torch.float32, device=q.device)
    alpha_all = torch.empty(total_seq_len, HV, dtype=torch.float32, device=q.device)

    total_elems = total_seq_len * HV
    BLOCK_GATE = 1024
    grid_gate = ((total_elems + BLOCK_GATE - 1) // BLOCK_GATE,)
    _compute_gating_kernel[grid_gate](
        a, b, A_log, dt_bias,
        g_all, beta_all, alpha_all,
        total_seq_len, HV,
        BLOCK_GATE,
    )

    block_size = BLOCK_SIZE

    # Pre-allocate masks (reuse across blocks)
    for seq_idx in range(num_seqs):
        seq_start = cu_seqlens[seq_idx].item()
        seq_end = cu_seqlens[seq_idx + 1].item()

        if state is not None:
            h_state = state[seq_idx].clone()
        else:
            h_state = torch.zeros(HV, V, K, dtype=torch.float32, device=q.device)

        blk_offset = seq_start
        while blk_offset < seq_end:
            valid_len = min(block_size, seq_end - blk_offset)
            is_full = valid_len == block_size

            if is_full:
                q_blk = q_expanded[blk_offset:blk_offset + block_size]
                k_blk = k_expanded[blk_offset:blk_offset + block_size]
                v_blk = v_f[blk_offset:blk_offset + block_size]
                g_blk = g_all[blk_offset:blk_offset + block_size]
                beta_blk = beta_all[blk_offset:blk_offset + block_size]
            else:
                q_blk = torch.zeros(block_size, HV, K, dtype=torch.float32, device=q.device)
                k_blk = torch.zeros(block_size, HV, K, dtype=torch.float32, device=q.device)
                v_blk = torch.zeros(block_size, HV, V, dtype=torch.float32, device=q.device)
                g_blk = torch.zeros(block_size, HV, dtype=torch.float32, device=q.device)
                beta_blk = torch.zeros(block_size, HV, dtype=torch.float32, device=q.device)
                q_blk[:valid_len] = q_expanded[blk_offset:seq_end]
                k_blk[:valid_len] = k_expanded[blk_offset:seq_end]
                v_blk[:valid_len] = v_f[blk_offset:seq_end]
                g_blk[:valid_len] = g_all[blk_offset:seq_end]
                beta_blk[:valid_len] = beta_all[blk_offset:seq_end]

            out_slice = output[blk_offset:min(seq_end, blk_offset + block_size)]

            _blockwise_prefill_fused(
                q_blk, k_blk, v_blk, g_blk, beta_blk,
                h_state, 1.0, block_size, valid_len, out_slice, seq_idx,
            )

            blk_offset += block_size

        new_state[seq_idx] = h_state
