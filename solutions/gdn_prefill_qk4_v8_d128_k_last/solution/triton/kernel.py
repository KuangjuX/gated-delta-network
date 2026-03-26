"""
Triton GDN Prefill Kernel - gdn_prefill_qk4_v8_d128_k_last

Gated Delta Net prefill with GVA configuration and k-last state layout.
Core recurrence via FLA chunk_gated_delta_rule_fwd; fused Triton precompute
for GVA head expansion, and gating (A_log, a, dt_bias, b -> g, beta).

Config: num_q_heads=4, num_k_heads=4, num_v_heads=8, head_size=128
State layout: k-last [num_seqs, HV, V, K]

DPS: output and new_state are pre-allocated and written via copy_.
"""

import torch
import triton
import triton.language as tl

from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_fwd


@triton.jit
def _fused_precompute_kernel(
    q_ptr, k_ptr, a_ptr, b_ptr,
    A_log_ptr, dt_bias_ptr,
    q_out_ptr, k_out_ptr, g_out_ptr, beta_out_ptr,
    T, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr,
    RATIO: tl.constexpr,
):
    pid = tl.program_id(0)
    i_t = pid

    k_offs = tl.arange(0, K)

    for i_h in range(H):
        qk_base = i_t * H * K + i_h * K
        q_raw = tl.load(q_ptr + qk_base + k_offs)
        k_raw = tl.load(k_ptr + qk_base + k_offs)

        for r in range(RATIO):
            i_hv = i_h * RATIO + r
            out_base = i_t * HV * K + i_hv * K
            tl.store(q_out_ptr + out_base + k_offs, q_raw)
            tl.store(k_out_ptr + out_base + k_offs, k_raw)

    for i_hv in range(HV):
        A_log_val = tl.load(A_log_ptr + i_hv)
        dt_bias_val = tl.load(dt_bias_ptr + i_hv)

        a_val = tl.load(a_ptr + i_t * HV + i_hv).to(tl.float32)
        b_val = tl.load(b_ptr + i_t * HV + i_hv).to(tl.float32)

        x = a_val + dt_bias_val
        softplus_x = tl.where(x <= 20.0, tl.math.log(1.0 + tl.math.exp(x)), x)
        g = -tl.math.exp(A_log_val) * softplus_x
        beta = 1.0 / (1.0 + tl.math.exp(-b_val))

        tl.store(g_out_ptr + i_t * HV + i_hv, g)
        tl.store(beta_out_ptr + i_t * HV + i_hv, beta.to(tl.bfloat16))


def kernel(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale, output, new_state):
    total_seq_len, H, K = q.shape
    HV, V = v.shape[1], v.shape[2]
    num_seqs = cu_seqlens.shape[0] - 1
    ratio = HV // H

    if isinstance(scale, torch.Tensor):
        scale = scale.item()

    h0 = state
    if h0 is None:
        h0 = torch.zeros(num_seqs, HV, V, K, dtype=torch.float32, device=q.device)

    q_exp = torch.empty(1, total_seq_len, HV, K, dtype=torch.bfloat16, device=q.device)
    k_exp = torch.empty(1, total_seq_len, HV, K, dtype=torch.bfloat16, device=q.device)
    g_raw = torch.empty(1, total_seq_len, HV, dtype=torch.float32, device=q.device)
    beta_out = torch.empty(1, total_seq_len, HV, dtype=torch.bfloat16, device=q.device)

    _fused_precompute_kernel[(total_seq_len,)](
        q, k, a, b,
        A_log, dt_bias,
        q_exp, k_exp, g_raw, beta_out,
        total_seq_len, H, HV, K,
        ratio,
        num_warps=1,
    )

    _, o, _, final_state, _ = chunk_gated_delta_rule_fwd(
        q=q_exp,
        k=k_exp,
        v=v.unsqueeze(0),
        g=g_raw,
        beta=beta_out,
        scale=scale,
        initial_state=h0,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        transpose_state_layout=True,
    )

    output.copy_(o.squeeze(0).to(dtype=output.dtype))
    if final_state is not None:
        new_state.copy_(final_state.to(dtype=new_state.dtype))
    else:
        new_state.zero_()
