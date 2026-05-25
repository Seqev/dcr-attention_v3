"""
Top-K Q-axis kernel pipeline — M2 + M3 entry points.

topk_qaxis_select  — M2: Pass 1+2, returns indices [B, H, k_eff] int32
topk_qaxis_attention — M3: full pipeline Pass 1+2+3, returns O [B, H, D] bf16

Same signature as M1 reference (topk_qaxis_attention_reference) — drop-in.
"""

from __future__ import annotations

import torch

from dcr_attention.kernel.qaxis_topk_reference import _compute_u_Q
from dcr_attention.kernel.triton.topk_axis_pass1 import topk_pass1_projection
from dcr_attention.kernel.triton.topk_axis_pass2 import topk_pass2_select
from dcr_attention.kernel.triton.fused_attn import fused_topk_attention


def topk_qaxis_select(
    Q: torch.Tensor,         # [B, H, D] bf16
    K_cache: torch.Tensor,   # [B, H_kv, N, D] bf16
    k_eff: int,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:           # [B, H, k_eff] int32, sorted ascending
    """
    Compute top-K key indices via Q-axis projection.

    Steps:
    1. u_Q = Q / ||Q||  (fp32, host-side Python — small op)
    2. Pass 1: project all N keys onto u_Q → scores [B, H, N] bf16  (Triton)
    3. Pass 2: top-K from scores → indices [B, H, k_eff] int32  (torch wrapper)

    Does NOT compute attention output — that is Pass 3 / M3.
    """
    B, H, D = Q.shape
    B_k, H_kv, N, D_k = K_cache.shape

    assert Q.dtype == torch.bfloat16,         f"Q must be bf16, got {Q.dtype}"
    assert K_cache.dtype == torch.bfloat16,   f"K_cache must be bf16, got {K_cache.dtype}"
    assert B == B_k and D == D_k,             "Q and K_cache batch/head-dim mismatch"
    assert H % H_kv == 0,                     f"H={H} must be divisible by H_kv={H_kv}"
    assert k_eff > 0 and k_eff < N,           f"k_eff={k_eff} must be in (0, N={N})"

    # Step 1: unit Q vector (fp32, host Python — not a kernel)
    u_Q = _compute_u_Q(Q, eps=eps)                        # [B, H, D] fp32

    # Pass 1: projection (Triton kernel)
    scores = topk_pass1_projection(K_cache, u_Q)          # [B, H, N] bf16

    # Pass 2: top-K selection (torch.argsort wrapper)
    indices = topk_pass2_select(scores, k_eff)            # [B, H, k_eff] int32

    return indices


def topk_qaxis_attention(
    Q: torch.Tensor,            # [B, H, D] bf16
    K_cache: torch.Tensor,      # [B, H_kv, N, D] bf16
    V_cache: torch.Tensor,      # [B, H_kv, N, D] bf16
    k_eff: int,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:              # [B, H, D] bf16
    """
    Full M2+M3 Triton pipeline: Q + K_cache + V_cache → O.

    Drop-in replacement for M1 topk_qaxis_attention_reference.
    Same signature (§8). Validated against M1 per spec §5.1.2.

    Steps:
    1. u_Q = Q / ||Q||  (fp32, Python)
    2. Pass 1: K → scores [B, H, N] bf16  (Triton)
    3. Pass 2: scores → indices [B, H, k_eff] int32  (torch.argsort)
    4. Pass 3: gather + fused softmax-attention → O [B, H, D] bf16  (Triton)
    """
    B, H, D     = Q.shape
    B_k, H_kv, N, D_k = K_cache.shape

    assert Q.dtype == torch.bfloat16
    assert K_cache.dtype == torch.bfloat16
    assert V_cache.dtype == torch.bfloat16
    assert B == B_k and D == D_k
    assert H % H_kv == 0
    assert k_eff > 0 and k_eff < N

    indices = topk_qaxis_select(Q, K_cache, k_eff, eps=eps)   # [B, H, k_eff] int32
    O       = fused_topk_attention(Q, K_cache, V_cache, indices)  # [B, H, D] bf16
    return O
