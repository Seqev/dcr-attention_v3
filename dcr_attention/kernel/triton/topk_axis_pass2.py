"""
Pass 2 — Top-K selection from projection scores.

M2 first iteration: torch.topk wrapper (no Triton kernel).
Triton native top-K deferred to M6 if profiling shows Pass 2 is bottleneck.

Per spec §6 tie-break: stable ordering, smaller index wins.
Per spec §2: returns indices sorted ascending (required for Pass 3 sequential access).
"""

from __future__ import annotations

import torch


def topk_pass2_select(
    scores: torch.Tensor,    # [B, H, N] bf16 (from Pass 1)
    k_eff: int,
) -> torch.Tensor:           # [B, H, k_eff] int32, sorted ascending
    """
    Top-K selection via stable argsort.

    torch.topk is NOT stable on CUDA; to match M1 reference tie-break (§6:
    smaller index wins on equal scores) we promote to fp32 and use argsort
    with stable=True — same as M1 _select_topk_indices.
    """
    B, H, N = scores.shape

    assert scores.dtype == torch.bfloat16,    f"scores must be bf16, got {scores.dtype}"
    assert k_eff > 0,                          f"k_eff must be > 0, got {k_eff}"
    assert k_eff < N, (
        f"§2 Step 0 / §6: k_eff={k_eff} must be < N={N} (caller's contract)"
    )

    # fp32 promotion before argsort: avoids bf16 tie-break ambiguity on CUDA
    scores_f32  = scores.float()
    sorted_idx  = torch.argsort(scores_f32, dim=-1, descending=True, stable=True)
    topk_idx    = sorted_idx[:, :, :k_eff]          # [B, H, k_eff]

    # Sort ascending for Pass 3 sequential access (spec §2)
    topk_idx_sorted, _ = topk_idx.sort(dim=-1)

    return topk_idx_sorted.to(torch.int32)           # int32 per spec §2
