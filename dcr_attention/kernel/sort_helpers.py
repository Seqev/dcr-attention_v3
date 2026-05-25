"""
Pre-computation of sort indices for rank-local attention.

All sorting is done outside the kernel (P2 in docs/design/phase1_forward_kernel.md).
This module provides the single entry point ``prepare_sort_indices`` and is pure
torch — no CUDA / Triton dependency.

**Phase 1.3 addition:** queries are now also sorted by their projection onto
the axis, enabling the Triton kernel to bound the per-block K-loop by the
window union (O(N·k·D) compute instead of O(N²·D)).  See
``docs/design/phase1_3_q_sort.md`` for the algorithmic motivation.
"""

from __future__ import annotations
from typing import NamedTuple

import torch


class SortIndices(NamedTuple):
    """
    Pre-computed rank-window bookkeeping.

    K-side fields (unchanged from Phase 1.2):

    sort_idx_k
        ``[B, H, N]`` int64.  ``sort_idx_k[b,h,t] = j`` means key ``j`` sits
        at sorted position ``t`` in ascending order of its axis projection.
    rank_of_k
        ``[B, H, N]`` int64, inverse permutation.  ``rank_of_k[b,h,j] = t``.

    Q-side fields (NEW in Phase 1.3):

    sort_idx_q
        ``[B, H, N]`` int64.  ``sort_idx_q[b,h,t] = i`` means query ``i``
        from the original order sits at sorted position ``t``.
    inv_q_perm
        ``[B, H, N]`` int64, inverse permutation.  Applied to kernel output
        in sorted Q-order to restore the caller's original Q-order.

    r_center
        ``[B, H, N]`` int64.  **Semantic change from Phase 1.2:**

        * Phase 1.2: ``r_center[i]`` = insertion rank for query ``i`` in
          original Q-order.
        * Phase 1.3: ``r_center[t]`` = insertion rank for query at sorted
          position ``t``.  Because Q is sorted by ``z_q`` and K is sorted
          by ``z_k``, ``r_center`` is **monotonically non-decreasing in t**.
          This is the property the Triton kernel exploits.

    z_q, z_k
        ``[B, H, N]`` projections onto the axis.  Diagnostic fields kept
        for debugging and future experiments.  ``z_q`` is in **original**
        order (pre-sort).
    """

    sort_idx_k: torch.Tensor
    rank_of_k: torch.Tensor
    sort_idx_q: torch.Tensor
    inv_q_perm: torch.Tensor
    r_center: torch.Tensor
    z_q: torch.Tensor
    z_k: torch.Tensor


def prepare_sort_indices(
    Q: torch.Tensor,
    K: torch.Tensor,
    axis: torch.Tensor,
) -> SortIndices:
    r"""
    Compute all sort-related indices needed by ``rank_local_fwd_triton``.

    Math.

    .. math::
        z^Q_{bhi} = \langle Q_{bhi}, u\rangle, \qquad
        z^K_{bhj} = \langle K_{bhj}, u\rangle.

    Let :math:`\pi^K_{bh}` be the permutation sorting :math:`z^K_{bh\cdot}`
    ascending; ``sort_idx_k[b,h,t] = π^K(t)``, ``rank_of_k[b,h,j] = (π^K)^{-1}(j)``.

    Let :math:`\pi^Q_{bh}` be the permutation sorting :math:`z^Q_{bh\cdot}`
    ascending; ``sort_idx_q[b,h,t] = π^Q(t)``, ``inv_q_perm[b,h,i] = (π^Q)^{-1}(i)``.

    Define ``r_center[b,h,t] = |{j : z^K_{bhj} < z^Q_{bh, π^Q(t)}}|`` — the
    insertion rank in the K-sorted sequence for the query at **sorted Q-position
    t**.  Because both sequences are sorted ascending, ``r_center`` is
    monotonically non-decreasing in t.

    Parameters
    ----------
    Q, K
        ``[B, H, N, D]`` same dtype and device.
    axis
        ``[D]`` for a single ordering axis shared across (b, h), or
        ``[B, H, D]`` for per-(b, h) axes (Phase 2a — Llama wrapper passes
        per-head PCA axes).  Caller's responsibility to keep unit-norm.

    Returns
    -------
    SortIndices
    """
    if Q.dim() != 4 or K.dim() != 4:
        raise ValueError(
            f"expected [B,H,N,D] for Q and K, got {tuple(Q.shape)} and {tuple(K.shape)}"
        )
    # Phase 2-pre: allow Q.N != K.N (decode shape: Q.N=1, K.N=context_len).
    # B, H, D must match; only the sequence dim may differ.
    if Q.shape[0] != K.shape[0] or Q.shape[1] != K.shape[1] or Q.shape[3] != K.shape[3]:
        raise ValueError(
            f"Q shape {tuple(Q.shape)} and K shape {tuple(K.shape)} must agree "
            f"on (B, H, D); only the N dimension may differ"
        )
    # Phase 2a (D4): axis may be [D] (shared) or [B, H, D] (per-head).
    B, H, _, D = Q.shape
    if axis.dim() == 1:
        if axis.shape[0] != D:
            raise ValueError(
                f"axis shape {tuple(axis.shape)} must be [D={D}]"
            )
    elif axis.dim() == 3:
        if axis.shape != (B, H, D):
            raise ValueError(
                f"axis shape {tuple(axis.shape)} must be [B={B}, H={H}, D={D}]"
            )
    else:
        raise ValueError(
            f"axis must be 1-D [D] or 3-D [B, H, D], got {axis.dim()}-D"
        )

    # Project onto axis — [B, H, N]
    if axis.dim() == 1:
        z_q = torch.einsum("bhnd,d->bhn", Q, axis.to(Q.dtype))
        z_k = torch.einsum("bhnd,d->bhn", K, axis.to(K.dtype))
    else:  # axis.dim() == 3
        z_q = torch.einsum("bhnd,bhd->bhn", Q, axis.to(Q.dtype))
        z_k = torch.einsum("bhnd,bhd->bhn", K, axis.to(K.dtype))

    # K-side permutation
    sort_idx_k = torch.argsort(z_k, dim=-1)                # [B,H,N]
    rank_of_k = torch.argsort(sort_idx_k, dim=-1)          # [B,H,N]

    # Q-side permutation (NEW in Phase 1.3)
    sort_idx_q = torch.argsort(z_q, dim=-1)                # [B,H,N]
    inv_q_perm = torch.argsort(sort_idx_q, dim=-1)         # [B,H,N]

    # z_q in sorted Q-order: z_q_sorted[t] = z_q[sort_idx_q[t]]
    z_q_sorted = torch.gather(z_q, -1, sort_idx_q).contiguous()

    # z_k in sorted K-order — needed for searchsorted
    z_k_sorted = torch.gather(z_k, -1, sort_idx_k).contiguous()

    # r_center[b,h,t] = insertion rank of z_q_sorted[b,h,t] into sorted z_k[b,h,:]
    # Because z_q_sorted is monotone non-decreasing in t, and z_k_sorted is also
    # monotone, r_center is monotone non-decreasing in t.
    #
    # Phase 2-pre fix: clamp upper bound is N_kv-1 (== K.shape[-2] - 1), NOT
    # N_q-1.  In prefill these coincide, but in decode (N_q=1, N_kv=context_len)
    # using N_q-1 would incorrectly clamp r_center to 0.
    r_center = torch.searchsorted(z_k_sorted, z_q_sorted)
    r_center = r_center.clamp_(0, K.shape[-2] - 1)

    return SortIndices(
        sort_idx_k=sort_idx_k,
        rank_of_k=rank_of_k,
        sort_idx_q=sort_idx_q,
        inv_q_perm=inv_q_perm,
        r_center=r_center,
        z_q=z_q,
        z_k=z_k,
    )


def gather_by_sort_idx(
    X: torch.Tensor,
    sort_idx: torch.Tensor,
) -> torch.Tensor:
    r"""
    Pre-gather ``X`` along the sequence dimension according to a permutation.

    Parameters
    ----------
    X
        ``[B, H, N, D]``.
    sort_idx
        ``[B, H, N]`` int64 — typically ``sort_idx_k`` or ``sort_idx_q``.

    Returns
    -------
    X_sorted : ``[B, H, N, D]``, where ``X_sorted[b,h,t,:] = X[b,h,sort_idx[b,h,t],:]``.
    """
    if X.dim() != 4 or sort_idx.dim() != 3:
        raise ValueError(
            f"X must be [B,H,N,D] (got {tuple(X.shape)}) and "
            f"sort_idx [B,H,N] (got {tuple(sort_idx.shape)})"
        )
    D = X.shape[-1]
    idx = sort_idx.unsqueeze(-1).expand(-1, -1, -1, D)
    return torch.gather(X, dim=2, index=idx)


def scatter_by_inv_perm(
    X_sorted: torch.Tensor,
    inv_perm: torch.Tensor,
) -> torch.Tensor:
    r"""
    Apply an inverse permutation to undo a previous gather.

    Equivalent to ``gather_by_sort_idx(X_sorted, inv_perm)``: indexing X_sorted
    at inv_perm[i] returns the element originally at position i.

    Used in Phase 1.3 to restore output to original Q-order after the kernel
    produces output in sorted Q-order.

    Parameters
    ----------
    X_sorted
        ``[B, H, N, D]`` or ``[B, H, N]`` (LSE case).
    inv_perm
        ``[B, H, N]`` — typically ``inv_q_perm`` from ``SortIndices``.

    Returns
    -------
    Tensor of the same shape as ``X_sorted``.
    """
    if inv_perm.dim() != 3:
        raise ValueError(f"inv_perm must be [B,H,N], got {tuple(inv_perm.shape)}")
    if X_sorted.dim() == 4:
        D = X_sorted.shape[-1]
        idx = inv_perm.unsqueeze(-1).expand(-1, -1, -1, D)
        return torch.gather(X_sorted, dim=2, index=idx)
    elif X_sorted.dim() == 3:
        return torch.gather(X_sorted, dim=2, index=inv_perm)
    else:
        raise ValueError(f"X_sorted must be [B,H,N,D] or [B,H,N], got {tuple(X_sorted.shape)}")
