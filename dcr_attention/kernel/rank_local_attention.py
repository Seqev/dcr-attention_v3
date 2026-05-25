"""
Public rank-local attention API.

Wraps the kernel (Triton on CUDA, torch fallback elsewhere) in a
``torch.autograd.Function`` with self-adaptive save-for-backward: nothing is
retained when all inputs are no-grad (pure inference path).

**Phase 1.3:** Q is sorted by axis projection before kernel call, output is
permuted back to original Q-order.  This is invisible to callers; the public
contract still takes Q in any order and returns output in the same order.
See ``docs/design/phase1_3_q_sort.md``.
"""

from __future__ import annotations
from typing import Optional

import torch

from dcr_attention.kernel.sort_helpers import (
    prepare_sort_indices,
    gather_by_sort_idx,
    scatter_by_inv_perm,
)
from dcr_attention.kernel.rank_local_fwd_torch import rank_local_fwd_torch

try:
    from dcr_attention.kernel.rank_local_fwd_triton import (
        rank_local_fwd_triton,
        TRITON_AVAILABLE,
    )
except ImportError:   # pragma: no cover — defensive, shouldn't trigger
    TRITON_AVAILABLE = False
    rank_local_fwd_triton = None  # type: ignore[assignment]


_EPS = 1e-12


def _normalize_axis(axis: torch.Tensor) -> torch.Tensor:
    r"""
    Unit-normalise ``axis`` along its last dim.

    Supports:
      * ``[D]``      — single shared axis.  Normalised over the only dim.
      * ``[B, H, D]`` — per-(b, h) axes.  Normalised independently per slot.
    """
    if axis.dim() == 1:
        return axis / (axis.norm() + _EPS)
    elif axis.dim() == 3:
        return axis / (axis.norm(dim=-1, keepdim=True) + _EPS)
    else:
        raise ValueError(
            f"axis must be [D] or [B, H, D], got shape {tuple(axis.shape)}"
        )


class RankLocalAttentionFn(torch.autograd.Function):
    """
    Forward: dispatch to Triton (CUDA) or torch fallback.
    Backward: not implemented yet (Phase 1.4 territory).
    """

    @staticmethod
    def forward(                                                # type: ignore[override]
        ctx,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        axis: torch.Tensor,
        k_window: int,
        scale: Optional[float],
    ) -> torch.Tensor:
        B, H, N, D = Q.shape
        if scale is None:
            scale = 1.0 / (D ** 0.5)

        axis_u = _normalize_axis(axis)

        # 1. Sort indices  (pure torch, P2 in design doc)
        idx = prepare_sort_indices(Q, K, axis_u)

        # 2. Pre-gather K and V along K-sort, Q along Q-sort.
        #    All three tensors enter the kernel in sorted order; output comes
        #    out in sorted Q-order; we reverse via scatter_by_inv_perm.
        K_sorted = gather_by_sort_idx(K, idx.sort_idx_k)
        V_sorted = gather_by_sort_idx(V, idx.sort_idx_k)
        Q_sorted = gather_by_sort_idx(Q, idx.sort_idx_q)

        # 3. Forward pass — Triton on CUDA, torch fallback otherwise.
        #    The torch fallback materialises the full N×N mask anyway, so
        #    Q-sort gains it nothing.  We pass Q_sorted to it for signature
        #    consistency; it produces output in sorted Q-order, same as Triton.
        use_triton = TRITON_AVAILABLE and Q.is_cuda
        if use_triton:
            O_sorted, LSE_sorted = rank_local_fwd_triton(
                Q_sorted, K_sorted, V_sorted,
                idx.rank_of_k, idx.r_center,
                k_window=k_window, scale=scale,
            )
        else:
            O_sorted, LSE_sorted = rank_local_fwd_torch(
                Q_sorted, K_sorted, V_sorted,
                idx.rank_of_k, idx.r_center,
                k_window=k_window, scale=scale, return_lse=True,
            )

        # 4. Restore output to original Q-order.
        O = scatter_by_inv_perm(O_sorted, idx.inv_q_perm)
        LSE = scatter_by_inv_perm(LSE_sorted, idx.inv_q_perm) if LSE_sorted is not None else None

        # 5. Self-adaptive save  (P1 in design doc)
        needs_backward = any(t.requires_grad for t in (Q, K, V, axis))
        if needs_backward:
            ctx.save_for_backward(
                Q, K, V, axis_u,
                idx.sort_idx_k, idx.sort_idx_q, idx.inv_q_perm,
                idx.r_center,
                O, LSE,
            )
            ctx.k_window = k_window
            ctx.scale = scale
        ctx.needs_backward = needs_backward

        return O

    @staticmethod
    def backward(ctx, grad_O):                                  # type: ignore[override]
        if not ctx.needs_backward:
            # Should not be reached — autograd wouldn't call us otherwise.
            return (None, None, None, None, None, None)
        raise NotImplementedError(
            "rank-local backward not implemented yet. "
            "Phase 1.3 forward contract: ctx holds Q, K, V, axis, sort_idx_k, "
            "sort_idx_q, inv_q_perm, r_center, O, LSE.  See "
            "docs/design/phase1_forward_kernel.md §6 and "
            "docs/design/phase1_3_q_sort.md §3."
        )


def rank_local_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    axis: torch.Tensor,
    k_window: int = 64,
    scale: Optional[float] = None,
) -> torch.Tensor:
    r"""
    Rank-local attention along an ordering axis.

    .. math::
        \mathrm{out}_i \;=\; \sum_{j : |r(j) - r_c(i)| \le k/2}
            \mathrm{softmax}_j\!\left(\tfrac{1}{\sqrt D} \langle Q_i, K_j\rangle\right) V_j,

    with :math:`r(j), r_c(i)` defined in
    ``dcr_attention.kernel.sort_helpers.prepare_sort_indices``.

    Parameters
    ----------
    Q, K, V
        ``[B, H, N, D]``.  Caller-side order is preserved on output.
    axis
        ``[D]`` for a single shared axis, or ``[B, H, D]`` for per-(b, h)
        axes (Phase 2a — Llama wrapper passes per-head PCA axes).
        Normalised internally; caller need not pass a unit vector.
    k_window
        Window size (``2 * half_k``).
    scale
        Defaults to :math:`1/\sqrt D`.

    Returns
    -------
    ``[B, H, N, D]``.
    """
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
            f"axis must be [D] or [B, H, D], got {axis.dim()}-D shape {tuple(axis.shape)}"
        )
    return RankLocalAttentionFn.apply(Q, K, V, axis, k_window, scale)
