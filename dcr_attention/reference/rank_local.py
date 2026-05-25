"""
Reference PyTorch implementation of rank-local attention.

Ported unchanged (semantically) from `dcr_sort_v2.1.0-rc1/dcr_benchmark/rank_local_attention.py`.
Used as ground-truth oracle for Triton kernel correctness tests at small N.

**Important**: this implementation materialises a full [B, H, N, N] mask, hence
is O(N^2) in both memory and compute. It is a *correctness reference only*;
the efficiency claim of DCR-Attention comes from the block-sparse Triton kernel,
not from this file. See INSIGHTS.md INS-1.
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn.functional as F


def rank_local_attention_reference(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    axis: torch.Tensor,
    k_window: int = 64,
    attention_mask: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
) -> torch.Tensor:
    r"""
    Rank-local attention via explicit [B, H, N, N] mask (reference only).

    Mathematical model.
    Let :math:`z_i = \langle Q_i, u \rangle` and :math:`\zeta_j = \langle K_j, u \rangle`
    be scalar projections onto the ordering axis :math:`u \in \mathbb{R}^D`.
    Let :math:`\pi: [N] \to [N]` be the permutation sorting keys by :math:`\zeta`
    ascending (so :math:`\zeta_{\pi(1)} \le \dots \le \zeta_{\pi(N)}`), and let
    :math:`r(j) = \pi^{-1}(j)` be the rank of key :math:`j`. For query :math:`i`,
    define the insertion rank :math:`r_c(i) = |\{j : \zeta_j < z_i\}|`.
    The rank-local attention with window :math:`k` is

    .. math::
        \mathrm{out}_i \;=\; \sum_{j : |r(j) - r_c(i)| \le k/2}
            \mathrm{softmax}_j\!\left(\tfrac{1}{\sqrt{D}}\, \langle Q_i, K_j\rangle\right) V_j.

    Parameters
    ----------
    Q, K, V
        Shape ``[B, H, N, D]``.
    axis
        Shape ``[D]``, treated as the ordering axis :math:`u`. No re-normalization —
        passed in as-is by the dispatcher (Stage B).
    k_window
        Size of the symmetric rank window per query (2 * half_k).
    attention_mask
        Optional additive mask of shape ``[B, 1, 1, N]`` (e.g. -inf for padding).
    scale
        Defaults to :math:`1/\sqrt{D}`.

    Returns
    -------
    torch.Tensor
        ``[B, H, N, D]``.

    Complexity
    ----------
    Memory: O(B · H · N²) from the window mask tensor. Compute: O(B · H · N² · D).
    Use only for N ≲ 4096. For production / long context, call the Triton kernel
    (``dcr_attention.kernel.rank_local_fwd``).

    Scope
    -----
    This is the **paper-grade reference**: prefill-only (Q.shape == K.shape).
    The production kernel and its torch fallback support decode shape
    (Q.N=1, K.N=N_kv) but the reference does not — it is an oracle for the
    correctness contract of full attention with rank-window mask, not a
    production-flexible function.
    """
    B, H, N, D = Q.shape
    if scale is None:
        scale = 1.0 / (D ** 0.5)

    # z_k, z_q: projections onto the ordering axis u. Shape [B, H, N].
    #   z_k[b,h,j] = <K[b,h,j,:], u>;  z_q[b,h,i] = <Q[b,h,i,:], u>
    z_k = torch.einsum("bhnd,d->bhn", K, axis)
    z_q = torch.einsum("bhnd,d->bhn", Q, axis)

    half_k = k_window // 2

    # sort_idx[b,h,t] = index of the key that is at sorted position t.
    # rank_of_k[b,h,j] = sorted position of key j, i.e. pi^{-1}(j).
    sort_idx = torch.argsort(z_k, dim=-1)          # [B, H, N]
    rank_of_k = torch.argsort(sort_idx, dim=-1)    # [B, H, N]

    # z_k_sorted[b,h,t] = zeta_{pi(t)}, non-decreasing along t.
    z_k_sorted = torch.gather(z_k, -1, sort_idx)   # [B, H, N]

    # r_center[b,h,i] = |{j : zeta_j < z_q[b,h,i]}|.
    # torch.searchsorted requires contiguous sorted arrays.
    r_center = torch.searchsorted(
        z_k_sorted.contiguous(), z_q.contiguous()
    )                                              # [B, H, N]
    r_center = r_center.clamp(0, N - 1)

    # Build rank-window mask:  |r(j) - r_c(i)| <= half_k
    #   rank_of_k_expanded: [B, H, 1, N]     (broadcast over queries i)
    #   r_center_expanded : [B, H, N, 1]     (broadcast over keys j)
    rank_of_k_expanded = rank_of_k.unsqueeze(-2)    # [B, H, 1, N]
    r_center_expanded = r_center.unsqueeze(-1)      # [B, H, N, 1]
    in_window = (rank_of_k_expanded - r_center_expanded).abs() <= half_k

    # Standard scaled dot product, masked outside the rank window.
    scores = torch.einsum("bhid,bhjd->bhij", Q, K) * scale
    scores = scores.masked_fill(~in_window, float("-inf"))
    if attention_mask is not None:
        scores = scores + attention_mask

    attn = F.softmax(scores, dim=-1)
    # Rows whose window is entirely masked (e.g. full-padding) produce NaN
    # after softmax over all -inf; we zero them out.
    attn = torch.nan_to_num(attn, nan=0.0)

    out = torch.einsum("bhij,bhjd->bhid", attn, V)
    return out


def dense_attention_reference(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
) -> torch.Tensor:
    r"""
    Reference scaled dot-product attention (no flash, no sparsity).

    .. math::
        \mathrm{out}_i \;=\; \sum_{j=1}^{N}
            \mathrm{softmax}_j\!\left(\tfrac{1}{\sqrt{D}} \langle Q_i, K_j\rangle\right) V_j.

    Used to verify the identity ``rank_local_attention_reference(k=2N) ≡ dense``.
    """
    _, _, _, D = Q.shape
    if scale is None:
        scale = 1.0 / (D ** 0.5)
    scores = torch.einsum("bhid,bhjd->bhij", Q, K) * scale
    if attention_mask is not None:
        scores = scores + attention_mask
    attn = F.softmax(scores, dim=-1)
    return torch.einsum("bhij,bhjd->bhid", attn, V)
