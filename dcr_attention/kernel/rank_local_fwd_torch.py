"""
Pure-torch rank-local forward using externally-supplied sort indices.

Role:
  * CPU/CI fallback when Triton is unavailable.
  * Ground-truth oracle for Triton kernel correctness at small N.
  * Debug surface for learnable-projector experiments.

Math identical to ``dcr_attention.reference.rank_local_attention_reference``
with one structural change: sort indices are *inputs*, not computed internally.
This matches Principle P2 of the Phase 1.2 design.
"""

from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def rank_local_fwd_torch(
    Q: torch.Tensor,
    K_sorted: torch.Tensor,
    V_sorted: torch.Tensor,
    rank_of_k: torch.Tensor,
    r_center: torch.Tensor,
    k_window: int,
    scale: Optional[float] = None,
    attention_mask: Optional[torch.Tensor] = None,
    return_lse: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    r"""
    Rank-local attention over pre-sorted K, V.

    Parameters
    ----------
    Q
        ``[B, H, N, D]`` queries in original order.
    K_sorted, V_sorted
        ``[B, H, N, D]`` keys / values gathered by the axis-sort permutation
        (i.e. ``K_sorted[b,h,t,:] = K[b,h,sort_idx[b,h,t],:]``).
    rank_of_k
        ``[B, H, N]``.  Retained in the signature for symmetry with the
        Triton kernel but unused in this path (window mask operates on sorted
        positions directly).
    r_center
        ``[B, H, N]`` per-query insertion rank in the sorted key sequence.
    k_window
        Window size; half is used on each side of ``r_center``.
    scale
        Defaults to :math:`1 / \sqrt{D}`.
    attention_mask
        Optional additive mask of shape ``[B, 1, 1, N]`` indexed in **original**
        key order (not sorted).  Applied after unsort alignment.
        Practical note: for simplicity of the fallback, pass ``None`` unless
        you know what you're doing — see TODO in the body.
    return_lse
        If True, also return the log-sum-exp per row (needed by backward and
        by the Triton path for consistency checks).

    Returns
    -------
    O : ``[B, H, N, D]``
    LSE : ``[B, H, N]`` or ``None``
    """
    B, H, N_q, D = Q.shape
    N_kv = K_sorted.shape[-2]
    if scale is None:
        scale = 1.0 / (D ** 0.5)
    half_k = k_window // 2

    # **Reference oracle precision contract.**  This fallback is the ground
    # truth used to validate the Triton kernel.  An oracle must be at least
    # as numerically accurate as the path it validates, otherwise its noise
    # masquerades as Triton bugs.  We therefore upcast Q/K/V to fp32 for all
    # intermediate math and downcast only at the end, matching Triton's
    # mixed-precision strategy.  See INS-18 for the full rationale.
    out_dtype = Q.dtype
    Q = Q.to(torch.float32)
    K_sorted = K_sorted.to(torch.float32)
    V_sorted = V_sorted.to(torch.float32)

    # Positions vector along the K-side [0, 1, ..., N_kv - 1] broadcast for
    # the window check.  In prefill N_kv == N_q; in decode N_q=1, N_kv=context.
    sorted_positions = torch.arange(N_kv, device=Q.device).view(1, 1, 1, N_kv)  # [1,1,1,N_kv]
    r_center_exp = r_center.unsqueeze(-1)                                       # [B,H,N_q,1]
    in_window = (sorted_positions - r_center_exp).abs() <= half_k               # [B,H,N_q,N_kv]

    # Scores against sorted K (fp32). Since we're in sorted space, softmax has
    # the same mathematical result — softmax is permutation-invariant when V
    # is permuted identically.
    scores = torch.einsum("bhid,bhjd->bhij", Q, K_sorted) * scale               # [B,H,N_q,N_kv] fp32
    scores = scores.masked_fill(~in_window, float("-inf"))

    if attention_mask is not None:
        # TODO(phase 1.3): to support padding masks in this fallback we need
        # to unsort the mask to sorted order.  For now, only unmasked case is
        # covered — correctness tests will assert this precondition.
        raise NotImplementedError(
            "attention_mask in the torch fallback requires sorted-order alignment; "
            "not implemented in Phase 1.2.  Pass None or use the reference."
        )

    # log-sum-exp per row over the window (ignoring -inf bins naturally).
    # torch.logsumexp is stable; output is fp32 because scores is fp32.
    lse = torch.logsumexp(scores, dim=-1) if return_lse else None        # [B,H,N] fp32

    attn = F.softmax(scores, dim=-1)
    attn = torch.nan_to_num(attn, nan=0.0)                                # all-masked rows
    out = torch.einsum("bhij,bhjd->bhid", attn, V_sorted)                 # [B,H,N,D] fp32

    # Cast output back to caller's dtype; LSE stays fp32 by contract.
    return out.to(out_dtype), lse
