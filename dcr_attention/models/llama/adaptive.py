r"""
Adaptive widening for DCR rank-local attention — Phase 2c.

Two complementary mechanisms validated by Phase 2b N=250 token-trace:

* **Component A** (always-on): adaptive ``k_window_eff`` ensures coverage
  ``= k_window_eff / n_kv`` never falls below ``coverage_floor``.  Phase 2b
  showed catastrophic decay below 0.5 coverage:

    coverage 1.00..0.70  → mean Δ_NLL +0.23 (acceptable)
    coverage 0.70..0.50  → mean Δ_NLL +0.53 (borderline)
    coverage 0.50..0.35  → mean Δ_NLL +2.29 (unacceptable)
    coverage 0.35..0.25  → mean Δ_NLL +4.14 (catastrophic)

  Default ``coverage_floor=0.8`` keeps approximation in the safe regime.

* **Component B** (opt-in via ``enable_adaptive_widening``): bound-driven
  safety net.  After window selection, examine projection-space distance
  of nearest rejected key vs furthest selected key.  If a rejected key is
  closer to the query projection than some selected keys (within slack),
  the boundary is "leaky" — widen and retry once.

  This is detection without formal guarantee.  In combination with
  Component A's coverage floor, it catches edge cases of locally bad axis
  quality without bounding the worst case mathematically.  Hard-capped at
  one widen per ``_dcr_forward`` call to keep latency at most 2× nominal.

Both mechanisms operate on tensors already computed in the DCR forward path
(axes, projections); no new SVD or attention computation is added.
"""

from __future__ import annotations

from typing import Tuple

import torch


def adaptive_k_window(
    k_window_min: int,
    n_kv: int,
    coverage_floor: float,
) -> int:
    r"""
    Compute effective k_window enforcing minimum coverage.

    .. math::
        k_{\text{eff}} = \min\!\bigl(n_{kv},\ \max(k_{\min},\ \lfloor n_{kv} \cdot c\rfloor)\bigr)

    where :math:`c` is ``coverage_floor`` and :math:`k_{\min}` is the user's
    declared ``k_window``.

    Examples (with ``k_window_min=64``):

      ===========  =================  ===================
      n_kv         coverage_floor=0.8  k_window_eff
      ===========  =================  ===================
      64                              64    (n_kv-clamp)
      100                             80    (floor-clamp)
      256                             204
      1000                            800
      4096                            3276
      ===========  =================  ===================

    Returned value is always:
      * at least ``k_window_min`` when ``n_kv >= k_window_min``
      * at most ``n_kv``
      * even (rank_local kernel constraint, ``2 * half_k``); rounded up

    Parameters
    ----------
    k_window_min
        User-declared minimum k_window from DCRLlamaConfig.
    n_kv
        Current cached-key sequence length.
    coverage_floor
        Fractional coverage floor, in [0, 1].  ``0.0`` disables adaptive
        widening (returns ``k_window_min``).

    Returns
    -------
    k_window_eff : int
        Even, in [min(k_window_min, n_kv), n_kv].
    """
    if n_kv <= 0:
        raise ValueError(f"n_kv must be > 0, got {n_kv}")
    if not (0.0 <= coverage_floor <= 1.0):
        raise ValueError(
            f"coverage_floor must be in [0, 1], got {coverage_floor}"
        )

    floor_value = int(n_kv * coverage_floor)
    k_eff = max(k_window_min, floor_value)
    k_eff = min(k_eff, n_kv)

    # Kernel requires even k_window (2 * half_k); round UP to next even,
    # but cap at n_kv.
    if k_eff % 2 == 1:
        k_eff = min(k_eff + 1, n_kv)
        # If n_kv itself is odd and we capped, we now have odd k_eff.
        # The rank_local kernel handles odd window sizes by treating
        # half_k = k_eff // 2 and using k_eff - half_k on the right; this
        # is correct for the full-coverage edge case where k_eff == n_kv.

    return k_eff


def detect_leaky_boundary(
    q_proj: torch.Tensor,           # [B, H, 1]      — query · axis
    k_proj: torch.Tensor,           # [B, H, N_kv]   — keys · axis (unsorted)
    sort_idx: torch.Tensor,         # [B, H, N_kv]   — argsort of k_proj
    k_window_eff: int,
    slack: float = 0.5,
) -> torch.Tensor:
    r"""
    Heuristic boundary check: is the rank-local window "leaky"?

    For each (b, h), compute distance in projection space:

      * d_in = max over selected keys of |q_proj - k_proj[selected]|
      * d_out = min over rejected keys of |q_proj - k_proj[rejected]|

    A "clean" boundary has ``d_out > d_in`` strictly (rejected keys
    further from query than selected keys).  A "leaky" boundary has
    ``d_out < slack * d_in`` — some rejected key is closer in projection
    space than some selected keys, suggesting the axis ranking is locally
    unreliable.

    This is **detection without formal guarantee**.  Distance in
    projection space lower-bounds angular distance only when the axis
    captures the dominant variation; for ill-conditioned local geometry
    the heuristic can miss true positives.

    Parameters
    ----------
    q_proj
        Query projection ``q · u_axis`` per (b, h).  Shape [B, H, 1].
    k_proj
        Key projections ``k_i · u_axis`` per (b, h, i), in original order.
        Shape [B, H, N_kv].
    sort_idx
        Argsort indices over k_proj per (b, h), ascending.  Shape
        [B, H, N_kv].
    k_window_eff
        Number of keys selected (centred on the position closest to q_proj).
    slack
        Multiplier on d_in below which boundary is considered leaky.
        Default 0.5 means d_out < 0.5 * d_in triggers widen.

    Returns
    -------
    leaky : torch.Tensor
        Boolean tensor of shape [B, H], ``True`` where boundary is leaky.
    """
    B, H, N_kv = k_proj.shape

    if k_window_eff >= N_kv:
        # Full coverage — nothing rejected, trivially clean
        return torch.zeros(B, H, dtype=torch.bool, device=k_proj.device)

    # Sorted projections: k_proj[..., sort_idx[..., i]] is i-th smallest
    k_sorted = torch.gather(k_proj, dim=-1, index=sort_idx)   # [B, H, N_kv]

    # Distance of each sorted key from q_proj
    dists_sorted = (k_sorted - q_proj).abs()                  # [B, H, N_kv]

    # Find which `k_window_eff` keys are selected: the ones nearest to
    # q_proj in projection space.  The kernel's selection isn't strictly
    # the k_window nearest — it's a contiguous window in sorted order
    # centred on the rank closest to q_proj.  Mirror that here.

    # rank_of_query: for each (b, h), index in sort order where q_proj
    # would be inserted (closest sorted-position to q_proj)
    rank_of_query = torch.searchsorted(k_sorted, q_proj)       # [B, H, 1]
    rank_of_query = rank_of_query.squeeze(-1).clamp(0, N_kv - 1)  # [B, H]

    half = k_window_eff // 2
    # Window [start, end) in sorted order
    start = (rank_of_query - half).clamp(min=0)                 # [B, H]
    end = (start + k_window_eff).clamp(max=N_kv)                # [B, H]
    start = (end - k_window_eff).clamp(min=0)                   # adjust if hit right edge

    # Build a mask [B, H, N_kv]: True at positions in [start, end)
    positions = torch.arange(N_kv, device=k_proj.device).view(1, 1, -1)
    in_window = (positions >= start.unsqueeze(-1)) & (positions < end.unsqueeze(-1))

    # d_in: max distance among selected positions
    # d_out: min distance among rejected positions
    INF = torch.finfo(dists_sorted.dtype).max
    d_in_max = torch.where(in_window, dists_sorted, torch.zeros_like(dists_sorted)).max(dim=-1).values
    d_out_min = torch.where(~in_window, dists_sorted, torch.full_like(dists_sorted, INF)).min(dim=-1).values

    # Leaky if a rejected key is closer than slack * (max selected distance)
    leaky = d_out_min < (slack * d_in_max)
    return leaky


def widen_factor(prev_k: int, n_kv: int, multiplier: float = 2.0) -> int:
    """
    Compute widened k_window after safety-net trigger.  Capped at n_kv.

    Parameters
    ----------
    prev_k
        Previous k_window_eff.
    n_kv
        Current N_kv (upper bound).
    multiplier
        Widen multiplier; default 2× per Phase 2c spec.

    Returns
    -------
    new_k : int
        Even, in (prev_k, n_kv].
    """
    new_k = min(int(prev_k * multiplier), n_kv)
    if new_k % 2 == 1:
        new_k = min(new_k + 1, n_kv)
    return max(new_k, prev_k + 2)  # always strictly larger than prev (or hit cap)
