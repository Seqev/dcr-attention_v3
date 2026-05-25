r"""
Spectral gap-theory metrics — shared between unit tests and benchmark scripts.

These helpers operate on the free-energy Hessian of attention,

.. math::
    H = -\beta\,\operatorname{Cov}_p(k),\qquad
    p = \mathrm{softmax}(\beta\,\langle q,k\rangle),

and report whether the spectrum has a gap (so that the negative-eigenvalue
count :math:`\dim E_-` is well-posed and threshold-free) and what that count
is.  See INSIGHTS.md INS-31 and ``tests/dispatcher/test_gap_theory.py`` for
the bridge to the dispatcher's ``d_eff(Cov(Q))`` signal.

Numerical contract
------------------
All inputs are promoted to fp32 (or fp64 if already fp64) before the
covariance and eigensolve.  Operating on bf16 keys / queries directly is
numerically unreliable for eigensolves (INS-28).  Callers passing bf16
tensors get fp32 internal precision; callers passing fp64 retain fp64.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


def make_clustered(
    m: int,
    n_per_cluster: int,
    D: int,
    sep: float,
    intra: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    ``m`` Gaussian clusters of ``n_per_cluster`` points in ``D`` dimensions.

    Cluster centres are unit vectors scaled by ``sep`` (inter-cluster
    separation); each point is its centre plus ``intra``-scaled Gaussian
    noise.  Returns ``(X, centres)`` with ``X`` of shape ``[m*n_per_cluster, D]``.
    Large ``sep`` → open spectral gap; small ``sep`` → gap closes.
    """
    g = torch.Generator().manual_seed(seed)
    centres = torch.randn(m, D, generator=g, dtype=torch.float64)
    centres = centres / centres.norm(dim=1, keepdim=True) * sep
    base = centres.repeat_interleave(n_per_cluster, dim=0)
    X = base + intra * torch.randn(base.shape, generator=g, dtype=torch.float64)
    return X, centres


def _promote(x: torch.Tensor) -> torch.Tensor:
    """Promote to fp32 if low-precision; fp64 passes through unchanged."""
    if x.dtype in (torch.float16, torch.bfloat16):
        return x.to(torch.float32)
    return x


def _robust_eigvalsh(H: torch.Tensor) -> np.ndarray:
    r"""
    Eigenvalues of a symmetric matrix, robust against ill-conditioning.

    Strategy:
      1. Try GPU/CPU torch.linalg.eigvalsh on the promoted (fp32+) input.
      2. On _LinAlgError (typical with bf16-derived Hessians having
         near-repeated eigenvalues), retry in fp64 on CPU via numpy.

    Returns a sorted numpy float64 array (ascending).
    """
    Hp = _promote(H)
    try:
        ev = torch.linalg.eigvalsh(Hp).cpu().numpy().astype(np.float64)
    except torch._C._LinAlgError:
        H_np = Hp.cpu().double().numpy()
        H_np = 0.5 * (H_np + H_np.T)        # enforce strict symmetry
        ev = np.linalg.eigvalsh(H_np).astype(np.float64)
    ev.sort()
    return ev


def attention_hessian(
    Q: torch.Tensor,
    K: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    r"""
    The theory's Hessian :math:`H = -\beta\operatorname{Cov}_p(k)` at the
    barycentre query, where :math:`p = \mathrm{softmax}(\beta\langle q,k\rangle)`.

    Linear logits ⇒ the mean-curvature term of the exponential-family
    identity vanishes, so ``H`` is exactly the (negated, scaled) covariance
    of the keys under the softmax weights.

    Parameters
    ----------
    Q : [N_q, D] tensor of query rows (the barycentre query is ``Q.mean(0)``)
    K : [N_kv, D] tensor of key rows
    beta : softmax temperature; for scaled-dot-product attention this is
        the same scale used by the model itself, i.e. :math:`1/\sqrt{D}`.
    """
    Q = _promote(Q)
    K = _promote(K)
    q = Q.mean(dim=0)
    p = torch.softmax(beta * (K @ q), dim=0)
    k_centred = K - p @ K
    cov_p = (k_centred * p[:, None]).T @ k_centred
    return -beta * cov_p


def attention_hessian_for_query(
    q: torch.Tensor,
    K: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    r"""
    Hessian for a SPECIFIC query vector :math:`q`, not the barycentre.

    Same formula as :func:`attention_hessian` but the softmax is taken at the
    supplied ``q`` rather than the mean of a query batch.  This is the form
    used by the real-attention validation: real Llama runs decode one query
    at a time, so the relevant Hessian is per-token-at-position.

    Parameters
    ----------
    q : [D] single query vector
    K : [N_kv, D] key rows currently in cache
    beta : softmax temperature, normally :math:`1/\sqrt{D}` for SDPA.
    """
    q = _promote(q)
    K = _promote(K)
    p = torch.softmax(beta * (K @ q), dim=0)
    k_bar = p @ K
    k_centred = K - k_bar
    cov_p = (k_centred * p[:, None]).T @ k_centred
    return -beta * cov_p


def spectral_gap(H: torch.Tensor) -> Tuple[float, float]:
    r"""
    The gap separating the strongly-negative inter-cluster eigenvalues from
    the near-zero intra-cluster bulk, and the threshold ``θ`` at its midpoint.

    For attention ``H ⪯ 0``: the spectrum has a block of strongly-negative
    inter-cluster eigenvalues and a dense bulk just below zero.  The relevant
    boundary is the one with the largest *multiplicative* jump
    ``|λ_lo| / |λ_hi|`` — a scale-invariant criterion, robust to the fact
    that the overall spectral scale itself shrinks as clusters merge (so any
    absolute "near-zero" tolerance would be miscalibrated across a sweep).

    Returns ``(Δ, θ)`` with ``Δ`` the additive gap width and
    ``θ = -½(λ_lo+λ_hi) > 0`` the midpoint threshold.  When the inter-cluster
    block has itself merged into the bulk the largest ratio is ``O(1)`` and
    the returned ``Δ`` is correspondingly tiny — the caller reads that as a
    closed gap.

    INS-31 bug-pattern note
    -----------------------
    Do NOT "simplify" to ``argmax(diff)``: the largest absolute gap can lie
    INSIDE the inter-cluster block when cluster centres are themselves
    unevenly spaced.  The multiplicative criterion is the correct one.
    """
    ev = _robust_eigvalsh(H)
    floor = 1e-9 * max(float(-ev[0]), 1.0)
    mag = np.maximum(np.abs(ev), floor)
    ratios = mag[:-1] / mag[1:]
    i = int(np.argmax(ratios))
    delta = float(ev[i + 1] - ev[i])
    midpoint = 0.5 * (ev[i] + ev[i + 1])
    return delta, -midpoint


def dim_E_minus(H: torch.Tensor, theta: float) -> int:
    r"""
    Threshold count :math:`\#\{\lambda(H) < -\theta\}`.

    In the gap-regime this equals the rank of the Riesz projector
    :math:`P_\theta = (2\pi i)^{-1}\oint(zI-H)^{-1}dz` and is independent of
    ``θ`` across the whole gap.
    """
    ev = _robust_eigvalsh(H)
    return int((ev < -theta).sum())


def negative_eigenvalues(H: torch.Tensor) -> np.ndarray:
    r"""
    Return the negative eigenvalues of ``H`` sorted **most-negative-first**.

    For attention :math:`H = -\beta\,\mathrm{Cov}_p(k)` is negative-
    semidefinite, so ALL eigenvalues are ≤ 0 in exact arithmetic;
    in floating point a few may sit just above zero due to rounding.
    Returned array contains only the genuinely negative ones (``< 0``).
    """
    ev = _robust_eigvalsh(H)
    return ev[ev < 0.0]


def chi_dominant_mass(H: torch.Tensor) -> float:
    r"""
    Dominant-mode mass fraction :math:`\chi = |\lambda_1| / \sum_i |\lambda_i|`.

    ``|lambda_1|`` is the largest magnitude (most-negative) eigenvalue of
    ``H``; the denominator is over ALL negative eigenvalues.  In
    :math:`[1/n, 1]` for ``n`` negative eigenvalues; ``chi = 1`` means a
    single direction carries all the curvature, ``chi → 1/n`` means a flat
    spectrum.

    A model that "decides via one dominant direction" has chi near 1; a
    model that "spreads attention curvature across many directions"
    has chi well below 1.
    """
    ev_neg = negative_eigenvalues(H)
    if ev_neg.size == 0:
        return float("nan")
    mags = np.abs(ev_neg)
    return float(mags[0] / mags.sum())


def spectral_entropy(H: torch.Tensor) -> float:
    r"""
    Spectral entropy of the negative-eigenvalue mass distribution,

    .. math::
        S_\lambda = -\sum_i \hat\ell_i \log \hat\ell_i,\qquad
        \hat\ell_i = |\lambda_i|/\sum_j|\lambda_j|.

    Naturally bounded in ``[0, log(n)]`` for ``n`` negative eigenvalues:
    zero ⇒ single mode, ``log(n)`` ⇒ flat.  Like Shannon entropy of a
    discrete distribution, but on the eigenvalue-magnitude simplex.
    """
    ev_neg = negative_eigenvalues(H)
    if ev_neg.size == 0:
        return float("nan")
    p = np.abs(ev_neg)
    p = p / p.sum()
    # 0 log 0 = 0 (limit); guard with eps inside log
    p = np.clip(p, 1e-30, 1.0)
    return float(-np.sum(p * np.log(p)))


def dominance_ratio(H: torch.Tensor) -> float:
    r"""
    Dominance ratio :math:`R = |\lambda_1| / |\lambda_2|`.

    The ratio of the two most-negative eigenvalues, a scale-free pair-wise
    statement of how much the first mode dominates the second.  Diverges
    when ``lambda_2 → 0``; returns ``+inf`` if only one negative eigenvalue
    is present.  ``R = 1`` means the top two modes are equal.
    """
    ev_neg = negative_eigenvalues(H)
    if ev_neg.size < 2:
        return float("inf") if ev_neg.size == 1 else float("nan")
    mags = np.abs(ev_neg)
    if mags[1] == 0:
        return float("inf")
    return float(mags[0] / mags[1])


def relative_gap(H: torch.Tensor) -> float:
    r"""
    Scale-free relative gap :math:`\Delta / |\lambda_{\max}|`.

    ``Delta`` from :func:`spectral_gap`; ``lambda_max`` is the largest-
    magnitude eigenvalue.  Filtering by ``relgap > 0.05`` selects the
    genuine-gap regime in a way that is invariant under rescaling of the
    Hessian (and so transports across models / scales).  An absolute
    threshold on ``Delta`` does not transport.

    Returns ``0.0`` if there are no negative eigenvalues (degenerate).
    """
    ev_all = _robust_eigvalsh(H)
    lam_max = float(np.abs(ev_all).max())
    if lam_max == 0:
        return 0.0
    delta, _ = spectral_gap(H)
    return float(delta / lam_max)


__all__ = [
    "make_clustered",
    "attention_hessian",
    "attention_hessian_for_query",
    "spectral_gap",
    "dim_E_minus",
    "negative_eigenvalues",
    "chi_dominant_mass",
    "spectral_entropy",
    "dominance_ratio",
    "relative_gap",
]
