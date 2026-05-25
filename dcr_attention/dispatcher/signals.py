"""
Scalar "shape-of-distribution" signals computed from a query matrix Q ∈ ℝ^{N×D}.

Mathematical map (all signals are normalised into [0, 1]):

  * ``d_eff``   — participation ratio of the row covariance Σ = Qᶜᵀ Qᶜ / N,
                  with Qᶜ the centred queries.
                  ``d_eff = (tr Σ)² / ‖Σ‖_F²  ∈  [1, D]``.
                  Large d_eff → isotropic; small d_eff → anisotropic.
  * ``s_deff``  — ``clip(1 − (d_eff − 1)/(D − 1), 0, 1)``.
                  1 if d_eff = 1 (rank-1); 0 if d_eff = D (fully isotropic).
  * ``s_local`` — 1 − 2·LNCR where LNCR is the mean k-NN radius normalised by a
                  global reference.  Large s_local → strong local clustering.
  * ``s_spectral`` — 1 minus normalised Shannon entropy of the eigenspectrum of Σ.
                    Large → spectrum concentrated.
  * ``s_intrinsic`` — 1 − (exp(H) − 1)/(D − 1), effective-rank complement of the
                     same spectrum.

The three local-spectral-intrinsic signals are combined into score_v1; score_v2
further mixes in s_deff.
"""

from __future__ import annotations
from typing import Tuple
import math

import torch

from dcr_attention.dispatcher.thresholds import DispatcherConfig, DEFAULT_CONFIG


# ---------------------------------------------------------------------------
# Effective dimensionality (participation ratio)
# ---------------------------------------------------------------------------

def compute_d_eff(Q: torch.Tensor) -> Tuple[float, float]:
    r"""
    Participation ratio of the row covariance of :math:`Q`.

    Let :math:`\bar Q = Q - \tfrac{1}{N}\sum_i Q_i` and
    :math:`\Sigma = \bar Q^\top \bar Q / N \in \mathbb{R}^{D\times D}`.
    Then

    .. math::
        d_{\mathrm{eff}} \;=\; \frac{(\mathrm{tr}\,\Sigma)^2}{\|\Sigma\|_F^2}
        \;\in\; [1, D],

    with :math:`d_{\mathrm{eff}} = 1` for a rank-1 covariance and
    :math:`d_{\mathrm{eff}} = D` for an isotropic one.

    The normalised score is

    .. math::
        s_{d_{\mathrm{eff}}} \;=\;
            \mathrm{clip}\!\left(1 - \frac{d_{\mathrm{eff}} - 1}{D - 1},\,0,\,1\right).

    Parameters
    ----------
    Q
        Shape ``[N, D]``.  The caller is responsible for reshaping multi-head
        tensors before passing in.

    Returns
    -------
    d_eff, s_deff : float, float
    """
    if Q.dim() != 2:
        raise ValueError(f"expected [N, D], got shape {tuple(Q.shape)}")
    N, D = Q.shape
    if D < 2:
        raise ValueError(f"need D >= 2, got D={D}")

    Qc = Q - Q.mean(dim=0, keepdim=True)
    # Σ = Qᶜᵀ Qᶜ / N
    cov = (Qc.T @ Qc) / N
    tr = torch.diagonal(cov).sum()
    frob_sq = (cov * cov).sum()
    d_eff = (tr * tr / (frob_sq + 1e-12)).item()

    s_deff = 1.0 - (d_eff - 1.0) / (D - 1.0)
    s_deff = max(0.0, min(1.0, s_deff))
    return d_eff, s_deff


# ---------------------------------------------------------------------------
# score_v1 components
# ---------------------------------------------------------------------------

def _compute_s_local(Q: torch.Tensor, cfg: DispatcherConfig) -> float:
    r"""
    k-NN Local Neighbourhood Coverage Ratio.

    For the first :math:`m = \text{n\_landmarks}` rows of :math:`Q` we compute the
    squared distance :math:`r_i` to their :math:`k`-th nearest neighbour in the
    full set.  We also compute :math:`r_{\text{global}}`, the mean squared pairwise
    distance among the first :math:`n_{\text{sample}}` rows.

    .. math::
        \mathrm{LNCR} \;=\; \frac{1}{m}\sum_{i=1}^{m}
            \frac{r_i}{r_{\text{global}} + \varepsilon},
        \qquad
        s_{\text{local}} \;=\; \mathrm{clip}(1 - 2\,\mathrm{LNCR},\, 0,\, 1).

    **Reproducibility note (INS-5).** The landmarks and global-sample rows are
    taken as the first ``m`` / ``n_global_sample`` rows — deterministic, but
    potentially biased on long-context inputs where early positions carry
    special tokens.  Future work: seeded random sampling.
    """
    N, D = Q.shape
    m = min(cfg.n_landmarks, N)
    k_nn = min(cfg.top_k_for_mass, N - 1)
    n_sample = min(cfg.n_global_sample, N)

    with torch.no_grad():
        landmarks = Q[:m]                                       # [m, D]
        sqdist = torch.cdist(landmarks, Q, p=2.0) ** 2          # [m, N]

        # (k_nn + 1) smallest includes self (distance 0), we want k-th neighbour
        knn_r, _ = torch.topk(sqdist, k=k_nn + 1, largest=False, dim=1)
        knn_r = knn_r[:, k_nn]                                  # [m]

        sample = Q[:n_sample]
        sample_dists = torch.cdist(sample, sample, p=2.0) ** 2
        r_global = sample_dists.mean()

        lncr = (knn_r / (r_global + 1e-12)).mean().item()

    return max(0.0, min(1.0, 1.0 - 2.0 * lncr))


def _compute_s_spectral_and_intrinsic(Q: torch.Tensor) -> Tuple[float, float]:
    r"""
    Joint computation of the spectral-entropy and effective-rank signals,
    factored together because both derive from the same eigenspectrum.

    .. math::
        H \;=\; -\sum_i p_i \log p_i, \qquad p_i = \lambda_i / \sum_j \lambda_j,

    with :math:`\{\lambda_i\}` the eigenvalues of :math:`\Sigma`.  Then

    .. math::
        s_{\text{spectral}} &\;=\; \mathrm{clip}\!\left(1 - \frac{H}{\log D},\,0,\,1\right), \\
        s_{\text{intrinsic}} &\;=\; \mathrm{clip}\!\left(
            1 - \frac{e^{H} - 1}{D - 1},\,0,\,1\right).
    """
    N, D = Q.shape
    Qc = Q - Q.mean(dim=0, keepdim=True)
    cov = (Qc.T @ Qc) / N
    eigs = torch.linalg.eigvalsh(cov)                           # ascending, real
    eigs = torch.clamp(eigs, min=1e-12)
    p = eigs / eigs.sum()
    H = -(p * torch.log(p)).sum().item()

    max_H = math.log(D)
    s_spectral = max(0.0, min(1.0, 1.0 - H / (max_H + 1e-12)))

    eff_rank = math.exp(H)
    s_intrinsic = max(0.0, min(1.0, 1.0 - (eff_rank - 1.0) / (D - 1.0)))
    return s_spectral, s_intrinsic


def compute_score_v1(
    Q: torch.Tensor,
    cfg: DispatcherConfig = DEFAULT_CONFIG,
) -> Tuple[float, float]:
    r"""
    UCI 5.0 composite score and its confidence.

    .. math::
        \text{score}_1 \;=\;
            w_L\, s_{\text{local}} +
            w_S\, s_{\text{spectral}} +
            w_I\, s_{\text{intrinsic}}.

    ``confidence = 1 − (max − min)`` across the three signals.  Low confidence
    means the three detectors disagree, which we treat as unreliable and route
    to dense.

    Returns
    -------
    score_v1, confidence : float, float
    """
    s_local = _compute_s_local(Q, cfg)
    s_spectral, s_intrinsic = _compute_s_spectral_and_intrinsic(Q)

    score_v1 = (
        cfg.score_v1_weight_local * s_local
        + cfg.score_v1_weight_spectral * s_spectral
        + cfg.score_v1_weight_intrinsic * s_intrinsic
    )

    signals = (s_local, s_spectral, s_intrinsic)
    spread = max(signals) - min(signals)
    confidence = max(0.0, min(1.0, 1.0 - spread))

    return score_v1, confidence


def compute_score_v2(
    score_v1: float,
    s_deff: float,
    cfg: DispatcherConfig = DEFAULT_CONFIG,
) -> float:
    r"""
    .. math::
        \text{score}_2 \;=\;
            \lambda\,\text{score}_1 \;+\; (1 - \lambda)\, s_{d_{\text{eff}}},
        \qquad \lambda = \text{score\_v2\_weight\_v1}.
    """
    lam = cfg.score_v2_weight_v1
    return lam * score_v1 + (1.0 - lam) * s_deff
