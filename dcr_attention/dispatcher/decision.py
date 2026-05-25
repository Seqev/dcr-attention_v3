"""
Result type for the dispatcher.  A single routing decision plus all diagnostic
signals that produced it — needed for logging, ablation, and the ``routing
distribution`` figure (roadmap Phase 2.4).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional

import torch


Mode = Literal["dense", "hybrid", "rank_local"]
AxisSource = Literal["positional", "pca", "none"]


@dataclass
class DispatcherDecision:
    r"""
    Routing decision for a single attention-layer forward call.

    Attributes
    ----------
    mode
        Chosen path: ``dense`` → full SDPA / FA fallback; ``hybrid`` → rank-local
        with halved window; ``rank_local`` → rank-local with full window.
    axis_hat
        Unit-norm ordering axis :math:`\hat u \in \mathbb{R}^D` for rank-local
        projection, or ``None`` for ``dense``.
    k_effective
        Actual window size used this call (``k`` for rank_local, ``k // 2``
        for hybrid, ``N`` for dense).
    score_v1
        UCI 5.0 composite signal (local + spectral + intrinsic).
    score_v2
        ``score_v2 = λ·score_v1 + (1-λ)·s_deff`` — the quantity compared to
        ``tau_low`` and ``tau_high``.
    s_deff
        Normalised effective-dimensionality score derived from
        :math:`(tr\,\Sigma)^2 / \|\Sigma\|_F^2`.
    d_eff
        Raw participation ratio / effective dimensionality.
    confidence
        Agreement between the three score_v1 signals (1 − spread).  Below
        ``tau_confidence`` → fall back to dense.
    axis_source
        How ``axis_hat`` was obtained: ``positional`` (from PE lstsq) or
        ``pca`` (from SVD of centred Q), or ``none`` if dense.
    """

    mode: Mode
    axis_hat: Optional[torch.Tensor]
    k_effective: int
    score_v1: float
    score_v2: float
    s_deff: float
    d_eff: float
    confidence: float
    axis_source: AxisSource

    def __repr__(self) -> str:  # compact logging-friendly repr
        ax = "-" if self.axis_hat is None else f"{self.axis_source}"
        return (
            f"DispatcherDecision(mode={self.mode}, k={self.k_effective}, "
            f"v2={self.score_v2:.3f}, s_deff={self.s_deff:.3f}, "
            f"conf={self.confidence:.3f}, axis={ax})"
        )
