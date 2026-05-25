"""
Dispatcher configuration.  All numerical thresholds live here as one dataclass
so experiments can override them without touching algorithmic code.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class DispatcherConfig:
    r"""
    Decision thresholds for the DCR two-stage dispatcher.
    Defaults match dcr_sort v2.1.0-rc1 (UCI 5.0 / Energy Projection, DOC_01 §§10, 20).

    Attributes
    ----------
    k_window_default
        Window size for the ``rank_local`` branch.  Ignored in ``hybrid``
        branch (which uses ``k_window_default // 2``).
    n_break_even
        Below this sequence length, dense attention is always cheaper than
        rank-local sort + gather overhead.  Hard fallback to dense.
    tau_confidence
        Minimum agreement between the three score_v1 signals required to trust
        the routing.  If signals disagree (spread high → confidence low),
        fall back to dense.
    tau_low
        Decision threshold for ``dense`` vs ``hybrid``: below → dense.
    tau_high
        Decision threshold for ``hybrid`` vs ``rank_local``: above → rank_local.
    score_v1_weight_local
        Weight of the k-NN Local Neighbourhood Coverage Ratio in score_v1.
    score_v1_weight_spectral
        Weight of the eigenspectrum-entropy term in score_v1.
    score_v1_weight_intrinsic
        Weight of the effective-rank term in score_v1.
    score_v2_weight_v1
        Weight of score_v1 in the composite score_v2; the remainder
        (``1 - score_v2_weight_v1``) weights ``s_deff``.
    n_landmarks
        Number of deterministic landmarks (first rows) used for k-NN
        reference distances in ``s_local``.  See INS-5 — deterministic sampling
        may bias long-context measurements.
    n_global_sample
        Number of deterministic sample rows for the global reference radius
        in ``s_local``.
    top_k_for_mass
        k in k-NN for the local neighbourhood coverage ratio.
    """

    k_window_default: int = 64
    n_break_even: int = 256

    tau_confidence: float = 0.275
    tau_low: float = 0.40
    tau_high: float = 0.62

    # score_v1 = w_L·s_local + w_S·s_spectral + w_I·s_intrinsic
    score_v1_weight_local: float = 0.75
    score_v1_weight_spectral: float = 0.20
    score_v1_weight_intrinsic: float = 0.05

    # score_v2 = λ·score_v1 + (1-λ)·s_deff
    score_v2_weight_v1: float = 0.60

    n_landmarks: int = 64
    n_global_sample: int = 200
    top_k_for_mass: int = 32


DEFAULT_CONFIG = DispatcherConfig()
