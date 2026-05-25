"""
DCR two-stage dispatcher.

Public API:
    * ``dispatch(Q, ...)``        — end-to-end routing decision.
    * ``DispatcherDecision``       — result dataclass.
    * ``DispatcherConfig``         — thresholds / weights (mutable via override).
    * ``compute_d_eff``, ``compute_score_v1``, ``compute_score_v2`` — signals.
    * ``compute_axis``             — Stage B alone (positional / PCA).

See individual module docstrings for the mathematical formulation.
"""

from dcr_attention.dispatcher.decision import (
    DispatcherDecision,
    Mode,
    AxisSource,
)
from dcr_attention.dispatcher.projector import (
    AxisProjector,
    axis_cosine_similarity,
    cosine_alignment_loss,
    pca_axis_target,
    synthetic_rank_k_batch,
)
from dcr_attention.dispatcher.signals import (
    compute_d_eff,
    compute_score_v1,
    compute_score_v2,
)
from dcr_attention.dispatcher.stage_a import dispatch
from dcr_attention.dispatcher.stage_b import compute_axis
from dcr_attention.dispatcher.thresholds import DEFAULT_CONFIG, DispatcherConfig

__all__ = [
    "dispatch",
    "DispatcherDecision",
    "DispatcherConfig",
    "DEFAULT_CONFIG",
    "Mode",
    "AxisSource",
    "compute_d_eff",
    "compute_score_v1",
    "compute_score_v2",
    "compute_axis",
    "AxisProjector",
    "pca_axis_target",
    "cosine_alignment_loss",
    "axis_cosine_similarity",
    "synthetic_rank_k_batch",
]
