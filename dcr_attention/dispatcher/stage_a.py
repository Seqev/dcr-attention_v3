"""
Stage A — coherence detection and routing orchestration.

The main public function ``dispatch(Q, ...)`` is the single entry point into
the dispatcher.  Decision logic, in order:

  1. If ``N < n_break_even``         → dense  (too short for sort overhead).
  2. If ``confidence < tau_conf``    → dense  (signals disagree — unreliable).
  3. If ``score_v2 < tau_low``       → dense  (no structure detected).
  4. If ``score_v2 < tau_high``      → hybrid (partial structure; narrow window).
  5. Else                             → rank_local (strong structure; full window).

For branches 4 and 5 we additionally call Stage B to resolve the ordering axis.
"""

from __future__ import annotations
from typing import Optional

import torch

from dcr_attention.dispatcher.decision import DispatcherDecision
from dcr_attention.dispatcher.signals import (
    compute_d_eff,
    compute_score_v1,
    compute_score_v2,
)
from dcr_attention.dispatcher.stage_b import compute_axis
from dcr_attention.dispatcher.thresholds import DispatcherConfig, DEFAULT_CONFIG


def _dense_decision(
    score_v1: float,
    score_v2: float,
    s_deff: float,
    d_eff: float,
    confidence: float,
    N: int,
) -> DispatcherDecision:
    return DispatcherDecision(
        mode="dense",
        axis_hat=None,
        k_effective=N,
        score_v1=score_v1,
        score_v2=score_v2,
        s_deff=s_deff,
        d_eff=d_eff,
        confidence=confidence,
        axis_source="none",
    )


def dispatch(
    Q: torch.Tensor,
    k: Optional[int] = None,
    positional_embedding: Optional[torch.Tensor] = None,
    config: DispatcherConfig = DEFAULT_CONFIG,
) -> DispatcherDecision:
    r"""
    Full two-stage dispatcher.

    Parameters
    ----------
    Q
        ``[N, D]``.  For multi-head use, either call once per head and combine
        decisions, or flatten heads into the N dimension — policy is caller's
        choice, the dispatcher is head-agnostic.
    k
        Target window size for ``rank_local`` branch.  Default from config.
    positional_embedding
        Optional ``[N, D]`` additive PE.  **Do not pass for RoPE models**
        (see ``compute_axis`` docstring).
    config
        Thresholds and algorithmic knobs.

    Returns
    -------
    DispatcherDecision
    """
    if Q.dim() != 2:
        raise ValueError(f"expected Q shape [N, D], got {tuple(Q.shape)}")
    N, _ = Q.shape
    k = k if k is not None else config.k_window_default

    d_eff, s_deff = compute_d_eff(Q)
    score_v1, confidence = compute_score_v1(Q, config)
    score_v2 = compute_score_v2(score_v1, s_deff, config)

    # Branch 1: length gate
    if N < config.n_break_even:
        return _dense_decision(score_v1, score_v2, s_deff, d_eff, confidence, N)

    # Branch 2: confidence gate
    if confidence < config.tau_confidence:
        return _dense_decision(score_v1, score_v2, s_deff, d_eff, confidence, N)

    # Branch 3: structure gate (low end)
    if score_v2 < config.tau_low:
        return _dense_decision(score_v1, score_v2, s_deff, d_eff, confidence, N)

    # Structure present — need an axis
    axis, axis_source = compute_axis(Q, positional_embedding=positional_embedding)

    # Branch 4: hybrid (partial structure)
    if score_v2 < config.tau_high:
        return DispatcherDecision(
            mode="hybrid",
            axis_hat=axis,
            k_effective=k // 2,
            score_v1=score_v1,
            score_v2=score_v2,
            s_deff=s_deff,
            d_eff=d_eff,
            confidence=confidence,
            axis_source=axis_source,
        )

    # Branch 5: rank_local (strong structure)
    return DispatcherDecision(
        mode="rank_local",
        axis_hat=axis,
        k_effective=k,
        score_v1=score_v1,
        score_v2=score_v2,
        s_deff=s_deff,
        d_eff=d_eff,
        confidence=confidence,
        axis_source=axis_source,
    )
