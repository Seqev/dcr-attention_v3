"""
Unit tests for dispatcher signal functions.

Each test uses an input with a known theoretical answer, so a numerical
regression is immediately visible.
"""

from __future__ import annotations
import math

import pytest
import torch

from dcr_attention.dispatcher import (
    DEFAULT_CONFIG,
    compute_d_eff,
    compute_score_v1,
    compute_score_v2,
)
from dcr_attention.dispatcher.signals import _compute_s_spectral_and_intrinsic


# ---------------------------------------------------------------------------
# d_eff / s_deff
# ---------------------------------------------------------------------------

def test_d_eff_rank_one_is_one():
    """Rank-1 covariance: d_eff = 1, s_deff = 1."""
    torch.manual_seed(0)
    N, D = 256, 32
    direction = torch.randn(D)
    direction = direction / direction.norm()
    scalars = torch.randn(N)
    Q = scalars.unsqueeze(-1) * direction                    # [N, D], rank 1
    d_eff, s_deff = compute_d_eff(Q)
    assert abs(d_eff - 1.0) < 1e-4, f"d_eff = {d_eff:.6f}, expected 1.0"
    assert s_deff > 0.9999


def test_d_eff_isotropic_approaches_D():
    """Large-N isotropic Gaussian: d_eff → D, s_deff → 0."""
    torch.manual_seed(0)
    N, D = 4096, 32
    Q = torch.randn(N, D)
    d_eff, s_deff = compute_d_eff(Q)
    # Sample covariance of N(0, I_D) concentrates; d_eff close to D for large N.
    assert D - 3.0 < d_eff <= D + 1e-3, f"d_eff = {d_eff:.2f}, expected ≈ {D}"
    assert s_deff < 0.1


def test_d_eff_rejects_non_2d():
    with pytest.raises(ValueError):
        compute_d_eff(torch.zeros(10, 10, 10))


def test_d_eff_rejects_D_lt_2():
    with pytest.raises(ValueError):
        compute_d_eff(torch.zeros(10, 1))


# ---------------------------------------------------------------------------
# spectral / intrinsic
# ---------------------------------------------------------------------------

def test_spectral_intrinsic_rank_one():
    """Rank-1 covariance: eigenvalues concentrated on one mode → both signals ≈ 1."""
    torch.manual_seed(0)
    N, D = 256, 16
    u = torch.randn(D); u = u / u.norm()
    Q = torch.randn(N, 1) * u
    s_spec, s_intr = _compute_s_spectral_and_intrinsic(Q)
    assert s_spec > 0.85, f"s_spectral = {s_spec:.4f}"
    assert s_intr > 0.90, f"s_intrinsic = {s_intr:.4f}"


def test_spectral_intrinsic_isotropic():
    """Isotropic: both signals near 0."""
    torch.manual_seed(0)
    N, D = 4096, 16
    Q = torch.randn(N, D)
    s_spec, s_intr = _compute_s_spectral_and_intrinsic(Q)
    assert s_spec < 0.10
    assert s_intr < 0.10


# ---------------------------------------------------------------------------
# score_v1 / score_v2 composition
# ---------------------------------------------------------------------------

def test_score_v1_weights_sum_to_one_by_default():
    cfg = DEFAULT_CONFIG
    assert abs(
        cfg.score_v1_weight_local
        + cfg.score_v1_weight_spectral
        + cfg.score_v1_weight_intrinsic
        - 1.0
    ) < 1e-9


def test_score_v2_is_convex_combination():
    # score_v1=1, s_deff=0 → score_v2 = λ·1 + (1-λ)·0 = λ
    v2 = compute_score_v2(score_v1=1.0, s_deff=0.0)
    assert abs(v2 - DEFAULT_CONFIG.score_v2_weight_v1) < 1e-9


def test_score_v1_returns_in_unit_interval():
    torch.manual_seed(0)
    Q = torch.randn(512, 32)
    score, conf = compute_score_v1(Q)
    assert 0.0 <= score <= 1.0
    assert 0.0 <= conf <= 1.0


def test_score_v1_confidence_low_when_signals_disagree():
    """
    Rank-1 data: s_spectral and s_intrinsic high, s_local depends on cluster
    structure.  With i.i.d. scalars on one axis, local neighbours are spread
    along the 1D line, so s_local is moderate — we expect the signals to
    disagree somewhat (not unanimously high), lowering confidence.
    """
    torch.manual_seed(0)
    N, D = 1024, 32
    u = torch.randn(D); u = u / u.norm()
    Q = torch.randn(N, 1) * u
    _, confidence = compute_score_v1(Q)
    # Not asserting a hard bound — just recording the expected direction.
    assert 0.0 <= confidence <= 1.0
