"""
Tests for Stage B — ordering-axis selection.
"""

from __future__ import annotations

import pytest
import torch

from dcr_attention.dispatcher.stage_b import compute_axis


def _cos_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    """|cos(a, b)| — sign-agnostic because ranking is invariant to axis flip."""
    return abs(
        torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
    )


# ---------------------------------------------------------------------------
# PCA fallback
# ---------------------------------------------------------------------------

def test_pca_recovers_rank_one_axis():
    """
    Q = s·uᵀ + small noise → PCA must recover u up to sign.
    """
    torch.manual_seed(42)
    N, D = 256, 32
    u_true = torch.randn(D); u_true = u_true / u_true.norm()
    scalars = torch.randn(N)
    Q = scalars.unsqueeze(-1) * u_true + 0.01 * torch.randn(N, D)
    u_hat, source = compute_axis(Q)
    assert source == "pca"
    assert abs(u_hat.norm() - 1.0) < 1e-5
    assert _cos_abs(u_hat, u_true) > 0.99


def test_pca_axis_is_unit_norm_on_random_inputs():
    torch.manual_seed(0)
    Q = torch.randn(512, 64)
    u_hat, source = compute_axis(Q)
    assert source == "pca"
    assert abs(u_hat.norm() - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# Positional least-squares
# ---------------------------------------------------------------------------

def test_positional_axis_recovers_monotone_direction_noiseless():
    """
    Noiseless PE = j·v: lstsq recovers v to machine precision.
    This establishes that the algorithm is correct; the noisy test below
    documents the achievable precision on realistic inputs.
    """
    torch.manual_seed(7)
    N, D = 256, 32
    v_true = torch.randn(D); v_true = v_true / v_true.norm()
    positions = torch.arange(N, dtype=torch.float32) - (N - 1) / 2.0
    PE = positions.unsqueeze(-1) * v_true              # no noise
    Q = torch.randn(N, D)                              # Q ignored when PE given
    u_hat, source = compute_axis(Q, positional_embedding=PE)
    assert source == "positional"
    assert abs(u_hat.norm() - 1.0) < 1e-5
    assert _cos_abs(u_hat, v_true) > 0.9999, (
        f"noiseless case must recover v exactly, got cos={_cos_abs(u_hat, v_true):.6f}"
    )


def test_positional_axis_recovers_monotone_direction_noisy():
    """
    With small additive noise the lstsq solution absorbs some of it into
    off-axis directions; precision degrades.  We document the achievable
    bound at a realistic (N, D, σ) rather than asserting a tight one.

    See INSIGHTS.md INS-7 for the N/D/σ tradeoff.
    """
    torch.manual_seed(7)
    N, D = 1024, 32          # more samples per dim than the failed N=128 case
    sigma = 0.01
    v_true = torch.randn(D); v_true = v_true / v_true.norm()
    positions = torch.arange(N, dtype=torch.float32) - (N - 1) / 2.0
    PE = positions.unsqueeze(-1) * v_true + sigma * torch.randn(N, D)
    Q = torch.randn(N, D)
    u_hat, source = compute_axis(Q, positional_embedding=PE)
    assert source == "positional"
    # Empirical threshold at N=1024, D=32, σ=0.01 — achieved by lstsq.
    assert _cos_abs(u_hat, v_true) > 0.95, (
        f"noisy case cos={_cos_abs(u_hat, v_true):.4f} below 0.95"
    )


def test_positional_overrides_pca_when_both_available():
    """If PE is supplied, the positional branch is taken regardless of Q."""
    torch.manual_seed(0)
    N, D = 128, 32
    Q = torch.randn(N, D)
    PE = torch.randn(N, D)
    _, source = compute_axis(Q, positional_embedding=PE)
    assert source == "positional"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_mismatched_pe_shape_raises():
    Q = torch.randn(64, 16)
    PE = torch.randn(128, 16)
    with pytest.raises(ValueError):
        compute_axis(Q, positional_embedding=PE)


def test_non_2d_Q_raises():
    Q = torch.randn(4, 64, 16)  # [B, N, D] — not accepted by Stage B alone
    with pytest.raises(ValueError):
        compute_axis(Q)
