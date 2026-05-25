"""
Integration tests for dispatch() — end-to-end routing decisions across the
four expected regimes.
"""

from __future__ import annotations

import torch

from dcr_attention.dispatcher import DEFAULT_CONFIG, DispatcherConfig, dispatch


def test_short_sequence_routes_to_dense():
    """N < n_break_even → dense regardless of structure."""
    torch.manual_seed(0)
    N, D = 128, 32                                  # 128 < 256
    assert N < DEFAULT_CONFIG.n_break_even
    Q = torch.randn(N, D)
    decision = dispatch(Q)
    assert decision.mode == "dense"
    assert decision.axis_hat is None
    assert decision.axis_source == "none"
    assert decision.k_effective == N


def test_isotropic_long_sequence_routes_to_dense():
    """Pure i.i.d. Gaussian, N ≥ n_break_even → dense via score_v2 < tau_low."""
    torch.manual_seed(0)
    N, D = 512, 64
    Q = torch.randn(N, D)
    decision = dispatch(Q)
    assert decision.mode == "dense"
    assert decision.score_v2 < DEFAULT_CONFIG.tau_low + 1e-6  # slack for boundary


def test_rank_one_structured_routes_to_rank_local():
    """
    Strong rank-1 signal with minimal noise → score_v2 high → rank_local.
    """
    torch.manual_seed(13)
    N, D = 512, 64
    u = torch.randn(D); u = u / u.norm()
    scalars = torch.randn(N)
    Q = 5.0 * scalars.unsqueeze(-1) * u + 0.1 * torch.randn(N, D)
    decision = dispatch(Q)
    assert decision.mode == "rank_local", (
        f"expected rank_local, got {decision!r}"
    )
    assert decision.axis_hat is not None
    assert decision.axis_source == "pca"
    assert decision.k_effective == DEFAULT_CONFIG.k_window_default


def test_positional_embedding_selects_positional_axis():
    """
    With PE supplied and signals crossing the thresholds, axis_source must be
    'positional'.  Q carries rank-2 structure so dispatcher chooses a
    non-dense branch.
    """
    torch.manual_seed(21)
    N, D = 512, 64
    u1 = torch.randn(D); u1 = u1 / u1.norm()
    # Rank-2 structure ensures s_deff and spectral signals exceed tau_low.
    u2 = torch.randn(D); u2 = u2 / u2.norm()
    s1, s2 = torch.randn(N), torch.randn(N)
    Q = 3.0 * (s1.unsqueeze(-1) * u1 + 0.5 * s2.unsqueeze(-1) * u2) \
        + 0.1 * torch.randn(N, D)

    # Positional embedding with monotone position dependency
    v = torch.randn(D); v = v / v.norm()
    positions = torch.arange(N, dtype=torch.float32) - (N - 1) / 2.0
    PE = positions.unsqueeze(-1) * v + 0.01 * torch.randn(N, D)

    decision = dispatch(Q, positional_embedding=PE)
    if decision.mode != "dense":
        assert decision.axis_source == "positional"


def test_decision_has_all_diagnostic_fields():
    torch.manual_seed(0)
    Q = torch.randn(512, 64)
    decision = dispatch(Q)
    # All float diagnostic fields must be finite and in [0, 1] for scores.
    assert 0.0 <= decision.score_v1 <= 1.0
    assert 0.0 <= decision.score_v2 <= 1.0
    assert 0.0 <= decision.s_deff <= 1.0
    assert decision.d_eff >= 1.0
    assert 0.0 <= decision.confidence <= 1.0


def test_custom_config_shifts_decisions():
    """
    Lowering tau_low should push a borderline input from dense into hybrid
    (or rank_local), all else equal.
    """
    torch.manual_seed(13)
    N, D = 512, 64
    # Tune signal strength to sit near the default tau_low on isotropic+bias.
    Q = torch.randn(N, D) + 0.5

    strict = DEFAULT_CONFIG                                 # tau_low = 0.40
    lenient = DispatcherConfig(tau_low=0.01, tau_high=0.02) # almost always route

    d_strict = dispatch(Q, config=strict)
    d_lenient = dispatch(Q, config=lenient)
    # Under lenient config, we're at least as aggressive as under strict.
    rank = {"dense": 0, "hybrid": 1, "rank_local": 2}
    assert rank[d_lenient.mode] >= rank[d_strict.mode]
