"""
Exhaustive tests of ``routing._route``.

Pure logic — no tensors, no GPU.  Goal: verify the decision tree from
``docs/design/phase2a_llama_wrapper.md`` §3 is implemented correctly,
including all early-exit branches.
"""

from __future__ import annotations

import pytest

from dcr_attention.models.llama.config import DCRLlamaConfig
from dcr_attention.models.llama.routing import Branch, route, explain_route


# ---------------------------------------------------------------------------
# Default-config decision tree
# ---------------------------------------------------------------------------

def test_prefill_always_sdpa():
    """N_q > 1 must always route to SDPA (causal correctness)."""
    cfg = DCRLlamaConfig()  # T_dispatch=4096
    for N_q in (2, 100, 16384):
        for N_kv in (N_q, 100_000):
            assert route(N_q, N_kv, layer_idx=0, cfg=cfg) is Branch.SDPA, (
                f"prefill N_q={N_q} N_kv={N_kv} should be SDPA"
            )


def test_decode_below_threshold_sdpa():
    """N_q == 1 and N_kv < T_dispatch → SDPA (short context)."""
    cfg = DCRLlamaConfig(T_dispatch=4096)
    for N_kv in (1, 100, 4095):
        assert route(1, N_kv, layer_idx=0, cfg=cfg) is Branch.SDPA, (
            f"short decode N_kv={N_kv} should be SDPA"
        )


def test_decode_at_threshold_dcr():
    """N_q == 1 and N_kv == T_dispatch → DCR (boundary case, ≥)."""
    cfg = DCRLlamaConfig(T_dispatch=4096)
    assert route(1, 4096, layer_idx=0, cfg=cfg) is Branch.DCR


def test_decode_above_threshold_dcr():
    """N_q == 1 and N_kv > T_dispatch → DCR."""
    cfg = DCRLlamaConfig(T_dispatch=4096)
    for N_kv in (4097, 16384, 131072):
        assert route(1, N_kv, layer_idx=0, cfg=cfg) is Branch.DCR


# ---------------------------------------------------------------------------
# Master switch
# ---------------------------------------------------------------------------

def test_enable_dcr_false_overrides_everything():
    """enable_dcr=False forces SDPA regardless of N_kv."""
    cfg = DCRLlamaConfig(enable_dcr=False, T_dispatch=4096)
    for N_q in (1, 100):
        for N_kv in (1, 100, 4096, 16384, 131072):
            assert route(N_q, N_kv, layer_idx=0, cfg=cfg) is Branch.SDPA


# ---------------------------------------------------------------------------
# Layer filter
# ---------------------------------------------------------------------------

def test_layer_filter_excludes_layers():
    """layer_filter only allows DCR for whitelisted layers."""
    cfg = DCRLlamaConfig(
        T_dispatch=4096,
        layer_filter=frozenset({16, 17, 18}),
    )
    # Whitelisted layer + decode + long context = DCR
    assert route(1, 8192, layer_idx=16, cfg=cfg) is Branch.DCR
    # Non-whitelisted layer = SDPA
    assert route(1, 8192, layer_idx=0, cfg=cfg) is Branch.SDPA
    assert route(1, 8192, layer_idx=15, cfg=cfg) is Branch.SDPA
    assert route(1, 8192, layer_idx=19, cfg=cfg) is Branch.SDPA


def test_layer_filter_does_not_force_dcr():
    """A whitelisted layer in prefill or short context still goes SDPA."""
    cfg = DCRLlamaConfig(
        T_dispatch=4096,
        layer_filter=frozenset({0}),
    )
    # Whitelisted but prefill
    assert route(100, 100, layer_idx=0, cfg=cfg) is Branch.SDPA
    # Whitelisted but short context
    assert route(1, 1024, layer_idx=0, cfg=cfg) is Branch.SDPA


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_T_dispatch_zero_forces_dcr_for_any_decode():
    """T_dispatch=0: every decode call goes DCR."""
    cfg = DCRLlamaConfig(T_dispatch=0)
    assert route(1, 1, layer_idx=0, cfg=cfg) is Branch.DCR
    assert route(1, 100, layer_idx=0, cfg=cfg) is Branch.DCR


def test_T_dispatch_very_large_disables_dcr():
    """T_dispatch=∞: every call goes SDPA."""
    cfg = DCRLlamaConfig(T_dispatch=10**9)
    assert route(1, 100_000, layer_idx=0, cfg=cfg) is Branch.SDPA


# ---------------------------------------------------------------------------
# explain_route — same decision logic, with prose reason
# ---------------------------------------------------------------------------

def test_explain_route_matches_route_decision():
    """Explanation must agree on which branch was taken."""
    cfg = DCRLlamaConfig(T_dispatch=4096)
    cases = [
        (100, 100, 0),
        (1, 1024, 0),
        (1, 4096, 0),
        (1, 8192, 0),
    ]
    for N_q, N_kv, layer_idx in cases:
        decision = route(N_q, N_kv, layer_idx, cfg)
        explanation = explain_route(N_q, N_kv, layer_idx, cfg)
        if decision is Branch.SDPA:
            assert explanation.startswith("SDPA"), explanation
        else:
            assert explanation.startswith("DCR"), explanation


def test_explain_route_includes_relevant_quantities():
    cfg = DCRLlamaConfig(T_dispatch=4096)
    e = explain_route(1, 1024, 0, cfg)
    assert "N_kv=1024" in e
    assert "T_dispatch=4096" in e


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

def test_config_rejects_odd_k_window():
    with pytest.raises(ValueError, match="k_window"):
        DCRLlamaConfig(k_window=63)


def test_config_rejects_zero_k_window():
    with pytest.raises(ValueError, match="k_window"):
        DCRLlamaConfig(k_window=0)


def test_config_rejects_negative_T_dispatch():
    with pytest.raises(ValueError, match="T_dispatch"):
        DCRLlamaConfig(T_dispatch=-1)


def test_config_rejects_unknown_axis_source():
    with pytest.raises(ValueError, match="axis_source"):
        DCRLlamaConfig(axis_source="random")  # type: ignore
