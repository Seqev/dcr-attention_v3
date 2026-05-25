"""
Unit tests for ``DCRLlamaAttention`` — CPU only, no transformers.

We mock ``LlamaAttention`` with a minimal ``nn.Module`` that exposes:
    q_proj, k_proj, v_proj, o_proj, num_heads, num_key_value_heads,
    head_dim, layer_idx, rotary_emb (optional)

This validates:
  * Composition / attribute proxying.
  * Forward signature compatibility.
  * Routing decisions surface correctly through the wrapper.
  * Both branches (SDPA, DCR) produce shape-correct, finite output.
  * SDPA branch is bit-identical to direct ``F.scaled_dot_product_attention``.

Real HF integration is validated GPU-side in ``test_dcr_llama_correctness.py``.
"""

from __future__ import annotations
from typing import Optional, Tuple
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from dcr_attention.models.llama.attention import DCRLlamaAttention
from dcr_attention.models.llama.config import DCRLlamaConfig


# ---------------------------------------------------------------------------
# Mock LlamaAttention
# ---------------------------------------------------------------------------

class MockLlamaAttention(nn.Module):
    """Minimal stand-in for ``LlamaAttention`` — just the surface DCRLlamaAttention reads."""

    def __init__(
        self,
        hidden_size: int = 256,
        num_heads: int = 8,
        num_key_value_heads: int = 4,
        head_dim: int = 32,
        layer_idx: int = 0,
        bias: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.layer_idx = layer_idx

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=bias)


def _make_inputs(B: int, N_q: int, hidden_size: int, dtype=torch.float32):
    torch.manual_seed(42)
    return torch.randn(B, N_q, hidden_size, dtype=dtype)


def _zero_cos_sin(B: int, N: int, head_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """RoPE-disabling cos/sin (cos=1, sin=0).  Useful for tests where we
    just want the projection + cache + branching machinery to work without
    RoPE math interfering."""
    cos = torch.ones(B, N, head_dim)
    sin = torch.zeros(B, N, head_dim)
    return cos, sin


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_construction_reads_attributes_by_reference():
    base = MockLlamaAttention(num_heads=8, num_key_value_heads=4, head_dim=32)
    wrap = DCRLlamaAttention(base)

    assert wrap.q_proj is base.q_proj          # reference, not copy
    assert wrap.num_heads == 8
    assert wrap.num_key_value_heads == 4
    assert wrap.head_dim == 32
    assert wrap.n_rep == 2                     # 8 // 4


def test_construction_rejects_invalid_gqa():
    base = MockLlamaAttention(num_heads=8, num_key_value_heads=3, head_dim=32)
    with pytest.raises(ValueError, match="num_heads"):
        DCRLlamaAttention(base)


def test_construction_rejects_missing_attrs():
    """If base layer is too minimal to resolve shape, raise diagnostic error."""

    class NoShape(nn.Module):
        """No num_heads, no head_dim, no config — irrecoverable."""
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(8, 8)
            self.k_proj = nn.Linear(8, 8)
            self.v_proj = nn.Linear(8, 8)
            self.o_proj = nn.Linear(8, 8)
            self.head_dim = None      # explicitly None
            self.num_heads = None

    with pytest.raises(ValueError, match="cannot resolve required shape parameters"):
        DCRLlamaAttention(NoShape())


# ---------------------------------------------------------------------------
# Forward shape correctness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("N_q", [1, 8, 64])
def test_forward_returns_correct_shape(N_q):
    """Wrapper output shape matches input hidden_states shape."""
    B, hidden = 2, 256
    base = MockLlamaAttention(hidden_size=hidden, num_heads=8,
                              num_key_value_heads=4, head_dim=32)
    cfg = DCRLlamaConfig(T_dispatch=4096)  # below threshold so SDPA branch
    wrap = DCRLlamaAttention(base, cfg=cfg)

    x = _make_inputs(B, N_q, hidden)
    cos, sin = _zero_cos_sin(B, N_q, 32)

    result = wrap(
        x,
        position_embeddings=(cos, sin),
        use_cache=False,
    )
    out, attn_weights = result[0], result[1]

    assert out.shape == (B, N_q, hidden)
    assert attn_weights is None


# ---------------------------------------------------------------------------
# Branch selection
# ---------------------------------------------------------------------------

def test_prefill_uses_sdpa_branch():
    """N_q > 1 must take SDPA branch (verified via patching DCR forward)."""
    B, hidden, head_dim, N_q = 1, 128, 16, 16
    base = MockLlamaAttention(hidden_size=hidden, num_heads=8,
                              num_key_value_heads=4, head_dim=head_dim)
    cfg = DCRLlamaConfig(T_dispatch=0)   # would force DCR if N_q==1
    wrap = DCRLlamaAttention(base, cfg=cfg)

    x = _make_inputs(B, N_q, hidden)
    cos, sin = _zero_cos_sin(B, N_q, head_dim)

    with patch.object(wrap, "_dcr_forward", side_effect=AssertionError("DCR called in prefill")):
        out = wrap(x, position_embeddings=(cos, sin))[0]
    assert out.shape == (B, N_q, hidden)


def test_short_decode_uses_sdpa_branch():
    """N_q == 1, N_kv < T_dispatch → SDPA."""
    B, hidden, head_dim = 1, 128, 16
    base = MockLlamaAttention(hidden_size=hidden, num_heads=8,
                              num_key_value_heads=4, head_dim=head_dim)
    cfg = DCRLlamaConfig(T_dispatch=4096)
    wrap = DCRLlamaAttention(base, cfg=cfg)

    x = _make_inputs(B, 1, hidden)
    cos, sin = _zero_cos_sin(B, 1, head_dim)

    with patch.object(wrap, "_dcr_forward", side_effect=AssertionError("DCR called below T")):
        out = wrap(x, position_embeddings=(cos, sin))[0]
    assert out.shape == (B, 1, hidden)


# ---------------------------------------------------------------------------
# SDPA branch correctness — bit-equivalent to direct F.sdpa
# ---------------------------------------------------------------------------

def test_sdpa_branch_matches_direct_sdpa_in_prefill():
    """Wrapper SDPA output should equal direct F.sdpa on the same Q/K/V/mask."""
    B, hidden, head_dim, num_heads = 1, 64, 16, 4
    N_q = 8
    base = MockLlamaAttention(hidden_size=hidden, num_heads=num_heads,
                              num_key_value_heads=num_heads,    # no GQA
                              head_dim=head_dim)
    cfg = DCRLlamaConfig(enable_dcr=False)
    wrap = DCRLlamaAttention(base, cfg=cfg)

    x = _make_inputs(B, N_q, hidden)
    cos, sin = _zero_cos_sin(B, N_q, head_dim)

    out_wrap = wrap(x, position_embeddings=(cos, sin))[0]

    # Recompute by hand using the same projections + RoPE (identity here)
    Q = base.q_proj(x).view(B, N_q, num_heads, head_dim).transpose(1, 2)
    K = base.k_proj(x).view(B, N_q, num_heads, head_dim).transpose(1, 2)
    V = base.v_proj(x).view(B, N_q, num_heads, head_dim).transpose(1, 2)
    # zero-RoPE → Q, K unchanged after cos=1, sin=0 multiplication
    out_direct = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
    out_direct = out_direct.transpose(1, 2).contiguous().view(B, N_q, -1)
    out_direct = base.o_proj(out_direct)

    diff = (out_wrap - out_direct).abs().max().item()
    assert diff < 1e-5, f"wrapper SDPA branch != direct SDPA, diff={diff:g}"


# ---------------------------------------------------------------------------
# output_attentions=True must raise
# ---------------------------------------------------------------------------

def test_output_attentions_raises():
    base = MockLlamaAttention()
    wrap = DCRLlamaAttention(base)
    x = _make_inputs(1, 4, base.hidden_size)
    cos, sin = _zero_cos_sin(1, 4, base.head_dim)
    with pytest.raises(NotImplementedError, match="output_attentions"):
        wrap(x, position_embeddings=(cos, sin), output_attentions=True)


# ---------------------------------------------------------------------------
# extra_repr
# ---------------------------------------------------------------------------

def test_extra_repr_contains_key_info():
    base = MockLlamaAttention(layer_idx=7)
    cfg = DCRLlamaConfig(k_window=128, T_dispatch=2048)
    wrap = DCRLlamaAttention(base, cfg=cfg)
    s = wrap.extra_repr()
    assert "layer_idx=7" in s
    assert "k_window=128" in s
    assert "T_dispatch=2048" in s


# ---------------------------------------------------------------------------
# compat.py dispatch — both HF API versions
# ---------------------------------------------------------------------------

def test_compat_apply_rotary_dispatches_for_legacy_4x_signature():
    """transformers 4.x: apply_rotary_pos_emb has position_ids → keep it."""
    from dcr_attention.models.llama import compat

    captured = {}
    def fake_4x(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        captured["nargs"] = 6
        captured["position_ids"] = position_ids
        captured["unsqueeze_dim"] = unsqueeze_dim
        return q, k

    compat._resolve_hf_apply_rotary.cache_clear()
    try:
        orig = compat._resolve_hf_apply_rotary
        compat._resolve_hf_apply_rotary = lambda: (fake_4x, True)

        Q = torch.randn(1, 2, 4, 8)
        K = torch.randn(1, 2, 4, 8)
        cos = torch.zeros(1, 4, 8)
        sin = torch.zeros(1, 4, 8)
        position_ids = torch.tensor([[1, 2, 3, 4]])

        compat.apply_rotary_pos_emb(
            Q, K, cos, sin, position_ids, unsqueeze_dim=1
        )
        assert captured["nargs"] == 6, "4.x fake should have been called"
        assert torch.equal(captured["position_ids"], position_ids), \
            "position_ids was not forwarded to 4.x-style HF apply"
        assert captured["unsqueeze_dim"] == 1
    finally:
        compat._resolve_hf_apply_rotary = orig
        compat._resolve_hf_apply_rotary.cache_clear()


def test_compat_apply_rotary_dispatches_for_5x_signature():
    """transformers 5.x: apply_rotary_pos_emb dropped position_ids → omit it."""
    from dcr_attention.models.llama import compat

    captured = {}
    def fake_5x(q, k, cos, sin, unsqueeze_dim=1):
        # Record the signature width we got called with
        captured["nargs"] = 5
        captured["unsqueeze_dim"] = unsqueeze_dim
        return q, k

    compat._resolve_hf_apply_rotary.cache_clear()
    try:
        orig = compat._resolve_hf_apply_rotary
        compat._resolve_hf_apply_rotary = lambda: (fake_5x, False)

        Q = torch.randn(1, 2, 4, 8)
        K = torch.randn(1, 2, 4, 8)
        cos = torch.zeros(1, 4, 8)
        sin = torch.zeros(1, 4, 8)

        # Pass position_ids — must be silently dropped under 5.x dispatch
        compat.apply_rotary_pos_emb(
            Q, K, cos, sin,
            position_ids=torch.tensor([[0, 1, 2, 3]]),
            unsqueeze_dim=1,
        )
        assert captured["nargs"] == 5, "5.x fake should have been called"
        assert captured["unsqueeze_dim"] == 1
    finally:
        compat._resolve_hf_apply_rotary = orig
        compat._resolve_hf_apply_rotary.cache_clear()


def test_compat_apply_rotary_falls_back_when_no_transformers():
    """No transformers installed → use inline reference."""
    from dcr_attention.models.llama import compat

    compat._resolve_hf_apply_rotary.cache_clear()
    try:
        orig = compat._resolve_hf_apply_rotary
        compat._resolve_hf_apply_rotary = lambda: (None, None)

        Q = torch.randn(1, 2, 4, 8)
        K = torch.randn(1, 2, 4, 8)
        cos = torch.ones(1, 4, 8)
        sin = torch.zeros(1, 4, 8)

        # Identity (cos=1, sin=0): output = input
        q_out, k_out = compat.apply_rotary_pos_emb(Q, K, cos, sin)
        assert torch.allclose(q_out, Q)
        assert torch.allclose(k_out, K)
    finally:
        compat._resolve_hf_apply_rotary = orig
        compat._resolve_hf_apply_rotary.cache_clear()


# ---------------------------------------------------------------------------
# Resilient shape-attribute resolution (transformers 4.x vs 5.x layout)
# ---------------------------------------------------------------------------

class _FakeLlamaConfig:
    """Stand-in for transformers' ``LlamaConfig`` — only the attrs we read."""
    def __init__(self, num_attention_heads, num_key_value_heads, head_dim):
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim


class MockLlamaAttention_5x(nn.Module):
    r"""
    Mimics transformers ≥ 5.x ``LlamaAttention``: ``head_dim`` is the only
    instance shape attribute; ``num_heads`` and ``num_key_value_heads`` live
    on ``config``.
    """
    def __init__(self, num_heads=8, num_kv_heads=4, head_dim=32,
                 hidden_size=256, layer_idx=0):
        super().__init__()
        self.head_dim = head_dim
        self.layer_idx = layer_idx
        # NOTE: NO num_heads, NO num_key_value_heads on instance
        self.config = _FakeLlamaConfig(
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            head_dim=head_dim,
        )
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)


def test_construction_works_with_transformers_5x_layout():
    """transformers 5.x: shape attrs only on .config, not on instance."""
    base = MockLlamaAttention_5x(num_heads=32, num_kv_heads=8, head_dim=64)
    wrap = DCRLlamaAttention(base)

    # All resolved correctly via config fallback
    assert wrap.num_heads == 32
    assert wrap.num_key_value_heads == 8
    assert wrap.head_dim == 64
    assert wrap.n_rep == 4               # 32 // 8


def test_construction_prefers_instance_attrs_over_config():
    """If both instance attr and config exist, instance attr wins."""
    base = MockLlamaAttention_5x(num_heads=32, num_kv_heads=8, head_dim=64)
    # Inject a conflicting instance attribute
    base.num_heads = 64                  # different from config
    wrap = DCRLlamaAttention(base)
    assert wrap.num_heads == 64          # instance wins


def test_construction_diagnostic_error_lists_what_was_checked():
    """If neither instance nor config has the attr, error must list both paths."""

    class TooMinimal(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(8, 8)
            self.k_proj = nn.Linear(8, 8)
            self.v_proj = nn.Linear(8, 8)
            self.o_proj = nn.Linear(8, 8)
            self.head_dim = None
            # No config, no num_heads, no head_dim resolvable

    with pytest.raises(ValueError, match="cannot resolve required shape parameters"):
        DCRLlamaAttention(TooMinimal())


def test_construction_falls_back_through_alternative_config_keys():
    """If config has 'num_heads' instead of 'num_attention_heads', still works."""

    class AltConfig:
        num_heads = 16          # alternative key
        num_key_value_heads = 4
        head_dim = 32

    class BaseWithAltConfig(nn.Module):
        def __init__(self):
            super().__init__()
            self.head_dim = 32
            self.layer_idx = 0
            self.config = AltConfig()
            self.q_proj = nn.Linear(64, 16 * 32, bias=False)
            self.k_proj = nn.Linear(64, 4 * 32, bias=False)
            self.v_proj = nn.Linear(64, 4 * 32, bias=False)
            self.o_proj = nn.Linear(16 * 32, 64, bias=False)

    wrap = DCRLlamaAttention(BaseWithAltConfig())
    assert wrap.num_heads == 16


# ---------------------------------------------------------------------------
# Phase 2b instrumentation — class-level routing counters
# ---------------------------------------------------------------------------

def test_phase2b_counters_initial_state():
    """Fresh counters start at zero."""
    DCRLlamaAttention.reset_counters()
    counters = DCRLlamaAttention.get_counters()
    assert counters == {"dcr": 0, "sdpa": 0}


def test_phase2b_sdpa_counter_increments():
    """SDPA branch invocation increments _sdpa_invocations."""
    DCRLlamaAttention.reset_counters()

    base = MockLlamaAttention_5x()
    cfg = DCRLlamaConfig(enable_dcr=False)  # forces SDPA branch
    wrap = DCRLlamaAttention(base, cfg=cfg)

    B, N_q, hidden = 1, 2, 256
    x = torch.randn(B, N_q, hidden)
    cos = torch.ones(B, N_q, 32)  # head_dim / 2 = 16; rope dim doubled
    sin = torch.zeros(B, N_q, 32)
    wrap(hidden_states=x, position_embeddings=(cos, sin))

    counters = DCRLlamaAttention.get_counters()
    assert counters["sdpa"] == 1, f"sdpa counter should be 1, got {counters}"
    assert counters["dcr"] == 0, f"dcr counter should be 0, got {counters}"


def test_phase2b_counters_reset_works():
    """reset_counters zeros both counters."""
    DCRLlamaAttention._sdpa_invocations = 42
    DCRLlamaAttention._dcr_invocations = 17
    DCRLlamaAttention.reset_counters()
    assert DCRLlamaAttention.get_counters() == {"dcr": 0, "sdpa": 0}


def test_phase2b_counters_are_class_level():
    """Counters are shared across all instances (process-wide tracking)."""
    DCRLlamaAttention.reset_counters()

    cfg = DCRLlamaConfig(enable_dcr=False)
    wrap1 = DCRLlamaAttention(MockLlamaAttention_5x(), cfg=cfg)
    wrap2 = DCRLlamaAttention(MockLlamaAttention_5x(), cfg=cfg)

    B, N_q, hidden = 1, 2, 256
    x = torch.randn(B, N_q, hidden)
    cos = torch.ones(B, N_q, 32)
    sin = torch.zeros(B, N_q, 32)

    wrap1(hidden_states=x, position_embeddings=(cos, sin))
    wrap2(hidden_states=x, position_embeddings=(cos, sin))

    # Both invocations contribute to the SAME class-level counter.
    counters = DCRLlamaAttention.get_counters()
    assert counters["sdpa"] == 2, (
        f"Class-level counter must aggregate across instances; got {counters}. "
        f"This matters for multi-layer models where we count total branch usage."
    )
