"""
HF 5.x API contract tests — derived directly from v12 diagnostic dump.

These tests assert the exact attribute / kwarg / signature contract observed
on transformers 5.6.2 with Llama-3.2 1B.  If the upstream API drifts again,
**these tests fail at CPU regression gate** before any GPU round — that's the
INS-27 discipline: detect drift early, not in production integration.

Each test references the diagnostic dump section it's based on.
"""

from __future__ import annotations
import inspect
from typing import Optional, Tuple
from unittest.mock import MagicMock

import math
import pytest
import torch
import torch.nn as nn

import transformers as _transformers

from dcr_attention.models.llama.attention import DCRLlamaAttention, _HF_RETURNS_PRESENT_KV
from dcr_attention.models.llama.compat import cache_update
from dcr_attention.models.llama.config import DCRLlamaConfig

# These contracts are specific to HF 5.x (2-tuple return).
# On 4.x the decoder expects a 3-tuple; skip the 5.x-only assertions.
_HF_5X = not _HF_RETURNS_PRESENT_KV
_skip_4x = pytest.mark.skipif(
    _HF_RETURNS_PRESENT_KV,
    reason=f"HF {_transformers.__version__} uses 4.x 3-tuple contract, not 5.x 2-tuple",
)


# ---------------------------------------------------------------------------
# Mock matching v12 diagnostic Section A-E (transformers 5.x layout)
# ---------------------------------------------------------------------------

class _LlamaConfig5x:
    """Mimics LlamaConfig with 5.x-relevant attributes only."""
    num_attention_heads = 32
    num_key_value_heads = 8
    head_dim = 64
    hidden_size = 2048


class MockLlamaAttention5x(nn.Module):
    """
    Mimics LlamaAttention as it appears in transformers 5.6.2 (per v12 dump).

    Section B of the diagnostic shows:
      * NO `num_heads` instance attr
      * NO `num_key_value_heads` instance attr
      * head_dim, num_key_value_groups, scaling on instance
      * config attribute holds the rest
    """
    def __init__(self):
        super().__init__()
        self.head_dim = 64
        self.num_key_value_groups = 4
        self.scaling = 1.0 / (64 ** 0.5)
        self.attention_dropout = 0.0
        self.is_causal = True
        self.layer_idx = 0
        self.config = _LlamaConfig5x()
        self.q_proj = nn.Linear(2048, 32 * 64, bias=False)
        self.k_proj = nn.Linear(2048, 8 * 64, bias=False)
        self.v_proj = nn.Linear(2048, 8 * 64, bias=False)
        self.o_proj = nn.Linear(32 * 64, 2048, bias=False)


# ---------------------------------------------------------------------------
# CONTRACT 1 — kwarg name `past_key_values` (with 's')
# Section J of v12 dump: HF 5.x calls attn.forward(...,
#                          past_key_values=cache, ...) as a kwarg.
# Our forward must accept this exact spelling.
# ---------------------------------------------------------------------------

def test_contract_forward_accepts_past_key_values_with_s():
    """
    HF 5.x passes the cache as kwarg `past_key_values` (plural form, per
    Section J of v12 diagnostic dump).  Our forward signature must bind it.

    Note: this test does NOT exercise the cache.  It only asserts that the
    parameter name is in the signature.  If our forward had `past_key_value`
    (no `s`), HF's kwarg would silently fall into **kwargs and our local
    `past_key_value` would stay None — the bug.
    """
    sig = inspect.signature(DCRLlamaAttention.forward)
    assert "past_key_values" in sig.parameters, (
        "forward() must accept `past_key_values` (HF 5.x kwarg name, with 's').  "
        f"Found parameters: {list(sig.parameters.keys())}"
    )


def test_contract_forward_does_not_silently_swallow_past_key_values():
    """
    End-to-end behavioural test: pass past_key_values as kwarg, verify the
    forward correctly extracts it (e.g. by calling its .update method).
    """
    base = MockLlamaAttention5x()
    wrap = DCRLlamaAttention(base)

    # Mock cache that records whether .update was called
    mock_cache = MagicMock()
    # Realistic 5.x cache.update returns (K, V) — full cached states.
    def fake_update(K, V, layer_idx, *args, **kwargs):
        return K, V
    mock_cache.update.side_effect = fake_update

    # Minimal forward inputs
    B, N_q, hidden = 1, 4, 2048
    x = torch.randn(B, N_q, hidden)
    cos = torch.ones(B, N_q, base.head_dim)
    sin = torch.zeros(B, N_q, base.head_dim)

    # HF 5.x calls forward with `past_key_values` kwarg (plural form)
    wrap(
        hidden_states=x,
        position_embeddings=(cos, sin),
        past_key_values=mock_cache,        # <-- the spelling that matters
        use_cache=True,
    )

    # Cache.update must have been called.  If our forward had `past_key_value`
    # (no `s`), the HF kwarg would land in **kwargs and never reach the cache
    # update code path.
    assert mock_cache.update.called, (
        "past_key_values=cache was passed to forward but cache.update was "
        "never invoked.  Likely cause: forward signature uses singular "
        "`past_key_value` and HF's plural kwarg falls into **kwargs."
    )


# ---------------------------------------------------------------------------
# CONTRACT 2 — DynamicCache.update signature is (K, V, layer_idx, *args, **kwargs)
# Section I of v12 dump.  cache_update must spread cache_kwargs as **kwargs,
# not pass them as a positional dict.
# ---------------------------------------------------------------------------

def test_contract_cache_update_spreads_kwargs_not_passes_dict_positionally():
    """
    Section I of v12 dump:
      DynamicCache.update(self, key, value, layer_idx, *args, **kwargs)

    Our `compat.cache_update(past, K, V, layer_idx, cache_kwargs)` must
    spread `cache_kwargs` as **kwargs — not pass it as a positional dict
    (which would land in *args[0] as a single dict, not get unpacked).
    """
    captured = {}

    class FakeCache:
        def update(self, K, V, layer_idx, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return K, V

    K = torch.randn(1, 4, 8, 16)
    V = torch.randn(1, 4, 8, 16)
    cache_kwargs = {"sin": torch.zeros(2), "cos": torch.ones(2)}

    cache_update(FakeCache(), K, V, layer_idx=0, cache_kwargs=cache_kwargs)

    # cache_kwargs must arrive as **kwargs, not as args[0]
    assert captured["args"] == (), (
        f"cache_kwargs leaked into positional args: {captured['args']!r}.  "
        f"Likely cause: cache_update passes cache_kwargs as a positional "
        f"dict instead of spreading it as **cache_kwargs."
    )
    assert "sin" in captured["kwargs"] and "cos" in captured["kwargs"], (
        f"cache_kwargs not spread into kwargs: {captured['kwargs']!r}"
    )


def test_contract_cache_update_handles_none_cache_kwargs():
    """compat.cache_update(..., cache_kwargs=None) must not crash."""

    class FakeCache:
        def update(self, K, V, layer_idx, *args, **kwargs):
            return K, V

    K = torch.randn(1, 2, 4, 8)
    V = torch.randn(1, 2, 4, 8)
    # No cache_kwargs supplied — common path in prefill
    out_K, out_V = cache_update(FakeCache(), K, V, layer_idx=0, cache_kwargs=None)
    assert out_K.shape == K.shape and out_V.shape == V.shape


# ---------------------------------------------------------------------------
# CONTRACT 3 — Section B: attn.head_dim is on instance, attn.num_heads is NOT
# Already covered by Phase 2a tests test_construction_works_with_transformers_5x_layout
# but assert it once more here as part of the contract surface.
# ---------------------------------------------------------------------------

def test_contract_5x_attention_layout_resolves_correctly():
    """v12 Section B: 5.x layout has head_dim on instance, num_heads on config."""
    base = MockLlamaAttention5x()
    wrap = DCRLlamaAttention(base)
    assert wrap.num_heads == 32, "should resolve from config.num_attention_heads"
    assert wrap.num_key_value_heads == 8, "should resolve from config.num_key_value_heads"
    assert wrap.head_dim == 64, "should read instance attr directly"
    assert wrap.n_rep == 4, "n_rep = num_heads // num_key_value_heads"


# ---------------------------------------------------------------------------
# CONTRACT 4 — Section J: HF kwargs that must not crash even if we don't use them
# ---------------------------------------------------------------------------

@_skip_4x
def test_contract_forward_accepts_full_5x_kwarg_set_without_crash():
    """
    Per Section J, HF 5.x calls our forward with these kwargs:
      hidden_states, position_embeddings, attention_mask, past_key_values,
      position_ids, use_cache.

    Forward must accept all of them (either explicit or via **kwargs) without
    raising TypeError.
    """
    base = MockLlamaAttention5x()
    cfg = DCRLlamaConfig(enable_dcr=False)   # force SDPA path — no kernel involvement
    wrap = DCRLlamaAttention(base, cfg=cfg)

    B, N_q, hidden = 1, 2, 2048
    x = torch.randn(B, N_q, hidden)
    cos = torch.ones(B, N_q, 64)
    sin = torch.zeros(B, N_q, 64)
    attn_mask = torch.zeros(B, 1, N_q, N_q)
    pos_ids = torch.arange(N_q).unsqueeze(0)

    # Call with the full 5.x kwarg set
    out = wrap(
        hidden_states=x,
        position_embeddings=(cos, sin),
        attention_mask=attn_mask,
        past_key_values=None,
        position_ids=pos_ids,
        use_cache=False,
    )[0]
    assert out.shape == (B, N_q, hidden)


# ---------------------------------------------------------------------------
# CONTRACT 5 — forward must return a 2-tuple (HF 5.x decoder layer contract)
# Per v13 GPU report (Section 5): LlamaDecoderLayer.forward:316 unpacks
#     hidden_states, _ = self.self_attn(...)
# A 3-tuple return raises ValueError: too many values to unpack (expected 2).
# This was the actual blocker of v13 (4 of 6 integration tests failed).
# This contract test prevents regression.
# ---------------------------------------------------------------------------

@_skip_4x
def test_contract_forward_returns_2tuple_runtime():
    """Runtime check: actual forward return value has exactly 2 elements (5.x only)."""
    base = MockLlamaAttention5x()
    cfg = DCRLlamaConfig(enable_dcr=False)
    wrap = DCRLlamaAttention(base, cfg=cfg)

    B, N_q, hidden = 1, 2, 2048
    x = torch.randn(B, N_q, hidden)
    cos = torch.ones(B, N_q, 64)
    sin = torch.zeros(B, N_q, 64)

    result = wrap(hidden_states=x, position_embeddings=(cos, sin))
    assert isinstance(result, tuple), (
        f"forward must return a tuple, got {type(result).__name__}"
    )
    assert len(result) == 2, (
        f"forward must return a 2-tuple per HF 5.x decoder contract; "
        f"got {len(result)}-tuple. This breaks "
        f"`hidden_states, _ = self.self_attn(...)` in LlamaDecoderLayer."
    )


@_skip_4x
def test_contract_forward_unpacks_with_two_targets():
    """Direct simulation of HF 5.x LlamaDecoderLayer's 2-value unpacking pattern."""
    base = MockLlamaAttention5x()
    cfg = DCRLlamaConfig(enable_dcr=False)
    wrap = DCRLlamaAttention(base, cfg=cfg)

    B, N_q, hidden = 1, 2, 2048
    x = torch.randn(B, N_q, hidden)
    cos = torch.ones(B, N_q, 64)
    sin = torch.zeros(B, N_q, 64)

    # This is exactly what LlamaDecoderLayer does — must not raise.
    try:
        hidden_states, _ = wrap(hidden_states=x, position_embeddings=(cos, sin))
    except ValueError as e:
        pytest.fail(
            f"LlamaDecoderLayer-style 2-tuple unpack failed: {e}. "
            f"This is the v13 blocker; forward must return exactly 2 values."
        )
    assert hidden_states.shape == (B, N_q, hidden)


# ---------------------------------------------------------------------------
# CONTRACT 6 — _sdpa_forward must use fp32 softmax (v14b fix)
# Per v14 GPU integration report (Outcome B):
#     test_enable_dcr_false_matches_unpatched   max|diff| = 0.302734
#     test_T_dispatch_infinity_matches_unpatched max|diff| = 0.140625
# Cause: F.scaled_dot_product_attention with bf16 inputs uses bf16 softmax
# (FLASH backend on Ampere/Ada). HF eager LlamaAttention forces fp32 softmax
# via dtype=torch.float32 then casts back. Difference accumulates over layers.
# Fix: replaced F.sdpa with manual attention matching HF eager exactly.
# This contract test prevents anyone "optimizing" back to F.sdpa.
# ---------------------------------------------------------------------------

def test_contract_sdpa_softmax_uses_fp32():
    """_sdpa_forward must compute softmax in fp32 to match HF eager precision."""
    import torch
    import torch.nn.functional as F
    from dcr_attention.models.llama.attention import DCRLlamaAttention

    base = MockLlamaAttention5x()
    cfg = DCRLlamaConfig(enable_dcr=False)
    wrap = DCRLlamaAttention(base, cfg=cfg)

    # Inputs designed to expose bf16 softmax precision loss.
    # Large dynamic range in pre-softmax scores → bf16 softmax loses precision.
    torch.manual_seed(0)
    B, H, N_q, N_kv, D = 1, 4, 8, 8, 16
    Q = torch.randn(B, H, N_q, D, dtype=torch.bfloat16) * 5.0  # large scale
    K = torch.randn(B, H, N_kv, D, dtype=torch.bfloat16) * 5.0
    V = torch.randn(B, H, N_kv, D, dtype=torch.bfloat16)

    # Reference: explicit fp32 softmax (HF eager pattern)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(D)
    attn_fp32 = F.softmax(scores, dim=-1, dtype=torch.float32).to(Q.dtype)
    out_ref = torch.matmul(attn_fp32, V)

    # Our SDPA path
    out_actual = wrap._sdpa_forward(Q, K, V, attention_mask=None, N_q=N_q, N_kv=N_kv)

    # Note: prefill path with attention_mask=None triggers manual causal mask
    # — apply same in reference for fair comparison.
    causal = torch.triu(
        torch.full((N_q, N_kv), torch.finfo(scores.dtype).min,
                   dtype=scores.dtype, device=scores.device),
        diagonal=N_kv - N_q + 1,
    )
    scores_causal = scores + causal
    attn_fp32_causal = F.softmax(scores_causal, dim=-1, dtype=torch.float32).to(Q.dtype)
    out_ref_causal = torch.matmul(attn_fp32_causal, V)

    # Match within bf16 round-off (matmul accumulator differences can give a few ULPs).
    max_diff = (out_actual - out_ref_causal).abs().max().item()
    assert max_diff < 1e-2, (
        f"_sdpa_forward output diverged from manual fp32-softmax reference: "
        f"max|diff| = {max_diff}. This indicates softmax is being done in bf16 "
        f"instead of fp32, which will cause integration tests to fail with HF eager."
    )


def test_contract_sdpa_softmax_dtype_inspection():
    """Static check: source code must contain dtype=torch.float32 in _sdpa_forward."""
    import inspect
    from dcr_attention.models.llama.attention import DCRLlamaAttention

    src = inspect.getsource(DCRLlamaAttention._sdpa_forward)
    # Strip docstring (which legitimately mentions F.sdpa for context)
    import ast
    import textwrap
    tree = ast.parse(textwrap.dedent(src))
    func = tree.body[0]
    if (isinstance(func.body[0], ast.Expr)
            and isinstance(func.body[0].value, ast.Constant)
            and isinstance(func.body[0].value.value, str)):
        # Remove docstring node, then unparse
        func.body = func.body[1:]
    code_only = ast.unparse(func)

    assert "dtype=torch.float32" in code_only, (
        "_sdpa_forward must explicitly use dtype=torch.float32 in softmax. "
        "Code (docstring stripped):\n" + code_only
    )
    assert "scaled_dot_product_attention" not in code_only, (
        "_sdpa_forward must not call F.scaled_dot_product_attention "
        "(uses bf16 softmax on FLASH backend, breaks HF eager equivalence). "
        "Use manual matmul + softmax(fp32) + matmul instead."
    )
