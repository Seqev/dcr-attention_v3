"""
Lifecycle tests for ``patch_llama_with_dcr`` / ``unpatch_llama``.

CPU-only.  Uses a mock model that mimics HF Llama's layout:

    MockLlamaModel
      └── model         (nn.Module)
           └── layers   (nn.ModuleList)
                └── [LlamaDecoderLayer × N]
                      └── self_attn   (MockLlamaAttention)

Real HF integration is validated GPU-side in v10.
"""

from __future__ import annotations
from typing import Optional

import pytest
import torch
import torch.nn as nn

from dcr_attention.models.llama.attention import DCRLlamaAttention
from dcr_attention.models.llama.config import DCRLlamaConfig
from dcr_attention.models.llama.monkey_patch import (
    is_patched,
    patch_llama_with_dcr,
    patched_layer_indices,
    unpatch_llama,
)


# ---------------------------------------------------------------------------
# Mock model layout
# ---------------------------------------------------------------------------

class MockAttention(nn.Module):
    """Minimal stand-in for ``LlamaAttention``."""

    def __init__(self, hidden=128, num_heads=4, num_kv_heads=2,
                 head_dim=32, layer_idx=0):
        super().__init__()
        self.hidden_size = hidden
        self.num_heads = num_heads
        self.num_key_value_heads = num_kv_heads
        self.head_dim = head_dim
        self.layer_idx = layer_idx
        self.q_proj = nn.Linear(hidden, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden, bias=False)


class MockDecoderLayer(nn.Module):
    """Stand-in for ``LlamaDecoderLayer``."""

    def __init__(self, layer_idx: int, hidden: int = 128):
        super().__init__()
        self.self_attn = MockAttention(hidden=hidden, layer_idx=layer_idx)
        # In real Llama there'd also be MLP, layernorms etc.; we don't need them.


class MockInner(nn.Module):
    """Stand-in for ``LlamaModel`` (the part inside ``LlamaForCausalLM``)."""

    def __init__(self, num_layers: int = 4, hidden: int = 128):
        super().__init__()
        self.layers = nn.ModuleList(
            MockDecoderLayer(i, hidden) for i in range(num_layers)
        )


class MockLlamaForCausalLM(nn.Module):
    """Stand-in for ``LlamaForCausalLM`` — ``model.model.layers`` layout."""

    def __init__(self, num_layers: int = 4, hidden: int = 128):
        super().__init__()
        self.model = MockInner(num_layers=num_layers, hidden=hidden)


class MockLlamaModel(nn.Module):
    """Stand-in for raw ``LlamaModel`` — ``model.layers`` layout (no causal LM)."""

    def __init__(self, num_layers: int = 4, hidden: int = 128):
        super().__init__()
        self.layers = nn.ModuleList(
            MockDecoderLayer(i, hidden) for i in range(num_layers)
        )


# ---------------------------------------------------------------------------
# Patch / unpatch basic
# ---------------------------------------------------------------------------

def test_patch_replaces_all_attention_modules():
    model = MockLlamaForCausalLM(num_layers=4)
    n = patch_llama_with_dcr(model, DCRLlamaConfig())
    assert n == 4
    assert is_patched(model)
    assert patched_layer_indices(model) == [0, 1, 2, 3]
    for layer in model.model.layers:
        assert isinstance(layer.self_attn, DCRLlamaAttention)


def test_patch_works_on_raw_model_layout():
    """``LlamaModel`` (no causal LM head) has ``model.layers`` directly."""
    model = MockLlamaModel(num_layers=3)
    n = patch_llama_with_dcr(model, DCRLlamaConfig())
    assert n == 3
    assert is_patched(model)


def test_patch_then_unpatch_restores_originals():
    model = MockLlamaForCausalLM(num_layers=4)
    originals = [layer.self_attn for layer in model.model.layers]

    patch_llama_with_dcr(model)
    n = unpatch_llama(model)
    assert n == 4
    assert not is_patched(model)
    for layer, orig in zip(model.model.layers, originals):
        assert layer.self_attn is orig


def test_unpatch_on_unpatched_model_is_noop():
    model = MockLlamaForCausalLM(num_layers=2)
    assert unpatch_llama(model) == 0


# ---------------------------------------------------------------------------
# Memory-neutrality: projections held by reference, not copied
# ---------------------------------------------------------------------------

def test_patch_reuses_projection_modules_by_reference():
    """The wrapper must hold q_proj, etc. by reference — no parameter copy."""
    model = MockLlamaForCausalLM(num_layers=2)
    original_q_projs = [layer.self_attn.q_proj for layer in model.model.layers]

    patch_llama_with_dcr(model)

    for layer, orig_q in zip(model.model.layers, original_q_projs):
        wrapper: DCRLlamaAttention = layer.self_attn  # type: ignore
        assert wrapper.q_proj is orig_q, (
            "wrapper.q_proj must be the same Linear instance as the original"
        )


def test_patch_does_not_clone_parameters():
    """Parameter count is preserved (no fresh nn.Parameters created)."""
    model = MockLlamaForCausalLM(num_layers=2)
    n_params_before = sum(p.numel() for p in model.parameters())
    patch_llama_with_dcr(model)
    n_params_after = sum(p.numel() for p in model.parameters())
    assert n_params_before == n_params_after, (
        f"parameter count changed: {n_params_before} → {n_params_after}; "
        f"wrapper must not introduce fresh parameters"
    )


# ---------------------------------------------------------------------------
# Idempotence
# ---------------------------------------------------------------------------

def test_double_patch_is_idempotent_with_warning():
    model = MockLlamaForCausalLM(num_layers=3)
    patch_llama_with_dcr(model)
    with pytest.warns(UserWarning, match="already patched"):
        n = patch_llama_with_dcr(model)
    assert n == 0       # nothing newly replaced
    assert is_patched(model)
    # Check no DCRLlamaAttention got nested
    for layer in model.model.layers:
        assert isinstance(layer.self_attn, DCRLlamaAttention)
        assert not isinstance(layer.self_attn._base_layer, DCRLlamaAttention)


def test_patch_unpatch_patch_roundtrip():
    """Patch → unpatch → patch must work cleanly."""
    model = MockLlamaForCausalLM(num_layers=2)
    patch_llama_with_dcr(model)
    unpatch_llama(model)
    n = patch_llama_with_dcr(model)
    assert n == 2
    assert is_patched(model)


# ---------------------------------------------------------------------------
# Layer filter
# ---------------------------------------------------------------------------

def test_layer_filter_only_patches_listed_layers():
    model = MockLlamaForCausalLM(num_layers=8)
    cfg = DCRLlamaConfig(layer_filter=frozenset({2, 5, 7}))
    n = patch_llama_with_dcr(model, cfg)
    assert n == 3
    assert patched_layer_indices(model) == [2, 5, 7]
    for i, layer in enumerate(model.model.layers):
        if i in (2, 5, 7):
            assert isinstance(layer.self_attn, DCRLlamaAttention)
        else:
            assert isinstance(layer.self_attn, MockAttention)


def test_unpatch_restores_only_replaced_layers():
    model = MockLlamaForCausalLM(num_layers=4)
    originals = [layer.self_attn for layer in model.model.layers]

    cfg = DCRLlamaConfig(layer_filter=frozenset({1, 3}))
    patch_llama_with_dcr(model, cfg)
    n = unpatch_llama(model)
    assert n == 2
    for i, layer in enumerate(model.model.layers):
        assert layer.self_attn is originals[i]


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

def test_patch_rejects_non_llama_model():
    """A model without `model.layers` or `layers` must raise."""

    class NotALlamaModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.something = nn.Linear(10, 10)

    with pytest.raises(ValueError, match="Cannot find transformer layers"):
        patch_llama_with_dcr(NotALlamaModel())


def test_patch_rejects_layers_without_self_attn():
    """If a decoder layer has no `self_attn`, patch must raise clearly."""

    class BrokenDecoderLayer(nn.Module):
        pass

    class BrokenInner(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([BrokenDecoderLayer()])

    class BrokenLlama(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = BrokenInner()

    with pytest.raises(ValueError, match="self_attn"):
        patch_llama_with_dcr(BrokenLlama())


# ---------------------------------------------------------------------------
# is_patched / patched_layer_indices on edge cases
# ---------------------------------------------------------------------------

def test_is_patched_false_on_fresh_model():
    model = MockLlamaForCausalLM(num_layers=2)
    assert not is_patched(model)
    assert patched_layer_indices(model) == []


def test_is_patched_handles_non_llama_model_gracefully():
    """is_patched should return False (not raise) for non-Llama models."""

    class Foo(nn.Module):
        pass

    assert is_patched(Foo()) is False
    assert patched_layer_indices(Foo()) == []
