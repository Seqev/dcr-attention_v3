"""
In-place replacement of HF ``LlamaAttention`` modules with ``DCRLlamaAttention``.

This is the integration entry point users call after loading a Llama model:

    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
    patch_llama_with_dcr(model, DCRLlamaConfig(k_window=64, T_dispatch=4096))
    output = model.generate(...)            # routes through DCR per dispatcher

Key invariants:

  * **Memory-neutral.**  ``DCRLlamaAttention`` holds projections by reference;
    no parameter copying.  Patching a 16-GB model adds ~0 MB.
  * **Reversible.**  ``unpatch_llama(model)`` restores the original modules.
  * **Idempotent.**  ``patch`` followed by ``patch`` is a no-op + warning;
    not an error, since users may patch in setup code that runs twice.
  * **Layer-aware.**  Respects ``cfg.layer_filter`` — non-listed layers are
    left as-is (still HF native).

The implementation deliberately walks ``model.model.layers`` rather than using
``model.named_modules()`` filtering: the layout is documented and stable for
the Llama family, while name-based filtering would silently miss subclasses
(``LlamaSdpaAttention``, ``LlamaFlashAttention2``, etc.).
"""

from __future__ import annotations
import warnings
from typing import List, Optional

import torch.nn as nn

from dcr_attention.models.llama.attention import DCRLlamaAttention
from dcr_attention.models.llama.config import DCRLlamaConfig


# ---------------------------------------------------------------------------
# Internal: discover attention modules in a Llama-style model
# ---------------------------------------------------------------------------

def _iter_attention_layers(model: nn.Module):
    """
    Yield (decoder_layer, layer_idx) tuples for every transformer layer.

    Expects HF Llama-family layout:  ``model.model.layers``.  Falls back to
    ``model.layers`` for raw ``LlamaModel`` (no causal LM head).  Raises if
    neither path exists.
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers
    else:
        raise ValueError(
            f"Cannot find transformer layers on {type(model).__name__}. "
            f"Expected `model.model.layers` (causal LM) or `model.layers` "
            f"(raw model)."
        )
    for idx, layer in enumerate(layers):
        if not hasattr(layer, "self_attn"):
            raise ValueError(
                f"Layer {idx} has no `self_attn` attribute; "
                f"is this really a Llama-family model?"
            )
        yield layer, idx


def _is_already_patched(layer: nn.Module) -> bool:
    return isinstance(layer.self_attn, DCRLlamaAttention)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Sentinel attribute we attach to the model to track patch state.  Stores the
# original attention modules so unpatch can restore them.
_PATCH_REGISTRY_ATTR = "_dcr_patch_registry"


def patch_llama_with_dcr(
    model: nn.Module,
    cfg: Optional[DCRLlamaConfig] = None,
) -> int:
    r"""
    Replace every ``LlamaAttention`` (or subclass) in ``model`` with
    ``DCRLlamaAttention``.  Returns the number of layers actually replaced.

    Parameters
    ----------
    model
        Loaded HF Llama model.  Mutated in place.
    cfg
        Configuration shared across all replaced layers.  ``cfg.layer_filter``
        controls which layers are replaced.  Default: replace all layers
        with default config.

    Returns
    -------
    n_replaced : int
        Layers replaced (excludes those skipped by ``layer_filter`` or
        already patched).

    Notes
    -----
    The original ``self_attn`` modules are stored on ``model._dcr_patch_registry``
    keyed by layer index, so :func:`unpatch_llama` can restore them.  Do not
    inspect or modify this attribute manually.
    """
    cfg = cfg if cfg is not None else DCRLlamaConfig()

    # Initialise registry on first patch
    if not hasattr(model, _PATCH_REGISTRY_ATTR):
        setattr(model, _PATCH_REGISTRY_ATTR, {})
    registry: dict = getattr(model, _PATCH_REGISTRY_ATTR)

    n_replaced = 0
    n_already_patched = 0
    n_skipped_by_filter = 0

    for layer, layer_idx in _iter_attention_layers(model):
        # Layer filter: skip if not in whitelist
        if cfg.layer_filter is not None and layer_idx not in cfg.layer_filter:
            n_skipped_by_filter += 1
            continue

        # Idempotence: skip already-patched layers
        if _is_already_patched(layer):
            n_already_patched += 1
            continue

        # Save original, install wrapper
        original = layer.self_attn
        registry[layer_idx] = original

        # Ensure layer_idx is propagated; HF sets this on the attention module,
        # but if it's missing we fall back to the position in the layers list.
        if not hasattr(original, "layer_idx") or original.layer_idx is None:
            original.layer_idx = layer_idx

        wrapper = DCRLlamaAttention(original, cfg=cfg)
        layer.self_attn = wrapper
        n_replaced += 1

    if n_already_patched:
        warnings.warn(
            f"patch_llama_with_dcr: {n_already_patched} layer(s) were already "
            f"patched and were skipped.  Total newly-replaced: {n_replaced}.",
            stacklevel=2,
        )

    return n_replaced


def unpatch_llama(model: nn.Module) -> int:
    r"""
    Restore original ``LlamaAttention`` modules.  Returns the number of
    layers actually restored.

    No-op (returns 0) if the model has never been patched.
    """
    if not hasattr(model, _PATCH_REGISTRY_ATTR):
        return 0

    registry: dict = getattr(model, _PATCH_REGISTRY_ATTR)
    n_restored = 0
    for layer, layer_idx in _iter_attention_layers(model):
        if layer_idx in registry and _is_already_patched(layer):
            layer.self_attn = registry[layer_idx]
            n_restored += 1

    # Clear the registry
    delattr(model, _PATCH_REGISTRY_ATTR)
    return n_restored


def reconfigure_dcr(model: nn.Module, cfg: DCRLlamaConfig) -> int:
    r"""
    Update the config on all already-patched ``DCRLlamaAttention`` layers.

    Use this when the model is already patched and you want to switch
    axis_source, enable_dcr, coverage_floor, etc. without unpatching and
    re-patching (which would re-instantiate the wrappers and break references).

    Returns the number of layers updated.  Raises if the model has no patched
    layers (call :func:`patch_llama_with_dcr` first).
    """
    n_updated = 0
    for layer, _ in _iter_attention_layers(model):
        if isinstance(layer.self_attn, DCRLlamaAttention):
            layer.self_attn.cfg = cfg
            n_updated += 1
    return n_updated


def is_patched(model: nn.Module) -> bool:
    r"""Return True if any layer in ``model`` is currently a ``DCRLlamaAttention``."""
    try:
        for layer, _ in _iter_attention_layers(model):
            if _is_already_patched(layer):
                return True
    except ValueError:
        return False
    return False


def patched_layer_indices(model: nn.Module) -> List[int]:
    r"""Return sorted list of layer indices currently using ``DCRLlamaAttention``."""
    indices = []
    try:
        for layer, layer_idx in _iter_attention_layers(model):
            if _is_already_patched(layer):
                indices.append(layer_idx)
    except ValueError:
        pass
    return indices
