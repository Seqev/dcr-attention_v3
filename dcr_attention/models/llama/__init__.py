"""
DCR Llama integration — drop-in replacement of HF ``LlamaAttention``.

Public surface:
    DCRLlamaAttention   — the wrapper class.
    DCRLlamaConfig      — user knobs.
    Branch              — routing decision enum.
    route               — pure-logic routing function.
"""

from dcr_attention.models.llama.attention import DCRLlamaAttention
from dcr_attention.models.llama.config import (
    DCRLlamaConfig,
    DEFAULT_DCR_LLAMA_CONFIG,
    AxisSource,
)
from dcr_attention.models.llama.monkey_patch import (
    is_patched,
    patch_llama_with_dcr,
    patched_layer_indices,
    unpatch_llama,
)
from dcr_attention.models.llama.routing import Branch, route, explain_route

__all__ = [
    "DCRLlamaAttention",
    "DCRLlamaConfig",
    "DEFAULT_DCR_LLAMA_CONFIG",
    "AxisSource",
    "Branch",
    "route",
    "explain_route",
    "patch_llama_with_dcr",
    "unpatch_llama",
    "is_patched",
    "patched_layer_indices",
]
