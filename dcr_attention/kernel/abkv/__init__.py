"""
ABKV (Axis-Block Key-Value) — pre-sorted KV cache for fast top-k decode.

Public API:
    ABKVCache               — pre-sorted cache dataclass
    build_abkv_cache        — prefill: sort KV by axis projection score
    abkv_attention_reference— decode: attention over prefix of sorted cache
    abkv_attention_with_signature — build + attend (one-shot for tests)
    sdpa_reference          — full-attention baseline for quality comparison
    cosine_similarity_output— quality metric helper
"""

from dcr_attention.kernel.abkv.abkv_cache import ABKVCache
from dcr_attention.kernel.abkv.abkv_reference import (
    abkv_attention_reference,
    abkv_attention_with_signature,
    build_abkv_cache,
    cosine_similarity_output,
    sdpa_reference,
)

__all__ = [
    "ABKVCache",
    "build_abkv_cache",
    "abkv_attention_reference",
    "abkv_attention_with_signature",
    "sdpa_reference",
    "cosine_similarity_output",
]
