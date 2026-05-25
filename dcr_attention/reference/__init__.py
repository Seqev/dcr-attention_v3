"""PyTorch reference implementations. Ground-truth for Triton correctness tests."""

from dcr_attention.reference.rank_local import (
    rank_local_attention_reference,
    dense_attention_reference,
)

__all__ = [
    "rank_local_attention_reference",
    "dense_attention_reference",
]
