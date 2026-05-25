"""
Rank-local attention kernels.

Public entry points:
    rank_local_attention       — autograd-wrapped, dispatches Triton/torch.
    rank_local_fwd_torch       — pure-torch path (takes external indices).
    rank_local_fwd_triton      — CUDA Triton path (takes external indices).
    prepare_sort_indices,
    gather_by_sort_idx         — sort-side bookkeeping helpers.

See docs/design/phase1_forward_kernel.md for the architecture.
"""

from dcr_attention.kernel.rank_local_attention import (
    RankLocalAttentionFn,
    rank_local_attention,
)
from dcr_attention.kernel.rank_local_fwd_torch import rank_local_fwd_torch
from dcr_attention.kernel.rank_local_fwd_triton import (
    TRITON_AVAILABLE,
    rank_local_fwd_triton,
)
from dcr_attention.kernel.sort_helpers import (
    SortIndices,
    gather_by_sort_idx,
    prepare_sort_indices,
    scatter_by_inv_perm,
)

__all__ = [
    "rank_local_attention",
    "RankLocalAttentionFn",
    "rank_local_fwd_torch",
    "rank_local_fwd_triton",
    "TRITON_AVAILABLE",
    "prepare_sort_indices",
    "gather_by_sort_idx",
    "scatter_by_inv_perm",
    "SortIndices",
]
