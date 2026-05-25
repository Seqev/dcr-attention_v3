"""
Correctness: ``rank_local_fwd_torch`` with external sort indices must produce
numerically identical output to the reference implementation (internal sort)
on small N where both fit in memory.

This is the ground-truth test for the *new architecture* (external indices).
A pass here means the sort/gather decomposition is mathematically correct.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from dcr_attention.reference import rank_local_attention_reference
from dcr_attention.kernel.sort_helpers import (
    gather_by_sort_idx,
    prepare_sort_indices,
    scatter_by_inv_perm,
)
from dcr_attention.kernel.rank_local_fwd_torch import rank_local_fwd_torch


@pytest.mark.parametrize("N", [64, 128, 256])
@pytest.mark.parametrize("k_window", [8, 32, 64, 256])
def test_torch_path_matches_reference(N, k_window):
    torch.manual_seed(42)
    B, H, D = 1, 2, 32
    Q = torch.randn(B, H, N, D)
    K = torch.randn(B, H, N, D)
    V = torch.randn(B, H, N, D)
    axis = torch.randn(D); axis = axis / axis.norm()

    out_ref = rank_local_attention_reference(Q, K, V, axis, k_window=k_window)

    # Phase 1.3 contract: kernel/fallback expects Q in sorted-Q-order.
    # Caller (or public API) is responsible for the perm + inverse-perm.
    idx = prepare_sort_indices(Q, K, axis)
    K_sorted = gather_by_sort_idx(K, idx.sort_idx_k)
    V_sorted = gather_by_sort_idx(V, idx.sort_idx_k)
    Q_sorted = gather_by_sort_idx(Q, idx.sort_idx_q)
    out_sorted, lse_sorted = rank_local_fwd_torch(
        Q_sorted, K_sorted, V_sorted,
        idx.rank_of_k, idx.r_center,
        k_window=k_window,
    )
    # Restore to original Q-order for comparison with reference
    out_torch = scatter_by_inv_perm(out_sorted, idx.inv_q_perm)
    lse = scatter_by_inv_perm(lse_sorted, idx.inv_q_perm) if lse_sorted is not None else None

    max_abs = (out_ref - out_torch).abs().max().item()
    assert max_abs < 1e-5, (
        f"N={N}, k={k_window}: max|ref - torch| = {max_abs:g} > 1e-5"
    )
    assert lse is not None
    assert lse.shape == (B, H, N)
    assert torch.isfinite(lse).all() or (lse == -float("inf")).any()


def test_torch_path_full_window_equals_dense():
    """Sanity: k=2N via external indices reproduces dense attention."""
    torch.manual_seed(0)
    B, H, N, D = 1, 2, 128, 32
    Q = torch.randn(B, H, N, D)
    K = torch.randn(B, H, N, D)
    V = torch.randn(B, H, N, D)
    axis = torch.randn(D); axis = axis / axis.norm()

    idx = prepare_sort_indices(Q, K, axis)
    K_sorted = gather_by_sort_idx(K, idx.sort_idx_k)
    V_sorted = gather_by_sort_idx(V, idx.sort_idx_k)
    Q_sorted = gather_by_sort_idx(Q, idx.sort_idx_q)

    out_sorted, _ = rank_local_fwd_torch(
        Q_sorted, K_sorted, V_sorted, idx.rank_of_k, idx.r_center,
        k_window=2 * N,
    )
    out = scatter_by_inv_perm(out_sorted, idx.inv_q_perm)

    # Dense reference (in original Q-order)
    scale = 1.0 / (D ** 0.5)
    scores = torch.einsum("bhid,bhjd->bhij", Q, K) * scale
    dense = torch.einsum("bhij,bhjd->bhid", F.softmax(scores, dim=-1), V)

    assert torch.allclose(out, dense, atol=1e-5)


def test_torch_path_rejects_attention_mask():
    """Phase 1.2: attention_mask not yet supported in the fallback — must raise."""
    torch.manual_seed(0)
    B, H, N, D = 1, 1, 32, 8
    Q = K = V = torch.randn(B, H, N, D)
    axis = torch.zeros(D); axis[0] = 1.0
    idx = prepare_sort_indices(Q, K, axis)
    K_s = gather_by_sort_idx(K, idx.sort_idx_k)
    V_s = gather_by_sort_idx(V, idx.sort_idx_k)
    mask = torch.zeros(B, 1, 1, N)

    with pytest.raises(NotImplementedError):
        rank_local_fwd_torch(
            Q, K_s, V_s, idx.rank_of_k, idx.r_center,
            k_window=8, attention_mask=mask,
        )
