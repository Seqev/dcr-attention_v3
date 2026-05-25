"""
Unit tests for sort_helpers (Phase 1.3 schema).

Verify identities:
  * sort_idx_k is a permutation of [0, N), rank_of_k is its inverse.
  * sort_idx_q is a permutation of [0, N), inv_q_perm is its inverse.
  * r_center[t] = |{j : z_k[j] < z_q[sort_idx_q[t]]}|  (sorted Q-position semantics).
  * r_center is monotonically non-decreasing in t.
  * gather_by_sort_idx and scatter_by_inv_perm round-trip.
"""

from __future__ import annotations

import pytest
import torch

from dcr_attention.kernel.sort_helpers import (
    gather_by_sort_idx,
    prepare_sort_indices,
    scatter_by_inv_perm,
)


def _random_qkv(B=2, H=3, N=128, D=32, seed=0):
    torch.manual_seed(seed)
    Q = torch.randn(B, H, N, D)
    K = torch.randn(B, H, N, D)
    return Q, K


# ---------------------------------------------------------------------------
# Permutation identities
# ---------------------------------------------------------------------------

def test_sort_idx_k_is_permutation():
    Q, K = _random_qkv()
    axis = torch.randn(K.shape[-1]); axis /= axis.norm()
    idx = prepare_sort_indices(Q, K, axis)
    N = K.shape[-2]
    ref = torch.arange(N)
    for b in range(K.shape[0]):
        for h in range(K.shape[1]):
            assert torch.equal(idx.sort_idx_k[b, h].sort().values, ref)


def test_sort_idx_q_is_permutation():
    Q, K = _random_qkv()
    axis = torch.randn(K.shape[-1]); axis /= axis.norm()
    idx = prepare_sort_indices(Q, K, axis)
    N = K.shape[-2]
    ref = torch.arange(N)
    for b in range(K.shape[0]):
        for h in range(K.shape[1]):
            assert torch.equal(idx.sort_idx_q[b, h].sort().values, ref)


def test_rank_of_k_is_inverse_of_sort_idx_k():
    Q, K = _random_qkv()
    axis = torch.randn(K.shape[-1]); axis /= axis.norm()
    idx = prepare_sort_indices(Q, K, axis)
    N = K.shape[-2]
    for b in range(K.shape[0]):
        for h in range(K.shape[1]):
            si = idx.sort_idx_k[b, h]
            ri = idx.rank_of_k[b, h]
            assert torch.equal(si.gather(0, ri), torch.arange(N))
            assert torch.equal(ri.gather(0, si), torch.arange(N))


def test_inv_q_perm_is_inverse_of_sort_idx_q():
    Q, K = _random_qkv()
    axis = torch.randn(K.shape[-1]); axis /= axis.norm()
    idx = prepare_sort_indices(Q, K, axis)
    N = K.shape[-2]
    for b in range(K.shape[0]):
        for h in range(K.shape[1]):
            sq = idx.sort_idx_q[b, h]
            iq = idx.inv_q_perm[b, h]
            assert torch.equal(sq.gather(0, iq), torch.arange(N))
            assert torch.equal(iq.gather(0, sq), torch.arange(N))


# ---------------------------------------------------------------------------
# r_center semantics — sorted Q-position version (Phase 1.3 change)
# ---------------------------------------------------------------------------

def test_r_center_matches_sorted_definition():
    """
    Phase 1.3 semantics:  r_center[t] = |{j : z_k[j] < z_q[sort_idx_q[t]]}|
    """
    Q, K = _random_qkv(B=1, H=1, N=64, D=16)
    axis = torch.zeros(16); axis[0] = 1.0
    idx = prepare_sort_indices(Q, K, axis)

    z_q, z_k = idx.z_q[0, 0], idx.z_k[0, 0]
    sort_idx_q = idx.sort_idx_q[0, 0]
    z_q_sorted = z_q.gather(0, sort_idx_q)

    ref = (z_k.unsqueeze(0) < z_q_sorted.unsqueeze(-1)).sum(dim=-1).clamp(0, 63)
    assert torch.equal(idx.r_center[0, 0], ref)


def test_r_center_is_monotone_in_t():
    """
    The whole point of Phase 1.3: r_center is non-decreasing along t after Q-sort.
    """
    Q, K = _random_qkv()
    axis = torch.randn(K.shape[-1]); axis /= axis.norm()
    idx = prepare_sort_indices(Q, K, axis)

    diffs = idx.r_center[..., 1:] - idx.r_center[..., :-1]
    assert (diffs >= 0).all(), (
        f"r_center has {(diffs < 0).sum().item()} non-monotone steps; "
        f"min diff = {diffs.min().item()}"
    )


# ---------------------------------------------------------------------------
# Gather / scatter round-trip
# ---------------------------------------------------------------------------

def test_gather_then_inverse_recovers_original():
    """
    For any permutation π and its inverse π⁻¹:
        scatter_by_inv_perm(gather_by_sort_idx(X, π), π⁻¹) == X
    """
    Q, K = _random_qkv()
    axis = torch.randn(K.shape[-1]); axis /= axis.norm()
    idx = prepare_sort_indices(Q, K, axis)

    K_sorted = gather_by_sort_idx(K, idx.sort_idx_k)
    K_recovered = scatter_by_inv_perm(K_sorted, idx.rank_of_k)
    assert torch.equal(K, K_recovered)

    Q_sorted = gather_by_sort_idx(Q, idx.sort_idx_q)
    Q_recovered = scatter_by_inv_perm(Q_sorted, idx.inv_q_perm)
    assert torch.equal(Q, Q_recovered)


def test_scatter_handles_3d_tensors_for_lse():
    """LSE is [B,H,N], not [B,H,N,D]; scatter must support both."""
    torch.manual_seed(0)
    B, H, N = 2, 3, 64
    LSE = torch.randn(B, H, N)
    perm = torch.stack([torch.randperm(N) for _ in range(B * H)]).reshape(B, H, N)
    inv_perm = torch.argsort(perm, dim=-1)

    LSE_perm = torch.gather(LSE, dim=2, index=perm)
    LSE_recovered = scatter_by_inv_perm(LSE_perm, inv_perm)
    assert torch.equal(LSE, LSE_recovered)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_rejects_mismatched_shapes():
    Q, K = _random_qkv()
    # Wrong axis size
    with pytest.raises(ValueError):
        prepare_sort_indices(Q, K, torch.randn(Q.shape[-1] + 1))
    # Phase 2-pre: Q.N != K.N is now ALLOWED (decode shape).
    # But B mismatch must still fail.
    with pytest.raises(ValueError):
        prepare_sort_indices(Q[:1], K, torch.randn(Q.shape[-1]))    # different B
    # Non-4D
    with pytest.raises(ValueError):
        prepare_sort_indices(Q[0], K[0], torch.randn(Q.shape[-1]))


def test_accepts_decode_shape_q_n_neq_k_n():
    """Phase 2-pre: Q.N=1, K.N=N_kv must work for decode."""
    torch.manual_seed(0)
    B, H, D = 1, 4, 16
    N_kv = 64
    Q = torch.randn(B, H, 1, D)
    K = torch.randn(B, H, N_kv, D)
    axis = torch.zeros(D); axis[0] = 1.0

    idx = prepare_sort_indices(Q, K, axis)

    assert idx.sort_idx_k.shape == (B, H, N_kv)
    assert idx.rank_of_k.shape == (B, H, N_kv)
    assert idx.sort_idx_q.shape == (B, H, 1)
    assert idx.inv_q_perm.shape == (B, H, 1)
    assert idx.r_center.shape == (B, H, 1)
    # r_center must be a valid index into K (0..N_kv-1).
    assert (idx.r_center >= 0).all() and (idx.r_center < N_kv).all()
    # sort_idx_q for N_q=1 is the trivial permutation [0].
    assert torch.equal(idx.sort_idx_q, torch.zeros(B, H, 1, dtype=torch.long))


def test_gather_by_sort_idx_validates_shapes():
    X = torch.randn(2, 3, 64, 16)
    bad_idx = torch.randint(0, 64, (2, 3))
    with pytest.raises(ValueError):
        gather_by_sort_idx(X, bad_idx)


def test_scatter_validates_shapes():
    bad = torch.randn(64, 16)
    inv_perm = torch.randint(0, 64, (1, 1, 64))
    with pytest.raises(ValueError):
        scatter_by_inv_perm(bad, inv_perm)
