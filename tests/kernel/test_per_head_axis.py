"""
Phase 2a (D4) acceptance: per-head axis API extension.

The kernel/sort_helpers/public-API now accept ``axis`` as either:
  * ``[D]``      — single shared axis (Phase 1.x compatibility).
  * ``[B, H, D]`` — per-(b, h) axes (Phase 2a Llama integration).

These tests verify:
  1. The two paths agree numerically when the [B, H, D] axis is the [D] axis
     broadcast (correctness equivalence).
  2. Per-head axes can differ and produce shape-correct, finite output.
  3. Validation rejects mis-shaped axes.

All tests CPU-only, fp32.  GPU parity is established by the existing
test_triton_gpu_correctness suite which uses [D] axis — D4 is purely a
host-side path through ``prepare_sort_indices``, so CPU validation suffices
for correctness.
"""

from __future__ import annotations

import pytest
import torch

from dcr_attention.kernel.rank_local_attention import rank_local_attention
from dcr_attention.kernel.sort_helpers import prepare_sort_indices


# ---------------------------------------------------------------------------
# Equivalence: [D] axis must agree with [B, H, D] axis broadcast
# ---------------------------------------------------------------------------

def test_prepare_sort_indices_axis_3d_matches_axis_1d_when_broadcast():
    """If we pass axis=[D] vs axis=[D].expand(B,H,D), SortIndices must match."""
    torch.manual_seed(0)
    B, H, N, D = 2, 4, 64, 16
    Q = torch.randn(B, H, N, D)
    K = torch.randn(B, H, N, D)
    axis_1d = torch.randn(D); axis_1d = axis_1d / axis_1d.norm()
    axis_3d = axis_1d.view(1, 1, D).expand(B, H, D).contiguous()

    idx_1d = prepare_sort_indices(Q, K, axis_1d)
    idx_3d = prepare_sort_indices(Q, K, axis_3d)

    # Same projections (within fp32 noise of einsum reordering)
    assert torch.allclose(idx_1d.z_q, idx_3d.z_q, atol=1e-6)
    assert torch.allclose(idx_1d.z_k, idx_3d.z_k, atol=1e-6)

    # Same permutations (identical projections → identical argsort)
    assert torch.equal(idx_1d.sort_idx_k, idx_3d.sort_idx_k)
    assert torch.equal(idx_1d.sort_idx_q, idx_3d.sort_idx_q)
    assert torch.equal(idx_1d.r_center, idx_3d.r_center)


def test_rank_local_attention_axis_3d_matches_axis_1d():
    """Public API: [D] axis vs broadcast [B, H, D] must give bitwise-close output."""
    torch.manual_seed(0)
    B, H, N, D = 2, 4, 64, 16
    Q = torch.randn(B, H, N, D)
    K = torch.randn(B, H, N, D)
    V = torch.randn(B, H, N, D)
    axis_1d = torch.randn(D); axis_1d = axis_1d / axis_1d.norm()
    axis_3d = axis_1d.view(1, 1, D).expand(B, H, D).contiguous()

    out_1d = rank_local_attention(Q, K, V, axis_1d, k_window=16)
    out_3d = rank_local_attention(Q, K, V, axis_3d, k_window=16)

    diff = (out_1d - out_3d).abs().max().item()
    assert diff < 1e-5, f"axis [D] vs [B,H,D] broadcast mismatch: {diff:g}"


# ---------------------------------------------------------------------------
# Per-head distinct axes: shape-correct + finite
# ---------------------------------------------------------------------------

def test_per_head_distinct_axes_finite_output():
    """Different axis per (b, h) — output must be finite and shape-correct."""
    torch.manual_seed(1)
    B, H, N, D = 2, 4, 64, 16
    Q = torch.randn(B, H, N, D)
    K = torch.randn(B, H, N, D)
    V = torch.randn(B, H, N, D)
    # Each (b, h) gets its own random axis
    axis_3d = torch.randn(B, H, D)
    axis_3d = axis_3d / axis_3d.norm(dim=-1, keepdim=True)

    out = rank_local_attention(Q, K, V, axis_3d, k_window=16)

    assert out.shape == Q.shape
    assert torch.isfinite(out).all()


def test_per_head_axes_produce_per_head_distinct_z_projections():
    """Different axes per (b, h) must produce different z-projections per (b, h)."""
    torch.manual_seed(2)
    B, H, N, D = 1, 4, 32, 8
    Q = torch.randn(B, H, N, D)
    K = torch.randn(B, H, N, D)
    # Two heads get axis e_0, two heads get axis e_1 — must give distinct z values
    axis_3d = torch.zeros(B, H, D)
    axis_3d[0, 0, 0] = 1.0
    axis_3d[0, 1, 0] = 1.0
    axis_3d[0, 2, 1] = 1.0
    axis_3d[0, 3, 1] = 1.0

    idx = prepare_sort_indices(Q, K, axis_3d)

    # heads 0,1 used axis e_0 → same z_k slice as Q[0,0,:,0]
    assert torch.allclose(idx.z_k[0, 0], K[0, 0, :, 0], atol=1e-6)
    assert torch.allclose(idx.z_k[0, 1], K[0, 1, :, 0], atol=1e-6)
    # heads 2,3 used axis e_1 → z_k slice from K[..., 1]
    assert torch.allclose(idx.z_k[0, 2], K[0, 2, :, 1], atol=1e-6)
    assert torch.allclose(idx.z_k[0, 3], K[0, 3, :, 1], atol=1e-6)


# ---------------------------------------------------------------------------
# Validation: bad axis shapes rejected
# ---------------------------------------------------------------------------

def test_axis_2d_rejected():
    """axis.dim() == 2 is neither [D] nor [B, H, D] — must be rejected."""
    Q = K = V = torch.randn(2, 4, 8, 16)
    axis_2d = torch.randn(2, 16)         # ambiguous
    with pytest.raises(ValueError, match="axis must be"):
        rank_local_attention(Q, K, V, axis_2d, k_window=4)


def test_axis_3d_wrong_B_rejected():
    """axis [B', H, D] with B' != B must be rejected."""
    Q = K = V = torch.randn(2, 4, 8, 16)
    axis_3d_bad = torch.randn(3, 4, 16)  # B mismatch
    with pytest.raises(ValueError, match="axis shape"):
        rank_local_attention(Q, K, V, axis_3d_bad, k_window=4)


def test_axis_3d_wrong_H_rejected():
    """axis [B, H', D] with H' != H must be rejected."""
    Q = K = V = torch.randn(2, 4, 8, 16)
    axis_3d_bad = torch.randn(2, 5, 16)  # H mismatch
    with pytest.raises(ValueError, match="axis shape"):
        rank_local_attention(Q, K, V, axis_3d_bad, k_window=4)


def test_axis_3d_wrong_D_rejected():
    """axis [B, H, D'] with D' != D must be rejected."""
    Q = K = V = torch.randn(2, 4, 8, 16)
    axis_3d_bad = torch.randn(2, 4, 17)  # D mismatch
    with pytest.raises(ValueError, match="axis shape"):
        rank_local_attention(Q, K, V, axis_3d_bad, k_window=4)


# ---------------------------------------------------------------------------
# Decode shape (Phase 2-pre) × per-head axis (Phase 2a) — combined
# ---------------------------------------------------------------------------

def test_per_head_axis_works_in_decode_shape():
    """The two Phase-2 features (decode shape + per-head axis) compose correctly."""
    torch.manual_seed(3)
    B, H, D = 1, 8, 16
    N_kv = 256
    Q = torch.randn(B, H, 1, D)        # decode: N_q = 1
    K = torch.randn(B, H, N_kv, D)
    V = torch.randn(B, H, N_kv, D)
    axis_3d = torch.randn(B, H, D)
    axis_3d = axis_3d / axis_3d.norm(dim=-1, keepdim=True)

    out = rank_local_attention(Q, K, V, axis_3d, k_window=32)

    assert out.shape == (B, H, 1, D)
    assert torch.isfinite(out).all()
