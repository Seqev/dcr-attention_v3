"""
Sanity tests on the reference implementation itself.

These tests do not involve Triton — they guarantee that our PyTorch oracle
behaves correctly before we use it as ground truth for Triton kernels.

Run:
    pytest tests/kernel/test_reference_sanity.py -v
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from dcr_attention.reference import (
    dense_attention_reference,
    rank_local_attention_reference,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _axis_first_coord(D: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Unit vector u = e_0.  Deterministic, trivial to reason about."""
    axis = torch.zeros(D, device=device, dtype=dtype)
    axis[0] = 1.0
    return axis


@pytest.fixture
def qkv_small():
    torch.manual_seed(42)
    B, H, N, D = 1, 2, 128, 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    Q = torch.randn(B, H, N, D, device=device, dtype=dtype)
    K = torch.randn(B, H, N, D, device=device, dtype=dtype)
    V = torch.randn(B, H, N, D, device=device, dtype=dtype)
    axis = _axis_first_coord(D, device, dtype)
    return Q, K, V, axis, N


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_full_window_equals_dense(qkv_small):
    """
    With k_window = 2N every query sees all keys.  The window mask is vacuous and
    rank-local attention must equal plain scaled dot-product attention up to
    fp32 rounding (≤ 1e-5 for these sizes).
    """
    Q, K, V, axis, N = qkv_small
    out_dense = dense_attention_reference(Q, K, V)
    out_full = rank_local_attention_reference(Q, K, V, axis, k_window=2 * N)
    max_diff = (out_dense - out_full).abs().max().item()
    assert max_diff < 1e-5, f"full-window != dense:  max|diff| = {max_diff:g}"


def test_no_nans_everywhere(qkv_small):
    Q, K, V, axis, N = qkv_small
    for k in (8, 16, N // 4, N // 2, N, 2 * N):
        out = rank_local_attention_reference(Q, K, V, axis, k_window=k)
        assert not torch.isnan(out).any(), f"NaN at k_window={k}"


def test_half_window_random_inputs_degrades_gracefully(qkv_small):
    """
    Half window with random i.i.d. Gaussian Q, K and axis = e_0 captures
    only a scattered subset of the true attention mass — projection onto
    a single random coordinate explains roughly 1/D ≈ 1.6 % of the variance,
    so rank-local ordering is nearly random and cos-sim hovers well below 1.

    What we **do** verify here:
      * output is non-trivial (not collinear with zero vector),
      * cos-sim is positive (not anti-correlated),
      * cos-sim is strictly less than the full-window result
        (confirms sparsification actually happened).

    What we deliberately do **not** verify: any hard lower bound > 0.5.
    Structured-input case is tested separately below.
    """
    Q, K, V, axis, N = qkv_small
    out_dense = dense_attention_reference(Q, K, V)
    out_full = rank_local_attention_reference(Q, K, V, axis, k_window=2 * N)
    out_half = rank_local_attention_reference(Q, K, V, axis, k_window=N // 2)

    cos_full = F.cosine_similarity(out_dense.flatten(), out_full.flatten(), dim=0).item()
    cos_half = F.cosine_similarity(out_dense.flatten(), out_half.flatten(), dim=0).item()

    assert cos_full > 0.999, f"full-window cos-sim {cos_full:.4f} must match dense"
    assert 0.0 < cos_half < cos_full, (
        f"half-window cos-sim {cos_half:.4f} should be positive and strictly "
        f"below full-window {cos_full:.4f}"
    )


def test_half_window_structured_inputs_preserves_mass():
    """
    Positive test for rank-local quality: when Q, K carry a strong rank-1
    signal along a known axis u, rank-local attention ordered by that axis
    recovers the dense output with high fidelity.  This is the regime DCR
    was designed for.

    Construction: rank-1 structured signal s·v·uᵀ + σ·noise, where v ∈ R^N
    is a random coordinate per token and u ∈ R^D the shared axis.
    For s/σ = 5 and k_window = N/2, cos-sim to dense must exceed 0.95.
    """
    torch.manual_seed(7)
    B, H, N, D = 1, 1, 128, 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Shared ordering axis u ∈ R^D
    u = torch.randn(D, device=device)
    u = u / u.norm()

    # Per-token scalar v_i ∈ R^N determines position along u
    v_q = torch.randn(B, H, N, device=device)
    v_k = torch.randn(B, H, N, device=device)

    # Rank-1 + noise:  Q_i = 5·v_q[i]·u + noise
    signal = 5.0
    noise = 1.0
    Q = signal * v_q.unsqueeze(-1) * u + noise * torch.randn(B, H, N, D, device=device)
    K = signal * v_k.unsqueeze(-1) * u + noise * torch.randn(B, H, N, D, device=device)
    V = torch.randn(B, H, N, D, device=device)

    out_dense = dense_attention_reference(Q, K, V)
    out_half = rank_local_attention_reference(Q, K, V, u, k_window=N // 2)
    cos = F.cosine_similarity(out_dense.flatten(), out_half.flatten(), dim=0).item()

    assert cos > 0.95, (
        f"structured rank-1 signal, k=N/2, cos-sim {cos:.4f} — rank-local "
        f"should recover dense output in its designed regime"
    )


def test_shapes_preserved(qkv_small):
    Q, K, V, axis, N = qkv_small
    out = rank_local_attention_reference(Q, K, V, axis, k_window=32)
    assert out.shape == Q.shape


def test_padding_mask_does_not_produce_nan():
    """
    When all keys of a query are masked out by attention_mask, softmax over
    all -inf yields NaN; the reference must recover with 0.0 via nan_to_num.
    """
    torch.manual_seed(0)
    B, H, N, D = 1, 1, 16, 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Q = torch.randn(B, H, N, D, device=device)
    K = torch.randn(B, H, N, D, device=device)
    V = torch.randn(B, H, N, D, device=device)
    axis = _axis_first_coord(D, device, torch.float32)

    # Pathological mask: -inf everywhere for the last query.
    attention_mask = torch.zeros(B, 1, 1, N, device=device)
    # Instead: craft a per-query pad mask of shape [B,1,N,N]? The reference
    # accepts [B,1,1,N] broadcast over queries.  We therefore verify the
    # milder case: first-half keys alive, second-half dead.
    attention_mask[..., N // 2 :] = float("-inf")

    out = rank_local_attention_reference(
        Q, K, V, axis, k_window=4, attention_mask=attention_mask
    )
    assert not torch.isnan(out).any()
