"""
Unit tests for dcr_attention.kernel.qaxis_topk_reference (M1 reference).

Tests map to spec §4.2 required cases.  All run on CPU — no GPU needed.

Naming convention:
  test_<case> — white-box tests exercising individual step helpers.
  test_m1_*   — black-box tests against the public entry point.
"""

from __future__ import annotations

import math

import pytest
import torch

from dcr_attention.kernel.qaxis_topk_reference import (
    _compute_u_Q,
    _compute_projection_scores,
    _select_topk_indices,
    _gather_kv,
    _fused_softmax_attention,
    topk_qaxis_attention_reference,
)

_BF16 = torch.bfloat16
_F32 = torch.float32


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rand_bf16(*shape, seed=0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(*shape).to(_BF16)


def _full_attention_bf16(
    Q: torch.Tensor,        # [B, H, D] bf16
    K: torch.Tensor,        # [B, H, N, D] bf16
    V: torch.Tensor,        # [B, H, N, D] bf16
) -> torch.Tensor:          # [B, H, D] bf16
    """Brute-force softmax attention over all N keys, for comparison."""
    B, H, N, D = K.shape
    scale = 1.0 / math.sqrt(D)
    Q_f = Q.float().unsqueeze(-2)                       # [B, H, 1, D]
    K_f = K.float()
    V_f = V.float()
    scores = torch.einsum("bhid,bhjd->bhij", Q_f, K_f) * scale  # [B, H, 1, N]
    weights = torch.softmax(scores, dim=-1)             # [B, H, 1, N]
    out = torch.einsum("bhij,bhjd->bhid", weights, V_f)  # [B, H, 1, D]
    return out.squeeze(-2).to(_BF16)                    # [B, H, D]


# ---------------------------------------------------------------------------
# §4.2.1  Degenerate: N <= k_eff raises ValueError
# ---------------------------------------------------------------------------

def test_degenerate_raises_value_error():
    Q = _rand_bf16(1, 4, 32)
    K = _rand_bf16(1, 2, 16, 32)
    V = _rand_bf16(1, 2, 16, 32)
    with pytest.raises(ValueError, match="caller must dispatch to SDPA"):
        topk_qaxis_attention_reference(Q, K, V, k_eff=16)


def test_degenerate_equal_raises():
    Q = _rand_bf16(1, 4, 32)
    K = _rand_bf16(1, 2, 8, 32)
    V = _rand_bf16(1, 2, 8, 32)
    with pytest.raises(ValueError):
        topk_qaxis_attention_reference(Q, K, V, k_eff=8)


# ---------------------------------------------------------------------------
# §4.2.2  Output shape and dtype
# ---------------------------------------------------------------------------

def test_output_shape_and_dtype():
    B, H, H_kv, N, D, k = 2, 8, 4, 64, 32, 16
    Q = _rand_bf16(B, H, D)
    K = _rand_bf16(B, H_kv, N, D)
    V = _rand_bf16(B, H_kv, N, D)
    O = topk_qaxis_attention_reference(Q, K, V, k_eff=k)
    assert O.shape == (B, H, D), f"expected ({B},{H},{D}), got {O.shape}"
    assert O.dtype == _BF16, f"expected bfloat16, got {O.dtype}"


# ---------------------------------------------------------------------------
# §4.2.3  Output dtype — separate minimal case
# ---------------------------------------------------------------------------

def test_output_is_bfloat16():
    Q = _rand_bf16(1, 1, 16)
    K = _rand_bf16(1, 1, 32, 16)
    V = _rand_bf16(1, 1, 32, 16)
    O = topk_qaxis_attention_reference(Q, K, V, k_eff=8)
    assert O.dtype == _BF16


# ---------------------------------------------------------------------------
# §4.2.4  Zero-query: eps guard prevents NaN
# ---------------------------------------------------------------------------

def test_zero_query_no_nan():
    Q = torch.zeros(1, 2, 32, dtype=_BF16)
    K = _rand_bf16(1, 2, 64, 32, seed=1)
    V = _rand_bf16(1, 2, 64, 32, seed=2)
    O = topk_qaxis_attention_reference(Q, K, V, k_eff=16)
    assert torch.isfinite(O).all(), "NaN/inf in output with zero query"


# ---------------------------------------------------------------------------
# §4.2.5  k_eff = N-1: near-full attention agrees with brute-force (soft)
# ---------------------------------------------------------------------------

def test_k_eff_near_full_matches_brute_force():
    """k_eff = N-1 drops only the worst key; expect close match to full attn."""
    torch.manual_seed(42)
    B, H, N, D = 1, 2, 32, 64
    Q = _rand_bf16(B, H, D)
    K = _rand_bf16(B, H, N, D)
    V = _rand_bf16(B, H, N, D)

    # M1 with k_eff = N-1
    O_m1 = topk_qaxis_attention_reference(Q, K, V, k_eff=N - 1)

    # Brute-force over all N keys
    O_ref = _full_attention_bf16(Q, K, V)

    # Allow generous tolerance — one key dropped, but still should be close
    diff = (O_m1.float() - O_ref.float()).abs().max().item()
    assert diff < 0.05, f"max|M1 - full_attn| = {diff:.4f} (expected < 0.05)"


# ---------------------------------------------------------------------------
# §4.2.6  Brute-force agreement: top-K selection must match manual argsort
# ---------------------------------------------------------------------------

def test_topk_selection_matches_manual_argsort():
    """_select_topk_indices must return the same k indices as manual argsort."""
    torch.manual_seed(7)
    B, H, N, D, k = 2, 4, 50, 32, 10
    Q = _rand_bf16(B, H, D)

    # unique values to avoid tie ambiguity
    scores = torch.arange(B * H * N, dtype=_F32).reshape(B, H, N)
    scores = scores + torch.rand_like(scores) * 0.01
    scores_bf16 = scores.to(_BF16)

    idx = _select_topk_indices(scores_bf16, k)
    assert idx.shape == (B, H, k)

    # Manual: argsort descending (stable=True to match §6 tie-breaking) → top-k → sort ascending
    expected_topk = scores_bf16.float().argsort(dim=-1, descending=True, stable=True)[:, :, :k]
    expected_sorted, _ = expected_topk.sort(dim=-1)

    assert torch.equal(idx, expected_sorted), "topk indices differ from manual argsort"


# ---------------------------------------------------------------------------
# §4.2.7  Tie-breaking: equal scores → smaller index wins (§6)
# ---------------------------------------------------------------------------

def test_tie_breaking_smaller_index_wins():
    """When all scores are identical, top-k must select indices 0..k-1."""
    B, H, N, k = 1, 1, 20, 5
    # All scores equal → stable argsort preserves input order → indices 0..k-1
    scores = torch.ones(B, H, N, dtype=_BF16)
    idx = _select_topk_indices(scores, k)
    expected = torch.arange(k).reshape(1, 1, k)
    assert torch.equal(idx, expected), f"tie-breaking failed: got {idx}"


# ---------------------------------------------------------------------------
# Step-level white-box tests
# ---------------------------------------------------------------------------

def test_compute_u_Q_unit_norm():
    """u_Q must have unit norm along D for every (b, h)."""
    Q = _rand_bf16(2, 4, 64)
    u = _compute_u_Q(Q, eps=1e-12)
    assert u.dtype == _F32
    norms = u.norm(dim=-1)  # [B, H]
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
        f"max |‖u_Q‖ - 1| = {(norms - 1).abs().max().item():.2e}"


def test_compute_projection_scores_shape_and_dtype():
    B, H_kv, N, D, H = 2, 4, 32, 64, 8
    K = _rand_bf16(B, H_kv, N, D)
    Q = _rand_bf16(B, H, D)
    u = _compute_u_Q(Q, eps=1e-12)
    scores = _compute_projection_scores(K, u)
    assert scores.shape == (B, H, N)
    assert scores.dtype == _BF16


def test_gather_kv_correct_values():
    """Gathered K_sel rows must equal the corresponding K_cache rows."""
    B, H, N, D, k = 1, 2, 16, 8, 4
    K = _rand_bf16(B, H, N, D)
    V = _rand_bf16(B, H, N, D)
    # Hand-craft indices
    idx = torch.tensor([[[0, 3, 7, 11], [1, 2, 5, 9]]])  # [1, 2, 4]
    K_sel, V_sel = _gather_kv(K, V, idx)
    for h in range(H):
        for ki, pos in enumerate(idx[0, h]):
            assert torch.equal(K_sel[0, h, ki], K[0, h, pos]), \
                f"K_sel mismatch at h={h}, ki={ki}, pos={pos}"


def test_fused_softmax_fp32_state():
    """Internal running state must be fp32 — verified by output finite-ness."""
    B, H, k, D = 1, 2, 128, 32
    K_sel = _rand_bf16(B, H, k, D)
    V_sel = _rand_bf16(B, H, k, D)
    Q = _rand_bf16(B, H, D)
    O = _fused_softmax_attention(K_sel, V_sel, Q, B_block=32)
    assert O.dtype == _BF16
    assert torch.isfinite(O).all()


# ---------------------------------------------------------------------------
# heap equivalence (mentioned in module docstring)
# ---------------------------------------------------------------------------

def test_heap_equivalence():
    """argsort(stable=True) top-K equals explicit loop over N with max-heap."""
    torch.manual_seed(99)
    B, H, N, k = 1, 1, 40, 8
    scores = torch.randn(B, H, N, dtype=_BF16)

    # M1 selection
    m1_idx = _select_topk_indices(scores, k)  # [1, 1, k] sorted asc

    # Explicit heap: iterate positions in order, track top-k with min-heap
    import heapq
    s = scores[0, 0].float().tolist()
    heap = []  # min-heap of (score, index)
    for i, v in enumerate(s):
        if len(heap) < k:
            heapq.heappush(heap, (v, i))
        elif v > heap[0][0]:
            heapq.heapreplace(heap, (v, i))
        elif v == heap[0][0] and i < heap[0][1]:
            heapq.heapreplace(heap, (v, i))
    heap_idx = sorted(idx for _, idx in heap)

    ref = torch.tensor(heap_idx, dtype=torch.int64).reshape(1, 1, k)
    assert torch.equal(m1_idx, ref), \
        f"heap vs argsort mismatch:\n  m1:   {m1_idx}\n  heap: {ref}"
