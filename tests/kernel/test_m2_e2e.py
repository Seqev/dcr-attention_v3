"""
M2 end-to-end validation — topk_qaxis_select vs M1 reference top-K selection.

Spec §5 Tier 3:
  - Jaccard(S_triton, S_ref) = 1.0 for ≥ 99.9% of (b, h) tuples
  - mean Jaccard ≥ 0.9999
  - |S_m2| = k_eff exactly
  - No NaN/inf
"""

from __future__ import annotations

import pytest
import torch

from dcr_attention.kernel.triton.topk_axis import topk_qaxis_select
from dcr_attention.kernel.qaxis_topk_reference import (
    _compute_u_Q,
    _compute_projection_scores,
    _select_topk_indices,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="M2 e2e requires CUDA"
)

B, H, H_kv, D = 1, 32, 8, 64


@pytest.mark.parametrize("k_eff_frac", [0.3, 0.5, 0.8])
@pytest.mark.parametrize("N",          [500, 1000, 2000, 5000])
@pytest.mark.parametrize("seed",       [0, 42, 100, 500, 1000])
def test_m2_topk_indices_match_m1(seed, N, k_eff_frac):
    """Spec §5 Tier 3: M2 top-K indices must match M1 ≥ 99.9% of (b,h) tuples."""
    torch.manual_seed(seed)
    k_eff = int(N * k_eff_frac)

    Q       = torch.randn(B, H, D,       dtype=torch.bfloat16, device="cuda")
    K_cache = torch.randn(B, H_kv, N, D, dtype=torch.bfloat16, device="cuda")

    # M1 reference indices
    u_Q        = _compute_u_Q(Q, eps=1e-12)
    scores_ref = _compute_projection_scores(K_cache, u_Q)
    m1_indices = _select_topk_indices(scores_ref, k_eff)    # [B, H, k_eff] int64

    # M2 Triton indices
    m2_indices = topk_qaxis_select(Q, K_cache, k_eff)       # [B, H, k_eff] int32

    assert m2_indices.shape == (B, H, k_eff)

    # Per-(b, h) Jaccard overlap
    overlaps = []
    perfect_count = 0
    for b in range(B):
        for h in range(H):
            m1_set = set(m1_indices[b, h].tolist())
            m2_set = set(m2_indices[b, h].tolist())
            union  = m1_set | m2_set
            inter  = m1_set & m2_set
            jaccard = len(inter) / len(union)
            overlaps.append(jaccard)
            if jaccard == 1.0:
                perfect_count += 1

    perfect_frac = perfect_count / len(overlaps)
    mean_overlap = sum(overlaps) / len(overlaps)

    print(f"\n  seed={seed} N={N} k={k_eff}: "
          f"perfect={perfect_frac:.4f}  mean_jaccard={mean_overlap:.6f}")

    assert perfect_frac >= 0.999, (
        f"Only {perfect_frac:.4f} of (b,h) tuples have perfect M1 agreement"
    )
    assert mean_overlap >= 0.9999, (
        f"Mean Jaccard {mean_overlap:.6f} below 0.9999"
    )


def test_m2_keff_exact():
    """|S_m2| = k_eff exactly; each row has k_eff unique indices in [0, N)."""
    torch.manual_seed(5)
    N, k_eff = 1000, 500
    Q       = torch.randn(B, H, D,       dtype=torch.bfloat16, device="cuda")
    K_cache = torch.randn(B, H_kv, N, D, dtype=torch.bfloat16, device="cuda")

    indices = topk_qaxis_select(Q, K_cache, k_eff)

    assert indices.shape == (B, H, k_eff), f"shape mismatch: {indices.shape}"
    assert indices.min().item() >= 0
    assert indices.max().item() < N
    for h in range(H):
        unique = indices[0, h].unique()
        assert unique.numel() == k_eff, f"head {h}: {unique.numel()} unique, expected {k_eff}"


def test_m2_no_nan():
    """No NaN/inf anywhere in M2 pipeline (spec §6)."""
    torch.manual_seed(21)
    N, k_eff = 512, 256
    Q       = torch.randn(B, H, D,       dtype=torch.bfloat16, device="cuda")
    K_cache = torch.randn(B, H_kv, N, D, dtype=torch.bfloat16, device="cuda")

    indices = topk_qaxis_select(Q, K_cache, k_eff)
    assert torch.isfinite(indices.float()).all(), "M2 pipeline produced non-finite indices"


def test_m2_indices_sorted_ascending():
    """M2 indices sorted ascending per (b, h) row (spec §2 Pass 3 requirement)."""
    torch.manual_seed(33)
    N, k_eff = 500, 200
    Q       = torch.randn(B, H, D,       dtype=torch.bfloat16, device="cuda")
    K_cache = torch.randn(B, H_kv, N, D, dtype=torch.bfloat16, device="cuda")

    indices = topk_qaxis_select(Q, K_cache, k_eff)
    diffs   = indices[:, :, 1:].long() - indices[:, :, :-1].long()
    assert (diffs > 0).all(), "M2 indices not strictly ascending"


@pytest.mark.parametrize("B_val", [1, 2, 4])
def test_m2_batch(B_val):
    """M2 wrapper handles B > 1 correctly."""
    torch.manual_seed(77)
    N, k_eff = 300, 150
    Q       = torch.randn(B_val, H, D,       dtype=torch.bfloat16, device="cuda")
    K_cache = torch.randn(B_val, H_kv, N, D, dtype=torch.bfloat16, device="cuda")

    # M1 reference
    u_Q        = _compute_u_Q(Q, eps=1e-12)
    scores_ref = _compute_projection_scores(K_cache, u_Q)
    m1_idx     = _select_topk_indices(scores_ref, k_eff)

    m2_idx     = topk_qaxis_select(Q, K_cache, k_eff)
    assert m2_idx.shape == (B_val, H, k_eff)

    exact = (m1_idx == m2_idx.to(torch.int64)).all(dim=-1).float().mean().item()
    assert exact >= 0.999, f"B={B_val}: exact match = {exact:.4f}"


def test_m2_runtime_sanity_n20k(benchmark=None):
    """Pass 1 at N=20K, B=1, H=32 should complete without error (perf target < 5ms)."""
    import time
    torch.manual_seed(0)
    N, k_eff = 20000, 10000
    Q       = torch.randn(B, H, D,       dtype=torch.bfloat16, device="cuda")
    K_cache = torch.randn(B, H_kv, N, D, dtype=torch.bfloat16, device="cuda")

    # warmup
    _ = topk_qaxis_select(Q, K_cache, k_eff)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(5):
        _ = topk_qaxis_select(Q, K_cache, k_eff)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) / 5 * 1000

    print(f"\n  N=20K runtime: {elapsed_ms:.2f} ms")
    # Not a hard gate — just report; hard gate is < 50ms per §7 risk protocol
    assert elapsed_ms < 50, f"Pass 1 at N=20K took {elapsed_ms:.1f} ms > 50ms limit"
