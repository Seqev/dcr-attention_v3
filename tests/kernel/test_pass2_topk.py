"""
M2.2 validation — Pass 2 top-K selection vs M1 _select_topk_indices.

Spec §5 Tier 3: ≥ 99.9% index overlap with M1 reference.
Both use stable=True argsort so overlap should be 100% on non-degenerate inputs.
"""

from __future__ import annotations

import pytest
import torch

from dcr_attention.kernel.triton.topk_axis_pass2 import topk_pass2_select
from dcr_attention.kernel.qaxis_topk_reference import _select_topk_indices

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)

B, H = 1, 32


@pytest.mark.parametrize("k_eff_frac", [0.3, 0.5, 0.8])
@pytest.mark.parametrize("N",          [500, 1000, 2000])
@pytest.mark.parametrize("seed",       [0, 42, 100])
def test_pass2_matches_m1_select(seed, N, k_eff_frac):
    """Pass 2 indices must match M1 _select_topk_indices ≥ 99.9% per (b,h)."""
    torch.manual_seed(seed)
    k_eff = int(N * k_eff_frac)

    scores = torch.randn(B, H, N, dtype=torch.bfloat16, device="cuda")

    m1_idx   = _select_topk_indices(scores, k_eff)           # int64, sorted asc
    pass2_idx = topk_pass2_select(scores, k_eff)              # int32, sorted asc

    assert pass2_idx.shape == (B, H, k_eff)
    assert pass2_idx.dtype == torch.int32

    # Overlap per (b, h)
    p2_i64 = pass2_idx.to(torch.int64)
    exact_matches = (m1_idx == p2_i64).all(dim=-1).float().mean().item()

    print(f"\n  seed={seed} N={N} k_eff_frac={k_eff_frac}: exact_match={exact_matches:.4f}")
    assert exact_matches >= 0.999, (
        f"Pass 2 vs M1 exact match = {exact_matches:.4f} < 0.999"
    )


def test_pass2_output_dtype_and_shape():
    """Output is [B, H, k_eff] int32."""
    scores = torch.randn(B, H, 1000, dtype=torch.bfloat16, device="cuda")
    idx    = topk_pass2_select(scores, k_eff=300)
    assert idx.shape == (B, H, 300)
    assert idx.dtype == torch.int32


def test_pass2_indices_in_range():
    """All returned indices must be in [0, N)."""
    N, k_eff = 800, 400
    scores = torch.randn(B, H, N, dtype=torch.bfloat16, device="cuda")
    idx    = topk_pass2_select(scores, k_eff)
    assert idx.min().item() >= 0
    assert idx.max().item() < N


def test_pass2_indices_sorted_ascending():
    """Indices must be sorted ascending within each (b, h) row (spec §2)."""
    N, k_eff = 500, 250
    scores = torch.randn(B, H, N, dtype=torch.bfloat16, device="cuda")
    idx    = topk_pass2_select(scores, k_eff)
    diffs  = idx[:, :, 1:] - idx[:, :, :-1]
    assert (diffs > 0).all(), "Pass 2 indices not strictly ascending"


def test_pass2_unique_per_head():
    """Each (b, h) row must have k_eff unique indices."""
    N, k_eff = 500, 250
    scores = torch.randn(B, H, N, dtype=torch.bfloat16, device="cuda")
    idx    = topk_pass2_select(scores, k_eff)
    for h in range(H):
        assert idx[0, h].unique().numel() == k_eff, f"head {h}: duplicate indices"
