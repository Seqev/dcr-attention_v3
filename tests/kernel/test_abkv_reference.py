"""
ABKV reference implementation unit tests.

Tests:
  T01  cache shape correctness
  T02  sort order invariant (axis_scores descending)
  T03  sort_perm is a valid permutation
  T04  K/V gathering correctness (sorted == gathered original)
  T05  axis_vec normalization
  T06  axis_vec custom injection
  T07  output dtype is bf16
  T08  output shape [B, H, D]
  T09  no NaN/inf in output
  T10  quality gate c=0.8: cos_sim >= 0.95
  T11  quality gate c=0.5: cos_sim >= 0.90
  T12  quality gate c=0.1: cos_sim >= 0.85
  T13  c=1.0 (full coverage) matches SDPA to within fp rounding
  T14  GQA (H_kv < H) shape and no-crash
  T15  scan_budget reranking improves quality vs prefix-only at c=0.1
  T16  num_bins_needed helper
  T17  bin_score_boundaries descending across bins
  T18  determinism: same inputs produce same output
  T19  custom axis_vec overrides computed axis
  T20  degenerate N=1 with k_eff=1
"""

from __future__ import annotations

import math

import pytest
import torch

from dcr_attention.kernel.abkv import (
    ABKVCache,
    abkv_attention_reference,
    abkv_attention_with_signature,
    build_abkv_cache,
    cosine_similarity_output,
    sdpa_reference,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

torch.manual_seed(0)

B, H, H_kv, N, D = 2, 8, 2, 256, 64  # GQA: 4 query heads per KV head
K_EFF_HIGH = int(0.8 * N)             # c=0.8
K_EFF_MED  = int(0.5 * N)             # c=0.5
K_EFF_LOW  = int(0.1 * N)             # c=0.1


def _rand_bf16(*shape) -> torch.Tensor:
    return torch.randn(*shape, dtype=torch.float32).to(torch.bfloat16)


def _make_inputs(b=B, h=H, h_kv=H_kv, n=N, d=D):
    Q      = _rand_bf16(b, h, d)
    K_full = _rand_bf16(b, h_kv, n, d)
    V_full = _rand_bf16(b, h_kv, n, d)
    return Q, K_full, V_full


def _make_structured_inputs(
    b=B, h=H, h_kv=H_kv, n=N, d=D,
    important_frac: float = 0.15,
    amplitude: float = 6.0,
    seed: int = 7,
) -> tuple:
    """
    Structured KV data with a dominant direction that Q is correlated with.

    A fraction `important_frac` of keys are generated along a shared axis
    vector `v` with amplitude `amplitude`.  The rest are random noise.
    Q is aligned with that same axis.  This models the locality structure of
    real transformer KV caches (subject tokens, boundary tokens, etc.).

    Pre-sorting K by the axis (mean-K direction) puts the important keys
    first, so ABKV at low coverage still captures the high-attention keys
    and achieves the quality gates.
    """
    rng = torch.Generator().manual_seed(seed)
    n_imp = max(1, int(n * important_frac))
    n_per_kv = h // h_kv

    # Dominant direction per KV head
    v = torch.randn(h_kv, d, generator=rng)
    v = v / v.norm(dim=-1, keepdim=True)

    # Important keys: strong signal along v + small noise
    K_imp  = v[None, :, None, :].expand(b, h_kv, n_imp, d) * amplitude
    K_imp  = K_imp + torch.randn(b, h_kv, n_imp, d, generator=rng) * 0.3
    K_rand = torch.randn(b, h_kv, n - n_imp, d, generator=rng)
    K = torch.cat([K_imp, K_rand], dim=2).to(torch.bfloat16)
    V = torch.randn(b, h_kv, n, d, generator=rng).to(torch.bfloat16)

    # Q: each head aligned with its KV group's v direction
    q_parts = []
    for hk in range(h_kv):
        qp = v[hk:hk+1, :].expand(b, n_per_kv, d) * amplitude
        qp = qp + torch.randn(b, n_per_kv, d, generator=rng) * 0.3
        q_parts.append(qp)
    Q = torch.cat(q_parts, dim=1).to(torch.bfloat16)

    return Q, K, V


@pytest.fixture(scope="module")
def default_inputs():
    return _make_inputs()


@pytest.fixture(scope="module")
def default_cache(default_inputs):
    _, K_full, V_full = default_inputs
    return build_abkv_cache(K_full, V_full, num_bins=32)


@pytest.fixture(scope="module")
def structured_inputs():
    """KV cache with axis-aligned structure; used for quality-gate tests."""
    return _make_structured_inputs()


@pytest.fixture(scope="module")
def structured_cache(structured_inputs):
    _, K_full, V_full = structured_inputs
    return build_abkv_cache(K_full, V_full, num_bins=32)


# ===========================================================================
# T01 — cache shape correctness
# ===========================================================================

def test_t01_cache_shapes(default_cache):
    c = default_cache
    assert c.K_sorted.shape    == (B, H_kv, N, D)
    assert c.V_sorted.shape    == (B, H_kv, N, D)
    assert c.axis_scores.shape == (B, H_kv, N)
    assert c.axis_vec.shape    == (H_kv, D)
    assert c.sort_perm.shape   == (B, H_kv, N)
    assert c.N == N
    assert c.H_kv == H_kv
    assert c.D == D


# ===========================================================================
# T02 — axis_scores are sorted descending
# ===========================================================================

def test_t02_scores_sorted_descending(default_cache):
    scores = default_cache.axis_scores  # [B, H_kv, N]
    diffs  = scores[:, :, 1:] - scores[:, :, :-1]
    assert (diffs <= 0).all(), "axis_scores must be non-increasing (descending)"


# ===========================================================================
# T03 — sort_perm is a valid permutation of 0..N-1
# ===========================================================================

def test_t03_sort_perm_valid_permutation(default_cache):
    perm = default_cache.sort_perm  # [B, H_kv, N]
    for b in range(B):
        for h in range(H_kv):
            idx = perm[b, h].sort().values
            expected = torch.arange(N)
            assert torch.equal(idx, expected), f"sort_perm[{b},{h}] is not a permutation of 0..{N-1}"


# ===========================================================================
# T04 — K_sorted correctly gathers from original K_cache
# ===========================================================================

def test_t04_ksorted_equals_gathered_original(default_inputs, default_cache):
    _, K_orig, _ = default_inputs
    perm  = default_cache.sort_perm                          # [B, H_kv, N]
    perm_exp = perm.unsqueeze(-1).expand(-1, -1, -1, D)
    K_reconstructed = torch.gather(K_orig, dim=2, index=perm_exp)
    assert torch.equal(default_cache.K_sorted, K_reconstructed)


# ===========================================================================
# T05 — axis_vec is unit norm per head
# ===========================================================================

def test_t05_axis_vec_unit_norm(default_cache):
    norms = torch.linalg.norm(default_cache.axis_vec.float(), dim=-1)  # [H_kv]
    assert torch.allclose(norms, torch.ones(H_kv), atol=1e-5), (
        f"axis_vec norms: {norms.tolist()} — must be unit"
    )


# ===========================================================================
# T06 — custom axis_vec is stored and used
# ===========================================================================

def test_t06_custom_axis_vec(default_inputs):
    _, K_full, V_full = default_inputs
    custom_axis = torch.randn(H_kv, D, dtype=torch.float32)
    custom_axis = custom_axis / torch.linalg.norm(custom_axis, dim=-1, keepdim=True)
    cache_custom = build_abkv_cache(K_full, V_full, num_bins=16, axis_vec=custom_axis)
    assert torch.allclose(cache_custom.axis_vec, custom_axis, atol=1e-6)

    # Verify the sort is consistent with the custom axis
    K_f32 = K_full.float()
    scores_expected = torch.einsum("bhnd,hd->bhn", K_f32, custom_axis)
    # After sorting descending, position 0 must have max score
    max_scores = scores_expected.amax(dim=-1)                # [B, H_kv]
    first_scores = cache_custom.axis_scores[:, :, 0]
    assert torch.allclose(first_scores, max_scores, atol=1e-4), (
        "First sorted position must have maximum axis score"
    )


# ===========================================================================
# T07 — output dtype is bf16
# ===========================================================================

def test_t07_output_dtype(default_inputs, default_cache):
    Q, _, _ = default_inputs
    out = abkv_attention_reference(Q, default_cache, k_eff=K_EFF_HIGH)
    assert out.dtype == torch.bfloat16, f"Expected bf16 output, got {out.dtype}"


# ===========================================================================
# T08 — output shape [B, H, D]
# ===========================================================================

def test_t08_output_shape(default_inputs, default_cache):
    Q, _, _ = default_inputs
    out = abkv_attention_reference(Q, default_cache, k_eff=K_EFF_HIGH)
    assert out.shape == (B, H, D), f"Expected ({B},{H},{D}), got {out.shape}"


# ===========================================================================
# T09 — no NaN/inf in output
# ===========================================================================

def test_t09_no_nan_inf(default_inputs, default_cache):
    Q, _, _ = default_inputs
    for k_eff in [K_EFF_LOW, K_EFF_MED, K_EFF_HIGH]:
        out = abkv_attention_reference(Q, default_cache, k_eff=k_eff)
        assert torch.isfinite(out).all(), f"Non-finite output at k_eff={k_eff}"


# ===========================================================================
# T10 — quality gate c=0.8: cos_sim >= 0.95 (structured KV data)
# ===========================================================================

def test_t10_quality_c08(structured_inputs, structured_cache):
    Q, K_full, V_full = structured_inputs
    O_abkv = abkv_attention_reference(Q, structured_cache, k_eff=K_EFF_HIGH)
    O_sdpa = sdpa_reference(Q, K_full, V_full)
    sim = cosine_similarity_output(O_abkv, O_sdpa)
    assert sim >= 0.95, f"cos_sim={sim:.4f} < 0.95 at c=0.8"


# ===========================================================================
# T11 — quality gate c=0.5: cos_sim >= 0.90 (structured KV data)
# ===========================================================================

def test_t11_quality_c05(structured_inputs, structured_cache):
    Q, K_full, V_full = structured_inputs
    O_abkv = abkv_attention_reference(Q, structured_cache, k_eff=K_EFF_MED)
    O_sdpa = sdpa_reference(Q, K_full, V_full)
    sim = cosine_similarity_output(O_abkv, O_sdpa)
    assert sim >= 0.90, f"cos_sim={sim:.4f} < 0.90 at c=0.5"


# ===========================================================================
# T12 — quality gate c=0.1: cos_sim >= 0.85 (structured KV data)
# ===========================================================================

def test_t12_quality_c01(structured_inputs, structured_cache):
    Q, K_full, V_full = structured_inputs
    O_abkv = abkv_attention_reference(Q, structured_cache, k_eff=K_EFF_LOW)
    O_sdpa = sdpa_reference(Q, K_full, V_full)
    sim = cosine_similarity_output(O_abkv, O_sdpa)
    assert sim >= 0.85, f"cos_sim={sim:.4f} < 0.85 at c=0.1"


# ===========================================================================
# T13 — c=1.0 (full coverage) matches SDPA within bf16 rounding
# ===========================================================================

def test_t13_full_coverage_matches_sdpa(structured_inputs, structured_cache):
    Q, K_full, V_full = structured_inputs
    O_abkv = abkv_attention_reference(Q, structured_cache, k_eff=N)
    O_sdpa = sdpa_reference(Q, K_full, V_full)
    sim = cosine_similarity_output(O_abkv, O_sdpa)
    # At full coverage the only difference is key ordering in online softmax
    # (mathematically equivalent); bf16 rounding may cause small gap
    assert sim >= 0.999, f"cos_sim={sim:.6f} at c=1.0 — should be ~1.0"


# ===========================================================================
# T14 — GQA (H_kv < H) works and output shape is correct
# ===========================================================================

def test_t14_gqa_multiple_queries_per_kv():
    # H=16 query heads, H_kv=2 KV heads → 8 per group
    Q_gqa = _rand_bf16(1, 16, 64)
    K_gqa = _rand_bf16(1, 2, 128, 64)
    V_gqa = _rand_bf16(1, 2, 128, 64)
    cache = build_abkv_cache(K_gqa, V_gqa, num_bins=8)
    out   = abkv_attention_reference(Q_gqa, cache, k_eff=32)
    assert out.shape == (1, 16, 64)
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out).all()


# ===========================================================================
# T15 — scan_budget=N recovers true-top-k quality on random (unstructured) data
# ===========================================================================

def test_t15_scan_budget_full_scan(default_inputs, default_cache):
    """
    With scan_budget=N (full axis-ordered scan + Q·K rerank), ABKV selects
    the true top-k_eff keys by Q·K score.  On random data the prefix-only
    mode picks an axis-ordered prefix with no Q·K correlation; reranking the
    full N recovers much better quality.
    """
    Q, K_full, V_full = default_inputs
    O_sdpa    = sdpa_reference(Q, K_full, V_full)
    O_prefix  = abkv_attention_reference(Q, default_cache, k_eff=K_EFF_LOW)
    O_fullscan = abkv_attention_reference(
        Q, default_cache, k_eff=K_EFF_LOW, scan_budget=N
    )
    sim_prefix   = cosine_similarity_output(O_prefix, O_sdpa)
    sim_fullscan = cosine_similarity_output(O_fullscan, O_sdpa)
    # scan_budget=N gives true top-k, must significantly beat random prefix
    assert sim_fullscan > sim_prefix + 0.05, (
        f"scan_budget=N should beat prefix on random data: "
        f"prefix={sim_prefix:.4f} fullscan={sim_fullscan:.4f}"
    )


# ===========================================================================
# T16 — num_bins_needed helper
# ===========================================================================

def test_t16_num_bins_needed(default_cache):
    c = default_cache
    assert c.num_bins_needed(c.bin_size) == 1           # one full bin
    assert c.num_bins_needed(c.N) == c.num_bins         # all bins
    assert c.num_bins_needed(1)   == 1                  # single key → first bin


# ===========================================================================
# T17 — bin_score_boundaries are non-increasing
# ===========================================================================

def test_t17_bin_boundaries_descending(default_cache):
    bounds = default_cache.bin_score_boundaries()       # [B, H_kv, num_bins]
    diffs  = bounds[:, :, 1:] - bounds[:, :, :-1]
    assert (diffs <= 1e-5).all(), "bin boundaries must be non-increasing"


# ===========================================================================
# T18 — determinism: same inputs → same output
# ===========================================================================

def test_t18_determinism(default_inputs, default_cache):
    Q, _, _ = default_inputs
    out1 = abkv_attention_reference(Q, default_cache, k_eff=K_EFF_MED)
    out2 = abkv_attention_reference(Q, default_cache, k_eff=K_EFF_MED)
    assert torch.equal(out1, out2), "Same inputs must produce identical outputs"


# ===========================================================================
# T19 — custom axis_vec produces different sort than computed axis
# ===========================================================================

def test_t19_custom_axis_changes_sort(default_inputs):
    _, K_full, V_full = default_inputs
    cache_auto   = build_abkv_cache(K_full, V_full, num_bins=32)
    # Reversed axis should give reversed sort order
    reversed_axis = -cache_auto.axis_vec
    cache_rev    = build_abkv_cache(K_full, V_full, num_bins=32, axis_vec=reversed_axis)
    # First positions should differ
    same = torch.equal(cache_auto.sort_perm, cache_rev.sort_perm)
    assert not same, "Reversed axis must produce a different sort order"


# ===========================================================================
# T20 — degenerate N=1 with k_eff=1
# ===========================================================================

def test_t20_degenerate_n1():
    Q = _rand_bf16(1, 4, 32)
    K = _rand_bf16(1, 1, 1, 32)
    V = _rand_bf16(1, 1, 1, 32)
    cache = build_abkv_cache(K, V, num_bins=1)
    out   = abkv_attention_reference(Q, cache, k_eff=1)
    assert out.shape == (1, 4, 32)
    assert torch.isfinite(out).all()
