"""
Measurement 3: Brute-force top-K oracle comparison (CPU only).

Theorem 1 (Paper 1 §3.2): ranking by K_i · u_Q is identical to ranking by
Q · K_i.  Therefore M1's top-K (via _select_topk_indices on Q-axis
projections) must match brute-force top-K (via full Q·K^T) with ≥ 99%
Jaccard overlap.

If overlap < 90% on many tests → M1 implementation diverges from theory.
If 90-99% → bf16 tie effects at boundary (investigate distribution).
If ≥ 99% consistently → Theorem 1 confirmed empirically.

All on CPU, no model loading.  45 parametrized cases.
"""
from __future__ import annotations

import torch
import pytest

from dcr_attention.kernel.qaxis_topk_reference import (
    _compute_u_Q,
    _compute_projection_scores,
    _select_topk_indices,
)

# Llama-3.2-1B dimensions (fixed for all tests)
B, H, H_kv, D = 1, 32, 8, 64


def _run_overlap(seed: int, N: int, k_eff_frac: float) -> dict:
    """Compute Jaccard overlap between M1 and brute-force top-K."""
    torch.manual_seed(seed)
    k_eff = max(1, int(N * k_eff_frac))

    Q = torch.randn(B, H, D, dtype=torch.bfloat16)
    K_cache = torch.randn(B, H_kv, N, D, dtype=torch.bfloat16)

    # ── M1 path: u_Q projection ──────────────────────────────────────────────
    u_Q = _compute_u_Q(Q, eps=1e-12)                        # [B, H, D] fp32
    scores_m1 = _compute_projection_scores(K_cache, u_Q)    # [B, H, N] bf16
    m1_idx = _select_topk_indices(scores_m1, k_eff)         # [B, H, k_eff]

    # ── Oracle path: full Q·K^T ───────────────────────────────────────────────
    n_per_kv = H // H_kv
    K_exp = K_cache.repeat_interleave(n_per_kv, dim=1)      # [B, H, N, D]
    Q_f32 = Q.float()
    K_f32 = K_exp.float()
    full_scores = torch.einsum("bhd,bhnd->bhn", Q_f32, K_f32)  # [B, H, N] fp32

    # bf16 round-trip to match M1's precision regime
    full_scores_rt = full_scores.to(torch.bfloat16).float()
    oracle_raw = full_scores_rt.argsort(dim=-1, descending=True, stable=True)
    oracle_idx, _ = oracle_raw[:, :, :k_eff].sort(dim=-1)   # [B, H, k_eff]

    # ── Jaccard overlap per (b, h) ────────────────────────────────────────────
    overlaps = []
    for b in range(B):
        for h in range(H):
            m1_set = set(m1_idx[b, h].tolist())
            ora_set = set(oracle_idx[b, h].tolist())
            jaccard = len(m1_set & ora_set) / len(m1_set | ora_set)
            overlaps.append(jaccard)

    return {
        "mean": sum(overlaps) / len(overlaps),
        "min": min(overlaps),
        "max": max(overlaps),
        "n_heads": len(overlaps),
        "n_below_99": sum(1 for o in overlaps if o < 0.99),
        "n_below_90": sum(1 for o in overlaps if o < 0.90),
    }


@pytest.mark.parametrize("seed", [0, 1, 2, 42, 100])
@pytest.mark.parametrize("N", [500, 1000, 2000])
@pytest.mark.parametrize("k_eff_frac", [0.3, 0.5, 0.8])
def test_m1_oracle_overlap(seed, N, k_eff_frac):
    """M1 top-K must overlap brute-force oracle with mean Jaccard ≥ 0.99."""
    r = _run_overlap(seed, N, k_eff_frac)
    k_eff = max(1, int(N * k_eff_frac))

    print(
        f"\n  seed={seed} N={N} k_eff={k_eff} ({k_eff_frac:.0%}): "
        f"mean={r['mean']:.4f}  min={r['min']:.4f}  "
        f"below_99={r['n_below_99']}/{r['n_heads']}  "
        f"below_90={r['n_below_90']}/{r['n_heads']}"
    )

    assert r["mean"] >= 0.99, (
        f"M1 top-K diverges from oracle: mean Jaccard={r['mean']:.4f} < 0.99 "
        f"at seed={seed}, N={N}, k_eff_frac={k_eff_frac}\n"
        f"  min={r['min']:.4f}, below_99={r['n_below_99']}/{r['n_heads']}"
    )
