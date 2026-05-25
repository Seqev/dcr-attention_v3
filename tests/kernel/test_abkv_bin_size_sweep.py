"""
ABKV bin-size sweep.

Measures how cos_sim(ABKV output, SDPA output) varies with:
  - num_bins  ∈ {8, 16, 32, 64, 128}
  - coverage c ∈ {0.10, 0.20, 0.30, 0.50, 0.80}

Structured KV data is used (important_frac=0.15, amp=6.0) so the axis
captures meaningful structure.  Output is written to /tmp/abkv_day1/bin_size_sweep.json.

Run:
  cd /home/user/dcr-attention
  PYTHONPATH=/home/user/dcr-attention python tests/kernel/test_abkv_bin_size_sweep.py \
    2>&1 | tee /tmp/abkv_day1/bin_size_sweep.log

Or via pytest (marks as integration, skipped in fast test runs):
  pytest tests/kernel/test_abkv_bin_size_sweep.py -v
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
import torch

from dcr_attention.kernel.abkv import (
    abkv_attention_reference,
    build_abkv_cache,
    cosine_similarity_output,
    sdpa_reference,
)

OUT_DIR = Path("/tmp/abkv_day1")

# Sweep parameters
NUM_BINS_LIST  = [8, 16, 32, 64, 128]
COVERAGE_LIST  = [0.10, 0.20, 0.30, 0.50, 0.80]

# Fixture dimensions (larger N for realistic sweep)
B_SW, H_SW, H_KV_SW, N_SW, D_SW = 2, 8, 2, 512, 64
IMP_FRAC, AMP = 0.15, 6.0


def _make_sweep_inputs():
    torch.manual_seed(42)
    rng = torch.Generator().manual_seed(42)
    n_imp = max(1, int(N_SW * IMP_FRAC))
    n_per_kv = H_SW // H_KV_SW

    v = torch.randn(H_KV_SW, D_SW, generator=rng)
    v = v / v.norm(dim=-1, keepdim=True)

    K_imp  = v[None, :, None, :].expand(B_SW, H_KV_SW, n_imp, D_SW) * AMP
    K_imp  = K_imp + torch.randn(B_SW, H_KV_SW, n_imp, D_SW, generator=rng) * 0.3
    K_rand = torch.randn(B_SW, H_KV_SW, N_SW - n_imp, D_SW, generator=rng)
    K = torch.cat([K_imp, K_rand], dim=2).to(torch.bfloat16)
    V = torch.randn(B_SW, H_KV_SW, N_SW, D_SW, generator=rng).to(torch.bfloat16)

    q_parts = []
    for hk in range(H_KV_SW):
        qp = v[hk:hk+1, :].expand(B_SW, n_per_kv, D_SW) * AMP
        qp = qp + torch.randn(B_SW, n_per_kv, D_SW, generator=rng) * 0.3
        q_parts.append(qp)
    Q = torch.cat(q_parts, dim=1).to(torch.bfloat16)

    return Q, K, V, n_imp


# ---------------------------------------------------------------------------
# pytest entry point — runs a quick smoke test
# ---------------------------------------------------------------------------

def test_bin_size_sweep_runs():
    """Verify the sweep runs without errors and all cos_sims are > 0."""
    Q, K, V, _ = _make_sweep_inputs()
    O_sdpa = sdpa_reference(Q, K, V)

    for num_bins in [8, 32]:
        cache = build_abkv_cache(K, V, num_bins=num_bins)
        for c in [0.10, 0.50]:
            k_eff = max(1, int(c * N_SW))
            O_abkv = abkv_attention_reference(Q, cache, k_eff=k_eff)
            sim = cosine_similarity_output(O_abkv, O_sdpa)
            assert sim > 0.0, f"cos_sim={sim} at num_bins={num_bins} c={c}"


# ---------------------------------------------------------------------------
# Standalone main — full sweep + JSON output
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    Q, K, V, n_imp = _make_sweep_inputs()
    O_sdpa = sdpa_reference(Q, K, V)

    print(f"ABKV Day 1 — bin-size sweep")
    print(f"B={B_SW} H={H_SW} H_kv={H_KV_SW} N={N_SW} D={D_SW}")
    print(f"important_frac={IMP_FRAC} n_imp={n_imp}  amplitude={AMP}")
    print()

    # Header
    coverage_labels = ["  ".join(f"c={c:.2f}" for c in COVERAGE_LIST)]
    print(f"{'num_bins':>10}  " + "  ".join(f"c={c:.2f}" for c in COVERAGE_LIST))
    print("-" * (12 + 9 * len(COVERAGE_LIST)))

    results = []

    for num_bins in NUM_BINS_LIST:
        t0 = time.monotonic()
        cache = build_abkv_cache(K, V, num_bins=num_bins)
        build_ms = (time.monotonic() - t0) * 1000

        row = {"num_bins": num_bins, "build_ms": round(build_ms, 2), "coverages": []}
        row_sims = []

        for c in COVERAGE_LIST:
            k_eff = max(1, int(c * N_SW))
            t1 = time.monotonic()
            O_abkv = abkv_attention_reference(Q, cache, k_eff=k_eff)
            attn_ms = (time.monotonic() - t1) * 1000
            sim = cosine_similarity_output(O_abkv, O_sdpa)

            # How many of the n_imp important keys are in the prefix?
            covered = int((cache.sort_perm[0, 0, :k_eff] < n_imp).sum())

            row["coverages"].append({
                "c": c,
                "k_eff": k_eff,
                "cos_sim": round(sim, 6),
                "important_covered": covered,
                "important_total": n_imp,
                "attn_ms": round(attn_ms, 2),
            })
            row_sims.append(f"{sim:.4f}")

        print(f"{num_bins:>10}  " + "  ".join(f"{s:>7}" for s in row_sims)
              + f"  (build {build_ms:.1f}ms)")
        results.append(row)

    out_path = OUT_DIR / "bin_size_sweep.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "config": {
                    "B": B_SW, "H": H_SW, "H_kv": H_KV_SW, "N": N_SW, "D": D_SW,
                    "important_frac": IMP_FRAC, "amplitude": AMP, "n_imp": n_imp,
                },
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nSaved -> {out_path}")

    # Quality gate summary
    print("\nQuality gate check (cos_sim >= gate):")
    gates = {0.10: 0.85, 0.50: 0.90, 0.80: 0.95}
    all_pass = True
    for row in results:
        for entry in row["coverages"]:
            c = entry["c"]
            if c in gates:
                ok = entry["cos_sim"] >= gates[c]
                if not ok:
                    all_pass = False
                    print(
                        f"  FAIL  num_bins={row['num_bins']} c={c:.2f} "
                        f"cos_sim={entry['cos_sim']:.4f} < {gates[c]:.2f}"
                    )
    if all_pass:
        print("  All gates PASSED")


if __name__ == "__main__":
    main()
