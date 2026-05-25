"""
Phase 2 Task 3 completion: v2.0 hero re-validation, seeds 2/42/100.

Power outage destroyed the original run. This version writes a persistent
JSON checkpoint after EACH seed to /data/raw/ (survives reboot).

Seeds 0 and 1 were preserved from the prior run and copied into v2_hero/
under the new naming convention. This script re-runs only 2, 42, 100 by
default; the combine step auto-detects all v2_hero_seed*.json files
present in the output directory.
"""
from __future__ import annotations

import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from pathlib import Path

from dcr_attention.models.llama.config import DCRLlamaConfig
from tests.kernel.test_m1_acceptance import (
    load_model_and_data,
    run_trace,
    standardized_json_output,
)


N = 20000
C_FLOOR = 0.5
SEEDS = [2, 42, 100]  # Seeds 0, 1 already saved in v2_hero/ from prior run.

OUT_DIR = Path("/home/user/dcr-attention/data/raw/phase_2_repro/v2_hero")
V20_PUBLISHED = 0.308   # v2.0 hero published mean (pp)


def measure_one_seed(seed: int) -> dict:
    """Run baseline + M1 reference for one seed; return standardized dict."""
    print(f"\n=== v2.0 hero seed={seed} (N={N}, c={C_FLOOR}) ===", flush=True)
    torch.cuda.empty_cache()
    torch.manual_seed(seed)

    model, _tok, ids = load_model_and_data(N=N + 1, seed=seed)

    cfg_sdpa = DCRLlamaConfig(enable_dcr=False)
    ppl_base = run_trace(model, cfg_sdpa, ids, N=N, label=f"base-s{seed}")
    print(f"  SDPA baseline PPL: {ppl_base:.4f}", flush=True)

    cfg_m1 = DCRLlamaConfig(
        enable_dcr=True,
        coverage_floor=C_FLOOR,
        axis_source="q_topk_reference",
        T_dispatch=1,
    )
    ppl_m1 = run_trace(model, cfg_m1, ids, N=N, label=f"hero-s{seed}")
    delta = (ppl_m1 - ppl_base) / ppl_base * 100.0
    print(f"  M1 PPL: {ppl_m1:.4f}  ΔPPL = {delta:+.4f}%", flush=True)

    return standardized_json_output(
        config_id=f"v2_hero_c{int(C_FLOOR*100):03d}_N{N}_seed{seed:03d}",
        N=N,
        c_floor=C_FLOOR,
        ppl_baseline=ppl_base,
        ppl_dcr=ppl_m1,
        delta_ppl_pct=delta,
        seed=seed,
        axis_source="q_topk_reference",
    )


def combine_and_classify():
    """Combine all v2_hero_seed*.json files present in OUT_DIR."""
    all_jsons = sorted(OUT_DIR.glob("v2_hero_seed*.json"))
    deltas, seeds_present = [], []
    for jp in all_jsons:
        d = json.loads(jp.read_text())
        deltas.append(d["delta_ppl_pct"])
        seeds_present.append(d["seed"])

    n = len(deltas)
    if n == 0:
        print("\n[WARN] No seed JSONs found in OUT_DIR. Skipping combine.")
        return

    mean = sum(deltas) / n
    if n > 1:
        std = (sum((d - mean) ** 2 for d in deltas) / (n - 1)) ** 0.5
    else:
        std = 0.0

    print(f"\n=== v2.0 hero {n}-seed summary ===")
    print(f"  seeds: {sorted(seeds_present)}")
    print(f"  mean: {mean:+.4f}%, std: ±{std:.4f} pp")
    print(f"  min: {min(deltas):+.4f}%, max: {max(deltas):+.4f}%")

    if mean <= 0.5:
        classification = "STRICT"
    elif mean <= 1.0:
        classification = "STANDARD"
    elif mean <= 2.0:
        classification = "PERMISSIVE"
    else:
        classification = "UNUSABLE"

    if mean <= 0.5 and std <= 0.2:
        scenario = "v2-A (confirmed; clean)"
    elif 0.5 < mean <= 0.7 and std <= 0.2:
        scenario = "v2-B-mild (disclosure needed)"
    elif mean > 0.7 or std > 0.3:
        scenario = "v2-B-severe (critical inversion)"
    else:
        scenario = "v2-B-other (architect review)"

    drift = mean - V20_PUBLISHED
    if drift < -0.1:
        direction = "OPTIMISTIC (reality better than published)"
    elif drift > 0.1:
        direction = "PESSIMISTIC (reality worse than published)"
    else:
        direction = "consistent with published"

    summary = {
        "config": "v2.0 hero (N=20K, c=0.5, M1 reference)",
        "seeds": sorted(seeds_present),
        "n_seeds": n,
        "deltas_ppl_pct": deltas,
        "mean_delta_pct": mean,
        "std_delta_pct": std,
        "min_delta_pct": min(deltas),
        "max_delta_pct": max(deltas),
        "classification": classification,
        "v2_0_published": V20_PUBLISHED,
        "drift_from_published": drift,
        "drift_direction": direction,
        "scenario": scenario,
    }

    summary_path = OUT_DIR / "v2_hero_5seed_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Classification: {classification}")
    print(f"  Drift from published: {drift:+.4f} pp — {direction}")
    print(f"  Scenario: {scenario}")
    print(f"  Saved: {summary_path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for seed in SEEDS:
        out_path = OUT_DIR / f"v2_hero_seed{seed:03d}_N{N}_c{int(C_FLOOR*100):02d}.json"
        if out_path.exists():
            print(f"\n[SKIP] {out_path.name} already exists.", flush=True)
            continue

        result = measure_one_seed(seed)
        # PERSISTENT checkpoint — write immediately to /data/, not /tmp/.
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  CHECKPOINT saved: {out_path}", flush=True)

    combine_and_classify()


if __name__ == "__main__":
    main()
