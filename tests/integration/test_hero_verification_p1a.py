"""
P1a multi-seed validation at N=32K, c=0.15.

Validates whether single-seed P1a (+0.561%) was lucky or representative.
Additional seeds: 1, 2, 42, 100. Original seed=0 already measured (+0.561%).

Runtime: ~4 hours per seed × 4 seeds = ~16 hours.

Run:
  cd /home/user/dcr-attention
  source /home/user/dcr-venv/bin/activate
  PYTHONPATH=/home/user/dcr-attention python tests/integration/test_hero_verification_p1a.py \
    2>&1 | tee /tmp/hero_verification/p1a_multiseed.log
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import torch

OUT_DIR = Path("/tmp/hero_verification")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N = 32000
C = 0.15
SEEDS_TO_RUN = [1, 2, 42, 100]
SEED0_DELTA = 0.561  # Original P1a result (seed=0, measured 2026-05-11)


def _print(msg: str) -> None:
    print(msg, flush=True)


def main() -> None:
    from dcr_attention.models.llama.config import DCRLlamaConfig
    from tests.kernel.test_m1_acceptance import load_model_and_data, run_trace

    _print(f"Hero Verification — P1a multi-seed: N={N}, c={C}")
    _print(f"Seeds to run: {SEEDS_TO_RUN}  (seed=0 already done: +{SEED0_DELTA}%)")
    _print(f"Output: {OUT_DIR}")
    _print("=" * 60)

    cfg_sdpa = DCRLlamaConfig(enable_dcr=False)
    cfg_m1 = DCRLlamaConfig(
        axis_source="q_topk_reference",
        coverage_floor=C,
        enable_dcr=True,
        enable_adaptive_widening=False,
    )

    per_seed_results = []

    for seed in SEEDS_TO_RUN:
        out_path = OUT_DIR / f"p1a_seed{seed:03d}_N{N}_c{int(C * 100):02d}.json"
        if out_path.exists():
            _print(f"\n[seed={seed}] Already exists: {out_path} — loading.")
            r = json.loads(out_path.read_text())
            per_seed_results.append(r)
            continue

        _print(f"\n[seed={seed}] Loading model + data (offset={seed * 1000})…")
        t0 = time.monotonic()
        model, tok, ids = load_model_and_data(N=N + 1, seed=seed)
        _print(f"[seed={seed}] Model loaded in {time.monotonic() - t0:.1f}s")

        torch.cuda.empty_cache()

        # SDPA baseline
        _print(f"[seed={seed}] Running SDPA baseline…")
        t_base = time.monotonic()
        ppl_base = run_trace(model, cfg_sdpa, ids, N=N, label=f"sdpa_baseline_seed{seed}")
        base_min = (time.monotonic() - t_base) / 60
        _print(f"[seed={seed}] ppl_baseline = {ppl_base:.4f}  ({base_min:.1f} min)")

        torch.cuda.empty_cache()

        # M1 at c=0.15
        _print(f"[seed={seed}] Running M1 c={C}…")
        t_m1 = time.monotonic()
        ppl_m1 = run_trace(model, cfg_m1, ids, N=N, label=f"M1_c{int(C*100):02d}_seed{seed}")
        m1_min = (time.monotonic() - t_m1) / 60
        delta = (ppl_m1 - ppl_base) / ppl_base * 100.0
        _print(f"[seed={seed}] ppl_m1 = {ppl_m1:.4f}  ({m1_min:.1f} min)")
        _print(f"[seed={seed}] ΔPPL = {delta:+.4f}%")

        result = {
            "seed": seed,
            "N": N,
            "c_floor": C,
            "ppl_baseline_sdpa": round(ppl_base, 6),
            "ppl_m1_topk": round(ppl_m1, 6),
            "delta_ppl_pct": round(delta, 6),
            "axis_source": "q_topk_reference",
            "runtime_baseline_min": round(base_min, 2),
            "runtime_m1_min": round(m1_min, 2),
            "gpu": torch.cuda.get_device_name(0),
        }
        out_path.write_text(json.dumps(result, indent=2))
        _print(f"[seed={seed}] Saved: {out_path}")
        per_seed_results.append(result)

        # Free model memory before next seed
        del model
        torch.cuda.empty_cache()

    # Combine with seed=0
    all_deltas = [SEED0_DELTA] + [r["delta_ppl_pct"] for r in per_seed_results]
    all_seeds = [0] + [r["seed"] for r in per_seed_results]
    n = len(all_deltas)
    mean = sum(all_deltas) / n
    var = sum((d - mean) ** 2 for d in all_deltas) / (n - 1)
    std = math.sqrt(var)
    min_d = min(all_deltas)
    max_d = max(all_deltas)

    _print("\n" + "=" * 60)
    _print("P1a 5-SEED SUMMARY")
    _print(f"  seeds:   {all_seeds}")
    _print(f"  deltas:  {[f'{d:+.4f}%' for d in all_deltas]}")
    _print(f"  mean:    {mean:+.4f}%")
    _print(f"  std:     ±{std:.4f}%")
    _print(f"  min:     {min_d:+.4f}%")
    _print(f"  max:     {max_d:+.4f}%")
    _print(f"  range:   {max_d - min_d:.4f} pp")

    # Classification
    if mean <= 0.5:
        classification = "STRICT"
    elif mean <= 1.0:
        classification = "STANDARD"
    elif mean <= 2.0:
        classification = "PERMISSIVE"
    else:
        classification = "UNUSABLE"

    # Verdict
    if mean <= 0.7 and std <= 0.15:
        verdict = "HERO_CONFIRMED"
    elif mean <= 1.0 and std <= 0.20:
        verdict = "HERO_MARGINAL"
    elif mean <= 1.0:
        verdict = "HERO_HIGH_VARIANCE"
    else:
        verdict = "HERO_COLLAPSED"

    _print(f"  classification: {classification}")
    _print(f"  verdict:        {verdict}")
    _print(f"  drift from single-seed: {mean - SEED0_DELTA:+.4f} pp")

    summary = {
        "config": f"P1a N={N} c={C} M1_q_topk_reference",
        "seeds_tested": all_seeds,
        "deltas_ppl_pct": [round(d, 6) for d in all_deltas],
        "n_seeds": n,
        "mean_delta_pct": round(mean, 6),
        "std_delta_pct": round(std, 6),
        "min_delta_pct": round(min_d, 6),
        "max_delta_pct": round(max_d, 6),
        "single_seed_ref_pct": SEED0_DELTA,
        "drift_from_single_seed_pp": round(mean - SEED0_DELTA, 6),
        "classification": classification,
        "verdict": verdict,
    }

    summary_path = OUT_DIR / "p1a_5seed_summary.json"
    summary_path.write_text(json.dumps({"summary": summary, "per_seed": per_seed_results}, indent=2))
    _print(f"\nSaved summary: {summary_path}")
    _print("DONE.")


if __name__ == "__main__":
    main()
