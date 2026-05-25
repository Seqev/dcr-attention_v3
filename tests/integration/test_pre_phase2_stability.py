"""
Pre-Phase-2 P0b — Q7 stability across 3 seeds + P1a/P1b quality at N=32K.

P0b: Re-runs Q7 (N=32K, c=0.10) at seeds 0, 1, 42 to verify reproducibility.
     Seed controls starting offset in WikiText-2 validation corpus (seed*1000 tokens).
P1a: N=32K, c=0.15  — wider operating range check
P1b: N=32K, c=0.30  — Theorem 3 validation at long context

Output:
  /tmp/pre_phase2/p0b_q7_stability.json
  /tmp/pre_phase2/p1a_N32K_c15.json
  /tmp/pre_phase2/p1b_N32K_c30.json

Run:
  cd /home/user/dcr-attention
  PYTHONPATH=/home/user/dcr-attention python tests/integration/test_pre_phase2_stability.py \
    2>&1 | tee /tmp/pre_phase2/p0b_p1_run.log
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch

OUT_DIR = Path("/tmp/pre_phase2")

# Q7 phase 1.5 single-seed reference
_Q7_REFERENCE_DELTA = 0.9693

# P0b seeds — different starting offsets in WikiText-2 validation corpus
Q7_SEEDS   = [0, 1, 42]
Q7_N       = 32000
Q7_C       = 0.10

# P1a / P1b configs
P1_CONFIGS = [
    ("P1a", 32000, 0.15),
    ("P1b", 32000, 0.30),
]

T_DISPATCH = 64
K_WINDOW   = 64


def _deployability(delta_pct: float) -> str:
    if delta_pct <= 0.5:
        return "strict"
    elif delta_pct <= 1.0:
        return "standard"
    elif delta_pct <= 2.0:
        return "permissive"
    return "unusable"


def run_one(model, ids, N, c, label, gpu) -> dict:
    from dcr_attention.models.llama.config import DCRLlamaConfig
    from tests.kernel.test_m1_acceptance import run_trace

    t0 = time.monotonic()
    cfg_sdpa = DCRLlamaConfig(enable_dcr=False)
    ppl_base = run_trace(model, cfg_sdpa, ids, N=N, label="sdpa_baseline")

    cfg_m1 = DCRLlamaConfig(
        axis_source="q_topk_reference",
        coverage_floor=c,
        k_window=K_WINDOW,
        T_dispatch=T_DISPATCH,
        enable_dcr=True,
        enable_adaptive_widening=False,
    )
    ppl_m1 = run_trace(model, cfg_m1, ids, N=N, label=label)
    elapsed = (time.monotonic() - t0) / 60.0

    delta = (ppl_m1 / ppl_base - 1.0) * 100.0
    deploy = _deployability(delta)

    print(
        f"  PPL_base = {ppl_base:.4f}  PPL_M1 = {ppl_m1:.4f}  "
        f"ΔPPL = {delta:+.4f}%  [{deploy.upper()}]  {elapsed:.1f} min",
        flush=True,
    )
    return {
        "N": N,
        "c_floor": c,
        "k_eff": int(c * N),
        "ppl_baseline_sdpa": round(ppl_base, 6),
        "ppl_m1_topk": round(ppl_m1, 6),
        "delta_ppl_pct": round(delta, 6),
        "deployability": deploy,
        "runtime_min": round(elapsed, 2),
        "gpu": gpu,
        "axis_source": "q_topk_reference",
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    from tests.kernel.test_m1_acceptance import load_model_and_data

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Pre-Phase-2: P0b Q7 stability + P1a/P1b quality at N=32K", flush=True)
    print(f"GPU: {gpu}", flush=True)

    # -----------------------------------------------------------------------
    # P0b — Q7 stability (3 seeds)
    # -----------------------------------------------------------------------
    print(f"\n{'='*65}", flush=True)
    print(f"P0b — Q7 stability: N={Q7_N}, c={Q7_C}, seeds={Q7_SEEDS}", flush=True)
    print(f"{'='*65}", flush=True)

    # Verify seeds produce different data
    print(f"\n  Seed verification (first 5 token IDs per seed):", flush=True)
    from transformers import AutoTokenizer
    from datasets import load_dataset
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    text = "\n\n".join(ds["text"])
    all_ids = tok(text, return_tensors="pt").input_ids[0]
    for s in Q7_SEEDS:
        off = s * 1000
        snippet = all_ids[off: off + 5].tolist()
        print(f"    seed={s} offset={off}: {snippet}", flush=True)
    del tok, ds, text, all_ids

    q7_per_seed = []
    for seed in Q7_SEEDS:
        print(f"\n--- Q7 seed={seed} ---", flush=True)
        torch.cuda.empty_cache()
        model, _, ids = load_model_and_data(N=Q7_N + 1, seed=seed)
        rec = run_one(model, ids, Q7_N, Q7_C, f"M1_c{Q7_C}_seed{seed}", gpu)
        rec["seed"] = seed
        q7_per_seed.append(rec)
        del model, ids
        torch.cuda.empty_cache()

    deltas = [r["delta_ppl_pct"] for r in q7_per_seed]
    mean_d = sum(deltas) / len(deltas)
    var_d  = sum((d - mean_d) ** 2 for d in deltas) / (len(deltas) - 1)
    std_d  = var_d ** 0.5

    if std_d < 0.05:
        stability = "stable"
    elif std_d < 0.15:
        stability = "acceptable"
    elif std_d < 0.30:
        stability = "marginal"
    else:
        stability = "unstable"

    drift_from_phase15 = abs(mean_d - _Q7_REFERENCE_DELTA)

    print(f"\n  === P0b SUMMARY ===", flush=True)
    print(f"  seeds: {Q7_SEEDS}", flush=True)
    print(f"  deltas: {[f'{d:+.4f}%' for d in deltas]}", flush=True)
    print(f"  mean ΔPPL: {mean_d:+.4f}%", flush=True)
    print(f"  std ΔPPL:  {std_d:.4f} pp", flush=True)
    print(f"  stability: {stability.upper()}", flush=True)
    print(f"  Phase 1.5 single-seed ref: {_Q7_REFERENCE_DELTA:+.4f}%", flush=True)
    print(f"  drift from ref: {drift_from_phase15:.4f} pp", flush=True)
    print(f"  deployability (mean): {_deployability(mean_d).upper()}", flush=True)

    p0b_out = {
        "config": f"N={Q7_N}, c={Q7_C}, axis=q_topk_reference",
        "seeds": Q7_SEEDS,
        "phase_1_5_single_seed_ref": _Q7_REFERENCE_DELTA,
        "mean_delta_pct": round(mean_d, 6),
        "std_delta_pct": round(std_d, 6),
        "min_delta_pct": round(min(deltas), 6),
        "max_delta_pct": round(max(deltas), 6),
        "drift_from_phase15_ref": round(drift_from_phase15, 6),
        "stability_verdict": stability,
        "deployability": _deployability(mean_d),
        "per_seed": q7_per_seed,
    }
    out_path = OUT_DIR / "p0b_q7_stability.json"
    with open(out_path, "w") as f:
        json.dump(p0b_out, f, indent=2)
    print(f"  Saved -> {out_path}", flush=True)

    # -----------------------------------------------------------------------
    # P1a / P1b — quality at N=32K, c=0.15 and c=0.30
    # -----------------------------------------------------------------------
    # Theorem 3 predictions at N=32K
    # Known: (N=2000, c=0.15): +1.526%;  (N=2000, c=0.30): +0.687%
    # Exponent at c=0.10 empirical: -0.29
    # At c=0.15 use Thm3 nominal: (32000/2000)^{-0.69} = 16^{-0.69} ≈ 0.148
    #   predicted = 1.526 * 0.148 ≈ 0.226%
    # At c=0.30 use Thm3 nominal: 0.687 * 0.148 ≈ 0.102%
    THM3_PREDICTIONS = {
        0.15: round(1.526 * (16 ** -0.69), 4),
        0.30: round(0.687 * (16 ** -0.69), 4),
    }

    for cfg_id, N, c in P1_CONFIGS:
        print(f"\n{'='*65}", flush=True)
        print(f"=== {cfg_id}: N={N}, c={c} ===", flush=True)
        print(f"{'='*65}", flush=True)
        torch.cuda.empty_cache()

        model, _, ids = load_model_and_data(N=N + 1, seed=0)
        rec = run_one(model, ids, N, c, f"M1_c{c}", gpu)
        rec["config_id"] = cfg_id

        pred = THM3_PREDICTIONS.get(c)
        if pred is not None:
            obs_ratio = rec["delta_ppl_pct"] / 1.526 if c == 0.15 else rec["delta_ppl_pct"] / 0.687
            thm3_ratio = 16 ** -0.69
            print(
                f"  Theorem 3 predicted: {pred:+.4f}%  "
                f"(predicted_ratio={thm3_ratio:.3f}, observed_ratio={obs_ratio:.3f})",
                flush=True,
            )
            rec["thm3_prediction_pct"] = pred
            rec["thm3_predicted_ratio"] = round(thm3_ratio, 4)
            rec["thm3_observed_ratio"]  = round(obs_ratio, 4)

        out_path = OUT_DIR / f"{cfg_id.lower()}_N{N}_c{int(c*100):02d}.json"
        with open(out_path, "w") as f:
            json.dump(rec, f, indent=2)
        print(f"  Saved -> {out_path}", flush=True)

        del model, ids
        torch.cuda.empty_cache()

    print(f"\nP0b + P1a + P1b complete.", flush=True)


if __name__ == "__main__":
    main()
