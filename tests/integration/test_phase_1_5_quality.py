"""
Phase 1.5 — Aggressive coverage quality sweep.

6 configurations: N ∈ {2000, 8000} × c_floor ∈ {0.05, 0.10, 0.15}
Method: M1 reference (Python q_topk_reference), measured against SDPA baseline.

Run as script (not via pytest) to allow sequential execution with explicit
output per config:

  cd /home/user/dcr-attention
  PYTHONPATH=/home/user/dcr-attention python tests/integration/test_phase_1_5_quality.py

Output JSONs at /tmp/phase_1_5/q[1-6]_N<N>_c<CC>.json
Run log at /tmp/phase_1_5/run.log (redirect stdout/stderr)
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Config grid (spec §3.1)
# ---------------------------------------------------------------------------
CONFIGS = [
    ("Q1", 2000, 0.15),
    ("Q2", 2000, 0.10),
    ("Q3", 2000, 0.05),
    ("Q4", 8000, 0.15),
    ("Q5", 8000, 0.10),
    ("Q6", 8000, 0.05),
]

T_DISPATCH = 64
K_WINDOW   = 64
OUT_DIR    = Path("/tmp/phase_1_5")

# ---------------------------------------------------------------------------
# Previously locked results (for cross-reference only)
# ---------------------------------------------------------------------------
LOCKED = {
    # N=2000, c=0.5 (M1 acceptance, Phase 1 measurement)
    (2000, 0.5): 0.285,
    # N=8000, c=0.5 (Pareto extension, Phase 1)
    (8000, 0.5): 0.118,
    # N=8000, c=0.3 (Pareto extension, Phase 1)
    (8000, 0.3): 0.412,
}


def _print_header(cfg_id: str, N: int, c_floor: float) -> None:
    k_eff = int(c_floor * N)
    print(f"\n{'='*65}", flush=True)
    print(f"=== {cfg_id}: N={N:,}  c_floor={c_floor}  k_eff={k_eff} ===", flush=True)
    print(f"{'='*65}", flush=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    from dcr_attention.models.llama.config import DCRLlamaConfig
    from tests.kernel.test_m1_acceptance import load_model_and_data, run_trace

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Phase 1.5 Quality Sweep", flush=True)
    print(f"GPU: {gpu_name}", flush=True)
    print(f"Configs: {len(CONFIGS)} × (N × c_floor)", flush=True)
    print(f"Output: {OUT_DIR}", flush=True)

    baselines: dict[int, float] = {}
    results: list[dict] = []

    for cfg_id, N, c_floor in CONFIGS:
        _print_header(cfg_id, N, c_floor)

        torch.cuda.empty_cache()

        t_start = time.monotonic()

        # Load fresh model for each N to avoid KV-cache state contamination.
        print(f"  Loading model + data (N={N:,}) …", flush=True)
        model, tokenizer, ids = load_model_and_data(N=N + 1)

        # ---- SDPA baseline (cached per N) ----
        if N not in baselines:
            print(f"  Running SDPA baseline (N={N:,}) …", flush=True)
            cfg_sdpa = DCRLlamaConfig(enable_dcr=False)
            ppl_base = run_trace(model, cfg_sdpa, ids, N=N, label="sdpa_baseline")
            baselines[N] = ppl_base
            print(f"  SDPA baseline PPL = {ppl_base:.4f}", flush=True)
        else:
            ppl_base = baselines[N]
            print(f"  SDPA baseline PPL = {ppl_base:.4f}  (cached)", flush=True)

        # ---- M1 q_topk_reference at this c_floor ----
        print(f"  Running M1 q_topk_reference (c={c_floor}) …", flush=True)
        cfg_m1 = DCRLlamaConfig(
            axis_source="q_topk_reference",
            k_window=K_WINDOW,
            coverage_floor=c_floor,
            T_dispatch=T_DISPATCH,
            enable_dcr=True,
            enable_adaptive_widening=False,
        )
        ppl_m1 = run_trace(model, cfg_m1, ids, N=N, label=f"M1_c{c_floor}")

        t_elapsed_min = (time.monotonic() - t_start) / 60.0

        delta_ppl = (ppl_m1 / ppl_base - 1.0) * 100.0

        # Deployability classification
        if delta_ppl <= 0.5:
            deploy = "strict"
        elif delta_ppl <= 1.0:
            deploy = "standard"
        elif delta_ppl <= 2.0:
            deploy = "permissive"
        else:
            deploy = "unusable"

        # Compare to known locked results (same N, adjacent c) for sanity
        sanity = {}
        for (lock_N, lock_c), lock_delta in LOCKED.items():
            if lock_N == N:
                ratio_1_over_c = (1.0 / c_floor) / (1.0 / lock_c)
                predicted = lock_delta * ratio_1_over_c
                sanity[f"predicted_from_c{lock_c}"] = round(predicted, 4)

        print(
            f"\n  --- {cfg_id} RESULT ---\n"
            f"  PPL_base (SDPA) = {ppl_base:.4f}\n"
            f"  PPL_M1          = {ppl_m1:.4f}\n"
            f"  ΔPPL            = {delta_ppl:+.4f}%\n"
            f"  Deployability   = {deploy.upper()}\n"
            f"  Runtime         = {t_elapsed_min:.1f} min\n"
            f"  Sanity (Thm3)   = {sanity}",
            flush=True,
        )

        rec = {
            "config_id": cfg_id,
            "N": N,
            "c_floor": c_floor,
            "k_eff": int(c_floor * N),
            "ppl_baseline_sdpa": round(ppl_base, 6),
            "ppl_m1_topk": round(ppl_m1, 6),
            "delta_ppl_pct": round(delta_ppl, 6),
            "deployability": deploy,
            "runtime_min": round(t_elapsed_min, 2),
            "gpu": gpu_name,
            "axis_source": "q_topk_reference",
            "T_dispatch": T_DISPATCH,
            "k_window": K_WINDOW,
            "theorem3_predictions": sanity,
        }
        results.append(rec)

        out_path = OUT_DIR / f"{cfg_id.lower()}_N{N}_c{int(c_floor * 100):02d}.json"
        with open(out_path, "w") as f:
            json.dump(rec, f, indent=2)
        print(f"  Saved → {out_path}", flush=True)

        # Free model memory before next config
        del model, tokenizer, ids
        torch.cuda.empty_cache()

    # ---- Summary table ----
    print(f"\n{'='*75}", flush=True)
    print(f"{'ID':>4}  {'N':>6}  {'c':>5}  {'k_eff':>6}  {'ΔPPL':>8}  {'Deploy':>10}", flush=True)
    print(f"{'-'*75}", flush=True)
    for r in results:
        print(
            f"{r['config_id']:>4}  {r['N']:>6}  {r['c_floor']:>5.2f}  "
            f"{r['k_eff']:>6}  {r['delta_ppl_pct']:>+8.4f}%  {r['deployability'].upper():>10}",
            flush=True,
        )

    summary_path = OUT_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary saved → {summary_path}", flush=True)
    print("Phase 1.5 measurements complete.", flush=True)


if __name__ == "__main__":
    main()
