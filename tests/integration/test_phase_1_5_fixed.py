"""
Phase 1.5 FIXED — Aggressive coverage quality sweep with corrected harness.

Fixes: _trace_nll N-argument bug (was always running N_TOKENS=2000 regardless
of N parameter). This file re-runs Q4/Q5/Q6 at true N=8000 and adds Q7 at
N=32000.

Run as script:
  cd /home/user/dcr-attention
  PYTHONPATH=/home/user/dcr-attention python tests/integration/test_phase_1_5_fixed.py \
    2>&1 | tee /tmp/phase_1_5/run_fixed.log

Configs:
  q1_recheck  N=2000  c=0.15  — validates fix reproduces +1.526%
  q4_fixed    N=8000  c=0.15
  q5_fixed    N=8000  c=0.10  — Phase 1 main win regime
  q6_fixed    N=8000  c=0.05  — extreme regime
  q7          N=32000 c=0.10  — vLLM hero scale, Theorem 3 validation

Output JSONs: /tmp/phase_1_5/{cfg_id}.json
Summary:      /tmp/phase_1_5/summary_fixed.json
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import torch

CONFIGS = [
    # (cfg_id,       N,     c_floor)
    ("q1_recheck",   2000,  0.15),
    ("q4_fixed",     8000,  0.15),
    ("q5_fixed",     8000,  0.10),
    ("q6_fixed",     8000,  0.05),
    ("q7",          32000,  0.10),
]

T_DISPATCH   = 64
K_WINDOW     = 64
OUT_DIR      = Path("/tmp/phase_1_5")

# Known reference results for cross-checks
KNOWN = {
    (2000, 0.15): 1.526,   # Q1 (Phase 1.5 original, N=2000 correct)
    (2000, 0.50): 0.285,   # M1 acceptance locked
    (8000, 0.50): 0.118,   # Phase 1 Pareto locked
    (8000, 0.30): 0.412,   # Phase 1 Pareto locked
}

# Theorem 3 exponent for N-scaling cross-check
_THM3_EXPONENT = -0.69


def _thm3_ratio(n1: int, n2: int) -> float:
    """Predicted ΔPPL(n2) / ΔPPL(n1) per Theorem 3."""
    return (n2 / n1) ** _THM3_EXPONENT


def _print_header(cfg_id: str, N: int, c: float) -> None:
    k_eff = int(c * N)
    print(f"\n{'='*65}", flush=True)
    print(f"=== {cfg_id}: N={N:,}  c_floor={c}  k_eff={k_eff} ===", flush=True)
    print(f"{'='*65}", flush=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    from dcr_attention.models.llama.config import DCRLlamaConfig
    from tests.kernel.test_m1_acceptance import load_model_and_data, run_trace

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Phase 1.5 FIXED Quality Sweep", flush=True)
    print(f"GPU: {gpu_name}", flush=True)
    print(f"Configs: {len(CONFIGS)}", flush=True)
    print(f"Output: {OUT_DIR}", flush=True)

    baselines: dict[int, float] = {}
    results: list[dict] = []

    for cfg_id, N, c_floor in CONFIGS:
        _print_header(cfg_id, N, c_floor)
        torch.cuda.empty_cache()
        t_start = time.monotonic()

        print(f"  Loading model + data (N={N:,}) ...", flush=True)
        model, tokenizer, ids = load_model_and_data(N=N + 1)

        # SDPA baseline (cache per N to avoid re-running when same N repeats)
        if N not in baselines:
            print(f"  Running SDPA baseline (N={N:,}) ...", flush=True)
            cfg_sdpa = DCRLlamaConfig(enable_dcr=False)
            ppl_base = run_trace(model, cfg_sdpa, ids, N=N, label="sdpa_baseline")
            baselines[N] = ppl_base
            print(f"  SDPA baseline PPL = {ppl_base:.4f}", flush=True)
        else:
            ppl_base = baselines[N]
            print(f"  SDPA baseline PPL = {ppl_base:.4f}  (cached)", flush=True)

        # M1 q_topk_reference
        print(f"  Running M1 q_topk_reference (c={c_floor}) ...", flush=True)
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

        # Q1_recheck sanity gate
        if cfg_id == "q1_recheck":
            expected = KNOWN.get((N, c_floor), None)
            if expected is not None:
                drift = abs(delta_ppl - expected)
                ok = drift <= 0.05
                print(
                    f"\n  Q1_RECHECK: ΔPPL = {delta_ppl:+.4f}%  "
                    f"expected ~{expected:+.3f}%  drift = {drift:.3f} pp  "
                    f"{'OK' if ok else 'FAIL — STOP AND INVESTIGATE'}",
                    flush=True,
                )
                if not ok:
                    print(
                        f"\n  ERROR: Q1 recheck drift {drift:.3f} pp > 0.05 pp gate.\n"
                        f"  Fix may have altered semantics. Aborting.",
                        flush=True,
                    )
                    sys.exit(1)

        # Theorem 3 cross-check against known N=2000 same-c or N=8000 same-c
        thm3_checks = {}
        for (ref_N, ref_c), ref_delta in KNOWN.items():
            if ref_c == c_floor and ref_N != N:
                predicted = ref_delta * _thm3_ratio(ref_N, N)
                ratio_obs = delta_ppl / ref_delta if ref_delta else None
                ratio_pred = _thm3_ratio(ref_N, N)
                thm3_checks[f"N{ref_N}"] = {
                    "ref_delta_pct": ref_delta,
                    "predicted_delta_pct": round(predicted, 4),
                    "observed_ratio": round(ratio_obs, 3) if ratio_obs else None,
                    "predicted_ratio": round(ratio_pred, 3),
                }

        print(
            f"\n  --- {cfg_id} RESULT ---\n"
            f"  PPL_base (SDPA) = {ppl_base:.4f}\n"
            f"  PPL_M1          = {ppl_m1:.4f}\n"
            f"  ΔPPL            = {delta_ppl:+.4f}%\n"
            f"  Deployability   = {deploy.upper()}\n"
            f"  Runtime         = {t_elapsed_min:.1f} min\n"
            f"  Thm3 checks     = {thm3_checks}",
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
            "theorem3_checks": thm3_checks,
            "harness_version": "fixed_post_bug",
        }
        results.append(rec)

        out_path = OUT_DIR / f"{cfg_id}.json"
        with open(out_path, "w") as f:
            json.dump(rec, f, indent=2)
        print(f"  Saved -> {out_path}", flush=True)

        del model, tokenizer, ids
        torch.cuda.empty_cache()

    # Summary table
    print(f"\n{'='*75}", flush=True)
    print(f"{'ID':>12}  {'N':>7}  {'c':>5}  {'k_eff':>7}  {'ΔPPL':>9}  {'Deploy':>10}", flush=True)
    print(f"{'-'*75}", flush=True)
    for r in results:
        print(
            f"{r['config_id']:>12}  {r['N']:>7}  {r['c_floor']:>5.2f}  "
            f"{r['k_eff']:>7}  {r['delta_ppl_pct']:>+9.4f}%  {r['deployability'].upper():>10}",
            flush=True,
        )

    summary_path = OUT_DIR / "summary_fixed.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary saved -> {summary_path}", flush=True)
    print("Phase 1.5 FIXED measurements complete.", flush=True)


if __name__ == "__main__":
    main()
