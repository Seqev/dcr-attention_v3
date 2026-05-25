"""
Phase 2 Task 2 — M4 multi-seed reproducibility at hero operating point.

Configuration: N=2000, c=0.5, axis_source=q_topk_triton (M4 Triton kernel).
Seeds: [0, 1, 2, 42, 100] — standard 5-seed set.

Acceptance criteria (STRICT ≤ 0.5% pp):
  - mean ΔPPL ≤ 0.5%
  - all individual seeds ΔPPL ≤ 0.8%
  - std pp ≤ 0.15 pp

Task 2.5 — M4–M1 drift cross-check:
  - |mean_M4 - mean_M1| ≤ 0.15 pp (v2.0 published value: -0.115 pp)

Output: /data/raw/phase_2_repro/m4_acceptance/{seed_N.json, summary.json, drift.json}
"""

from __future__ import annotations
import json
import math
import os
import sys
import time

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from tests.kernel.test_m1_acceptance import (
    load_model_and_data,
    standardized_json_output,
    PREFILL_SIZE,
    N_TOKENS,
)
from dcr_attention.models.llama.config import DCRLlamaConfig
from dcr_attention.models.llama.monkey_patch import (
    patch_llama_with_dcr,
    reconfigure_dcr,
    is_patched,
)

# ---------------------------------------------------------------------------
# Phase 2 configuration
# ---------------------------------------------------------------------------

SEEDS = [0, 1, 2, 42, 100]
N_PHASE2 = 2000
C_FLOOR = 0.5
AXIS_SOURCE_M4 = "q_topk_triton"
AXIS_SOURCE_M1 = "q_topk_reference"

MEAN_DELTA_STRICT = 0.5
MAX_SEED_DELTA    = 0.8
STD_CAP           = 0.15
DRIFT_CAP         = 0.15   # pp — M4 vs M1 mean drift cap

OUTPUT_DIR_M4 = os.path.join(
    os.path.dirname(__file__), '..', '..', 'data', 'raw', 'phase_2_repro', 'm4_acceptance'
)
OUTPUT_DIR_M1 = os.path.join(
    os.path.dirname(__file__), '..', '..', 'data', 'raw', 'phase_2_repro', 'm1_acceptance'
)


# ---------------------------------------------------------------------------
# Core measurement helper
# ---------------------------------------------------------------------------

def _trace_nll(model, ids: torch.Tensor, N: int, label: str = "") -> torch.Tensor:
    """Teacher-forced NLL trace."""
    device = next(model.parameters()).device
    ids = ids.to(device)
    nll = torch.zeros(N, dtype=torch.float64)
    t0 = time.perf_counter()

    with torch.no_grad():
        out = model(ids[:PREFILL_SIZE].unsqueeze(0), use_cache=True, return_dict=True)
        past = out.past_key_values
        logit_prev = out.logits[0, -1]

        for t in range(N):
            target = ids[PREFILL_SIZE + t]
            nll[t] = -F.log_softmax(logit_prev.float(), dim=-1)[target].item()

            if t < N - 1:
                out = model(
                    ids[PREFILL_SIZE + t].unsqueeze(0).unsqueeze(0),
                    past_key_values=past,
                    use_cache=True,
                    return_dict=True,
                )
                torch.cuda.synchronize()
                past = out.past_key_values
                logit_prev = out.logits[0, -1]

            if (t + 1) % 200 == 0 or (t + 1) == N:
                elapsed = time.perf_counter() - t0
                cum_ppl = math.exp(nll[:t + 1].mean().item())
                rate = (t + 1) / elapsed
                print(f"  [{label}] t={t+1:>5}/{N}  cum_ppl={cum_ppl:.4f}  {rate:.1f} tok/s",
                      flush=True)

    assert torch.isfinite(nll).all(), f"Non-finite NLL in {label}"
    ppl = math.exp(nll.mean().item())
    assert 0.0 < ppl < 10_000.0, f"PPL={ppl:.2f} out of range in {label}"
    return nll


def _measure_seed_m4(seed: int) -> dict:
    """Run M4 (baseline + q_topk_triton) for one seed."""
    model, _tok, ids = load_model_and_data(N=N_PHASE2 + 1, seed=seed)

    cfg_base = DCRLlamaConfig(enable_dcr=False)
    if is_patched(model):
        reconfigure_dcr(model, cfg_base)
    else:
        patch_llama_with_dcr(model, cfg_base)

    nll_base = _trace_nll(model, ids[:N_PHASE2 + 1], N=N_PHASE2, label=f"base-s{seed}")
    ppl_base = math.exp(nll_base.mean().item())

    cfg_m4 = DCRLlamaConfig(
        enable_dcr=True,
        coverage_floor=C_FLOOR,
        axis_source=AXIS_SOURCE_M4,
        T_dispatch=1,
    )
    reconfigure_dcr(model, cfg_m4)

    nll_m4 = _trace_nll(model, ids[:N_PHASE2 + 1], N=N_PHASE2, label=f"m4-s{seed}")
    ppl_m4 = math.exp(nll_m4.mean().item())
    delta_pct = (ppl_m4 - ppl_base) / ppl_base * 100.0

    return standardized_json_output(
        config_id=f"m4_c{int(C_FLOOR*100):03d}_N{N_PHASE2}_seed{seed:03d}",
        N=N_PHASE2,
        c_floor=C_FLOOR,
        ppl_baseline=ppl_base,
        ppl_dcr=ppl_m4,
        delta_ppl_pct=delta_pct,
        seed=seed,
        axis_source=AXIS_SOURCE_M4,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("seed", SEEDS)
def test_m4_phase2_single_seed(seed):
    """Run M4 measurement for one seed and save result JSON."""
    os.makedirs(OUTPUT_DIR_M4, exist_ok=True)

    result = _measure_seed_m4(seed)
    delta = result["delta_ppl_pct"]

    out_path = os.path.join(OUTPUT_DIR_M4, f"m4_seed{seed:03d}_N{N_PHASE2}_c{int(C_FLOOR*100):03d}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved: {out_path}")
    print(f"  Seed {seed}: M4 ΔPPL = {delta:+.4f}%")

    assert delta <= MAX_SEED_DELTA, (
        f"Seed {seed}: M4 ΔPPL={delta:.4f}% exceeds per-seed cap {MAX_SEED_DELTA}%"
    )


@pytest.mark.slow
def test_m4_phase2_multiseed_summary():
    """Aggregate 5-seed M4 results and run M4–M1 drift cross-check (Task 2.5)."""
    os.makedirs(OUTPUT_DIR_M4, exist_ok=True)
    deltas_m4 = []

    for seed in SEEDS:
        path = os.path.join(OUTPUT_DIR_M4, f"m4_seed{seed:03d}_N{N_PHASE2}_c{int(C_FLOOR*100):03d}.json")
        if not os.path.exists(path):
            result = _measure_seed_m4(seed)
            with open(path, "w") as f:
                json.dump(result, f, indent=2)
        else:
            with open(path) as f:
                result = json.load(f)
        deltas_m4.append(result["delta_ppl_pct"])

    mean_m4 = float(np.mean(deltas_m4))
    std_m4 = float(np.std(deltas_m4, ddof=1))
    max_m4 = float(np.max(deltas_m4))

    if mean_m4 <= MEAN_DELTA_STRICT:
        classification = "STRICT"
    elif mean_m4 <= 1.0:
        classification = "STANDARD"
    elif mean_m4 <= 2.0:
        classification = "PERMISSIVE"
    else:
        classification = "UNUSABLE"

    summary = {
        "phase": "2_m4_multiseed",
        "config": {"N": N_PHASE2, "c_floor": C_FLOOR, "axis_source": AXIS_SOURCE_M4, "seeds": SEEDS},
        "summary": {
            "seeds_tested": SEEDS,
            "mean_delta_pct": mean_m4,
            "std_delta_pct": std_m4,
            "max_delta_pct": max_m4,
            "classification": classification,
            "per_seed": {str(s): d for s, d in zip(SEEDS, deltas_m4)},
        },
    }

    summary_path = os.path.join(OUTPUT_DIR_M4, "m4_5seed_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  M4 5-seed summary: mean={mean_m4:+.4f}%  std={std_m4:.4f} pp  [{classification}]")

    # Task 2.5: M4–M1 drift cross-check
    m1_summary_path = os.path.join(OUTPUT_DIR_M1, "m1_5seed_summary.json")
    if os.path.exists(m1_summary_path):
        with open(m1_summary_path) as f:
            m1_summary = json.load(f)
        mean_m1 = m1_summary["summary"]["mean_delta_pct"]
        drift = mean_m4 - mean_m1

        drift_result = {
            "mean_m4_delta_pct": mean_m4,
            "mean_m1_delta_pct": mean_m1,
            "drift_m4_minus_m1_pp": drift,
            "drift_cap_pp": DRIFT_CAP,
            "v20_published_drift_pp": -0.115,
            "verdict": "PASS" if abs(drift) <= DRIFT_CAP else "FAIL",
        }
        drift_path = os.path.join(OUTPUT_DIR_M4, "m4_m1_drift.json")
        with open(drift_path, "w") as f:
            json.dump(drift_result, f, indent=2)

        print(f"  M4–M1 drift: {drift:+.4f} pp  (v2.0 published: -0.115 pp)  "
              f"cap: ±{DRIFT_CAP} pp  → {'PASS' if abs(drift) <= DRIFT_CAP else 'FAIL'}")
        assert abs(drift) <= DRIFT_CAP, (
            f"M4–M1 drift={drift:+.4f} pp exceeds cap ±{DRIFT_CAP} pp. "
            f"M4 kernel may have diverged from M1 reference."
        )
    else:
        print(f"  [WARN] M1 summary not found at {m1_summary_path}; skipping drift check")

    assert mean_m4 <= MEAN_DELTA_STRICT, (
        f"M4 mean ΔPPL={mean_m4:.4f}% exceeds STRICT threshold {MEAN_DELTA_STRICT}%"
    )
    assert std_m4 <= STD_CAP, (
        f"M4 std={std_m4:.4f} pp exceeds cap {STD_CAP} pp"
    )
    assert max_m4 <= MAX_SEED_DELTA, (
        f"M4 max per-seed ΔPPL={max_m4:.4f}% exceeds cap {MAX_SEED_DELTA}%"
    )

    print(f"\n  PHASE 2 M4 ACCEPTANCE: {classification} — PASS")
