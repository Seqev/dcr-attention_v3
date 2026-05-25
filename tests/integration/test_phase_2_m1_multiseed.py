"""
Phase 2 Task 1 — M1 multi-seed reproducibility at hero operating point.

Configuration: N=2000, c=0.5, axis_source=q_topk_reference (M1 Python reference).
Seeds: [0, 1, 2, 42, 100] — standard 5-seed set.

Acceptance criteria (STRICT ≤ 0.5% pp):
  - mean ΔPPL ≤ 0.5%
  - all individual seeds ΔPPL ≤ 0.8%   (guard: no outlier)
  - std pp ≤ 0.15 pp

Locked reference (M1 Outcome 3, seed=0):
  ΔPPL = +0.285% (q_topk_reference vs SDPA at N=2000 c=0.5)

Output: /data/raw/phase_2_repro/m1_acceptance/{seed_N.json, summary.json}
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
N_PHASE2 = 2000          # matches Phase 1 / M1 Outcome 3
C_FLOOR = 0.5            # hero phase 2 config (c=0.5 GREEN from Phase 1.5)
AXIS_SOURCE = "q_topk_reference"

# Acceptance thresholds
MEAN_DELTA_STRICT = 0.5    # pp — mean must be ≤ this
MAX_SEED_DELTA    = 0.8    # pp — individual seed cap
STD_CAP           = 0.15   # pp — std cap

# Locked reference seed-0 result (M1 Outcome 3)
REF_SEED0_DELTA = 0.285    # pp
REF_SEED0_TOL   = 0.05     # pp — tolerance around seed-0 reference

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', 'data', 'raw', 'phase_2_repro', 'm1_acceptance'
)


# ---------------------------------------------------------------------------
# Core measurement helper
# ---------------------------------------------------------------------------

def _measure_seed(seed: int) -> dict:
    """Run M1 (baseline + q_topk_reference) for one seed, return result dict."""
    model, _tok, ids = load_model_and_data(N=N_PHASE2 + 1, seed=seed)

    # SDPA baseline
    cfg_base = DCRLlamaConfig(enable_dcr=False)
    if is_patched(model):
        reconfigure_dcr(model, cfg_base)
    else:
        patch_llama_with_dcr(model, cfg_base)

    nll_base = _trace_nll(model, ids[:N_PHASE2 + 1], N=N_PHASE2, label=f"base-s{seed}")
    ppl_base = math.exp(nll_base.mean().item())

    # M1 q_topk_reference
    cfg_m1 = DCRLlamaConfig(
        enable_dcr=True,
        coverage_floor=C_FLOOR,
        axis_source=AXIS_SOURCE,
        T_dispatch=1,     # force DCR on every decode step
    )
    reconfigure_dcr(model, cfg_m1)

    nll_m1 = _trace_nll(model, ids[:N_PHASE2 + 1], N=N_PHASE2, label=f"m1-s{seed}")
    ppl_m1 = math.exp(nll_m1.mean().item())

    delta_pct = (ppl_m1 - ppl_base) / ppl_base * 100.0

    result = standardized_json_output(
        config_id=f"m1_c{int(C_FLOOR*100):03d}_N{N_PHASE2}_seed{seed:03d}",
        N=N_PHASE2,
        c_floor=C_FLOOR,
        ppl_baseline=ppl_base,
        ppl_dcr=ppl_m1,
        delta_ppl_pct=delta_pct,
        seed=seed,
        axis_source=AXIS_SOURCE,
    )
    return result


def _trace_nll(model, ids: torch.Tensor, N: int, label: str = "") -> torch.Tensor:
    """Teacher-forced NLL trace — mirrors test_m1_acceptance._trace_nll."""
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("seed", SEEDS)
def test_m1_phase2_single_seed(seed):
    """Run M1 measurement for one seed and save result JSON."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    result = _measure_seed(seed)
    delta = result["delta_ppl_pct"]

    out_path = os.path.join(OUTPUT_DIR, f"m1_seed{seed:03d}_N{N_PHASE2}_c{int(C_FLOOR*100):03d}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved: {out_path}")
    print(f"  Seed {seed}: ΔPPL = {delta:+.4f}%  baseline={result['ppl_baseline_sdpa']:.4f}  dcr={result['ppl_dcr']:.4f}")

    # Seed-0 locked reference check
    if seed == 0:
        assert abs(delta - REF_SEED0_DELTA) <= REF_SEED0_TOL, (
            f"Seed-0 ΔPPL={delta:.4f}% deviates from locked reference "
            f"{REF_SEED0_DELTA:.4f}% by more than {REF_SEED0_TOL} pp. "
            f"Potential regression."
        )

    assert delta <= MAX_SEED_DELTA, (
        f"Seed {seed}: ΔPPL={delta:.4f}% exceeds per-seed cap {MAX_SEED_DELTA}%"
    )


@pytest.mark.slow
def test_m1_phase2_multiseed_summary():
    """
    Aggregate 5-seed results into summary JSON.
    Requires all 5 per-seed JSONs to exist (run test_m1_phase2_single_seed first).
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    deltas = []
    seed_results = []

    for seed in SEEDS:
        path = os.path.join(OUTPUT_DIR, f"m1_seed{seed:03d}_N{N_PHASE2}_c{int(C_FLOOR*100):03d}.json")
        if not os.path.exists(path):
            # Run inline if not already saved
            result = _measure_seed(seed)
            with open(path, "w") as f:
                json.dump(result, f, indent=2)
        else:
            with open(path) as f:
                result = json.load(f)
        deltas.append(result["delta_ppl_pct"])
        seed_results.append(result)

    mean_d = float(np.mean(deltas))
    std_d = float(np.std(deltas, ddof=1))
    max_d = float(np.max(deltas))
    min_d = float(np.min(deltas))

    if mean_d <= MEAN_DELTA_STRICT:
        classification = "STRICT"
    elif mean_d <= 1.0:
        classification = "STANDARD"
    elif mean_d <= 2.0:
        classification = "PERMISSIVE"
    else:
        classification = "UNUSABLE"

    summary = {
        "phase": "2_m1_multiseed",
        "config": {
            "N": N_PHASE2,
            "c_floor": C_FLOOR,
            "axis_source": AXIS_SOURCE,
            "seeds": SEEDS,
        },
        "summary": {
            "seeds_tested": SEEDS,
            "mean_delta_pct": mean_d,
            "std_delta_pct": std_d,
            "min_delta_pct": min_d,
            "max_delta_pct": max_d,
            "classification": classification,
            "per_seed": {str(s): d for s, d in zip(SEEDS, deltas)},
        },
    }

    summary_path = os.path.join(OUTPUT_DIR, "m1_5seed_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  5-seed summary: mean={mean_d:+.4f}%  std={std_d:.4f} pp  [{classification}]")
    for s, d in zip(SEEDS, deltas):
        print(f"    seed={s:3d}: {d:+.4f}%")

    # Acceptance gates
    assert mean_d <= MEAN_DELTA_STRICT, (
        f"Mean ΔPPL={mean_d:.4f}% exceeds STRICT threshold {MEAN_DELTA_STRICT}%. "
        f"Classification: {classification}"
    )
    assert std_d <= STD_CAP, (
        f"Std ΔPPL={std_d:.4f} pp exceeds cap {STD_CAP} pp — seeds are inconsistent"
    )
    assert max_d <= MAX_SEED_DELTA, (
        f"Max per-seed ΔPPL={max_d:.4f}% exceeds cap {MAX_SEED_DELTA}%"
    )

    print(f"\n  PHASE 2 M1 ACCEPTANCE: {classification} — PASS")
