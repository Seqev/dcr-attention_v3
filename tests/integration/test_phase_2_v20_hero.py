"""
Phase 2 Task 3 — v2.0 hero operating point re-validation.

Configuration: N=20000, c=0.5, axis_source=q_topk_reference (M1 Python reference).
Seeds: [0, 1, 2, 42, 100] — standard 5-seed set.
Estimated runtime: ~15-20 GPU hours (run over 2-3 nights).

Acceptance criteria (v2-A: STRICT ≤ 0.5% pp):
  - mean ΔPPL ≤ 0.5%  → v2-A (STRICT, no paper action)
  - 0.5% < mean ≤ 1.0% → v2-B-mild (STANDARD, update supplementary)
  - mean > 1.0%        → v2-B-severe (paper section update required)

Theorem 3 cross-check (N-scaling):
  Effective exponent α estimated from N=2K (Phase 2 Task 1) and N=20K (this task).
  Expected slope: (1-α_eff) ≈ 0.46 at c=0.5 (from §10 of kernel_spec.md).

Output: /data/raw/phase_2_repro/v20_hero/{seed_N.json, summary.json, scaling.json}
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
)
from dcr_attention.models.llama.config import DCRLlamaConfig
from dcr_attention.models.llama.monkey_patch import (
    patch_llama_with_dcr,
    reconfigure_dcr,
    is_patched,
)

# ---------------------------------------------------------------------------
# Phase 2 Task 3 configuration
# ---------------------------------------------------------------------------

SEEDS = [0, 1, 2, 42, 100]
N_HERO = 20_000          # v2.0 hero operating point
N_PHASE2 = 2_000         # N=2K from Task 1 (for Theorem 3 cross-check)
C_FLOOR = 0.5
AXIS_SOURCE = "q_topk_reference"

# Scenario thresholds
STRICT_THRESHOLD   = 0.5   # pp — v2-A
STANDARD_THRESHOLD = 1.0   # pp — v2-B-mild
STD_CAP            = 0.20  # pp — std cap at N=20K (wider than N=2K)
MAX_SEED_DELTA     = 1.0   # pp — per-seed cap at N=20K

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', 'data', 'raw', 'phase_2_repro', 'v20_hero'
)
M1_2K_DIR = os.path.join(
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

            if (t + 1) % 1000 == 0 or (t + 1) == N:
                elapsed = time.perf_counter() - t0
                cum_ppl = math.exp(nll[:t + 1].mean().item())
                rate = (t + 1) / elapsed
                eta = (N - t - 1) / rate if rate > 0 else 0
                print(
                    f"  [{label}] t={t+1:>6}/{N}  cum_ppl={cum_ppl:.4f}  "
                    f"{rate:.1f} tok/s  ETA {eta/60:.0f}m",
                    flush=True,
                )

    assert torch.isfinite(nll).all(), f"Non-finite NLL in {label}"
    ppl = math.exp(nll.mean().item())
    assert 0.0 < ppl < 10_000.0, f"PPL={ppl:.2f} out of range in {label}"
    return nll


def _measure_seed(seed: int) -> dict:
    """Run M1 (baseline + q_topk_reference) at N=20K for one seed."""
    model, _tok, ids = load_model_and_data(N=N_HERO + 1, seed=seed)

    cfg_base = DCRLlamaConfig(enable_dcr=False)
    if is_patched(model):
        reconfigure_dcr(model, cfg_base)
    else:
        patch_llama_with_dcr(model, cfg_base)

    nll_base = _trace_nll(model, ids[:N_HERO + 1], N=N_HERO, label=f"base-s{seed}")
    ppl_base = math.exp(nll_base.mean().item())

    cfg_m1 = DCRLlamaConfig(
        enable_dcr=True,
        coverage_floor=C_FLOOR,
        axis_source=AXIS_SOURCE,
        T_dispatch=1,
    )
    reconfigure_dcr(model, cfg_m1)

    nll_m1 = _trace_nll(model, ids[:N_HERO + 1], N=N_HERO, label=f"hero-s{seed}")
    ppl_m1 = math.exp(nll_m1.mean().item())
    delta_pct = (ppl_m1 - ppl_base) / ppl_base * 100.0

    return standardized_json_output(
        config_id=f"v20hero_c{int(C_FLOOR*100):03d}_N{N_HERO}_seed{seed:03d}",
        N=N_HERO,
        c_floor=C_FLOOR,
        ppl_baseline=ppl_base,
        ppl_dcr=ppl_m1,
        delta_ppl_pct=delta_pct,
        seed=seed,
        axis_source=AXIS_SOURCE,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("seed", SEEDS)
def test_v20_hero_single_seed(seed):
    """Run v2.0 hero measurement (N=20K) for one seed and save result JSON."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    result = _measure_seed(seed)
    delta = result["delta_ppl_pct"]

    out_path = os.path.join(
        OUTPUT_DIR,
        f"v20hero_seed{seed:03d}_N{N_HERO}_c{int(C_FLOOR*100):03d}.json"
    )
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved: {out_path}")
    print(f"  Seed {seed}: N=20K ΔPPL = {delta:+.4f}%")

    assert delta <= MAX_SEED_DELTA, (
        f"Seed {seed}: ΔPPL={delta:.4f}% exceeds per-seed cap {MAX_SEED_DELTA}%"
    )


@pytest.mark.slow
def test_v20_hero_multiseed_summary():
    """
    Aggregate 5-seed v2.0 hero results and classify scenario (v2-A/B-mild/B-severe).
    Also runs Theorem 3 N-scaling cross-check if N=2K results exist.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    deltas = []

    for seed in SEEDS:
        path = os.path.join(
            OUTPUT_DIR,
            f"v20hero_seed{seed:03d}_N{N_HERO}_c{int(C_FLOOR*100):03d}.json"
        )
        if not os.path.exists(path):
            result = _measure_seed(seed)
            with open(path, "w") as f:
                json.dump(result, f, indent=2)
        else:
            with open(path) as f:
                result = json.load(f)
        deltas.append(result["delta_ppl_pct"])

    mean_d = float(np.mean(deltas))
    std_d = float(np.std(deltas, ddof=1))
    max_d = float(np.max(deltas))

    if mean_d <= STRICT_THRESHOLD:
        scenario = "v2-A"
        classification = "STRICT"
        paper_action = "none"
    elif mean_d <= STANDARD_THRESHOLD:
        scenario = "v2-B-mild"
        classification = "STANDARD"
        paper_action = "update supplementary table"
    else:
        scenario = "v2-B-severe"
        classification = "NONSTANDARD"
        paper_action = "paper section update required"

    # Theorem 3 N-scaling cross-check
    theorem3_result = None
    m1_2k_path = os.path.join(M1_2K_DIR, "m1_5seed_summary.json")
    if os.path.exists(m1_2k_path):
        with open(m1_2k_path) as f:
            m1_2k = json.load(f)
        mean_2k = m1_2k["summary"]["mean_delta_pct"]
        # α from N-scaling: ΔPPL ∝ N^{-(1-α)}
        # (1-α) = log(ΔPPL_20K/ΔPPL_2K) / log(N_20K/N_2K)
        if mean_2k > 0 and mean_d > 0:
            log_ratio_ppl = math.log(mean_d / mean_2k)
            log_ratio_N = math.log(N_HERO / N_PHASE2)
            alpha_eff = 1.0 - log_ratio_ppl / log_ratio_N
            pred_20k_from_2k = mean_2k * (N_HERO / N_PHASE2) ** (-(1.0 - alpha_eff))
            theorem3_result = {
                "mean_delta_2k": mean_2k,
                "mean_delta_20k": mean_d,
                "effective_alpha": alpha_eff,
                "one_minus_alpha": 1.0 - alpha_eff,
                "expected_one_minus_alpha_at_c05": 0.46,
                "theorem3_prediction_20k": mean_2k * (N_HERO / N_PHASE2) ** (-0.46),
            }
            print(f"\n  Theorem 3 cross-check:")
            print(f"    N=2K ΔPPL: {mean_2k:+.4f}%  N=20K ΔPPL: {mean_d:+.4f}%")
            print(f"    Effective (1-α): {1-alpha_eff:.3f}  (spec predicts 0.46 at c=0.5)")

    summary = {
        "phase": "2_v20_hero",
        "config": {
            "N": N_HERO, "c_floor": C_FLOOR,
            "axis_source": AXIS_SOURCE, "seeds": SEEDS,
        },
        "summary": {
            "seeds_tested": SEEDS,
            "mean_delta_pct": mean_d,
            "std_delta_pct": std_d,
            "max_delta_pct": max_d,
            "classification": classification,
            "scenario": scenario,
            "paper_action": paper_action,
            "per_seed": {str(s): d for s, d in zip(SEEDS, deltas)},
        },
        "theorem3": theorem3_result,
    }

    summary_path = os.path.join(OUTPUT_DIR, "v20hero_5seed_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  v2.0 Hero 5-seed: mean={mean_d:+.4f}%  std={std_d:.4f} pp")
    print(f"  Scenario: {scenario}  ({classification})")
    print(f"  Paper action: {paper_action}")
    for s, d in zip(SEEDS, deltas):
        print(f"    seed={s:3d}: {d:+.4f}%")

    # Always pass — log results for architect decision; don't fail on scenario classification
    # (v2-B scenarios require paper action but are not test failures)
    print(f"\n  PHASE 2 v2.0 HERO CLASSIFICATION: {scenario}")
