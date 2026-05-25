"""
Theorem 3 causal intervention — PILOT v5 (Fix A + Fix B + recalibration).

Treatment:  λ-interpolation toward uniform (v4 Option H — unchanged).
Control:    exact mass permutation, greedy over DISTANCE-k pairs (Fix B).
Cell pool:  strict per-cell base_H filter (Fix A).
target_tv:  recalibrated on cleaned+widened pool for ΔH_treat ≥ 0.30 nats.

Validates intervention construction. Does NOT test the Theorem 3 hypothesis.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from pathlib import Path

from tests.kernel.test_m1_acceptance import load_model_and_data


PILOT_OUT = Path("/home/user/dcr-attention/data/raw/phase_3_causal_pilot")
CONTEXT = 512
N_PROMPTS = 5
SEED = 0
# widened candidate scan (Fix A contingency — more pairs to survive strict filter)
CANDIDATE_PAIRS = [
    (8, 4), (8, 12), (10, 6), (6, 10), (12, 4),
    (14, 8), (9, 2), (11, 14), (7, 6), (13, 10),
]
BASE_H_LO = 1.5
BASE_H_HI = float(np.log(CONTEXT)) - 1.5            # ≈ 4.74
TV_SWEEP = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
PAIR_DISTANCES = (1, 2, 4, 8, 16, 32, 64)
SIGNAL_FLOOR_NATS = 0.30                            # meaningful signal floor
MIN_POOL = 10                                       # min admitted cells


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def entropy_of(p: torch.Tensor) -> float:
    p = p.float().clamp_min(1e-12)
    return float(-(p * torch.log(p)).sum())


def tv_distance(p: torch.Tensor, q: torch.Tensor) -> float:
    return 0.5 * float(torch.abs(p.float() - q.float()).sum())


# ---------------------------------------------------------------------------
# Treatment — λ interpolation (v4 Option H — unchanged)
# ---------------------------------------------------------------------------

def treatment_perturbation(p_base: torch.Tensor, target_tv: float):
    """p_treated = (1−λ)·p + λ·uniform, λ closed form to hit target_tv."""
    N = p_base.shape[-1]
    uniform = torch.full_like(p_base, 1.0 / N)
    half_l1 = 0.5 * float(torch.abs(uniform - p_base).sum())
    if half_l1 < 1e-12:
        return p_base.clone(), 0.0, 0.0
    lam = min(target_tv / half_l1, 1.0)
    p_treated = (1.0 - lam) * p_base + lam * uniform
    return p_treated, tv_distance(p_treated, p_base), lam


# ---------------------------------------------------------------------------
# Fix B — distance-k candidate pairs
# ---------------------------------------------------------------------------

def candidate_pairs(order: torch.Tensor, distances=PAIR_DISTANCES) -> list:
    """Swap pairs at multiple sort-order distances.

    A swap of order[i] and order[i+d] moves |p_order[i] − p_order[i+d]| of mass;
    larger d → larger step. Multiple d values give the greedy a spectrum.
    """
    N = order.shape[-1]
    pairs = []
    for d in distances:
        for i in range(0, N - d):
            pairs.append((order[i].item(), order[i + d].item()))
    return pairs


def control_perturbation(p_base: torch.Tensor, target_tv: float,
                         rng: torch.Generator):
    """Exact mass permutation, greedy over distance-k pairs (Fix B)."""
    N = p_base.shape[-1]
    order = torch.argsort(p_base, descending=True)
    perm = torch.arange(N, device=p_base.device)
    pairs = candidate_pairs(order)
    idx = torch.randperm(len(pairs), generator=rng,
                         device=p_base.device).tolist()

    def tv_of(pm):
        return tv_distance(p_base.index_select(-1, pm), p_base)

    n_swaps = 0
    for ci in idx:
        a, b = pairs[ci]
        trial = perm.clone()
        t = trial[a].item()
        trial[a] = trial[b]
        trial[b] = t
        if abs(tv_of(trial) - target_tv) < abs(tv_of(perm) - target_tv):
            perm = trial
            n_swaps += 1
        if abs(tv_of(perm) - target_tv) < 0.1 * target_tv:
            break
    p_ctrl = p_base.index_select(-1, perm)
    return p_ctrl, tv_of(perm), n_swaps


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------

def capture_attention_p(model, ids_2d: torch.Tensor, layers_needed: list) -> dict:
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        out = model(
            ids_2d.to(device),
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )
    return {l: out.attentions[l].float() for l in layers_needed}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    PILOT_OUT.mkdir(parents=True, exist_ok=True)
    rng = torch.Generator(device="cuda")
    rng.manual_seed(SEED)

    total_needed = CONTEXT * N_PROMPTS
    model, _tok, ids_full = load_model_and_data(N=total_needed, seed=SEED)
    prompts = [
        ids_full[i * CONTEXT:(i + 1) * CONTEXT].unsqueeze(0)
        for i in range(N_PROMPTS)
    ]

    # --- Step 1: Fix A — per-cell base_H scan ---
    layers = sorted({l for (l, _h) in CANDIDATE_PAIRS})
    cell_pool = []
    for p_idx, prompt in enumerate(prompts):
        cap = capture_attention_p(model, prompt, layers)
        for (layer, head) in CANDIDATE_PAIRS:
            bH = entropy_of(cap[layer][0, head, -1, :])
            in_band = BASE_H_LO <= bH <= BASE_H_HI
            cell_pool.append({
                "layer": layer, "head": head, "prompt_idx": p_idx,
                "base_H": bH, "in_band": in_band,
            })

    admitted = [c for c in cell_pool if c["in_band"]]
    rejected = [c for c in cell_pool if not c["in_band"]]
    print(f"Fix A — strict per-cell base_H filter:")
    print(f"  admitted: {len(admitted)} cells  rejected: {len(rejected)} cells")
    if rejected:
        print("  rejected list:")
        for c in rejected:
            print(f"    L{c['layer']:2d}H{c['head']:2d} p{c['prompt_idx']}: "
                  f"base_H={c['base_H']:.3f}")
    if len(admitted) < MIN_POOL:
        print(f"\nWARNING: only {len(admitted)} cells admitted "
              f"(< {MIN_POOL} required). Widen CANDIDATE_PAIRS and re-run.",
              flush=True)
    if not admitted:
        print("ERROR: no admitted cells — cannot proceed.", flush=True)
        return

    # --- Step 2: Recalibration TV-band sweep on first admitted cell ---
    cal = admitted[0]
    cap_cal = capture_attention_p(model, prompts[cal["prompt_idx"]],
                                  [cal["layer"]])
    p_cal = cap_cal[cal["layer"]][0, cal["head"], -1, :]
    base_H_cal = entropy_of(p_cal)
    print(f"\nRecalibration sweep on L{cal['layer']}H{cal['head']} "
          f"p{cal['prompt_idx']} (base_H={base_H_cal:.3f}):",
          flush=True)
    sweep: dict = {}
    for tv in TV_SWEEP:
        p_t, tv_t, lam = treatment_perturbation(p_cal, tv)
        dH_t = entropy_of(p_t) - base_H_cal
        p_c, tv_c, nsw = control_perturbation(p_cal, tv, rng)
        dH_c = entropy_of(p_c) - base_H_cal
        matches = abs(tv_c - tv) < 0.10 * tv
        sweep[tv] = {
            "dH_treat": dH_t, "tv_treat": tv_t, "lambda": lam,
            "dH_ctrl": dH_c, "tv_ctrl": tv_c, "n_swaps": nsw,
            "ctrl_matches": matches,
        }
        print(f"  target_tv={tv:.2f}: ΔH_t={dH_t:+.4f} (λ={lam:.3f})  "
              f"tv_c={tv_c:.4f}  matches={matches}  n_swaps={nsw}",
              flush=True)

    ok = [tv for tv in TV_SWEEP
          if sweep[tv]["ctrl_matches"]
          and sweep[tv]["dH_treat"] >= SIGNAL_FLOOR_NATS]
    if not ok:
        print(f"\nWARNING: no target_tv meets ctrl-match AND "
              f"ΔH ≥ {SIGNAL_FLOOR_NATS}. Surfacing.",
              flush=True)
        matching = [tv for tv in TV_SWEEP if sweep[tv]["ctrl_matches"]]
        chosen_tv = min(matching) if matching else TV_SWEEP[0]
    else:
        chosen_tv = min(ok)
    print(f"  chosen target_tv = {chosen_tv}", flush=True)

    # V5 monotonicity from the sweep
    dH_seq = [sweep[tv]["dH_treat"] for tv in TV_SWEEP]
    V5 = bool(all(dH_seq[i] < dH_seq[i + 1] for i in range(len(dH_seq) - 1)))

    # --- Step 3: pilot measurement at chosen_tv over the admitted pool ---
    by_prompt: dict = {}
    for c in admitted:
        by_prompt.setdefault(c["prompt_idx"], []).append(c)

    records = []
    for p_idx, cells in sorted(by_prompt.items()):
        layers_here = sorted({c["layer"] for c in cells})
        cap = capture_attention_p(model, prompts[p_idx], layers_here)
        for c in cells:
            p_base = cap[c["layer"]][0, c["head"], -1, :]
            base_H = entropy_of(p_base)
            p_t, tv_t, lam = treatment_perturbation(p_base, chosen_tv)
            H_t = entropy_of(p_t)
            p_c, tv_c, nsw = control_perturbation(p_base, chosen_tv, rng)
            H_c = entropy_of(p_c)
            rec = {
                "layer": c["layer"], "head": c["head"], "prompt_idx": p_idx,
                "context": CONTEXT, "target_tv": chosen_tv,
                "base_H_nats": base_H,
                "treatment_H_nats": H_t,
                "control_H_nats": H_c,
                "treatment_delta_H": H_t - base_H,
                "control_delta_H": H_c - base_H,
                "tv_treatment": tv_t,
                "tv_control": tv_c,
                "lambda": lam,
                "control_n_swaps": nsw,
                "tv_match_rel_error": abs(tv_t - tv_c) / max(chosen_tv, 1e-12),
            }
            records.append(rec)
            print(
                f"  L{c['layer']:2d}H{c['head']:2d} p{p_idx}: "
                f"base_H={base_H:.3f}  "
                f"ΔH_t={rec['treatment_delta_H']:+.4f}  "
                f"ΔH_c={rec['control_delta_H']:+.2e}  "
                f"tv_t={tv_t:.4f}  tv_c={tv_c:.4f}  "
                f"match_err={rec['tv_match_rel_error']:.4f}  "
                f"n_swaps={nsw}",
                flush=True,
            )

    # --- Step 4: V1'-V5 ---
    treat_dH = np.array([r["treatment_delta_H"] for r in records])
    ctrl_dH = np.array([r["control_delta_H"] for r in records])
    tv_err = np.array([r["tv_match_rel_error"] for r in records])

    all_positive = bool(np.all(treat_dH > 0))
    V1p = bool(treat_dH.mean() > 0.05 and all_positive)
    V2 = bool(np.abs(ctrl_dH).mean() < 0.02)
    V3p = bool(tv_err.max() < 0.10)
    V4 = bool(treat_dH.mean() > 5 * np.abs(ctrl_dH).mean() + 1e-9)

    validation = {
        "V1prime_treatment_meaningful_signed_allpos": V1p,
        "V2_control_preserves_entropy": V2,
        "V3prime_tv_matched": V3p,
        "V4_treatment_discriminates": V4,
        "V5_treatment_dose_monotone": V5,
        "all_pass": V1p and V2 and V3p and V4 and V5,
        "stats": {
            "treatment_delta_H_mean": float(treat_dH.mean()),
            "treatment_delta_H_std": float(treat_dH.std(ddof=1)),
            "treatment_delta_H_min": float(treat_dH.min()),
            "treatment_delta_H_max": float(treat_dH.max()),
            "treatment_delta_H_all_positive": all_positive,
            "control_delta_H_absmean": float(np.abs(ctrl_dH).mean()),
            "tv_match_rel_error_max": float(tv_err.max()),
            "tv_match_rel_error_mean": float(tv_err.mean()),
            "chosen_target_tv": chosen_tv,
            "n_cells_admitted": len(admitted),
            "n_cells_rejected": len(rejected),
            "n_cells_measured": len(records),
        },
        "tv_band_sweep": {str(tv): sweep[tv] for tv in TV_SWEEP},
        "cell_admit_reject": {
            "admitted": [{"layer": c["layer"], "head": c["head"],
                          "prompt": c["prompt_idx"], "base_H": c["base_H"]}
                         for c in admitted],
            "rejected": [{"layer": c["layer"], "head": c["head"],
                          "prompt": c["prompt_idx"], "base_H": c["base_H"]}
                         for c in rejected],
        },
        "thresholds": {
            "V1prime_min_nats": 0.05,
            "V2_cap_nats": 0.02,
            "V3prime_rel_error_cap": 0.10,
            "V4_ratio_min": 5.0,
            "signal_floor_nats": SIGNAL_FLOOR_NATS,
            "min_pool": MIN_POOL,
        },
    }

    out_path = PILOT_OUT / "causal_pilot_v5_results.json"
    with open(out_path, "w") as f:
        json.dump({"validation": validation, "records": records}, f, indent=2)

    print("\n=== Causal pilot v5 validation ===")
    for k in (
        "V1prime_treatment_meaningful_signed_allpos",
        "V2_control_preserves_entropy",
        "V3prime_tv_matched",
        "V4_treatment_discriminates",
        "V5_treatment_dose_monotone",
    ):
        print(f"  {k}: {'PASS' if validation[k] else 'FAIL'}")
    print(f"  ALL PASS: {validation['all_pass']}")
    print(
        f"\n  pool: {len(admitted)} admitted / {len(rejected)} rejected / "
        f"{len(records)} measured"
    )
    print(
        f"  treatment ΔH: {treat_dH.mean():+.4f} ± {treat_dH.std(ddof=1):.4f}, "
        f"min {treat_dH.min():+.4f}, max {treat_dH.max():+.4f}"
    )
    print(f"  all positive: {all_positive}")
    print(f"  V3' TV match max rel err: {tv_err.max():.4f} "
          f"(mean {tv_err.mean():.4f})")
    print(f"  V5 monotone sweep: {[round(x, 4) for x in dH_seq]}")
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
