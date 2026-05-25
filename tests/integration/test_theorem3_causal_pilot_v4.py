"""
Theorem 3 causal intervention — PILOT v4 (Option H: entropy interpolation).

Treatment: p_treated = (1 − λ)·p + λ·uniform   — monotone, one-sided H↑
Control:   exact mass permutation              — H preserved bit-exactly
Matching:  ‖Δp‖ TV distance

Fixes the v3 root cause: additive isotropic noise treatment is two-sided in
sign and non-monotone in dose (see v3 raw-data analysis). λ-interpolation is
monotone and one-sided by entropy concavity.

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
CANDIDATE_PAIRS = [(8, 4), (8, 12), (10, 6), (14, 2), (6, 10), (10, 20)]
BASE_H_LO = 1.5
BASE_H_HI = float(np.log(CONTEXT)) - 1.5
N_PAIRS_KEEP = 3
TV_SWEEP = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def entropy_of(p: torch.Tensor) -> float:
    p = p.float().clamp_min(1e-12)
    return float(-(p * torch.log(p)).sum())


def tv_distance(p: torch.Tensor, q: torch.Tensor) -> float:
    return 0.5 * float(torch.abs(p.float() - q.float()).sum())


# ---------------------------------------------------------------------------
# Treatment — closed-form λ interpolation toward uniform
# ---------------------------------------------------------------------------

def treatment_perturbation(p_base: torch.Tensor, target_tv: float):
    """p_treated = (1−λ)·p + λ·uniform with λ s.t. TV(p_treated, p_base) == target_tv.

    TV(p_treated, p_base) = ½‖λ·(uniform − p_base)‖₁ = λ · ½‖uniform − p_base‖₁
    ⇒ λ = target_tv / (½‖uniform − p_base‖₁), clamped to [0, 1].
    """
    N = p_base.shape[-1]
    uniform = torch.full_like(p_base, 1.0 / N)
    half_l1 = 0.5 * float(torch.abs(uniform - p_base).sum())
    if half_l1 < 1e-12:
        return p_base.clone(), 0.0, 0.0
    lam = min(target_tv / half_l1, 1.0)
    p_treated = (1.0 - lam) * p_base + lam * uniform
    return p_treated, tv_distance(p_treated, p_base), lam


# ---------------------------------------------------------------------------
# Control — exact mass permutation, greedy partial-perm to match target TV
# ---------------------------------------------------------------------------

def control_perturbation(p_base: torch.Tensor, target_tv: float,
                         rng: torch.Generator):
    """Greedy partial permutation: swap pairs ADJACENT in p-sorted order
    until TV reaches within 10% of target_tv. p_ctrl is an exact permutation
    of p_base ⇒ H preserved bit-exactly.
    """
    N = p_base.shape[-1]
    order = torch.argsort(p_base, descending=True)
    perm = torch.arange(N, device=p_base.device)

    def tv_of(pm):
        return tv_distance(p_base.index_select(-1, pm), p_base)

    cand = [(order[k].item(), order[k + 1].item()) for k in range(N - 1)]
    idx = torch.randperm(len(cand), generator=rng,
                         device=p_base.device).tolist()
    n_swaps = 0
    for ci in idx:
        a, b = cand[ci]
        trial = perm.clone()
        t = trial[a].item()
        trial[a] = trial[b]
        trial[b] = t
        if abs(tv_of(trial) - target_tv) < abs(tv_of(perm) - target_tv):
            perm = trial
            n_swaps += 1
        if abs(tv_of(perm) - target_tv) < 0.1 * target_tv:
            break
    return p_base.index_select(-1, perm), tv_of(perm), n_swaps


# ---------------------------------------------------------------------------
# Capture post-softmax attention probabilities
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

    # --- Step 1: base_H scan ---
    layers = sorted({l for (l, _h) in CANDIDATE_PAIRS})
    base_H_by_pair: dict = {}
    for prompt in prompts:
        cap = capture_attention_p(model, prompt, layers)
        for (layer, head) in CANDIDATE_PAIRS:
            p = cap[layer][0, head, -1, :]
            base_H_by_pair.setdefault((layer, head), []).append(entropy_of(p))
    pair_meanH = {p: float(np.mean(hs)) for p, hs in base_H_by_pair.items()}
    selected = [p for p in CANDIDATE_PAIRS
                if BASE_H_LO <= pair_meanH[p] <= BASE_H_HI][:N_PAIRS_KEEP]
    print("base_H scan:")
    for p in CANDIDATE_PAIRS:
        mark = "SELECTED" if p in selected else "skip"
        print(f"  L{p[0]:2d}H{p[1]:2d}: base_H={pair_meanH[p]:.3f}  [{mark}]",
              flush=True)

    # --- Step 2: TV-band sweep on first selected cell ---
    l0, h0 = selected[0]
    cap0 = capture_attention_p(model, prompts[0], [l0])
    p0 = cap0[l0][0, h0, -1, :]
    base_H0 = entropy_of(p0)
    print(f"\nTV-band sweep on L{l0}H{h0} p0 (base_H={base_H0:.3f}):", flush=True)
    sweep: dict = {}
    for tv in TV_SWEEP:
        p_t, tv_t, lam = treatment_perturbation(p0, tv)
        dH_t = entropy_of(p_t) - base_H0
        p_c, tv_c, nsw = control_perturbation(p0, tv, rng)
        dH_c = entropy_of(p_c) - base_H0
        ctrl_ok = abs(tv_c - tv) < 0.10 * tv
        sweep[tv] = {
            "dH_treat": dH_t, "tv_treat": tv_t, "lambda": lam,
            "dH_ctrl": dH_c, "tv_ctrl": tv_c, "n_swaps": nsw,
            "ctrl_matches": ctrl_ok,
        }
        print(f"  target_tv={tv:.2f}: ΔH_t={dH_t:+.4f}  λ={lam:.3f}  "
              f"ΔH_c={dH_c:+.2e}  tv_c={tv_c:.4f}  n_swaps={nsw}  "
              f"ctrl_ok={ctrl_ok}",
              flush=True)

    ok = [tv for tv in TV_SWEEP
          if sweep[tv]["dH_treat"] > 0.05 and sweep[tv]["ctrl_matches"]]
    if not ok:
        chosen_tv = TV_SWEEP[len(TV_SWEEP) // 2]
        print(f"  WARNING: no target_tv satisfies both criteria — "
              f"using fallback {chosen_tv}", flush=True)
    else:
        chosen_tv = min(ok)
        print(f"  chosen target_tv = {chosen_tv}", flush=True)

    # V5 monotonicity: ΔH_treat strictly increasing in target_tv
    dH_sweep = [sweep[tv]["dH_treat"] for tv in TV_SWEEP]
    V5 = bool(all(dH_sweep[i] < dH_sweep[i + 1]
                  for i in range(len(dH_sweep) - 1)))

    # --- Step 3: pilot at chosen_tv ---
    records = []
    for (layer, head) in selected:
        for p_idx, prompt in enumerate(prompts):
            cap = capture_attention_p(model, prompt, [layer])
            p_base = cap[layer][0, head, -1, :]
            base_H = entropy_of(p_base)

            p_t, tv_t, lam = treatment_perturbation(p_base, chosen_tv)
            H_t = entropy_of(p_t)
            p_c, tv_c, nsw = control_perturbation(p_base, chosen_tv, rng)
            H_c = entropy_of(p_c)

            rec = {
                "layer": layer, "head": head, "prompt_idx": p_idx,
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
                f"  L{layer:2d}H{head:2d} p{p_idx}: base_H={base_H:.3f}  "
                f"ΔH_t={rec['treatment_delta_H']:+.4f}  "
                f"ΔH_c={rec['control_delta_H']:+.2e}  "
                f"tv_t={tv_t:.4f}  tv_c={tv_c:.4f}  λ={lam:.3f}  "
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
            "chosen_target_tv": chosen_tv,
            "n_cells": len(records),
        },
        "tv_band_sweep": {str(tv): sweep[tv] for tv in TV_SWEEP},
        "selected_pairs": [list(p) for p in selected],
        "base_H_by_pair": {f"L{l}H{h}": pair_meanH[(l, h)]
                           for (l, h) in CANDIDATE_PAIRS},
        "thresholds": {
            "V1prime_min_nats": 0.05,
            "V2_cap_nats": 0.02,
            "V3prime_rel_error_cap": 0.10,
            "V4_ratio_min": 5.0,
        },
    }

    out_path = PILOT_OUT / "causal_pilot_v4_results.json"
    with open(out_path, "w") as f:
        json.dump({"validation": validation, "records": records}, f, indent=2)

    print("\n=== Causal pilot v4 validation ===")
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
        f"\n  treatment ΔH: {treat_dH.mean():+.4f} ± {treat_dH.std(ddof=1):.4f} "
        f"nats  min {treat_dH.min():+.4f}  max {treat_dH.max():+.4f}"
    )
    print(f"  all positive: {all_positive}")
    print(f"  control ΔH abs mean: {np.abs(ctrl_dH).mean():.2e}")
    print(f"  TV match max rel err: {tv_err.max():.2e}")
    print(f"  V5 monotone sweep: {[round(x, 4) for x in dH_sweep]}")
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
