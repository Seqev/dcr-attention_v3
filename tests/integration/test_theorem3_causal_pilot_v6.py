"""
Theorem 3 causal intervention — PILOT v6 (Option I: terminating refinement greedy).

Only the control's greedy changes vs v5: single-pass → terminating refinement
greedy with undo, strict monotone error decrease, hard pass cap.

Treatment (Option H, v4), Fix A (per-cell base_H filter, v5),
Fix B (distance-k pairs, v5), recalibration — all UNCHANGED.

Termination argument:
  Every accepted swap STRICTLY decreases |tv − target|. On a finite set of
  permutations, tv takes finitely many values; |tv − target| takes finitely
  many values; a strictly-decreasing sequence over a finite set is finite.
  Independently, the pass count is hard-capped at 1 + MAX_REFINE_PASSES.
  Either bound alone guarantees termination. The loop also early-stops on
  success (within tolerance) and on stall (a full pass accepts zero swaps).
  Therefore the loop always halts; the returned permutation is exact (a
  composition of transpositions).

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
CANDIDATE_PAIRS = [
    (8, 4), (8, 12), (10, 6), (6, 10), (12, 4),
    (14, 8), (9, 2), (11, 14), (7, 6), (13, 10),
]
BASE_H_LO = 1.5
BASE_H_HI = float(np.log(CONTEXT)) - 1.5
TV_SWEEP = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
PAIR_DISTANCES = (1, 2, 4, 8, 16, 32, 64)
SIGNAL_FLOOR_NATS = 0.30
MIN_POOL = 10
MAX_REFINE_PASSES = 5
TV_MATCH_TOL = 0.10


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def entropy_of(p: torch.Tensor) -> float:
    p = p.float().clamp_min(1e-12)
    return float(-(p * torch.log(p)).sum())


def tv_distance(p: torch.Tensor, q: torch.Tensor) -> float:
    return 0.5 * float(torch.abs(p.float() - q.float()).sum())


# ---------------------------------------------------------------------------
# Treatment — λ-interpolation (v4 Option H, UNCHANGED)
# ---------------------------------------------------------------------------

def treatment_perturbation(p_base: torch.Tensor, target_tv: float):
    N = p_base.shape[-1]
    uniform = torch.full_like(p_base, 1.0 / N)
    half_l1 = 0.5 * float(torch.abs(uniform - p_base).sum())
    if half_l1 < 1e-12:
        return p_base.clone(), 0.0, 0.0
    lam = min(target_tv / half_l1, 1.0)
    p_treated = (1.0 - lam) * p_base + lam * uniform
    return p_treated, tv_distance(p_treated, p_base), lam


# ---------------------------------------------------------------------------
# Fix B distance-k pairs (v5, UNCHANGED)
# ---------------------------------------------------------------------------

def candidate_pairs(order: torch.Tensor, distances=PAIR_DISTANCES) -> list:
    N = order.shape[-1]
    pairs = []
    for d in distances:
        for i in range(0, N - d):
            pairs.append((order[i].item(), order[i + d].item()))
    return pairs


# ---------------------------------------------------------------------------
# Control — Option I: terminating refinement greedy
# ---------------------------------------------------------------------------

def control_perturbation(p_base: torch.Tensor, target_tv: float,
                         rng: torch.Generator):
    """
    Exact mass permutation matched to target_tv. Pass 0 = build; passes 1..5 =
    refinement with undo. Strict monotone descent in |tv − target| + hard pass
    cap ⇒ provable termination.
    """
    N = p_base.shape[-1]
    order = torch.argsort(p_base, descending=True)
    perm = torch.arange(N, device=p_base.device)
    pairs = candidate_pairs(order)

    def tv_of(pm):
        return tv_distance(p_base.index_select(-1, pm), p_base)

    def err(pm):
        return abs(tv_of(pm) - target_tv)

    n_swaps_total = 0
    n_passes_used = 0
    converged = False

    for pass_idx in range(1 + MAX_REFINE_PASSES):
        n_passes_used = pass_idx + 1
        idx = torch.randperm(len(pairs), generator=rng,
                             device=p_base.device).tolist()
        accepted = 0
        for ci in idx:
            a, b = pairs[ci]
            trial = perm.clone()
            t = trial[a].item()
            trial[a] = trial[b]
            trial[b] = t
            if err(trial) < err(perm):              # STRICT decrease only
                perm = trial
                n_swaps_total += 1
                accepted += 1
            if err(perm) < TV_MATCH_TOL * target_tv:
                converged = True
                break
        if converged:
            break
        if accepted == 0:                            # stall — discrete local min
            break

    p_ctrl = p_base.index_select(-1, perm)
    return p_ctrl, tv_of(perm), n_swaps_total, n_passes_used, converged


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
            cell_pool.append({
                "layer": layer, "head": head, "prompt_idx": p_idx,
                "base_H": bH,
                "in_band": BASE_H_LO <= bH <= BASE_H_HI,
            })

    admitted = [c for c in cell_pool if c["in_band"]]
    rejected = [c for c in cell_pool if not c["in_band"]]
    print(f"Fix A — strict per-cell base_H filter:")
    print(f"  admitted: {len(admitted)} cells  rejected: {len(rejected)} cells",
          flush=True)
    if len(admitted) < MIN_POOL:
        print(f"WARNING: pool {len(admitted)} < {MIN_POOL}. Surfacing.",
              flush=True)
    if not admitted:
        print("ERROR: no admitted cells.", flush=True)
        return

    # --- Step 2: recalibration sweep ---
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
        p_c, tv_c, nsw, npass, conv = control_perturbation(p_cal, tv, rng)
        dH_c = entropy_of(p_c) - base_H_cal
        matches = abs(tv_c - tv) < TV_MATCH_TOL * tv
        sweep[tv] = {
            "dH_treat": dH_t, "tv_treat": tv_t, "lambda": lam,
            "dH_ctrl": dH_c, "tv_ctrl": tv_c,
            "n_swaps": nsw, "n_passes": npass, "converged": conv,
            "ctrl_matches": matches,
        }
        print(f"  target_tv={tv:.2f}: ΔH_t={dH_t:+.4f}  tv_c={tv_c:.4f}  "
              f"matches={matches}  passes={npass}  conv={conv}",
              flush=True)

    ok = [tv for tv in TV_SWEEP
          if sweep[tv]["ctrl_matches"]
          and sweep[tv]["dH_treat"] >= SIGNAL_FLOOR_NATS]
    if not ok:
        matching = [tv for tv in TV_SWEEP if sweep[tv]["ctrl_matches"]]
        chosen_tv = min(matching) if matching else TV_SWEEP[0]
        print(f"WARNING: no tv meets both criteria; fallback {chosen_tv}.",
              flush=True)
    else:
        chosen_tv = min(ok)
    print(f"  chosen target_tv = {chosen_tv}", flush=True)

    dH_seq = [sweep[tv]["dH_treat"] for tv in TV_SWEEP]
    V5 = bool(all(dH_seq[i] < dH_seq[i + 1] for i in range(len(dH_seq) - 1)))

    # --- Step 3: pilot at chosen_tv ---
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
            p_c, tv_c, nsw, npass, conv = control_perturbation(
                p_base, chosen_tv, rng,
            )
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
                "control_n_passes": npass,
                "control_converged": conv,
                "tv_match_rel_error": abs(tv_t - tv_c) / max(chosen_tv, 1e-12),
            }
            records.append(rec)
            print(
                f"  L{c['layer']:2d}H{c['head']:2d} p{p_idx}: "
                f"base_H={base_H:.3f}  "
                f"ΔH_t={rec['treatment_delta_H']:+.4f}  "
                f"tv_c={tv_c:.4f}  err={rec['tv_match_rel_error']:.4f}  "
                f"passes={npass}  conv={conv}",
                flush=True,
            )

    # --- Step 4: V1'-V5 ---
    treat_dH = np.array([r["treatment_delta_H"] for r in records])
    ctrl_dH = np.array([r["control_delta_H"] for r in records])
    tv_err = np.array([r["tv_match_rel_error"] for r in records])
    conv_n = sum(1 for r in records if r["control_converged"])
    npass_arr = np.array([r["control_n_passes"] for r in records])

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
            "tv_match_rel_error_median": float(np.median(tv_err)),
            "tv_match_rel_error_mean": float(tv_err.mean()),
            "chosen_target_tv": chosen_tv,
            "n_cells_admitted": len(admitted),
            "n_cells_rejected": len(rejected),
            "n_cells_measured": len(records),
            "n_cells_v3_pass": int((tv_err < 0.10).sum()),
            "n_cells_control_converged": conv_n,
            "n_passes_max": int(npass_arr.max()),
            "n_passes_mean": float(npass_arr.mean()),
            "max_refine_passes_cap": MAX_REFINE_PASSES,
        },
        "tv_band_sweep": {str(tv): sweep[tv] for tv in TV_SWEEP},
        "cell_admit_reject": {
            "n_admitted": len(admitted), "n_rejected": len(rejected),
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
            "max_refine_passes": MAX_REFINE_PASSES,
            "tv_match_tol": TV_MATCH_TOL,
        },
    }

    out_path = PILOT_OUT / "causal_pilot_v6_results.json"
    with open(out_path, "w") as f:
        json.dump({"validation": validation, "records": records}, f, indent=2)

    print("\n=== Causal pilot v6 validation ===")
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
        f"\n  V3' cells passing: "
        f"{validation['stats']['n_cells_v3_pass']}/{len(records)}  "
        f"(max rel err {tv_err.max():.4f}, median {np.median(tv_err):.4f})"
    )
    print(
        f"  control converged: {conv_n}/{len(records)} cells  "
        f"(passes: mean {npass_arr.mean():.1f}, max {npass_arr.max()})"
    )
    print(
        f"  treatment ΔH: {treat_dH.mean():+.4f} ± {treat_dH.std(ddof=1):.4f}, "
        f"min {treat_dH.min():+.4f}  (all positive: {all_positive})"
    )
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
