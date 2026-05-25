"""
Theorem 3 causal intervention — PILOT v3 (C+C′ design).

C  — control is adjacent-key-swap permutation (small natural magnitude)
C′ — treatment matched to control on ‖Δp‖ (TV distance), the operationally
     meaningful invariant, NOT ‖δ logits‖_F.

Fixes the v2 geometric failure (wrong invariant + magnitude too large).
Validates intervention construction. Does NOT test the Theorem 3 hypothesis.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

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
N_SWAPS_SWEEP = [16, 32, 64, 128, 256]  # re-sweep upward per architect §6.2 (v1 too small)
TV_TARGET_LO, TV_TARGET_HI = 0.05, 0.20


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def attention_entropy(logits: torch.Tensor) -> torch.Tensor:
    p = F.softmax(logits.float(), dim=-1)
    logp = torch.log(p.clamp_min(1e-12))
    return -(p * logp).sum(dim=-1)


def tv_distance(p: torch.Tensor, q: torch.Tensor) -> float:
    """Total-variation distance ½‖p − q‖₁, in [0, 1]."""
    return 0.5 * torch.abs(p.float() - q.float()).sum().item()


# ---------------------------------------------------------------------------
# Perturbations
# ---------------------------------------------------------------------------

def control_perturbation(logits: torch.Tensor, n_swaps: int,
                         rng: torch.Generator) -> torch.Tensor:
    """Adjacent-key-swap permutation. Exact → H_N preserved bit-exactly."""
    N_keys = logits.shape[-1]
    perm = torch.arange(N_keys, device=logits.device)
    candidates = torch.arange(0, N_keys - 1, 2, device=logits.device)
    n = min(n_swaps, len(candidates))
    pick = torch.randperm(len(candidates), generator=rng,
                          device=logits.device)[:n]
    sel = candidates[pick]
    for i in sel.tolist():
        tmp = perm[i].item()
        perm[i] = perm[i + 1]
        perm[i + 1] = tmp
    permuted = logits.index_select(-1, perm)
    return permuted - logits


def treatment_perturbation(logits: torch.Tensor, target_tv: float,
                           rng: torch.Generator,
                           max_iter: int = 40):
    """Isotropic noise scaled so the softmax TV shift equals target_tv."""
    p_base = torch.softmax(logits.float(), dim=-1)
    z = torch.randn(
        logits.shape, generator=rng,
        device=logits.device, dtype=logits.dtype,
    )

    def tv_at(std: float) -> float:
        perturbed = torch.softmax((logits + std * z).float(), dim=-1)
        return tv_distance(perturbed, p_base)

    lo, hi = 0.0, 1.0
    bracketed = False
    for _ in range(30):
        if tv_at(hi) >= target_tv:
            bracketed = True
            break
        hi *= 1.7
    if not bracketed:
        return z * hi, tv_at(hi)

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if tv_at(mid) < target_tv:
            lo = mid
        else:
            hi = mid
    std = 0.5 * (lo + hi)
    return std * z, tv_at(std)


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------

def capture_attention_logits(model, ids_2d: torch.Tensor,
                             layers_needed: list) -> dict:
    """Forward with output_attentions=True; return {layer: log p tensor}."""
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        out = model(
            ids_2d.to(device),
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )
    captured = {}
    for layer_idx in layers_needed:
        p = out.attentions[layer_idx]
        captured[layer_idx] = torch.log(p.float().clamp_min(1e-12))
    return captured


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    PILOT_OUT.mkdir(parents=True, exist_ok=True)
    rng = torch.Generator(device="cuda")
    rng.manual_seed(SEED)

    # Load model + 5 non-overlapping prompt slices of length CONTEXT
    # load_model_and_data returns ids as 1D tensor [N+1]; we add batch dim.
    total_needed = CONTEXT * N_PROMPTS
    model, _tok, ids_full = load_model_and_data(N=total_needed, seed=SEED)
    prompts = [
        ids_full[i * CONTEXT:(i + 1) * CONTEXT].unsqueeze(0)
        for i in range(N_PROMPTS)
    ]

    # --- Step 1: base_H scan ---
    layers_to_probe = sorted({l for (l, _h) in CANDIDATE_PAIRS})
    base_H_by_pair: dict = {}
    for prompt in prompts:
        cap = capture_attention_logits(model, prompt, layers_to_probe)
        for (layer, head) in CANDIDATE_PAIRS:
            h = attention_entropy(cap[layer][0, head, -1, :]).item()
            base_H_by_pair.setdefault((layer, head), []).append(h)

    pair_meanH = {p: float(np.mean(hs)) for p, hs in base_H_by_pair.items()}
    selected = [p for p in CANDIDATE_PAIRS
                if BASE_H_LO <= pair_meanH[p] <= BASE_H_HI][:N_PAIRS_KEEP]

    print("base_H scan:")
    for p in CANDIDATE_PAIRS:
        mark = "SELECTED" if p in selected else "skip"
        print(f"  L{p[0]:2d}H{p[1]:2d}: base_H={pair_meanH[p]:.3f}  [{mark}]",
              flush=True)
    if len(selected) < N_PAIRS_KEEP:
        print(f"WARNING: only {len(selected)} pairs in band "
              f"[{BASE_H_LO}, {BASE_H_HI:.2f}].", flush=True)

    # --- Step 2: n_swaps sweep on first selected cell ---
    l0, h0 = selected[0]
    cap0 = capture_attention_logits(model, prompts[0], [l0])
    logits0 = cap0[l0][0, h0, -1, :]
    p_base0 = torch.softmax(logits0.float(), dim=-1)
    swap_sweep: dict = {}
    for ns in N_SWAPS_SWEEP:
        d = control_perturbation(logits0, ns, rng)
        p_perm = torch.softmax((logits0 + d).float(), dim=-1)
        swap_sweep[ns] = tv_distance(p_perm, p_base0)

    print("\nn_swaps sweep (TV of control on first cell L"
          f"{l0}H{h0} prompt 0):", flush=True)
    for ns, tv in swap_sweep.items():
        print(f"  n_swaps={ns:3d}: TV={tv:.4f}", flush=True)

    in_band = [ns for ns, tv in swap_sweep.items()
               if TV_TARGET_LO <= tv <= TV_TARGET_HI]
    if not in_band:
        target_mid = 0.5 * (TV_TARGET_LO + TV_TARGET_HI)
        chosen_n_swaps = min(N_SWAPS_SWEEP,
                             key=lambda ns: abs(swap_sweep[ns] - target_mid))
        print(f"  WARNING: no n_swaps gives TV in "
              f"[{TV_TARGET_LO}, {TV_TARGET_HI}]; using closest to "
              f"{target_mid:.3f}: n_swaps={chosen_n_swaps}", flush=True)
    else:
        chosen_n_swaps = min(in_band)
        print(f"  chosen n_swaps = {chosen_n_swaps} "
              f"(TV={swap_sweep[chosen_n_swaps]:.4f})", flush=True)

    # --- Step 3: pilot measurement ---
    records = []
    for (layer, head) in selected:
        for p_idx, prompt in enumerate(prompts):
            cap = capture_attention_logits(model, prompt, [layer])
            logits = cap[layer][0, head, -1, :]
            p_base = torch.softmax(logits.float(), dim=-1)
            base_H = attention_entropy(logits).item()

            # Control: adjacent-swap permutation, exact
            delta_c = control_perturbation(logits, chosen_n_swaps, rng)
            p_ctrl = torch.softmax((logits + delta_c).float(), dim=-1)
            H_ctrl = attention_entropy(logits + delta_c).item()
            tv_ctrl = tv_distance(p_ctrl, p_base)

            # Treatment: noise scaled to match control's TV
            delta_t, tv_treat = treatment_perturbation(logits, tv_ctrl, rng)
            H_treat = attention_entropy(logits + delta_t).item()

            rec = {
                "layer": layer, "head": head, "prompt_idx": p_idx,
                "context": CONTEXT, "n_swaps": chosen_n_swaps,
                "base_H_nats": base_H,
                "treatment_H_nats": H_treat,
                "control_H_nats": H_ctrl,
                "treatment_delta_H": H_treat - base_H,
                "control_delta_H": H_ctrl - base_H,
                "tv_control": tv_ctrl,
                "tv_treatment": tv_treat,
                "tv_match_rel_error": abs(tv_treat - tv_ctrl) / max(tv_ctrl, 1e-12),
            }
            records.append(rec)
            print(
                f"  L{layer:2d}H{head:2d} p{p_idx}: base_H={base_H:.3f}  "
                f"ΔH_t={rec['treatment_delta_H']:+.3f}  "
                f"ΔH_c={rec['control_delta_H']:+.3f}  "
                f"TV_c={tv_ctrl:.4f}  TV_t={tv_treat:.4f}  "
                f"match_err={rec['tv_match_rel_error']:.2e}",
                flush=True,
            )

    # --- Step 4: V1'-V4 ---
    treat_dH = np.array([r["treatment_delta_H"] for r in records])
    ctrl_dH = np.array([r["control_delta_H"] for r in records])
    tv_err = np.array([r["tv_match_rel_error"] for r in records])
    tv_c = np.array([r["tv_control"] for r in records])

    V1p = bool(treat_dH.mean() > 0.05)
    V2 = bool(np.abs(ctrl_dH).mean() < 0.02)
    V3p = bool(tv_err.max() < 0.05)
    V4 = bool(treat_dH.mean() > 5 * np.abs(ctrl_dH).mean() + 1e-9)
    sign_consistent = bool(np.all(treat_dH > 0))

    validation = {
        "V1prime_treatment_meaningful_signed": V1p,
        "V2_control_preserves_entropy": V2,
        "V3prime_tv_fit_converged": V3p,
        "V4_treatment_discriminates": V4,
        "all_pass": V1p and V2 and V3p and V4,
        "stats": {
            "treatment_delta_H_mean": float(treat_dH.mean()),
            "treatment_delta_H_std": float(treat_dH.std(ddof=1)),
            "treatment_delta_H_sign_consistent": sign_consistent,
            "treatment_delta_H_min": float(treat_dH.min()),
            "treatment_delta_H_max": float(treat_dH.max()),
            "control_delta_H_absmean": float(np.abs(ctrl_dH).mean()),
            "tv_match_rel_error_max": float(tv_err.max()),
            "tv_control_min": float(tv_c.min()),
            "tv_control_max": float(tv_c.max()),
            "tv_control_mean": float(tv_c.mean()),
            "n_cells": len(records),
            "chosen_n_swaps": chosen_n_swaps,
        },
        "n_swaps_sweep": {str(k): v for k, v in swap_sweep.items()},
        "selected_pairs": [list(p) for p in selected],
        "base_H_by_pair": {f"L{l}H{h}": pair_meanH[(l, h)]
                           for (l, h) in CANDIDATE_PAIRS},
        "thresholds": {
            "V1prime_min_nats": 0.05,
            "V2_cap_nats": 0.02,
            "V3prime_rel_error_cap": 0.05,
            "V4_ratio_min": 5.0,
            "tv_target_band": [TV_TARGET_LO, TV_TARGET_HI],
        },
    }

    out_path = PILOT_OUT / "causal_pilot_v3_results.json"
    with open(out_path, "w") as f:
        json.dump({"validation": validation, "records": records}, f, indent=2)

    print("\n=== Causal pilot v3 validation ===")
    for k in (
        "V1prime_treatment_meaningful_signed",
        "V2_control_preserves_entropy",
        "V3prime_tv_fit_converged",
        "V4_treatment_discriminates",
    ):
        print(f"  {k}: {'PASS' if validation[k] else 'FAIL'}")
    print(f"  ALL PASS: {validation['all_pass']}")
    print(
        f"\n  treatment ΔH: {treat_dH.mean():+.4f} ± {treat_dH.std(ddof=1):.4f} "
        f"nats  (sign-consistent positive: {sign_consistent})"
    )
    print(
        f"  control ΔH: {np.abs(ctrl_dH).mean():.6f} nats (abs mean)"
    )
    print(
        f"  TV match max rel err: {tv_err.max():.2e}"
    )
    print(
        f"  control TV spread: [{tv_c.min():.4f}, {tv_c.max():.4f}]  "
        f"mean {tv_c.mean():.4f}"
    )
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
