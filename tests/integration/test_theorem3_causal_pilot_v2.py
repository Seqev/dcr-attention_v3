"""
Theorem 3 causal intervention — PILOT v2 (norm-leading design).

Fixes 3 bugs from v1 (see CAUSAL_PILOT_V2 prompt §0):
  - Bug 1: control is now exact permutation, no rescale
  - Bug 2: treatment scaled by norm (monotonic), not entropy
  - Bug 3: V3' is now a genuine convergence check, not a tautology

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
# candidate (layer, head) pairs to scan; keep first N_PAIRS_KEEP in the base_H band
CANDIDATE_PAIRS = [(8, 4), (8, 12), (10, 6), (14, 2), (6, 10), (10, 20)]
BASE_H_LO = 1.5
BASE_H_HI = float(np.log(CONTEXT)) - 1.5   # ≈ 4.74 for CONTEXT=512
N_PAIRS_KEEP = 3


# ---------------------------------------------------------------------------
# Entropy
# ---------------------------------------------------------------------------

def attention_entropy(logits: torch.Tensor) -> torch.Tensor:
    """H_N per row, nats, fp32."""
    p = F.softmax(logits.float(), dim=-1)
    logp = torch.log(p.clamp_min(1e-12))
    return -(p * logp).sum(dim=-1)


# ---------------------------------------------------------------------------
# Perturbations — norm-leading
# ---------------------------------------------------------------------------

def control_perturbation(logits: torch.Tensor, rng: torch.Generator) -> torch.Tensor:
    """Exact permutation of logit values across keys. No rescale.

    A permutation leaves the multiset {softmax(logits)_i} invariant, hence H_N
    is preserved EXACTLY (up to fp precision). The returned δ has whatever
    Frobenius norm the permutation naturally produces — that norm is what
    treatment will be matched to.
    """
    N_keys = logits.shape[-1]
    perm = torch.randperm(N_keys, generator=rng, device=logits.device)
    permuted = logits.index_select(-1, perm)
    return permuted - logits


def treatment_perturbation(logits: torch.Tensor, target_norm: float,
                           rng: torch.Generator) -> torch.Tensor:
    """Isotropic Gaussian noise scaled so ‖δ‖_F = target_norm.

    Norm scales linearly in the scaling factor, so a single division suffices —
    no bisection. (v1's entropy bisection had the non-monotonicity bug; norm
    bisection cannot.)
    """
    noise = torch.randn(
        logits.shape, generator=rng,
        device=logits.device, dtype=logits.dtype,
    )
    noise_norm = torch.linalg.norm(noise.flatten()).clamp_min(1e-12)
    delta = noise * (target_norm / noise_norm)
    achieved = torch.linalg.norm(delta.flatten()).item()
    assert abs(achieved - target_norm) / max(target_norm, 1e-12) < 1e-4, (
        f"norm match failed: achieved {achieved}, target {target_norm}"
    )
    return delta


# ---------------------------------------------------------------------------
# Capture post-softmax attentions, return log-probs as logit-equivalents.
# ---------------------------------------------------------------------------

def capture_attention_logits(model, ids_2d: torch.Tensor, layers_needed: list) -> dict:
    """Forward with output_attentions=True; return {layer: log p tensor [B,H,q,k]}."""
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

    # Load model once; carve 5 non-overlapping prompts of length CONTEXT.
    # ids returned 1D: [CONTEXT*N_PROMPTS + 1]
    total_needed = CONTEXT * N_PROMPTS
    model, _tok, ids_full = load_model_and_data(N=total_needed, seed=SEED)
    prompts = [
        ids_full[i * CONTEXT:(i + 1) * CONTEXT].unsqueeze(0)
        for i in range(N_PROMPTS)
    ]

    # --- Step 1: base_H scan ---
    layers_to_probe = sorted({l for (l, _h) in CANDIDATE_PAIRS})
    base_H_by_pair: dict = {}
    for p_idx, prompt in enumerate(prompts):
        cap = capture_attention_logits(model, prompt, layers_to_probe)
        for (layer, head) in CANDIDATE_PAIRS:
            logits_lastq = cap[layer][0, head, -1, :]
            h = attention_entropy(logits_lastq).item()
            base_H_by_pair.setdefault((layer, head), []).append(h)

    pair_meanH = {pair: float(np.mean(hs)) for pair, hs in base_H_by_pair.items()}
    selected = [
        pair for pair in CANDIDATE_PAIRS
        if BASE_H_LO <= pair_meanH[pair] <= BASE_H_HI
    ][:N_PAIRS_KEEP]

    print("base_H scan:")
    for pair in CANDIDATE_PAIRS:
        mark = "SELECTED" if pair in selected else "skip"
        print(f"  L{pair[0]:2d}H{pair[1]:2d}: base_H={pair_meanH[pair]:.3f}  [{mark}]",
              flush=True)

    if len(selected) < N_PAIRS_KEEP:
        print(f"WARNING: only {len(selected)} pairs in band "
              f"[{BASE_H_LO}, {BASE_H_HI:.2f}]; full pilot may underrun.",
              flush=True)

    # --- Step 2: pilot measurement ---
    records = []
    for (layer, head) in selected:
        for p_idx, prompt in enumerate(prompts):
            cap = capture_attention_logits(model, prompt, [layer])
            logits = cap[layer][0, head, -1, :]
            base_H = attention_entropy(logits).item()

            delta_c = control_perturbation(logits, rng)
            H_ctrl = attention_entropy(logits + delta_c).item()
            norm_c = torch.linalg.norm(delta_c.flatten()).item()

            delta_t = treatment_perturbation(logits, norm_c, rng)
            H_treat = attention_entropy(logits + delta_t).item()
            norm_t = torch.linalg.norm(delta_t.flatten()).item()

            rec = {
                "layer": layer, "head": head, "prompt_idx": p_idx,
                "context": CONTEXT,
                "base_H_nats": base_H,
                "treatment_H_nats": H_treat,
                "control_H_nats": H_ctrl,
                "treatment_delta_H": H_treat - base_H,
                "control_delta_H": H_ctrl - base_H,
                "control_norm_F": norm_c,
                "treatment_norm_F": norm_t,
                "norm_match_rel_error": abs(norm_t - norm_c) / max(norm_c, 1e-12),
            }
            records.append(rec)
            print(
                f"  L{layer:2d}H{head:2d} p{p_idx}: base_H={base_H:.3f}  "
                f"ΔH_t={rec['treatment_delta_H']:+.3f}  "
                f"ΔH_c={rec['control_delta_H']:+.3f}  "
                f"‖δ‖={norm_c:.2f}  match_err={rec['norm_match_rel_error']:.2e}",
                flush=True,
            )

    # --- Step 3: V1'-V4 ---
    treat_dH = np.array([r["treatment_delta_H"] for r in records])
    ctrl_dH = np.array([r["control_delta_H"] for r in records])
    norm_err = np.array([r["norm_match_rel_error"] for r in records])
    norms = np.array([r["control_norm_F"] for r in records])

    V1p = bool(treat_dH.mean() > 0.20)
    V2 = bool(np.abs(ctrl_dH).mean() < 0.02)
    V3p = bool(norm_err.max() < 1e-3)
    V4 = bool(treat_dH.mean() > 5 * np.abs(ctrl_dH).mean() + 1e-9)

    validation = {
        "V1prime_treatment_meaningful": V1p,
        "V2_control_preserves_entropy": V2,
        "V3prime_norm_fit_converged": V3p,
        "V4_treatment_discriminates": V4,
        "all_pass": V1p and V2 and V3p and V4,
        "stats": {
            "treatment_delta_H_mean": float(treat_dH.mean()),
            "treatment_delta_H_std": float(treat_dH.std(ddof=1)),
            "control_delta_H_mean": float(ctrl_dH.mean()),
            "control_delta_H_std": float(ctrl_dH.std(ddof=1)),
            "control_delta_H_absmean": float(np.abs(ctrl_dH).mean()),
            "norm_match_rel_error_max": float(norm_err.max()),
            "perturbation_norm_min": float(norms.min()),
            "perturbation_norm_max": float(norms.max()),
            "perturbation_norm_mean": float(norms.mean()),
            "perturbation_norm_std": float(norms.std(ddof=1)),
            "n_cells": len(records),
        },
        "selected_pairs": [list(p) for p in selected],
        "base_H_by_pair": {f"L{l}H{h}": pair_meanH[(l, h)]
                           for (l, h) in CANDIDATE_PAIRS},
        "thresholds": {
            "V1prime_min_nats": 0.20,
            "V2_cap_nats": 0.02,
            "V3prime_rel_error_cap": 1e-3,
            "V4_ratio_min": 5.0,
        },
    }

    out_path = PILOT_OUT / "causal_pilot_v2_results.json"
    with open(out_path, "w") as f:
        json.dump({"validation": validation, "records": records}, f, indent=2)

    print("\n=== Causal pilot v2 validation ===")
    for k in (
        "V1prime_treatment_meaningful",
        "V2_control_preserves_entropy",
        "V3prime_norm_fit_converged",
        "V4_treatment_discriminates",
    ):
        print(f"  {k}: {'PASS' if validation[k] else 'FAIL'}")
    print(f"  ALL PASS: {validation['all_pass']}")
    print(
        f"\n  treatment ΔH: {treat_dH.mean():+.4f} ± {treat_dH.std(ddof=1):.4f} nats"
    )
    print(
        f"  control  ΔH: {ctrl_dH.mean():+.4f} ± {ctrl_dH.std(ddof=1):.4f} nats"
        f"  (|mean| = {np.abs(ctrl_dH).mean():.4f})"
    )
    print(
        f"  norm match max rel err: {norm_err.max():.2e}"
    )
    print(
        f"  perturbation norm spread: "
        f"[{norms.min():.2f}, {norms.max():.2f}]  "
        f"mean {norms.mean():.2f} ± {norms.std(ddof=1):.2f}"
    )
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
