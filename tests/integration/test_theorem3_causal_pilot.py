"""
Theorem 3 causal intervention — PILOT.

Validates the treatment / matched-control construction BEFORE the full
causal run. Does NOT test the Theorem 3 hypothesis; validates only that
the intervention is correctly constructed.

Pilot scale: 2 layer/head pairs × 5 prompts × 1 query position (last),
context 512. ~10 measurement cells.

Capture strategy:
  We do not modify attention internals. We run the model with
  `output_attentions=True` and capture post-softmax attention weights
  `p[B, H, q, N_keys]`. The pre-softmax logits are recovered up to a
  per-row additive constant as `log p` — and that is sufficient,
  because softmax is shift-invariant, so perturbations `s' = log p + δ`
  give the same softmax as `s' = true_logits + δ`. The Frobenius norm
  `‖δ‖_F` is identical regardless of which "logits" reference frame
  we use. (See report §B.0 for the proof.)

Last-query-position simplification:
  We use only the last query position (index q = CONTEXT - 1). At that
  position every key 0..CONTEXT-1 is causally valid, so the
  distribution has full support with no -inf masking. This avoids
  permutation-control breaking causality. The architect's V1-V4
  validation semantics are preserved.

Validation checks:
  V1: treatment ΔH_N within ±0.15 of target 0.5 nats (mean over cells)
  V2: control ΔH_N preserves entropy: |mean(ctrl_dH)| < 0.05 nats
  V3: ‖δ_treat‖_F ≈ ‖δ_ctrl‖_F: max rel error < 2%
  V4: treatment ≫ control in H effect: treat_dH > 5 × |ctrl_dH|

Output: /data/raw/phase_3_causal_pilot/causal_pilot_results.json
"""
from __future__ import annotations

import json
import math
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
LAYER_HEAD_PAIRS = [(8, 4), (12, 8)]
TARGET_DELTA_NATS = 0.5
SEED = 0
BISECT_ITERS = 25
CTRL_TOL_V3 = 0.02
V1_TOL = 0.15
V2_CAP = 0.05
V4_RATIO = 5.0


# ---------------------------------------------------------------------------
# Entropy helper
# ---------------------------------------------------------------------------

def softmax_entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """H(softmax(logits)) per row (last-dim distribution). Returns shape logits.shape[:-1]."""
    p = F.softmax(logits.float(), dim=-1)
    logp = torch.log(p.clamp_min(1e-30))
    return -(p * logp).sum(dim=-1)


# ---------------------------------------------------------------------------
# Perturbations
# ---------------------------------------------------------------------------

def treatment_delta(base_logits: torch.Tensor, target_delta_nats: float,
                    rng: torch.Generator) -> torch.Tensor:
    """
    Build an isotropic Gaussian noise tensor δ such that
        H(softmax(base_logits + δ)) − H(softmax(base_logits)) ≈ target_delta_nats.

    Entropy as a function of noise std is NOT monotonic: it rises from base_H,
    peaks near log(N_keys), then drops back toward 0 as the noise dominates
    and softmax concentrates on the random arg-max. We must find the LOWER
    root (rising branch). Approach:

      1. Sweep std on a geometric grid until either H exceeds target
         (success — bisect in the last interval) or H starts decreasing
         (we passed the peak without ever reaching target — clamp to peak).
      2. Bisect within the last bracketing interval.

    The same noise pattern z (drawn once) is rescaled — so the returned δ is
    deterministic given (base_logits, rng_state).
    """
    base_H = softmax_entropy_from_logits(base_logits).mean().item()
    z = torch.randn(
        base_logits.shape, generator=rng,
        device=base_logits.device, dtype=base_logits.dtype,
    )

    def H_after(std: float) -> float:
        return softmax_entropy_from_logits(base_logits + z * std).mean().item()

    # Sweep std on a geometric grid; stop when H first exceeds target or peaks.
    target_H = base_H + target_delta_nats
    grid_lo, grid_hi = 0.0, None
    last_std, last_H = 0.0, base_H
    std_try = 0.05
    peak_H, peak_std = base_H, 0.0
    for _ in range(40):
        H_here = H_after(std_try)
        if H_here > peak_H:
            peak_H, peak_std = H_here, std_try
        if H_here >= target_H:
            grid_lo, grid_hi = last_std, std_try
            break
        if H_here < last_H - 0.01:
            # Past the peak without reaching target — use peak std.
            grid_lo, grid_hi = peak_std, peak_std
            break
        last_std, last_H = std_try, H_here
        std_try *= 1.4
    else:
        grid_lo, grid_hi = peak_std, peak_std

    if grid_hi > grid_lo:
        for _ in range(BISECT_ITERS):
            mid = 0.5 * (grid_lo + grid_hi)
            if H_after(mid) - base_H < target_delta_nats:
                grid_lo = mid
            else:
                grid_hi = mid
        std = 0.5 * (grid_lo + grid_hi)
    else:
        std = grid_lo

    return z * std


def matched_control_delta(base_logits: torch.Tensor, treat_delta: torch.Tensor,
                          rng: torch.Generator) -> torch.Tensor:
    """
    Permutation-based control: permute logit values across the last (keys) axis.

    A permutation leaves the multiset of softmax values unchanged → H unchanged.
    It DOES move mass to different keys → perturbs the attention output.

    The resulting raw delta is rescaled so its Frobenius norm exactly matches
    `‖treat_delta‖_F`. We match the PERTURBATION norm, NOT the logit norm —
    matching the wrong norm is the Thread A failure mode.
    """
    N_keys = base_logits.shape[-1]
    perm = torch.randperm(N_keys, generator=rng, device=base_logits.device)
    permuted = base_logits.index_select(-1, perm)
    delta_raw = permuted - base_logits

    norm_treat = torch.linalg.norm(treat_delta.flatten())
    norm_raw = torch.linalg.norm(delta_raw.flatten()).clamp_min(1e-12)
    return delta_raw * (norm_treat / norm_raw)


# ---------------------------------------------------------------------------
# Capture attention weights via output_attentions=True
# ---------------------------------------------------------------------------

def capture_attentions(model, ids: torch.Tensor, layer_indices: list) -> dict:
    """
    Run one forward pass with output_attentions=True. Return a dict
    {layer_idx: tensor[B, H, q, N_keys]} (post-softmax weights) on CPU.
    """
    device = next(model.parameters()).device
    with torch.no_grad():
        out = model(
            ids.unsqueeze(0).to(device),
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )
    attentions = out.attentions  # tuple of length n_layers, each [B, H, q, N_keys]
    captured = {}
    for li in layer_indices:
        captured[li] = attentions[li].detach().cpu()
    return captured


# ---------------------------------------------------------------------------
# Main pilot
# ---------------------------------------------------------------------------

def run_pilot():
    PILOT_OUT.mkdir(parents=True, exist_ok=True)
    print(f"Theorem 3 causal pilot — context {CONTEXT}, {N_PROMPTS} prompts, "
          f"{len(LAYER_HEAD_PAIRS)} (layer, head) pairs", flush=True)

    rng = torch.Generator(device="cpu")
    rng.manual_seed(SEED)

    layer_indices = sorted({li for li, _ in LAYER_HEAD_PAIRS})

    # Load model once; slice different prompts from the same data.
    # load_model_and_data uses offset = seed * 1000, so seed=N_PROMPTS-1=4 gives
    # offset 4000 < 252853 corpus size — safe. We then take CONTEXT-length slices
    # from offsets [0, 1000, 2000, ...] within the returned ids tensor.
    # Simpler: just load with seed=0 and a generous N, then slice.
    max_offset = (N_PROMPTS - 1) * CONTEXT
    total_needed = max_offset + CONTEXT
    model, _tok, ids_full = load_model_and_data(N=total_needed, seed=0)

    records = []
    for prompt_idx in range(N_PROMPTS):
        offset = prompt_idx * CONTEXT
        print(f"\n--- prompt {prompt_idx} (offset {offset}) ---", flush=True)
        ids = ids_full[offset:offset + CONTEXT]

        captured = capture_attentions(model, ids, layer_indices)

        for (layer, head) in LAYER_HEAD_PAIRS:
            # Post-softmax weights for the LAST query position.
            #   captured[layer] shape: [B=1, H, q=CONTEXT, N_keys=CONTEXT]
            p = captured[layer][0, head, -1, :]  # [CONTEXT]
            # Recover "logits" up to a per-row additive constant.
            logits = torch.log(p.clamp_min(1e-30)).to(torch.float64)

            base_H = softmax_entropy_from_logits(logits).item()

            d_t = treatment_delta(logits, TARGET_DELTA_NATS, rng)
            H_t = softmax_entropy_from_logits(logits + d_t).item()
            norm_t = torch.linalg.norm(d_t.flatten()).item()

            d_c = matched_control_delta(logits, d_t, rng)
            H_c = softmax_entropy_from_logits(logits + d_c).item()
            norm_c = torch.linalg.norm(d_c.flatten()).item()

            rec = {
                "layer": layer,
                "head": head,
                "prompt_idx": prompt_idx,
                "prompt_offset": offset,
                "context": CONTEXT,
                "base_H_nats": base_H,
                "treatment_H_nats": H_t,
                "control_H_nats": H_c,
                "treatment_delta_H": H_t - base_H,
                "control_delta_H": H_c - base_H,
                "treatment_norm_F": norm_t,
                "control_norm_F": norm_c,
                "norm_match_rel_error": abs(norm_t - norm_c) / max(norm_t, 1e-12),
            }
            records.append(rec)
            print(
                f"  L{layer:2d}H{head:2d}: base_H={base_H:.3f}  "
                f"ΔH_treat={rec['treatment_delta_H']:+.3f}  "
                f"ΔH_ctrl={rec['control_delta_H']:+.3f}  "
                f"norm_err={rec['norm_match_rel_error']:.4f}",
                flush=True,
            )

        torch.cuda.empty_cache()

    # ---- V1-V4 ----
    treat_dH = np.array([r["treatment_delta_H"] for r in records])
    ctrl_dH = np.array([r["control_delta_H"] for r in records])
    norm_err = np.array([r["norm_match_rel_error"] for r in records])

    V1 = bool(abs(treat_dH.mean() - TARGET_DELTA_NATS) < V1_TOL)
    V2 = bool(abs(ctrl_dH).mean() < V2_CAP)
    V3 = bool(norm_err.max() < CTRL_TOL_V3)
    V4 = bool(treat_dH.mean() > V4_RATIO * abs(ctrl_dH).mean() + 1e-12)

    validation = {
        "V1_treatment_hits_target_delta": V1,
        "V2_control_preserves_entropy": V2,
        "V3_norms_matched_within_2pct": V3,
        "V4_treatment_discriminates_from_control": V4,
        "all_pass": V1 and V2 and V3 and V4,
        "stats": {
            "treatment_delta_H_mean": float(treat_dH.mean()),
            "treatment_delta_H_std": float(treat_dH.std(ddof=1)),
            "control_delta_H_mean": float(ctrl_dH.mean()),
            "control_delta_H_std": float(ctrl_dH.std(ddof=1)),
            "norm_match_rel_error_max": float(norm_err.max()),
            "target_delta_nats": TARGET_DELTA_NATS,
            "n_cells": len(records),
        },
        "thresholds": {
            "V1_tol_nats": V1_TOL,
            "V2_cap_nats": V2_CAP,
            "V3_rel_error_cap": CTRL_TOL_V3,
            "V4_ratio_min": V4_RATIO,
        },
    }

    out_path = PILOT_OUT / "causal_pilot_results.json"
    with open(out_path, "w") as f:
        json.dump({"validation": validation, "records": records}, f, indent=2)

    print("\n=== Causal pilot V1-V4 validation ===")
    for k in (
        "V1_treatment_hits_target_delta",
        "V2_control_preserves_entropy",
        "V3_norms_matched_within_2pct",
        "V4_treatment_discriminates_from_control",
    ):
        print(f"  {k}: {'PASS' if validation[k] else 'FAIL'}")
    print(f"  ALL PASS: {validation['all_pass']}")
    print(f"\n  Treatment ΔH mean: {treat_dH.mean():+.4f} (target {TARGET_DELTA_NATS:.2f})")
    print(f"  Control ΔH mean: {ctrl_dH.mean():+.4f}")
    print(f"  Norm match max rel error: {norm_err.max():.5f}")
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    run_pilot()
