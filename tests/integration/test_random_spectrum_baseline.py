"""
Random-spectrum baseline for Theorem 3 — α-fit with untrained K projections.

For each of 5 seeds:
  1. Load Llama-3.2-1B
  2. Re-initialize the K projection weights in every attention layer
     (matching the original kaiming_uniform_ default of nn.Linear)
  3. Compute mean base_H across (layer, head, prompt) at each context
  4. Fit α from H_N = α · log N + β

Compare α_random (mean ± std over 5 seeds) to α_trained (from the main causal
run's analysis). Pre-registered interpretation (§1):
  α_random ≈ α_trained → α<1 is algebraic
  α_random ≠ α_trained → α<1 is a fact about training

Output: data/raw/phase_3_causal_full/random_spectrum_results.json
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from pathlib import Path

from tests.kernel.test_m1_acceptance import load_model_and_data
from tests.integration.test_theorem3_causal_pilot_v6 import entropy_of


OUT_PATH = Path("/home/user/dcr-attention/data/raw/phase_3_causal_full/random_spectrum_results.json")
CONTEXTS = [256, 512, 1024, 2048]   # match causal run; C=4096 added if feasible
N_PROMPTS = 10                       # fewer than main run (cheap per-α-fit)
SEEDS = [0, 1, 2, 42, 100]


def reinit_k_projections(model, seed):
    """Re-initialize k_proj weights in every attention layer.

    Uses the default nn.Linear init: kaiming_uniform_(a=sqrt(5)).
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    for layer in model.model.layers:
        k = layer.self_attn.k_proj
        # Replicate nn.Linear reset_parameters
        # Use the original layer's tensor shape for the new random init
        with torch.no_grad():
            new_weight = torch.empty(k.weight.shape, dtype=k.weight.dtype)
            torch.nn.init.kaiming_uniform_(new_weight, a=math.sqrt(5),
                                            generator=g)
            k.weight.copy_(new_weight.to(k.weight.device))
            if k.bias is not None:
                fan_in = k.weight.shape[1]
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                new_bias = torch.empty(k.bias.shape, dtype=k.bias.dtype)
                torch.nn.init.uniform_(new_bias, -bound, bound, generator=g)
                k.bias.copy_(new_bias.to(k.bias.device))


@torch.no_grad()
def mean_base_H_at_context(model, ids_full, context, n_prompts):
    """Mean base_H across all (layer, head) at the last query position, over prompts."""
    device = next(model.parameters()).device
    base_offset = context
    Hs = []
    for p_idx in range(n_prompts):
        off = p_idx * base_offset
        if off + context + 1 > ids_full.shape[0]:
            break
        ids_2d = ids_full[off:off + context].unsqueeze(0).to(device)
        out = model(ids_2d, output_attentions=True, use_cache=False, return_dict=True)
        for layer_idx, attn in enumerate(out.attentions):
            for head in range(attn.shape[1]):
                p = attn[0, head, -1, :].float()
                Hs.append(entropy_of(p))
        del out
        torch.cuda.empty_cache()
    return float(np.mean(Hs)), float(np.std(Hs)), len(Hs)


def fit_alpha(contexts, mean_Hs):
    """H_N = α log N + β; return (α, β)."""
    x = np.log(np.array(contexts, dtype=float))
    y = np.array(mean_Hs, dtype=float)
    alpha, beta = np.polyfit(x, y, 1)
    return float(alpha), float(beta)


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Load model + corpus once
    max_ctx = max(CONTEXTS)
    needed = max_ctx * N_PROMPTS + max_ctx
    model, _tok, ids_full = load_model_and_data(N=needed, seed=0)

    # Save original K weights so we can restore between seeds (avoid drift)
    original_k_weights = {}
    for i, layer in enumerate(model.model.layers):
        original_k_weights[i] = layer.self_attn.k_proj.weight.detach().clone()

    results = []
    for seed in SEEDS:
        # Restore original K → then re-init from seed
        for i, layer in enumerate(model.model.layers):
            with torch.no_grad():
                layer.self_attn.k_proj.weight.copy_(original_k_weights[i])
        reinit_k_projections(model, seed)
        print(f"\nSeed {seed}: re-initialized K projections", flush=True)

        contexts_done = []
        means = []
        stds = []
        ns = []
        for ctx in CONTEXTS:
            try:
                meanH, stdH, n = mean_base_H_at_context(model, ids_full, ctx, N_PROMPTS)
                contexts_done.append(ctx)
                means.append(meanH)
                stds.append(stdH)
                ns.append(n)
                print(f"  C={ctx}: mean base_H={meanH:.4f} ± {stdH:.4f} "
                      f"(n={n})", flush=True)
            except torch.cuda.OutOfMemoryError:
                print(f"  C={ctx}: OOM — skipping", flush=True)
                torch.cuda.empty_cache()
                continue

        if len(contexts_done) >= 2:
            alpha, beta = fit_alpha(contexts_done, means)
        else:
            alpha = beta = None

        results.append({
            "seed": seed, "contexts": contexts_done,
            "mean_base_H": means, "std_base_H": stds, "n_per_ctx": ns,
            "alpha": alpha, "beta": beta,
        })
        print(f"  α_random (seed={seed}) = {alpha:.4f}, β={beta:.4f}",
              flush=True)

    # Aggregate
    alphas = [r["alpha"] for r in results if r["alpha"] is not None]
    summary = {
        "config": {
            "contexts": CONTEXTS, "n_prompts": N_PROMPTS, "seeds": SEEDS,
        },
        "per_seed_results": results,
        "alpha_random_mean": float(np.mean(alphas)) if alphas else None,
        "alpha_random_std": float(np.std(alphas, ddof=1)) if len(alphas) > 1 else 0.0,
        "alpha_random_seeds": alphas,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Random-spectrum baseline ===")
    print(f"  α_random per seed: {alphas}")
    if alphas:
        print(f"  α_random mean ± std: {summary['alpha_random_mean']:.4f} ± "
              f"{summary['alpha_random_std']:.4f}")
    print(f"  Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
