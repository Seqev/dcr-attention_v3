"""
Theorem 3 full causal run — measurement script (resumable).

Reuses v6 intervention construction:
  - Treatment: p_treated = (1-λ)·p + λ·uniform     (Option H, v4 — closed form)
  - Control:   exact mass permutation matched on TV (v6 terminating greedy)

Per-cell flow at one (layer, head, prompt, context):
  1. Baseline forward → NLL_base; capture post-softmax p[layer, head, q=-1, :]
  2. For each λ ∈ LAMBDA_SWEEP:
       p_treated = (1-λ)·p + λ·uniform
       Forward with hook replacing p[layer, head, q=-1, :] = p_treated → NLL_treat
       Compute p_ctrl via v6 greedy matched on TV(p_treated, p_base)
       Forward with hook replacing p[layer, head, q=-1, :] = p_ctrl → NLL_ctrl

Persistent per-cell JSON checkpoint to /data/raw/phase_3_causal_full/checkpoints/.
Resumable: re-invocation skips cells whose checkpoint exists.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from pathlib import Path

from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

from tests.kernel.test_m1_acceptance import load_model_and_data
from tests.integration.test_theorem3_causal_pilot_v6 import (
    candidate_pairs,
    control_perturbation,
    entropy_of,
    tv_distance,
)


# ---------------------------------------------------------------------------
# Configuration (locked per pre-registration)
# ---------------------------------------------------------------------------

RUN_OUT = Path("/home/user/dcr-attention/data/raw/phase_3_causal_full")
CKPT = RUN_OUT / "checkpoints"
LOG_DIR = Path("/home/user/dcr-attention/data/internal")

CONTEXTS = [256, 512, 1024, 2048, 4096]   # 8192 dropped: C=8192 baseline fwd =
                                          # 58s on this hardware AND OOMs on
                                          # treatment forward (architect §8 risk
                                          # protocol allows reducing N-axis).
LAMBDA_SWEEP = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]   # λ=0 is identity
N_PROMPTS = 25
N_LAYERHEADS = 12
BASE_H_LO = 1.5
SEED = 0


# ---------------------------------------------------------------------------
# Patched attention forward — replicates eager LlamaAttention.forward with
# a single intervention point at post-softmax.
# ---------------------------------------------------------------------------

class InterventionState:
    """Mutable container for the intervention parameters.

    Three independent flags:
      - active:  if True, replace attn_weights[0, target_head, -1, :] with p_override
      - capture: if True, store attn_weights[0, :, -1, :] into captured_p (last-q only)
      - Both False ⇒ patched forward is bit-identical to original.
    """
    def __init__(self):
        self.active = False
        self.capture = False
        self.target_head: int = 0
        self.p_override: torch.Tensor | None = None
        self.captured_p: torch.Tensor | None = None    # [H, N_keys] at q=-1


def build_patched_forward(attn_module, state: InterventionState):
    """Return a forward function that mimics LlamaAttention.forward + intervention.

    Capture-on-demand at q=-1 only (memory cheap even at CONTEXT=8192).
    Override of attn_weights[0, target_head, -1, :] only when state.active.
    """
    def patched(
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()
        Q = attn_module.q_proj(hidden_states)
        K = attn_module.k_proj(hidden_states)
        V = attn_module.v_proj(hidden_states)
        Q = Q.view(bsz, q_len, -1, attn_module.head_dim).transpose(1, 2)
        K = K.view(bsz, q_len, -1, attn_module.head_dim).transpose(1, 2)
        V = V.view(bsz, q_len, -1, attn_module.head_dim).transpose(1, 2)
        if position_embeddings is None:
            cos, sin = attn_module.rotary_emb(V, position_ids)
        else:
            cos, sin = position_embeddings
        Q, K = apply_rotary_pos_emb(Q, K, cos, sin)
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            K, V = past_key_value.update(K, V, attn_module.layer_idx, cache_kwargs)
        K = repeat_kv(K, attn_module.num_key_value_groups)
        V = repeat_kv(V, attn_module.num_key_value_groups)
        attn_weights = torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(attn_module.head_dim)
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, :K.shape[-2]]
            attn_weights = attn_weights + causal_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(Q.dtype)
        attn_weights = F.dropout(attn_weights, p=attn_module.attention_dropout,
                                 training=attn_module.training)

        # Capture-only-at-q=-1 (memory-cheap)
        if state.capture:
            state.captured_p = attn_weights[0, :, -1, :].detach().clone()

        # Intervention point — in-place modification (no_grad → safe;
        # avoids cloning the full attn_weights tensor at large CONTEXT).
        if state.active and state.p_override is not None:
            h = state.target_head
            override = state.p_override.to(attn_weights.dtype).to(attn_weights.device)
            attn_weights[0, h, -1, :] = override

        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = attn_module.o_proj(attn_output)
        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_value
    return patched


def install_intervention(model, target_layer: int) -> tuple[InterventionState, callable]:
    """Monkey-patch the target layer's self_attn.forward. Return (state, restore)."""
    attn_module = model.model.layers[target_layer].self_attn
    state = InterventionState()
    original_forward = attn_module.forward
    attn_module.forward = build_patched_forward(attn_module, state)

    def restore():
        attn_module.forward = original_forward

    return state, restore


# ---------------------------------------------------------------------------
# Forward helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def forward_get_nll(model, ids_2d):
    """Single prefill forward. Returns NLL of next token at position CONTEXT.

    ids_2d: [1, CONTEXT+1] — input is ids[:CONTEXT], target is ids[CONTEXT].
    Does NOT use output_attentions=True; capture is handled by the patched
    forward's InterventionState (memory-cheap at large CONTEXT).
    """
    device = next(model.parameters()).device
    ids_2d = ids_2d.to(device)
    inp = ids_2d[:, :-1]            # [1, CONTEXT]
    target = ids_2d[0, -1]          # scalar
    out = model(inp, use_cache=False, return_dict=True)
    logit_last = out.logits[0, -1, :]      # [vocab]
    logp = F.log_softmax(logit_last.float(), dim=-1)
    return -logp[target].item()


# ---------------------------------------------------------------------------
# Per-cell measurement
# ---------------------------------------------------------------------------

def measure_cell(model, ids_full, layer, head, prompt_idx, context,
                 base_offset_per_prompt, rng_cpu, ckpt_dir):
    """Run baseline + treatment(λ) + control(λ) over LAMBDA_SWEEP. Save JSON."""
    out_path = ckpt_dir / f"cell_L{layer}H{head}_p{prompt_idx:02d}_n{context}.json"
    if out_path.exists():
        return "skip"

    offset = base_offset_per_prompt[prompt_idx]
    ids_2d = ids_full[offset:offset + context + 1].unsqueeze(0)
    if ids_2d.shape[1] < context + 1:
        return "short"

    # Install intervention on target layer
    state, restore = install_intervention(model, layer)
    try:
        # Baseline pass: state.active=False, state.capture=True →
        # patched forward writes p[:, -1, :] into state.captured_p (cheap)
        state.active = False
        state.capture = True
        state.captured_p = None
        nll_base = forward_get_nll(model, ids_2d)
        state.capture = False
        assert state.captured_p is not None, "capture failed"
        p_base = state.captured_p[head, :].float()   # [N_keys]
        base_H = entropy_of(p_base)
        uniform = torch.full_like(p_base, 1.0 / p_base.shape[-1])
        half_l1_to_uniform = 0.5 * float(torch.abs(uniform - p_base).sum())

        records_per_lambda = []
        for lam in LAMBDA_SWEEP:
            if lam == 0.0:
                # λ=0 sanity: treatment == baseline
                nll_t = nll_base
                nll_c = nll_base
                tv_t = 0.0
                tv_c = 0.0
                conv = True
                n_swaps = 0
                n_passes = 0
                dH_t = 0.0
                dH_c = 0.0
            else:
                # Treatment via λ-interpolation toward uniform
                p_treated = (1.0 - lam) * p_base + lam * uniform
                tv_t = tv_distance(p_treated, p_base)
                dH_t = entropy_of(p_treated) - base_H

                state.active = True
                state.target_head = head
                state.p_override = p_treated
                nll_t = forward_get_nll(model, ids_2d)

                # Matched control via v6 terminating refinement greedy.
                # Run on CPU to avoid GPU-sync bottleneck inside the
                # candidate-evaluation loop (~57K candidates × 2 syncs/iter at N=8192).
                p_base_cpu = p_base.detach().cpu()
                p_ctrl_cpu, tv_c, n_swaps, n_passes, conv = control_perturbation(
                    p_base_cpu, tv_t, rng_cpu,
                )
                dH_c = entropy_of(p_ctrl_cpu) - base_H
                state.p_override = p_ctrl_cpu.to(p_base.device)
                nll_c = forward_get_nll(model, ids_2d)

            records_per_lambda.append({
                "lambda": lam,
                "nll_treat": nll_t,
                "nll_ctrl": nll_c,
                "D_diff": nll_t - nll_c,
                "tv_treatment": tv_t,
                "tv_control": tv_c,
                "treatment_delta_H": dH_t,
                "control_delta_H": dH_c,
                "control_n_swaps": n_swaps,
                "control_n_passes": n_passes,
                "control_converged": conv,
            })

        cell = {
            "layer": layer, "head": head,
            "prompt_idx": prompt_idx, "prompt_offset": offset,
            "context": context,
            "nll_baseline": nll_base,
            "base_H_nats": base_H,
            "half_l1_to_uniform": half_l1_to_uniform,
            "in_band": BASE_H_LO <= base_H <= (math.log(context) - 1.5),
            "lambda_sweep": records_per_lambda,
            "seed": SEED,
        }
    finally:
        state.active = False
        state.p_override = None
        restore()

    with open(out_path, "w") as f:
        json.dump(cell, f, indent=2)
    return "done"


# ---------------------------------------------------------------------------
# Cell grid selection — scan candidate (layer, head) pairs, pick 12 with the
# most prompts in band at the smallest context.
# ---------------------------------------------------------------------------

def select_layer_head_grid(model, ids_full, base_offset_per_prompt, scan_context=512):
    """Return list of (layer, head) pairs that maximize in-band prompt count."""
    print(f"\n=== Cell grid selection (scan_context={scan_context}) ===", flush=True)
    # Scan all (layer, head) at scan_context
    band_hi = math.log(scan_context) - 1.5
    counts = {}    # (l, h) -> in_band count over prompts
    base_H_log = {}
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads

    for prompt_idx in range(N_PROMPTS):
        offset = base_offset_per_prompt[prompt_idx]
        ids_2d = ids_full[offset:offset + scan_context + 1].unsqueeze(0)
        if ids_2d.shape[1] < scan_context + 1:
            continue
        with torch.no_grad():
            out = model(
                ids_2d[:, :-1].to(next(model.parameters()).device),
                output_attentions=True, use_cache=False, return_dict=True,
            )
        for layer in range(n_layers):
            attn = out.attentions[layer]    # [1, H, q, k]
            for head in range(n_heads):
                p = attn[0, head, -1, :].float()
                bH = entropy_of(p)
                key = (layer, head)
                base_H_log.setdefault(key, []).append(bH)
                if BASE_H_LO <= bH <= band_hi:
                    counts[key] = counts.get(key, 0) + 1
        del out
        torch.cuda.empty_cache()

    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    top = ranked[:N_LAYERHEADS]
    print(f"  Top {N_LAYERHEADS} (layer, head) pairs by in-band-prompt count:")
    for (l, h), c in top:
        mean_bH = float(np.mean(base_H_log[(l, h)]))
        print(f"    L{l:2d}H{h:2d}: in-band {c}/{N_PROMPTS}, mean base_H={mean_bH:.3f}",
              flush=True)
    return [pair for pair, _c in top]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    RUN_OUT.mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # We need enough corpus for all (prompt, context) combinations
    max_context = max(CONTEXTS)
    base_offset_per_prompt = [i * max_context for i in range(N_PROMPTS)]
    total_needed = base_offset_per_prompt[-1] + max_context

    print(f"Loading model + corpus (need {total_needed} tokens) ...", flush=True)
    model, _tok, ids_full = load_model_and_data(N=total_needed, seed=SEED)

    # Cell-grid selection (cached)
    grid_path = RUN_OUT / "selected_layer_head.json"
    if grid_path.exists():
        sel = json.loads(grid_path.read_text())
        layer_head_pairs = [tuple(x) for x in sel["pairs"]]
        print(f"Loaded selected grid from {grid_path}", flush=True)
    else:
        layer_head_pairs = select_layer_head_grid(model, ids_full,
                                                  base_offset_per_prompt)
        with open(grid_path, "w") as f:
            json.dump({"pairs": [list(p) for p in layer_head_pairs]}, f, indent=2)
        print(f"Saved grid to {grid_path}", flush=True)

    # CPU rng — used by the v6 control_perturbation (running on CPU to avoid
    # GPU-sync bottleneck at large N).
    rng_cpu = torch.Generator(device="cpu")
    rng_cpu.manual_seed(SEED)

    # Smallest first — guarantees complete data through C=2048 even if C=4096
    # OOMs on this consumer GPU. Architect §8 risk-protocol: reduce N-axis if
    # large contexts infeasible.
    contexts_ordered = sorted(CONTEXTS)

    total = len(contexts_ordered) * len(layer_head_pairs) * N_PROMPTS
    done_init = sum(1 for c in contexts_ordered for (l, h) in layer_head_pairs
                    for p in range(N_PROMPTS)
                    if (CKPT / f"cell_L{l}H{h}_p{p:02d}_n{c}.json").exists())
    print(f"\nTotal cells: {total}  already done: {done_init}", flush=True)

    n_done = 0
    n_skip = 0
    n_short = 0
    t0 = time.perf_counter()

    for ctx in contexts_ordered:
        for (layer, head) in layer_head_pairs:
            for prompt_idx in range(N_PROMPTS):
                status = measure_cell(
                    model, ids_full, layer, head, prompt_idx, ctx,
                    base_offset_per_prompt, rng_cpu, CKPT,
                )
                if status == "done":
                    n_done += 1
                    elapsed = time.perf_counter() - t0
                    rate = n_done / max(elapsed, 1e-6)
                    eta_sec = (total - done_init - n_done) / max(rate, 1e-6)
                    print(
                        f"  [done] L{layer}H{head} p{prompt_idx:02d} n{ctx}  "
                        f"({n_done} new, rate {rate:.2f} cells/s, "
                        f"ETA {eta_sec / 3600:.1f}h)",
                        flush=True,
                    )
                elif status == "skip":
                    n_skip += 1
                elif status == "short":
                    n_short += 1
                    print(f"  [short] L{layer}H{head} p{prompt_idx:02d} n{ctx} "
                          f"— corpus exhausted", flush=True)

    print(f"\n=== Run complete ===")
    print(f"  new cells written: {n_done}")
    print(f"  skipped (already done): {n_skip}")
    print(f"  short (no data): {n_short}")
    print(f"  total time: {(time.perf_counter() - t0) / 3600:.2f} hours")
    print(f"  checkpoints in: {CKPT}")


if __name__ == "__main__":
    main()
