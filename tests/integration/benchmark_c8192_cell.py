"""Benchmark a single C=8192 cell: time forwards and CPU greedy.

This is to estimate full-run runtime before committing.
"""
from __future__ import annotations

import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from tests.kernel.test_m1_acceptance import load_model_and_data
from tests.integration.test_theorem3_causal_full import (
    forward_get_nll,
    install_intervention,
)
from tests.integration.test_theorem3_causal_pilot_v6 import (
    control_perturbation, entropy_of, tv_distance,
)


CONTEXT = 8192
LAYER = 0
HEAD = 4


def main():
    print(f"Benchmark cell at CONTEXT={CONTEXT}", flush=True)
    print("Loading model + data ...", flush=True)
    t_load_start = time.perf_counter()
    model, _tok, ids_full = load_model_and_data(N=CONTEXT + 1, seed=0)
    t_load = time.perf_counter() - t_load_start
    print(f"  load time: {t_load:.1f}s", flush=True)
    ids_2d = ids_full[:CONTEXT + 1].unsqueeze(0)
    rng_cpu = torch.Generator(device="cpu")
    rng_cpu.manual_seed(0)

    state, restore = install_intervention(model, LAYER)
    try:
        # Baseline + capture
        state.active = False
        state.capture = True
        state.captured_p = None
        t = time.perf_counter()
        nll_base = forward_get_nll(model, ids_2d)
        torch.cuda.synchronize()
        t_base = time.perf_counter() - t
        state.capture = False
        p_base = state.captured_p[HEAD, :].float()
        print(f"  baseline forward (+capture): {t_base:.2f}s  NLL={nll_base:.4f}",
              flush=True)
        base_H = entropy_of(p_base)
        uniform = torch.full_like(p_base, 1.0 / p_base.shape[-1])

        # 5 (λ, treat+ctrl) iterations to estimate per-cell cost
        cell_t0 = time.perf_counter()
        for lam in (0.05, 0.10, 0.15, 0.20, 0.30):
            # Treatment
            p_treated = (1.0 - lam) * p_base + lam * uniform
            tv_t = tv_distance(p_treated, p_base)
            state.active = True
            state.target_head = HEAD
            state.p_override = p_treated
            t = time.perf_counter()
            nll_t = forward_get_nll(model, ids_2d)
            torch.cuda.synchronize()
            t_treat_fwd = time.perf_counter() - t

            # Greedy on CPU
            p_base_cpu = p_base.detach().cpu()
            t = time.perf_counter()
            p_ctrl_cpu, tv_c, nsw, npass, conv = control_perturbation(
                p_base_cpu, tv_t, rng_cpu,
            )
            t_greedy = time.perf_counter() - t

            # Control forward
            state.p_override = p_ctrl_cpu.to(p_base.device)
            t = time.perf_counter()
            nll_c = forward_get_nll(model, ids_2d)
            torch.cuda.synchronize()
            t_ctrl_fwd = time.perf_counter() - t

            print(f"  λ={lam}: treat_fwd={t_treat_fwd:.2f}s  "
                  f"greedy={t_greedy:.2f}s (passes={npass}, conv={conv})  "
                  f"ctrl_fwd={t_ctrl_fwd:.2f}s  "
                  f"D={nll_t - nll_c:+.4f}",
                  flush=True)
        t_cell = time.perf_counter() - cell_t0
        print(f"\n  Total per-cell time (post-baseline): {t_cell:.1f}s", flush=True)
        print(f"  Total per-cell time including baseline: {t_cell + t_base:.1f}s",
              flush=True)

    finally:
        restore()

    # Project: 300 cells per context × 6 contexts = 1800 cells
    # But C=8192 is the slowest; smaller contexts << this
    print(f"\nIf each C=8192 cell ≈ {t_cell + t_base:.0f}s, "
          f"300 cells = {(t_cell + t_base) * 300 / 3600:.1f}h",
          flush=True)


if __name__ == "__main__":
    main()
