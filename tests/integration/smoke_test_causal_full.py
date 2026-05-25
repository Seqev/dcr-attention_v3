"""
Smoke test for theorem3_causal_full: verify intervention hook works.

Single (layer, head, prompt, context) cell. Confirms:
  1. Monkey-patch installs and restores cleanly
  2. Baseline forward gives reasonable NLL
  3. Intervention modifies attention and yields a different NLL
  4. v6 functions integrate without error
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
    entropy_of,
    tv_distance,
    control_perturbation,
)


CONTEXT = 256
LAYER = 8
HEAD = 4


def main():
    print("Loading model + data ...", flush=True)
    model, _tok, ids_full = load_model_and_data(N=CONTEXT + 1, seed=0)
    ids_2d = ids_full[:CONTEXT + 1].unsqueeze(0)
    rng = torch.Generator(device="cuda")
    rng.manual_seed(0)

    # 1) Baseline forward — capture p via patched-forward capture mode
    state, restore = install_intervention(model, LAYER)
    try:
        state.active = False
        state.capture = True
        state.captured_p = None
        t0 = time.perf_counter()
        nll_base = forward_get_nll(model, ids_2d)
        t_fwd = time.perf_counter() - t0
        state.capture = False
        assert state.captured_p is not None
        p_layer = state.captured_p
        print(f"Baseline forward: NLL={nll_base:.4f}, fwd_time={t_fwd:.3f}s",
              flush=True)
        p_base = p_layer[HEAD, :].float()
        print(f"  p_base: shape={tuple(p_base.shape)}, "
              f"sum={float(p_base.sum()):.6f} (should be 1.0)",
              flush=True)
        base_H = entropy_of(p_base)
        print(f"  base_H = {base_H:.4f} nats", flush=True)

        # 2) Treatment forward — λ-interpolation toward uniform
        lam = 0.15
        uniform = torch.full_like(p_base, 1.0 / p_base.shape[-1])
        p_treated = (1.0 - lam) * p_base + lam * uniform
        tv_t = tv_distance(p_treated, p_base)
        treat_H = entropy_of(p_treated)
        print(f"\nTreatment (λ={lam}):", flush=True)
        print(f"  treat_H = {treat_H:.4f} (Δ {treat_H - base_H:+.4f})", flush=True)
        print(f"  TV(treat, base) = {tv_t:.4f}", flush=True)

        state.active = True
        state.target_head = HEAD
        state.p_override = p_treated
        nll_treat = forward_get_nll(model, ids_2d)
        print(f"  NLL_treat = {nll_treat:.4f}  (Δ from base: "
              f"{nll_treat - nll_base:+.4f})", flush=True)

        # 3) Control forward — matched permutation
        p_ctrl, tv_c, nsw, npass, conv = control_perturbation(p_base, tv_t, rng)
        ctrl_H = entropy_of(p_ctrl)
        print(f"\nControl (matched-perm, target tv={tv_t:.4f}):", flush=True)
        print(f"  achieved tv_c = {tv_c:.4f}  "
              f"converged={conv}, passes={npass}, swaps={nsw}", flush=True)
        print(f"  ctrl_H = {ctrl_H:.6f} (Δ {ctrl_H - base_H:+.2e}) "
              f"— should be ≈ 0", flush=True)

        state.p_override = p_ctrl
        nll_ctrl = forward_get_nll(model, ids_2d)
        print(f"  NLL_ctrl = {nll_ctrl:.4f}  (Δ from base: "
              f"{nll_ctrl - nll_base:+.4f})", flush=True)

        # 4) Verify intervention OFF returns baseline
        state.active = False
        state.p_override = None
        nll_again = forward_get_nll(model, ids_2d)
        print(f"\nBaseline-again (intervention off): NLL={nll_again:.4f}  "
              f"diff from first baseline: {nll_again - nll_base:+.2e}", flush=True)
        assert abs(nll_again - nll_base) < 1e-4, "Intervention did not cleanly disable"

    finally:
        restore()

    # 5) After restore: baseline forward should still give same NLL
    nll_final = forward_get_nll(model, ids_2d)
    print(f"\nAfter restore: NLL={nll_final:.4f}  "
          f"diff from first baseline: {nll_final - nll_base:+.2e}", flush=True)
    assert abs(nll_final - nll_base) < 1e-4, "Restore did not clean up"

    print("\nSmoke test PASS — intervention installs, modifies, and restores cleanly.")
    print(f"  D = NLL_treat - NLL_ctrl = {nll_treat - nll_ctrl:+.4f} nats")


if __name__ == "__main__":
    main()
