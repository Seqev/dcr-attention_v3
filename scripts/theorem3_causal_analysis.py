"""
Theorem 3 causal analysis — stratified, pre-registered.

Runs ONLY after all cell checkpoints exist in /data/raw/phase_3_causal_full/checkpoints/.
Pure analysis; no GPU.

Applies pre-registered (§1 of CLAUDE_CODE_PROMPT_CAUSAL_FULL) verdict logic
MECHANICALLY — no post-hoc reinterpretation.

Outputs:
  data/raw/phase_3_causal_full/analysis.json
  data/raw/phase_3_causal_full/analysis_by_context.json
"""
from __future__ import annotations

import json
import math
import os
import sys

from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


RUN_OUT = Path("/home/user/dcr-attention/data/raw/phase_3_causal_full")
CKPT = RUN_OUT / "checkpoints"
ANALYSIS_PATH = RUN_OUT / "analysis.json"
ANALYSIS_BY_CTX_PATH = RUN_OUT / "analysis_by_context.json"

# Pre-registered reference λ for the primary test (mid-range of LAMBDA_SWEEP).
# Frozen here before viewing data; verdict logic in §6 applies at this λ.
REGISTERED_LAMBDA = 0.15

ALPHA_PRIMARY = 0.05    # one-sided Wilcoxon
ALPHA_BIAS = 0.05       # two-sided Mann-Whitney
N_BOOTSTRAP = 10_000


def load_cells():
    """Load all cell JSONs from CKPT/."""
    files = sorted(CKPT.glob("cell_*.json"))
    cells = []
    for f in files:
        with open(f) as fh:
            cells.append(json.load(fh))
    return cells


def bootstrap_ci_median(x, n=N_BOOTSTRAP, alpha=0.05):
    """Bootstrap 95% CI for the median."""
    x = np.asarray(x)
    if x.size < 2:
        return float(np.median(x)), float("nan"), float("nan")
    rng = np.random.default_rng(0)
    boots = np.array([
        np.median(rng.choice(x, size=x.size, replace=True))
        for _ in range(n)
    ])
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(np.median(x)), float(lo), float(hi)


def collect_D_at_lambda(cells, lam, only_converged=None, only_in_band=True,
                       context=None):
    """Return list of D = NLL_treat − NLL_ctrl across cells at given lam."""
    D = []
    for c in cells:
        if context is not None and c["context"] != context:
            continue
        if only_in_band and not c.get("in_band", True):
            continue
        for rec in c["lambda_sweep"]:
            if abs(rec["lambda"] - lam) > 1e-9:
                continue
            if only_converged is True and not rec["control_converged"]:
                continue
            if only_converged is False and rec["control_converged"]:
                continue
            D.append(rec["D_diff"])
    return D


def primary_test(cells, lam, context=None):
    """Pre-registered one-sided Wilcoxon on converged=True, in-band cells."""
    D = collect_D_at_lambda(cells, lam, only_converged=True, context=context)
    n = len(D)
    if n < 2:
        return {"n": n, "median": None, "p_value": None,
                "verdict": "INSUFFICIENT_DATA"}
    median_D = float(np.median(D))
    # Wilcoxon one-sided (alternative='greater' tests H1: median > 0)
    try:
        w_stat, p = stats.wilcoxon(D, alternative="greater")
    except ValueError:
        # all zeros etc — treat as p=1
        return {"n": n, "median": median_D, "p_value": 1.0,
                "verdict": "DESCRIPTIVE"}
    p = float(p)
    if p >= ALPHA_PRIMARY:
        verdict = "DESCRIPTIVE"
    elif median_D > 0:
        verdict = "LOAD_BEARING"
    else:
        verdict = "ANOMALOUS_NEGATIVE"
    med, lo, hi = bootstrap_ci_median(D)
    return {
        "n": n,
        "median": median_D,
        "median_ci95_lo": lo,
        "median_ci95_hi": hi,
        "p_value": p,
        "verdict": verdict,
        "alpha": ALPHA_PRIMARY,
    }


def bias_check(cells, lam, context=None):
    """Pre-registered Mann-Whitney U: primary (converged) vs secondary (stalled)."""
    D_primary = collect_D_at_lambda(cells, lam, only_converged=True, context=context)
    D_secondary = collect_D_at_lambda(cells, lam, only_converged=False, context=context)
    n_p = len(D_primary)
    n_s = len(D_secondary)
    if n_p < 2 or n_s < 2:
        return {"n_primary": n_p, "n_secondary": n_s, "p_value": None,
                "verdict": "INSUFFICIENT_SECONDARY"}
    u_stat, p = stats.mannwhitneyu(D_primary, D_secondary, alternative="two-sided")
    p = float(p)
    return {
        "n_primary": n_p, "n_secondary": n_s,
        "median_primary": float(np.median(D_primary)),
        "median_secondary": float(np.median(D_secondary)),
        "u_statistic": float(u_stat),
        "p_value": p,
        "alpha": ALPHA_BIAS,
        "verdict": "GENERALIZABLE" if p >= ALPHA_BIAS else "CONDITIONAL",
    }


def dose_response(cells, context=None):
    """median(D) vs λ across the sweep, on converged primary cells."""
    lambdas = sorted({
        rec["lambda"]
        for c in cells for rec in c["lambda_sweep"]
        if rec["lambda"] > 0
    })
    out = []
    for lam in lambdas:
        D = collect_D_at_lambda(cells, lam, only_converged=True, context=context)
        if not D:
            continue
        med, lo, hi = bootstrap_ci_median(D)
        out.append({
            "lambda": lam,
            "n": len(D),
            "median_D": med,
            "ci95_lo": lo,
            "ci95_hi": hi,
            "mean_D": float(np.mean(D)),
        })
    # slope dD/dλ via simple linear regression on (lam, median_D)
    if len(out) >= 2:
        xs = np.array([o["lambda"] for o in out])
        ys = np.array([o["median_D"] for o in out])
        slope, intercept = np.polyfit(xs, ys, 1)
        return {"sweep": out, "slope_dD_dlambda": float(slope),
                "intercept": float(intercept)}
    return {"sweep": out, "slope_dD_dlambda": None, "intercept": None}


def alpha_fit_from_baseline(cells):
    """Fit H_N = α log N + β using per-cell baseline_H across contexts.

    Aggregate per-cell base_H by context, then fit (α, β) via least squares
    on (log N, mean_base_H).
    """
    by_ctx = defaultdict(list)
    for c in cells:
        # only in-band, primary signal — use all baseline_H though (descriptive fit)
        by_ctx[c["context"]].append(c["base_H_nats"])
    contexts = sorted(by_ctx.keys())
    mean_H = [float(np.mean(by_ctx[c])) for c in contexts]
    if len(contexts) < 2:
        return None
    log_N = np.log(np.array(contexts, dtype=float))
    H = np.array(mean_H, dtype=float)
    alpha, beta = np.polyfit(log_N, H, 1)
    return {
        "contexts": list(contexts),
        "mean_base_H": mean_H,
        "alpha": float(alpha),
        "beta": float(beta),
        "log_N_axis_used": [float(x) for x in log_N],
    }


def summarize_pool(cells):
    by_ctx = defaultdict(lambda: {"total": 0, "in_band": 0, "converged_any": 0})
    for c in cells:
        d = by_ctx[c["context"]]
        d["total"] += 1
        if c.get("in_band", True):
            d["in_band"] += 1
        # converged_any: at least one λ converged
        if any(rec["control_converged"] for rec in c["lambda_sweep"]
               if rec["lambda"] > 0):
            d["converged_any"] += 1
    return {str(k): v for k, v in sorted(by_ctx.items())}


def main():
    cells = load_cells()
    print(f"Loaded {len(cells)} cells")

    pool = summarize_pool(cells)
    print("\nPool by context:")
    for ctx, d in pool.items():
        print(f"  C={ctx}: total={d['total']}  in_band={d['in_band']}  "
              f"any_converged={d['converged_any']}")

    # Primary test at registered λ, pooled across contexts
    primary = primary_test(cells, REGISTERED_LAMBDA, context=None)
    bias = bias_check(cells, REGISTERED_LAMBDA, context=None)
    dose = dose_response(cells, context=None)
    alpha_fit = alpha_fit_from_baseline(cells)

    # Per-context view
    by_context = {}
    for ctx in sorted({c["context"] for c in cells}):
        by_context[str(ctx)] = {
            "primary": primary_test(cells, REGISTERED_LAMBDA, context=ctx),
            "bias": bias_check(cells, REGISTERED_LAMBDA, context=ctx),
            "dose_response": dose_response(cells, context=ctx),
        }

    summary = {
        "registered_lambda": REGISTERED_LAMBDA,
        "pool_summary_by_context": pool,
        "primary_pooled": primary,
        "bias_check_pooled": bias,
        "dose_response_pooled": dose,
        "alpha_fit_descriptive": alpha_fit,
    }
    with open(ANALYSIS_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    with open(ANALYSIS_BY_CTX_PATH, "w") as f:
        json.dump(by_context, f, indent=2)

    # --- Apply pre-registered verdict mapping mechanically ---
    print("\n=== PRE-REGISTERED PRIMARY TEST (pooled) ===")
    print(f"  registered λ = {REGISTERED_LAMBDA}")
    print(f"  n converged = {primary['n']}")
    if primary["median"] is not None:
        print(f"  median(D) = {primary['median']:+.6f} nats "
              f"[CI95: {primary['median_ci95_lo']:+.6f}, "
              f"{primary['median_ci95_hi']:+.6f}]")
        print(f"  Wilcoxon one-sided p = {primary['p_value']:.6f}")
    print(f"  VERDICT: {primary['verdict']}")

    print("\n=== PRE-REGISTERED BIAS CHECK ===")
    print(f"  n_primary = {bias.get('n_primary')}  n_secondary = {bias.get('n_secondary')}")
    if bias.get("p_value") is not None:
        print(f"  Mann-Whitney U p = {bias['p_value']:.6f}")
        print(f"  median primary = {bias['median_primary']:+.6f} ; "
              f"secondary = {bias['median_secondary']:+.6f}")
    print(f"  VERDICT: {bias.get('verdict')}")

    print("\n=== DOSE RESPONSE (pooled, converged primary cells) ===")
    for o in dose["sweep"]:
        print(f"  λ={o['lambda']:.2f}: median(D)={o['median_D']:+.4f}  "
              f"[CI95 {o['ci95_lo']:+.4f}, {o['ci95_hi']:+.4f}]  n={o['n']}")
    if dose["slope_dD_dlambda"] is not None:
        print(f"  slope dD/dλ = {dose['slope_dD_dlambda']:+.4f}")

    print("\n=== α-FIT (descriptive, H_N = α log N + β) ===")
    if alpha_fit:
        print(f"  α_trained = {alpha_fit['alpha']:+.4f}")
        print(f"  β = {alpha_fit['beta']:+.4f}")
        print(f"  contexts = {alpha_fit['contexts']}")

    print(f"\nSaved: {ANALYSIS_PATH}\n        {ANALYSIS_BY_CTX_PATH}")


if __name__ == "__main__":
    main()
