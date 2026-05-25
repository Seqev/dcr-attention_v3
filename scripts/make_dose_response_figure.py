"""
Phase 5.1 figure: dose-response D vs λ.

Source: data/raw/phase_3_causal_full/analysis.json → dose_response_pooled.sweep
Output: paper/figures/dose_response.pdf  (paper-ready, single-panel)

Architect-revised layout (2026-05-21):
  - figsize 6.0 × 3.8 (was 5.0 × 3.3 — clipped)
  - no in-figure title; LaTeX caption does that work
  - no sub-caption with per-λ n; those go in tab:causal-result
  - short y-label: median(D) [nats]; full formula lives in caption
  - tight_layout(pad=0.5) to prevent overflow
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ANALYSIS = Path("/home/user/dcr-attention/data/raw/phase_3_causal_full/analysis.json")
FIG_OUT = Path("/home/user/dcr-attention/paper/figures/dose_response.pdf")


def main():
    a = json.loads(ANALYSIS.read_text())
    sweep = a["dose_response_pooled"]["sweep"]
    slope = a["dose_response_pooled"]["slope_dD_dlambda"]

    lams = [s["lambda"] for s in sweep]
    med = [s["median_D"] for s in sweep]
    lo = [s["ci95_lo"] for s in sweep]
    hi = [s["ci95_hi"] for s in sweep]
    err_lo = [max(m - l, 0.0) for m, l in zip(med, lo)]
    err_hi = [max(h - m, 0.0) for h, m in zip(hi, med)]

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "pdf.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.errorbar(
        lams, med,
        yerr=[err_lo, err_hi],
        fmt="o-", color="C0", capsize=4, markersize=5,
        linewidth=1.2, elinewidth=1.0,
        label="median$(D)$ with 95% bootstrap CI",
    )
    ax.axhline(0, ls="--", color="gray", lw=0.8, label=r"$D=0$ (null)")
    ax.set_xlabel(r"$\lambda$  (treatment dose, interpolation toward uniform)")
    ax.set_ylabel(r"median$(D)$  [nats]")
    ax.text(
        0.05, 0.08,
        fr"slope $dD/d\lambda = {slope:.2e}$",
        transform=ax.transAxes, fontsize=9,
        bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9),
    )
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.25, linestyle=":")
    plt.tight_layout(pad=0.5)
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_OUT, bbox_inches="tight")
    print(f"Saved: {FIG_OUT}")
    print(f"  slope dD/dλ = {slope:.4e}")
    print(f"  median(D) range: [{min(med):.2e}, {max(med):.2e}]")


if __name__ == "__main__":
    main()
