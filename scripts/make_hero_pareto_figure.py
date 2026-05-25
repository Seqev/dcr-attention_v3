"""
Phase 5.2 figure: hero/Pareto operating points on (c, ΔPPL) plane.

Sources:
  m1   = data/raw/phase_2_repro/m1_acceptance/m1_5seed_summary.json
  m4   = data/raw/phase_2_repro/m4_acceptance/m4_5seed_summary.json
  v2h  = data/raw/phase_2_repro/v2_hero/v2_hero_5seed_summary.json
  hero = data/raw/hero_verification/p1a_5seed_summary.json
  q7   = data/raw/pre_phase2/p0b_q7_stability.json

Output: paper/figures/hero_pareto.pdf
Paper-ready: serif font, figsize 6.0×4.0, tier bands, marker size scales with N.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path("/home/user/dcr-attention")
FIG_OUT = ROOT / "paper/figures/hero_pareto.pdf"


def load_pt(path, mean_key, std_key, c, N, label, is_summary=True):
    d = json.loads((ROOT / path).read_text())
    src = d.get("summary", d)
    return {
        "c": c,
        "mean": float(src[mean_key]),
        "std": float(src[std_key]),
        "N": N,
        "label": label,
    }


def main():
    pts = [
        load_pt("data/raw/phase_2_repro/m1_acceptance/m1_5seed_summary.json",
                "mean_delta_pct", "std_delta_pct",
                c=0.5, N=2000, label="M1 acceptance"),
        load_pt("data/raw/phase_2_repro/m4_acceptance/m4_5seed_summary.json",
                "mean_delta_pct", "std_delta_pct",
                c=0.5, N=2000, label="M4 acceptance"),
        load_pt("data/raw/phase_2_repro/v2_hero/v2_hero_5seed_summary.json",
                "mean_delta_pct", "std_delta_pct",
                c=0.5, N=20000, label="v2.0 hero re-val"),
        load_pt("data/raw/hero_verification/p1a_5seed_summary.json",
                "mean_delta_pct", "std_delta_pct",
                c=0.15, N=32000, label="HERO (primary)"),
        load_pt("data/raw/pre_phase2/p0b_q7_stability.json",
                "mean_delta_pct", "std_delta_pct",
                c=0.10, N=32000, label="Q7 secondary (3-seed)"),
    ]

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(6.0, 4.0))

    # Tier bands (shaded backgrounds + dashed threshold lines)
    ax.axhspan(-0.2, 0.5, facecolor="#c5e1c5", alpha=0.35, zorder=0)
    ax.axhspan(0.5, 1.0, facecolor="#fde9c5", alpha=0.45, zorder=0)
    ax.axhspan(1.0, 2.0, facecolor="#f5c5c5", alpha=0.40, zorder=0)
    ax.axhline(0.5, ls="--", color="#2a7", lw=0.8, alpha=0.9, zorder=1)
    ax.axhline(1.0, ls="--", color="#c80", lw=0.8, alpha=0.9, zorder=1)
    ax.axhline(2.0, ls="--", color="#a33", lw=0.8, alpha=0.9, zorder=1)
    # tier labels at right edge — empty zone above the c=0.5 data cluster
    ax.text(0.58, 0.45, "STRICT", color="#176", fontsize=8,
            va="top", ha="right", weight="bold")
    ax.text(0.58, 0.95, "STANDARD", color="#a60", fontsize=8,
            va="top", ha="right", weight="bold")
    ax.text(0.58, 1.55, "PERMISSIVE", color="#822", fontsize=8,
            va="top", ha="right", weight="bold")

    # Marker size scales with N
    size_map = {2000: 70, 20000: 130, 32000: 210}
    color_map = {2000: "#4c72b0", 20000: "#dd8452", 32000: "#55a868"}

    for pt in pts:
        ax.errorbar(
            pt["c"], pt["mean"],
            yerr=pt["std"],
            fmt="o", capsize=4,
            markersize=(size_map[pt["N"]]) ** 0.5,
            markeredgecolor="black", markeredgewidth=0.5,
            color=color_map[pt["N"]],
            label=f"{pt['label']}  (N={pt['N']//1000}K)",
            zorder=10,
        )

    ax.set_xlabel(r"Coverage floor $c$")
    ax.set_ylabel(r"$\Delta\mathrm{PPL}$  [%]")
    ax.set_xlim(0.05, 0.60)
    ax.set_ylim(-0.2, 1.6)
    ax.set_xticks([0.10, 0.15, 0.30, 0.50])
    ax.grid(True, alpha=0.20, linestyle=":")
    # legend in the empty middle gap (between c=0.15 HERO and c=0.5 cluster)
    ax.legend(loc="center", framealpha=0.95, ncol=1, fontsize=8,
              bbox_to_anchor=(0.55, 0.70))
    plt.tight_layout(pad=0.5)
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_OUT, bbox_inches="tight")
    print(f"Saved: {FIG_OUT}")
    for pt in pts:
        print(f"  c={pt['c']:.2f} N={pt['N']:>6} mean={pt['mean']:+.4f} "
              f"std={pt['std']:.4f} -- {pt['label']}")


if __name__ == "__main__":
    main()
