"""
Phase 5.3 figure: M4 speedup landscape vs SDPA across (N, B, c).

Sources:
  Phase HERO Phase B at c=0.15: data/raw/hero_verification/latency_c015_c030.json
                                  (rows L1-L4 = c=0.15; L5 = c=0.30 control)
  Phase 1.5 P0a at c=0.10: HERO_VERIFICATION_REPORT.md  (comparison-only points:
                          N=8K B=4 → 1.12×; N=32K B=1 → 0.995×)

Output: paper/figures/speedup_landscape.pdf

Visualization: speedup vs N, one curve per (c, B) regime.
Parity line at 1.0× clearly marked.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path("/home/user/dcr-attention")
FIG_OUT = ROOT / "paper/figures/speedup_landscape.pdf"


def main():
    hero = json.loads(
        (ROOT / "data/raw/hero_verification/latency_c015_c030.json").read_text()
    )
    # rows L1-L4 at c=0.15, L5 at c=0.30
    c015 = {(r["N"], r["batch"]): r["speedup_vs_sdpa"]
            for r in hero if abs(r["c_floor"] - 0.15) < 1e-9}
    c030 = {(r["N"], r["batch"]): r["speedup_vs_sdpa"]
            for r in hero if abs(r["c_floor"] - 0.30) < 1e-9}
    # Phase 1.5 P0a c=0.10 — comparison-only points (from HERO_VERIFICATION_REPORT)
    c010 = {(8000, 4): 1.12, (32000, 1): 0.995}

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(6.0, 3.8))

    # parity band
    ax.axhspan(0.95, 1.05, facecolor="#dddddd", alpha=0.4,
               zorder=0, label=None)
    ax.axhline(1.0, ls="--", color="gray", lw=0.8, alpha=0.9, zorder=1)
    ax.text(6000, 1.03, "parity (1.0×)", color="dimgray", fontsize=8,
            ha="left", va="bottom")

    # plot points; group by (c, B)
    series = [
        ("c=0.10, B=1", c010, 1, "o", "#1f77b4"),
        ("c=0.10, B=4", c010, 4, "s", "#1f77b4"),
        ("c=0.15, B=1", c015, 1, "o", "#d62728"),
        ("c=0.15, B=4", c015, 4, "s", "#d62728"),
        ("c=0.30, B=1", c030, 1, "o", "#2ca02c"),
    ]

    for label, table, B, marker, color in series:
        Ns = sorted(N for (N, b) in table.keys() if b == B)
        ys = [table[(N, B)] for N in Ns]
        if len(Ns) >= 2:
            ax.plot(Ns, ys, marker=marker, color=color, linewidth=1.3,
                    markersize=7, markeredgecolor="black",
                    markeredgewidth=0.5, label=label)
        elif len(Ns) == 1:
            ax.plot(Ns, ys, marker=marker, color=color, linewidth=0,
                    markersize=8, markeredgecolor="black",
                    markeredgewidth=0.5, label=label)

    # Annotate the L4 deciding point — text in upper-left zone (data-free)
    ax.annotate(
        "L4 (HeroQualityOnly):\n$N{=}32K,B{=}4,c{=}0.15$\n$0.895\\times$",
        xy=(32000, 0.895), xytext=(8500, 0.66),
        fontsize=8, ha="left", va="top",
        arrowprops=dict(arrowstyle="->", color="black", lw=0.8,
                        connectionstyle="arc3,rad=0.2"),
        bbox=dict(boxstyle="round", fc="#fff5e8", ec="#d62728", alpha=0.95),
    )

    ax.set_xlabel(r"Context length $N$")
    ax.set_ylabel(r"M4 / SDPA speedup")
    ax.set_xscale("log")
    # explicit ticks, no minor tick labels
    ax.set_xticks([8000, 32000])
    ax.set_xticklabels(["8K", "32K"])
    ax.set_xticks([], minor=True)
    ax.set_xlim(5500, 48000)
    ax.set_ylim(0.55, 1.25)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=8,
              bbox_to_anchor=(0.99, 0.99))
    ax.grid(True, alpha=0.25, linestyle=":")
    plt.tight_layout(pad=0.5)
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_OUT, bbox_inches="tight")
    print(f"Saved: {FIG_OUT}")
    for label, table, B, _, _ in series:
        for (N, b), v in sorted(table.items()):
            if b == B:
                print(f"  {label}: N={N} -> {v}")


if __name__ == "__main__":
    main()
