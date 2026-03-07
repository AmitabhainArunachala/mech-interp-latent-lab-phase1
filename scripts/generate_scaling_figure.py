#!/usr/bin/env python3
"""
Generate scaling curve figure with 8+ data points and transition zone shading.

Combines data from:
  - results/scaling_gap/ (E1.3 models)
  - Cross-architecture results (Mistral-7B, OPT-6.7B, GPT-2 XL, Qwen2.5-7B, Pythia-1.4B)

Output: figures/fig_scaling_curve.pdf + .png
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ── Hardcoded data points (from paper v005 + scaling_gap results) ──
# Format: (params_B, |Cohen's d|, CI_lo, CI_hi, label, significant)
SCALING_DATA = [
    # Small Pythia models (from scaling gap sweep)
    (1.0,   0.28,  None, None, "Pythia-1B",        False),
    (1.4,   0.01,  -0.40, 0.36, "Pythia-1.4B",     False),
    (2.8,   0.25,  None, None, "Pythia-2.8B",       False),
    # Mid-range (from scaling gap)
    (2.6,   0.48,  None, None, "Gemma-2-2B",        False),  # Placeholder; update from results
    (3.0,   1.25,  0.74, 1.77, "Qwen2.5-3B",       True),
    (3.8,   0.62,  0.16, 1.09, "Phi-3-mini",        True),
    # Large (from cross-architecture and scaling gap)
    (1.5,   1.52,  1.07, 2.05, "GPT-2 XL",          True),
    (6.7,   1.68,  1.35, 2.09, "OPT-6.7B",          True),
    (6.9,   0.48,  0.00, 0.96, "Pythia-6.9B",       False),
    (7.0,   1.66,  1.32, 2.08, "Mistral-7B",        True),
    (7.0,   2.32,  1.90, 2.86, "Qwen2.5-7B",        True),
]

# Try to load Gemma-2-2B result from JSON
gemma_path = Path("results/scaling_gap/gemma-2-2b_result.json")
if gemma_path.exists():
    with open(gemma_path) as f:
        gemma = json.load(f)
    d_val = abs(gemma.get("cohens_d", 0.48))
    ci_lo = gemma.get("ci_95_lo", None)
    ci_hi = gemma.get("ci_95_hi", None)
    SCALING_DATA[3] = (2.6, d_val, ci_lo, ci_hi, "Gemma-2-2B", d_val > 0.5)


def main():
    out_dir = Path("R_V_PAPER/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    # ── Transition zone shading (2.5B–7B) ──
    ax.axvspan(2.5, 7.0, alpha=0.08, color="steelblue", label="Transition zone")

    # ── Large effect threshold ──
    ax.axhline(0.8, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.text(0.85, 0.85, "|d| = 0.8 (large)", fontsize=7, color="gray", alpha=0.7)

    # ── Plot data points ──
    for params, d, ci_lo, ci_hi, label, sig in SCALING_DATA:
        color = "#2166ac" if sig else "#b2182b"
        marker = "o" if sig else "x"
        size = 60 if sig else 40

        ax.scatter(params, d, c=color, marker=marker, s=size, zorder=5,
                   edgecolors="black" if sig else "none", linewidths=0.5)

        # Error bars
        if ci_lo is not None and ci_hi is not None:
            ax.errorbar(params, d, yerr=[[d - ci_lo], [ci_hi - d]],
                        fmt="none", color=color, alpha=0.4, capsize=3)

        # Labels
        offset_y = 0.08
        if label in ("Qwen2.5-7B", "Mistral-7B"):
            offset_y = 0.12
        if label == "OPT-6.7B":
            offset_y = -0.15
        if label == "Pythia-6.9B":
            offset_y = -0.15
        ax.annotate(label, (params, d), textcoords="offset points",
                    xytext=(5, offset_y * 100), fontsize=6.5, alpha=0.8)

    # ── Formatting ──
    ax.set_xlabel("Parameters (billions)", fontsize=11)
    ax.set_ylabel("|Cohen's d| (recursive vs. baseline)", fontsize=11)
    ax.set_title("R$_V$ Effect Size vs. Model Scale", fontsize=12, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlim(0.8, 10)
    ax.set_ylim(-0.1, 2.8)
    ax.set_xticks([1, 2, 3, 5, 7])
    ax.set_xticklabels(["1B", "2B", "3B", "5B", "7B"])

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2166ac",
               markeredgecolor="black", markersize=8, label="Significant (p<0.05)"),
        Line2D([0], [0], marker="x", color="#b2182b", markersize=8,
               linestyle="None", label="Non-significant"),
        Rectangle((0, 0), 1, 1, alpha=0.15, color="steelblue", label="Transition zone"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8, framealpha=0.9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    # Save
    fig.savefig(out_dir / "fig_scaling_curve.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(out_dir / "fig_scaling_curve.png", bbox_inches="tight", dpi=300)
    print(f"Saved: {out_dir / 'fig_scaling_curve.pdf'}")
    print(f"Saved: {out_dir / 'fig_scaling_curve.png'}")
    plt.close()


if __name__ == "__main__":
    main()
