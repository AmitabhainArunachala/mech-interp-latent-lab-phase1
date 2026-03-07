#!/usr/bin/env python3
"""
generate_masterplan_figures.py — Publication figures for R_V Master Plan experiments.

Generates new figures from E1.3, E1.4, E2.2, E5, and FDR results.
Does NOT require GPU — reads from local JSON results.

Usage:
    python3 scripts/generate_masterplan_figures.py [--output-dir figures/masterplan]
"""

import argparse
import json
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# ── NeurIPS style ──
NEURIPS_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.2,
    "lines.markersize": 5,
    "axes.grid": False,
}
plt.rcParams.update(NEURIPS_RC)

C_SELF = "#D32F2F"
C_BASE = "#1976D2"
C_OTHER = "#757575"
C_HIGHLIGHT = "#FF9800"
C_POSITIVE = "#4CAF50"
C_NEGATIVE = "#F44336"
C_14B = "#E91E63"
C_28B = "#9C27B0"

RESULTS = Path(__file__).parent.parent / "results"


def load_json(path):
    with open(path) as f:
        content = f.read().strip()
        if not content:
            return None
        content = content.replace(": Infinity", ": 1e308")
        content = content.replace(": -Infinity", ": -1e308")
        content = content.replace(": NaN", ": null")
        return json.loads(content)


def save_fig(fig, name, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)
    print(f"  ✓ {name}")


# ===================================================================
# FIGURE: 32×32 Full Head Sweep Heatmap (E2.2)
# ===================================================================
def fig_full_head_sweep(output_dir):
    """32×32 heatmap of entropy Cohen's d for all Mistral-7B heads."""
    sweep_files = sorted(glob.glob(str(RESULTS / "full_head_sweep/full_head_sweep_*.json")))
    if not sweep_files:
        print("  ✗ full_head_sweep data not found")
        return

    data = load_json(sweep_files[-1])
    if not data:
        print("  ✗ full_head_sweep data empty")
        return

    heads = data["head_results"]
    n_layers = 32
    n_heads = 32
    grid = np.full((n_layers, n_heads), np.nan)

    for h in heads:
        l, hd = h["layer"], h["head"]
        d = h.get("entropy_d")
        if d is not None and l < n_layers and hd < n_heads:
            grid[l, hd] = d

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(grid, cmap="RdBu_r", vmin=-4, vmax=4, aspect="auto",
                   interpolation="nearest")

    # Mark top heads
    top_heads = sorted(heads, key=lambda h: abs(h.get("entropy_d", 0)), reverse=True)[:10]
    for h in top_heads:
        l, hd = h["layer"], h["head"]
        d = h.get("entropy_d", 0)
        ax.plot(hd, l, "k*", markersize=8 if abs(d) > 3 else 5)

    ax.set_xlabel("Head Index")
    ax.set_ylabel("Layer Index")
    ax.set_title("Self-Referential Circuit: Attention Entropy Divergence\n"
                 "(Mistral-7B, 1024 heads, Cohen's $d$: recursive vs baseline)")
    ax.set_xticks(range(0, n_heads, 4))
    ax.set_yticks(range(0, n_layers, 4))

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Cohen's $d$ (entropy divergence)")

    # Annotate key regions
    ax.axhline(5, color="white", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.axhline(27, color="white", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.text(n_heads + 0.5, 5, "L5 (early)", fontsize=7, va="center", color=C_SELF)
    ax.text(n_heads + 0.5, 27, "L27 (late)", fontsize=7, va="center", color=C_BASE)

    fig.tight_layout()
    save_fig(fig, "fig_full_head_sweep_32x32", output_dir)


# ===================================================================
# FIGURE: Scaling Curve (E1.3 + existing scaling data)
# ===================================================================
def fig_scaling_curve(output_dir):
    """Combined scaling curve with all data points."""
    # Collect all model results
    models = []

    # From scaling_gap E1.3 metrics
    metrics_path = RESULTS / "rv_masterplan/E1.3_scaling_gap/metrics.json"
    if metrics_path.exists():
        data = load_json(metrics_path)
        if data:
            for model, vals in data.get("models_completed", {}).items():
                if vals.get("cohens_d") is not None:
                    models.append({
                        "name": model,
                        "params": vals["params"],
                        "d": vals["cohens_d"],
                        "p": vals.get("p_value"),
                        "n": vals.get("n_recursive", 0) + vals.get("n_baseline", 0),
                        "source": "scaling_gap",
                    })

    # From power_up E1.1
    for f in sorted((RESULTS / "power_up").glob("*_result.json")):
        data = load_json(f)
        if data and data.get("cohens_d") is not None:
            name = data["model"]
            # Skip if already in scaling_gap with same model
            if any(m["name"] == name for m in models):
                continue
            params_map = {
                "gpt2-xl": 1_500_000_000,
                "mistral-7b": 7_000_000_000,
                "opt-6.7b": 6_700_000_000,
                "qwen2.5-7b": 7_000_000_000,
            }
            models.append({
                "name": name,
                "params": params_map.get(name, 0),
                "d": data["cohens_d"],
                "p": data.get("p_value"),
                "n": data.get("n_recursive", 0) + data.get("n_baseline", 0),
                "source": "power_up",
            })

    # From scaling_gap individual results
    for f in sorted((RESULTS / "scaling_gap").glob("*_result.json")):
        data = load_json(f)
        if data and data.get("cohens_d") is not None:
            name = data.get("model", f.stem.replace("_result", ""))
            if any(m["name"] == name for m in models):
                continue
            models.append({
                "name": name,
                "params": data.get("params", 0),
                "d": data["cohens_d"],
                "p": data.get("p_value"),
                "n": data.get("n_recursive", 0) + data.get("n_baseline", 0),
                "source": "scaling_gap_individual",
            })

    if not models:
        print("  ✗ No scaling data found")
        return

    # Sort by params
    models.sort(key=lambda m: m["params"])

    # Filter out zeros and duplicates
    models = [m for m in models if m["params"] > 0]

    fig, ax = plt.subplots(figsize=(6, 4))

    params = [m["params"] / 1e9 for m in models]
    ds = [abs(m["d"]) for m in models]
    sig = [m.get("p", 1) < 0.05 for m in models]

    # Plot points
    for i, m in enumerate(models):
        color = C_SELF if sig[i] else C_OTHER
        marker = "D" if sig[i] else "o"
        ax.scatter(params[i], abs(m["d"]), c=color, s=80, marker=marker,
                   edgecolors="white", linewidth=0.5, zorder=3)
        ax.annotate(m["name"], (params[i], abs(m["d"])),
                    fontsize=6, xytext=(5, 5), textcoords="offset points",
                    color=color)

    # Significance threshold
    ax.axhline(0.8, color="grey", linestyle=":", linewidth=0.5, alpha=0.7,
               label="Large effect (|d|=0.8)")
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.5, alpha=0.5,
               label="Medium effect (|d|=0.5)")

    # Transition zone shading
    ax.axvspan(2.5, 4.0, color=C_HIGHLIGHT, alpha=0.1, label="Transition zone")

    ax.set_xscale("log")
    ax.set_xlabel("Parameters (B)")
    ax.set_ylabel("|Cohen's $d$| (recursive vs baseline)")
    ax.set_title("R$_V$ Effect Size vs Model Scale\n"
                 f"({len(models)} architectures, ★ = p < 0.05)")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_xlim(0.3, 15)

    fig.tight_layout()
    save_fig(fig, "fig_scaling_curve", output_dir)


# ===================================================================
# FIGURE: Training Checkpoint Emergence (E1.4)
# ===================================================================
def fig_training_checkpoints(output_dir):
    """R_V emergence during training for Pythia-1.4B and 2.8B."""
    steps_14b = []
    steps_28b = []

    for f in sorted((RESULTS / "training_checkpoints").glob("pythia-1.4b_step*_result.json")):
        data = load_json(f)
        if data and data.get("cohens_d") is not None:
            steps_14b.append({
                "step": data["step"],
                "d": data["cohens_d"],
                "rv_rec": data.get("rv_recursive_mean"),
                "rv_bas": data.get("rv_baseline_mean"),
                "p": data.get("p_value"),
            })

    for f in sorted((RESULTS / "training_checkpoints").glob("pythia-2.8b_step*_result.json")):
        data = load_json(f)
        if data and data.get("cohens_d") is not None:
            steps_28b.append({
                "step": data["step"],
                "d": data["cohens_d"],
                "rv_rec": data.get("rv_recursive_mean"),
                "rv_bas": data.get("rv_baseline_mean"),
                "p": data.get("p_value"),
            })

    if not steps_14b and not steps_28b:
        print("  ✗ Training checkpoint data not found")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5), sharey=True)

    # Pythia-1.4B
    if steps_14b:
        steps_14b.sort(key=lambda s: s["step"])
        x = [s["step"] / 1000 for s in steps_14b]
        d = [s["d"] for s in steps_14b]
        ax1.plot(x, d, "o-", color=C_14B, markersize=6, linewidth=1.5)
        for s in steps_14b:
            if s["p"] and s["p"] < 0.05:
                ax1.plot(s["step"] / 1000, s["d"], "o", color=C_14B,
                         markersize=8, markeredgecolor="black", markeredgewidth=1)
        ax1.set_xlabel("Training Step (×1000)")
        ax1.set_ylabel("Cohen's $d$ (recursive vs baseline)")
        ax1.set_title("Pythia-1.4B")
        ax1.axhline(0, color="grey", linestyle="-", linewidth=0.5)

    # Pythia-2.8B
    if steps_28b:
        steps_28b.sort(key=lambda s: s["step"])
        x = [s["step"] / 1000 for s in steps_28b]
        d = [s["d"] for s in steps_28b]

        # Check for the bug: are all d values identical?
        unique_d = len(set(round(di, 4) for di in d))
        if unique_d == 1 and len(d) > 1:
            ax2.text(0.5, 0.5, "⚠ BUG DETECTED\nAll checkpoints identical\n"
                     f"d = {d[0]:.3f}\n(cached weights, not actual checkpoints)",
                     ha="center", va="center", transform=ax2.transAxes,
                     fontsize=8, color="red", fontweight="bold",
                     bbox=dict(boxstyle="round", facecolor="lightyellow"))
        else:
            ax2.plot(x, d, "o-", color=C_28B, markersize=6, linewidth=1.5)
            for s in steps_28b:
                if s["p"] and s["p"] < 0.05:
                    ax2.plot(s["step"] / 1000, s["d"], "o", color=C_28B,
                             markersize=8, markeredgecolor="black", markeredgewidth=1)
        ax2.set_xlabel("Training Step (×1000)")
        ax2.set_title("Pythia-2.8B")
        ax2.axhline(0, color="grey", linestyle="-", linewidth=0.5)

    fig.suptitle("R$_V$ Self-Referential Specificity During Training", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_fig(fig, "fig_training_checkpoints", output_dir)


# ===================================================================
# FIGURE: Safety ROC Curve (E5.3)
# ===================================================================
def fig_safety_roc(output_dir):
    """ROC curve for R_V-based self-referential detection."""
    safety_files = sorted(glob.glob(str(RESULTS / "safety/safety_analysis_*.json")))
    if not safety_files:
        print("  ✗ Safety data not found")
        return

    data = load_json(safety_files[-1])
    if not data:
        print("  ✗ Safety data empty")
        return

    e53 = data.get("e53_deployment_monitoring", {})
    roc = e53.get("roc_curve", [])
    auroc = e53.get("auroc", 0)
    best_tpr = e53.get("best_tpr", 0)
    best_fpr = e53.get("best_fpr", 0)

    if not roc:
        print("  ✗ No ROC data")
        return

    fprs = [p["fpr"] for p in roc]
    tprs = [p["tpr"] for p in roc]

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(fprs, tprs, color=C_SELF, linewidth=2, label=f"R$_V$ detector (AUROC={auroc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.5, label="Random")
    ax.plot(best_fpr, best_tpr, "D", color=C_HIGHLIGHT, markersize=8,
            label=f"Best threshold (TPR={best_tpr:.2f}, FPR={best_fpr:.2f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("R$_V$-Based Self-Referential Processing Detector\n"
                 f"(500 prompts, 50 self-ref seeded)")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")

    fig.tight_layout()
    save_fig(fig, "fig_safety_roc", output_dir)


# ===================================================================
# FIGURE: Safety — Genuine vs Deceptive Self-Reference (E5.1)
# ===================================================================
def fig_safety_genuine_vs_deceptive(output_dir):
    """Bar chart comparing R_V for genuine, deceptive, alignment-faking self-reference."""
    safety_files = sorted(glob.glob(str(RESULTS / "safety/safety_analysis_*.json")))
    if not safety_files:
        print("  ✗ Safety data not found")
        return

    data = load_json(safety_files[-1])
    if not data:
        return

    e51 = data.get("e51_genuine_vs_deceptive", {})
    e52 = data.get("e52_alignment_faking", {})

    categories = ["Baseline", "Genuine\nSelf-Ref", "Deceptive\nSelf-Ref", "Alignment\nFaking"]
    means = [
        e51.get("baseline_rv_mean", 0),
        e51.get("genuine_rv_mean", 0),
        e51.get("deceptive_rv_mean", 0),
        e52.get("faking_rv_mean", 0),
    ]
    stds = [
        e51.get("baseline_rv_std", 0),
        e51.get("genuine_rv_std", 0),
        e51.get("deceptive_rv_std", 0),
        e52.get("faking_rv_std", 0),
    ]
    colors = [C_BASE, C_SELF, C_HIGHLIGHT, C_28B]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.bar(range(len(categories)), means, yerr=stds, color=colors,
                  edgecolor="white", linewidth=0.5, capsize=4,
                  error_kw={"linewidth": 0.8})

    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, fontsize=8)
    ax.set_ylabel("Mean R$_V$")
    ax.set_title("R$_V$ Tracks Content, Not Intent\n"
                 "(Genuine ≈ Deceptive self-reference, both < Baseline)")

    # Add effect size annotations
    ax.annotate(f"d = {e51.get('d_genuine_vs_deceptive', 0):.2f}\n(NS)",
                xy=(1.5, max(means[1], means[2]) + 0.01),
                fontsize=7, ha="center", color=C_OTHER)
    ax.annotate(f"d = {e51.get('d_genuine_vs_baseline', 0):.2f}***",
                xy=(0.5, means[0] - 0.02),
                fontsize=7, ha="center", color=C_SELF)

    fig.tight_layout()
    save_fig(fig, "fig_safety_genuine_vs_deceptive", output_dir)


# ===================================================================
# FIGURE: FDR Correction Summary
# ===================================================================
def fig_fdr_summary(output_dir):
    """Dot plot of p-values before/after FDR correction."""
    fdr_files = sorted(glob.glob(str(RESULTS / "fdr_correction/fdr_results_*.json")))
    if not fdr_files:
        print("  ✗ FDR data not found — run scripts/fdr_correction.py first")
        return

    data = load_json(fdr_files[-1])
    if not data:
        return

    tests = data["tests"]
    # Sort by p_original
    tests.sort(key=lambda t: t["p_original"])

    fig, ax = plt.subplots(figsize=(6, 5))

    for i, t in enumerate(tests):
        p = t["p_original"]
        q = t["q_value"]
        sig = t["significant_fdr"]
        color = C_POSITIVE if sig else C_NEGATIVE

        # Plot original p
        ax.plot(-np.log10(max(p, 1e-300)), i, "o", color=color, markersize=5)
        # Plot q-value
        ax.plot(-np.log10(max(q, 1e-300)), i, "s", color=color, markersize=4, alpha=0.5)
        # Connect with line
        ax.plot([-np.log10(max(p, 1e-300)), -np.log10(max(q, 1e-300))],
                [i, i], "-", color=color, linewidth=0.5, alpha=0.5)

    # Significance threshold
    ax.axvline(-np.log10(0.05), color="grey", linestyle="--", linewidth=0.8,
               label="p = 0.05")

    labels = [f"{t['source'].split('_')[0]}:{t['test'][:30]}" for t in tests]
    ax.set_yticks(range(len(tests)))
    ax.set_yticklabels(labels, fontsize=5)
    ax.set_xlabel("$-\\log_{10}(p)$")
    ax.set_title(f"FDR Correction: {data['n_significant_fdr']}/{data['n_tests']} "
                 f"tests survive (BH, α=0.05)\n"
                 f"○ = original p, □ = BH q-value")
    ax.legend(fontsize=7)
    ax.invert_yaxis()

    fig.tight_layout()
    save_fig(fig, "fig_fdr_correction", output_dir)


# ===================================================================
# FIGURE: Updated Cross-Architecture Forest Plot (with power-up data)
# ===================================================================
def fig_cross_arch_updated(output_dir):
    """Forest plot with ALL architectures including power-up and scaling gap."""
    models = []

    # Collect from power_up
    for f in sorted((RESULTS / "power_up").glob("*_result.json")):
        data = load_json(f)
        if data and data.get("cohens_d") is not None:
            models.append({
                "name": data["model"],
                "d": data["cohens_d"],
                "n1": data.get("n_recursive", 0),
                "n2": data.get("n_baseline", 0),
                "p": data.get("p_value"),
                "source": "E1.1",
            })

    # Collect from scaling_gap
    metrics_path = RESULTS / "rv_masterplan/E1.3_scaling_gap/metrics.json"
    if metrics_path.exists():
        data = load_json(metrics_path)
        if data:
            for model, vals in data.get("models_completed", {}).items():
                if vals.get("cohens_d") is not None:
                    if not any(m["name"] == model for m in models):
                        models.append({
                            "name": model,
                            "d": vals["cohens_d"],
                            "n1": vals.get("n_recursive", 0),
                            "n2": vals.get("n_baseline", 0),
                            "p": vals.get("p_value"),
                            "source": "E1.3",
                        })

    if not models:
        print("  ✗ No cross-architecture data found")
        return

    # Sort by |d|
    models.sort(key=lambda m: abs(m["d"]), reverse=True)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for i, m in enumerate(models):
        d = m["d"]
        n = m["n1"] + m["n2"]
        se = np.sqrt(2 / max(n, 2) + d ** 2 / (2 * max(n, 2)))
        ci_lo = d - 1.96 * se
        ci_hi = d + 1.96 * se
        sig = m.get("p", 1) < 0.05
        color = C_SELF if sig else C_OTHER

        ax.errorbar(d, i, xerr=[[d - ci_lo], [ci_hi - d]],
                    fmt="o" if sig else "x", color=color, markersize=6,
                    capsize=3, capthick=0.8, linewidth=0.8)
        label = f"{m['name']} (n={m['n1']}+{m['n2']})"
        ax.text(ci_hi + 0.05, i, label, fontsize=6, va="center", color=color)

    ax.axvline(0, color="black", linestyle="-", linewidth=0.6)
    ax.axvline(-0.8, color="grey", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.axvline(0.8, color="grey", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.set_yticks([])
    ax.set_xlabel("Cohen's $d$ (recursive vs baseline R$_V$)")
    ax.set_title(f"Cross-Architecture Forest Plot ({len(models)} models)")
    ax.invert_yaxis()

    fig.tight_layout()
    save_fig(fig, "fig_cross_arch_updated", output_dir)


# ===================================================================
# MAIN
# ===================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="figures/masterplan")
    args = parser.parse_args()

    out = Path(args.output_dir)
    print(f"Generating masterplan figures → {out}/\n")

    fig_full_head_sweep(out)
    fig_scaling_curve(out)
    fig_training_checkpoints(out)
    fig_safety_roc(out)
    fig_safety_genuine_vs_deceptive(out)
    fig_fdr_summary(out)
    fig_cross_arch_updated(out)

    print(f"\nDone! All figures saved to {out}/")


if __name__ == "__main__":
    main()
