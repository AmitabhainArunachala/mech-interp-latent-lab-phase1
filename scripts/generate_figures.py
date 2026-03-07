#!/usr/bin/env python3
"""
generate_figures.py — Publication-quality figures for NeurIPS 2026 paper.

Reads result JSONs from experiments and produces 12+ matplotlib figures
in paper/figures/ directory.

Usage:
    python scripts/generate_figures.py [--output-dir paper/figures]
"""

import argparse
import json
import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec
from pathlib import Path

# ---------------------------------------------------------------------------
# NeurIPS 2026 house style
# ---------------------------------------------------------------------------
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

# Colour palette
C_SELF = "#D32F2F"       # red for self-referential
C_BASE = "#1976D2"       # blue for baseline
C_OTHER = "#757575"      # grey for other modes
C_HIGHLIGHT = "#FF9800"  # orange accent
C_POSITIVE = "#4CAF50"   # green
C_NEGATIVE = "#F44336"   # red

RESULTS = Path("results")
PAPER_FIG = Path("paper/figures")


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_fig(fig, name, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)
    print(f"  ✓ {name}")


# ===================================================================
# FIGURE 1: Mode Atlas — R_V spectral fingerprint (hero figure)
# ===================================================================
def fig_mode_atlas_bar(output_dir):
    """Bar chart of mean R_V across 10 computational modes."""
    atlas_files = sorted(glob.glob(str(RESULTS / "mode_atlas/atlas_summary_*.json")))
    if not atlas_files:
        print("  ✗ mode atlas data not found")
        return
    data = load_json(atlas_files[-1])
    fp = data["fingerprint"]

    modes = []
    means = []
    stds = []
    for mode, vals in fp.items():
        modes.append(mode.replace("_", " ").title())
        means.append(vals["rv"]["mean"])
        stds.append(vals["rv"]["std"])

    # Sort by mean R_V
    idx = np.argsort(means)
    modes = [modes[i] for i in idx]
    means = [means[i] for i in idx]
    stds = [stds[i] for i in idx]

    colors = [C_SELF if "Self" in m else C_OTHER for m in modes]

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    bars = ax.barh(range(len(modes)), means, xerr=stds, height=0.65,
                   color=colors, edgecolor="white", linewidth=0.5,
                   capsize=3, error_kw={"linewidth": 0.8})
    ax.set_yticks(range(len(modes)))
    ax.set_yticklabels(modes)
    ax.set_xlabel("$R_V$ (participation ratio ratio)")
    ax.axvline(1.0, color="black", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_xlim(0.4, 1.3)
    ax.set_title("$R_V$ Spectral Fingerprint Across 10 Computational Modes")
    ax.invert_yaxis()

    # Add annotation
    self_idx = [i for i, m in enumerate(modes) if "Self" in m][0]
    ax.annotate(f"d = −1.67 (overall)",
                xy=(means[self_idx], self_idx),
                xytext=(means[self_idx] - 0.15, self_idx + 1.5),
                fontsize=7, color=C_SELF, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C_SELF, lw=0.8))

    fig.tight_layout()
    save_fig(fig, "fig1_mode_atlas_rv", output_dir)


# ===================================================================
# FIGURE 2: Cross-architecture forest plot
# ===================================================================
def fig_cross_architecture(output_dir):
    """Forest plot of Cohen's d across 5 architectures."""
    archs = [
        ("Mistral-7B", -2.26, 45),
        ("OPT-6.7B", -1.84, 45),
        ("GPT-2 XL", -1.14, 45),
        ("Qwen2.5-7B", -0.72, 45),
        ("Pythia-1.4B", -0.31, 63),
    ]

    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    for i, (name, d, n) in enumerate(archs):
        se = np.sqrt(2 / n + d ** 2 / (2 * n))
        ci_lo = d - 1.96 * se
        ci_hi = d + 1.96 * se
        color = C_SELF if abs(d) > 0.5 else C_OTHER
        ax.errorbar(d, i, xerr=[[d - ci_lo], [ci_hi - d]],
                    fmt="o", color=color, markersize=7,
                    capsize=4, capthick=1.0, linewidth=1.0)
        ax.text(d - 0.08, i + 0.25, f"d = {d:.2f}", fontsize=7,
                ha="right", color=color)

    ax.axvline(0, color="black", linestyle="-", linewidth=0.6)
    ax.axvline(-0.8, color="grey", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.set_yticks(range(len(archs)))
    ax.set_yticklabels([a[0] for a in archs])
    ax.set_xlabel("Cohen's $d$ (recursive vs baseline $R_V$)")
    ax.set_title("Cross-Architecture Replication")
    ax.invert_yaxis()
    fig.tight_layout()
    save_fig(fig, "fig2_cross_architecture", output_dir)


# ===================================================================
# FIGURE 3: Statistical hardening forest plot
# ===================================================================
def fig_stat_hardening(output_dir):
    """Forest plot with CIs and Bayes factors for all key effects."""
    harden_files = sorted(glob.glob(str(RESULTS / "statistical_hardening/hardening_summary_*.json")))
    if not harden_files:
        print("  ✗ hardening data not found")
        return
    data = load_json(harden_files[-1])

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    effects = data["effects"]
    for i, e in enumerate(effects):
        d = e["d_observed"]
        lo = e["ci_95_lower"]
        hi = e["ci_95_upper"]
        bf = e["bf_interpretation"]
        color = C_SELF if bf in ("decisive", "very strong") else C_OTHER
        ax.errorbar(d, i, xerr=[[d - lo], [hi - d]],
                    fmt="s", color=color, markersize=5,
                    capsize=3, capthick=0.8, linewidth=0.8)
        ax.text(hi + 0.1, i, f"BF={bf}", fontsize=6, va="center", color=color)

    ax.axvline(0, color="black", linestyle="-", linewidth=0.6)
    ax.set_yticks(range(len(effects)))
    ax.set_yticklabels([e["name"] for e in effects], fontsize=7)
    ax.set_xlabel("Cohen's $d$ (with 95% CI)")
    ax.set_title("Statistical Hardening: Key Effects with Confidence Intervals")
    ax.invert_yaxis()
    fig.tight_layout()
    save_fig(fig, "fig3_statistical_hardening", output_dir)


# ===================================================================
# FIGURE 4: Per-head attention entropy heatmap
# ===================================================================
def fig_per_head_heatmap(output_dir):
    """Heatmap of Cohen's d for entropy difference across heads at L5 and L27."""
    head_files = sorted(glob.glob(str(RESULTS / "per_head_attention/per_head_summary_*.json")))
    if not head_files:
        print("  ✗ per-head data not found")
        return
    data = load_json(head_files[-1])

    layers = sorted(set(h["layer"] for h in data["head_results"]))
    n_heads = data["n_heads"]

    fig, axes = plt.subplots(1, len(layers), figsize=(5.5, 2.5),
                             sharey=True)
    if len(layers) == 1:
        axes = [axes]

    for ax, layer in zip(axes, layers):
        heads = [h for h in data["head_results"] if h["layer"] == layer]
        heads.sort(key=lambda h: h["head"])
        ds = [h["cohens_d"] for h in heads]

        # Reshape for heatmap display (4 x 8 grid)
        n_rows, n_cols = 4, 8
        grid = np.array(ds[:n_rows * n_cols]).reshape(n_rows, n_cols)

        im = ax.imshow(grid, cmap="RdBu_r", vmin=-1, vmax=4,
                       aspect="auto")
        ax.set_title(f"Layer {layer}", fontsize=9)
        ax.set_xlabel("Head (mod 8)")
        if ax == axes[0]:
            ax.set_ylabel("Head (div 8)")
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(range(n_cols), fontsize=6)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels([f"{r * 8}-{r * 8 + 7}" for r in range(n_rows)], fontsize=6)

        # Annotate top heads
        for r in range(n_rows):
            for c in range(n_cols):
                val = grid[r, c]
                if abs(val) > 2.0:
                    ax.text(c, r, f"{val:.1f}", ha="center", va="center",
                            fontsize=5, fontweight="bold",
                            color="white" if val > 2.5 else "black")

    cbar = fig.colorbar(im, ax=axes, shrink=0.8, label="Cohen's $d$")
    fig.suptitle("Per-Head Attention Entropy: Recursive vs Baseline", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_fig(fig, "fig4_per_head_entropy", output_dir)


# ===================================================================
# FIGURE 5: Mode atlas pairwise comparison heatmap
# ===================================================================
def fig_mode_pairwise(output_dir):
    """Heatmap of Cohen's d between all mode pairs."""
    atlas_files = sorted(glob.glob(str(RESULTS / "mode_atlas/atlas_summary_*.json")))
    if not atlas_files:
        print("  ✗ mode atlas data not found")
        return
    data = load_json(atlas_files[-1])
    comps = data["comparisons"]

    # Extract unique modes in order
    mode_set = list(data["fingerprint"].keys())
    # Sort by mean R_V
    mode_set.sort(key=lambda m: data["fingerprint"][m]["rv"]["mean"])
    n = len(mode_set)

    mat = np.zeros((n, n))
    for key, val in comps.items():
        parts = key.split("_vs_")
        a, b = parts[0], parts[1]
        if a in mode_set and b in mode_set:
            i, j = mode_set.index(a), mode_set.index(b)
            mat[i, j] = val["d"]
            mat[j, i] = -val["d"]

    labels = [m.replace("_", "\n").title() for m in mode_set]

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-4, vmax=4)
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, fontsize=6, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=6)

    # Annotate significant cells
    for i in range(n):
        for j in range(n):
            if i != j:
                key1 = f"{mode_set[i]}_vs_{mode_set[j]}"
                key2 = f"{mode_set[j]}_vs_{mode_set[i]}"
                p = None
                if key1 in comps:
                    p = comps[key1]["p"]
                elif key2 in comps:
                    p = comps[key2]["p"]
                sig = "***" if p and p < 0.001 else ("**" if p and p < 0.01 else ("*" if p and p < 0.05 else ""))
                if sig:
                    ax.text(j, i, sig, ha="center", va="center", fontsize=5,
                            color="white" if abs(mat[i, j]) > 2 else "black")

    fig.colorbar(im, ax=ax, shrink=0.7, label="Cohen's $d$")
    ax.set_title("Pairwise $R_V$ Comparisons Between Modes", fontsize=10)
    fig.tight_layout()
    save_fig(fig, "fig5_mode_pairwise_heatmap", output_dir)


# ===================================================================
# FIGURE 6: R_V distribution violin plot
# ===================================================================
def fig_rv_distribution(output_dir):
    """Violin/box plot of R_V for self-referential vs all other modes."""
    atlas_files = sorted(glob.glob(str(RESULTS / "mode_atlas/atlas_summary_*.json")))
    if not atlas_files:
        print("  ✗ mode atlas data not found")
        return
    data = load_json(atlas_files[-1])

    self_rvs = [r["rv"] for r in data["all_results"]["self_referential"]
                if r["rv"] is not None and not np.isnan(r["rv"])]
    other_rvs = []
    for mode, results in data["all_results"].items():
        if mode == "self_referential":
            continue
        for r in results:
            if r["rv"] is not None and not np.isnan(r["rv"]):
                other_rvs.append(r["rv"])

    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    parts = ax.violinplot([self_rvs, other_rvs], positions=[0, 1],
                          showmeans=True, showmedians=True)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(C_SELF if i == 0 else C_BASE)
        pc.set_alpha(0.6)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Self-Referential\n(n={})".format(len(self_rvs)),
                        "All Other Modes\n(n={})".format(len(other_rvs))])
    ax.set_ylabel("$R_V$")
    ax.set_title("$R_V$ Distribution: Self-Referential vs Other Modes")
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.5, alpha=0.5)

    # Add stats annotation
    from scipy import stats
    u, p = stats.mannwhitneyu(self_rvs, other_rvs, alternative="less")
    d = (np.mean(self_rvs) - np.mean(other_rvs)) / np.sqrt(
        (np.std(self_rvs)**2 + np.std(other_rvs)**2) / 2)
    ax.text(0.5, 1.15, f"$d$ = {d:.2f}, $p$ < {p:.1e}",
            ha="center", fontsize=7, transform=ax.transAxes)

    fig.tight_layout()
    save_fig(fig, "fig6_rv_distribution", output_dir)


# ===================================================================
# FIGURE 7: Layer sweep (from canonical session data)
# ===================================================================
def fig_layer_sweep(output_dir):
    """R_V separation by layer depth."""
    sweep_files = sorted(glob.glob(str(RESULTS / "canonical/session_2_final/layer_sweep/layer_sweep_results.json")))
    if not sweep_files:
        sweep_files = sorted(glob.glob(str(RESULTS / "canonical/layer_sweep_results.json")))
    if not sweep_files:
        print("  ✗ layer sweep data not found")
        return
    data = load_json(sweep_files[-1])

    if isinstance(data, dict) and "results" in data:
        results = data["results"]
        layers = [r["layer"] for r in results]
        # Compute separation as proxy for effect size
        ds = [r.get("separation", r.get("cohens_d", 0)) for r in results]
    elif isinstance(data, list):
        layers = [d.get("layer", d.get("late_layer", i)) for i, d in enumerate(data)]
        ds = [d.get("cohens_d", d.get("d", 0)) for d in data]
    elif isinstance(data, dict) and "layers" in data:
        layers = data["layers"]
        ds = data["cohens_d"]
    else:
        print("  ✗ layer sweep format not recognized")
        return

    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.plot(layers, ds, "o-", color=C_SELF, markersize=4, linewidth=1.2)
    ax.axhline(0, color="black", linestyle="-", linewidth=0.5)
    ax.axhline(-0.8, color="grey", linestyle=":", linewidth=0.5,
               label="Large effect threshold")
    ax.set_xlabel("Late Layer Index")
    ax.set_ylabel("Cohen's $d$ (recursive vs baseline)")
    ax.set_title("$R_V$ Effect Size Across Layers (Mistral-7B)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    save_fig(fig, "fig7_layer_sweep", output_dir)


# ===================================================================
# FIGURE 8: Necessity / Sufficiency schematic
# ===================================================================
def fig_necessity_sufficiency(output_dir):
    """2x2 schematic showing necessity and sufficiency results."""
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.5, 2.5)
    ax.set_aspect("equal")

    # Draw grid
    cells = {
        (0, 1): ("Intact\nGeometry", "Intact\nKV Context", "56% BT+ART\n(recursive)", C_POSITIVE),
        (1, 1): ("Ablated\nGeometry", "Intact\nKV Context", "27.7% BT+ART\n(KV sufficient)", C_HIGHLIGHT),
        (0, 0): ("Intact\nGeometry", "Ablated\nKV Context", "3.7% BT+ART\n(geometry alone\nnot sufficient)", C_NEGATIVE),
        (1, 0): ("Ablated\nGeometry", "Ablated\nKV Context", "2.7% BT+ART\n(baseline)", C_OTHER),
    }

    for (x, y), (xlabel, ylabel, text, color) in cells.items():
        rect = FancyBboxPatch((x - 0.4, y - 0.4), 0.8, 0.8,
                              boxstyle="round,pad=0.05",
                              facecolor=color, alpha=0.3,
                              edgecolor=color, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x, y, text, ha="center", va="center", fontsize=7,
                fontweight="bold")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Intact V-proj\nGeometry", "Ablated V-proj\nGeometry"],
                       fontsize=8)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Ablated\nKV Context", "Intact\nKV Context"], fontsize=8)
    ax.set_title("Necessity–Sufficiency Dissociation", fontsize=10)

    # Arrows for necessity / sufficiency
    ax.annotate("Necessary\n($d$=3.29)", xy=(0.6, 0.5), fontsize=7,
                color=C_SELF, fontweight="bold", ha="center")
    ax.annotate("Sufficient\n(OR=13.96)", xy=(1, 1.5), fontsize=7,
                color=C_HIGHLIGHT, fontweight="bold", ha="center")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_fig(fig, "fig8_necessity_sufficiency", output_dir)


# ===================================================================
# FIGURE 9: Spectral fingerprint — PR early vs late
# ===================================================================
def fig_spectral_scatter(output_dir):
    """Scatter plot of PR_early vs PR_late for each mode."""
    atlas_files = sorted(glob.glob(str(RESULTS / "mode_atlas/atlas_summary_*.json")))
    if not atlas_files:
        print("  ✗ mode atlas data not found")
        return
    data = load_json(atlas_files[-1])

    fig, ax = plt.subplots(figsize=(4.5, 4.0))

    for mode, vals in data["fingerprint"].items():
        pr_e = vals["pr_early"]["mean"]
        pr_l = vals["pr_late"]["mean"]
        color = C_SELF if mode == "self_referential" else C_OTHER
        size = 80 if mode == "self_referential" else 40
        marker = "D" if mode == "self_referential" else "o"
        label = mode.replace("_", " ").title()
        ax.scatter(pr_e, pr_l, c=color, s=size, marker=marker,
                   edgecolors="white", linewidth=0.5, zorder=3)
        ax.annotate(label, (pr_e, pr_l), fontsize=5,
                    xytext=(5, 5), textcoords="offset points")

    # Add diagonal
    lims = [5, 11]
    ax.plot(lims, lims, "--", color="grey", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("PR(early) — Layer 5")
    ax.set_ylabel("PR(late) — Layer 27")
    ax.set_title("Spectral Fingerprint: Early vs Late Participation Ratio")
    fig.tight_layout()
    save_fig(fig, "fig9_spectral_scatter", output_dir)


# ===================================================================
# FIGURE 10: Circularity controls
# ===================================================================
def fig_circularity_controls(output_dir):
    """R_V by circularity control type."""
    circ_files = sorted(glob.glob(str(RESULTS / "circularity_controls/circularity_controls_*.json")))
    if not circ_files:
        print("  ✗ circularity data not found")
        return
    data = load_json(circ_files[-1])

    if isinstance(data, dict) and "groups" in data:
        categories = {}
        for gname, gdata in data["groups"].items():
            categories[gname] = gdata.get("rvs", [])
    elif isinstance(data, dict) and "categories" in data:
        categories = data["categories"]
    elif isinstance(data, dict) and "results" in data:
        categories = {}
        for r in data["results"]:
            cat = r.get("category", "unknown")
            if cat not in categories:
                categories[cat] = []
            rv = r.get("rv")
            if rv is not None and not np.isnan(rv):
                categories[cat].append(rv)
    elif isinstance(data, list):
        categories = {}
        for r in data:
            cat = r.get("category", "unknown")
            if cat not in categories:
                categories[cat] = []
            rv = r.get("rv")
            if rv is not None and not np.isnan(rv):
                categories[cat].append(rv)
    else:
        print("  ✗ circularity format not recognized")
        return

    if not categories:
        print("  ✗ no circularity categories found")
        return

    names = sorted(categories.keys())
    means = [np.mean(categories[n]) for n in names]
    stds = [np.std(categories[n]) for n in names]
    labels = [n.replace("_", "\n") for n in names]

    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    colors = [C_SELF if "self" in n.lower() or "recursive" in n.lower() and "non" not in n.lower()
              else C_OTHER for n in names]
    ax.bar(range(len(names)), means, yerr=stds, color=colors,
           edgecolor="white", capsize=3, error_kw={"linewidth": 0.8})
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel("$R_V$")
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.5)
    ax.set_title("Circularity Controls: Structure vs Semantics")
    fig.tight_layout()
    save_fig(fig, "fig10_circularity_controls", output_dir)


# ===================================================================
# FIGURE 11: Self-feeding loop result
# ===================================================================
def fig_self_feeding(output_dir):
    """Self-feeding vs scaffolded recursive generation."""
    # Hard-coded from established findings
    conditions = ["Gnani\nScaffolded", "Self-Feeding\nRecursive"]
    bt_art = [42.4, 10.0]
    colors = [C_POSITIVE, C_NEGATIVE]

    fig, ax = plt.subplots(figsize=(3.0, 3.0))
    bars = ax.bar(conditions, bt_art, color=colors, edgecolor="white",
                  width=0.5)
    ax.set_ylabel("BT+ART Rate (%)")
    ax.set_title("Self-Feeding Loop is Negative\n($d$ = −4.28)")
    for bar, val in zip(bars, bt_art):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val}%", ha="center", fontsize=8, fontweight="bold")
    ax.set_ylim(0, 55)
    fig.tight_layout()
    save_fig(fig, "fig11_self_feeding", output_dir)


# ===================================================================
# FIGURE 12: Multi-metric discriminant
# ===================================================================
def fig_multi_metric(output_dir):
    """Multi-metric radar chart for self-referential vs other modes."""
    atlas_files = sorted(glob.glob(str(RESULTS / "mode_atlas/atlas_summary_*.json")))
    if not atlas_files:
        print("  ✗ mode atlas data not found")
        return
    data = load_json(atlas_files[-1])
    fp = data["fingerprint"]

    metrics = ["rv", "attn_entropy", "spectral_late_top1_ratio",
               "spectral_late_spectral_gap", "spectral_late_effective_rank"]
    metric_labels = ["$R_V$", "Attn Entropy", "Top-1 σ Ratio",
                     "Spectral Gap", "Effective Rank"]

    # Normalize each metric across modes for radar
    self_vals = []
    other_vals = []
    for m in metrics:
        all_vals = [fp[mode][m]["mean"] for mode in fp if m in fp[mode]]
        min_v, max_v = min(all_vals), max(all_vals)
        rng = max_v - min_v if max_v != min_v else 1
        self_vals.append((fp["self_referential"][m]["mean"] - min_v) / rng)
        other_means = [fp[mode][m]["mean"] for mode in fp
                       if mode != "self_referential" and m in fp[mode]]
        other_vals.append((np.mean(other_means) - min_v) / rng)

    # Radar chart
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    self_vals += self_vals[:1]
    other_vals += other_vals[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(4.0, 4.0), subplot_kw=dict(polar=True))
    ax.fill(angles, self_vals, alpha=0.25, color=C_SELF)
    ax.plot(angles, self_vals, "o-", color=C_SELF, linewidth=1.2,
            markersize=4, label="Self-Referential")
    ax.fill(angles, other_vals, alpha=0.15, color=C_BASE)
    ax.plot(angles, other_vals, "s--", color=C_BASE, linewidth=1.0,
            markersize=3, label="Other Modes (mean)")
    ax.set_thetagrids(np.degrees(angles[:-1]), metric_labels, fontsize=7)
    ax.set_title("Multi-Metric Discriminant Profile", fontsize=10, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=7)
    fig.tight_layout()
    save_fig(fig, "fig12_multi_metric_radar", output_dir)


# ===================================================================
# MAIN
# ===================================================================
def main():
    parser = argparse.ArgumentParser(description="Generate publication figures")
    parser.add_argument("--output-dir", default="paper/figures",
                        help="Output directory for figures")
    args = parser.parse_args()

    print(f"Generating figures → {args.output_dir}/")
    print("=" * 50)

    fig_mode_atlas_bar(args.output_dir)
    fig_cross_architecture(args.output_dir)
    fig_stat_hardening(args.output_dir)
    fig_per_head_heatmap(args.output_dir)
    fig_mode_pairwise(args.output_dir)
    fig_rv_distribution(args.output_dir)
    fig_layer_sweep(args.output_dir)
    fig_necessity_sufficiency(args.output_dir)
    fig_spectral_scatter(args.output_dir)
    fig_circularity_controls(args.output_dir)
    fig_self_feeding(args.output_dir)
    fig_multi_metric(args.output_dir)

    print("=" * 50)
    print("Done!")


if __name__ == "__main__":
    main()
