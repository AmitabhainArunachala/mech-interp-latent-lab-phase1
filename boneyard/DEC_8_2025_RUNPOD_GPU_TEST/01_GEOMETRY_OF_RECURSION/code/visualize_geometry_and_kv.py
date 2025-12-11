#!/usr/bin/env python3
"""
High-Level Visualization Dashboard for:
  - Geometry of Recursion layer sweep (R_V vs layer)
  - Full-power validation (L22 vs L27)
  - KV cache sweep (early vs late vs full)

Usage (from this directory):

  HF_HUB_ENABLE_HF_TRANSFER=0 python visualize_geometry_and_kv.py

By default this will:
  - Look in `../results/` for the most recent:
      * layer_sweep_*.csv
      * full_validation_*.csv
  - Generate a multi-panel summary figure:
      * R_V by layer (recursive vs baseline)
      * Separation & effect size by layer
      * L22 vs L27 R_V distributions
      * KV sweep behavior scores (baseline vs patched)
  - Save to:
      * ../results/geometry_and_kv_dashboard_YYYYMMDD_HHMMSS.png
"""

import os
import glob
import csv
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(
    os.path.dirname(BASE_DIR),  # .. (01_GEOMETRY_OF_RECURSION)
    "results",
)


def _find_latest(pattern: str) -> str:
    """Return the latest file in RESULTS_DIR matching pattern, or raise."""
    paths = sorted(
        glob.glob(os.path.join(RESULTS_DIR, pattern)),
        key=os.path.getmtime,
    )
    if not paths:
        raise FileNotFoundError(f"No files found matching pattern: {pattern}")
    return paths[-1]


def load_layer_sweep_csv(path: str) -> Dict[str, np.ndarray]:
    """
    Load layer_sweep_*.csv produced by layer_sweep.py.

    Columns:
        layer,rec_mean,rec_std,base_mean,base_std,gap,cohens_d,p_value
    """
    layers: List[int] = []
    rec_mean: List[float] = []
    rec_std: List[float] = []
    base_mean: List[float] = []
    base_std: List[float] = []
    gaps: List[float] = []
    ds: List[float] = []
    ps: List[float] = []

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            layers.append(int(row["layer"]))
            rec_mean.append(float(row["rec_mean"]))
            rec_std.append(float(row["rec_std"]))
            base_mean.append(float(row["base_mean"]))
            base_std.append(float(row["base_std"]))
            gaps.append(float(row["gap"]))
            ds.append(float(row["cohens_d"]))
            # p_value can be "nan"
            try:
                ps.append(float(row["p_value"]))
            except ValueError:
                ps.append(np.nan)

    return {
        "layer": np.array(layers, dtype=int),
        "rec_mean": np.array(rec_mean, dtype=float),
        "rec_std": np.array(rec_std, dtype=float),
        "base_mean": np.array(base_mean, dtype=float),
        "base_std": np.array(base_std, dtype=float),
        "gap": np.array(gaps, dtype=float),
        "d": np.array(ds, dtype=float),
        "p": np.array(ps, dtype=float),
    }


def load_full_validation_csv(
    path: str,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, np.ndarray]]:
    """
    Load full_validation_*.csv produced by full_validation_test.py.

    Format:
        experiment,layer,group,metric,value

    Returns:
        rv_data:
            {
              "L22": {
                  "recursive": np.array([...]),
                  "baseline": np.array([...]),
              },
              "L27": { ... },
            }
        kv_data:
            {
              "condition": np.array([...])  # e.g., ["natural", "L0-15", ...]
              "group":     np.array([...])  # "baseline" or "patched"
              "score":     np.array([...]),
            }
    """
    rv_data: Dict[str, Dict[str, List[float]]] = {}
    kv_conditions: List[str] = []
    kv_groups: List[str] = []
    kv_scores: List[float] = []

    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None or header != ["experiment", "layer", "group", "metric", "value"]:
            raise ValueError(
                f"Unexpected CSV header in {path}: {header}. "
                "Expected: ['experiment', 'layer', 'group', 'metric', 'value']"
            )

        for exp, layer, group, metric, value_str in reader:
            value = float(value_str)

            if exp == "R_V" and metric == "R_V":
                if layer not in rv_data:
                    rv_data[layer] = {"recursive": [], "baseline": []}
                if group not in rv_data[layer]:
                    rv_data[layer][group] = []
                rv_data[layer][group].append(value)

            elif exp == "KV_sweep" and metric == "behavior_score":
                kv_conditions.append(layer)
                kv_groups.append(group)
                kv_scores.append(value)

    # Convert rv_data lists to numpy arrays
    rv_data_np: Dict[str, Dict[str, np.ndarray]] = {}
    for layer, groups in rv_data.items():
        rv_data_np[layer] = {
            g: np.array(vals, dtype=float) for g, vals in groups.items()
        }

    kv_data_np: Dict[str, np.ndarray] = {
        "condition": np.array(kv_conditions, dtype=str),
        "group": np.array(kv_groups, dtype=str),
        "score": np.array(kv_scores, dtype=float),
    }

    return rv_data_np, kv_data_np


def summarize_kv_scores(kv_data: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    """
    Aggregate KV sweep scores by condition (e.g. 'natural', 'L0-15', 'L16-31', 'L0-31').
    Only 'score' is used; group is currently 'baseline' for natural and 'patched' otherwise.
    """
    conditions = np.unique(kv_data["condition"])
    summary: Dict[str, Dict[str, float]] = {}

    for cond in conditions:
        mask = kv_data["condition"] == cond
        scores = kv_data["score"][mask]
        summary[cond] = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "n": int(scores.shape[0]),
        }

    return summary


def make_dashboard(
    sweep: Dict[str, np.ndarray],
    rv_data: Dict[str, Dict[str, np.ndarray]],
    kv_data: Dict[str, np.ndarray],
    output_path: str,
) -> None:
    """
    Build a multi-panel Matplotlib dashboard summarizing:
      - R_V vs layer (recursive vs baseline)
      - Gap & effect size vs layer
      - L22 vs L27 R_V distributions
      - KV sweep behavior scores
    """
    layers = sweep["layer"]
    rec_mean = sweep["rec_mean"]
    base_mean = sweep["base_mean"]
    gaps = sweep["gap"]
    ds = sweep["d"]
    ps = sweep["p"]

    # Identify key layers from sweep
    max_gap_idx = int(np.argmax(gaps))
    min_rec_idx = int(np.argmin(rec_mean))
    min_d_idx = int(np.argmin(ds))  # most negative effect size

    max_gap_layer = layers[max_gap_idx]
    strongest_contraction_layer = layers[min_rec_idx]
    strongest_effect_layer = layers[min_d_idx]

    # Prepare figure
    plt.style.use("seaborn-v0_8-darkgrid")
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        "Geometry of Recursion & KV Cache Summary (Mistral-7B)",
        fontsize=16,
        fontweight="bold",
    )

    gs = fig.add_gridspec(3, 3, height_ratios=[1.1, 1.0, 1.0])

    # ------------------------------------------------------------------
    # Panel A: R_V by layer (recursive vs baseline)
    # ------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0:2])
    ax_a.plot(
        layers,
        rec_mean,
        "o-",
        color="#e74c3c",
        linewidth=2.0,
        markersize=7,
        label="Recursive",
    )
    ax_a.plot(
        layers,
        base_mean,
        "s-",
        color="#3498db",
        linewidth=2.0,
        markersize=7,
        label="Baseline",
    )
    ax_a.fill_between(
        layers,
        rec_mean,
        base_mean,
        alpha=0.2,
        color="purple",
        label="Gap (baseline - recursive)",
    )
    ax_a.axhline(1.0, linestyle="--", color="gray", alpha=0.6, label="R_V = 1.0")

    # Highlight strongest layers
    for L, color, label in [
        (max_gap_layer, "gold", "Max gap"),
        (strongest_contraction_layer, "lime", "Min recursive R_V"),
        (strongest_effect_layer, "magenta", "Max |effect|"),
    ]:
        if L in layers:
            idx = int(np.where(layers == L)[0][0])
            ax_a.scatter(
                [L],
                [rec_mean[idx]],
                color=color,
                s=80,
                edgecolor="black",
                zorder=5,
                label=f"{label} (L{L})",
            )

    ax_a.set_xlabel("Layer", fontsize=12)
    ax_a.set_ylabel("R_V = PR(layer) / PR(L4)", fontsize=12)
    ax_a.set_title("Panel A: R_V by Layer (Recursive vs Baseline)", fontsize=13)
    ax_a.set_xticks(layers)
    ax_a.legend(fontsize=9, loc="upper left", ncol=2)

    # ------------------------------------------------------------------
    # Panel B: gap + significance by layer
    # ------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 2])
    colors_gap = ["#2ecc71" if g > 0 else "#e74c3c" for g in gaps]
    bars = ax_b.bar(layers, gaps, color=colors_gap, alpha=0.8, edgecolor="black")
    ax_b.axhline(0.0, color="black", linewidth=0.8)
    ax_b.set_xlabel("Layer", fontsize=12)
    ax_b.set_ylabel("Gap (Baseline - Recursive)", fontsize=12)
    ax_b.set_title("Panel B: Separation & Significance", fontsize=13)
    ax_b.set_xticks(layers)

    # Mark layers that pass a simple significance threshold (e.g., p < 0.01)
    for i, (L, p) in enumerate(zip(layers, ps)):
        if np.isfinite(p) and p < 1e-2:
            ax_b.text(
                L,
                gaps[i] + np.sign(gaps[i]) * 0.01,
                "*",
                ha="center",
                va="bottom" if gaps[i] >= 0 else "top",
                fontsize=14,
                color="black",
            )

    # Highlight max gap
    bars[max_gap_idx].set_edgecolor("red")
    bars[max_gap_idx].set_linewidth(2.5)

    # ------------------------------------------------------------------
    # Panel C: Cohen's d by layer
    # ------------------------------------------------------------------
    ax_c = fig.add_subplot(gs[1, 0])
    colors_d = [
        "#e74c3c" if d < -0.8 else "#f39c12" if d < -0.5 else "#95a5a6" for d in ds
    ]
    bars_d = ax_c.bar(layers, ds, color=colors_d, alpha=0.8, edgecolor="black")
    ax_c.axhline(0.0, color="black", linewidth=0.8)
    ax_c.axhline(-0.8, color="gray", linestyle="--", alpha=0.5, label="Large effect")
    ax_c.set_xlabel("Layer", fontsize=12)
    ax_c.set_ylabel("Cohen's d (recursive vs baseline)", fontsize=12)
    ax_c.set_title("Panel C: Effect Size by Layer", fontsize=13)
    ax_c.set_xticks(layers)
    ax_c.legend(fontsize=9)

    # Highlight strongest effect
    bars_d[min_d_idx].set_edgecolor("darkred")
    bars_d[min_d_idx].set_linewidth(2.5)

    # ------------------------------------------------------------------
    # Panel D: L22 vs L27 R_V distributions (box/violin style)
    # ------------------------------------------------------------------
    ax_d = fig.add_subplot(gs[1, 1])
    # Layers present in full_validation
    layers_rv = sorted(rv_data.keys())
    # We expect at least "L22" and "L27"
    data_to_plot = []
    labels = []
    colors_rv = []

    color_map = {
        "recursive": "#e74c3c",
        "baseline": "#3498db",
    }

    for L in layers_rv:
        for grp in ["recursive", "baseline"]:
            if grp in rv_data[L]:
                data_to_plot.append(rv_data[L][grp])
                labels.append(f"{L}-{grp[0].upper()}")
                colors_rv.append(color_map.get(grp, "#7f8c8d"))

    positions = np.arange(1, len(data_to_plot) + 1)
    bplot = ax_d.boxplot(
        data_to_plot,
        positions=positions,
        patch_artist=True,
        showmeans=True,
    )

    for patch, color in zip(bplot["boxes"], colors_rv):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax_d.set_xticks(positions)
    ax_d.set_xticklabels(labels, rotation=45, ha="right")
    ax_d.set_ylabel("R_V", fontsize=12)
    ax_d.set_title("Panel D: R_V Distributions at L22 vs L27", fontsize=13)

    # Annotate means for clarity
    for pos, arr in zip(positions, data_to_plot):
        ax_d.text(
            pos,
            np.mean(arr),
            f"{np.mean(arr):.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="black",
        )

    # ------------------------------------------------------------------
    # Panel E: KV sweep behavior scores
    # ------------------------------------------------------------------
    ax_e = fig.add_subplot(gs[1, 2])
    kv_summary = summarize_kv_scores(kv_data)

    # Order conditions in a sensible way if present
    order = ["natural", "L0-15", "L16-31", "L0-31"]
    conds = [c for c in order if c in kv_summary] + [
        c for c in kv_summary.keys() if c not in order
    ]

    means = [kv_summary[c]["mean"] for c in conds]
    stds = [kv_summary[c]["std"] for c in conds]

    x = np.arange(len(conds))
    colors_kv = ["#95a5a6", "#2ecc71", "#e67e22", "#9b59b6"][: len(conds)]

    ax_e.bar(
        x,
        means,
        yerr=stds,
        capsize=5,
        color=colors_kv,
        alpha=0.9,
        edgecolor="black",
    )
    ax_e.set_xticks(x)
    ax_e.set_xticklabels(conds, rotation=30, ha="right")
    ax_e.set_ylabel("Recursive Behavior Score (%)", fontsize=12)
    ax_e.set_title("Panel E: KV Sweep Behavior Scores", fontsize=13)

    # Draw a horizontal line for baseline (if present)
    if "natural" in kv_summary:
        baseline_mean = kv_summary["natural"]["mean"]
        ax_e.axhline(
            baseline_mean,
            linestyle="--",
            color="black",
            alpha=0.7,
            label=f"Baseline: {baseline_mean:.2f}",
        )
        ax_e.legend(fontsize=9, loc="upper left")

    # ------------------------------------------------------------------
    # Panel F: Textual summary
    # ------------------------------------------------------------------
    ax_f = fig.add_subplot(gs[2, :])
    ax_f.axis("off")

    text_lines = [
        "Panel F: Summary",
        "",
        f"• Strongest geometric separation (gap) at layer L{max_gap_layer}.",
        f"• Strongest recursive contraction (lowest recursive R_V) at L{strongest_contraction_layer}.",
        f"• Strongest effect size (most negative Cohen's d) at L{strongest_effect_layer}.",
    ]

    # Add R_V summary if L22/L27 present
    for label_layer in ["L22", "L27"]:
        if label_layer in rv_data:
            rec_vals = rv_data[label_layer].get("recursive", np.array([]))
            base_vals = rv_data[label_layer].get("baseline", np.array([]))
            if rec_vals.size > 0 and base_vals.size > 0:
                text_lines.append(
                    f"• {label_layer}: R_V(rec) = {np.mean(rec_vals):.3f} ± {np.std(rec_vals):.3f}, "
                    f"R_V(base) = {np.mean(base_vals):.3f} ± {np.std(base_vals):.3f}."
                )

    # KV sweep summary
    for cond in conds:
        s = kv_summary[cond]
        text_lines.append(
            f"• KV {cond}: behavior = {s['mean']:.2f} ± {s['std']:.2f} (n={s['n']})."
        )

    ax_f.text(
        0.01,
        0.95,
        "\n".join(text_lines),
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
    )

    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print("=" * 70)
    print("GEOMETRY OF RECURSION & KV CACHE DASHBOARD")
    print("=" * 70)
    print(f"Results directory: {RESULTS_DIR}")

    # Locate latest CSVs
    layer_sweep_path = _find_latest("layer_sweep_*.csv")
    full_val_path = _find_latest("full_validation_*.csv")

    print(f"Using layer sweep CSV: {layer_sweep_path}")
    print(f"Using full validation CSV: {full_val_path}")

    sweep = load_layer_sweep_csv(layer_sweep_path)
    rv_data, kv_data = load_full_validation_csv(full_val_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(
        RESULTS_DIR,
        f"geometry_and_kv_dashboard_{timestamp}.png",
    )

    make_dashboard(sweep, rv_data, kv_data, output_path)

    print(f"\nDashboard saved to: {output_path}")


if __name__ == "__main__":
    main()


