#!/usr/bin/env python3
"""Analyze seed bridge replication matrix outputs.

Inputs:
- configs/canonical/seed_bridge_2026_02_20/RUN_MATRIX.csv
- results/phase1_mechanism/runs/*_<run_name>/summary.json
- results/phase1_mechanism/runs/*_<run_name>/per_sample.csv

Outputs:
- industry_grade/2026-02-20/evidence/seed_bridge_analysis.json
- industry_grade/2026-02-20/evidence/seed_bridge_analysis.md
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    matplotlib = None
    plt = None


@dataclass
class RunRecord:
    run_id: str
    condition: str
    seed: int
    config_path: Path
    run_name: str
    run_dir: Path
    summary: Dict[str, object]
    rv_delta: np.ndarray
    pair_keys: List[Tuple[str, str]]


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = len(x), len(y)
    sx, sy = x.std(ddof=1), y.std(ddof=1)
    sp = math.sqrt(((nx - 1) * sx * sx + (ny - 1) * sy * sy) / (nx + ny - 2))
    return float((x.mean() - y.mean()) / sp)


def random_effects_meta(estimates: List[float], variances: List[float]) -> Dict[str, float]:
    yi = np.array(estimates, dtype=float)
    vi = np.array(variances, dtype=float)
    wi = 1.0 / vi

    mu_fe = float(np.sum(wi * yi) / np.sum(wi))
    q = float(np.sum(wi * (yi - mu_fe) ** 2))
    df = len(yi) - 1
    c = float(np.sum(wi) - (np.sum(wi**2) / np.sum(wi)))
    tau2 = max(0.0, (q - df) / c) if c > 0 else 0.0

    wi_re = 1.0 / (vi + tau2)
    mu_re = float(np.sum(wi_re * yi) / np.sum(wi_re))
    se_re = float(math.sqrt(1.0 / np.sum(wi_re)))

    i2 = max(0.0, (q - df) / q) * 100.0 if q > 0 else 0.0

    return {
        "k": int(len(yi)),
        "mu_re": mu_re,
        "ci95_low": float(mu_re - 1.96 * se_re),
        "ci95_high": float(mu_re + 1.96 * se_re),
        "tau2": float(tau2),
        "Q": float(q),
        "I2_pct": float(i2),
    }


def _search_run_roots(repo_root: Path) -> List[Path]:
    roots = [
        repo_root / "results" / "phase1_mechanism" / "runs",
        repo_root / "results" / "remote_gpu_sync" / "2026-02-20" / "phase1_mechanism",
    ]
    return [r for r in roots if r.exists()]


def latest_run_dir(results_roots: List[Path], run_name: str) -> Path | None:
    candidates: List[Path] = []
    for root in results_roots:
        candidates.extend(root.glob(f"*_{run_name}"))
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]


def _load_pair_keys(frame: pd.DataFrame) -> List[Tuple[str, str]]:
    if "rec_id" not in frame.columns or "base_id" not in frame.columns:
        return []
    return list(zip(frame["rec_id"].astype(str), frame["base_id"].astype(str)))


def _paired_or_unpaired(x_rec: RunRecord, y_rec: RunRecord) -> Dict[str, float | str]:
    x = x_rec.rv_delta.astype(float)
    y = y_rec.rv_delta.astype(float)

    x_map = dict(zip(x_rec.pair_keys, x)) if x_rec.pair_keys else {}
    y_map = dict(zip(y_rec.pair_keys, y)) if y_rec.pair_keys else {}

    shared_keys = sorted(set(x_map).intersection(y_map))
    if len(shared_keys) >= 2:
        xx = np.array([x_map[k] for k in shared_keys], dtype=float)
        yy = np.array([y_map[k] for k in shared_keys], dtype=float)
        mask = (~np.isnan(xx)) & (~np.isnan(yy))
        xx = xx[mask]
        yy = yy[mask]
        if len(xx) >= 2:
            diff = xx - yy
            t_stat, p_val = stats.ttest_rel(xx, yy)
            d = float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 0 else 0.0
            var = float(diff.var(ddof=1) / len(diff)) if len(diff) > 1 else float("nan")
            return {
                "test_type": "paired_t",
                "n_a": int(len(xx)),
                "n_b": int(len(yy)),
                "n_overlap": int(len(xx)),
                "mean_diff": float(xx.mean() - yy.mean()),
                "t_stat": float(t_stat),
                "p_value": float(p_val),
                "cohens_d": d,
                "var_for_meta": var,
            }

    # Fallback if pair keys unavailable or overlap too small.
    x_clean = x[~np.isnan(x)]
    y_clean = y[~np.isnan(y)]
    if len(x_clean) < 2 or len(y_clean) < 2:
        return {
            "test_type": "insufficient_data",
            "n_a": int(len(x_clean)),
            "n_b": int(len(y_clean)),
            "n_overlap": 0,
            "mean_diff": float("nan"),
            "t_stat": float("nan"),
            "p_value": float("nan"),
            "cohens_d": float("nan"),
            "var_for_meta": float("nan"),
        }
    t_stat, p_val = stats.ttest_ind(x_clean, y_clean, equal_var=False)
    var = float(x_clean.var(ddof=1) / len(x_clean) + y_clean.var(ddof=1) / len(y_clean))
    return {
        "test_type": "welch_t",
        "n_a": int(len(x_clean)),
        "n_b": int(len(y_clean)),
        "n_overlap": 0,
        "mean_diff": float(x_clean.mean() - y_clean.mean()),
        "t_stat": float(t_stat),
        "p_value": float(p_val),
        "cohens_d": cohens_d(x_clean, y_clean),
        "var_for_meta": var,
    }


def _paired_diffs(x_rec: RunRecord, y_rec: RunRecord) -> np.ndarray:
    x = x_rec.rv_delta.astype(float)
    y = y_rec.rv_delta.astype(float)

    x_map = dict(zip(x_rec.pair_keys, x)) if x_rec.pair_keys else {}
    y_map = dict(zip(y_rec.pair_keys, y)) if y_rec.pair_keys else {}
    shared_keys = sorted(set(x_map).intersection(y_map))
    if len(shared_keys) < 2:
        return np.array([], dtype=float)

    xx = np.array([x_map[k] for k in shared_keys], dtype=float)
    yy = np.array([y_map[k] for k in shared_keys], dtype=float)
    mask = (~np.isnan(xx)) & (~np.isnan(yy))
    if int(mask.sum()) < 2:
        return np.array([], dtype=float)
    return xx[mask] - yy[mask]


def _pooled_paired_test(diffs: np.ndarray) -> Dict[str, float | int | None]:
    if len(diffs) < 3:
        return {
            "n_pairs": int(len(diffs)),
            "mean_diff": None,
            "cohens_d": None,
            "t_stat": None,
            "p_value": None,
            "ci95_low": None,
            "ci95_high": None,
        }
    t_stat, p_val = stats.ttest_1samp(diffs, 0.0)
    mean = float(np.mean(diffs))
    std = float(np.std(diffs, ddof=1))
    d = float(mean / std) if std > 0 else 0.0
    sem = float(stats.sem(diffs))
    ci = stats.t.interval(0.95, len(diffs) - 1, loc=mean, scale=sem)
    return {
        "n_pairs": int(len(diffs)),
        "mean_diff": mean,
        "cohens_d": d,
        "t_stat": float(t_stat),
        "p_value": float(p_val),
        "ci95_low": float(ci[0]),
        "ci95_high": float(ci[1]),
    }


def _plot_seed_paired_means(
    per_seed: Dict[str, Dict[str, Dict[str, float | int | str]]],
    by_seed: Dict[int, Dict[str, RunRecord]],
    out_path: Path,
) -> None:
    if plt is None:
        return
    seeds = sorted([int(s) for s in per_seed.keys()])
    if not seeds:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=180)
    comparisons = [
        ("random_head_control", "Head-Specific vs Random-Head"),
        ("baseline_donor_control", "Head-Specific vs Baseline-Donor"),
    ]

    for ax, (ctrl_cond, title) in zip(axes, comparisons):
        xs = [0, 1]
        for seed in seeds:
            runs = by_seed[seed]
            head_mean = float(np.nanmean(runs["head_specific"].rv_delta))
            ctrl_mean = float(np.nanmean(runs[ctrl_cond].rv_delta))
            ax.plot(xs, [head_mean, ctrl_mean], marker="o", alpha=0.75, linewidth=1.4)
            ax.text(1.02, ctrl_mean, f"s{seed}", fontsize=7, alpha=0.75)
        ax.set_xticks(xs)
        ax.set_xticklabels(["head_specific", ctrl_cond], rotation=18)
        ax.set_ylabel("mean rv_delta")
        ax.set_title(title)
        ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def load_run_matrix(repo_root: Path) -> List[RunRecord]:
    matrix_path = repo_root / "configs" / "canonical" / "seed_bridge_2026_02_20" / "RUN_MATRIX.csv"
    results_roots = _search_run_roots(repo_root)

    records: List[RunRecord] = []
    with matrix_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            config_path = repo_root / row["config_path"]
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            run_name = cfg["run_name"]
            run_dir = latest_run_dir(results_roots, run_name)
            if run_dir is None:
                continue

            per_sample_path = run_dir / "per_sample.csv"
            summary_path = run_dir / "summary.json"
            if not per_sample_path.exists() or not summary_path.exists():
                continue
            per_sample = pd.read_csv(per_sample_path)
            rv_delta = per_sample["rv_delta"].to_numpy(dtype=float)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

            records.append(
                RunRecord(
                    run_id=row["run_id"],
                    condition=row["condition"],
                    seed=int(row["seed"]),
                    config_path=config_path,
                    run_name=run_name,
                    run_dir=run_dir,
                    summary=summary,
                    rv_delta=rv_delta,
                    pair_keys=_load_pair_keys(per_sample),
                )
            )

    return records


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = repo_root / "industry_grade" / "2026-02-20" / "evidence"
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_run_matrix(repo_root)

    by_seed: Dict[int, Dict[str, RunRecord]] = {}
    for r in records:
        by_seed.setdefault(r.seed, {})[r.condition] = r

    pair_defs = [
        ("head_specific", "random_head_control"),
        ("head_specific", "baseline_donor_control"),
        ("random_head_control", "baseline_donor_control"),
    ]

    per_seed = {}
    meta_inputs: Dict[Tuple[str, str], Dict[str, List[float]]] = {
        p: {"est": [], "var": []} for p in pair_defs
    }
    pooled_pair_diffs: Dict[Tuple[str, str], List[np.ndarray]] = {p: [] for p in pair_defs}

    for seed, seed_runs in sorted(by_seed.items()):
        if not all(cond in seed_runs for cond in ["head_specific", "random_head_control", "baseline_donor_control"]):
            continue

        seed_out = {}
        for a, b in pair_defs:
            contrast = _paired_or_unpaired(seed_runs[a], seed_runs[b])
            est = float(contrast["mean_diff"])
            var = float(contrast["var_for_meta"])
            if not (math.isnan(est) or math.isnan(var) or var <= 0):
                meta_inputs[(a, b)]["est"].append(est)
                meta_inputs[(a, b)]["var"].append(var)
            diffs = _paired_diffs(seed_runs[a], seed_runs[b])
            if len(diffs) > 0:
                pooled_pair_diffs[(a, b)].append(diffs)

            seed_out[f"{a}_vs_{b}"] = {
                "test_type": contrast["test_type"],
                "n_a": int(contrast["n_a"]),
                "n_b": int(contrast["n_b"]),
                "n_overlap": int(contrast["n_overlap"]),
                "mean_diff": est,
                "t_stat": float(contrast["t_stat"]),
                "p_value": float(contrast["p_value"]),
                "cohens_d": float(contrast["cohens_d"]),
            }

        per_seed[str(seed)] = seed_out

    signal_gate = {
        "criterion": "head_specific must beat both controls with mean_diff<0 and p<0.01",
        "seed_passes": [],
        "n_seed_passes": 0,
    }
    for seed in sorted(per_seed, key=lambda x: int(x)):
        hr = per_seed[seed]["head_specific_vs_random_head_control"]
        hb = per_seed[seed]["head_specific_vs_baseline_donor_control"]
        if (
            hr["mean_diff"] < 0
            and hb["mean_diff"] < 0
            and hr["p_value"] < 0.01
            and hb["p_value"] < 0.01
        ):
            signal_gate["seed_passes"].append(int(seed))
    signal_gate["n_seed_passes"] = len(signal_gate["seed_passes"])

    pooled = {}
    for a, b in pair_defs:
        ests = meta_inputs[(a, b)]["est"]
        vars_ = meta_inputs[(a, b)]["var"]
        if ests:
            pooled[f"{a}_vs_{b}"] = random_effects_meta(ests, vars_)

    pooled_paired = {}
    for a, b in pair_defs:
        key = f"{a}_vs_{b}"
        chunks = pooled_pair_diffs[(a, b)]
        if chunks:
            pooled_diffs = np.concatenate(chunks, axis=0)
        else:
            pooled_diffs = np.array([], dtype=float)
        pooled_paired[key] = _pooled_paired_test(pooled_diffs)

    summary = {
        "n_runs_found": len(records),
        "n_seeds_complete": len(per_seed),
        "seeds_complete": sorted([int(s) for s in per_seed.keys()]),
        "run_roots_scanned": [str(p) for p in _search_run_roots(repo_root)],
        "per_seed": per_seed,
        "pooled_random_effects": pooled,
        "pooled_paired_ttest": pooled_paired,
        "standout_signal_gate": signal_gate,
    }

    json_path = out_dir / "seed_bridge_analysis.json"
    md_path = out_dir / "seed_bridge_analysis.md"

    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    lines = ["# Seed Bridge Analysis\n"]
    lines.append(f"- runs found: `{summary['n_runs_found']}`\n")
    lines.append(f"- complete seeds: `{summary['n_seeds_complete']}`\n")
    lines.append(f"- standout seed passes: `{summary['standout_signal_gate']['n_seed_passes']}` ({summary['standout_signal_gate']['seed_passes']})\n\n")

    if per_seed:
        lines.append("## Per-seed contrasts\n")
        for seed in sorted(per_seed, key=lambda x: int(x)):
            lines.append(f"### Seed {seed}\n")
            for contrast, vals in per_seed[seed].items():
                lines.append(
                    f"- {contrast} [{vals['test_type']}]: diff={vals['mean_diff']:.6f}, p={vals['p_value']:.6g}, d={vals['cohens_d']:.4f}, overlap={vals['n_overlap']}\n"
                )
            lines.append("\n")

    if pooled:
        lines.append("## Pooled random-effects\n")
        for contrast, vals in pooled.items():
            lines.append(
                f"- {contrast}: mu={vals['mu_re']:.6f}, 95%CI=[{vals['ci95_low']:.6f},{vals['ci95_high']:.6f}], I2={vals['I2_pct']:.2f}%\n"
            )

    if pooled_paired:
        lines.append("\n## Pooled paired t-tests\n")
        for contrast, vals in pooled_paired.items():
            if vals["mean_diff"] is None:
                lines.append(f"- {contrast}: insufficient data\n")
            else:
                lines.append(
                    f"- {contrast}: mean_diff={vals['mean_diff']:.6f}, "
                    f"p={vals['p_value']:.6g}, d={vals['cohens_d']:.4f}, n_pairs={vals['n_pairs']}\n"
                )

    md_path.write_text("".join(lines), encoding="utf-8")

    fig_path = out_dir / "seed_bridge_paired_dotplot.png"
    _plot_seed_paired_means(per_seed, by_seed, fig_path)
    summary["artifacts"] = {
        "paired_dotplot_png": str(fig_path) if plt is not None else None,
        "paired_dotplot_status": "written" if (plt is not None and fig_path.exists()) else "matplotlib_unavailable",
    }
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json_path)
    print(md_path)
    print(fig_path if plt is not None else "paired_dotplot_skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
