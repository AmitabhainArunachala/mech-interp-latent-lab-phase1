#!/usr/bin/env python3
"""
Offline indexing + meta-experiment harness for mech-interp-latent-lab-phase1.

This script is designed for environments where model-forward execution is blocked
or expensive. It mines existing run artifacts and performs new statistical checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


RUN_TS_RE = re.compile(r"(\d{8}_\d{6})")


def _now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _extract_ts(text: str) -> str:
    m = RUN_TS_RE.search(text)
    return m.group(1) if m else "00000000_000000"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _cohens_d_1samp(x: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    sd = float(np.std(x, ddof=1))
    if sd == 0:
        return float("nan")
    return float(np.mean(x) / sd)


def _cohens_d_ind(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    sa = float(np.var(a, ddof=1))
    sb = float(np.var(b, ddof=1))
    na, nb = len(a), len(b)
    pooled = ((na - 1) * sa + (nb - 1) * sb) / max(na + nb - 2, 1)
    if pooled <= 0:
        return float("nan")
    return float((np.mean(a) - np.mean(b)) / math.sqrt(pooled))


def _normalize_model(raw: str) -> str:
    m = (raw or "unknown").lower()
    if "mistral" in m:
        return "mistral_7b"
    if "gpt2" in m:
        return "gpt2_xl"
    if "pythia" in m:
        return "pythia_1_4b"
    if "qwen" in m:
        return "qwen2_7b"
    if "opt-6.7b" in m or "opt_6_7b" in m:
        return "opt_6_7b"
    if "gemma-2-9b" in m or "gemma_2_9b" in m:
        return "gemma_2_9b"
    return m.replace("/", "_")


def index_project(repo_root: Path, summary_paths: List[Path]) -> Dict[str, Any]:
    skip_dirs = {".git", ".venv", "__pycache__", ".pytest_cache"}
    top_level_counts: Counter[str] = Counter()
    ext_counts: Counter[str] = Counter()
    total_files = 0

    for root, dirs, files in os.walk(repo_root):
        rel_root = Path(root).relative_to(repo_root)
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for file_name in files:
            total_files += 1
            p = rel_root / file_name
            top_key = p.parts[0] if p.parts else "."
            top_level_counts[top_key] += 1
            ext = Path(file_name).suffix.lower() or "[no_ext]"
            ext_counts[ext] += 1

    pipeline_py = list((repo_root / "src" / "pipelines").rglob("*.py"))
    config_json = list((repo_root / "configs").rglob("*.json"))
    scripts_py = list((repo_root / "scripts").rglob("*.py"))

    exp_config_counts: Counter[str] = Counter()
    for cfg_path in config_json:
        cfg = _safe_json(cfg_path)
        if not cfg:
            continue
        exp = str(cfg.get("experiment") or "UNKNOWN")
        exp_config_counts[exp] += 1

    results_root = repo_root / "results"
    results_summary_count = len(summary_paths)
    results_csv_count = len(list(results_root.rglob("*.csv"))) if results_root.exists() else 0
    results_report_count = len(list(results_root.rglob("report.md"))) if results_root.exists() else 0

    run_index_entries = 0
    run_index_success = 0
    run_index_path = results_root / "RUN_INDEX.jsonl"
    if run_index_path.exists():
        for line in run_index_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                run_index_entries += 1
                if row.get("success") is True:
                    run_index_success += 1
            except Exception:
                continue

    return {
        "generated_at": datetime.now().isoformat(),
        "total_files": total_files,
        "top_level_file_counts": dict(top_level_counts.most_common()),
        "extension_counts": dict(ext_counts.most_common()),
        "code_inventory": {
            "pipeline_python_files": len(pipeline_py),
            "script_python_files": len(scripts_py),
            "config_json_files": len(config_json),
        },
        "config_experiment_counts": dict(exp_config_counts.most_common()),
        "results_inventory": {
            "summary_json_files": results_summary_count,
            "csv_files": results_csv_count,
            "report_md_files": results_report_count,
            "run_index_entries": run_index_entries,
            "run_index_success": run_index_success,
        },
    }


def experiment_cross_arch(repo_root: Path, summary_paths: List[Path]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for sp in summary_paths:
        s = _safe_json(sp)
        if not s:
            continue
        if s.get("experiment") != "rv_l27_causal_validation":
            continue

        run_dir = sp.parent
        cfg = _safe_json(run_dir / "config.json") or {}
        model = (
            s.get("model")
            or s.get("model_name")
            or (cfg.get("model") or {}).get("name")
            or (cfg.get("params") or {}).get("model")
            or "unknown"
        )
        if model == "unknown":
            maybe = run_dir.name.split("rv_l27_causal_validation_")
            if len(maybe) > 1:
                model = maybe[-1]

        delta_mean = s.get("rv_delta_mean")
        delta_std = s.get("rv_delta_std")
        n_pairs = s.get("n_pairs")
        dm = s.get("delta_main")
        if isinstance(dm, dict):
            if delta_mean is None:
                delta_mean = dm.get("mean")
            if delta_std is None:
                delta_std = dm.get("std")
            if n_pairs is None:
                n_pairs = dm.get("n")

        p_val = s.get("rv_p_value")
        if p_val is None:
            p_val = (((s.get("tests") or {}).get("main_effect_ttest_1samp_less_0") or {}).get("p"))

        d_val = s.get("rv_cohens_d")
        if d_val is None:
            d_val = (((s.get("tests") or {}).get("main_effect_ttest_1samp_less_0") or {}).get("cohens_d"))

        if delta_mean is None:
            continue

        rows.append(
            {
                "run_dir": str(run_dir.relative_to(repo_root)),
                "timestamp": _extract_ts(run_dir.name),
                "model_raw": str(model),
                "model_norm": _normalize_model(str(model)),
                "n_pairs": float(n_pairs) if n_pairs is not None else np.nan,
                "rv_delta_mean": float(delta_mean),
                "rv_delta_std": float(delta_std) if delta_std is not None else np.nan,
                "rv_p_value": float(p_val) if p_val is not None else np.nan,
                "rv_cohens_d": float(d_val) if d_val is not None else np.nan,
            }
        )

    if not rows:
        return {"status": "no_data"}

    all_df = pd.DataFrame(rows).sort_values(["model_norm", "timestamp"])
    latest_df = all_df.groupby("model_norm", as_index=False).tail(1).copy()

    neg_count = int((latest_df["rv_delta_mean"] < 0).sum())
    total_models = int(len(latest_df))
    sign_test_p = float(
        stats.binomtest(neg_count, total_models, p=0.5, alternative="greater").pvalue
    ) if total_models > 0 else float("nan")

    usable = latest_df.dropna(subset=["rv_delta_std", "n_pairs"]).copy()
    usable = usable[usable["n_pairs"] > 1]
    meta = {}
    if len(usable) >= 2:
        vi = (usable["rv_delta_std"] ** 2) / usable["n_pairs"]
        wi = 1.0 / vi
        fixed_mean = float((wi * usable["rv_delta_mean"]).sum() / wi.sum())
        fixed_se = float(math.sqrt(1.0 / wi.sum()))
        fixed_ci = [fixed_mean - 1.96 * fixed_se, fixed_mean + 1.96 * fixed_se]

        q = float((wi * ((usable["rv_delta_mean"] - fixed_mean) ** 2)).sum())
        df_q = int(len(usable) - 1)
        c = float(wi.sum() - (wi.pow(2).sum() / wi.sum()))
        tau2 = float(max((q - df_q) / c, 0.0)) if c > 0 else 0.0
        wi_re = 1.0 / (vi + tau2)
        re_mean = float((wi_re * usable["rv_delta_mean"]).sum() / wi_re.sum())
        re_se = float(math.sqrt(1.0 / wi_re.sum()))
        re_ci = [re_mean - 1.96 * re_se, re_mean + 1.96 * re_se]
        i2 = float(max((q - df_q) / q, 0.0) * 100.0) if q > 0 else 0.0
        q_p = float(1.0 - stats.chi2.cdf(q, df_q)) if df_q > 0 else np.nan

        meta = {
            "fixed_effect_mean_delta": fixed_mean,
            "fixed_effect_ci95": fixed_ci,
            "random_effect_mean_delta": re_mean,
            "random_effect_ci95": re_ci,
            "tau2": tau2,
            "heterogeneity_q": q,
            "heterogeneity_df": df_q,
            "heterogeneity_p": q_p,
            "i2_percent": i2,
        }

    return {
        "status": "ok",
        "all_runs": all_df.to_dict(orient="records"),
        "latest_per_model": latest_df.to_dict(orient="records"),
        "latest_models_all_negative": bool(neg_count == total_models and total_models > 0),
        "latest_negative_count": neg_count,
        "latest_total_models": total_models,
        "sign_test_p": sign_test_p,
        "meta_analysis": meta,
    }


def experiment_bridge_specificity(repo_root: Path, summary_paths: List[Path]) -> Dict[str, Any]:
    run_rows: List[Dict[str, Any]] = []
    arrays: Dict[str, np.ndarray] = {}

    for sp in summary_paths:
        s = _safe_json(sp)
        if not s or s.get("experiment") != "rv_l27_activation_patching_bridge":
            continue
        run_dir = sp.parent
        csv_path = run_dir / "per_sample.csv"
        if not csv_path.exists():
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if "rv_delta" not in df.columns:
            continue

        arr = pd.to_numeric(df["rv_delta"], errors="coerce").dropna().to_numpy(dtype=float)
        if arr.size == 0:
            continue

        hash_id = _sha256(csv_path)
        trunc_rate = np.nan
        if "baseline_truncated" in df.columns and "patched_truncated" in df.columns:
            b = df["baseline_truncated"].astype(bool).to_numpy()
            p = df["patched_truncated"].astype(bool).to_numpy()
            trunc_rate = float(np.mean(np.logical_or(b, p)))

        t_stat, p_two = stats.ttest_1samp(arr, 0.0)
        row = {
            "run_dir": str(run_dir.relative_to(repo_root)),
            "timestamp": _extract_ts(run_dir.name),
            "hash_id": hash_id,
            "version": s.get("version"),
            "patch_mode": s.get("patch_mode"),
            "donor_type": s.get("donor_type"),
            "n": int(arr.size),
            "rv_delta_mean": float(np.mean(arr)),
            "rv_delta_std": float(np.std(arr, ddof=1)) if arr.size > 1 else np.nan,
            "rv_delta_t_two_sided": float(t_stat),
            "rv_delta_p_two_sided": float(p_two),
            "cohens_d_1samp": _cohens_d_1samp(arr),
            "truncation_rate": trunc_rate,
        }
        run_rows.append(row)
        arrays[hash_id] = arr

    if not run_rows:
        return {"status": "no_data"}

    run_df = pd.DataFrame(run_rows).sort_values(["timestamp", "run_dir"])
    unique_df = run_df.drop_duplicates(subset=["hash_id"]).copy()

    def _pick(version: str, patch_mode: str, donor_type: Optional[str]) -> Optional[np.ndarray]:
        mask = (run_df["version"] == version) & (run_df["patch_mode"] == patch_mode)
        if donor_type is None:
            mask = mask & (run_df["donor_type"].isna())
        else:
            mask = mask & (run_df["donor_type"] == donor_type)
        if not mask.any():
            return None
        hash_id = run_df.loc[mask].iloc[-1]["hash_id"]
        return arrays.get(hash_id)

    v4_head = _pick("v4_gqa_headspace", "head_specific", "recursive")
    v4_random = _pick("v4_gqa_headspace", "random_head", "recursive")
    v4_baseline_donor = _pick("v4_gqa_headspace", "head_specific", "baseline")
    v2_random = _pick("v2_head_specific", "random_head", None)

    comparisons: Dict[str, Any] = {}

    def _welch(name: str, a: Optional[np.ndarray], b: Optional[np.ndarray]) -> None:
        if a is None or b is None:
            comparisons[name] = {"status": "missing"}
            return
        t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
        comparisons[name] = {
            "status": "ok",
            "n_a": int(len(a)),
            "n_b": int(len(b)),
            "mean_a": float(np.mean(a)),
            "mean_b": float(np.mean(b)),
            "delta_mean_a_minus_b": float(np.mean(a) - np.mean(b)),
            "t_welch": float(t_stat),
            "p_welch_two_sided": float(p_val),
            "cohens_d": _cohens_d_ind(a, b),
        }

    _welch("v4_head_specific_vs_v4_random_head", v4_head, v4_random)
    _welch("v4_head_specific_vs_v4_baseline_donor", v4_head, v4_baseline_donor)
    _welch("v4_random_head_vs_v4_baseline_donor", v4_random, v4_baseline_donor)
    _welch("v2_random_head_vs_v4_random_head", v2_random, v4_random)

    sign_flip = None
    if v2_random is not None and v4_random is not None:
        sign_flip = {
            "v2_random_mean": float(np.mean(v2_random)),
            "v4_random_mean": float(np.mean(v4_random)),
            "sign_changed": bool(np.sign(np.mean(v2_random)) != np.sign(np.mean(v4_random))),
        }

    return {
        "status": "ok",
        "all_runs": run_df.to_dict(orient="records"),
        "unique_runs": unique_df.to_dict(orient="records"),
        "comparisons": comparisons,
        "random_head_version_flip": sign_flip,
    }


def experiment_multi_token_truncation(summary_paths: List[Path]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for sp in summary_paths:
        s = _safe_json(sp)
        if not s or s.get("experiment") != "multi_token_bridge":
            continue
        analysis = s.get("analysis") or {}
        for temp_key, temp_data in analysis.items():
            if not isinstance(temp_data, dict):
                continue
            rows.append(
                {
                    "run_dir": str(sp.parent),
                    "timestamp": _extract_ts(sp.parent.name),
                    "version": s.get("version"),
                    "temp": temp_key,
                    "pct_truncated": temp_data.get("pct_truncated"),
                    "n_total": temp_data.get("n_total"),
                    "n_non_truncated": temp_data.get("n_non_truncated"),
                    "h1_r": temp_data.get("h1_spearman_r"),
                    "h1_p": temp_data.get("h1_spearman_p"),
                    "h1_significant": temp_data.get("h1_significant"),
                    "h2_d": temp_data.get("h2_cohens_d"),
                    "h2_p": temp_data.get("h2_p_value"),
                    "h2_significant": temp_data.get("h2_significant"),
                    "h3_r": temp_data.get("h3_point_biserial_r"),
                    "h3_p": temp_data.get("h3_point_biserial_p"),
                    "h3_significant": temp_data.get("h3_significant"),
                }
            )

    if not rows:
        return {"status": "no_data"}

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["pct_truncated", "h3_r"])
    if df.empty:
        return {"status": "insufficient_data", "points": rows}

    spearman_r, spearman_p = stats.spearmanr(df["pct_truncated"], df["h3_r"])

    high = df[df["pct_truncated"] >= 85.0]
    low = df[df["pct_truncated"] < 85.0]
    high_sig = int(high["h3_significant"].fillna(False).astype(bool).sum())
    low_sig = int(low["h3_significant"].fillna(False).astype(bool).sum())
    fisher_p = np.nan
    if len(high) > 0 and len(low) > 0:
        table = np.array(
            [
                [high_sig, len(high) - high_sig],
                [low_sig, len(low) - low_sig],
            ]
        )
        _, fisher_p = stats.fisher_exact(table)

    return {
        "status": "ok",
        "points": df.to_dict(orient="records"),
        "n_points": int(len(df)),
        "truncation_vs_h3_spearman_r": float(spearman_r),
        "truncation_vs_h3_spearman_p": float(spearman_p),
        "h3_significance_high_truncation_rate": float(high_sig / len(high)) if len(high) else np.nan,
        "h3_significance_low_truncation_rate": float(low_sig / len(low)) if len(low) else np.nan,
        "high_vs_low_h3_significance_fisher_p": float(fisher_p) if not np.isnan(fisher_p) else np.nan,
    }


def choose_candidate(
    cross_arch: Dict[str, Any],
    bridge: Dict[str, Any],
    truncation: Dict[str, Any],
) -> Dict[str, Any]:
    candidates = []

    if cross_arch.get("status") == "ok":
        meta = cross_arch.get("meta_analysis") or {}
        sign_p = float(cross_arch.get("sign_test_p", np.nan))
        re_delta = float(meta.get("random_effect_mean_delta", np.nan))
        n_models = int(cross_arch.get("latest_total_models", 0))
        score = (
            n_models
            + (abs(re_delta) * 30.0 if not np.isnan(re_delta) else 0.0)
            + (max(0.0, -math.log10(max(sign_p, 1e-300))) if not np.isnan(sign_p) else 0.0)
        )
        candidates.append(
            {
                "name": "cross_arch_universal_contraction",
                "score": score,
                "evidence": {
                    "n_models": n_models,
                    "sign_test_p": sign_p,
                    "random_effect_mean_delta": re_delta,
                },
            }
        )

    if bridge.get("status") == "ok":
        c1 = (bridge.get("comparisons") or {}).get("v4_head_specific_vs_v4_random_head", {})
        c2 = (bridge.get("comparisons") or {}).get("v4_head_specific_vs_v4_baseline_donor", {})
        p1 = float(c1.get("p_welch_two_sided", np.nan))
        p2 = float(c2.get("p_welch_two_sided", np.nan))
        d1 = float(c1.get("cohens_d", np.nan))
        d2 = float(c2.get("cohens_d", np.nan))
        score = 0.0
        for p in [p1, p2]:
            if not np.isnan(p):
                score += max(0.0, -math.log10(max(p, 1e-300)))
        for d in [d1, d2]:
            if not np.isnan(d):
                score += abs(d)
        if (bridge.get("random_head_version_flip") or {}).get("sign_changed"):
            score += 2.0
        candidates.append(
            {
                "name": "gqa_headspace_specificity_bridge",
                "score": score,
                "evidence": {
                    "head_vs_random_p": p1,
                    "head_vs_baseline_donor_p": p2,
                    "head_vs_random_d": d1,
                    "head_vs_baseline_donor_d": d2,
                    "random_head_sign_flip_v2_to_v4": (bridge.get("random_head_version_flip") or {}).get(
                        "sign_changed"
                    ),
                },
            }
        )

    if truncation.get("status") == "ok":
        p = float(truncation.get("truncation_vs_h3_spearman_p", np.nan))
        r = float(truncation.get("truncation_vs_h3_spearman_r", np.nan))
        n = int(truncation.get("n_points", 0))
        score = n * 0.3 + (max(0.0, -math.log10(max(p, 1e-300))) if not np.isnan(p) else 0.0) + abs(r)
        candidates.append(
            {
                "name": "truncation_driven_h3_instability",
                "score": score,
                "evidence": {
                    "n_points": n,
                    "truncation_vs_h3_r": r,
                    "truncation_vs_h3_p": p,
                },
            }
        )

    if not candidates:
        return {"status": "no_candidate"}

    candidates.sort(key=lambda x: x["score"], reverse=True)
    best = candidates[0]
    return {
        "status": "ok",
        "best": best,
        "ranking": candidates,
    }


def write_outputs(
    out_dir: Path,
    index_data: Dict[str, Any],
    cross_arch: Dict[str, Any],
    bridge: Dict[str, Any],
    truncation: Dict[str, Any],
    candidate: Dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    if cross_arch.get("status") == "ok":
        pd.DataFrame(cross_arch.get("all_runs", [])).to_csv(out_dir / "cross_arch_all_runs.csv", index=False)
        pd.DataFrame(cross_arch.get("latest_per_model", [])).to_csv(
            out_dir / "cross_arch_latest_per_model.csv", index=False
        )
    if bridge.get("status") == "ok":
        pd.DataFrame(bridge.get("all_runs", [])).to_csv(out_dir / "bridge_all_runs.csv", index=False)
        pd.DataFrame(bridge.get("unique_runs", [])).to_csv(out_dir / "bridge_unique_runs.csv", index=False)
    if truncation.get("status") == "ok":
        pd.DataFrame(truncation.get("points", [])).to_csv(out_dir / "multi_token_truncation_points.csv", index=False)

    summary = {
        "generated_at": datetime.now().isoformat(),
        "project_index": index_data,
        "experiments": {
            "cross_arch_meta": cross_arch,
            "bridge_specificity": bridge,
            "multi_token_truncation": truncation,
        },
        "candidate_finding": candidate,
    }

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    best = ((candidate.get("best") or {}).get("name")) if candidate.get("status") == "ok" else "none"
    report_lines = [
        "# Offline YOLO Meta-Experiment Report",
        "",
        f"- Generated: `{summary['generated_at']}`",
        f"- Candidate finding selected: `{best}`",
        "",
        "## Project Index",
        f"- Total files scanned: {index_data.get('total_files')}",
        f"- Pipeline Python files: {index_data.get('code_inventory', {}).get('pipeline_python_files')}",
        f"- Config JSON files: {index_data.get('code_inventory', {}).get('config_json_files')}",
        f"- Results summary.json files: {index_data.get('results_inventory', {}).get('summary_json_files')}",
        "",
        "## Experiment A: Cross-Architecture Meta",
        f"- Status: {cross_arch.get('status')}",
    ]

    if cross_arch.get("status") == "ok":
        report_lines.extend(
            [
                f"- Latest models: {cross_arch.get('latest_total_models')} "
                f"(negative deltas: {cross_arch.get('latest_negative_count')})",
                f"- Sign test p-value: {cross_arch.get('sign_test_p')}",
            ]
        )
        meta = cross_arch.get("meta_analysis") or {}
        if meta:
            report_lines.extend(
                [
                    f"- Random-effects mean delta: {meta.get('random_effect_mean_delta')}",
                    f"- Random-effects 95% CI: {meta.get('random_effect_ci95')}",
                    f"- Heterogeneity I2: {meta.get('i2_percent')}",
                ]
            )

    report_lines.extend(
        [
            "",
            "## Experiment B: Bridge Specificity Controls",
            f"- Status: {bridge.get('status')}",
        ]
    )
    if bridge.get("status") == "ok":
        c = bridge.get("comparisons") or {}
        for key in [
            "v4_head_specific_vs_v4_random_head",
            "v4_head_specific_vs_v4_baseline_donor",
            "v2_random_head_vs_v4_random_head",
        ]:
            if c.get(key, {}).get("status") == "ok":
                report_lines.append(
                    f"- {key}: delta_mean={c[key].get('delta_mean_a_minus_b')}, "
                    f"p={c[key].get('p_welch_two_sided')}, d={c[key].get('cohens_d')}"
                )
        flip = bridge.get("random_head_version_flip")
        if flip:
            report_lines.append(
                f"- Random-head sign flip v2->v4: {flip.get('sign_changed')} "
                f"(v2 mean={flip.get('v2_random_mean')}, v4 mean={flip.get('v4_random_mean')})"
            )

    report_lines.extend(
        [
            "",
            "## Experiment C: Multi-Token Truncation Stress",
            f"- Status: {truncation.get('status')}",
        ]
    )
    if truncation.get("status") == "ok":
        report_lines.extend(
            [
                f"- Points analyzed: {truncation.get('n_points')}",
                f"- Truncation vs H3 rho: {truncation.get('truncation_vs_h3_spearman_r')}",
                f"- Truncation vs H3 p-value: {truncation.get('truncation_vs_h3_spearman_p')}",
                f"- H3 significance rate @ high truncation: "
                f"{truncation.get('h3_significance_high_truncation_rate')}",
                f"- H3 significance rate @ low truncation: "
                f"{truncation.get('h3_significance_low_truncation_rate')}",
            ]
        )

    report_lines.extend(
        [
            "",
            "## Candidate Ranking",
        ]
    )
    if candidate.get("status") == "ok":
        for i, cnd in enumerate(candidate.get("ranking", []), start=1):
            report_lines.append(f"{i}. {cnd.get('name')} (score={cnd.get('score')})")
            report_lines.append(f"   evidence={cnd.get('evidence')}")
    else:
        report_lines.append("- No candidate selected.")

    (out_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline index + YOLO meta-experiments")
    parser.add_argument("--repo-root", default=".", help="Path to repository root")
    parser.add_argument(
        "--output-root",
        default="results/meta_yolo/runs",
        help="Root folder for outputs",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    output_root = (repo_root / args.output_root).resolve()
    run_dir = output_root / f"{_now_ts()}_offline_meta_yolo"

    summary_paths = list((repo_root / "results").rglob("summary.json"))
    index_data = index_project(repo_root, summary_paths)
    cross_arch = experiment_cross_arch(repo_root, summary_paths)
    bridge = experiment_bridge_specificity(repo_root, summary_paths)
    truncation = experiment_multi_token_truncation(summary_paths)
    candidate = choose_candidate(cross_arch, bridge, truncation)
    write_outputs(run_dir, index_data, cross_arch, bridge, truncation, candidate)

    print(json.dumps({"status": "ok", "run_dir": str(run_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
