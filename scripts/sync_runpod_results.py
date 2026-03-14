#!/usr/bin/env python3
"""Sync and summarize RunPod experiment results, cross-referencing paper claims.

Reads JSON results from three RunPod experiment directories:
  1. results/full_head_sweep/       (E2.2 full 1024-head sweep)
  2. results/path_patching/         (full layer x component path patching)
  3. results/persistent_patching_v3/ (dual-layer bridge: L18 residual + L27 V-proj)

For each directory, prints a summary and cross-references the COLM 2026 paper
claims. Outputs a timestamped Markdown report.

This script is pure Python -- no model loading, no GPU, no torch.

Usage:
    python3 scripts/sync_runpod_results.py
    python3 scripts/sync_runpod_results.py --results-dir /path/to/results
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"

# Paper claims (from paper_colm2026_v005.tex and CLAUDE.md / MEMORY.md)
PAPER_CLAIMS = {
    "head_sweep_significant_count": 606,
    "head_sweep_total": 1024,
    "head_sweep_sig_pct": 59.2,
    "head_sweep_top_head_location": "L10H20",
    "head_sweep_top_head_d": 3.90,
    "head_sweep_strongest_cluster": "L8--L14",
    "path_patching_vproj_max_abs_d": 0.22,
    "path_patching_early_residual_top_layer": 4,
    "path_patching_early_residual_d": 1.96,
    "dual_layer_necessity_or": 33.4,
    "dual_layer_break_bt_art_from": 0.56,
    "dual_layer_break_bt_art_to": 0.037,
    "dual_layer_cohens_h": 1.31,
    "dual_layer_kv_transfer_or": 13.96,
    "dual_layer_rv_transfer_d": 0.11,
    "dual_layer_d_session": 3.29,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class Discrepancy:
    """One claim vs. data comparison."""

    claim_label: str
    paper_value: float | str
    data_value: float | str | None
    match: bool
    note: str = ""


def _load_json_safe(path: Path) -> dict[str, Any] | None:
    """Load a JSON file, returning None on any error."""
    try:
        with open(path) as f:
            text = f.read()
        # Handle NaN/Infinity which are not valid JSON but numpy writes them
        text = text.replace(": NaN", ": null")
        text = text.replace(": Infinity", ": null")
        text = text.replace(": -Infinity", ": null")
        return json.loads(text)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"  [WARN] Could not load {path}: {exc}")
        return None


def _find_latest_json(directory: Path, prefix: str = "") -> Path | None:
    """Find the most recently modified JSON file in a directory."""
    if not directory.is_dir():
        return None
    jsons = sorted(directory.glob(f"{prefix}*.json"), key=lambda p: p.stat().st_mtime)
    return jsons[-1] if jsons else None


def _find_all_jsons(directory: Path, prefix: str = "") -> list[Path]:
    """Return all JSON files in a directory, sorted by modification time."""
    if not directory.is_dir():
        return []
    return sorted(directory.glob(f"{prefix}*.json"), key=lambda p: p.stat().st_mtime)


def _approx_equal(a: float, b: float, rtol: float = 0.05, atol: float = 0.01) -> bool:
    """Check approximate equality with relative and absolute tolerance."""
    if math.isnan(a) or math.isnan(b):
        return False
    return abs(a - b) <= max(atol, rtol * max(abs(a), abs(b)))


def _fmt(val: float | str | None, decimals: int = 3) -> str:
    """Format a numeric value for display."""
    if val is None:
        return "N/A"
    if isinstance(val, str):
        return val
    if math.isnan(val) or math.isinf(val):
        return str(val)
    return f"{val:.{decimals}f}"


# ---------------------------------------------------------------------------
# Head Sweep Analysis
# ---------------------------------------------------------------------------

def analyze_head_sweep(results_dir: Path) -> tuple[list[str], list[Discrepancy]]:
    """Analyze full head sweep results.

    Returns:
        Tuple of (summary_lines, discrepancies).
    """
    lines: list[str] = []
    discs: list[Discrepancy] = []
    sweep_dir = results_dir / "full_head_sweep"

    if not sweep_dir.is_dir():
        lines.append("Directory `results/full_head_sweep/` does not exist.")
        lines.append("Run the head sweep experiment on RunPod first.")
        return lines, discs

    all_files = _find_all_jsons(sweep_dir, prefix="full_head_sweep")
    if not all_files:
        lines.append("No `full_head_sweep_*.json` files found.")
        return lines, discs

    lines.append(f"Found {len(all_files)} result file(s):")
    for f in all_files:
        lines.append(f"  - `{f.name}`")

    latest = all_files[-1]
    data = _load_json_safe(latest)
    if data is None:
        lines.append(f"Could not parse latest file: `{latest.name}`")
        return lines, discs

    lines.append(f"\nAnalyzing latest: `{latest.name}`")
    lines.append(f"Model: {data.get('model', 'unknown')}")
    lines.append(f"Prompt bank: {data.get('prompt_bank_version', 'unknown')}")
    lines.append(f"Prompt subset: {data.get('prompt_subset_name', 'unknown')}")

    n_layers = data.get("n_layers", 0)
    n_heads = data.get("n_heads", 0)
    total_heads = n_layers * n_heads
    n_recursive = data.get("n_recursive_prompts", 0)
    n_baseline = data.get("n_baseline_prompts", 0)

    lines.append(f"Layers: {n_layers}, Heads/layer: {n_heads}, Total heads: {total_heads}")
    lines.append(f"Prompts: {n_recursive} recursive, {n_baseline} baseline")

    head_results: list[dict[str, Any]] = data.get("head_results", [])
    if not head_results:
        lines.append("[WARN] No head_results array found.")
        return lines, discs

    # Count significant heads (p < 0.05 on either metric)
    sig_entropy = 0
    sig_rank = 0
    sig_either = 0
    for hr in head_results:
        ent_p = hr.get("entropy_p")
        rank_p = hr.get("rank_p")
        e_sig = ent_p is not None and ent_p < 0.05
        r_sig = rank_p is not None and rank_p < 0.05
        if e_sig:
            sig_entropy += 1
        if r_sig:
            sig_rank += 1
        if e_sig or r_sig:
            sig_either += 1

    lines.append(f"\nSignificant heads (p < 0.05, uncorrected):")
    lines.append(f"  Entropy metric:  {sig_entropy}/{total_heads} ({100*sig_entropy/max(total_heads,1):.1f}%)")
    lines.append(f"  OV rank metric:  {sig_rank}/{total_heads} ({100*sig_rank/max(total_heads,1):.1f}%)")
    lines.append(f"  Either metric:   {sig_either}/{total_heads} ({100*sig_either/max(total_heads,1):.1f}%)")

    # Paper claims 606/1024 -- compare
    discs.append(Discrepancy(
        claim_label="Head sweep: significant heads",
        paper_value=f"{PAPER_CLAIMS['head_sweep_significant_count']}/{PAPER_CLAIMS['head_sweep_total']} ({PAPER_CLAIMS['head_sweep_sig_pct']}%)",
        data_value=f"{sig_either}/{total_heads} ({100*sig_either/max(total_heads,1):.1f}%)",
        match=(sig_either == PAPER_CLAIMS["head_sweep_significant_count"] and total_heads == PAPER_CLAIMS["head_sweep_total"]),
        note="Counting heads significant on entropy OR rank metric at p<0.05 uncorrected.",
    ))

    # Top 5 heads by |d| (entropy)
    sorted_ent = sorted(
        head_results,
        key=lambda r: abs(r.get("entropy_d", 0) or 0),
        reverse=True,
    )
    lines.append(f"\nTop 5 heads by |entropy d|:")
    lines.append(f"  {'Head':>8} {'d_entropy':>10} {'p_entropy':>12} {'d_rank':>10}")
    lines.append(f"  {'-'*44}")
    for hr in sorted_ent[:5]:
        loc = f"L{hr['layer']:02d}.H{hr['head']:02d}"
        d_e = hr.get("entropy_d", float("nan"))
        p_e = hr.get("entropy_p", float("nan"))
        d_r = hr.get("rank_d", float("nan"))
        lines.append(f"  {loc:>8} {_fmt(d_e):>10} {_fmt(p_e, 6):>12} {_fmt(d_r):>10}")

    # Top 5 heads by |d| (OV rank)
    sorted_rank = sorted(
        head_results,
        key=lambda r: abs(r.get("rank_d", 0) or 0),
        reverse=True,
    )
    lines.append(f"\nTop 5 heads by |OV rank d|:")
    lines.append(f"  {'Head':>8} {'d_rank':>10} {'p_rank':>12} {'d_entropy':>10}")
    lines.append(f"  {'-'*44}")
    for hr in sorted_rank[:5]:
        loc = f"L{hr['layer']:02d}.H{hr['head']:02d}"
        d_r = hr.get("rank_d", float("nan"))
        p_r = hr.get("rank_p", float("nan"))
        d_e = hr.get("entropy_d", float("nan"))
        lines.append(f"  {loc:>8} {_fmt(d_r):>10} {_fmt(p_r, 6):>12} {_fmt(d_e):>10}")

    # Check paper claim about top head (L10H20, d=3.90)
    if sorted_ent:
        top = sorted_ent[0]
        top_loc = f"L{top['layer']}H{top['head']}"
        top_d = abs(top.get("entropy_d", 0) or 0)
        paper_top = PAPER_CLAIMS["head_sweep_top_head_location"]
        paper_d = PAPER_CLAIMS["head_sweep_top_head_d"]
        discs.append(Discrepancy(
            claim_label="Head sweep: top head location",
            paper_value=f"{paper_top} (|d|={paper_d})",
            data_value=f"{top_loc} (|d|={top_d:.3f})",
            match=(top_loc == paper_top and _approx_equal(top_d, paper_d, rtol=0.1)),
        ))

    # Layer-averaged |d| to identify strongest cluster
    layer_avg_d: dict[int, float] = {}
    for layer_idx in range(n_layers):
        layer_heads = [hr for hr in head_results if hr.get("layer") == layer_idx]
        abs_ds = [
            abs(hr.get("entropy_d", 0) or 0)
            for hr in layer_heads
            if hr.get("entropy_d") is not None and not math.isnan(hr.get("entropy_d", float("nan")))
        ]
        if abs_ds:
            layer_avg_d[layer_idx] = sum(abs_ds) / len(abs_ds)

    if layer_avg_d:
        top_layers = sorted(layer_avg_d.items(), key=lambda kv: kv[1], reverse=True)[:5]
        lines.append(f"\nTop 5 layers by avg |entropy d|:")
        for lay, avg_d in top_layers:
            lines.append(f"  L{lay:02d}: avg |d| = {avg_d:.3f}")

    return lines, discs


# ---------------------------------------------------------------------------
# Path Patching Analysis
# ---------------------------------------------------------------------------

def analyze_path_patching(results_dir: Path) -> tuple[list[str], list[Discrepancy]]:
    """Analyze full path patching results.

    Returns:
        Tuple of (summary_lines, discrepancies).
    """
    lines: list[str] = []
    discs: list[Discrepancy] = []
    pp_dir = results_dir / "path_patching"

    if not pp_dir.is_dir():
        lines.append("Directory `results/path_patching/` does not exist.")
        lines.append("Run the path patching experiment on RunPod first.")
        return lines, discs

    all_files = _find_all_jsons(pp_dir, prefix="path_patching")
    if not all_files:
        lines.append("No `path_patching_*.json` files found.")
        return lines, discs

    lines.append(f"Found {len(all_files)} result file(s):")
    for f in all_files:
        lines.append(f"  - `{f.name}`")

    latest = all_files[-1]
    data = _load_json_safe(latest)
    if data is None:
        lines.append(f"Could not parse latest file: `{latest.name}`")
        return lines, discs

    lines.append(f"\nAnalyzing latest: `{latest.name}`")
    lines.append(f"Model: {data.get('model', 'unknown')}")
    lines.append(f"Target layers: {data.get('target_layers', [])}")
    lines.append(f"Components: {data.get('components', [])}")
    lines.append(f"N prompts: {data.get('n_prompts', 0)}")

    results: list[dict[str, Any]] = data.get("results", [])
    if not results:
        lines.append("[WARN] No results array found.")
        return lines, discs

    # Build layer x component table
    components = sorted(set(r.get("component", "") for r in results))
    layers = sorted(set(r.get("layer", 0) for r in results))

    lines.append(f"\nPath patching heatmap (Cohen's d, break direction):")
    header = f"  {'Layer':>6}"
    for comp in components:
        header += f"  {comp:>12}"
    lines.append(header)
    lines.append(f"  {'-' * (8 + 14 * len(components))}")

    for layer_idx in layers:
        row = f"  L{layer_idx:>4}"
        for comp in components:
            r = next(
                (r for r in results if r.get("layer") == layer_idx and r.get("component") == comp),
                None,
            )
            if r and r.get("cohens_d") is not None:
                d = r["cohens_d"]
                sig = "***" if abs(d) > 1.0 else " * " if abs(d) > 0.5 else "   "
                row += f"  {d:>+8.3f}{sig}"
            else:
                row += f"  {'N/A':>12}"
        lines.append(row)

    # Find top causal layer/component
    valid_results = [
        r for r in results
        if r.get("cohens_d") is not None and not math.isnan(r.get("cohens_d", float("nan")))
    ]

    if valid_results:
        # Sort by |d| descending
        sorted_by_d = sorted(valid_results, key=lambda r: abs(r["cohens_d"]), reverse=True)
        lines.append(f"\nTop 5 causal sites by |d|:")
        lines.append(f"  {'Layer':>6} {'Component':>12} {'d':>10} {'delta_rv':>10}")
        lines.append(f"  {'-'*42}")
        for r in sorted_by_d[:5]:
            lines.append(
                f"  L{r['layer']:>4} {r['component']:>12} {r['cohens_d']:>+10.3f} {_fmt(r.get('delta_rv')):>10}"
            )

        # V-proj max |d| -- paper claims 0.22
        vproj_results = [r for r in valid_results if r.get("component") == "v_proj"]
        if vproj_results:
            vproj_top = max(vproj_results, key=lambda r: abs(r["cohens_d"]))
            vproj_max_d = abs(vproj_top["cohens_d"])
            lines.append(f"\nV-proj max |d|: {vproj_max_d:.3f} at L{vproj_top['layer']}")
            discs.append(Discrepancy(
                claim_label="Path patching: V-proj max |d|",
                paper_value=PAPER_CLAIMS["path_patching_vproj_max_abs_d"],
                data_value=round(vproj_max_d, 3),
                match=_approx_equal(vproj_max_d, PAPER_CLAIMS["path_patching_vproj_max_abs_d"], rtol=0.15, atol=0.05),
                note=f"V-proj top site: L{vproj_top['layer']}",
            ))

        # Residual top site -- paper claims L4 d=1.96
        residual_results = [r for r in valid_results if r.get("component") == "residual"]
        if residual_results:
            residual_top = max(residual_results, key=lambda r: abs(r["cohens_d"]))
            residual_max_d = residual_top["cohens_d"]
            residual_top_layer = residual_top["layer"]
            lines.append(f"Residual top |d|: {abs(residual_max_d):.3f} at L{residual_top_layer}")

            paper_layer = PAPER_CLAIMS["path_patching_early_residual_top_layer"]
            paper_d = PAPER_CLAIMS["path_patching_early_residual_d"]
            discs.append(Discrepancy(
                claim_label="Path patching: top residual causal site",
                paper_value=f"L{paper_layer} (d={paper_d})",
                data_value=f"L{residual_top_layer} (d={residual_max_d:+.3f})",
                match=(residual_top_layer == paper_layer and _approx_equal(abs(residual_max_d), paper_d, rtol=0.1)),
                note="Paper claims L4 residual d=1.96 is top causal site.",
            ))

        # MLP summary
        mlp_results = [r for r in valid_results if r.get("component") == "mlp"]
        if mlp_results:
            mlp_top = max(mlp_results, key=lambda r: abs(r["cohens_d"]))
            lines.append(f"MLP top |d|: {abs(mlp_top['cohens_d']):.3f} at L{mlp_top['layer']}")

    return lines, discs


# ---------------------------------------------------------------------------
# Dual Layer Bridge Analysis
# ---------------------------------------------------------------------------

def analyze_dual_layer_bridge(results_dir: Path) -> tuple[list[str], list[Discrepancy]]:
    """Analyze dual-layer bridge (persistent patching v3) results.

    Searches both results/dual_layer_bridge/ and results/persistent_patching_v3/
    since the experiment script writes to the latter.

    Returns:
        Tuple of (summary_lines, discrepancies).
    """
    lines: list[str] = []
    discs: list[Discrepancy] = []

    # Check both possible directories
    candidates = [
        results_dir / "dual_layer_bridge",
        results_dir / "persistent_patching_v3",
    ]
    all_files: list[Path] = []
    found_dirs: list[str] = []
    for d in candidates:
        if d.is_dir():
            found_dirs.append(str(d.relative_to(REPO_ROOT)))
            jsons = _find_all_jsons(d, prefix="persistent_patching_v3_dual")
            # Also check for any JSON that might have a different prefix
            if not jsons:
                jsons = _find_all_jsons(d)
            all_files.extend(jsons)

    if not found_dirs:
        lines.append("Neither `results/dual_layer_bridge/` nor `results/persistent_patching_v3/` exists.")
        lines.append("Run the dual-layer bridge experiment on RunPod first.")
        return lines, discs

    lines.append(f"Searched directories: {', '.join(found_dirs)}")

    if not all_files:
        lines.append("No result JSON files found in searched directories.")
        return lines, discs

    # Deduplicate by filename
    seen: set[str] = set()
    unique_files: list[Path] = []
    for f in all_files:
        if f.name not in seen:
            seen.add(f.name)
            unique_files.append(f)
    unique_files.sort(key=lambda p: p.stat().st_mtime)

    lines.append(f"Found {len(unique_files)} result file(s):")
    for f in unique_files:
        lines.append(f"  - `{f.name}`")

    # Use the latest file with meaningful data (n_sessions > 1)
    best_data: dict[str, Any] | None = None
    best_path: Path | None = None

    for f in reversed(unique_files):
        data = _load_json_safe(f)
        if data is None:
            continue
        n_sess = data.get("n_sessions_per_condition", 0)
        agg = data.get("aggregated", {})
        # Prefer files with more sessions (the production run has n=10)
        if best_data is None or n_sess > best_data.get("n_sessions_per_condition", 0):
            best_data = data
            best_path = f

    if best_data is None or best_path is None:
        lines.append("Could not parse any result files.")
        return lines, discs

    data = best_data
    lines.append(f"\nAnalyzing best file (most sessions): `{best_path.name}`")
    lines.append(f"Model: {data.get('model', 'unknown')}")
    lines.append(f"Experiment: {data.get('experiment', 'unknown')}")
    n_sessions = data.get("n_sessions_per_condition", 0)
    max_turns = data.get("max_turns_per_session", 0)
    lines.append(f"Sessions per condition: {n_sessions}")
    lines.append(f"Max turns per session: {max_turns}")
    lines.append(f"Total turns per condition: {n_sessions * max_turns}")
    lines.append(f"V-layer: {data.get('v_layer')}, R-layer: {data.get('r_layer')}")

    # Aggregated condition summaries
    agg = data.get("aggregated", {})
    if agg:
        lines.append(f"\nCondition summaries:")
        lines.append(f"  {'Condition':>28} {'BT+ART rate':>12} {'Mean R_V':>10} {'N turns':>8}")
        lines.append(f"  {'-'*62}")
        for cond_key in ["recursive_clean", "recursive_dual_patched", "baseline_clean", "baseline_dual_patched"]:
            c = agg.get(cond_key, {})
            label_map = {
                "recursive_clean": "A: recursive_clean",
                "recursive_dual_patched": "B: recursive_dual_patched",
                "baseline_clean": "C: baseline_clean",
                "baseline_dual_patched": "D: baseline_dual_patched",
            }
            label = label_map.get(cond_key, cond_key)
            bt_rate = c.get("bt_art_rate")
            mean_rv = c.get("mean_rv")
            n_turns = c.get("total_turns", 0)
            lines.append(
                f"  {label:>28} {_fmt(bt_rate):>12} {_fmt(mean_rv):>10} {n_turns:>8}"
            )

    # Comparisons
    comparisons = data.get("comparisons", {})

    # Break test (A vs B)
    break_test = comparisons.get("break_test", {})
    if break_test:
        turn_level = break_test.get("turn_level", {})
        session_level = break_test.get("session_level", {})

        a_rate = turn_level.get("a_rate")
        b_rate = turn_level.get("b_rate")
        break_or = turn_level.get("or")
        break_p = turn_level.get("p")
        break_d = session_level.get("cohens_d")

        lines.append(f"\nBreak test (A vs B):")
        lines.append(f"  A (recursive clean) BT+ART rate: {_fmt(a_rate)}")
        lines.append(f"  B (recursive patched) BT+ART rate: {_fmt(b_rate)}")
        lines.append(f"  Odds ratio: {_fmt(break_or)}")
        lines.append(f"  Turn-level p: {_fmt(break_p, 6)}")
        lines.append(f"  Session-level Cohen's d: {_fmt(break_d)}")

        # Paper claims: OR=33.4, bt_art drops from 56% to 3.7%, d=3.29
        if a_rate is not None and b_rate is not None:
            discs.append(Discrepancy(
                claim_label="Dual layer: BT+ART rate (recursive clean)",
                paper_value=f"{PAPER_CLAIMS['dual_layer_break_bt_art_from']*100:.0f}%",
                data_value=f"{a_rate*100:.1f}%",
                match=_approx_equal(a_rate, PAPER_CLAIMS["dual_layer_break_bt_art_from"], rtol=0.15),
            ))
            discs.append(Discrepancy(
                claim_label="Dual layer: BT+ART rate (recursive patched)",
                paper_value=f"{PAPER_CLAIMS['dual_layer_break_bt_art_to']*100:.1f}%",
                data_value=f"{b_rate*100:.1f}%",
                match=_approx_equal(b_rate, PAPER_CLAIMS["dual_layer_break_bt_art_to"], rtol=0.2, atol=0.02),
            ))

        if break_or is not None and not math.isnan(break_or) and not math.isinf(break_or):
            discs.append(Discrepancy(
                claim_label="Dual layer: necessity OR",
                paper_value=PAPER_CLAIMS["dual_layer_necessity_or"],
                data_value=round(break_or, 1),
                match=_approx_equal(break_or, PAPER_CLAIMS["dual_layer_necessity_or"], rtol=0.1),
            ))

        if break_d is not None and not math.isnan(break_d):
            discs.append(Discrepancy(
                claim_label="Dual layer: session-level Cohen's d (break)",
                paper_value=PAPER_CLAIMS["dual_layer_d_session"],
                data_value=round(break_d, 2),
                match=_approx_equal(break_d, PAPER_CLAIMS["dual_layer_d_session"], rtol=0.15),
            ))

    # Induce test (C vs D)
    induce_test = comparisons.get("induce_test", {})
    if induce_test:
        turn_level = induce_test.get("turn_level", {})
        session_level = induce_test.get("session_level", {})

        c_rate = turn_level.get("c_rate")
        d_rate = turn_level.get("d_rate")
        induce_or = turn_level.get("or")
        induce_d = session_level.get("cohens_d")

        lines.append(f"\nInduce test (C vs D):")
        lines.append(f"  C (baseline clean) BT+ART rate: {_fmt(c_rate)}")
        lines.append(f"  D (baseline patched) BT+ART rate: {_fmt(d_rate)}")
        lines.append(f"  Odds ratio: {_fmt(induce_or)}")
        lines.append(f"  Session-level Cohen's d: {_fmt(induce_d)}")

        if induce_or is not None and not math.isnan(induce_or) and not math.isinf(induce_or):
            discs.append(Discrepancy(
                claim_label="Dual layer: KV transfer OR (induce)",
                paper_value=PAPER_CLAIMS["dual_layer_kv_transfer_or"],
                data_value=round(induce_or, 2),
                match=_approx_equal(induce_or, PAPER_CLAIMS["dual_layer_kv_transfer_or"], rtol=0.15),
            ))

    # R_V session contrasts
    rv_contrasts = comparisons.get("rv_session_contrasts", {})
    if rv_contrasts:
        lines.append(f"\nR_V session contrasts:")
        for direction in ["break", "induce"]:
            rc = rv_contrasts.get(direction, {})
            if rc:
                mean_diff = rc.get("mean_diff")
                cohens_d_val = rc.get("cohens_d")
                perm_p = rc.get("permutation_p")
                lines.append(f"  {direction}: mean_diff={_fmt(mean_diff)}, d={_fmt(cohens_d_val)}, perm_p={_fmt(perm_p, 6)}")

        induce_rv = rv_contrasts.get("induce", {})
        rv_d = induce_rv.get("cohens_d")
        if rv_d is not None and not math.isnan(rv_d):
            lines.append(
                "  Note: dual-layer induce R_V shift is not directly comparable to the "
                "paper's KV-only geometry/behavior dissociation claim (d=0.11, NS), "
                "which comes from a different experiment family."
            )

    # Sanity check
    sanity = comparisons.get("sanity", {})
    if sanity:
        san_turn = sanity.get("turn_level", {})
        san_or = san_turn.get("or")
        san_p = san_turn.get("p")
        lines.append(f"\nSanity check (A vs C): OR={_fmt(san_or)}, p={_fmt(san_p, 6)}")

    return lines, discs


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------

def generate_report(
    results_dir: Path,
    head_lines: list[str],
    head_discs: list[Discrepancy],
    pp_lines: list[str],
    pp_discs: list[Discrepancy],
    dl_lines: list[str],
    dl_discs: list[Discrepancy],
) -> str:
    """Generate a Markdown report from analysis results."""
    now = datetime.now(timezone.utc)
    all_discs = head_discs + pp_discs + dl_discs
    n_match = sum(1 for d in all_discs if d.match)
    n_mismatch = sum(1 for d in all_discs if not d.match)

    report: list[str] = []
    report.append(f"# RunPod Results Sync Report")
    report.append(f"")
    report.append(f"**Generated**: {now.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    report.append(f"**Results directory**: `{results_dir}`")
    report.append(f"**Paper reference**: `R_V_PAPER/paper_colm2026_v005.tex`")
    report.append(f"")

    # Executive summary
    report.append(f"## Executive Summary")
    report.append(f"")
    report.append(f"| Metric | Value |")
    report.append(f"|--------|-------|")
    report.append(f"| Total claims checked | {len(all_discs)} |")
    report.append(f"| Claims matching paper | {n_match} |")
    report.append(f"| Claims diverging from paper | {n_mismatch} |")
    if all_discs:
        report.append(f"| Match rate | {100*n_match/len(all_discs):.0f}% |")
    report.append(f"")

    # Head sweep
    report.append(f"---")
    report.append(f"")
    report.append(f"## 1. Full Head Sweep (E2.2)")
    report.append(f"")
    for line in head_lines:
        report.append(line)
    report.append(f"")

    # Path patching
    report.append(f"---")
    report.append(f"")
    report.append(f"## 2. Full Path Patching")
    report.append(f"")
    for line in pp_lines:
        report.append(line)
    report.append(f"")

    # Dual layer bridge
    report.append(f"---")
    report.append(f"")
    report.append(f"## 3. Dual-Layer Bridge (L18 Residual + L27 V-proj)")
    report.append(f"")
    for line in dl_lines:
        report.append(line)
    report.append(f"")

    # Cross-reference table
    report.append(f"---")
    report.append(f"")
    report.append(f"## 4. Paper Claim Cross-Reference")
    report.append(f"")
    if all_discs:
        report.append(f"| # | Claim | Paper Value | Data Value | Match |")
        report.append(f"|---|-------|-------------|------------|-------|")
        for i, d in enumerate(all_discs, 1):
            status = "YES" if d.match else "**NO**"
            note = f" -- {d.note}" if d.note and not d.match else ""
            report.append(f"| {i} | {d.claim_label} | {d.paper_value} | {d.data_value} | {status}{note} |")
    else:
        report.append(f"No claims could be checked (no data found).")
    report.append(f"")

    # Action items for mismatches
    mismatches = [d for d in all_discs if not d.match]
    if mismatches:
        report.append(f"---")
        report.append(f"")
        report.append(f"## 5. Action Items (Mismatches)")
        report.append(f"")
        for d in mismatches:
            report.append(f"- **{d.claim_label}**: Paper says `{d.paper_value}`, data shows `{d.data_value}`.")
            if d.note:
                report.append(f"  - Note: {d.note}")
            report.append(f"  - Action: Investigate whether this is a new run, different prompt set, or genuine discrepancy.")
        report.append(f"")

    report.append(f"---")
    report.append(f"*Report generated by `scripts/sync_runpod_results.py`*")
    return "\n".join(report)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """Run the sync and report pipeline.

    Returns:
        Exit code: 0 if all claims match, 1 if any mismatch, 2 if no data.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Sync and summarize RunPod experiment results against paper claims.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"Path to results directory (default: {DEFAULT_RESULTS_DIR})",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip writing the Markdown report file.",
    )
    args = parser.parse_args()

    results_dir: Path = args.results_dir.resolve()
    print(f"{'='*70}")
    print(f"  RUNPOD RESULTS SYNC")
    print(f"  Results dir: {results_dir}")
    print(f"  Timestamp:   {datetime.now().isoformat()}")
    print(f"{'='*70}")

    if not results_dir.is_dir():
        print(f"\n[ERROR] Results directory does not exist: {results_dir}")
        return 2

    # ── 1. Head Sweep ──
    print(f"\n{'='*70}")
    print(f"  1. FULL HEAD SWEEP (E2.2)")
    print(f"{'='*70}")
    head_lines, head_discs = analyze_head_sweep(results_dir)
    for line in head_lines:
        print(f"  {line}")

    # ── 2. Path Patching ──
    print(f"\n{'='*70}")
    print(f"  2. FULL PATH PATCHING")
    print(f"{'='*70}")
    pp_lines, pp_discs = analyze_path_patching(results_dir)
    for line in pp_lines:
        print(f"  {line}")

    # ── 3. Dual Layer Bridge ──
    print(f"\n{'='*70}")
    print(f"  3. DUAL-LAYER BRIDGE")
    print(f"{'='*70}")
    dl_lines, dl_discs = analyze_dual_layer_bridge(results_dir)
    for line in dl_lines:
        print(f"  {line}")

    # ── 4. Cross-Reference Summary ──
    all_discs = head_discs + pp_discs + dl_discs
    n_match = sum(1 for d in all_discs if d.match)
    n_mismatch = sum(1 for d in all_discs if not d.match)

    print(f"\n{'='*70}")
    print(f"  4. PAPER CLAIM CROSS-REFERENCE")
    print(f"{'='*70}")
    if all_discs:
        for d in all_discs:
            status = "MATCH" if d.match else "MISMATCH"
            print(f"  [{status:>8}] {d.claim_label}")
            print(f"           Paper: {d.paper_value}")
            print(f"           Data:  {d.data_value}")
            if d.note:
                print(f"           Note:  {d.note}")
        print(f"\n  Total: {n_match} match, {n_mismatch} mismatch out of {len(all_discs)} claims")
    else:
        print(f"  No claims could be checked (no data found in any directory).")

    # ── 5. Write report ──
    if not args.no_report:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = results_dir / f"runpod_sync_report_{timestamp}.md"
        report_content = generate_report(
            results_dir, head_lines, head_discs, pp_lines, pp_discs, dl_lines, dl_discs,
        )
        report_path.write_text(report_content, encoding="utf-8")
        print(f"\n  Report written to: {report_path}")

    # Exit code
    if not all_discs:
        return 2
    return 0 if n_mismatch == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
