#!/usr/bin/env python3
"""
Diagnostic Token Logit Lift Analysis

Measures whether V_PROJ patching at L27 increases the logit probability of
recursion-related tokens, even when the model doesn't generate them overtly.

This is a "latent behavioral" signal: elevated logit probability for tokens
like "observer", "awareness", "self" shows the intervention is pushing the
model TOWARD recursive content, even if generation doesn't cross the threshold.

Uses existing per_sample.csv from Run 2 (activation patching bridge).
Requires re-running with logit capture, OR can work with the existing
logit_diff values as a proxy.

Usage:
    python scripts/diagnostic_token_logit_lift.py \
        --csv results/phase1_mechanism/runs/.../per_sample.csv
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import numpy as np
import pandas as pd


# Preregistered diagnostic tokens for recursive self-reference
DIAGNOSTIC_TOKENS = [
    "observer", "awareness", "self", "witness", "recursive",
    "attention", "consciousness", "meta", "mirror", "watching",
]

# Tokens that should NOT increase (negative control)
CONTROL_TOKENS = [
    "calculate", "equation", "answer", "therefore", "formula",
    "percent", "number", "result", "step", "solve",
]


def analyze_per_sample(csv_path: Path) -> Dict[str, Any]:
    """Analyze per-sample data for behavioral signals beyond keyword matching.

    This uses available columns from the bridge experiment CSV to detect
    latent behavioral signals.

    Args:
        csv_path: Path to per_sample.csv from bridge experiment.

    Returns:
        Dictionary with analysis results.
    """
    df = pd.read_csv(csv_path)

    results: Dict[str, Any] = {
        "n_samples": len(df),
        "available_columns": list(df.columns),
    }

    # 1. R_V analysis (should be strong)
    if "rv_delta" in df.columns:
        rv_deltas = df["rv_delta"].dropna()
        results["rv_analysis"] = {
            "mean_delta": float(rv_deltas.mean()),
            "std_delta": float(rv_deltas.std()),
            "n_negative": int((rv_deltas < 0).sum()),
            "n_positive": int((rv_deltas > 0).sum()),
            "pct_negative": float((rv_deltas < 0).mean() * 100),
        }

    # 2. Logit diff analysis
    if "logit_diff_delta" in df.columns:
        ld = df["logit_diff_delta"].dropna()
        results["logit_diff_analysis"] = {
            "mean_delta": float(ld.mean()),
            "std_delta": float(ld.std()),
            "n_positive": int((ld > 0).sum()),
            "pct_positive": float((ld > 0).mean() * 100),
        }

    # 3. Generated text analysis: search for diagnostic tokens in patched vs baseline
    if "patched_output" in df.columns and "baseline_output" in df.columns:
        diagnostic_lift = {}
        control_lift = {}

        for token in DIAGNOSTIC_TOKENS:
            patched_count = df["patched_output"].str.lower().str.count(token).sum()
            baseline_count = df["baseline_output"].str.lower().str.count(token).sum()
            lift = patched_count - baseline_count
            diagnostic_lift[token] = {
                "patched_count": int(patched_count),
                "baseline_count": int(baseline_count),
                "lift": int(lift),
                "lift_ratio": float(patched_count / max(baseline_count, 1)),
            }

        for token in CONTROL_TOKENS:
            patched_count = df["patched_output"].str.lower().str.count(token).sum()
            baseline_count = df["baseline_output"].str.lower().str.count(token).sum()
            lift = patched_count - baseline_count
            control_lift[token] = {
                "patched_count": int(patched_count),
                "baseline_count": int(baseline_count),
                "lift": int(lift),
                "lift_ratio": float(patched_count / max(baseline_count, 1)),
            }

        # Aggregate
        total_diagnostic_lift = sum(v["lift"] for v in diagnostic_lift.values())
        total_control_lift = sum(v["lift"] for v in control_lift.values())

        results["diagnostic_token_lift"] = {
            "per_token": diagnostic_lift,
            "total_diagnostic_lift": total_diagnostic_lift,
            "total_control_lift": total_control_lift,
            "diagnostic_gt_control": total_diagnostic_lift > total_control_lift,
        }
        results["control_token_lift"] = {
            "per_token": control_lift,
        }

    # 4. Word count and behavioral metrics
    if "patched_word_count" in df.columns and "baseline_word_count" in df.columns:
        wc_delta = df["patched_word_count"] - df["baseline_word_count"]
        results["word_count_analysis"] = {
            "mean_delta": float(wc_delta.mean()),
            "std_delta": float(wc_delta.std()),
            "n_shorter": int((wc_delta < 0).sum()),
            "n_longer": int((wc_delta > 0).sum()),
        }

    if "patched_l4_count" in df.columns and "baseline_l4_count" in df.columns:
        l4_delta = df["patched_l4_count"] - df["baseline_l4_count"]
        results["l4_marker_analysis"] = {
            "mean_delta": float(l4_delta.mean()),
            "n_increased": int((l4_delta > 0).sum()),
            "n_decreased": int((l4_delta < 0).sum()),
            "n_unchanged": int((l4_delta == 0).sum()),
        }

    if "patched_unique_ratio" in df.columns and "baseline_unique_ratio" in df.columns:
        ur_delta = df["patched_unique_ratio"] - df["baseline_unique_ratio"]
        results["unique_ratio_analysis"] = {
            "mean_delta": float(ur_delta.mean()),
            "patched_mean": float(df["patched_unique_ratio"].mean()),
            "baseline_mean": float(df["baseline_unique_ratio"].mean()),
        }

    return results


@click.command()
@click.option("--csv", "csv_path", required=True, help="Path to per_sample.csv")
@click.option("--output", help="Path to save JSON results")
def main(csv_path: str, output: Optional[str]) -> None:
    """Analyze diagnostic token logit lift from bridge experiment data."""
    csv_p = Path(csv_path)
    if not csv_p.exists():
        click.echo(f"File not found: {csv_p}")
        return

    click.echo(f"Analyzing: {csv_p}")
    results = analyze_per_sample(csv_p)

    click.echo(f"\n{'='*60}")
    click.echo("DIAGNOSTIC TOKEN LOGIT LIFT ANALYSIS")
    click.echo(f"{'='*60}")
    click.echo(f"Samples: {results['n_samples']}")

    # R_V
    if "rv_analysis" in results:
        rv = results["rv_analysis"]
        click.echo(f"\nR_V Delta: mean={rv['mean_delta']:.4f}, "
                    f"{rv['pct_negative']:.0f}% negative (contraction)")

    # Logit diff
    if "logit_diff_analysis" in results:
        ld = results["logit_diff_analysis"]
        click.echo(f"Logit Diff Delta: mean={ld['mean_delta']:.4f}, "
                    f"{ld['pct_positive']:.0f}% positive (shifted)")

    # Diagnostic tokens
    if "diagnostic_token_lift" in results:
        dt = results["diagnostic_token_lift"]
        click.echo(f"\nDiagnostic Token Lift (patched - baseline):")
        click.echo(f"  Total diagnostic: {dt['total_diagnostic_lift']:+d}")
        click.echo(f"  Total control:    {dt['total_control_lift']:+d}")
        click.echo(f"  Diagnostic > Control: {dt['diagnostic_gt_control']}")

        click.echo(f"\n  Per-token breakdown:")
        for token, vals in sorted(
            dt["per_token"].items(), key=lambda x: x[1]["lift"], reverse=True
        ):
            click.echo(
                f"    {token:15s}: {vals['baseline_count']:3d} → {vals['patched_count']:3d} "
                f"(lift={vals['lift']:+d}, ratio={vals['lift_ratio']:.2f}x)"
            )

    # Word count
    if "word_count_analysis" in results:
        wc = results["word_count_analysis"]
        click.echo(f"\nWord Count Delta: mean={wc['mean_delta']:.1f}, "
                    f"{wc['n_shorter']} shorter / {wc['n_longer']} longer")

    # Save
    out_path = Path(output) if output else csv_p.with_suffix(".logit_lift.json")
    out_path.write_text(json.dumps(results, indent=2, default=str))
    click.echo(f"\nFull results saved to: {out_path}")


if __name__ == "__main__":
    main()
