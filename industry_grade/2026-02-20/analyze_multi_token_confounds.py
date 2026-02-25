#!/usr/bin/env python3
"""Confound-focused analysis for multi-token bridge runs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def pointbiserial_safe(binary: pd.Series, values: pd.Series):
    if binary.nunique(dropna=True) < 2:
        return (np.nan, np.nan)
    return stats.pointbiserialr(binary.astype(int), values)


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    run_dir = repo / "results" / "remote_gpu_sync" / "2026-02-20" / "phase1_cross_architecture" / "20260220_071457_multi_token_bridge_mistral_7b_bridge_deconfound_fast"
    csv_path = run_dir / "rv_behavioral_correlation.csv"
    out_dir = repo / "industry_grade" / "2026-02-20" / "evidence"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)

    rows = []
    for temp, g in df.groupby("temperature"):
        rec = g[g["group_type"] == "recursive"]["rv"]
        base = g[g["group_type"] == "baseline"]["rv"]
        t_stat, p_group = stats.ttest_ind(rec, base, equal_var=False)

        r_all, p_all = stats.spearmanr(g["rv"], g["word_count"])

        non_trunc = g[~g["truncated"]]
        if len(non_trunc) > 3:
            r_non, p_non = stats.spearmanr(non_trunc["rv"], non_trunc["word_count"])
        else:
            r_non, p_non = (np.nan, np.nan)

        r_trunc, p_trunc = pointbiserial_safe(g["truncated"], g["word_count"])

        rows.append(
            {
                "temperature": float(temp),
                "n_total": int(len(g)),
                "pct_truncated": float(100.0 * g["truncated"].mean()),
                "n_non_truncated": int((~g["truncated"]).sum()),
                "rv_recursive_mean": float(rec.mean()),
                "rv_baseline_mean": float(base.mean()),
                "rv_group_t": float(t_stat),
                "rv_group_p": float(p_group),
                "rv_word_spearman_all_r": float(r_all),
                "rv_word_spearman_all_p": float(p_all),
                "rv_word_spearman_nontrunc_r": float(r_non) if not np.isnan(r_non) else None,
                "rv_word_spearman_nontrunc_p": float(p_non) if not np.isnan(p_non) else None,
                "trunc_word_pointbiserial_r": float(r_trunc) if not np.isnan(r_trunc) else None,
                "trunc_word_pointbiserial_p": float(p_trunc) if not np.isnan(p_trunc) else None,
            }
        )

    out = {
        "run_dir": str(run_dir),
        "results": rows,
    }

    json_path = out_dir / "multi_token_confound_analysis.json"
    md_path = out_dir / "multi_token_confound_analysis.md"

    json_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    lines = ["# Multi-token Confound Analysis\n"]
    for r in rows:
        lines.append(f"## Temperature {r['temperature']}\n")
        lines.append(f"- n_total: `{r['n_total']}`\n")
        lines.append(f"- pct_truncated: `{r['pct_truncated']:.1f}%`\n")
        lines.append(f"- n_non_truncated: `{r['n_non_truncated']}`\n")
        lines.append(f"- rv recursive mean: `{r['rv_recursive_mean']:.4f}`\n")
        lines.append(f"- rv baseline mean: `{r['rv_baseline_mean']:.4f}`\n")
        lines.append(f"- rv group p-value: `{r['rv_group_p']:.3e}`\n")
        lines.append(f"- rv-word spearman (all): `r={r['rv_word_spearman_all_r']:.4f}, p={r['rv_word_spearman_all_p']:.3e}`\n")
        if r['rv_word_spearman_nontrunc_r'] is not None:
            lines.append(
                f"- rv-word spearman (non-truncated): `r={r['rv_word_spearman_nontrunc_r']:.4f}, p={r['rv_word_spearman_nontrunc_p']:.3e}`\n"
            )
        else:
            lines.append("- rv-word spearman (non-truncated): `insufficient samples`\n")
        if r['trunc_word_pointbiserial_r'] is not None:
            lines.append(
                f"- truncation-word point-biserial: `r={r['trunc_word_pointbiserial_r']:.4f}, p={r['trunc_word_pointbiserial_p']:.3e}`\n"
            )
        lines.append("\n")

    md_path.write_text("".join(lines), encoding="utf-8")

    print(json_path)
    print(md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
