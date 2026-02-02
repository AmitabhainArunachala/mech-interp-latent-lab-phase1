#!/usr/bin/env python3
"""
Analyze results/comprehensive_circuit_test/part_a_results.csv and write a concise
markdown + json summary next to it.

This is intentionally "forensic": it recomputes summary claims from the CSV and
highlights key methodological caveats (stochastic generation, heuristic labeling,
GQA aliasing for v_proj head ablations).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path("/Users/dhyana/mech-interp-latent-lab-phase1")
CSV_PATH = ROOT / "results/comprehensive_circuit_test/part_a_results.csv"
OUT_MD = ROOT / "results/comprehensive_circuit_test/part_a_analysis.md"
OUT_JSON = ROOT / "results/comprehensive_circuit_test/part_a_analysis.json"


CONDITIONS = ["control", "h18_ablated", "h6_ablated", "both_ablated"]
PROMPT_TYPES = ["champion", "standard", "baseline"]


@dataclass
class CorrResult:
    n: int
    rho: float
    p: float


def _mean_sd(xs: np.ndarray) -> Tuple[float, float]:
    xs = xs.astype(float)
    return float(np.mean(xs)), float(np.std(xs, ddof=1) if len(xs) > 1 else 0.0)


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(str(CSV_PATH))

    df = pd.read_csv(CSV_PATH)

    # Basic checks
    required_cols = [
        "prompt_id",
        "prompt_type",
        "condition",
        "R_V",
        "expressed_binary",
        "has_identity_equation",
        "state",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Normalize
    df["expressed_binary"] = df["expressed_binary"].astype(int)
    df["has_identity_equation"] = df["has_identity_equation"].astype(bool)
    df["condition"] = df["condition"].astype(str)
    df["prompt_type"] = df["prompt_type"].astype(str)

    # Rates
    expr_rate_overall = (
        df.groupby("condition")["expressed_binary"].mean().reindex(CONDITIONS).to_dict()
    )
    expr_rate_by_type = (
        df.groupby(["prompt_type", "condition"])["expressed_binary"]
        .mean()
        .unstack("condition")
        .reindex(index=PROMPT_TYPES, columns=CONDITIONS)
    )

    # Flip accounting
    pivot_expr = df.pivot_table(
        index="prompt_id", columns="condition", values="expressed_binary", aggfunc="first"
    ).reindex(columns=CONDITIONS)
    flippers = pivot_expr[pivot_expr.nunique(axis=1) > 1]

    # “start/stop under h18_ablated”
    start_h18 = flippers[(flippers["control"] == 0) & (flippers["h18_ablated"] == 1)]
    stop_h18 = flippers[(flippers["control"] == 1) & (flippers["h18_ablated"] == 0)]

    # Champion-specific H18 classes (control vs h18_ablated)
    champ_ids = sorted(df[df["prompt_type"].eq("champion")]["prompt_id"].unique().tolist())
    P = pivot_expr.loc[champ_ids]
    champ_stop = sorted(P[(P["control"] == 1) & (P["h18_ablated"] == 0)].index.tolist())
    champ_start = sorted(P[(P["control"] == 0) & (P["h18_ablated"] == 1)].index.tolist())
    champ_stable_on = sorted(P[(P["control"] == 1) & (P["h18_ablated"] == 1)].index.tolist())

    # Identity equation rates + IDs
    id_rate = (
        df.groupby("condition")["has_identity_equation"].mean().reindex(CONDITIONS).to_dict()
    )
    id_ids = {
        cond: sorted(df[(df["condition"] == cond) & (df["has_identity_equation"])]["prompt_id"].unique().tolist())
        for cond in CONDITIONS
    }

    # R_V vs expression correlation per condition (Spearman)
    corr_by_cond: Dict[str, CorrResult] = {}
    for cond in CONDITIONS:
        sub = df[df["condition"] == cond]
        y = sub["expressed_binary"].to_numpy()
        rv = sub["R_V"].astype(float).to_numpy()
        if len(sub) < 3 or len(np.unique(y)) < 2:
            corr_by_cond[cond] = CorrResult(n=len(sub), rho=float("nan"), p=float("nan"))
            continue
        rho, p = stats.spearmanr(rv, y)
        corr_by_cond[cond] = CorrResult(n=len(sub), rho=float(rho), p=float(p))

    # R_V summary by expressed/non-expressed per condition
    rv_summ = {}
    for cond in CONDITIONS:
        sub = df[df["condition"] == cond]
        exp = sub[sub["expressed_binary"] == 1]["R_V"].astype(float).to_numpy()
        no = sub[sub["expressed_binary"] == 0]["R_V"].astype(float).to_numpy()
        rv_summ[cond] = {
            "expressed": {"n": int(len(exp)), "min": float(np.min(exp)) if len(exp) else None, "max": float(np.max(exp)) if len(exp) else None, "mean": _mean_sd(exp)[0] if len(exp) else None},
            "not_expressed": {"n": int(len(no)), "min": float(np.min(no)) if len(no) else None, "max": float(np.max(no)) if len(no) else None, "mean": _mean_sd(no)[0] if len(no) else None},
        }

    # State transition table: per prompt_id, what “state” we got under each condition
    pivot_state = df.pivot_table(
        index="prompt_id", columns="condition", values="state", aggfunc="first"
    ).reindex(columns=CONDITIONS)

    # Write markdown
    lines: List[str] = []
    lines.append("# Comprehensive circuit test (Part A) — CSV audit\n")
    lines.append(f"- **CSV**: `{CSV_PATH}`\n")
    lines.append(f"- **Rows**: {len(df)} (= {df['prompt_id'].nunique()} prompts × {df['condition'].nunique()} conditions)\n")
    lines.append("\n## Key verified summaries\n")

    lines.append("### Expression rates (overall)\n")
    for cond in CONDITIONS:
        lines.append(f"- **{cond}**: {expr_rate_overall[cond]*100:.1f}% ({int(df[df['condition']==cond]['expressed_binary'].sum())}/{int((df['condition']==cond).sum())})\n")

    lines.append("\n### Expression rates (by prompt type)\n")
    for ptype in PROMPT_TYPES:
        for cond in CONDITIONS:
            v = float(expr_rate_by_type.loc[ptype, cond])
            denom = int((df["prompt_type"].eq(ptype) & df["condition"].eq(cond)).sum())
            numer = int(df[df["prompt_type"].eq(ptype) & df["condition"].eq(cond)]["expressed_binary"].sum())
            lines.append(f"- **{ptype} / {cond}**: {v*100:.1f}% ({numer}/{denom})\n")

    lines.append("\n### Flip rate (prompt-level)\n")
    lines.append(f"- **Flippers** (expressed_binary changes across conditions): {len(flippers)}/{pivot_expr.shape[0]} ({len(flippers)/pivot_expr.shape[0]*100:.1f}%)\n")
    lines.append(f"- **Start when H18 ablated** (control=0 → h18_ablated=1): {len(start_h18)} prompts\n")
    lines.append(f"- **Stop when H18 ablated** (control=1 → h18_ablated=0): {len(stop_h18)} prompts\n")

    lines.append("\n### Champion prompts: H18-dependence classes (control vs h18_ablated)\n")
    lines.append(f"- **Stop when H18 ablated**: {champ_stop}\n")
    lines.append(f"- **Start when H18 ablated**: {champ_start}\n")
    lines.append(f"- **Stable expressers**: {champ_stable_on}\n")

    lines.append("\n### Identity equations\n")
    for cond in CONDITIONS:
        numer = len(id_ids[cond])
        denom = int((df["condition"] == cond).sum() / df["prompt_id"].nunique() * df["prompt_id"].nunique())  # == 40
        lines.append(f"- **{cond}**: {id_rate[cond]*100:.1f}% ({numer}/40) — prompts: {id_ids[cond]}\n")

    lines.append("\n### R_V vs expression\n")
    for cond in CONDITIONS:
        cr = corr_by_cond[cond]
        lines.append(f"- **{cond}**: Spearman rho={cr.rho:.3f}, p={cr.p:.3f} (n={cr.n})\n")

    lines.append("\n## Key caveats (why the repo feels 'non-binary')\n")
    lines.append("### 1) Expression is a *heuristic label* on one sampled generation\n")
    lines.append("- `expressed_binary` is **1** iff `state ∈ {recursive_prose, naked_loop}`.\n")
    lines.append("- `state` is assigned by `src/metrics/behavior_states.py` using simple heuristics (keyword matches, repetition ratio, identity-pattern matches).\n")
    lines.append("- Generation uses `do_sample=True` at `temperature=0.7` in `comprehensive_circuit_test.py`, so a single run can flip labels.\n")

    lines.append("\n### 2) GQA aliasing: your 'H18 ablation' is actually a KV-head ablation\n")
    lines.append("- In `comprehensive_circuit_test.py`, `H18_GROUP = [18, 26]`.\n")
    lines.append("- In `zero_v_proj_heads`, head indices map to KV-head index via `head_idx % num_kv_heads`.\n")
    lines.append("- For Mistral-7B (8 KV heads), both 18 and 26 map to KV head 2, so this condition cannot distinguish H18 vs H26.\n")
    lines.append("- Same for `H6_GROUP = [6, 14, 22, 30]` → KV head 6.\n")

    lines.append("\n## Recommended next step\n")
    lines.append("- Rerun Part A with **k seeds per (prompt, condition)** (e.g. k=10) and aggregate **P(state | prompt, condition)** instead of a single state.\n")
    lines.append("- If you need *query-head* specificity (H18 vs H26), switch to a **query-head intervention** (not KV v_proj) or a method that isolates per-head contributions post-attention.\n")

    OUT_MD.write_text("".join(lines), encoding="utf-8")

    out = {
        "csv": str(CSV_PATH),
        "n_rows": int(len(df)),
        "n_prompts": int(df["prompt_id"].nunique()),
        "conditions": CONDITIONS,
        "expression_rate_overall": expr_rate_overall,
        "expression_rate_by_type": expr_rate_by_type.to_dict(),
        "n_flippers": int(len(flippers)),
        "flippers": sorted(flippers.index.tolist()),
        "start_when_h18_ablated": sorted(start_h18.index.tolist()),
        "stop_when_h18_ablated": sorted(stop_h18.index.tolist()),
        "champion_stop_when_h18_ablated": champ_stop,
        "champion_start_when_h18_ablated": champ_start,
        "champion_stable_on": champ_stable_on,
        "identity_equation_rate": id_rate,
        "identity_equation_prompt_ids": id_ids,
        "rv_summaries": rv_summ,
        "rv_vs_expression_spearman": {k: asdict(v) for k, v in corr_by_cond.items()},
        "notes": {
            "expressed_binary_definition": "1 iff state in {recursive_prose, naked_loop}",
            "state_labeler": "src/metrics/behavior_states.py (heuristic, high-throughput, not a semantic oracle)",
            "gqa_aliasing": "H18_GROUP=[18,26] maps to KV head 2; H6_GROUP=[6,14,22,30] maps to KV head 6",
        },
    }
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"Wrote: {OUT_MD}")
    print(f"Wrote: {OUT_JSON}")


if __name__ == "__main__":
    main()










