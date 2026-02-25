#!/usr/bin/env python3
"""Build a compact final-correlation report from seed + semantic artifacts."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    evidence = repo_root / "industry_grade" / "2026-02-20" / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)

    seed_path = evidence / "seed_bridge_analysis.json"
    semantic_path = evidence / "semantic_behavior_analysis.json"

    seed = json.loads(seed_path.read_text(encoding="utf-8"))
    sem = json.loads(semantic_path.read_text(encoding="utf-8"))

    pooled = seed.get("pooled_paired_ttest", {})
    per_seed = seed.get("per_seed", {})
    standout = seed.get("standout_signal_gate", {})

    seed_corr = sem.get("seed_bridge", {})
    c2_corr = sem.get("c2", {})

    out = {
        "seed_bridge": {
            "n_runs_found": seed.get("n_runs_found"),
            "n_seeds_complete": seed.get("n_seeds_complete"),
            "seeds_complete": seed.get("seeds_complete"),
            "standout_signal_gate": standout,
            "pooled_paired_ttest": {
                "head_specific_vs_random_head_control": pooled.get("head_specific_vs_random_head_control"),
                "head_specific_vs_baseline_donor_control": pooled.get("head_specific_vs_baseline_donor_control"),
                "random_head_control_vs_baseline_donor_control": pooled.get("random_head_control_vs_baseline_donor_control"),
            },
            "per_seed": {
                k: {
                    "head_specific_vs_random_head_control": v.get("head_specific_vs_random_head_control"),
                    "head_specific_vs_baseline_donor_control": v.get("head_specific_vs_baseline_donor_control"),
                }
                for k, v in per_seed.items()
            },
        },
        "semantic": {
            "seed_bridge_rows": seed_corr.get("n_rows"),
            "seed_bridge_runs": seed_corr.get("runs"),
            "seed_bridge_seeds": seed_corr.get("seeds"),
            "seed_bridge_rates": seed_corr.get("semantic_recursive_rate_by_condition"),
            "seed_bridge_spearman": seed_corr.get("spearman_overall"),
            "seed_bridge_semantic_score_contrasts": seed_corr.get("semantic_score_contrasts"),
            "c2_rows": c2_corr.get("n_rows"),
            "c2_sources": c2_corr.get("sources"),
            "c2_spearman": c2_corr.get("spearman_overall"),
        },
    }

    json_out = evidence / "final_correlations.json"
    md_out = evidence / "final_correlations.md"
    json_out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    lines = ["# Final Correlation Summary\n"]
    lines.append("## Seed Bridge\n")
    lines.append(f"- runs found: `{out['seed_bridge']['n_runs_found']}`\n")
    lines.append(f"- complete seeds: `{out['seed_bridge']['n_seeds_complete']}` -> `{out['seed_bridge']['seeds_complete']}`\n")
    lines.append(
        f"- standout gate seed passes: `{out['seed_bridge']['standout_signal_gate'].get('seed_passes', [])}`\n"
    )
    for key, vals in out["seed_bridge"]["pooled_paired_ttest"].items():
        if not vals:
            continue
        lines.append(
            f"- `{key}`: mean_diff={vals.get('mean_diff')}, p={vals.get('p_value')}, "
            f"d={vals.get('cohens_d')}, n_pairs={vals.get('n_pairs')}\n"
        )

    lines.append("\n## Semantic (Seed Bridge)\n")
    lines.append(
        f"- rows/runs/seeds: `{out['semantic']['seed_bridge_rows']}` / `{out['semantic']['seed_bridge_runs']}` / `{out['semantic']['seed_bridge_seeds']}`\n"
    )
    lines.append("- semantic_recursive_rate by condition:\n")
    rates = out["semantic"]["seed_bridge_rates"] or {}
    for cond, vals in rates.items():
        lines.append(
            f"  - `{cond}`: rate={vals.get('semantic_recursive_rate')}, "
            f"mean_score={vals.get('semantic_score_mean')}, n={vals.get('n')}\n"
        )
    sp = out["semantic"]["seed_bridge_spearman"] or {}
    lines.append(
        f"- Spearman rv_delta vs semantic_score: `{sp.get('rv_delta_vs_semantic_score')}`\n"
    )
    lines.append(
        f"- Spearman rv_patch vs semantic_score: `{sp.get('rv_patch_vs_semantic_score')}`\n"
    )

    lines.append("\n## Semantic (C2)\n")
    lines.append(f"- rows/sources: `{out['semantic']['c2_rows']}` / `{out['semantic']['c2_sources']}`\n")
    c2_sp = (out["semantic"]["c2_spearman"] or {}).get("rv_mean_vs_semantic_score")
    lines.append(f"- Spearman rv_mean vs semantic_score: `{c2_sp}`\n")

    md_out.write_text("".join(lines), encoding="utf-8")

    print(json_out)
    print(md_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
