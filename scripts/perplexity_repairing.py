#!/usr/bin/env python3
"""
POST-HOC PERPLEXITY RE-PAIRING (Method A)

For each recursive prompt, find the baseline prompt with the closest perplexity,
creating perplexity-matched pairs. Then re-test R_V on matched pairs only.

If the effect survives → R_V is NOT an artifact of perplexity differences.

This is a P0 requirement from the COLM gap analysis.

Usage:
    python3 scripts/perplexity_repairing.py
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats


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


def main():
    print("=" * 70)
    print("POST-HOC PERPLEXITY RE-PAIRING (Method A)")
    print("=" * 70)

    # Load circularity controls data
    circ_files = sorted((RESULTS / "circularity_controls").glob("circularity_perplexity_v2_*.json"))
    if not circ_files:
        print("ERROR: No circularity_perplexity_v2 data found")
        return

    source_file = circ_files[-1]
    data = load_json(source_file)
    if not data:
        print("ERROR: Empty data file")
        return

    groups = data["groups"]

    # Extract recursive and baseline with (rv, ppl) pairs
    rec_prompts = []
    for d in groups["recursive_reference"]["details"]:
        if d.get("rv") is not None and d.get("ppl") is not None:
            rec_prompts.append({"id": d["id"], "rv": d["rv"], "ppl": d["ppl"]})

    bas_prompts = []
    for d in groups["baseline_reference"]["details"]:
        if d.get("rv") is not None and d.get("ppl") is not None:
            bas_prompts.append({"id": d["id"], "rv": d["rv"], "ppl": d["ppl"]})

    print(f"\nRecursive prompts: {len(rec_prompts)}")
    print(f"Baseline prompts:  {len(bas_prompts)}")
    print(f"Recursive PPL: {np.mean([p['ppl'] for p in rec_prompts]):.1f} ± {np.std([p['ppl'] for p in rec_prompts]):.1f}")
    print(f"Baseline PPL:  {np.mean([p['ppl'] for p in bas_prompts]):.1f} ± {np.std([p['ppl'] for p in bas_prompts]):.1f}")

    # ── Method A: Nearest-neighbor perplexity matching ──
    print(f"\n{'=' * 50}")
    print("Method A: Nearest-neighbor perplexity matching")
    print(f"{'=' * 50}")

    matched_pairs = []
    used_bas = set()

    for rp in rec_prompts:
        best_idx = None
        best_dist = float("inf")
        for i, bp in enumerate(bas_prompts):
            if i in used_bas:
                continue
            dist = abs(rp["ppl"] - bp["ppl"])
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        if best_idx is not None:
            used_bas.add(best_idx)
            matched_pairs.append({
                "rec_id": rp["id"],
                "bas_id": bas_prompts[best_idx]["id"],
                "rec_rv": rp["rv"],
                "bas_rv": bas_prompts[best_idx]["rv"],
                "rec_ppl": rp["ppl"],
                "bas_ppl": bas_prompts[best_idx]["ppl"],
                "ppl_diff": abs(rp["ppl"] - bas_prompts[best_idx]["ppl"]),
            })

    print(f"Matched pairs: {len(matched_pairs)}")
    ppl_diffs = [p["ppl_diff"] for p in matched_pairs]
    print(f"PPL difference: {np.mean(ppl_diffs):.1f} ± {np.std(ppl_diffs):.1f} (max={max(ppl_diffs):.1f})")

    rec_matched = [p["rec_rv"] for p in matched_pairs]
    bas_matched = [p["bas_rv"] for p in matched_pairs]

    print(f"\nMatched R_V recursive: {np.mean(rec_matched):.4f} ± {np.std(rec_matched):.4f}")
    print(f"Matched R_V baseline:  {np.mean(bas_matched):.4f} ± {np.std(bas_matched):.4f}")

    # Paired t-test (since these are matched pairs)
    t_stat, p_paired = stats.ttest_rel(rec_matched, bas_matched)
    # Also unpaired for comparison
    u_stat, p_unpaired = stats.mannwhitneyu(rec_matched, bas_matched, alternative="two-sided")

    # Cohen's d (paired)
    diff = np.array(rec_matched) - np.array(bas_matched)
    d_paired = np.mean(diff) / np.std(diff, ddof=1)

    # Cohen's d (unpaired, standard)
    pooled_std = np.sqrt((np.var(rec_matched, ddof=1) + np.var(bas_matched, ddof=1)) / 2)
    d_unpaired = (np.mean(rec_matched) - np.mean(bas_matched)) / pooled_std

    print(f"\nPaired t-test:   t={t_stat:.3f}, p={p_paired:.2e}")
    print(f"Cohen's d (paired):   {d_paired:.3f}")
    print(f"Cohen's d (unpaired): {d_unpaired:.3f}")
    print(f"Mann-Whitney U:  p={p_unpaired:.2e}")

    # ── Method A': Strict matching (PPL diff < 10) ──
    print(f"\n{'=' * 50}")
    print("Method A': Strict matching (PPL diff < 10)")
    print(f"{'=' * 50}")

    strict_pairs = [p for p in matched_pairs if p["ppl_diff"] < 10]
    print(f"Pairs with PPL diff < 10: {len(strict_pairs)}")

    if len(strict_pairs) >= 5:
        rec_strict = [p["rec_rv"] for p in strict_pairs]
        bas_strict = [p["bas_rv"] for p in strict_pairs]
        t_strict, p_strict = stats.ttest_rel(rec_strict, bas_strict)
        diff_strict = np.array(rec_strict) - np.array(bas_strict)
        d_strict = np.mean(diff_strict) / np.std(diff_strict, ddof=1) if np.std(diff_strict, ddof=1) > 0 else 0

        print(f"R_V recursive: {np.mean(rec_strict):.4f} ± {np.std(rec_strict):.4f}")
        print(f"R_V baseline:  {np.mean(bas_strict):.4f} ± {np.std(bas_strict):.4f}")
        print(f"Paired t-test: t={t_strict:.3f}, p={p_strict:.2e}")
        print(f"Cohen's d (paired): {d_strict:.3f}")
    else:
        t_strict, p_strict, d_strict = None, None, None
        print("Insufficient strict pairs for analysis")

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print("CONCLUSION")
    print(f"{'=' * 70}")

    if p_paired < 0.05:
        print(f"✓ R_V effect SURVIVES perplexity re-pairing (p={p_paired:.2e}, d={d_paired:.3f})")
        print(f"  Perplexity confound ruled out by Method A.")
    else:
        print(f"⚠ R_V effect does NOT survive perplexity re-pairing (p={p_paired:.2e})")
        print(f"  Possible perplexity confound.")

    # Save results
    out_dir = RESULTS / "perplexity_repairing"
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result = {
        "timestamp": timestamp,
        "model": data.get("model", "mistralai/Mistral-7B-v0.1"),
        "source_artifact": str(source_file),
        "method": "nearest_neighbor_perplexity_matching",
        "n_recursive": len(rec_prompts),
        "n_baseline": len(bas_prompts),
        "n_matched_pairs": len(matched_pairs),
        "mean_ppl_diff": float(np.mean(ppl_diffs)),
        "max_ppl_diff": float(max(ppl_diffs)),
        "matched_results": {
            "rec_rv_mean": float(np.mean(rec_matched)),
            "rec_rv_std": float(np.std(rec_matched)),
            "bas_rv_mean": float(np.mean(bas_matched)),
            "bas_rv_std": float(np.std(bas_matched)),
            "t_stat": float(t_stat),
            "p_paired": float(p_paired),
            "d_paired": float(d_paired),
            "d_unpaired": float(d_unpaired),
            "p_mannwhitney": float(p_unpaired),
        },
        "strict_results": {
            "n_pairs": len(strict_pairs),
            "t_stat": float(t_strict) if t_strict is not None else None,
            "p_paired": float(p_strict) if p_strict is not None else None,
            "d_paired": float(d_strict) if d_strict is not None else None,
        },
        "matched_pairs": matched_pairs,
    }

    path = out_dir / f"repairing_results_{timestamp}.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
