#!/usr/bin/env python3
"""
CLUSTER-ROBUST STANDARD ERRORS

Prompts within templates are not independent — they share structural patterns
that can inflate effective sample sizes. This script:

1. Identifies template clusters from prompt IDs
2. Computes intra-cluster correlation (ICC)
3. Computes design effect (DEFF) and effective sample size
4. Reports cluster-robust CIs for the main R_V effects

P0 for COLM submission.

Usage:
    python3 scripts/cluster_robust_se.py
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


def intra_cluster_correlation(clusters):
    """Compute ICC(1) — intra-class correlation coefficient.

    ICC(1) estimates the proportion of total variance that is between-cluster.
    High ICC means prompts within a cluster are similar → effective n is lower.
    """
    all_vals = []
    cluster_means = []
    cluster_sizes = []

    for vals in clusters.values():
        if len(vals) < 2:
            continue
        all_vals.extend(vals)
        cluster_means.append(np.mean(vals))
        cluster_sizes.append(len(vals))

    if len(cluster_means) < 2:
        return 0.0, 0, 0

    grand_mean = np.mean(all_vals)
    N = len(all_vals)
    k = len(cluster_means)
    n_avg = N / k

    # Between-cluster variance (MSB)
    MSB = sum(n * (m - grand_mean) ** 2 for n, m in zip(cluster_sizes, cluster_means)) / (k - 1)

    # Within-cluster variance (MSW)
    SSW = 0
    for vals, m in zip(clusters.values(), cluster_means):
        if len(vals) < 2:
            continue
        SSW += sum((v - m) ** 2 for v in vals)
    MSW = SSW / (N - k) if N > k else 1e-10

    # ICC(1) = (MSB - MSW) / (MSB + (n_avg - 1) * MSW)
    icc = (MSB - MSW) / (MSB + (n_avg - 1) * MSW) if (MSB + (n_avg - 1) * MSW) > 0 else 0.0
    icc = max(0, min(1, icc))  # Clamp to [0, 1]

    return icc, k, n_avg


def design_effect(icc, avg_cluster_size):
    """DEFF = 1 + (avg_cluster_size - 1) * ICC."""
    return 1 + (avg_cluster_size - 1) * icc


def cluster_robust_ci(d, n1, n2, deff, alpha=0.05):
    """Compute cluster-robust CI for Cohen's d."""
    n1_eff = n1 / deff
    n2_eff = n2 / deff

    se = np.sqrt(2 / max(n1_eff, 1) + d ** 2 / (2 * max(n1_eff + n2_eff, 2)))
    z = stats.norm.ppf(1 - alpha / 2)
    return d - z * se, d + z * se, se


def extract_clusters(prompt_ids, rv_values):
    """Group prompts by template prefix."""
    clusters = {}
    for pid, rv in zip(prompt_ids, rv_values):
        # Template = everything before the last _NN number
        parts = pid.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            template = parts[0]
        else:
            template = pid
        clusters.setdefault(template, []).append(rv)
    return clusters


def main():
    print("=" * 70)
    print("CLUSTER-ROBUST STANDARD ERRORS")
    print("=" * 70)

    results = []

    # ── 1. Circularity Controls v2 (has prompt IDs) ──
    print("\n--- Circularity Controls v2 ---")
    circ_files = sorted((RESULTS / "circularity_controls").glob("circularity_perplexity_v2_*.json"))
    if circ_files:
        data = load_json(circ_files[-1])
        if data and "groups" in data:
            # Recursive group
            rec_group = data["groups"]["recursive_reference"]
            rec_ids = [d["id"] for d in rec_group["details"]]
            rec_rvs = [d["rv"] for d in rec_group["details"] if d.get("rv") is not None]
            rec_clusters = extract_clusters(rec_ids, rec_rvs)

            print(f"  Recursive: n={len(rec_rvs)}, clusters={len(rec_clusters)}")
            for tpl, vals in rec_clusters.items():
                print(f"    {tpl}: n={len(vals)}, mean={np.mean(vals):.4f}, std={np.std(vals):.4f}")

            icc_rec, k_rec, n_avg_rec = intra_cluster_correlation(rec_clusters)
            deff_rec = design_effect(icc_rec, n_avg_rec)
            n_eff_rec = len(rec_rvs) / deff_rec

            print(f"  ICC(1) = {icc_rec:.4f}")
            print(f"  DEFF = {deff_rec:.2f}")
            print(f"  Effective n: {len(rec_rvs)} → {n_eff_rec:.1f}")

            # Baseline group
            bas_group = data["groups"]["baseline_reference"]
            bas_ids = [d["id"] for d in bas_group["details"]]
            bas_rvs = [d["rv"] for d in bas_group["details"] if d.get("rv") is not None]
            bas_clusters = extract_clusters(bas_ids, bas_rvs)

            print(f"\n  Baseline: n={len(bas_rvs)}, clusters={len(bas_clusters)}")
            for tpl, vals in bas_clusters.items():
                print(f"    {tpl}: n={len(vals)}, mean={np.mean(vals):.4f}, std={np.std(vals):.4f}")

            icc_bas, k_bas, n_avg_bas = intra_cluster_correlation(bas_clusters)
            deff_bas = design_effect(icc_bas, n_avg_bas)
            n_eff_bas = len(bas_rvs) / deff_bas

            print(f"  ICC(1) = {icc_bas:.4f}")
            print(f"  DEFF = {deff_bas:.2f}")
            print(f"  Effective n: {len(bas_rvs)} → {n_eff_bas:.1f}")

            # Combined DEFF (average)
            deff_combined = (deff_rec + deff_bas) / 2

            # Original effect
            pooled_std = np.sqrt((np.var(rec_rvs, ddof=1) + np.var(bas_rvs, ddof=1)) / 2)
            d_orig = (np.mean(rec_rvs) - np.mean(bas_rvs)) / pooled_std

            # Original CI
            se_orig = np.sqrt(2 / len(rec_rvs) + d_orig ** 2 / (2 * (len(rec_rvs) + len(bas_rvs))))
            ci_orig = (d_orig - 1.96 * se_orig, d_orig + 1.96 * se_orig)

            # Cluster-robust CI
            ci_lo, ci_hi, se_robust = cluster_robust_ci(d_orig, len(rec_rvs), len(bas_rvs), deff_combined)

            print(f"\n  Effect size: d = {d_orig:.3f}")
            print(f"  Original CI:        [{ci_orig[0]:.3f}, {ci_orig[1]:.3f}] (SE={se_orig:.3f})")
            print(f"  Cluster-robust CI:  [{ci_lo:.3f}, {ci_hi:.3f}] (SE={se_robust:.3f})")
            print(f"  CI width inflation:  {(ci_hi - ci_lo) / (ci_orig[1] - ci_orig[0]):.2f}x")

            results.append({
                "comparison": "recursive_vs_baseline (circularity_v2)",
                "d": d_orig,
                "n_rec": len(rec_rvs),
                "n_bas": len(bas_rvs),
                "n_clusters_rec": k_rec,
                "n_clusters_bas": k_bas,
                "icc_rec": icc_rec,
                "icc_bas": icc_bas,
                "deff_rec": deff_rec,
                "deff_bas": deff_bas,
                "deff_combined": deff_combined,
                "n_eff_rec": n_eff_rec,
                "n_eff_bas": n_eff_bas,
                "ci_original": list(ci_orig),
                "ci_robust": [ci_lo, ci_hi],
                "se_original": se_orig,
                "se_robust": se_robust,
                "significant_robust": (ci_lo > 0 or ci_hi < 0),
            })

    # ── 2. Scaling gap models (have individual RV values) ──
    print("\n--- Scaling Gap Models ---")
    for model_file in sorted((RESULTS / "scaling_gap").glob("*_result.json")):
        data = load_json(model_file)
        if not data or "recursive_rv_values" not in data:
            continue

        model = data.get("model", model_file.stem)
        rec_rvs = [v for v in data["recursive_rv_values"] if v is not None]
        bas_rvs = [v for v in data["baseline_rv_values"] if v is not None]
        d = data.get("cohens_d", 0)

        if not rec_rvs or not bas_rvs:
            continue

        # For scaling gap, prompts come from the same bank with no sub-templates
        # → treat as single cluster (conservative: ICC=0, DEFF=1)
        # But for thoroughness, we can check if there's within-prompt correlation
        # by computing a permutation-based ICC estimate

        # Without template IDs, we assume DEFF=1 (worst case: all independent)
        # The conservative approach: report both naive and worst-case DEFF=2
        se_naive = np.sqrt(2 / len(rec_rvs) + d ** 2 / (2 * (len(rec_rvs) + len(bas_rvs))))
        ci_naive = (d - 1.96 * se_naive, d + 1.96 * se_naive)

        # Conservative DEFF=2 (assumes moderate template dependence)
        ci_lo, ci_hi, se_cons = cluster_robust_ci(d, len(rec_rvs), len(bas_rvs), deff=2.0)

        significant_naive = (ci_naive[0] > 0 or ci_naive[1] < 0)
        significant_cons = (ci_lo > 0 or ci_hi < 0)

        print(f"  {model}: d={d:.3f}, naive CI=[{ci_naive[0]:.3f}, {ci_naive[1]:.3f}], "
              f"conservative CI=[{ci_lo:.3f}, {ci_hi:.3f}] "
              f"{'✓' if significant_cons else '✗ (loses sig)'}")

        results.append({
            "comparison": f"{model} (scaling_gap)",
            "d": d,
            "n_rec": len(rec_rvs),
            "n_bas": len(bas_rvs),
            "deff_assumed": 2.0,
            "ci_naive": list(ci_naive),
            "ci_conservative": [ci_lo, ci_hi],
            "se_naive": se_naive,
            "se_conservative": se_cons,
            "significant_naive": significant_naive,
            "significant_conservative": significant_cons,
        })

    # ── 3. Statistical hardening effects ──
    print("\n--- Statistical Hardening (cluster-robust) ---")
    hard_files = sorted((RESULTS / "statistical_hardening").glob("hardening_summary_*.json"))
    if hard_files:
        data = load_json(hard_files[-1])
        if data:
            for e in data["effects"]:
                d = e["d_observed"]
                n1, n2 = e["n1"], e["n2"]

                # Naive CI
                se = np.sqrt(2 / n1 + d ** 2 / (2 * (n1 + n2)))
                ci_naive = (d - 1.96 * se, d + 1.96 * se)

                # Conservative DEFF=2
                ci_lo, ci_hi, se_cons = cluster_robust_ci(d, n1, n2, deff=2.0)

                significant = (ci_lo > 0 or ci_hi < 0)

                print(f"  {e['name'][:45]}: d={d:.2f}, "
                      f"conservative CI=[{ci_lo:.2f}, {ci_hi:.2f}] "
                      f"{'✓' if significant else '✗'}")

                results.append({
                    "comparison": e["name"],
                    "d": d,
                    "n1": n1,
                    "n2": n2,
                    "deff_assumed": 2.0,
                    "ci_naive": list(ci_naive),
                    "ci_conservative": [ci_lo, ci_hi],
                    "significant_conservative": significant,
                })

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    n_tested = len(results)
    n_sig_naive = sum(1 for r in results if r.get("significant_naive", r.get("significant_conservative", False)))
    n_sig_cons = sum(1 for r in results if r.get("significant_conservative", r.get("significant_robust", False)))

    print(f"Total comparisons: {n_tested}")
    print(f"Significant (naive):        {n_sig_naive}/{n_tested}")
    print(f"Significant (conservative): {n_sig_cons}/{n_tested}")

    lost = [r for r in results
            if r.get("significant_naive", True) and not r.get("significant_conservative", r.get("significant_robust", True))]
    if lost:
        print(f"\n⚠ Effects that LOSE significance with cluster-robust SEs (DEFF=2):")
        for r in lost:
            print(f"  - {r['comparison']} (d={r['d']:.3f})")
    else:
        print(f"\n✓ All effects remain significant under conservative cluster-robust SEs")

    # Circularity-specific: report the measured ICC and DEFF
    circ_results = [r for r in results if "circularity" in r.get("comparison", "")]
    if circ_results:
        r = circ_results[0]
        print(f"\nCIRCULARITY CONTROLS — measured cluster structure:")
        print(f"  Recursive ICC(1) = {r.get('icc_rec', 'N/A'):.4f}")
        print(f"  Baseline ICC(1)  = {r.get('icc_bas', 'N/A'):.4f}")
        print(f"  Combined DEFF    = {r.get('deff_combined', 'N/A'):.2f}")
        print(f"  Effect: d = {r['d']:.3f}")
        print(f"  Robust CI: [{r.get('ci_robust', [0,0])[0]:.3f}, {r.get('ci_robust', [0,0])[1]:.3f}]")
        print(f"  Still significant: {r.get('significant_robust', 'N/A')}")

    # Save
    out_dir = RESULTS / "cluster_robust_se"
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {
        "timestamp": timestamp,
        "n_comparisons": n_tested,
        "n_significant_conservative": n_sig_cons,
        "method": "cluster-robust SE with DEFF correction",
        "note": "For circularity controls: ICC computed from template clusters. "
                "For other experiments: conservative DEFF=2 assumed.",
        "results": results,
    }
    path = out_dir / f"cluster_robust_results_{timestamp}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
