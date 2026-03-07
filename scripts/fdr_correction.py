#!/usr/bin/env python3
"""
FDR CORRECTION — Benjamini-Hochberg across all R_V statistical tests.

Collects every p-value from all experiments, applies BH correction,
and outputs a corrected table with q-values.

This is a P0 requirement for COLM submission.

Usage:
    python3 scripts/fdr_correction.py
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime


RESULTS = Path(__file__).parent.parent / "results"


def load_json(path):
    with open(path) as f:
        content = f.read().strip()
        if not content:
            return None
        # Handle Infinity / NaN in JSON
        content = content.replace(": Infinity", ": 1e308")
        content = content.replace(": -Infinity", ": -1e308")
        content = content.replace(": NaN", ": null")
        return json.loads(content)


def collect_pvalues():
    """Collect all p-values from all experiments into a structured list."""
    tests = []

    # ── 1. Statistical hardening (9 effects) ──
    path = sorted((RESULTS / "statistical_hardening").glob("hardening_summary_*.json"))
    if path:
        data = load_json(path[-1])
        if data:
            for e in data["effects"]:
                # BF-derived p-values: approximate from BF
                # For hardening, we use the effect sizes to compute p
                n1, n2 = e["n1"], e["n2"]
                d = e["d_observed"]
                # Two-sided t-test p-value from d and n
                se = np.sqrt(1/n1 + 1/n2)
                t_stat = d / se
                from scipy import stats as st
                p = 2 * st.t.sf(abs(t_stat), df=n1+n2-2)
                tests.append({
                    "source": "statistical_hardening",
                    "test": e["name"],
                    "d": d,
                    "n1": n1,
                    "n2": n2,
                    "p_original": p,
                    "bf_interpretation": e["bf_interpretation"],
                })

    # ── 2. Power-up (E1.1) — cross-architecture tests ──
    for f in sorted((RESULTS / "power_up").glob("*_result.json")):
        data = load_json(f)
        if data and "p_value" in data and data.get("p_value") is not None:
            tests.append({
                "source": "E1.1_power_up",
                "test": f"{data['model']} (n={data.get('n_recursive', '?')}/{data.get('n_baseline', '?')})",
                "d": data.get("cohens_d"),
                "n1": data.get("n_recursive"),
                "n2": data.get("n_baseline"),
                "p_original": data["p_value"],
            })

    # ── 3. Scaling gap (E1.3) — per-model tests ──
    metrics_path = RESULTS / "rv_masterplan/E1.3_scaling_gap/metrics.json"
    if metrics_path.exists():
        data = load_json(metrics_path)
        if data:
            for model, vals in data.get("models_completed", {}).items():
                if vals.get("p_value") is not None:
                    tests.append({
                        "source": "E1.3_scaling_gap",
                        "test": f"{model} (n={vals.get('n_recursive', '?')}/{vals.get('n_baseline', '?')})",
                        "d": vals.get("cohens_d"),
                        "n1": vals.get("n_recursive"),
                        "n2": vals.get("n_baseline"),
                        "p_original": vals["p_value"],
                    })

    # ── 4. Training checkpoints (E1.4) ──
    for f in sorted((RESULTS / "training_checkpoints").glob("*_result.json")):
        data = load_json(f)
        if data and "p_value" in data and data.get("p_value") is not None:
            tests.append({
                "source": "E1.4_training_checkpoints",
                "test": f"{data['model']} step={data.get('step', '?')} (n={data.get('n_recursive', '?')}/{data.get('n_baseline', '?')})",
                "d": data.get("cohens_d"),
                "n1": data.get("n_recursive"),
                "n2": data.get("n_baseline"),
                "p_original": data["p_value"],
            })

    # ── 5. Circularity controls (perplexity v2) ──
    circ_files = sorted((RESULTS / "circularity_controls").glob("circularity_perplexity_v2_*.json"))
    if circ_files:
        data = load_json(circ_files[-1])
        if data and "groups" in data:
            # Pairwise comparisons between groups
            from scipy import stats as st
            groups = {}
            for gname, gdata in data["groups"].items():
                rvs = [d["rv"] for d in gdata.get("details", []) if d.get("rv") is not None]
                if rvs:
                    groups[gname] = rvs

            if "recursive_reference" in groups:
                ref = groups["recursive_reference"]
                for gname, rvs in groups.items():
                    if gname == "recursive_reference":
                        continue
                    if len(ref) >= 2 and len(rvs) >= 2:
                        u, p = st.mannwhitneyu(ref, rvs, alternative="two-sided")
                        pooled_std = np.sqrt((np.var(ref, ddof=1) + np.var(rvs, ddof=1)) / 2)
                        d = (np.mean(ref) - np.mean(rvs)) / pooled_std if pooled_std > 0 else 0
                        tests.append({
                            "source": "circularity_controls_v2",
                            "test": f"recursive_reference vs {gname}",
                            "d": d,
                            "n1": len(ref),
                            "n2": len(rvs),
                            "p_original": p,
                        })

    # ── 6. Safety monitoring (E5) ──
    safety_files = sorted((RESULTS / "safety").glob("safety_analysis_*.json"))
    if safety_files:
        data = load_json(safety_files[-1])
        if data:
            e51 = data.get("e51_genuine_vs_deceptive", {})
            # Approximate p from d and assumed n
            # We can compute via the d values
            for label, d_key in [
                ("genuine vs baseline", "d_genuine_vs_baseline"),
                ("deceptive vs baseline", "d_deceptive_vs_baseline"),
                ("genuine vs deceptive", "d_genuine_vs_deceptive"),
            ]:
                d_val = e51.get(d_key)
                if d_val is not None:
                    # Approximate n=20 per group
                    se = np.sqrt(2/20 + d_val**2 / (2*20))
                    from scipy import stats as st
                    t = d_val / se if se > 0 else 0
                    p = 2 * st.t.sf(abs(t), df=38)
                    tests.append({
                        "source": "E5_safety",
                        "test": label,
                        "d": d_val,
                        "n1": 20,
                        "n2": 20,
                        "p_original": p,
                    })

    return tests


def benjamini_hochberg(p_values, alpha=0.05):
    """Apply Benjamini-Hochberg FDR correction.

    Returns q-values (adjusted p-values) and significance mask.
    """
    p = np.array(p_values, dtype=float)
    n = len(p)

    # Sort p-values
    sorted_idx = np.argsort(p)
    sorted_p = p[sorted_idx]

    # Compute BH critical values: (rank / n) * alpha
    ranks = np.arange(1, n + 1)
    bh_critical = (ranks / n) * alpha

    # Compute q-values (adjusted p-values)
    q_values = np.zeros(n)
    q_values[sorted_idx[-1]] = sorted_p[-1]
    for i in range(n - 2, -1, -1):
        q_values[sorted_idx[i]] = min(
            sorted_p[i] * n / (i + 1),
            q_values[sorted_idx[i + 1]]
        )

    # Significance
    significant = q_values < alpha

    return q_values, significant


def main():
    print("=" * 70)
    print("FDR CORRECTION — Benjamini-Hochberg")
    print("=" * 70)

    tests = collect_pvalues()
    print(f"\nCollected {len(tests)} statistical tests:\n")

    # Group by source
    sources = {}
    for t in tests:
        sources.setdefault(t["source"], []).append(t)
    for src, ts in sources.items():
        print(f"  {src}: {len(ts)} tests")

    # Extract p-values
    p_values = [t["p_original"] for t in tests]

    # Apply BH correction
    q_values, significant = benjamini_hochberg(p_values, alpha=0.05)

    # Attach to tests
    for i, t in enumerate(tests):
        t["q_value"] = float(q_values[i])
        t["significant_fdr"] = bool(significant[i])

    # Print results
    print(f"\n{'=' * 100}")
    print(f"{'Source':<30} {'Test':<45} {'d':>7} {'p_orig':>12} {'q_value':>12} {'Sig?':>5}")
    print(f"{'=' * 100}")
    for t in sorted(tests, key=lambda x: x["p_original"]):
        sig_mark = "✓" if t["significant_fdr"] else "✗"
        print(f"{t['source']:<30} {t['test']:<45} {t.get('d', 0):>7.3f} "
              f"{t['p_original']:>12.2e} {t['q_value']:>12.2e} {sig_mark:>5}")

    # Summary
    n_sig = sum(1 for t in tests if t["significant_fdr"])
    n_total = len(tests)
    print(f"\n{'=' * 70}")
    print(f"SUMMARY: {n_sig}/{n_total} tests survive FDR correction (α=0.05)")
    print(f"{'=' * 70}")

    # Tests that LOSE significance after FDR
    lost = [t for t in tests if t["p_original"] < 0.05 and not t["significant_fdr"]]
    if lost:
        print(f"\n⚠ Tests significant at p<0.05 but NOT after FDR:")
        for t in lost:
            print(f"  - {t['source']}: {t['test']} (p={t['p_original']:.4f}, q={t['q_value']:.4f})")
    else:
        print(f"\n✓ All tests significant at p<0.05 remain significant after FDR correction.")

    # Save results
    out_dir = RESULTS / "fdr_correction"
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {
        "timestamp": timestamp,
        "n_tests": n_total,
        "n_significant_fdr": n_sig,
        "alpha": 0.05,
        "method": "Benjamini-Hochberg",
        "tests": tests,
    }
    path = out_dir / f"fdr_results_{timestamp}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
