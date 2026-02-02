#!/usr/bin/env python3
"""
Compare prompt-pass-only results to original generation-based results.

Usage:
    python3 compare_prompt_pass_results.py

This script:
1. Finds all prompt-pass results in results/phase1_mechanism/
2. Finds corresponding original results
3. Compares key metrics (rv_delta, p-values, effect sizes)
4. Generates comparison table
5. Identifies which component (PR_early or PR_late) drives effects

Output:
- Comparison table printed to console
- Summary of findings
- Recommendations for paper
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np


def find_results(experiment_name, layer):
    """Find most recent results for given experiment and layer."""
    results_dir = Path("results/phase1_mechanism")

    # Find directories matching pattern
    pattern = f"{experiment_name}_l{layer}_*"
    matching_dirs = list(results_dir.glob(pattern))

    if not matching_dirs:
        return None

    # Get most recent (by directory name timestamp)
    most_recent = sorted(matching_dirs)[-1]

    summary_path = most_recent / "summary.json"
    csv_path = most_recent / f"{experiment_name}.csv"

    if not summary_path.exists():
        return None

    with open(summary_path, "r") as f:
        summary = json.load(f)

    csv_data = None
    if csv_path.exists():
        csv_data = pd.read_csv(csv_path)

    return {
        "dir": most_recent,
        "summary": summary,
        "csv": csv_data
    }


def compare_layer(layer):
    """Compare prompt-pass vs generation results for a single layer."""
    # Find prompt-pass results
    pp_results = find_results("mlp_ablation_necessity_prompt_pass", layer)

    # Find original generation results
    gen_results = find_results("mlp_ablation_necessity", layer)

    if pp_results is None and gen_results is None:
        return None

    comparison = {
        "layer": layer,
        "has_prompt_pass": pp_results is not None,
        "has_generation": gen_results is not None,
    }

    # Prompt-pass metrics
    if pp_results:
        pp_sum = pp_results["summary"]
        comparison.update({
            "pp_rv_delta": pp_sum.get("rv_delta_mean"),
            "pp_rv_pvalue": pp_sum.get("rv_pvalue"),
            "pp_rv_cohens_d": pp_sum.get("rv_cohens_d"),
            "pp_pr_early_delta": pp_sum.get("pr_early_delta_mean"),
            "pp_pr_early_pvalue": pp_sum.get("pr_early_pvalue"),
            "pp_pr_late_delta": pp_sum.get("pr_late_delta_mean"),
            "pp_pr_late_pvalue": pp_sum.get("pr_late_pvalue"),
            "pp_dominant_component": pp_sum.get("dominant_component"),
            "pp_verdict": pp_sum.get("verdict"),
            "pp_n_pairs": pp_sum.get("n_pairs"),
        })

    # Generation metrics
    if gen_results:
        gen_sum = gen_results["summary"]
        comparison.update({
            "gen_rv_delta": gen_sum.get("rv_delta_mean"),
            "gen_rv_pvalue": gen_sum.get("rv_pvalue"),
            "gen_rv_cohens_d": gen_sum.get("rv_cohens_d"),
            "gen_verdict": gen_sum.get("verdict"),
            "gen_n_pairs": gen_sum.get("n_pairs"),
        })

    # Compute differences
    if pp_results and gen_results:
        pp_delta = comparison["pp_rv_delta"]
        gen_delta = comparison["gen_rv_delta"]

        if pp_delta is not None and gen_delta is not None:
            comparison["delta_difference"] = pp_delta - gen_delta
            comparison["delta_ratio"] = pp_delta / gen_delta if gen_delta != 0 else float("nan")
            comparison["same_sign"] = np.sign(pp_delta) == np.sign(gen_delta)

            # Effect size comparison
            pp_d = comparison["pp_rv_cohens_d"]
            gen_d = comparison["gen_rv_cohens_d"]
            if pp_d is not None and gen_d is not None:
                comparison["cohens_d_difference"] = pp_d - gen_d
                comparison["cohens_d_ratio"] = abs(pp_d) / abs(gen_d) if gen_d != 0 else float("nan")

    return comparison


def print_comparison_table(comparisons):
    """Print formatted comparison table."""
    print("\n" + "="*120)
    print("PROMPT-PASS vs GENERATION MODE COMPARISON")
    print("="*120)

    # Header
    print(f"\n{'Layer':<6} {'Mode':<12} {'R_V Δ':<10} {'p-value':<12} {'Cohen d':<10} "
          f"{'Dominant':<12} {'Verdict':<40}")
    print("-"*120)

    for comp in comparisons:
        if comp is None:
            continue

        layer = comp["layer"]

        # Prompt-pass row
        if comp["has_prompt_pass"]:
            pp_delta = f"{comp['pp_rv_delta']:.4f}" if comp.get("pp_rv_delta") is not None else "N/A"
            pp_pval = f"{comp['pp_rv_pvalue']:.2e}" if comp.get("pp_rv_pvalue") is not None else "N/A"
            pp_cohens = f"{comp['pp_rv_cohens_d']:.3f}" if comp.get("pp_rv_cohens_d") is not None else "N/A"
            pp_dom = comp.get("pp_dominant_component", "N/A")[:11]
            pp_verdict = comp.get("pp_verdict", "N/A")[:39]

            print(f"L{layer:<5} {'Prompt-Pass':<12} {pp_delta:<10} {pp_pval:<12} {pp_cohens:<10} "
                  f"{pp_dom:<12} {pp_verdict:<40}")

            # Component breakdown (if available)
            if comp.get("pp_pr_early_delta") is not None:
                early_delta = f"{comp['pp_pr_early_delta']:.4f}"
                late_delta = f"{comp['pp_pr_late_delta']:.4f}"
                early_pval = f"{comp['pp_pr_early_pvalue']:.2e}" if comp.get("pp_pr_early_pvalue") is not None else "N/A"
                late_pval = f"{comp['pp_pr_late_pvalue']:.2e}" if comp.get("pp_pr_late_pvalue") is not None else "N/A"

                print(f"       {'  PR_early:':<12} {early_delta:<10} {early_pval:<12}")
                print(f"       {'  PR_late:':<12} {late_delta:<10} {late_pval:<12}")

        # Generation row
        if comp["has_generation"]:
            gen_delta = f"{comp['gen_rv_delta']:.4f}" if comp.get("gen_rv_delta") is not None else "N/A"
            gen_pval = f"{comp['gen_rv_pvalue']:.2e}" if comp.get("gen_rv_pvalue") is not None else "N/A"
            gen_cohens = f"{comp['gen_rv_cohens_d']:.3f}" if comp.get("gen_rv_cohens_d") is not None else "N/A"
            gen_verdict = comp.get("gen_verdict", "N/A")[:39]

            print(f"       {'Generation':<12} {gen_delta:<10} {gen_pval:<12} {gen_cohens:<10} "
                  f"{'N/A':<12} {gen_verdict:<40}")

        # Difference row (if both exist)
        if comp.get("delta_difference") is not None:
            diff = comp["delta_difference"]
            ratio = comp["delta_ratio"]
            same_sign = "✓" if comp["same_sign"] else "✗"

            print(f"       {'Difference:':<12} {diff:+.4f}    "
                  f"{'Ratio:':<6} {ratio:.3f}    "
                  f"{'Same sign:':<10} {same_sign}")

        print("-"*120)


def analyze_findings(comparisons):
    """Analyze overall findings and print summary."""
    print("\n" + "="*120)
    print("ANALYSIS SUMMARY")
    print("="*120)

    valid_comparisons = [c for c in comparisons if c is not None and c.get("delta_difference") is not None]

    if not valid_comparisons:
        print("\n❌ No valid comparisons found. Run experiments first.")
        return

    # Check if inverse pattern persists
    inverse_layers_gen = []
    inverse_layers_pp = []

    for comp in valid_comparisons:
        layer = comp["layer"]

        # Generation mode inverse pattern
        if comp.get("gen_rv_delta") is not None and comp["gen_rv_delta"] < -0.05:
            if comp.get("gen_rv_pvalue") is not None and comp["gen_rv_pvalue"] < 0.01:
                inverse_layers_gen.append(layer)

        # Prompt-pass mode inverse pattern
        if comp.get("pp_rv_delta") is not None and comp["pp_rv_delta"] < -0.05:
            if comp.get("pp_rv_pvalue") is not None and comp["pp_rv_pvalue"] < 0.01:
                inverse_layers_pp.append(layer)

    print(f"\n1. INVERSE PATTERN ANALYSIS (R_V delta < -0.05, p < 0.01)")
    print(f"   Generation mode:  Layers {inverse_layers_gen if inverse_layers_gen else 'None'}")
    print(f"   Prompt-pass mode: Layers {inverse_layers_pp if inverse_layers_pp else 'None'}")

    if set(inverse_layers_gen) == set(inverse_layers_pp) and inverse_layers_gen:
        print(f"   ✅ PATTERN PERSISTS - Real geometric effect at layers {inverse_layers_pp}")
    elif inverse_layers_gen and not inverse_layers_pp:
        print(f"   ❌ PATTERN DISAPPEARED - Was measurement artifact")
    elif inverse_layers_pp and not inverse_layers_gen:
        print(f"   ⚠️  NEW PATTERN EMERGED - Requires investigation")
    else:
        print(f"   ℹ️  No clear inverse pattern in either mode")

    # Component analysis
    print(f"\n2. COMPONENT ANALYSIS (Prompt-pass mode)")
    for comp in valid_comparisons:
        layer = comp["layer"]
        dominant = comp.get("pp_dominant_component", "Unknown")

        if dominant != "Unknown" and dominant is not None:
            early_p = comp.get("pp_pr_early_pvalue")
            late_p = comp.get("pp_pr_late_pvalue")

            early_sig = "✓" if early_p is not None and early_p < 0.01 else "✗"
            late_sig = "✓" if late_p is not None and late_p < 0.01 else "✗"

            print(f"   L{layer}: Dominant={dominant:<10} (PR_early sig={early_sig}, PR_late sig={late_sig})")

    # Effect size comparison
    print(f"\n3. EFFECT SIZE COMPARISON")
    for comp in valid_comparisons:
        layer = comp["layer"]
        ratio = comp.get("cohens_d_ratio")

        if ratio is not None and not np.isnan(ratio):
            if 0.8 <= ratio <= 1.2:
                status = "✓ Similar"
            elif ratio < 0.8:
                status = "⚠️ Weaker in prompt-pass"
            else:
                status = "⚠️ Stronger in prompt-pass"

            print(f"   L{layer}: |Cohen's d| ratio = {ratio:.3f}  ({status})")

    # Recommendations
    print(f"\n4. RECOMMENDATIONS FOR PAPER")

    if set(inverse_layers_gen) == set(inverse_layers_pp) and inverse_layers_gen:
        print(f"   • Report inverse pattern as VALIDATED geometric effect")
        print(f"   • Emphasize component analysis (which PR component drives effect)")
        print(f"   • Discuss compensatory mechanisms at layers {inverse_layers_pp}")
    elif inverse_layers_gen and not inverse_layers_pp:
        print(f"   • Report inverse pattern as MEASUREMENT ARTIFACT")
        print(f"   • Revise methodology section to use prompt-pass approach")
        print(f"   • Caution against measuring R_V on generated text")
    else:
        print(f"   • Report nuanced findings - pattern varies by measurement approach")
        print(f"   • Discuss implications of generation vs prompt-pass measurement")

    print("\n" + "="*120 + "\n")


def main():
    """Main comparison workflow."""
    print("\n" + "="*120)
    print("COMPARING PROMPT-PASS vs GENERATION ABLATION RESULTS")
    print("="*120)

    # Compare layers 0-5
    comparisons = []
    for layer in range(6):
        comp = compare_layer(layer)
        comparisons.append(comp)

    # Print table
    print_comparison_table(comparisons)

    # Analyze findings
    analyze_findings(comparisons)

    # Export to CSV
    valid_comparisons = [c for c in comparisons if c is not None]
    if valid_comparisons:
        df = pd.DataFrame(valid_comparisons)
        output_path = Path("results/comparison_prompt_pass_vs_generation.csv")
        df.to_csv(output_path, index=False)
        print(f"📊 Comparison data saved to: {output_path}")


if __name__ == "__main__":
    main()
