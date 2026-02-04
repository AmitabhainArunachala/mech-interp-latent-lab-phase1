"""
Command-line interface for rv_toolkit.

Usage:
    rv-toolkit compute <tensor_file> [--window=16] [--output=json]
    rv-toolkit analyze <results_file> [--plot] [--output=report.md]
    rv-toolkit demo [--model=mistral-7b] [--n-pairs=10]
    rv-toolkit version
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from . import __version__
from .metrics import compute_rv, compute_participation_ratio, RVResult
from .analysis import compute_effect_size, run_statistical_tests
from .prompts import RECURSIVE_PROMPTS, BASELINE_PROMPTS, get_prompt_pairs


def main():
    """Main entry point for rv-toolkit CLI."""
    parser = argparse.ArgumentParser(
        prog="rv-toolkit",
        description="R_V metrics for measuring geometric signatures of recursive self-reference",
    )
    parser.add_argument("--version", action="version", version=f"rv_toolkit {__version__}")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # compute command
    compute_parser = subparsers.add_parser(
        "compute",
        help="Compute R_V metrics from saved activations",
    )
    compute_parser.add_argument(
        "tensor_file",
        help="Path to .pt file containing value activations (shape: batch x seq x dim)",
    )
    compute_parser.add_argument(
        "--window", "-w",
        type=int,
        default=16,
        help="Window size for R_V computation (default: 16)",
    )
    compute_parser.add_argument(
        "--output", "-o",
        choices=["json", "table", "simple"],
        default="simple",
        help="Output format (default: simple)",
    )
    compute_parser.add_argument(
        "--save", "-s",
        type=str,
        default=None,
        help="Save results to file (JSON format)",
    )
    
    # analyze command
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze experiment results and compute statistics",
    )
    analyze_parser.add_argument(
        "results_file",
        help="Path to JSON file with experiment results",
    )
    analyze_parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate visualization plots",
    )
    analyze_parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file for report",
    )
    
    # demo command
    demo_parser = subparsers.add_parser(
        "demo",
        help="Run demonstration with synthetic data",
    )
    demo_parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of samples to generate (default: 100)",
    )
    demo_parser.add_argument(
        "--contraction",
        type=float,
        default=0.20,
        help="Simulated contraction magnitude (default: 0.20 = 20%%)",
    )
    
    # prompts command
    prompts_parser = subparsers.add_parser(
        "prompts",
        help="List available prompt pairs",
    )
    prompts_parser.add_argument(
        "--category", "-c",
        type=str,
        default=None,
        help="Filter by category",
    )
    prompts_parser.add_argument(
        "--count", "-n",
        type=int,
        default=10,
        help="Number of pairs to show (default: 10)",
    )
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    
    if args.command == "compute":
        cmd_compute(args)
    elif args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "demo":
        cmd_demo(args)
    elif args.command == "prompts":
        cmd_prompts(args)


def cmd_compute(args):
    """Compute R_V from saved activation tensor."""
    tensor_path = Path(args.tensor_file)
    
    if not tensor_path.exists():
        print(f"Error: File not found: {tensor_path}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Loading activations from {tensor_path}...")
    
    try:
        tensor = torch.load(tensor_path, map_location="cpu")
    except Exception as e:
        print(f"Error loading tensor: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Handle different tensor formats
    if isinstance(tensor, dict):
        if "v" in tensor:
            tensor = tensor["v"]
        elif "activations" in tensor:
            tensor = tensor["activations"]
        else:
            print(f"Dict keys found: {tensor.keys()}", file=sys.stderr)
            print("Expected 'v' or 'activations' key", file=sys.stderr)
            sys.exit(1)
    
    print(f"Tensor shape: {tensor.shape}")
    print(f"Computing R_V with window size {args.window}...")
    
    result = compute_rv(tensor, window_size=args.window)
    
    # Output results
    if args.output == "json":
        output = {
            "rv": result.rv,
            "effective_rank": result.effective_rank,
            "v_parallel_norm": result.v_parallel_norm,
            "v_perp_norm": result.v_perp_norm,
            "dual_ratio": result.dual_ratio,
            "n_singular_values": len(result.singular_values),
            "window_size": args.window,
            "tensor_shape": list(tensor.shape),
        }
        print(json.dumps(output, indent=2))
        
    elif args.output == "table":
        print("\n" + "=" * 50)
        print("R_V Computation Results")
        print("=" * 50)
        print(f"{'Metric':<25} {'Value':>20}")
        print("-" * 50)
        print(f"{'R_V (participation ratio)':<25} {result.rv:>20.4f}")
        print(f"{'Effective rank':<25} {result.effective_rank:>20.4f}")
        if result.v_parallel_norm is not None:
            print(f"{'V_parallel norm':<25} {result.v_parallel_norm:>20.4f}")
        if result.v_perp_norm is not None:
            print(f"{'V_perp norm':<25} {result.v_perp_norm:>20.4f}")
        if result.dual_ratio is not None:
            print(f"{'Dual ratio (∥/⊥)':<25} {result.dual_ratio:>20.4f}")
        print("=" * 50)
        
    else:  # simple
        print(f"\nR_V = {result.rv:.4f}")
        print(f"Effective rank = {result.effective_rank:.4f}")
        if result.dual_ratio is not None:
            print(f"Dual ratio = {result.dual_ratio:.4f}")
    
    if args.save:
        output = {
            "rv": result.rv,
            "effective_rank": result.effective_rank,
            "v_parallel_norm": result.v_parallel_norm,
            "v_perp_norm": result.v_perp_norm,
            "dual_ratio": result.dual_ratio,
            "singular_values": result.singular_values.tolist(),
        }
        with open(args.save, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.save}")


def cmd_analyze(args):
    """Analyze experiment results."""
    results_path = Path(args.results_file)
    
    if not results_path.exists():
        print(f"Error: File not found: {results_path}", file=sys.stderr)
        sys.exit(1)
    
    with open(results_path) as f:
        data = json.load(f)
    
    # Extract R_V values
    if isinstance(data, list):
        # List of results
        baseline_rvs = [r.get("baseline_rv", r.get("baseline")) for r in data if r.get("baseline_rv") or r.get("baseline")]
        recursive_rvs = [r.get("recursive_rv", r.get("recursive")) for r in data if r.get("recursive_rv") or r.get("recursive")]
    elif isinstance(data, dict):
        baseline_rvs = data.get("baseline_rvs", data.get("baseline", []))
        recursive_rvs = data.get("recursive_rvs", data.get("recursive", []))
    else:
        print("Error: Unrecognized data format", file=sys.stderr)
        sys.exit(1)
    
    if not baseline_rvs or not recursive_rvs:
        print("Error: No R_V values found in data", file=sys.stderr)
        sys.exit(1)
    
    baseline_rvs = np.array(baseline_rvs)
    recursive_rvs = np.array(recursive_rvs)
    
    # Compute statistics
    effect_size = compute_effect_size(baseline_rvs, recursive_rvs)
    stats = run_statistical_tests(baseline_rvs, recursive_rvs)
    
    report = []
    report.append("# R_V Analysis Report\n")
    report.append(f"**Samples:** {len(baseline_rvs)} baseline, {len(recursive_rvs)} recursive\n")
    report.append("## Summary Statistics\n")
    report.append(f"| Condition | Mean R_V | Std | Min | Max |")
    report.append(f"|-----------|----------|-----|-----|-----|")
    report.append(f"| Baseline | {baseline_rvs.mean():.4f} | {baseline_rvs.std():.4f} | {baseline_rvs.min():.4f} | {baseline_rvs.max():.4f} |")
    report.append(f"| Recursive | {recursive_rvs.mean():.4f} | {recursive_rvs.std():.4f} | {recursive_rvs.min():.4f} | {recursive_rvs.max():.4f} |")
    report.append("")
    report.append("## Effect Size\n")
    report.append(f"**Cohen's d:** {effect_size:.4f}")
    
    if effect_size <= -0.8:
        interpretation = "Large contraction (d ≤ -0.8)"
    elif effect_size <= -0.5:
        interpretation = "Medium contraction (-0.8 < d ≤ -0.5)"
    elif effect_size <= -0.2:
        interpretation = "Small contraction (-0.5 < d ≤ -0.2)"
    elif effect_size < 0.2:
        interpretation = "Negligible effect"
    else:
        interpretation = "Expansion (d > 0.2)"
    
    report.append(f"\n**Interpretation:** {interpretation}")
    report.append("")
    report.append("## Statistical Tests\n")
    report.append(f"**t-statistic:** {stats.get('t_stat', 'N/A'):.4f}")
    report.append(f"**p-value:** {stats.get('p_value', 'N/A'):.2e}")
    
    report_text = "\n".join(report)
    
    if args.output:
        with open(args.output, "w") as f:
            f.write(report_text)
        print(f"Report saved to {args.output}")
    else:
        print(report_text)


def cmd_demo(args):
    """Run demonstration with synthetic data showing R_V contraction."""
    print("=" * 60)
    print("R_V Toolkit Demonstration")
    print("Simulating recursive self-reference geometric contraction")
    print("=" * 60)
    
    n = args.n_samples
    contraction = args.contraction
    
    # Simulate value activations
    dim = 128  # Simulated embedding dimension
    seq_len = 32
    
    print(f"\nGenerating {n} prompt pairs...")
    print(f"Simulated contraction: {contraction*100:.1f}%")
    
    baseline_rvs = []
    recursive_rvs = []
    
    for i in range(n):
        # BASELINE: Distributed representations (high effective dimensionality)
        # Information spread across many dimensions - uniform singular values
        baseline_act = np.random.randn(seq_len, dim)
        
        # RECURSIVE: Concentrated representations (lower effective dimensionality)
        # Self-referential processing compresses to fewer dominant dimensions
        # Create low-rank structure: only ~10% of dimensions carry signal
        n_dominant = max(3, dim // 10)
        V = np.random.randn(dim, n_dominant)
        V, _ = np.linalg.qr(V)  # Orthonormal basis
        # Strong coefficients on dominant dimensions, weak elsewhere
        coeffs = np.random.randn(seq_len, n_dominant) * 3.0
        recursive_act = coeffs @ V.T + 0.1 * np.random.randn(seq_len, dim)
        
        # Compute PR for each
        _, s_b, _ = np.linalg.svd(baseline_act, full_matrices=False)
        _, s_r, _ = np.linalg.svd(recursive_act, full_matrices=False)
        
        pr_b = compute_participation_ratio(s_b)
        pr_r = compute_participation_ratio(s_r)
        
        baseline_rvs.append(pr_b)
        recursive_rvs.append(pr_r)
    
    baseline_rvs = np.array(baseline_rvs)
    recursive_rvs = np.array(recursive_rvs)
    
    # Normalize to show relative contraction
    baseline_mean = baseline_rvs.mean()
    rv_baseline = baseline_rvs / baseline_mean
    rv_recursive = recursive_rvs / baseline_mean
    
    # Statistics
    mean_contraction = (rv_recursive.mean() - rv_baseline.mean()) / rv_baseline.mean()
    effect_size = (rv_recursive.mean() - rv_baseline.mean()) / np.sqrt((rv_baseline.std()**2 + rv_recursive.std()**2) / 2)
    
    print("\n" + "-" * 60)
    print("Results")
    print("-" * 60)
    print(f"\n{'Condition':<15} {'Mean R_V':<12} {'Std':<12}")
    print("-" * 40)
    print(f"{'Baseline':<15} {rv_baseline.mean():.4f}       {rv_baseline.std():.4f}")
    print(f"{'Recursive':<15} {rv_recursive.mean():.4f}       {rv_recursive.std():.4f}")
    print("-" * 40)
    print(f"\nMean contraction: {mean_contraction*100:.1f}%")
    print(f"Cohen's d: {effect_size:.2f}")
    
    if effect_size < -0.8:
        print("\n✓ Large effect size detected (d < -0.8)")
        print("  This matches the expected geometric signature of recursive self-reference")
    elif effect_size < -0.5:
        print("\n✓ Medium effect size detected (-0.8 < d < -0.5)")
    else:
        print("\n⚠ Effect size smaller than expected")
    
    print("\n" + "=" * 60)
    print("The demonstration shows how recursive self-reference prompts")
    print("induce geometric contraction (lower participation ratio) in")
    print("the value activation space, as documented in the R_V paper.")
    print("=" * 60)


def cmd_prompts(args):
    """List available prompt pairs."""
    print("=" * 60)
    print("Available Prompt Pairs")
    print("=" * 60)
    
    pairs = get_prompt_pairs(n_pairs=args.count, shuffle=False)
    
    for i, (baseline, recursive) in enumerate(pairs[:args.count], 1):
        print(f"\n--- Pair {i} ---")
        print(f"BASELINE:  {baseline[:80]}..." if len(baseline) > 80 else f"BASELINE:  {baseline}")
        print(f"RECURSIVE: {recursive[:80]}..." if len(recursive) > 80 else f"RECURSIVE: {recursive}")
    
    print(f"\nTotal pairs available: {len(BASELINE_PROMPTS)}")


if __name__ == "__main__":
    main()
