#!/usr/bin/env python3
"""
Reproduce Results: Standard Battery Entry Point

Run this script to reproduce the "Geometric Contraction" results.
Generates clean, indisputable graphs before your eyes.

Usage:
    python reproduce_results.py
    python reproduce_results.py --model mistralai/Mistral-7B-v0.1 --device cuda
"""

import argparse
import os
from pathlib import Path

import torch

from src.pipelines.phase1_existence import run_phase1_existence_proof


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce R_V geometric contraction results"
    )
    parser.add_argument(
        "--model",
        default="mistralai/Mistral-7B-v0.1",
        help="Model identifier. Default: Mistral-7B Base (reference reality)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Target device",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--results_dir",
        default="results",
        help="Directory for output CSVs",
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("OPERATION SAMURAI: REPRODUCE RESULTS")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Seed: {args.seed}")
    print(f"Results: {args.results_dir}")
    print("=" * 60)
    print()
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Run Phase 1: Existence Proof
    print("Running Phase 1: Existence Proof...")
    try:
        layer_sweep_csv, battery_csv = run_phase1_existence_proof(
            model_name=args.model,
            device=args.device,
            seed=args.seed,
            results_dir=args.results_dir,
        )
        print()
        print("✓ Phase 1 complete")
        print(f"  - Layer sweep: {layer_sweep_csv}")
        print(f"  - Prompt battery: {battery_csv}")
    except Exception as e:
        print(f"✗ Phase 1 failed: {e}")
        raise
    
    print()
    print("=" * 60)
    print("REPRODUCTION COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {args.results_dir}/")
    print()
    print("Next steps:")
    print("  1. Analyze CSVs to verify R_V < 1.0 for recursive prompts")
    print("  2. Plot layer sweep to visualize contraction point")
    print("  3. Compare recursive vs baseline separation")


if __name__ == "__main__":
    main()

