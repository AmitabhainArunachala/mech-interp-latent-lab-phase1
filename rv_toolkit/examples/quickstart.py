#!/usr/bin/env python3
"""
Quickstart Example: Measuring R_V in Transformer Value Spaces

This example demonstrates how to use rv_toolkit to measure geometric signatures
of recursive self-reference in transformer value activations.

The key finding from the R_V paper: recursive self-reference induces a 
characteristic ~20% contraction in participation ratio at late integration
layers (L25-L27, 78-84% depth).

Requirements:
    pip install rv_toolkit
    pip install transformers accelerate  # For model loading
    
Usage:
    python quickstart.py
    python quickstart.py --model mistralai/Mistral-7B-v0.1
    python quickstart.py --synthetic  # Use synthetic data (no GPU needed)
"""

import argparse
import numpy as np
import torch

# Core rv_toolkit imports
from rv_toolkit import (
    compute_rv,
    compute_participation_ratio,
    compute_effect_size,
    RECURSIVE_PROMPTS,
    BASELINE_PROMPTS,
)


def run_synthetic_demo():
    """
    Demonstrate R_V computation with synthetic data.
    
    This simulates the geometric signature without requiring a GPU or
    actual transformer model. Useful for understanding the metric.
    """
    print("=" * 70)
    print("R_V Quickstart: Synthetic Data Demo")
    print("=" * 70)
    
    n_samples = 50
    dim = 512  # Typical hidden dimension
    seq_len = 64
    
    print(f"\nSimulating {n_samples} prompt pairs...")
    print(f"Activation shape: (seq_len={seq_len}, dim={dim})")
    
    baseline_rvs = []
    recursive_rvs = []
    
    for i in range(n_samples):
        # BASELINE: Distributed representations (high effective dimensionality)
        # Information spread across many dimensions
        baseline_noise = 0.1 * np.random.randn(seq_len, dim)
        baseline_structure = np.random.randn(seq_len, dim) @ np.diag(np.random.exponential(1.0, dim))
        baseline_act = baseline_structure + baseline_noise
        
        # RECURSIVE: Concentrated representations (lower effective dimensionality)
        # Self-referential processing compresses to fewer dimensions
        # This simulates the "attentional focusing" that occurs during recursive observation
        n_dominant = 15  # Fewer dominant directions
        V = np.random.randn(dim, n_dominant)
        V, _ = np.linalg.qr(V)  # Orthonormal basis
        coeffs = np.random.randn(seq_len, n_dominant) * np.array([3.0 / (k+1) for k in range(n_dominant)])
        recursive_act = coeffs @ V.T + 0.05 * np.random.randn(seq_len, dim)
        
        # Compute participation ratios
        _, s_b, _ = np.linalg.svd(baseline_act, full_matrices=False)
        _, s_r, _ = np.linalg.svd(recursive_act, full_matrices=False)
        
        pr_b = compute_participation_ratio(s_b)
        pr_r = compute_participation_ratio(s_r)
        
        baseline_rvs.append(pr_b)
        recursive_rvs.append(pr_r)
    
    baseline_rvs = np.array(baseline_rvs)
    recursive_rvs = np.array(recursive_rvs)
    
    # Compute R_V as ratio (normalized by baseline layer 5 proxy)
    # In actual experiments, R_V = PR(L27) / PR(L5)
    # Here we simulate by using baseline mean as reference
    ref = baseline_rvs.mean()
    rv_baseline = baseline_rvs / ref
    rv_recursive = recursive_rvs / ref
    
    # Effect size
    d = compute_effect_size(rv_baseline, rv_recursive)
    contraction = (rv_recursive.mean() - rv_baseline.mean()) / rv_baseline.mean() * 100
    
    print("\n" + "-" * 70)
    print("Results")
    print("-" * 70)
    print(f"\n{'Condition':<15} {'Mean R_V':<12} {'Std':<12} {'N':<8}")
    print("-" * 50)
    print(f"{'Baseline':<15} {rv_baseline.mean():.4f}       {rv_baseline.std():.4f}       {len(rv_baseline)}")
    print(f"{'Recursive':<15} {rv_recursive.mean():.4f}       {rv_recursive.std():.4f}       {len(rv_recursive)}")
    
    print("\n" + "-" * 70)
    print("Key Metrics")
    print("-" * 70)
    print(f"Geometric contraction: {contraction:.1f}%")
    print(f"Cohen's d: {d:.2f}")
    
    # Interpret
    if d < -0.8:
        print("\n✓ LARGE effect (d < -0.8)")
        print("  Matches signature of recursive self-reference")
    elif d < -0.5:
        print("\n✓ MEDIUM effect (-0.8 < d < -0.5)")
    elif d < -0.2:
        print("\n○ SMALL effect (-0.5 < d < -0.2)")
    else:
        print("\n○ NEGLIGIBLE effect (|d| < 0.2)")
    
    print("\n" + "=" * 70)
    print("The participation ratio (PR) measures effective dimensionality.")
    print("Recursive self-reference concentrates representations into fewer")
    print("dimensions, yielding geometric CONTRACTION (lower R_V).")
    print("")
    print("In real Mistral-7B experiments:")
    print("  - Contraction: -16.5%")
    print("  - Cohen's d: -3.56")
    print("  - Location: Layer 27 (84% depth)")
    print("=" * 70)
    
    return rv_baseline, rv_recursive


def run_model_demo(model_name: str):
    """
    Demonstrate R_V computation with an actual transformer model.
    
    Requires: transformers, accelerate, GPU with sufficient memory
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("Error: transformers not installed")
        print("Run: pip install transformers accelerate")
        return
    
    print("=" * 70)
    print(f"R_V Quickstart: Model Demo ({model_name})")
    print("=" * 70)
    
    print(f"\nLoading model: {model_name}")
    print("This may take a few minutes...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    if device == "cpu":
        print("\n⚠ Warning: Running on CPU will be slow for large models")
        print("  Consider using a GPU or the --synthetic flag")
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
    )
    
    # Get target layer (84% depth for late integration)
    n_layers = model.config.num_hidden_layers
    target_layer = int(n_layers * 0.84)
    print(f"\nModel has {n_layers} layers")
    print(f"Target layer: {target_layer} (84% depth)")
    
    # Sample prompts
    n_pairs = 5
    print(f"\nProcessing {n_pairs} prompt pairs...")
    
    baseline_rvs = []
    recursive_rvs = []
    
    for i in range(n_pairs):
        baseline_prompt = BASELINE_PROMPTS[i]
        recursive_prompt = RECURSIVE_PROMPTS[i]
        
        print(f"\nPair {i+1}:")
        print(f"  Baseline: {baseline_prompt[:50]}...")
        print(f"  Recursive: {recursive_prompt[:50]}...")
        
        # Get activations at target layer
        def get_value_activations(prompt, layer_idx):
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True,
                )
            # Hidden state at target layer
            hidden = outputs.hidden_states[layer_idx]
            return hidden.cpu()
        
        baseline_act = get_value_activations(baseline_prompt, target_layer)
        recursive_act = get_value_activations(recursive_prompt, target_layer)
        
        # Compute R_V
        rv_b = compute_rv(baseline_act, window_size=16)
        rv_r = compute_rv(recursive_act, window_size=16)
        
        baseline_rvs.append(rv_b.rv)
        recursive_rvs.append(rv_r.rv)
        
        print(f"  R_V baseline: {rv_b.rv:.4f}")
        print(f"  R_V recursive: {rv_r.rv:.4f}")
    
    baseline_rvs = np.array(baseline_rvs)
    recursive_rvs = np.array(recursive_rvs)
    
    d = compute_effect_size(baseline_rvs, recursive_rvs)
    contraction = (recursive_rvs.mean() - baseline_rvs.mean()) / baseline_rvs.mean() * 100
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Mean baseline R_V: {baseline_rvs.mean():.4f}")
    print(f"Mean recursive R_V: {recursive_rvs.mean():.4f}")
    print(f"Contraction: {contraction:.1f}%")
    print(f"Cohen's d: {d:.2f}")
    
    return baseline_rvs, recursive_rvs


def main():
    parser = argparse.ArgumentParser(
        description="Quickstart example for rv_toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python quickstart.py --synthetic     # No GPU needed
    python quickstart.py --model mistralai/Mistral-7B-v0.1
    python quickstart.py --model meta-llama/Llama-2-7b-hf
        """,
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data (no GPU/model required)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name or path (HuggingFace format)",
    )
    
    args = parser.parse_args()
    
    if args.synthetic or args.model is None:
        run_synthetic_demo()
    else:
        run_model_demo(args.model)


if __name__ == "__main__":
    main()
