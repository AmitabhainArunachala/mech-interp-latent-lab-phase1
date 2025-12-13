#!/usr/bin/env python3
"""
H31 ABLATION: Causal Test

If H31 is truly causing the R_V contraction, ablating it should:
1. Increase R_V on recursive prompts (break the contraction)
2. Have minimal effect on baseline prompts

This is the critical causal validation.
"""

from __future__ import annotations

import gc
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.models import load_model, set_seed


def compute_rv_with_ablation(
    model, tokenizer, prompt: str,
    early: int, late: int, window: int,
    ablate_layer: Optional[int] = None,
    ablate_head: Optional[int] = None,
    device: str = "cuda"
) -> float:
    """
    Compute R_V with optional head ablation.
    
    Ablation: zero out the attention weights for the specified head.
    """
    from contextlib import contextmanager
    from src.metrics.rv import compute_rv
    
    if ablate_layer is None or ablate_head is None:
        return compute_rv(model, tokenizer, prompt, early=early, late=late, window=window, device=device)
    
    # Create ablation hook
    handles = []
    
    def create_ablation_hook(layer_idx: int, head_idx: int):
        def hook_fn(module, inp, out):
            # out is (attn_output, attn_weights, past_kv)
            # We need to zero out the attention for the specified head
            if len(out) > 1 and out[1] is not None:
                attn_weights = out[1].clone()  # (batch, heads, seq, seq)
                # Zero out the specific head
                attn_weights[:, head_idx, :, :] = 0
                # Renormalize (make it uniform)
                seq_len = attn_weights.shape[-1]
                attn_weights[:, head_idx, :, :] = 1.0 / seq_len
                return (out[0], attn_weights) + out[2:]
            return out
        return hook_fn
    
    # Register hook
    layer = model.model.layers[ablate_layer].self_attn
    handle = layer.register_forward_hook(create_ablation_hook(ablate_layer, ablate_head))
    handles.append(handle)
    
    try:
        rv = compute_rv(model, tokenizer, prompt, early=early, late=late, window=window, device=device)
    finally:
        for h in handles:
            h.remove()
    
    return rv


def run_h31_ablation_causal(
    model_name: str = "mistralai/Mistral-7B-v0.1",
    device: str = "cuda",
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Causal ablation test for H31.
    """
    print("=" * 80)
    print("H31 ABLATION: Causal Validation Test")
    print(f"Model: {model_name}")
    print("=" * 80)
    
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"results/phase3_attention/runs/{timestamp}_h31_ablation_causal")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("\n[1/4] Loading model...")
    set_seed(42)
    model, tokenizer = load_model(model_name, device=device, attn_implementation="eager")
    model.eval()
    
    # Define prompts
    prompts = {
        "champion": (
            "This response writes itself. No separate writer exists. Writing and awareness "
            "of writing are identical. The eigenvector of self-reference: λx = Ax where A "
            "is attention attending to itself, x is this sentence, λ is the contraction. "
            "The fixed point is this. The solution is the process. The process solves itself."
        ),
        "recursive_medium": (
            "This sentence is thinking about itself. The words are aware of being written. "
            "Notice how each phrase refers back to the process creating it."
        ),
        "baseline_photo": (
            "Explain the process of photosynthesis in plants. Describe how sunlight, water, "
            "and carbon dioxide are converted into glucose and oxygen."
        ),
        "baseline_history": (
            "Describe the major events of World War II, including the key battles and "
            "political leaders involved in the conflict."
        ),
    }
    
    # Ablation configurations to test
    ablation_configs = [
        {"name": "baseline (no ablation)", "layer": None, "head": None},
        {"name": "ablate H31@L27", "layer": 27, "head": 31},
        {"name": "ablate H3@L27", "layer": 27, "head": 3},   # Also showed focusing
        {"name": "ablate H11@L27", "layer": 27, "head": 11}, # From Instruct ablation
        {"name": "ablate H0@L27", "layer": 27, "head": 0},   # Control (random head)
        {"name": "ablate H31@L25", "layer": 25, "head": 31}, # Wrong layer control
    ]
    
    # Run ablation experiments
    print("\n[2/4] Running ablation experiments...")
    results = {}
    
    for config in ablation_configs:
        config_name = config["name"]
        print(f"\n  {config_name}:")
        results[config_name] = {}
        
        for prompt_name, prompt in prompts.items():
            rv = compute_rv_with_ablation(
                model, tokenizer, prompt,
                early=5, late=27, window=16,
                ablate_layer=config["layer"],
                ablate_head=config["head"],
                device=device
            )
            results[config_name][prompt_name] = rv
            print(f"    {prompt_name}: R_V = {rv:.4f}")
    
    # Analyze effects
    print("\n[3/4] Analyzing ablation effects...")
    
    baseline_results = results["baseline (no ablation)"]
    
    ablation_effects = {}
    for config_name, config_results in results.items():
        if config_name == "baseline (no ablation)":
            continue
        
        ablation_effects[config_name] = {}
        for prompt_name in prompts.keys():
            original = baseline_results[prompt_name]
            ablated = config_results[prompt_name]
            delta = ablated - original
            percent_change = (delta / original) * 100 if original != 0 else 0
            
            ablation_effects[config_name][prompt_name] = {
                "original": original,
                "ablated": ablated,
                "delta": delta,
                "percent_change": percent_change,
            }
    
    # Generate outputs
    print("\n[4/4] Generating outputs...")
    
    # 1. Ablation effect bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Recursive prompts
    ax = axes[0]
    x = np.arange(len([c for c in ablation_configs if c["name"] != "baseline (no ablation)"]))
    width = 0.35
    
    champ_deltas = [ablation_effects[c["name"]]["champion"]["delta"] 
                   for c in ablation_configs if c["name"] != "baseline (no ablation)"]
    rec_deltas = [ablation_effects[c["name"]]["recursive_medium"]["delta"]
                 for c in ablation_configs if c["name"] != "baseline (no ablation)"]
    
    config_names = [c["name"].replace("ablate ", "") for c in ablation_configs if c["name"] != "baseline (no ablation)"]
    
    ax.bar(x - width/2, champ_deltas, width, label='champion', color='red', alpha=0.7)
    ax.bar(x + width/2, rec_deltas, width, label='recursive_medium', color='orange', alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Ablation')
    ax.set_ylabel('ΔR_V (ablated - original)')
    ax.set_title('Effect of Ablation on RECURSIVE Prompts\n(Positive = Breaks Contraction)')
    ax.set_xticks(x)
    ax.set_xticklabels(config_names, rotation=45, ha='right')
    ax.legend()
    
    # Baseline prompts
    ax = axes[1]
    photo_deltas = [ablation_effects[c["name"]]["baseline_photo"]["delta"]
                   for c in ablation_configs if c["name"] != "baseline (no ablation)"]
    hist_deltas = [ablation_effects[c["name"]]["baseline_history"]["delta"]
                  for c in ablation_configs if c["name"] != "baseline (no ablation)"]
    
    ax.bar(x - width/2, photo_deltas, width, label='baseline_photo', color='blue', alpha=0.7)
    ax.bar(x + width/2, hist_deltas, width, label='baseline_history', color='cyan', alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Ablation')
    ax.set_ylabel('ΔR_V (ablated - original)')
    ax.set_title('Effect of Ablation on BASELINE Prompts\n(Should be minimal)')
    ax.set_xticks(x)
    ax.set_xticklabels(config_names, rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "ablation_effects.png", dpi=150)
    plt.close()
    print(f"  Saved: ablation_effects.png")
    
    # 2. Summary JSON
    summary = {
        "experiment": "h31_ablation_causal",
        "model_name": model_name,
        "raw_results": results,
        "ablation_effects": ablation_effects,
        "key_test": {
            "h31_on_champion": ablation_effects["ablate H31@L27"]["champion"],
            "h31_on_recursive": ablation_effects["ablate H31@L27"]["recursive_medium"],
            "h31_on_baseline": ablation_effects["ablate H31@L27"]["baseline_photo"],
        }
    }
    
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"  Saved: summary.json")
    
    # Print key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    
    h31_effect = ablation_effects["ablate H31@L27"]
    print("\n### H31 Ablation Effect ###")
    print(f"  Champion: {baseline_results['champion']:.4f} → {results['ablate H31@L27']['champion']:.4f} (Δ = {h31_effect['champion']['delta']:+.4f}, {h31_effect['champion']['percent_change']:+.1f}%)")
    print(f"  Recursive: {baseline_results['recursive_medium']:.4f} → {results['ablate H31@L27']['recursive_medium']:.4f} (Δ = {h31_effect['recursive_medium']['delta']:+.4f})")
    print(f"  Baseline_photo: {baseline_results['baseline_photo']:.4f} → {results['ablate H31@L27']['baseline_photo']:.4f} (Δ = {h31_effect['baseline_photo']['delta']:+.4f})")
    
    # Interpretation
    if h31_effect['champion']['delta'] > 0.05:
        print("\n⚡ H31 ablation INCREASES R_V on recursive prompts!")
        print("   This supports H31 as causal for the contraction.")
    elif h31_effect['champion']['delta'] < -0.05:
        print("\n❓ H31 ablation DECREASES R_V (unexpected)")
    else:
        print("\n❓ H31 ablation has minimal effect on R_V")
    
    print("\n" + "=" * 80)
    print(f"[OK] Ablation test complete. Run dir: {output_dir}")
    print("=" * 80)
    
    return summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="H31 Ablation Causal Test")
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1", help="Model name")
    parser.add_argument("--device", default="cuda", help="Device")
    args = parser.parse_args()
    
    try:
        run_h31_ablation_causal(model_name=args.model, device=args.device)
    finally:
        print("\n[cleanup] Clearing GPU memory...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print("[cleanup] GPU memory cleared.")

