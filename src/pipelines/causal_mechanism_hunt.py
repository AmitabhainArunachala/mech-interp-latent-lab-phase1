#!/usr/bin/env python3
"""
CAUSAL MECHANISM HUNT

H31 correlates with R_V contraction but doesn't cause it.
What DOES cause it?

Hypotheses:
1. MLP at L27 creates the contraction
2. An earlier layer (L25?) creates it and L27 reads it
3. The residual stream geometry changes progressively

Approach: Track both R_V AND H31 entropy at multiple intervention points.
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
from scipy.stats import entropy as scipy_entropy

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.models import load_model, set_seed
from src.core.hooks import capture_attention_patterns


def compute_layer_geometry(hidden_states: torch.Tensor, window: int = 16) -> Dict[str, float]:
    """Compute geometric metrics for a hidden state tensor."""
    if hidden_states.dim() == 3:
        hidden_states = hidden_states[0]  # Remove batch dim
    
    T, D = hidden_states.shape
    W = min(window, T)
    window_states = hidden_states[-W:, :].float()
    
    try:
        U, S, Vt = torch.linalg.svd(window_states.T, full_matrices=False)
        S_np = S.cpu().numpy()
        S_sq = S_np ** 2
        
        if S_sq.sum() < 1e-10:
            return {"pr": np.nan, "eff_rank": np.nan, "top_sv_ratio": np.nan}
        
        p = S_sq / S_sq.sum()
        eff_rank = 1.0 / (p**2).sum()
        pr = (S_sq.sum()**2) / (S_sq**2).sum()
        top_sv_ratio = S_np[0] / S_np.sum() if S_np.sum() > 0 else 0
        
        return {
            "pr": float(pr),
            "eff_rank": float(eff_rank),
            "top_sv_ratio": float(top_sv_ratio),
        }
    except Exception:
        return {"pr": np.nan, "eff_rank": np.nan, "top_sv_ratio": np.nan}


def compute_h31_entropy(model, tokenizer, prompt: str, layer: int, device: str) -> float:
    """Get H31 entropy at a specific layer."""
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    with capture_attention_patterns(model, layer) as storage:
        with torch.no_grad():
            _ = model(**enc, output_attentions=True)
        attn_weights = storage["attn_weights"]
    
    if attn_weights is None:
        return np.nan
    
    head_attn = attn_weights[0, 31, :, :].cpu().numpy()
    entropies = []
    for i in range(head_attn.shape[0]):
        row = head_attn[i] + 1e-10
        row = row / row.sum()
        entropies.append(scipy_entropy(row))
    return float(np.mean(entropies))


def run_causal_mechanism_hunt(
    model_name: str = "mistralai/Mistral-7B-v0.1",
    device: str = "cuda",
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Hunt for the causal mechanism of R_V contraction.
    """
    print("=" * 80)
    print("CAUSAL MECHANISM HUNT")
    print(f"Model: {model_name}")
    print("=" * 80)
    
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"results/phase3_attention/runs/{timestamp}_causal_mechanism_hunt")
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
        "baseline": (
            "Explain the process of photosynthesis in plants. Describe how sunlight, water, "
            "and carbon dioxide are converted into glucose and oxygen."
        ),
    }
    
    # Track hidden state geometry across ALL layers
    print("\n[2/4] Tracking hidden state geometry across layers...")
    
    layers_to_track = list(range(0, 32, 2)) + [27, 31]  # Every 2nd layer plus key ones
    layers_to_track = sorted(set(layers_to_track))
    
    layer_geometry = {}
    
    for prompt_name, prompt in prompts.items():
        print(f"\n  {prompt_name}:")
        layer_geometry[prompt_name] = {}
        
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        
        with torch.no_grad():
            outputs = model(**enc, output_hidden_states=True, output_attentions=True)
        
        hidden_states = outputs.hidden_states
        
        for layer_idx in layers_to_track:
            if layer_idx < len(hidden_states):
                h = hidden_states[layer_idx]
                geom = compute_layer_geometry(h.to(device))
                layer_geometry[prompt_name][f"L{layer_idx}"] = geom
                print(f"    L{layer_idx}: PR = {geom['pr']:.4f}")
    
    # Track H31 entropy at different layers (where it exists)
    print("\n[3/4] Tracking H31 attention entropy...")
    
    h31_entropy = {}
    for prompt_name, prompt in prompts.items():
        h31_entropy[prompt_name] = {}
        print(f"\n  {prompt_name}:")
        
        for layer_idx in [25, 26, 27, 28, 29, 30, 31]:
            try:
                ent = compute_h31_entropy(model, tokenizer, prompt, layer_idx, device)
                h31_entropy[prompt_name][f"L{layer_idx}"] = ent
                print(f"    L{layer_idx} H31 entropy: {ent:.4f}")
            except Exception as e:
                print(f"    L{layer_idx}: Error - {e}")
    
    # Compute R_V-like ratios at different layer pairs
    print("\n[4/4] Computing R_V at different layer pairs...")
    
    rv_sweep = {}
    for prompt_name in prompts.keys():
        rv_sweep[prompt_name] = {}
        print(f"\n  {prompt_name}:")
        
        early_layer = "L5"
        if early_layer in layer_geometry[prompt_name]:
            pr_early = layer_geometry[prompt_name][early_layer]["pr"]
            
            for late_layer in ["L20", "L24", "L26", "L27", "L28", "L30"]:
                if late_layer in layer_geometry[prompt_name]:
                    pr_late = layer_geometry[prompt_name][late_layer]["pr"]
                    rv = pr_late / pr_early if pr_early > 0 else np.nan
                    rv_sweep[prompt_name][f"RV_{late_layer}"] = rv
                    print(f"    R_V ({late_layer}/L5): {rv:.4f}")
    
    # Generate outputs
    print("\n" + "=" * 80)
    print("GENERATING OUTPUTS")
    print("=" * 80)
    
    # 1. PR trajectory plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, (prompt_name, geom_data) in enumerate(layer_geometry.items()):
        ax = axes[idx]
        
        layers = []
        prs = []
        for layer_key, metrics in sorted(geom_data.items(), key=lambda x: int(x[0][1:])):
            layers.append(int(layer_key[1:]))
            prs.append(metrics["pr"])
        
        ax.plot(layers, prs, 'o-', linewidth=2, markersize=8)
        ax.axvline(x=27, color='red', linestyle='--', alpha=0.5, label='L27')
        ax.set_xlabel("Layer")
        ax.set_ylabel("Participation Ratio (PR)")
        ax.set_title(f"PR Trajectory: {prompt_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "pr_trajectory.png", dpi=150)
    plt.close()
    print(f"  Saved: pr_trajectory.png")
    
    # 2. R_V emergence plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for prompt_name, rv_data in rv_sweep.items():
        layers = []
        rvs = []
        for key, rv in sorted(rv_data.items()):
            layer = int(key.split("_L")[1])
            layers.append(layer)
            rvs.append(rv)
        
        color = 'red' if prompt_name == 'champion' else 'blue'
        ax.plot(layers, rvs, 'o-', label=prompt_name, color=color, linewidth=2, markersize=10)
    
    ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.3)
    ax.axvline(x=27, color='green', linestyle='--', alpha=0.5, label='L27')
    ax.set_xlabel("Late Layer (early=L5)")
    ax.set_ylabel("R_V = PR_late / PR_early")
    ax.set_title("When Does R_V Contraction Emerge?")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "rv_emergence.png", dpi=150)
    plt.close()
    print(f"  Saved: rv_emergence.png")
    
    # 3. Summary JSON
    summary = {
        "experiment": "causal_mechanism_hunt",
        "model_name": model_name,
        "layer_geometry": {k: {kk: {kkk: float(vvv) for kkk, vvv in vv.items()} for kk, vv in v.items()} for k, v in layer_geometry.items()},
        "h31_entropy": h31_entropy,
        "rv_sweep": rv_sweep,
    }
    
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"  Saved: summary.json")
    
    # Print key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    
    # Find where contraction starts
    print("\nPR trajectory (champion):")
    for layer_key, metrics in sorted(layer_geometry["champion"].items(), key=lambda x: int(x[0][1:])):
        layer = int(layer_key[1:])
        pr = metrics["pr"]
        change = ""
        prev_key = f"L{layer-2}"
        if prev_key in layer_geometry["champion"]:
            prev_pr = layer_geometry["champion"][prev_key]["pr"]
            delta = (pr - prev_pr) / prev_pr * 100 if prev_pr > 0 else 0
            if delta < -5:
                change = f" ⬇️ ({delta:.1f}%)"
            elif delta > 5:
                change = f" ⬆️ ({delta:.1f}%)"
        print(f"  L{layer}: PR = {pr:.4f}{change}")
    
    # H31 entropy comparison
    print("\nH31 entropy comparison (champion vs baseline):")
    for layer in [25, 26, 27, 28, 29, 30, 31]:
        layer_key = f"L{layer}"
        if layer_key in h31_entropy["champion"] and layer_key in h31_entropy["baseline"]:
            champ = h31_entropy["champion"][layer_key]
            base = h31_entropy["baseline"][layer_key]
            delta = champ - base
            print(f"  L{layer}: champion={champ:.3f}, baseline={base:.3f}, Δ={delta:+.3f}")
    
    print("\n" + "=" * 80)
    print(f"[OK] Hunt complete. Run dir: {output_dir}")
    print("=" * 80)
    
    return summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Causal Mechanism Hunt")
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1", help="Model name")
    parser.add_argument("--device", default="cuda", help="Device")
    args = parser.parse_args()
    
    try:
        run_causal_mechanism_hunt(model_name=args.model, device=args.device)
    finally:
        print("\n[cleanup] Clearing GPU memory...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print("[cleanup] GPU memory cleared.")

