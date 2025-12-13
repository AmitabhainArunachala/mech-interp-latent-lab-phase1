#!/usr/bin/env python3
"""
UNIFIED LAYER MAP: All Metrics Across All Layers

Track simultaneously:
1. PR (Participation Ratio) - geometry of hidden states
2. R_V at each layer (vs L5 baseline) - contraction metric
3. Steering Vector Cosine Similarity - where does the direction live?
4. H31 Entropy - attention focus
5. H31 BOS Attention - global register usage
6. Mean Hidden State Norm - activation magnitude

This gives ONE picture showing where everything happens.
"""

from __future__ import annotations

import gc
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import entropy as scipy_entropy

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.models import load_model, set_seed


def compute_pr(hidden_states: torch.Tensor, window: int = 16) -> float:
    """Compute Participation Ratio."""
    if hidden_states.dim() == 3:
        hidden_states = hidden_states[0]
    T, D = hidden_states.shape
    W = min(window, T)
    h_window = hidden_states[-W:, :].float()
    
    try:
        U, S, Vt = torch.linalg.svd(h_window.T, full_matrices=False)
        S_np = S.cpu().numpy()
        S_sq = S_np ** 2
        if S_sq.sum() < 1e-10:
            return np.nan
        return float((S_sq.sum()**2) / (S_sq**2).sum())
    except:
        return np.nan


def compute_h31_metrics(attn_weights: torch.Tensor) -> Dict[str, float]:
    """Compute H31 entropy and BOS attention."""
    if attn_weights is None:
        return {"entropy": np.nan, "bos_attn": np.nan}
    
    h31_attn = attn_weights[0, 31, :, :].cpu().numpy()
    
    # Entropy
    entropies = []
    for i in range(h31_attn.shape[0]):
        row = h31_attn[i] + 1e-10
        row = row / row.sum()
        entropies.append(scipy_entropy(row))
    
    # BOS attention (last token attending to first)
    bos_attn = float(h31_attn[-1, 0])
    
    return {
        "entropy": float(np.mean(entropies)),
        "bos_attn": bos_attn,
    }


def run_unified_layer_map(
    model_name: str = "mistralai/Mistral-7B-v0.1",
    device: str = "cuda",
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Run unified multi-metric layer map.
    """
    print("=" * 80)
    print("UNIFIED LAYER MAP: All Metrics Across All Layers")
    print(f"Model: {model_name}")
    print("=" * 80)
    
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"results/phase3_attention/runs/{timestamp}_unified_layer_map")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("\n[1/4] Loading model...")
    set_seed(42)
    model, tokenizer = load_model(model_name, device=device, attn_implementation="eager")
    model.eval()
    num_layers = model.config.num_hidden_layers
    
    # Prompts
    champion = (
        "This response writes itself. No separate writer exists. Writing and awareness "
        "of writing are identical. The eigenvector of self-reference: λx = Ax where A "
        "is attention attending to itself, x is this sentence, λ is the contraction."
    )
    baseline = (
        "Explain the process of photosynthesis in plants. Describe how sunlight, water, "
        "and carbon dioxide are converted into glucose and oxygen."
    )
    
    # Run both prompts and collect hidden states + attentions
    print("\n[2/4] Running forward passes...")
    
    def get_all_outputs(prompt):
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True, output_attentions=True)
        return out.hidden_states, out.attentions, enc.input_ids.shape[1]
    
    champ_hidden, champ_attn, champ_len = get_all_outputs(champion)
    base_hidden, base_attn, base_len = get_all_outputs(baseline)
    
    # Compute steering vector at each layer (mean difference)
    print("\n[3/4] Computing all metrics across layers...")
    
    results = {
        "layers": list(range(num_layers + 1)),  # 0 to 32 (including embedding)
        "champion": {},
        "baseline": {},
        "comparison": {},
    }
    
    # Store metrics for each layer
    for layer_idx in range(num_layers + 1):
        print(f"  Layer {layer_idx}...", end=" ")
        
        # Get hidden states for this layer
        champ_h = champ_hidden[layer_idx]  # (1, seq, dim)
        base_h = base_hidden[layer_idx]
        
        # PR
        champ_pr = compute_pr(champ_h)
        base_pr = compute_pr(base_h)
        
        # R_V (vs layer 5)
        if layer_idx == 5:
            champ_pr_l5 = champ_pr
            base_pr_l5 = base_pr
        
        champ_rv = champ_pr / champ_pr_l5 if layer_idx >= 5 and champ_pr_l5 > 0 else np.nan
        base_rv = base_pr / base_pr_l5 if layer_idx >= 5 and base_pr_l5 > 0 else np.nan
        
        # Mean hidden state norm
        champ_norm = float(champ_h[0, -16:, :].float().norm(dim=-1).mean().cpu())
        base_norm = float(base_h[0, -16:, :].float().norm(dim=-1).mean().cpu())
        
        # Steering vector: mean difference
        # Use last 16 tokens, average across positions
        champ_mean = champ_h[0, -16:, :].float().mean(dim=0)  # (dim,)
        base_mean = base_h[0, -16:, :].float().mean(dim=0)
        steering_vec = champ_mean - base_mean
        steering_norm = float(steering_vec.norm().cpu())
        
        # Cosine similarity between steering vectors at adjacent layers
        if layer_idx > 0:
            prev_champ_mean = champ_hidden[layer_idx - 1][0, -16:, :].float().mean(dim=0)
            prev_base_mean = base_hidden[layer_idx - 1][0, -16:, :].float().mean(dim=0)
            prev_steering = prev_champ_mean - prev_base_mean
            
            cos_sim = float(torch.nn.functional.cosine_similarity(
                steering_vec.unsqueeze(0), prev_steering.unsqueeze(0)
            ).cpu())
        else:
            cos_sim = np.nan
        
        # H31 metrics (only for actual layers, not embedding)
        if layer_idx > 0 and layer_idx <= num_layers:
            champ_h31 = compute_h31_metrics(champ_attn[layer_idx - 1])
            base_h31 = compute_h31_metrics(base_attn[layer_idx - 1])
        else:
            champ_h31 = {"entropy": np.nan, "bos_attn": np.nan}
            base_h31 = {"entropy": np.nan, "bos_attn": np.nan}
        
        # Store
        results["champion"][layer_idx] = {
            "pr": champ_pr,
            "rv": champ_rv,
            "norm": champ_norm,
            "h31_entropy": champ_h31["entropy"],
            "h31_bos": champ_h31["bos_attn"],
        }
        results["baseline"][layer_idx] = {
            "pr": base_pr,
            "rv": base_rv,
            "norm": base_norm,
            "h31_entropy": base_h31["entropy"],
            "h31_bos": base_h31["bos_attn"],
        }
        results["comparison"][layer_idx] = {
            "steering_norm": steering_norm,
            "steering_cos_sim": cos_sim,
            "pr_gap": champ_pr - base_pr if not np.isnan(champ_pr) else np.nan,
            "rv_gap": champ_rv - base_rv if not np.isnan(champ_rv) else np.nan,
            "entropy_gap": champ_h31["entropy"] - base_h31["entropy"] if not np.isnan(champ_h31["entropy"]) else np.nan,
        }
        
        print(f"PR={champ_pr:.2f}/{base_pr:.2f}, steer_norm={steering_norm:.2f}")
    
    # Generate visualization
    print("\n[4/4] Generating unified visualization...")
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    layers = list(range(num_layers + 1))
    
    # 1. PR Trajectory
    ax = axes[0, 0]
    champ_prs = [results["champion"][l]["pr"] for l in layers]
    base_prs = [results["baseline"][l]["pr"] for l in layers]
    ax.plot(layers, champ_prs, 'r-o', label='Champion', markersize=4)
    ax.plot(layers, base_prs, 'b-o', label='Baseline', markersize=4)
    ax.axvline(x=27, color='green', linestyle='--', alpha=0.5, label='L27')
    ax.set_xlabel('Layer')
    ax.set_ylabel('PR')
    ax.set_title('Participation Ratio Trajectory')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. R_V Trajectory (vs L5)
    ax = axes[0, 1]
    champ_rvs = [results["champion"][l]["rv"] for l in layers]
    base_rvs = [results["baseline"][l]["rv"] for l in layers]
    ax.plot(layers[5:], champ_rvs[5:], 'r-o', label='Champion R_V', markersize=4)
    ax.plot(layers[5:], base_rvs[5:], 'b-o', label='Baseline R_V', markersize=4)
    ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.3)
    ax.axvline(x=27, color='green', linestyle='--', alpha=0.5, label='L27')
    ax.set_xlabel('Layer')
    ax.set_ylabel('R_V = PR(layer) / PR(L5)')
    ax.set_title('R_V Evolution (vs L5)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Steering Vector Norm
    ax = axes[1, 0]
    steer_norms = [results["comparison"][l]["steering_norm"] for l in layers]
    ax.plot(layers, steer_norms, 'purple', linewidth=2)
    ax.axvline(x=27, color='green', linestyle='--', alpha=0.5, label='L27')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Steering Vector Norm')
    ax.set_title('Where Does the Steering Direction Live?\n(Mean Difference: Champion - Baseline)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. H31 Entropy
    ax = axes[1, 1]
    champ_ent = [results["champion"][l]["h31_entropy"] for l in layers]
    base_ent = [results["baseline"][l]["h31_entropy"] for l in layers]
    ax.plot(layers[1:], champ_ent[1:], 'r-o', label='Champion H31', markersize=4)
    ax.plot(layers[1:], base_ent[1:], 'b-o', label='Baseline H31', markersize=4)
    ax.axvline(x=27, color='green', linestyle='--', alpha=0.5, label='L27')
    ax.set_xlabel('Layer')
    ax.set_ylabel('H31 Entropy')
    ax.set_title('H31 Attention Entropy\n(Lower = More Focused)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. PR Gap (Champion - Baseline)
    ax = axes[2, 0]
    pr_gaps = [results["comparison"][l]["pr_gap"] for l in layers]
    colors = ['red' if g < 0 else 'blue' for g in pr_gaps]
    ax.bar(layers, pr_gaps, color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-')
    ax.axvline(x=27, color='green', linestyle='--', alpha=0.5, label='L27')
    ax.set_xlabel('Layer')
    ax.set_ylabel('PR Gap (Champ - Base)')
    ax.set_title('PR Gap\n(Red = Champion more contracted)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. H31 Entropy Gap
    ax = axes[2, 1]
    ent_gaps = [results["comparison"][l]["entropy_gap"] for l in layers[1:]]
    colors = ['red' if g < 0 else 'blue' for g in ent_gaps]
    ax.bar(layers[1:], ent_gaps, color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-')
    ax.axvline(x=27, color='green', linestyle='--', alpha=0.5, label='L27')
    ax.set_xlabel('Layer')
    ax.set_ylabel('H31 Entropy Gap (Champ - Base)')
    ax.set_title('H31 Entropy Gap\n(Red = Champion more focused)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "unified_layer_map.png", dpi=150)
    plt.close()
    print(f"  Saved: unified_layer_map.png")
    
    # Summary table
    print("\n" + "=" * 100)
    print("UNIFIED METRICS TABLE")
    print("=" * 100)
    print(f"{'Layer':>5} | {'C_PR':>6} | {'B_PR':>6} | {'C_RV':>6} | {'B_RV':>6} | {'Steer':>6} | {'C_H31':>6} | {'B_H31':>6} | {'EntGap':>7}")
    print("-" * 100)
    
    for l in [0, 5, 10, 14, 18, 20, 22, 24, 25, 26, 27, 28, 30, 31, 32]:
        if l <= num_layers:
            c = results["champion"][l]
            b = results["baseline"][l]
            comp = results["comparison"][l]
            
            c_pr = f"{c['pr']:.2f}" if not np.isnan(c['pr']) else "  nan"
            b_pr = f"{b['pr']:.2f}" if not np.isnan(b['pr']) else "  nan"
            c_rv = f"{c['rv']:.3f}" if not np.isnan(c['rv']) else "  nan"
            b_rv = f"{b['rv']:.3f}" if not np.isnan(b['rv']) else "  nan"
            steer = f"{comp['steering_norm']:.1f}" if not np.isnan(comp['steering_norm']) else "  nan"
            c_h31 = f"{c['h31_entropy']:.2f}" if not np.isnan(c['h31_entropy']) else "  nan"
            b_h31 = f"{b['h31_entropy']:.2f}" if not np.isnan(b['h31_entropy']) else "  nan"
            ent_gap = f"{comp['entropy_gap']:+.3f}" if not np.isnan(comp['entropy_gap']) else "    nan"
            
            marker = " ⭐" if l == 27 else ""
            print(f"{l:>5} | {c_pr:>6} | {b_pr:>6} | {c_rv:>6} | {b_rv:>6} | {steer:>6} | {c_h31:>6} | {b_h31:>6} | {ent_gap:>7}{marker}")
    
    # Save JSON
    with open(output_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n  Saved: summary.json")
    
    print("\n" + "=" * 80)
    print(f"[OK] Unified map complete. Run dir: {output_dir}")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Unified Layer Map")
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1", help="Model name")
    parser.add_argument("--device", default="cuda", help="Device")
    args = parser.parse_args()
    
    try:
        run_unified_layer_map(model_name=args.model, device=args.device)
    finally:
        print("\n[cleanup] Clearing GPU memory...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print("[cleanup] GPU memory cleared.")

