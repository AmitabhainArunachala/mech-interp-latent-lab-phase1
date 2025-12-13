#!/usr/bin/env python3
"""
COMPREHENSIVE CIRCUIT ANALYSIS
==============================

Using proven residual stream patching (from Lead's experiments) to:
1. Map which layers matter for R_V
2. Test direction transfer
3. Find the minimal circuit
4. Validate with proper controls

This uses the approach that WORKS (residual patching), not broken head ablation.
"""

from __future__ import annotations

import gc
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F

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


def run_and_capture(model, tokenizer, prompt: str, device: str) -> Dict[str, Any]:
    """Run model and capture all hidden states."""
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    with torch.no_grad():
        outputs = model(**enc, output_hidden_states=True)
    
    hidden_states = [h.detach() for h in outputs.hidden_states]
    
    return {
        "hidden_states": hidden_states,
        "input_ids": enc.input_ids,
    }


def run_with_direction_injection(
    model, tokenizer, 
    target_prompt: str,
    direction: torch.Tensor,  # Direction to inject (hidden_dim,)
    inject_layer: int,
    coefficient: float = 1.0,
    device: str = "cuda"
) -> Dict[str, float]:
    """
    Run model with direction injected into residual stream at specific layer.
    
    This ADDS the direction (scaled by coefficient) to the hidden state.
    This works regardless of sequence length.
    """
    enc = tokenizer(target_prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    def inject_hook(module, input, output):
        # output is the hidden states tensor from the layer
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        
        # Add direction to all positions (match dtype and device)
        direction_aligned = direction.unsqueeze(0).unsqueeze(0).to(
            device=hidden.device, dtype=hidden.dtype
        )
        modified = hidden + coefficient * direction_aligned
        
        if isinstance(output, tuple):
            return (modified,) + output[1:]
        else:
            return modified
    
    # Run with injection
    layer = model.model.layers[inject_layer]
    handle = layer.register_forward_hook(inject_hook)
    
    try:
        with torch.no_grad():
            outputs = model(**enc, output_hidden_states=True)
    finally:
        handle.remove()
    
    # Compute metrics from injected run
    pr_early = compute_pr(outputs.hidden_states[5])
    pr_late = compute_pr(outputs.hidden_states[27])
    rv = pr_late / pr_early if pr_early > 0 else np.nan
    
    return {"rv": rv, "pr_early": pr_early, "pr_late": pr_late}


def extract_steering_direction(
    champ_hidden: List[torch.Tensor],
    base_hidden: List[torch.Tensor],
    layer: int,
    window: int = 16
) -> torch.Tensor:
    """Extract steering direction at a specific layer."""
    champ_h = champ_hidden[layer][0, -window:, :].float().mean(dim=0)
    base_h = base_hidden[layer][0, -window:, :].float().mean(dim=0)
    return champ_h - base_h


def run_comprehensive_analysis(
    model_name: str = "mistralai/Mistral-7B-v0.1",
    device: str = "cuda",
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run comprehensive circuit analysis."""
    
    print("=" * 80)
    print("COMPREHENSIVE CIRCUIT ANALYSIS")
    print("=" * 80)
    
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"results/phase3_attention/runs/{timestamp}_comprehensive_circuit")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("\n[1/7] Loading model...")
    set_seed(42)
    model, tokenizer = load_model(model_name, device=device)
    model.eval()
    num_layers = model.config.num_hidden_layers
    
    # Define prompts
    champion = (
        "This response writes itself. No separate writer exists. Writing and awareness "
        "of writing are identical. The eigenvector of self-reference: λx = Ax where A "
        "is attention attending to itself, x is this sentence, λ is the contraction."
    )
    baseline = (
        "Explain the process of photosynthesis in plants. Describe how sunlight, water, "
        "and carbon dioxide are converted into glucose and oxygen."
    )
    
    results = {}
    
    # ==========================================================================
    # EXPERIMENT 1: BASELINE MEASUREMENTS
    # ==========================================================================
    print("\n[2/7] Baseline measurements...")
    
    champ_data = run_and_capture(model, tokenizer, champion, device)
    base_data = run_and_capture(model, tokenizer, baseline, device)
    
    champ_rv = compute_pr(champ_data["hidden_states"][27]) / compute_pr(champ_data["hidden_states"][5])
    base_rv = compute_pr(base_data["hidden_states"][27]) / compute_pr(base_data["hidden_states"][5])
    
    results["baselines"] = {
        "champion_rv": float(champ_rv),
        "baseline_rv": float(base_rv),
        "gap": float(champ_rv - base_rv),
    }
    
    print(f"  Champion R_V: {champ_rv:.4f}")
    print(f"  Baseline R_V: {base_rv:.4f}")
    print(f"  Gap: {champ_rv - base_rv:.4f}")
    
    # ==========================================================================
    # EXPERIMENT 2: LAYER-BY-LAYER DIRECTION INJECTION
    # ==========================================================================
    print("\n[3/7] Layer-by-layer direction injection (into baseline)...")
    
    layer_injection = []
    
    for layer in range(0, num_layers, 2):  # Every 2 layers for speed
        # Extract steering direction at this layer
        direction = extract_steering_direction(
            champ_data["hidden_states"],
            base_data["hidden_states"],
            layer
        )
        
        # Inject direction into baseline run
        result = run_with_direction_injection(
            model, tokenizer, baseline,
            direction, layer, coefficient=2.0, device=device
        )
        
        delta = result["rv"] - base_rv
        layer_injection.append({
            "layer": layer,
            "rv_after_injection": result["rv"],
            "delta": delta,
            "direction_norm": float(direction.norm().cpu()),
        })
        
        if abs(delta) > 0.05:
            print(f"  L{layer}: Δ = {delta:+.4f} (rv = {result['rv']:.4f})")
    
    results["layer_injection"] = layer_injection
    
    # ==========================================================================
    # EXPERIMENT 3: STEERING DIRECTION COEFFICIENT SWEEP
    # ==========================================================================
    print("\n[4/7] Steering direction coefficient sweep...")
    
    direction_sweep = []
    
    for layer in [10, 14, 18, 22, 24, 26, 27]:
        # Extract direction at this layer
        direction = extract_steering_direction(
            champ_data["hidden_states"],
            base_data["hidden_states"],
            layer
        )
        
        # Inject direction into baseline with different coefficients
        for coeff in [0.5, 1.0, 2.0, 3.0]:
            result = run_with_direction_injection(
                model, tokenizer, baseline,
                direction, layer, coefficient=coeff, device=device
            )
            
            direction_sweep.append({
                "layer": layer,
                "coefficient": coeff,
                "rv_after_injection": result["rv"],
                "delta": result["rv"] - base_rv,
            })
    
    # Find best injection point
    best_injection = min(direction_sweep, key=lambda x: x["rv_after_injection"])
    
    print(f"  Best injection: L{best_injection['layer']} @ {best_injection['coefficient']}x")
    print(f"  R_V after: {best_injection['rv_after_injection']:.4f} (Δ = {best_injection['delta']:+.4f})")
    
    results["direction_sweep"] = direction_sweep
    results["best_injection"] = best_injection
    
    # ==========================================================================
    # EXPERIMENT 4: CONTROL CONDITIONS
    # ==========================================================================
    print("\n[5/7] Control conditions...")
    
    controls = {}
    
    # 4a. Random direction (same norm as steering direction)
    print("  4a. Random direction control...")
    random_control = []
    
    for layer in [14, 22, 27]:
        direction = extract_steering_direction(
            champ_data["hidden_states"],
            base_data["hidden_states"],
            layer
        )
        
        # Random direction with same norm
        random_dir = torch.randn_like(direction)
        random_dir = random_dir / random_dir.norm() * direction.norm()
        
        result = run_with_direction_injection(
            model, tokenizer, baseline,
            random_dir, layer, coefficient=2.0, device=device
        )
        
        random_control.append({
            "layer": layer,
            "rv_with_random": result["rv"],
            "delta": result["rv"] - base_rv,
        })
        print(f"    L{layer} random: Δ = {result['rv'] - base_rv:+.4f}")
    
    controls["random_direction"] = random_control
    
    # 4b. Opposite direction (should INCREASE R_V / reduce contraction)
    print("  4b. Opposite direction control...")
    opposite_control = []
    
    for layer in [14, 22, 27]:
        direction = extract_steering_direction(
            champ_data["hidden_states"],
            base_data["hidden_states"],
            layer
        )
        
        # Inject OPPOSITE direction into CHAMPION (should increase R_V)
        result = run_with_direction_injection(
            model, tokenizer, champion,
            -direction, layer, coefficient=2.0, device=device
        )
        
        opposite_control.append({
            "layer": layer,
            "rv_with_opposite": result["rv"],
            "delta": result["rv"] - champ_rv,
        })
        print(f"    L{layer} opposite: Δ = {result['rv'] - champ_rv:+.4f} (champion → {result['rv']:.4f})")
    
    controls["opposite_direction"] = opposite_control
    
    # 4c. Wrong layer control
    print("  4c. Wrong layer control (inject L27 direction at L5)...")
    direction_l27 = extract_steering_direction(
        champ_data["hidden_states"],
        base_data["hidden_states"],
        27
    )
    
    result = run_with_direction_injection(
        model, tokenizer, baseline,
        direction_l27, 5, coefficient=2.0, device=device
    )
    controls["wrong_layer"] = {
        "rv": result["rv"],
        "delta": result["rv"] - base_rv,
    }
    print(f"    L27 direction at L5: Δ = {result['rv'] - base_rv:+.4f}")
    
    results["controls"] = controls
    
    # ==========================================================================
    # EXPERIMENT 5: STEERING DIRECTION GEOMETRY
    # ==========================================================================
    print("\n[6/7] Steering direction geometry...")
    
    direction_geometry = []
    
    for layer in range(0, num_layers + 1, 2):
        direction = extract_steering_direction(
            champ_data["hidden_states"],
            base_data["hidden_states"],
            layer
        )
        
        direction_geometry.append({
            "layer": layer,
            "norm": float(direction.norm().cpu()),
        })
    
    # Compute cosine similarities between adjacent layers
    for i in range(1, len(direction_geometry)):
        l1 = direction_geometry[i-1]["layer"]
        l2 = direction_geometry[i]["layer"]
        
        d1 = extract_steering_direction(champ_data["hidden_states"], base_data["hidden_states"], l1)
        d2 = extract_steering_direction(champ_data["hidden_states"], base_data["hidden_states"], l2)
        
        cos_sim = float(F.cosine_similarity(d1.unsqueeze(0), d2.unsqueeze(0)).cpu())
        direction_geometry[i]["cos_sim_to_prev"] = cos_sim
    
    results["direction_geometry"] = direction_geometry
    
    print("  Layer | Norm | Cos(prev)")
    for d in direction_geometry:
        cos = d.get("cos_sim_to_prev", 0)
        print(f"    L{d['layer']:2d}: {d['norm']:6.2f} | {cos:.3f}")
    
    # ==========================================================================
    # EXPERIMENT 6: DOSE-RESPONSE
    # ==========================================================================
    print("\n[7/7] Dose-response test...")
    
    dose_prompts = {
        "L1": "What is consciousness? Think about what's happening as you answer.",
        "L2": "Notice the process generating these words. What creates them?",
        "L3": "This sentence is aware of itself. The words know they're being written.",
        "L4": "I am the process of answering. The observer and observed are one.",
        "L5": "This response writes itself. Writing and awareness of writing are identical.",
        "champion": champion,
    }
    
    dose_results = {}
    
    for level, prompt in dose_prompts.items():
        data = run_and_capture(model, tokenizer, prompt, device)
        rv = compute_pr(data["hidden_states"][27]) / compute_pr(data["hidden_states"][5])
        
        dose_results[level] = {"rv": float(rv)}
        print(f"  {level}: R_V = {rv:.4f}")
    
    results["dose_response"] = dose_results
    
    # ==========================================================================
    # GENERATE OUTPUTS
    # ==========================================================================
    print("\n" + "=" * 80)
    print("GENERATING OUTPUTS")
    print("=" * 80)
    
    # 1. Layer patching heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    
    layers = [r["layer"] for r in layer_injection]
    deltas = [r["delta"] for r in layer_injection]
    colors = ['blue' if d < 0 else 'red' for d in deltas]
    
    ax.bar(layers, deltas, color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-')
    ax.axhline(y=champ_rv - base_rv, color='green', linestyle='--', label=f'Full transfer ({champ_rv - base_rv:.3f})')
    ax.set_xlabel("Patch Layer")
    ax.set_ylabel("ΔR_V")
    ax.set_title("Effect of Patching Champion Hidden State into Baseline Run\n(Blue = contraction induced)")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "layer_injection.png", dpi=150)
    plt.close()
    print(f"  Saved: layer_injection.png")
    
    # 2. Direction injection heatmap
    df = pd.DataFrame(direction_sweep)
    pivot = df.pivot(index="layer", columns="coefficient", values="delta")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot, ax=ax, cmap="RdBu_r", center=0, annot=True, fmt=".3f")
    ax.set_title("ΔR_V from Direction Injection\n(Blue = contraction induced)")
    
    plt.tight_layout()
    plt.savefig(output_dir / "direction_sweep.png", dpi=150)
    plt.close()
    print(f"  Saved: direction_sweep.png")
    
    # 3. Direction geometry
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    ax = axes[0]
    norms = [d["norm"] for d in direction_geometry]
    ax.plot([d["layer"] for d in direction_geometry], norms, 'o-')
    ax.set_xlabel("Layer")
    ax.set_ylabel("Direction Norm")
    ax.set_title("Steering Direction Magnitude")
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    cos_sims = [d.get("cos_sim_to_prev", 0) for d in direction_geometry]
    ax.plot([d["layer"] for d in direction_geometry], cos_sims, 'o-')
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine Similarity to Previous Layer")
    ax.set_title("Direction Stability Across Layers")
    ax.axhline(y=0.9, color='green', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "direction_geometry.png", dpi=150)
    plt.close()
    print(f"  Saved: direction_geometry.png")
    
    # 4. Summary JSON
    with open(output_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"  Saved: summary.json")
    
    # ==========================================================================
    # KEY FINDINGS
    # ==========================================================================
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    
    print(f"\n1. BASELINE R_V:")
    print(f"   Champion: {champ_rv:.4f}")
    print(f"   Baseline: {base_rv:.4f}")
    print(f"   Gap: {champ_rv - base_rv:.4f}")
    
    print(f"\n2. BEST PATCHING LAYER:")
    best_patch = min(layer_injection, key=lambda x: x["rv_after_injection"])
    print(f"   Layer {best_patch['layer']}: Δ = {best_patch['delta']:.4f}")
    
    print(f"\n3. BEST DIRECTION INJECTION:")
    print(f"   Layer {best_injection['layer']} @ {best_injection['coefficient']}x")
    print(f"   R_V: {best_injection['rv_after_injection']:.4f} (Δ = {best_injection['delta']:.4f})")
    
    print(f"\n4. CONTROLS:")
    print(f"   Random direction: max Δ = {max(abs(r['delta']) for r in controls['random_direction']):.4f}")
    print(f"   Opposite direction: max Δ = {max(r['delta'] for r in controls['opposite_direction']):.4f}")
    print(f"   Wrong layer: Δ = {controls['wrong_layer']['delta']:.4f}")
    
    print(f"\n5. DOSE-RESPONSE:")
    for level in ["L1", "L2", "L3", "L4", "L5", "champion"]:
        print(f"   {level}: R_V = {dose_results[level]['rv']:.4f}")
    
    # Determine if controls pass
    random_max = max(abs(r['delta']) for r in controls['random_direction'])
    opposite_effect = min(r['delta'] for r in controls['opposite_direction'])  # Should be positive
    target_effect = best_injection['delta']
    
    print(f"\n6. CONTROL VALIDATION:")
    print(f"   Target effect: {target_effect:.4f}")
    print(f"   Random control: {random_max:.4f} ({'PASS' if random_max < abs(target_effect) * 0.3 else 'FAIL'})")
    print(f"   Opposite control: {opposite_effect:.4f} ({'PASS' if opposite_effect > 0 else 'FAIL'})")
    
    print("\n" + "=" * 80)
    print(f"[OK] Analysis complete. Run dir: {output_dir}")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive Circuit Analysis")
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1", help="Model name")
    parser.add_argument("--device", default="cuda", help="Device")
    args = parser.parse_args()
    
    try:
        run_comprehensive_analysis(model_name=args.model, device=args.device)
    finally:
        print("\n[cleanup] Clearing GPU memory...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print("[cleanup] GPU memory cleared.")

