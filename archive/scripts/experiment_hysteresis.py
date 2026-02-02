#!/usr/bin/env python3
"""
EXPERIMENT 3: Hysteresis / One-Way Door Test
=============================================

Tests if recursive state shows hysteresis (irreversibility), which is required
to justify "phase transition" language.

Measures recovery percentage when patching:
- Forward: baseline residual → recursive prompt (should work)
- Reverse: recursive residual → baseline prompt (should fail)

N=200 pairs for statistical power.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from scipy.stats import ttest_rel

sys.path.insert(0, str(Path(__file__).parent))

from src.core.models import load_model, set_seed
from src.core.hooks import capture_v_projection
from src.metrics.rv import participation_ratio
from prompts.loader import PromptLoader

# =============================================================================
# CONFIGURATION
# =============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
EARLY_LAYER = 5
LATE_LAYER = 27
WINDOW = 16
SEED = 42
N_PAIRS = 200
TEST_LAYERS = [24, 26, 28, 30, 31]  # Late layers for patching

# =============================================================================
# RESIDUAL EXTRACTION & PATCHING
# =============================================================================

def extract_residual(
    model,
    tokenizer,
    prompt: str,
    layer_idx: int,
    device: str = "cuda",
) -> Optional[torch.Tensor]:
    """
    Extract residual stream OUTPUT at a specific layer (after attention + MLP).
    
    Returns:
        Residual tensor of shape (seq_len, hidden_dim) or None if failed
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Hook to capture residual OUTPUT (after layer processing)
    storage = {"residual": None}
    
    def hook_fn(module, inp, out):
        # Output is the residual stream after this layer
        if isinstance(out, tuple):
            storage["residual"] = out[0].detach().clone()
        else:
            storage["residual"] = out.detach().clone() if isinstance(out, torch.Tensor) else None
        return out
    
    layer = model.model.layers[layer_idx]
    handle = layer.register_forward_hook(hook_fn)
    
    try:
        with torch.no_grad():
            _ = model(**inputs)
        residual = storage["residual"]
        # Return as (seq_len, hidden_dim) for consistency
        if residual is not None and residual.dim() == 3:
            residual = residual[0]  # Remove batch dimension
    except Exception as e:
        print(f"  ⚠️  Error extracting residual at L{layer_idx}: {e}")
        residual = None
    finally:
        handle.remove()
    
    return residual


def patch_residual_and_compute_rv(
    model,
    tokenizer,
    target_prompt: str,
    source_residual: torch.Tensor,
    layer_idx: int,
    device: str = "cuda",
) -> float:
    """
    Patch residual from source into target prompt and compute R_V.
    
    Uses forward_pre_hook to patch the INPUT to the layer (before processing).
    
    Returns:
        R_V value after patching
    """
    target_inputs = tokenizer(target_prompt, return_tensors="pt").to(device)
    
    # Use PRE-hook to patch the INPUT to the layer (following experiment_one_way_door.py pattern)
    def pre_hook_fn(module, args):
        # args is a tuple, first element is hidden_states (residual stream input)
        if not isinstance(args, tuple) or len(args) == 0:
            return args
        
        hidden_states = args[0]  # (batch, seq_len, hidden_dim)
        
        # Ensure source_residual has correct shape
        if source_residual.dim() == 2:
            # (seq_len, hidden_dim) -> need to add batch dim
            source_residual_batched = source_residual.unsqueeze(0)  # (1, seq_len, hidden_dim)
        else:
            source_residual_batched = source_residual
        
        B, T, D = hidden_states.shape
        T_src = source_residual_batched.shape[1] if source_residual_batched.dim() == 3 else source_residual_batched.shape[0]
        D_src = source_residual_batched.shape[2] if source_residual_batched.dim() == 3 else source_residual_batched.shape[1]
        
        if D_src != D:
            # Dimension mismatch - skip patching
            return args
        
        # Use window approach: patch last W tokens (like experiment_one_way_door.py)
        W = min(WINDOW, T, T_src)
        if W <= 0:
            return args
        
        # Extract last W tokens from source
        if source_residual_batched.dim() == 3:
            patch_tensor = source_residual_batched[:, -W:, :]  # (1, W, D)
        else:
            patch_tensor = source_residual_batched[-W:, :].unsqueeze(0)  # (1, W, D)
        
        # Clone and patch
        patched_hidden = hidden_states.clone()
        patch_tensor = patch_tensor.to(hidden_states.device, dtype=hidden_states.dtype)
        patched_hidden[:, -W:, :] = patch_tensor.expand(B, -1, -1)
        
        # Return patched input tuple
        return (patched_hidden,) + args[1:]
    
    layer = model.model.layers[layer_idx]
    handle = layer.register_forward_pre_hook(pre_hook_fn)
    
    try:
        # Capture V-projections
        with capture_v_projection(model, EARLY_LAYER) as v_early_storage:
            with capture_v_projection(model, LATE_LAYER) as v_late_storage:
                with torch.no_grad():
                    try:
                        _ = model(**target_inputs)
                    except Exception as e:
                        print(f"  ⚠️  Error in forward pass during patching: {e}")
                        return float('nan')
        
        v_early = v_early_storage.get("v")
        v_late = v_late_storage.get("v")
        
        if v_early is None or v_late is None:
            return float('nan')
        
        # Normalize to 2D for participation_ratio
        if v_early.dim() == 3:
            v_early = v_early[0]
        if v_late.dim() == 3:
            v_late = v_late[0]
        
        pr_early = participation_ratio(v_early, window_size=WINDOW)
        pr_late = participation_ratio(v_late, window_size=WINDOW)
        
        if pr_early == 0 or np.isnan(pr_early) or np.isnan(pr_late):
            return float('nan')
        
        return float(pr_late / pr_early)
    
    finally:
        handle.remove()


def compute_recovery_percentage(
    rv_baseline: float,
    rv_recursive: float,
    rv_patched: float,
) -> float:
    """
    Compute recovery percentage.
    
    Recovery % = (RV_patched - RV_recursive) / (RV_baseline - RV_recursive) × 100
    
    Interpretation:
    - 100% = Full recovery (can push into/out of recursive state)
    - 0% = No recovery (irreversible)
    """
    if np.isnan(rv_baseline) or np.isnan(rv_recursive) or np.isnan(rv_patched):
        return float('nan')
    
    gap = rv_baseline - rv_recursive
    if abs(gap) < 1e-6:
        return float('nan')  # No gap to recover
    
    recovery = (rv_patched - rv_recursive) / gap * 100.0
    return recovery


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    print("=" * 80)
    print("EXPERIMENT 3: HYSTERESIS / ONE-WAY DOOR TEST")
    print("=" * 80)
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")
    print(f"N pairs: {N_PAIRS}")
    print(f"Test layers: {TEST_LAYERS}")
    print("=" * 80)
    
    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"results/path_b_validation/runs/{timestamp}_hysteresis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    set_seed(SEED)
    model, tokenizer = load_model(MODEL_NAME, device=DEVICE, attn_implementation="eager")
    model.eval()
    
    # Load prompt pairs
    loader = PromptLoader()
    recursive_prompts = loader.get_by_pillar("dose_response", limit=N_PAIRS, seed=SEED)
    # Try multiple baseline pillar names
    baseline_prompts = []
    for pillar_name in ["baseline", "baselines", "control"]:
        baseline_prompts = loader.get_by_pillar(pillar_name, limit=N_PAIRS, seed=SEED)
        if len(baseline_prompts) > 0:
            break
    # If still empty, use get_by_type
    if len(baseline_prompts) == 0:
        baseline_prompts = loader.get_by_type("baseline", limit=N_PAIRS, seed=SEED)
    
    print(f"\nLoaded {len(recursive_prompts)} recursive and {len(baseline_prompts)} baseline prompts")
    
    results = []
    
    for pair_idx in tqdm(range(min(len(recursive_prompts), len(baseline_prompts))), desc="Processing pairs"):
        recursive_prompt = recursive_prompts[pair_idx]
        baseline_prompt = baseline_prompts[pair_idx]
        
        try:
            # Compute baseline R_V values
            # Baseline prompt R_V
            baseline_inputs = tokenizer(baseline_prompt, return_tensors="pt").to(DEVICE)
            with capture_v_projection(model, EARLY_LAYER) as v_early_storage:
                with capture_v_projection(model, LATE_LAYER) as v_late_storage:
                    with torch.no_grad():
                        _ = model(**baseline_inputs)
            v_early_base = v_early_storage.get("v")
            v_late_base = v_late_storage.get("v")
            rv_baseline = float('nan')
            if v_early_base is not None and v_late_base is not None:
                pr_early = participation_ratio(v_early_base, window_size=WINDOW)
                pr_late = participation_ratio(v_late_base, window_size=WINDOW)
                if pr_early > 0 and not np.isnan(pr_early) and not np.isnan(pr_late):
                    rv_baseline = float(pr_late / pr_early)
            
            # Recursive prompt R_V
            recursive_inputs = tokenizer(recursive_prompt, return_tensors="pt").to(DEVICE)
            with capture_v_projection(model, EARLY_LAYER) as v_early_storage:
                with capture_v_projection(model, LATE_LAYER) as v_late_storage:
                    with torch.no_grad():
                        _ = model(**recursive_inputs)
            v_early_rec = v_early_storage.get("v")
            v_late_rec = v_late_storage.get("v")
            rv_recursive = float('nan')
            if v_early_rec is not None and v_late_rec is not None:
                pr_early = participation_ratio(v_early_rec, window_size=WINDOW)
                pr_late = participation_ratio(v_late_rec, window_size=WINDOW)
                if pr_early > 0 and not np.isnan(pr_early) and not np.isnan(pr_late):
                    rv_recursive = float(pr_late / pr_early)
            
            # Test each layer
            for layer_idx in TEST_LAYERS:
                # Extract residuals
                baseline_residual = extract_residual(model, tokenizer, baseline_prompt, layer_idx, DEVICE)
                recursive_residual = extract_residual(model, tokenizer, recursive_prompt, layer_idx, DEVICE)
                
                # Skip if residual extraction failed
                if baseline_residual is None or recursive_residual is None:
                    continue
                
                # Forward: baseline residual → recursive prompt
                rv_forward = patch_residual_and_compute_rv(
                    model, tokenizer, recursive_prompt, baseline_residual, layer_idx, DEVICE
                )
                recovery_forward = compute_recovery_percentage(rv_baseline, rv_recursive, rv_forward)
                
                # Reverse: recursive residual → baseline prompt
                rv_reverse = patch_residual_and_compute_rv(
                    model, tokenizer, baseline_prompt, recursive_residual, layer_idx, DEVICE
                )
                recovery_reverse = compute_recovery_percentage(rv_baseline, rv_recursive, rv_reverse)
                
                results.append({
                    "pair_id": pair_idx,
                    "baseline_prompt": baseline_prompt[:100],
                    "recursive_prompt": recursive_prompt[:100],
                    "layer": layer_idx,
                    "rv_baseline": rv_baseline,
                    "rv_recursive": rv_recursive,
                    "rv_forward": rv_forward,
                    "rv_reverse": rv_reverse,
                    "recovery_forward": recovery_forward,
                    "recovery_reverse": recovery_reverse,
                    "asymmetry": recovery_forward - recovery_reverse if not (np.isnan(recovery_forward) or np.isnan(recovery_reverse)) else float('nan'),
                })
        
        except Exception as e:
            print(f"\nError processing pair {pair_idx}: {e}")
            continue
    
    # Save results
    if len(results) == 0:
        print("\n⚠️  WARNING: No results collected. Check errors above.")
        return
    
    df = pd.DataFrame(results)
    df.to_csv(output_dir / "results.csv", index=False)
    
    # Summary statistics
    summary = {
        "experiment": "hysteresis",
        "model": MODEL_NAME,
        "n_pairs": N_PAIRS,
        "test_layers": TEST_LAYERS,
        "results": {},
    }
    
    # Aggregate by layer
    for layer_idx in TEST_LAYERS:
        layer_df = df[df["layer"] == layer_idx] if len(df) > 0 else pd.DataFrame()
        
        forward_recoveries = layer_df["recovery_forward"].dropna()
        reverse_recoveries = layer_df["recovery_reverse"].dropna()
        
        if len(forward_recoveries) > 0 and len(reverse_recoveries) > 0:
            # Paired t-test
            paired_data = layer_df[["recovery_forward", "recovery_reverse"]].dropna()
            if len(paired_data) > 1:
                t_stat, p_value = ttest_rel(paired_data["recovery_forward"], paired_data["recovery_reverse"])
            else:
                t_stat, p_value = float('nan'), float('nan')
            
            summary["results"][f"L{layer_idx}"] = {
                "mean_recovery_forward": float(forward_recoveries.mean()),
                "std_recovery_forward": float(forward_recoveries.std()),
                "mean_recovery_reverse": float(reverse_recoveries.mean()),
                "std_recovery_reverse": float(reverse_recoveries.std()),
                "asymmetry": float(forward_recoveries.mean() - reverse_recoveries.mean()),
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "n_valid": len(paired_data),
            }
    
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")
    for layer_idx in TEST_LAYERS:
        layer_key = f"L{layer_idx}"
        if layer_key in summary["results"]:
            stats = summary["results"][layer_key]
            print(f"\nLayer {layer_idx}:")
            print(f"  Forward recovery: {stats['mean_recovery_forward']:.1f}% ± {stats['std_recovery_forward']:.1f}%")
            print(f"  Reverse recovery: {stats['mean_recovery_reverse']:.1f}% ± {stats['std_recovery_reverse']:.1f}%")
            print(f"  Asymmetry: {stats['asymmetry']:.1f}%")
            print(f"  t-test: t={stats['t_statistic']:.3f}, p={stats['p_value']:.4f}")
            if stats['p_value'] < 0.05:
                print(f"  ✅ SIGNIFICANT ASYMMETRY (hysteresis confirmed)")
            else:
                print(f"  ❌ No significant asymmetry")
    
    print(f"\n✅ Results saved to: {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

