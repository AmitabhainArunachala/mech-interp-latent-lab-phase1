"""
CIRCUIT HUNT V2 FOCUSED: Fast, Targeted Tests

This version focuses on the MOST PROMISING hypotheses:
1. Early-layer head ablation (L8-L23 ramp region)
2. Reverse patching (baseline → recursive to undo effect)
3. Mean ablation vs zeroing

Runs faster by:
- Sampling fewer prompts
- Testing fewer heads (but strategically)
- Focusing on ramp layers
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

sys.path.insert(0, os.path.abspath("."))

from src.core.models import load_model, set_seed
from src.metrics.rv import participation_ratio
from prompts.loader import PromptLoader

# =============================================================================
# CONFIGURATION
# =============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
WINDOW = 16
EARLY_LAYER = 5
MEASURE_LAYER = 27
NUM_LAYERS = 32
NUM_HEADS = 32
SEED = 42

# Focus on ramp layers where effect builds
RAMP_LAYERS = [10, 12, 15, 18, 20, 22]  # Key ramp layers
LATE_LAYERS = [24, 25, 27]

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def compute_rv(v_early: torch.Tensor, v_late: torch.Tensor, window: int = 16) -> float:
    """Compute R_V from V-projection outputs."""
    pr_early = participation_ratio(v_early, window_size=window)
    pr_late = participation_ratio(v_late, window_size=window)
    if pr_early == 0 or np.isnan(pr_early) or np.isnan(pr_late):
        return float('nan')
    return float(pr_late / pr_early)


def clear_gpu():
    """Clear GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


@contextmanager
def capture_v_projections(model, layers: List[int]):
    """Capture V-projection outputs at multiple layers."""
    storage = {l: None for l in layers}
    handles = []
    
    def make_hook(layer_idx):
        def hook_fn(module, inp, out):
            storage[layer_idx] = out.detach()
            return out
        return hook_fn
    
    for l in layers:
        h = model.model.layers[l].self_attn.v_proj.register_forward_hook(make_hook(l))
        handles.append(h)
    
    try:
        yield storage
    finally:
        for h in handles:
            h.remove()


# =============================================================================
# INTERVENTION FUNCTIONS
# =============================================================================

@contextmanager
def zero_heads_at_layer(model, layer_idx: int, head_indices: List[int]):
    """Zero out specific attention heads at a layer."""
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads
    
    def hook_fn(module, inp, out):
        if isinstance(out, tuple):
            attn_output = out[0]
        else:
            attn_output = out
            
        batch, seq, hidden = attn_output.shape
        reshaped = attn_output.clone().view(batch, seq, num_heads, head_dim)
        for h in head_indices:
            reshaped[:, :, h, :] = 0
        modified = reshaped.view(batch, seq, hidden)
        
        if isinstance(out, tuple):
            return (modified,) + out[1:]
        return modified
    
    handle = model.model.layers[layer_idx].self_attn.register_forward_hook(hook_fn)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def mean_ablate_heads_at_layer(model, layer_idx: int, head_indices: List[int]):
    """Mean ablation: replace head outputs with their mean activation."""
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads
    
    def hook_fn(module, inp, out):
        if isinstance(out, tuple):
            attn_output = out[0]
        else:
            attn_output = out
            
        batch, seq, hidden = attn_output.shape
        reshaped = attn_output.clone().view(batch, seq, num_heads, head_dim)
        
        for h in head_indices:
            # Compute mean over sequence dimension
            mean_val = reshaped[:, :, h, :].mean(dim=1, keepdim=True)
            reshaped[:, :, h, :] = mean_val
        
        modified = reshaped.view(batch, seq, hidden)
        
        if isinstance(out, tuple):
            return (modified,) + out[1:]
        return modified
    
    handle = model.model.layers[layer_idx].self_attn.register_forward_hook(hook_fn)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def patch_residual_at_layer(model, layer_idx: int, source_residual: torch.Tensor, window: int):
    """Patch residual stream at a layer with source activation."""
    def hook_fn(module, inputs):
        hidden = inputs[0]
        batch, seq, dim = hidden.shape
        src_seq = source_residual.shape[1]
        
        W = min(window, seq, src_seq)
        if W > 0:
            patched = hidden.clone()
            patched[:, -W:, :] = source_residual[:, -W:, :].to(hidden.device).to(hidden.dtype)
            return (patched, *inputs[1:])
        return inputs
    
    handle = model.model.layers[layer_idx].register_forward_pre_hook(hook_fn)
    try:
        yield
    finally:
        handle.remove()


def get_residual_at_layer(model, tokenizer, prompt: str, layer: int) -> torch.Tensor:
    """Get residual stream at a specific layer."""
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    storage = {"resid": None}
    
    def hook_fn(module, inputs):
        storage["resid"] = inputs[0].detach().cpu()
        return inputs
    
    handle = model.model.layers[layer].register_forward_pre_hook(hook_fn)
    with torch.no_grad():
        model(**enc)
    handle.remove()
    
    return storage["resid"]


def run_and_measure_rv(model, tokenizer, prompt: str, intervention_fn=None) -> float:
    """Run model with optional intervention and measure R_V."""
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    
    with capture_v_projections(model, [EARLY_LAYER, MEASURE_LAYER]) as storage:
        if intervention_fn is not None:
            with intervention_fn(model):
                with torch.no_grad():
                    model(**enc)
        else:
            with torch.no_grad():
                model(**enc)
    
    v_early = storage[EARLY_LAYER]
    v_late = storage[MEASURE_LAYER]
    
    if v_early is None or v_late is None:
        return float('nan')
    
    return compute_rv(v_early[0], v_late[0], WINDOW)


# =============================================================================
# EXPERIMENT 1: EARLY-LAYER HEAD ABLATION (FOCUSED)
# =============================================================================

def run_early_layer_head_ablation_focused(model, tokenizer, prompts: Dict[str, List[str]]) -> Dict:
    """
    Test head ablation at ramp layers (L10-L22).
    
    Strategy: Test all heads at key ramp layers, but sample prompts.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 1: Early-Layer Head Ablation (Ramp Region)")
    print("="*60)
    
    results = []
    
    # Use fewer prompts but more systematic head testing
    recursive_prompts = prompts["recursive"][:10]
    baseline_prompts = prompts["baseline"][:10]
    all_prompts = recursive_prompts + baseline_prompts
    
    for layer in tqdm(RAMP_LAYERS, desc="Layers"):
        # Test all heads individually
        for head in tqdm(range(NUM_HEADS), desc=f"L{layer} heads", leave=False):
            for prompt in all_prompts[:8]:  # 8 prompts per layer-head combo
                rv_baseline = run_and_measure_rv(model, tokenizer, prompt)
                
                rv_zero = run_and_measure_rv(
                    model, tokenizer, prompt,
                    intervention_fn=lambda m, l=layer, h=head: zero_heads_at_layer(m, l, [h])
                )
                
                results.append({
                    "experiment": "early_layer_head_ablation",
                    "layer": layer,
                    "head": head,
                    "prompt_type": "recursive" if prompt in recursive_prompts else "baseline",
                    "rv_baseline": rv_baseline,
                    "rv_zero": rv_zero,
                    "delta": rv_zero - rv_baseline,
                })
                
                clear_gpu()
    
    return {"early_layer_head_ablation": results}


# =============================================================================
# EXPERIMENT 2: MEAN ABLATION VS ZEROING (FOCUSED)
# =============================================================================

def run_mean_ablation_focused(model, tokenizer, prompts: Dict[str, List[str]]) -> Dict:
    """
    Compare mean ablation vs zeroing at key layers.
    
    Strategy: Test at L15, L20, L27 (representative layers).
    """
    print("\n" + "="*60)
    print("EXPERIMENT 2: Mean Ablation vs Zeroing")
    print("="*60)
    
    results = []
    
    test_layers = [15, 20, 27]  # Representative: ramp, strong ramp, measurement
    recursive_prompts = prompts["recursive"][:8]
    
    for layer in tqdm(test_layers, desc="Layers"):
        for head in tqdm(range(0, NUM_HEADS, 2), desc=f"L{layer} heads", leave=False):  # Every other head
            for prompt in recursive_prompts:
                rv_baseline = run_and_measure_rv(model, tokenizer, prompt)
                
                # Zero ablation
                rv_zero = run_and_measure_rv(
                    model, tokenizer, prompt,
                    intervention_fn=lambda m, l=layer, h=head: zero_heads_at_layer(m, l, [h])
                )
                
                # Mean ablation
                rv_mean = run_and_measure_rv(
                    model, tokenizer, prompt,
                    intervention_fn=lambda m, l=layer, h=head: mean_ablate_heads_at_layer(m, l, [h])
                )
                
                results.append({
                    "experiment": "mean_ablation_comparison",
                    "layer": layer,
                    "head": head,
                    "rv_baseline": rv_baseline,
                    "rv_zero": rv_zero,
                    "rv_mean": rv_mean,
                    "delta_zero": rv_zero - rv_baseline,
                    "delta_mean": rv_mean - rv_baseline,
                    "difference": (rv_mean - rv_baseline) - (rv_zero - rv_baseline),
                })
                
                clear_gpu()
    
    return {"mean_ablation_comparison": results}


# =============================================================================
# EXPERIMENT 3: REVERSE PATCHING (FOCUSED)
# =============================================================================

def run_reverse_patching_focused(model, tokenizer, prompts: Dict[str, List[str]]) -> Dict:
    """
    Patch baseline → recursive to undo effect.
    
    Strategy: Test at all ramp + late layers, but fewer prompt pairs.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 3: Reverse Patching (Baseline → Recursive)")
    print("="*60)
    
    results = []
    
    recursive_prompts = prompts["recursive"][:8]
    baseline_prompts = prompts["baseline"][:8]
    
    for rec_prompt, base_prompt in tqdm(zip(recursive_prompts, baseline_prompts), 
                                         desc="Prompt pairs", total=len(recursive_prompts)):
        rv_rec_baseline = run_and_measure_rv(model, tokenizer, rec_prompt)
        rv_base_baseline = run_and_measure_rv(model, tokenizer, base_prompt)
        
        # Test patching baseline residual into recursive at different layers
        test_layers = RAMP_LAYERS + LATE_LAYERS
        
        for layer in test_layers:
            base_residual = get_residual_at_layer(model, tokenizer, base_prompt, layer)
            
            rv_patched = run_and_measure_rv(
                model, tokenizer, rec_prompt,
                intervention_fn=lambda m, l=layer, r=base_residual: patch_residual_at_layer(m, l, r, WINDOW)
            )
            
            results.append({
                "experiment": "reverse_patching",
                "layer": layer,
                "rv_rec_baseline": rv_rec_baseline,
                "rv_base_baseline": rv_base_baseline,
                "rv_patched": rv_patched,
                "delta": rv_patched - rv_rec_baseline,
                "recovery": rv_patched - rv_base_baseline,  # How close to baseline?
                "recovery_pct": ((rv_patched - rv_rec_baseline) / (rv_base_baseline - rv_rec_baseline) * 100) 
                                if (rv_base_baseline - rv_rec_baseline) != 0 else 0,
            })
            
            clear_gpu()
    
    return {"reverse_patching": results}


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*60)
    print("CIRCUIT HUNT V2 FOCUSED: Fast, Targeted Tests")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_NAME}")
    print(f"Start time: {datetime.now().isoformat()}")
    print("="*60)
    
    set_seed(SEED)
    
    # Load model
    print("\nLoading model...")
    model, tokenizer = load_model(MODEL_NAME, device=DEVICE)
    model.eval()
    
    # Load prompts
    print("Loading prompts...")
    loader = PromptLoader()
    
    recursive_prompts = loader.get_by_pillar("dose_response", limit=30)
    baseline_prompts = loader.get_by_pillar("baselines", limit=30)
    
    prompts = {
        "recursive": recursive_prompts,
        "baseline": baseline_prompts,
    }
    
    print(f"Loaded {len(recursive_prompts)} recursive, {len(baseline_prompts)} baseline prompts")
    
    # Run experiments
    all_results = {}
    
    try:
        # Experiment 1: Early-layer head ablation
        print("\n>>> Starting Experiment 1: Early-Layer Head Ablation")
        results_1 = run_early_layer_head_ablation_focused(model, tokenizer, prompts)
        all_results.update(results_1)
        clear_gpu()
        
        # Experiment 2: Mean ablation comparison
        print("\n>>> Starting Experiment 2: Mean Ablation Comparison")
        results_2 = run_mean_ablation_focused(model, tokenizer, prompts)
        all_results.update(results_2)
        clear_gpu()
        
        # Experiment 3: Reverse patching
        print("\n>>> Starting Experiment 3: Reverse Patching")
        results_3 = run_reverse_patching_focused(model, tokenizer, prompts)
        all_results.update(results_3)
        clear_gpu()
        
    except Exception as e:
        print(f"\nError during experiment: {e}")
        import traceback
        traceback.print_exc()
    
    # Save results
    output_dir = Path("results/circuit_hunt_v2_focused")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save full results
    with open(output_dir / f"results_{timestamp}.json", "w") as f:
        def convert(obj):
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, torch.Tensor):
                return obj.cpu().numpy().tolist()
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        
        json.dump(convert(all_results), f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    # Find significant effects
    for exp_name, exp_results in all_results.items():
        if not exp_results:
            continue
            
        df = pd.DataFrame(exp_results)
        
        print(f"\n--- {exp_name} ---")
        
        if "delta" in df.columns:
            # Find significant effects
            significant = df[df["delta"].abs() > 0.02]
            if len(significant) > 0:
                print(f"Found {len(significant)} interventions with |Δ| > 0.02")
                print("\nTop effects:")
                top = significant.nlargest(15, "delta", key=abs)
                cols = [c for c in ["layer", "head", "delta"] if c in top.columns]
                print(top[cols].to_string())
            else:
                print("No interventions with |Δ| > 0.02")
        
        if "recovery_pct" in df.columns:
            # For reverse patching, show recovery percentages
            high_recovery = df[df["recovery_pct"].abs() > 20]
            if len(high_recovery) > 0:
                print(f"\nFound {len(high_recovery)} layers with >20% recovery:")
                print(high_recovery[["layer", "recovery_pct", "rv_patched"]].to_string())
    
    print(f"\nResults saved to: {output_dir}")
    print(f"End time: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()

