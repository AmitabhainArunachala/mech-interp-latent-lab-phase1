"""
KITCHEN SINK EXPERIMENT: Systematic search for causal source of R_V contraction.

Tests:
1. MLP vs Attention decomposition (which component drives PR drop?)
2. Multi-head ablation (redundant heads masking single effects?)
3. Activation direction analysis (is there a "recursive mode" direction?)
4. Hysteresis test (point of no return?)

Run with: python experiment_kitchen_sink.py
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

# Layers to focus component decomposition on
FOCUS_LAYERS = [10, 15, 20, 24, 25, 26, 27]

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


@contextmanager  
def zero_attention_at_layer(model, layer_idx: int):
    """Zero out attention output at a specific layer."""
    def hook_fn(module, inp, out):
        # out is tuple: (attn_output, attn_weights, past_key_values)
        # We zero just the attn_output
        if isinstance(out, tuple):
            return (torch.zeros_like(out[0]),) + out[1:]
        return torch.zeros_like(out)
    
    handle = model.model.layers[layer_idx].self_attn.register_forward_hook(hook_fn)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def zero_mlp_at_layer(model, layer_idx: int):
    """Zero out MLP output at a specific layer."""
    def hook_fn(module, inp, out):
        if isinstance(out, tuple):
            return (torch.zeros_like(out[0]),) + out[1:]
        return torch.zeros_like(out)
    
    handle = model.model.layers[layer_idx].mlp.register_forward_hook(hook_fn)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def zero_heads_at_layer(model, layer_idx: int, head_indices: List[int]):
    """Zero out specific attention heads at a layer."""
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads
    
    def hook_fn(module, inp, out):
        # out is tuple: (attn_output, attn_weights, past_key_values)
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
def inject_direction_at_layer(model, layer_idx: int, direction: torch.Tensor, coefficient: float):
    """Inject a direction into the residual stream at a layer."""
    def hook_fn(module, inputs):
        hidden = inputs[0]
        # Add direction (broadcast across batch and sequence)
        # direction is (hidden_dim,), hidden is (batch, seq, hidden_dim)
        injected = hidden + coefficient * direction.unsqueeze(0).unsqueeze(0).to(hidden.dtype).to(hidden.device)
        return (injected, *inputs[1:])
    
    handle = model.model.layers[layer_idx].register_forward_pre_hook(hook_fn)
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


def run_and_measure_rv(model, tokenizer, prompt: str, intervention_fn=None) -> float:
    """Run model with optional intervention and measure R_V.
    
    Args:
        intervention_fn: Optional callable that takes (model,) and returns a context manager
    """
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


# =============================================================================
# EXPERIMENT 1: COMPONENT DECOMPOSITION
# =============================================================================

def run_component_decomposition(model, tokenizer, prompts: Dict[str, List[str]]) -> Dict:
    """Test if PR drop is in attention or MLPs."""
    print("\n" + "="*60)
    print("EXPERIMENT 1: Component Decomposition (Attention vs MLP)")
    print("="*60)
    
    results = []
    
    for prompt_type, prompt_list in prompts.items():
        for prompt in tqdm(prompt_list[:10], desc=f"{prompt_type}"):
            # Baseline R_V
            rv_baseline = run_and_measure_rv(model, tokenizer, prompt)
            
            for layer in FOCUS_LAYERS:
                # Zero attention at this layer
                rv_zero_attn = run_and_measure_rv(
                    model, tokenizer, prompt,
                    intervention_fn=lambda m, l=layer: zero_attention_at_layer(m, l)
                )
                
                # Zero MLP at this layer
                rv_zero_mlp = run_and_measure_rv(
                    model, tokenizer, prompt,
                    intervention_fn=lambda m, l=layer: zero_mlp_at_layer(m, l)
                )
                
                results.append({
                    "prompt_type": prompt_type,
                    "layer": layer,
                    "rv_baseline": rv_baseline,
                    "rv_zero_attn": rv_zero_attn,
                    "rv_zero_mlp": rv_zero_mlp,
                    "delta_attn": rv_zero_attn - rv_baseline,
                    "delta_mlp": rv_zero_mlp - rv_baseline,
                })
            
            clear_gpu()
    
    return {"component_decomposition": results}


# =============================================================================
# EXPERIMENT 2: MULTI-HEAD ABLATION
# =============================================================================

def run_multihead_ablation(model, tokenizer, prompts: Dict[str, List[str]]) -> Dict:
    """Test if redundant heads mask single-head effects."""
    print("\n" + "="*60)
    print("EXPERIMENT 2: Multi-Head Ablation")
    print("="*60)
    
    results = []
    
    # Head groups to ablate at L27
    head_groups = {
        "top5_by_entropy": [31, 3, 13, 10, 8],  # From earlier analysis
        "bottom5_by_entropy": [0, 1, 2, 4, 5],
        "all_except_h31": [i for i in range(32) if i != 31],
        "first_half": list(range(16)),
        "second_half": list(range(16, 32)),
        "random_8": [3, 7, 12, 18, 22, 25, 28, 31],
    }
    
    for prompt_type, prompt_list in prompts.items():
        for prompt in tqdm(prompt_list[:10], desc=f"{prompt_type}"):
            rv_baseline = run_and_measure_rv(model, tokenizer, prompt)
            
            for group_name, heads in head_groups.items():
                rv_ablated = run_and_measure_rv(
                    model, tokenizer, prompt,
                    intervention_fn=lambda m, h=heads: zero_heads_at_layer(m, MEASURE_LAYER, h)
                )
                
                results.append({
                    "prompt_type": prompt_type,
                    "head_group": group_name,
                    "num_heads_ablated": len(heads),
                    "rv_baseline": rv_baseline,
                    "rv_ablated": rv_ablated,
                    "delta": rv_ablated - rv_baseline,
                })
            
            clear_gpu()
    
    return {"multihead_ablation": results}


# =============================================================================
# EXPERIMENT 3: ACTIVATION DIRECTION ANALYSIS
# =============================================================================

def run_direction_analysis(model, tokenizer, prompts: Dict[str, List[str]]) -> Dict:
    """Find and test the 'recursive mode' direction."""
    print("\n" + "="*60)
    print("EXPERIMENT 3: Activation Direction Analysis")
    print("="*60)
    
    results = {"directions": {}, "injection_tests": []}
    
    # Collect residuals for direction extraction
    rec_residuals = {l: [] for l in [12, 20, 24, 27]}
    base_residuals = {l: [] for l in [12, 20, 24, 27]}
    
    print("Collecting residuals...")
    for prompt in tqdm(prompts["recursive"][:20], desc="Recursive"):
        for l in rec_residuals.keys():
            resid = get_residual_at_layer(model, tokenizer, prompt, l)
            rec_residuals[l].append(resid[0, -WINDOW:, :].mean(dim=0))  # Mean over window
        clear_gpu()
    
    for prompt in tqdm(prompts["baseline"][:20], desc="Baseline"):
        for l in base_residuals.keys():
            resid = get_residual_at_layer(model, tokenizer, prompt, l)
            base_residuals[l].append(resid[0, -WINDOW:, :].mean(dim=0))
        clear_gpu()
    
    # Compute difference directions
    print("Computing directions...")
    directions = {}
    for l in rec_residuals.keys():
        rec_mean = torch.stack(rec_residuals[l]).mean(dim=0)
        base_mean = torch.stack(base_residuals[l]).mean(dim=0)
        
        diff = rec_mean - base_mean
        diff_norm = diff / (diff.norm() + 1e-8)
        directions[l] = diff_norm
        
        results["directions"][l] = {
            "diff_norm": float(diff.norm().cpu()),
            "rec_mean_norm": float(rec_mean.norm().cpu()),
            "base_mean_norm": float(base_mean.norm().cpu()),
        }
    
    # Test direction injection
    print("Testing direction injection...")
    injection_layers = [12, 20, 24]
    coefficients = [0.5, 1.0, 2.0, 5.0]
    
    for prompt in tqdm(prompts["baseline"][:10], desc="Injection tests"):
        rv_baseline = run_and_measure_rv(model, tokenizer, prompt)
        
        for inject_layer in injection_layers:
            direction = directions[inject_layer].to(DEVICE)
            
            for coeff in coefficients:
                rv_injected = run_and_measure_rv(
                    model, tokenizer, prompt,
                    intervention_fn=lambda m, il=inject_layer, d=direction, c=coeff: inject_direction_at_layer(m, il, d, c)
                )
                
                results["injection_tests"].append({
                    "inject_layer": inject_layer,
                    "coefficient": coeff,
                    "rv_baseline": rv_baseline,
                    "rv_injected": rv_injected,
                    "delta": rv_injected - rv_baseline,
                })
        
        clear_gpu()
    
    return {"direction_analysis": results}


# =============================================================================
# EXPERIMENT 4: HYSTERESIS TEST
# =============================================================================

def run_hysteresis_test(model, tokenizer, prompts: Dict[str, List[str]]) -> Dict:
    """Test point of no return with two-stage patching."""
    print("\n" + "="*60)
    print("EXPERIMENT 4: Hysteresis / Point of No Return")
    print("="*60)
    
    results = []
    
    # Get source residuals
    push_layers = [15, 20, 24]
    undo_layers = [20, 24, 27]
    
    for rec_prompt, base_prompt in zip(prompts["recursive"][:10], prompts["baseline"][:10]):
        # Cache residuals
        rec_residuals = {}
        base_residuals = {}
        
        for l in set(push_layers + undo_layers):
            rec_residuals[l] = get_residual_at_layer(model, tokenizer, rec_prompt, l)
            base_residuals[l] = get_residual_at_layer(model, tokenizer, base_prompt, l)
        
        # Baseline R_V
        rv_baseline = run_and_measure_rv(model, tokenizer, base_prompt)
        rv_recursive = run_and_measure_rv(model, tokenizer, rec_prompt)
        
        for push_layer in push_layers:
            # Single push (no undo)
            rv_push_only = run_and_measure_rv(
                model, tokenizer, base_prompt,
                intervention_fn=lambda m, pl=push_layer, rr=rec_residuals[push_layer]: patch_residual_at_layer(m, pl, rr, WINDOW)
            )
            
            results.append({
                "test_type": "push_only",
                "push_layer": push_layer,
                "undo_layer": None,
                "rv_baseline": rv_baseline,
                "rv_result": rv_push_only,
                "delta": rv_push_only - rv_baseline,
            })
            
            # Push then attempt undo
            for undo_layer in [l for l in undo_layers if l > push_layer]:
                # Need to do two patches in same forward pass
                # This requires a custom combined hook
                enc = tokenizer(base_prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
                
                v_storage = {EARLY_LAYER: None, MEASURE_LAYER: None}
                handles = []
                
                # V-proj capture hooks
                def make_v_hook(layer_idx):
                    def hook_fn(module, inp, out):
                        v_storage[layer_idx] = out.detach()
                        return out
                    return hook_fn
                
                handles.append(model.model.layers[EARLY_LAYER].self_attn.v_proj.register_forward_hook(make_v_hook(EARLY_LAYER)))
                handles.append(model.model.layers[MEASURE_LAYER].self_attn.v_proj.register_forward_hook(make_v_hook(MEASURE_LAYER)))
                
                # Push patch hook
                def push_hook(module, inputs):
                    hidden = inputs[0]
                    patched = hidden.clone()
                    src = rec_residuals[push_layer].to(hidden.device).to(hidden.dtype)
                    W = min(WINDOW, hidden.shape[1], src.shape[1])
                    patched[:, -W:, :] = src[:, -W:, :]
                    return (patched, *inputs[1:])
                
                # Undo patch hook
                def undo_hook(module, inputs):
                    hidden = inputs[0]
                    patched = hidden.clone()
                    src = base_residuals[undo_layer].to(hidden.device).to(hidden.dtype)
                    W = min(WINDOW, hidden.shape[1], src.shape[1])
                    patched[:, -W:, :] = src[:, -W:, :]
                    return (patched, *inputs[1:])
                
                handles.append(model.model.layers[push_layer].register_forward_pre_hook(push_hook))
                handles.append(model.model.layers[undo_layer].register_forward_pre_hook(undo_hook))
                
                with torch.no_grad():
                    model(**enc)
                
                for h in handles:
                    h.remove()
                
                rv_push_undo = compute_rv(v_storage[EARLY_LAYER][0], v_storage[MEASURE_LAYER][0], WINDOW)
                
                results.append({
                    "test_type": "push_then_undo",
                    "push_layer": push_layer,
                    "undo_layer": undo_layer,
                    "rv_baseline": rv_baseline,
                    "rv_result": rv_push_undo,
                    "delta": rv_push_undo - rv_baseline,
                    "rv_push_only": rv_push_only,
                    "undo_effect": rv_push_undo - rv_push_only,
                })
        
        clear_gpu()
    
    return {"hysteresis": results}


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*60)
    print("KITCHEN SINK EXPERIMENT: Systematic Causal Source Hunt")
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
        # Experiment 1: Component decomposition
        results_1 = run_component_decomposition(model, tokenizer, prompts)
        all_results.update(results_1)
        clear_gpu()
        
        # Experiment 2: Multi-head ablation
        results_2 = run_multihead_ablation(model, tokenizer, prompts)
        all_results.update(results_2)
        clear_gpu()
        
        # Experiment 3: Direction analysis
        results_3 = run_direction_analysis(model, tokenizer, prompts)
        all_results.update(results_3)
        clear_gpu()
        
        # Experiment 4: Hysteresis
        results_4 = run_hysteresis_test(model, tokenizer, prompts)
        all_results.update(results_4)
        clear_gpu()
        
    except Exception as e:
        print(f"\nError during experiment: {e}")
        import traceback
        traceback.print_exc()
    
    # Save results
    output_dir = Path("results/dec13_kitchen_sink")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save full results
    with open(output_dir / f"results_{timestamp}.json", "w") as f:
        # Convert numpy/torch to python types
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
    
    # Component decomposition summary
    if "component_decomposition" in all_results:
        df = pd.DataFrame(all_results["component_decomposition"])
        print("\n--- Component Decomposition ---")
        print("Effect of zeroing attention vs MLP on R_V (delta from baseline):")
        summary = df.groupby(["prompt_type", "layer"])[["delta_attn", "delta_mlp"]].mean()
        print(summary.round(4))
    
    # Multi-head ablation summary
    if "multihead_ablation" in all_results:
        df = pd.DataFrame(all_results["multihead_ablation"])
        print("\n--- Multi-Head Ablation at L27 ---")
        print("Effect of ablating head groups on R_V:")
        summary = df.groupby(["prompt_type", "head_group"])["delta"].mean()
        print(summary.round(4))
    
    # Direction injection summary
    if "direction_analysis" in all_results:
        tests = all_results["direction_analysis"].get("injection_tests", [])
        if tests:
            df = pd.DataFrame(tests)
            print("\n--- Direction Injection ---")
            print("Effect of injecting (rec-base) direction into baseline prompts:")
            summary = df.groupby(["inject_layer", "coefficient"])["delta"].mean()
            print(summary.round(4))
    
    # Hysteresis summary
    if "hysteresis" in all_results:
        df = pd.DataFrame(all_results["hysteresis"])
        print("\n--- Hysteresis Test ---")
        print("Can we undo the push? (undo_effect < 0 means undo worked)")
        push_undo = df[df["test_type"] == "push_then_undo"]
        if len(push_undo) > 0:
            summary = push_undo.groupby(["push_layer", "undo_layer"])[["rv_result", "undo_effect"]].mean()
            print(summary.round(4))
    
    print(f"\nResults saved to: {output_dir}")
    print(f"End time: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()

