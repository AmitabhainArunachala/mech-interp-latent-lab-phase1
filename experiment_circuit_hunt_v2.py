"""
CIRCUIT HUNT V2: Testing Hypotheses You Didn't Try

This script tests multiple alternative approaches to find the causal circuit:

1. Early-layer head ablation (L8-L23 where ramp occurs, not just L27)
2. Mean ablation instead of zeroing (preserves statistics)
3. Reverse patching (baseline → recursive to undo effect)
4. Path patching (trace information flow from early to late)
5. Head interaction effects (pairs/triplets, not just singles)
6. Attention output patching (not just residual stream)

The key insight: Maybe the circuit is NOT at L27 (where we measure), but at L8-L23 (where the ramp happens).
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
from typing import Dict, List, Optional, Tuple, Callable
from tqdm import tqdm
from itertools import combinations

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

# Key insight: Ramp happens at L8-L23, not L27
RAMP_LAYERS = [8, 10, 12, 15, 18, 20, 22, 23]
LATE_LAYERS = [24, 25, 26, 27]

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
def mean_ablate_heads_at_layer(model, layer_idx: int, head_indices: List[int], 
                                mean_activations: Optional[torch.Tensor] = None):
    """
    Mean ablation: replace head outputs with their mean activation.
    
    If mean_activations is None, computes mean on-the-fly from current forward pass.
    """
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
            if mean_activations is not None:
                # Use pre-computed mean
                reshaped[:, :, h, :] = mean_activations[h].unsqueeze(0).unsqueeze(0).expand(
                    batch, seq, -1
                )
            else:
                # Compute mean on-the-fly (mean over sequence)
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
def patch_head_outputs_at_layer(model, layer_idx: int, head_indices: List[int],
                                source_outputs: torch.Tensor):
    """
    Patch specific head outputs with source outputs.
    
    Args:
        source_outputs: Tensor of shape (batch, seq, num_heads, head_dim) or (seq, num_heads, head_dim)
    """
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads
    
    def hook_fn(module, inp, out):
        if isinstance(out, tuple):
            attn_output = out[0]
        else:
            attn_output = out
            
        batch, seq, hidden = attn_output.shape
        reshaped = attn_output.clone().view(batch, seq, num_heads, head_dim)
        
        # Reshape source if needed
        if source_outputs.dim() == 3:
            src = source_outputs.unsqueeze(0)  # Add batch dim
        else:
            src = source_outputs
        
        # Match sequence length
        src_seq = src.shape[1]
        W = min(seq, src_seq)
        
        for h in head_indices:
            reshaped[:, -W:, h, :] = src[:, -W:, h, :].to(reshaped.device).to(reshaped.dtype)
        
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


def get_head_outputs_at_layer(model, tokenizer, prompt: str, layer: int) -> torch.Tensor:
    """Get all head outputs at a specific layer."""
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    storage = {"head_outputs": None}
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads
    
    def hook_fn(module, inp, out):
        if isinstance(out, tuple):
            attn_output = out[0]
        else:
            attn_output = out
        
        batch, seq, hidden = attn_output.shape
        reshaped = attn_output.view(batch, seq, num_heads, head_dim)
        storage["head_outputs"] = reshaped.detach().cpu()
        return out
    
    handle = model.model.layers[layer].self_attn.register_forward_hook(hook_fn)
    with torch.no_grad():
        model(**enc)
    handle.remove()
    
    return storage["head_outputs"]


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
# EXPERIMENT 1: EARLY-LAYER HEAD ABLATION (RAMP REGION)
# =============================================================================

def run_early_layer_head_ablation(model, tokenizer, prompts: Dict[str, List[str]]) -> Dict:
    """
    Test head ablation at L8-L23 (where ramp occurs), not just L27.
    
    Hypothesis: The circuit might be in the ramp region, not at measurement.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 1: Early-Layer Head Ablation (Ramp Region)")
    print("="*60)
    
    results = []
    
    # Test all heads individually at ramp layers
    for layer in tqdm(RAMP_LAYERS, desc="Layers"):
        for head in tqdm(range(NUM_HEADS), desc=f"L{layer} heads", leave=False):
            for prompt_type, prompt_list in prompts.items():
                for prompt in prompt_list[:5]:  # Sample 5 prompts per type
                    rv_baseline = run_and_measure_rv(model, tokenizer, prompt)
                    
                    rv_ablated = run_and_measure_rv(
                        model, tokenizer, prompt,
                        intervention_fn=lambda m, l=layer, h=head: zero_heads_at_layer(m, l, [h])
                    )
                    
                    results.append({
                        "experiment": "early_layer_head_ablation",
                        "layer": layer,
                        "head": head,
                        "prompt_type": prompt_type,
                        "rv_baseline": rv_baseline,
                        "rv_ablated": rv_ablated,
                        "delta": rv_ablated - rv_baseline,
                    })
                    
                    clear_gpu()
    
    return {"early_layer_head_ablation": results}


# =============================================================================
# EXPERIMENT 2: MEAN ABLATION VS ZEROING
# =============================================================================

def run_mean_ablation_comparison(model, tokenizer, prompts: Dict[str, List[str]]) -> Dict:
    """
    Compare mean ablation vs zeroing.
    
    Hypothesis: Zeroing might break the network in a way that masks effects.
    Mean ablation preserves statistics while removing information.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 2: Mean Ablation vs Zeroing")
    print("="*60)
    
    results = []
    
    # Test at ramp layers and late layers
    test_layers = RAMP_LAYERS[:4] + LATE_LAYERS[:2]  # Sample layers
    
    for layer in tqdm(test_layers, desc="Layers"):
        # Test all heads individually
        for head in tqdm(range(NUM_HEADS), desc=f"L{layer} heads", leave=False):
            for prompt_type, prompt_list in prompts.items():
                for prompt in prompt_list[:5]:
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
                        "prompt_type": prompt_type,
                        "rv_baseline": rv_baseline,
                        "rv_zero": rv_zero,
                        "rv_mean": rv_mean,
                        "delta_zero": rv_zero - rv_baseline,
                        "delta_mean": rv_mean - rv_baseline,
                    })
                    
                    clear_gpu()
    
    return {"mean_ablation_comparison": results}


# =============================================================================
# EXPERIMENT 3: REVERSE PATCHING (BASELINE → RECURSIVE)
# =============================================================================

def run_reverse_patching(model, tokenizer, prompts: Dict[str, List[str]]) -> Dict:
    """
    Patch FROM baseline TO recursive (undo the effect).
    
    Hypothesis: If we can undo the contraction by patching baseline activations
    into recursive prompts, we can identify what creates the effect.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 3: Reverse Patching (Baseline → Recursive)")
    print("="*60)
    
    results = []
    
    # Get matched prompt pairs
    recursive_prompts = prompts["recursive"][:10]
    baseline_prompts = prompts["baseline"][:10]
    
    for rec_prompt, base_prompt in tqdm(zip(recursive_prompts, baseline_prompts), 
                                         desc="Prompt pairs", total=len(recursive_prompts)):
        # Baseline R_V for both
        rv_rec_baseline = run_and_measure_rv(model, tokenizer, rec_prompt)
        rv_base_baseline = run_and_measure_rv(model, tokenizer, base_prompt)
        
        # Test patching baseline residual into recursive at different layers
        for layer in RAMP_LAYERS + LATE_LAYERS:
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
            })
            
            clear_gpu()
    
    return {"reverse_patching": results}


# =============================================================================
# EXPERIMENT 4: HEAD OUTPUT PATCHING (NOT RESIDUAL)
# =============================================================================

def run_head_output_patching(model, tokenizer, prompts: Dict[str, List[str]]) -> Dict:
    """
    Patch head outputs directly (not residual stream).
    
    Hypothesis: Maybe the circuit is in attention head outputs, not residual stream.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 4: Head Output Patching")
    print("="*60)
    
    results = []
    
    recursive_prompts = prompts["recursive"][:5]
    baseline_prompts = prompts["baseline"][:5]
    
    for rec_prompt, base_prompt in tqdm(zip(recursive_prompts, baseline_prompts),
                                         desc="Prompt pairs", total=len(recursive_prompts)):
        rv_rec_baseline = run_and_measure_rv(model, tokenizer, rec_prompt)
        rv_base_baseline = run_and_measure_rv(model, tokenizer, base_prompt)
        
        # Test patching baseline head outputs into recursive
        for layer in RAMP_LAYERS[:4]:  # Sample ramp layers
            base_head_outputs = get_head_outputs_at_layer(model, tokenizer, base_prompt, layer)
            
            # Test individual heads
            for head in range(0, NUM_HEADS, 4):  # Sample every 4th head
                rv_patched = run_and_measure_rv(
                    model, tokenizer, rec_prompt,
                    intervention_fn=lambda m, l=layer, h=head, src=base_head_outputs: 
                        patch_head_outputs_at_layer(m, l, [h], src)
                )
                
                results.append({
                    "experiment": "head_output_patching",
                    "layer": layer,
                    "head": head,
                    "rv_rec_baseline": rv_rec_baseline,
                    "rv_base_baseline": rv_base_baseline,
                    "rv_patched": rv_patched,
                    "delta": rv_patched - rv_rec_baseline,
                    "recovery": rv_patched - rv_base_baseline,
                })
                
                clear_gpu()
    
    return {"head_output_patching": results}


# =============================================================================
# EXPERIMENT 5: HEAD INTERACTION EFFECTS
# =============================================================================

def run_head_interactions(model, tokenizer, prompts: Dict[str, List[str]]) -> Dict:
    """
    Test head pairs/triplets (interaction effects).
    
    Hypothesis: Maybe individual heads have no effect, but pairs/triplets do.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 5: Head Interaction Effects")
    print("="*60)
    
    results = []
    
    # Focus on L20 (strong ramp) and L27 (measurement)
    test_layers = [20, 27]
    
    for layer in tqdm(test_layers, desc="Layers"):
        # Test pairs of heads
        # Sample pairs: top 10 heads by index (to reduce combinatorics)
        test_heads = list(range(0, min(16, NUM_HEADS), 2))  # Every other head, up to 16
        
        for head_pair in tqdm(list(combinations(test_heads, 2))[:20], 
                              desc=f"L{layer} pairs", leave=False):
            for prompt_type, prompt_list in prompts.items():
                for prompt in prompt_list[:5]:
                    rv_baseline = run_and_measure_rv(model, tokenizer, prompt)
                    
                    rv_ablated = run_and_measure_rv(
                        model, tokenizer, prompt,
                        intervention_fn=lambda m, l=layer, h=head_pair: zero_heads_at_layer(m, l, list(h))
                    )
                    
                    results.append({
                        "experiment": "head_interactions",
                        "layer": layer,
                        "head_1": head_pair[0],
                        "head_2": head_pair[1],
                        "prompt_type": prompt_type,
                        "rv_baseline": rv_baseline,
                        "rv_ablated": rv_ablated,
                        "delta": rv_ablated - rv_baseline,
                    })
                    
                    clear_gpu()
    
    return {"head_interactions": results}


# =============================================================================
# EXPERIMENT 6: PATH PATCHING (TRACE INFO FLOW)
# =============================================================================

def run_path_patching(model, tokenizer, prompts: Dict[str, List[str]]) -> Dict:
    """
    Path patching: patch at source layer, measure at target layer.
    
    Hypothesis: Trace information flow from early layers to L27.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 6: Path Patching (Trace Info Flow)")
    print("="*60)
    
    results = []
    
    recursive_prompts = prompts["recursive"][:5]
    baseline_prompts = prompts["baseline"][:5]
    
    for rec_prompt, base_prompt in tqdm(zip(recursive_prompts, baseline_prompts),
                                         desc="Prompt pairs", total=len(recursive_prompts)):
        rv_rec_baseline = run_and_measure_rv(model, tokenizer, rec_prompt)
        rv_base_baseline = run_and_measure_rv(model, tokenizer, base_prompt)
        
        # Patch at source layers, measure effect at target
        source_layers = [8, 12, 15, 20]
        target_layer = MEASURE_LAYER
        
        for source_layer in source_layers:
            # Patch recursive residual into baseline at source
            rec_residual = get_residual_at_layer(model, tokenizer, rec_prompt, source_layer)
            
            rv_patched = run_and_measure_rv(
                model, tokenizer, base_prompt,
                intervention_fn=lambda m, l=source_layer, r=rec_residual: 
                    patch_residual_at_layer(m, l, r, WINDOW)
            )
            
            results.append({
                "experiment": "path_patching",
                "source_layer": source_layer,
                "target_layer": target_layer,
                "rv_rec_baseline": rv_rec_baseline,
                "rv_base_baseline": rv_base_baseline,
                "rv_patched": rv_patched,
                "delta": rv_patched - rv_base_baseline,
                "effect_size": (rv_patched - rv_base_baseline) / (rv_rec_baseline - rv_base_baseline) if (rv_rec_baseline - rv_base_baseline) != 0 else 0,
            })
            
            clear_gpu()
    
    return {"path_patching": results}


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*60)
    print("CIRCUIT HUNT V2: Testing Alternative Hypotheses")
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
        results_1 = run_early_layer_head_ablation(model, tokenizer, prompts)
        all_results.update(results_1)
        clear_gpu()
        
        # Experiment 2: Mean ablation comparison
        print("\n>>> Starting Experiment 2: Mean Ablation Comparison")
        results_2 = run_mean_ablation_comparison(model, tokenizer, prompts)
        all_results.update(results_2)
        clear_gpu()
        
        # Experiment 3: Reverse patching
        print("\n>>> Starting Experiment 3: Reverse Patching")
        results_3 = run_reverse_patching(model, tokenizer, prompts)
        all_results.update(results_3)
        clear_gpu()
        
        # Experiment 4: Head output patching
        print("\n>>> Starting Experiment 4: Head Output Patching")
        results_4 = run_head_output_patching(model, tokenizer, prompts)
        all_results.update(results_4)
        clear_gpu()
        
        # Experiment 5: Head interactions
        print("\n>>> Starting Experiment 5: Head Interactions")
        results_5 = run_head_interactions(model, tokenizer, prompts)
        all_results.update(results_5)
        clear_gpu()
        
        # Experiment 6: Path patching
        print("\n>>> Starting Experiment 6: Path Patching")
        results_6 = run_path_patching(model, tokenizer, prompts)
        all_results.update(results_6)
        clear_gpu()
        
    except Exception as e:
        print(f"\nError during experiment: {e}")
        import traceback
        traceback.print_exc()
    
    # Save results
    output_dir = Path("results/circuit_hunt_v2")
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
            significant = df[df["delta"].abs() > 0.01]
            if len(significant) > 0:
                print(f"Found {len(significant)} interventions with |Δ| > 0.01")
                print("\nTop effects:")
                top = significant.nlargest(10, "delta", key=abs)
                print(top[["layer", "head", "delta"]].to_string() if "head" in top.columns 
                      else top[["layer", "delta"]].to_string())
            else:
                print("No interventions with |Δ| > 0.01")
    
    print(f"\nResults saved to: {output_dir}")
    print(f"End time: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()

