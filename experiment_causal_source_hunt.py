import os
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from contextlib import contextmanager

# Add project root to path
sys.path.insert(0, os.path.abspath("."))

from src.core.models import load_model, set_seed
from src.metrics.rv import compute_rv
from prompts.loader import PromptLoader

print("Starting Causal Source Hunt...", flush=True)

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
WINDOW_SIZE = 16
MEASUREMENT_LAYER = 27

# Ranges to test for "Accumulated Effect"
LAYER_RANGES = [
    (0, 2),   # Early Layer Knockout
    (0, 5),   # Early block
    (5, 10),  # Mid-early
    (10, 15), # Middle
    (15, 20), # Mid-late
    (20, 27)  # Late (Control Knob area)
]

@contextmanager
def apply_ablation(model, layer_idx, mode="mean", noise_std=0.1):
    """
    Ablates the residual stream at layer_idx.
    mode: "mean" (batch mean), "zero", "noise"
    """
    def hook_fn(module, args):
        hidden_states = args[0] # (B, T, D)
        
        if mode == "zero":
            new_hidden = torch.zeros_like(hidden_states)
        elif mode == "mean":
            # Replace with batch mean across B dimension (preserving T structure? or global mean?)
            # Usually preserving T structure is better for positional info
            mean_act = hidden_states.mean(dim=0, keepdim=True)
            new_hidden = mean_act.expand_as(hidden_states)
        elif mode == "noise":
            noise = torch.randn_like(hidden_states) * noise_std
            new_hidden = hidden_states + noise
        else:
            new_hidden = hidden_states

        return (new_hidden, *args[1:])

    handle = model.model.layers[layer_idx].register_forward_pre_hook(hook_fn)
    try:
        yield
    finally:
        handle.remove()

@contextmanager
def patch_layer_range(model, layer_range, source_cache):
    """
    Patches a range of layers [start, end) from source_cache.
    source_cache: dict mapping layer_idx -> activation_tensor
    """
    start, end = layer_range
    handles = []
    
    def make_hook(l_idx):
        def hook_fn(module, args):
            if l_idx in source_cache:
                # We overwrite the input to this layer with the cached source
                # Note: This is simplified. Strictly we should patch the *output* of the previous components
                # or the residual stream *at* this layer.
                # args[0] is residual stream entrance.
                source_act = source_cache[l_idx].to(args[0].device)
                
                # Handle sequence length mismatch if any
                curr_len = args[0].shape[1]
                src_len = source_act.shape[1]
                min_len = min(curr_len, src_len)
                
                new_hidden = args[0].clone()
                new_hidden[:, -min_len:, :] = source_act[:, -min_len:, :]
                return (new_hidden, *args[1:])
            return args
        return hook_fn

    for l in range(start, end):
        h = model.model.layers[l].register_forward_pre_hook(make_hook(l))
        handles.append(h)
        
    try:
        yield
    finally:
        for h in handles:
            h.remove()

def cache_activations(model, tokenizer, prompt):
    """
    Runs forward pass and caches residual stream at EVERY layer entrance.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    cache = {}
    
    def make_capture_hook(l_idx):
        def hook(module, args):
            cache[l_idx] = args[0].detach().cpu() # Move to CPU
        return hook

    handles = []
    for l in range(len(model.model.layers)):
        handles.append(model.model.layers[l].register_forward_pre_hook(make_capture_hook(l)))
        
    with torch.no_grad():
        model(**inputs)
        
    for h in handles:
        h.remove()
        
    return cache

def run_experiment():
    print(f"Running Causal Source Hunt on {DEVICE}...")
    set_seed(42)
    model, tokenizer = load_model("mistralai/Mistral-7B-v0.1", device=DEVICE)
    model.eval()
    
    loader = PromptLoader()
    # 20 pairs for initial scan
    pairs = loader.get_balanced_pairs(n_pairs=20, seed=42)
    
    results = []
    
    for pair_idx, (rec_prompt, base_prompt) in enumerate(tqdm(pairs, desc="Pairs")):
        
        # 1. Establish Baselines
        try:
            rv_rec = compute_rv(model, tokenizer, rec_prompt, device=DEVICE)
            rv_base = compute_rv(model, tokenizer, base_prompt, device=DEVICE)
        except:
            continue
            
        # 2. Layer-wise Ablation Sweep (Option B.2)
        # We check every 2nd layer to save time
        for layer in range(0, 28, 2):
            # Mean Ablation on Recursive Prompt
            # Hypothesis: If we kill the "Source", R_V should go UP (Expansion / Normalcy)
            try:
                with apply_ablation(model, layer, mode="mean"):
                    rv_ablated = compute_rv(model, tokenizer, rec_prompt, device=DEVICE)
            except:
                rv_ablated = np.nan
            
            results.append({
                "pair_idx": pair_idx,
                "type": "ablation_mean",
                "layer_or_range": str(layer),
                "rv": rv_ablated,
                "baseline_rv": rv_rec
            })

        # 3. Range Patching (Option B.3 & B.4)
        # We want to see if patching Base -> Rec in a range *breaks* the recursion (increases R_V)
        # Or if patching Rec -> Base in a range *induces* the recursion (decreases R_V)
        
        # Cache source activations
        base_cache = cache_activations(model, tokenizer, base_prompt)
        rec_cache = cache_activations(model, tokenizer, rec_prompt)
        
        for (start, end) in LAYER_RANGES:
            range_str = f"{start}-{end}"
            
            # Experiment A: Induce Contraction (Rec -> Base)
            # Patch Recursive info into Baseline context
            try:
                with patch_layer_range(model, (start, end), rec_cache):
                    rv_induced = compute_rv(model, tokenizer, base_prompt, device=DEVICE)
            except:
                rv_induced = np.nan
                
            results.append({
                "pair_idx": pair_idx,
                "type": "patch_induction",
                "layer_or_range": range_str,
                "rv": rv_induced,
                "baseline_rv": rv_base
            })
            
            # Experiment B: Break Contraction (Base -> Rec)
            # Patch Baseline info into Recursive context (Restoration)
            try:
                with patch_layer_range(model, (start, end), base_cache):
                    rv_restored = compute_rv(model, tokenizer, rec_prompt, device=DEVICE)
            except:
                rv_restored = np.nan
                
            results.append({
                "pair_idx": pair_idx,
                "type": "patch_restoration",
                "layer_or_range": range_str,
                "rv": rv_restored,
                "baseline_rv": rv_rec
            })

    # Save
    df = pd.DataFrame(results)
    os.makedirs("results/dec13_source_hunt", exist_ok=True)
    df.to_csv("results/dec13_source_hunt/results.csv", index=False)
    
    print("\n--- Summary ---")
    summary = df.groupby(["type", "layer_or_range"])["rv"].mean().unstack(level=0)
    print(summary)

if __name__ == "__main__":
    run_experiment()

