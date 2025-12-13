
import os
import sys
print("Starting script...", flush=True)
import torch
print(f"Torch imported. CUDA: {torch.cuda.is_available()}", flush=True)
import numpy as np
import pandas as pd
from tqdm import tqdm
from contextlib import contextmanager

# Add project root to path
sys.path.insert(0, os.path.abspath("."))

print("Importing project modules...", flush=True)
from src.core.models import load_model, set_seed
from src.metrics.rv import compute_rv
from prompts.loader import PromptLoader
print("Modules imported.", flush=True)

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
WINDOW_SIZE = 16
MEASUREMENT_LAYER = 27 # The "Graveyard" where we measure R_V

# We want to sweep L5 to L27 to see the "Slippery Slope"
# Patching after L27 is useless for R_V@L27 (causality)
PATCH_LAYERS = list(range(5, 28, 2)) # [5, 7, ..., 27]

@contextmanager
def patch_residual_stream(model, layer_idx, source_activations):
    """
    Patches the residual stream at the START of layer_idx with source_activations.
    source_activations: (batch, seq, hidden)
    """
    def hook_fn(module, args):
        hidden_states = args[0] # (batch, seq, hidden)
        
        # Ensure we don't crash on shape mismatch (seq len)
        # We take the last W tokens from source and apply to last W of target
        B, T, D = hidden_states.shape
        src_B, src_T, src_D = source_activations.shape
        
        # Align sequence lengths (take suffix)
        W = min(WINDOW_SIZE, T, src_T)
        
        # Create patch tensor
        patch = source_activations[:, -W:, :].to(hidden_states.device).to(hidden_states.dtype)
        
        # Overwrite
        # Note: hidden_states is a tuple in some implementations, but typically tensor in hook
        # args[0] is the tensor. We modify it in-place or return new.
        # Cloning is safer.
        new_hidden = hidden_states.clone()
        new_hidden[:, -W:, :] = patch
        
        return (new_hidden, *args[1:])

    handle = model.model.layers[layer_idx].register_forward_pre_hook(hook_fn)
    try:
        yield
    finally:
        handle.remove()

def get_residual_activations(model, tokenizer, prompt, layer_idx):
    """
    Capture residual stream at input of layer_idx.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    activations = None
    
    def capture_hook(module, args):
        nonlocal activations
        activations = args[0].detach().cpu() # Move to CPU to save VRAM
        return None 

    handle = model.model.layers[layer_idx].register_forward_pre_hook(capture_hook)
    with torch.no_grad():
        model(**inputs)
    handle.remove()
    return activations

def run_experiment():
    print("="*60)
    print("EXPERIMENT: L5-L27 Coarse Sweep (The Slippery Slope)")
    print("="*60)
    
    set_seed(42)
    model, tokenizer = load_model("mistralai/Mistral-7B-v0.1", device=DEVICE)
    model.eval()
    
    loader = PromptLoader()
    # Use 20 pairs for speed but statistical significance
    pairs = loader.get_balanced_pairs(n_pairs=20, seed=42) 
    
    results = []
    
    # We want to test: 
    # 1. Patch Recursive -> Baseline (Does it INDUCE recursion? Expansion -> Contraction)
    # 2. Patch Baseline -> Recursive (Does it BREAK recursion? Contraction -> Expansion)
    
    for pair_idx, (rec_prompt, base_prompt) in enumerate(tqdm(pairs, desc="Pairs")):
        
        # 0. Measure Baselines (Natural State)
        try:
            rv_rec_natural = compute_rv(model, tokenizer, rec_prompt, device=DEVICE)
            rv_base_natural = compute_rv(model, tokenizer, base_prompt, device=DEVICE)
        except Exception:
            continue # Skip bad SVDs
            
        # For each layer in the sweep
        for layer in PATCH_LAYERS:
            
            # --- Condition A: INDUCTION (Recursive Content -> Baseline Context) ---
            # We take the residual from the Recursive prompt at Layer L
            # And inject it into the Baseline prompt at Layer L
            # Expected: R_V of Baseline should drop (Contract) as L increases
            
            rec_resid = get_residual_activations(model, tokenizer, rec_prompt, layer)
            
            try:
                with patch_residual_stream(model, layer, rec_resid):
                    rv_induced = compute_rv(model, tokenizer, base_prompt, device=DEVICE)
            except Exception as e:
                rv_induced = np.nan
                
            # --- Condition B: BREAKING (Baseline Content -> Recursive Context) ---
            # We take residual from Baseline
            # Inject into Recursive
            # Expected: R_V of Recursive should rise (Expand) as L increases
            
            base_resid = get_residual_activations(model, tokenizer, base_prompt, layer)
            
            try:
                with patch_residual_stream(model, layer, base_resid):
                    rv_broken = compute_rv(model, tokenizer, rec_prompt, device=DEVICE)
            except Exception as e:
                rv_broken = np.nan

            # --- Condition C: NOISE (Random -> Recursive Context) ---
            # Inject random noise with same norm as baseline
            # To test "Fragility" vs "Specificity"
            
            if base_resid is not None:
                noise = torch.randn_like(base_resid)
                # Normalize to match baseline energy
                noise = noise * (base_resid.norm() / noise.norm())
                
                try:
                    with patch_residual_stream(model, layer, noise):
                        rv_noise = compute_rv(model, tokenizer, rec_prompt, device=DEVICE)
                except:
                    rv_noise = np.nan
            else:
                rv_noise = np.nan

            results.append({
                "pair_idx": pair_idx,
                "layer": layer,
                "rv_rec_natural": rv_rec_natural,
                "rv_base_natural": rv_base_natural,
                "rv_induced": rv_induced, # Did we make the baseline contract?
                "rv_broken": rv_broken,   # Did we make the recursive expand?
                "rv_noise": rv_noise      # Did noise make it expand?
            })
        
    # Save Results
    df = pd.DataFrame(results)
    os.makedirs("results/dec13_slope", exist_ok=True)
    df.to_csv("results/dec13_slope/l5_l27_sweep.csv", index=False)
    
    # Print Quick Summary
    print("\n--- Summary Results (Mean R_V) ---")
    summary = df.groupby("layer")[["rv_rec_natural", "rv_base_natural", "rv_induced", "rv_broken", "rv_noise"]].mean()
    print(summary)
             
if __name__ == "__main__":
    run_experiment()
