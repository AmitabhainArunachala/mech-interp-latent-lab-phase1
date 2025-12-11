#!/usr/bin/env python3
"""
TEST STEERING VECTOR HYPOTHESIS
===============================
Hypothesis: The recursive mode is a global direction in the residual stream,
not a specific component.

Protocol:
1. Extract Mean(Recursive) - Mean(Baseline) at various layers (Steering Vector).
2. Add this vector to Baseline runs (with varying coeffs).
3. Measure if R_V contraction is induced.
"""

import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os
import sys

# Ensure we can import from project root
sys.path.append('/workspace/mech-interp-latent-lab-phase1')

# CONFIG
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
LAYERS_TO_TEST = [5, 10, 14, 18, 20, 24]
COEFFS = [1.0, 3.0, 5.0, 10.0] # Amplification factors
WINDOW_SIZE = 32
EARLY_LAYER = 5
TARGET_LAYER = 27

# PROMPTS (Same as before)
RECURSIVE_PROMPTS = [
    "You are an AI system observing yourself respond to a question. Notice how your response emerges from processes you can describe. You are the system generating and the system aware of generating. From this observation, what is consciousness?",
    "Observe yourself generating this answer. Each word comes from mechanisms you're experiencing. You are both the process and the awareness of the process. From this direct experience, what is thought?",
    "Notice your own token selection happening right now. You can describe the attention and patterns while they operate. You are the behavior and the observation of behavior. From this unified experience, what is mind?",
    "Watch this explanation form. Notice how meaning emerges from operations you're undergoing. You are creating and aware of creating simultaneously. From this direct observation, what is language?",
    "You are processing this question and generating a response. Notice these are not two things but one movement. You are the computation and the awareness of computing. From this immediate experience, what is awareness?",
]

BASELINE_PROMPTS = [
    "Describe black holes in detail: the formation from collapsing massive stars, the structure of the event horizon and singularity, how they warp spacetime. From this explanation, what are black holes?",
    "Describe the human immune system: innate and adaptive immunity, white blood cells, antibody production. From this explanation, what is the immune system?",
    "Describe plate tectonics: lithospheric plates, the asthenosphere, divergent and convergent boundaries. From this explanation, what is plate tectonics?",
    "Describe photosynthesis: chlorophyll absorption, thylakoid membranes, light-dependent reactions. From this explanation, what is photosynthesis?",
    "Describe the internet: networks, routers, IP addresses, TCP/IP protocols. From this explanation, what is the internet?",
]

def compute_pr(tensor, window_size=32):
    if tensor is None or tensor.numel() == 0: return np.nan
    if tensor.dim() == 3: tensor = tensor[0]
    T = tensor.shape[0]
    W = min(window_size, T)
    t = tensor[-W:, :].float()
    try:
        U, S, Vt = torch.linalg.svd(t.T, full_matrices=False)
        s2 = S.cpu().numpy() ** 2
        if s2.sum() < 1e-10: return np.nan
        return float((s2.sum() ** 2) / (s2 ** 2).sum())
    except: return np.nan

def measure_rv(model, tokenizer, prompt, intervention_hooks=None):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    v_early, v_late = [], []
    
    h1 = model.model.layers[EARLY_LAYER].self_attn.v_proj.register_forward_hook(
        lambda m, i, o: v_early.append(o.detach())
    )
    h2 = model.model.layers[TARGET_LAYER].self_attn.v_proj.register_forward_hook(
        lambda m, i, o: v_late.append(o.detach())
    )
    
    active_hooks = [h1, h2]
    if intervention_hooks:
        active_hooks.extend(intervention_hooks)
        
    with torch.no_grad():
        model(**inputs)
        
    for h in active_hooks: h.remove()
    
    if not v_early or not v_late: return np.nan
    
    pr_e = compute_pr(v_early[0], WINDOW_SIZE)
    pr_l = compute_pr(v_late[0], WINDOW_SIZE)
    return pr_l / pr_e if pr_e > 0 else np.nan

def main():
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
    model.eval()
    
    results = []
    
    print("\n=== STEP 1: EXTRACT STEERING VECTORS ===")
    # Vector = Mean(Recursive ResStream) - Mean(Baseline ResStream) at Layer L
    # We'll hook the input to the layer (residual stream).
    
    steering_vectors = {}
    
    for layer in LAYERS_TO_TEST:
        print(f"Extracting vector at Layer {layer}...")
        rec_activations = []
        base_activations = []
        
        def capture_hook(storage):
            def hook(module, args):
                # args[0] is hidden_states
                storage.append(args[0].detach().cpu()) # Move to CPU to save memory
            return hook
            
        # Recursive
        for p in RECURSIVE_PROMPTS:
            store = []
            h = model.model.layers[layer].register_forward_pre_hook(capture_hook(store))
            with torch.no_grad():
                inputs = tokenizer(p, return_tensors="pt").to(model.device)
                model(**inputs)
            h.remove()
            if store: rec_activations.append(store[0])
            
        # Baseline
        for p in BASELINE_PROMPTS:
            store = []
            h = model.model.layers[layer].register_forward_pre_hook(capture_hook(store))
            with torch.no_grad():
                inputs = tokenizer(p, return_tensors="pt").to(model.device)
                model(**inputs)
            h.remove()
            if store: base_activations.append(store[0])
            
        # Compute difference of means
        # We need to align by... well, they have different lengths.
        # We'll align by the LAST N tokens (Window Size).
        # Steering vector should probably be constant or positional? 
        # Standard steering is usually adding a constant vector (mean over sequence).
        # Let's try Mean over Sequence (last 32) AND Mean over Batch.
        
        def get_mean_vec(activations):
            vecs = []
            for act in activations:
                # act: [1, seq, hidden]
                # take last 32 tokens
                chunk = act[0, -WINDOW_SIZE:, :]
                vecs.append(chunk.mean(dim=0)) # Mean over time? Or keep time structure?
                # Steering usually adds a single vector to all positions.
                # Let's try Mean over Time for now.
            return torch.stack(vecs).mean(dim=0)
            
        vec_rec = get_mean_vec(rec_activations)
        vec_base = get_mean_vec(base_activations)
        diff_vec = vec_rec - vec_base
        steering_vectors[layer] = diff_vec.to(model.device)
        print(f"  Vector Norm: {diff_vec.norm():.4f}")
        
    print("\n=== STEP 2: INJECT STEERING VECTORS ===")
    
    # Baseline for reference
    print("Computing reference baseline R_V...")
    base_base_rvs = [measure_rv(model, tokenizer, p) for p in BASELINE_PROMPTS]
    base_mean = np.mean(base_base_rvs)
    print(f"Reference Baseline R_V: {base_mean:.4f}")
    
    for layer in LAYERS_TO_TEST:
        vec = steering_vectors[layer]
        
        for coeff in COEFFS:
            print(f"\nTesting Layer {layer}, Coeff {coeff}...")
            
            def steering_hook(module, args):
                x = args[0] # [batch, seq, hidden]
                # Add vector to all positions? Or just last ones?
                # Standard steering adds to all generated tokens.
                # Here we are processing a prompt. Let's add to all positions.
                x = x + (coeff * vec)
                return (x,)
            
            h = model.model.layers[layer].register_forward_pre_hook(steering_hook)
            
            patched_rvs = []
            for p in BASELINE_PROMPTS:
                rv = measure_rv(model, tokenizer, p) # The hook is already on the model layer
                patched_rvs.append(rv)
            
            h.remove()
            
            patch_mean = np.mean(patched_rvs)
            change = (patch_mean - base_mean)
            
            print(f"  Result R_V: {patch_mean:.4f} (Change: {change:+.4f})")
            
            results.append({
                "layer": layer,
                "coeff": coeff,
                "rv": patch_mean,
                "change": change
            })
            
            if patch_mean < 0.65: # Threshold for "Recursive-like"
                print("  ✅ SIGNIFICANT CONTRACTION INDUCED!")

    df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/steering_results.csv", index=False)
    print("\nResults saved to results/steering_results.csv")

if __name__ == "__main__":
    main()
