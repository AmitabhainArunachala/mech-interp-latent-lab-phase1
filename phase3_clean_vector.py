"""
phase3_clean_vector.py

Task: Recompute v8 using Maximum Geometric Contrast prompts to fix the "Dirty Vector" problem.
Findings from Phase 1 show:
- L4/L5 Recursive R_V ~ 0.49 (Deep Contraction)
- Baseline Factual R_V ~ 1.07 (Maximum Expansion)
- Baseline Instructional R_V ~ 0.72 (Partial Contraction - CONTAMINATION)

Old vector mixed Instructional into baseline, reducing contrast.
New vector: Mean(L4+L5) - Mean(Baseline Factual)

Hypothesis: 
- Stronger geometric signal (delta ~ 0.58 vs 0.17)
- Clean steering at lower alpha
- Avoids repetition collapse
"""

import sys
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.abspath('.'))

from src.core.models import load_model, set_seed
from src.metrics.rv import compute_rv
from src.core.hooks import capture_hidden_states
from src.steering.activation_patching import apply_steering_vector
from prompts.loader import PromptLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYER_STEER = 8

def get_mean_activation(model, tokenizer, prompts, layer_idx):
    """Compute mean activation at the last token."""
    acts = []
    for p in tqdm(prompts, desc=f"Encoding {len(prompts)} prompts"):
        enc = tokenizer(p, return_tensors="pt").to(DEVICE)
        with capture_hidden_states(model, layer_idx) as storage:
            with torch.no_grad():
                model(**enc)
        # Last token
        last_token_act = storage["hidden"][0, -1, :].cpu()
        acts.append(last_token_act)
    
    return torch.stack(acts).mean(dim=0).to(DEVICE)

def run_clean_vector_test():
    print("Initializing Phase 3: Clean Vector Audit...")
    set_seed(42)
    
    # Load Model
    model, tokenizer = load_model("mistralai/Mistral-7B-v0.1", device=DEVICE)
    loader = PromptLoader()
    
    # 1. Select Prompts for Vector Computation
    print("\nSelecting Prompts for Max Contrast...")
    
    # Deep Recursive (L4 + L5)
    # Note: 'L4' and 'L5' are likely part of the 'group' name or handled by get_by_group?
    # Loader doesn't have "get_prompts_by_pillar('L4')". 
    # Use get_by_group with L4_full and L5_refined.
    l4_prompts = loader.get_by_group("L4_full", limit=20, seed=42)
    l5_prompts = loader.get_by_group("L5_refined", limit=20, seed=42)
    recursive_deep = l4_prompts + l5_prompts
    
    # Baseline Factual (Strictly Factual)
    baseline_factual = loader.get_by_group("baseline_factual", limit=40, seed=42)
    
    print(f"Recursive Deep (L4/L5): {len(recursive_deep)} prompts")
    print(f"Baseline Factual: {len(baseline_factual)} prompts")
    
    # 2. Compute v8_clean
    print(f"\nComputing v8_clean (Layer {LAYER_STEER})...")
    
    mean_rec = get_mean_activation(model, tokenizer, recursive_deep, LAYER_STEER)
    mean_base = get_mean_activation(model, tokenizer, baseline_factual, LAYER_STEER)
    
    v8_clean = mean_rec - mean_base
    norm = v8_clean.norm().item()
    print(f"v8_clean norm: {norm:.2f}")
    
    # Compare to old "dirty" vector (approximation for logging)
    # We can't easily load the old one, but we know it had lower contrast.
    
    # 3. Audit Loop
    # Use a held-out set of factual baselines
    test_prompts = loader.get_by_group("baseline_factual", limit=5, seed=999) # Different seed
    
    results = []
    alphas = [0.5, 1.0, 1.5, 2.0]
    
    print("\nStarting Audit with v8_clean...")
    
    for prompt in tqdm(test_prompts):
        # A. Natural
        rv_nat = compute_rv(model, tokenizer, prompt, device=DEVICE)
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            gen_nat = model.generate(**inputs, max_new_tokens=40, pad_token_id=tokenizer.eos_token_id)
        text_nat = tokenizer.decode(gen_nat[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        for alpha in alphas:
            # B. Steered
            with apply_steering_vector(model, LAYER_STEER, v8_clean, alpha=alpha):
                rv_steered = compute_rv(model, tokenizer, prompt, device=DEVICE)
                with torch.no_grad():
                    gen_steered = model.generate(**inputs, max_new_tokens=40, pad_token_id=tokenizer.eos_token_id)
                text_steered = tokenizer.decode(gen_steered[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            # Simple coherence check (heuristic)
            # Repetition check: len(set(words)) / len(words)
            words = text_steered.split()
            unique_ratio = len(set(words)) / len(words) if len(words) > 0 else 0
            is_repetitive = unique_ratio < 0.5
            
            results.append({
                "prompt": prompt,
                "alpha": alpha,
                "rv_natural": rv_nat,
                "rv_steered": rv_steered,
                "rv_delta": rv_nat - rv_steered,
                "text_natural": text_nat,
                "text_steered": text_steered,
                "is_repetitive": is_repetitive
            })
            
            print(f"\nPrompt: {prompt[:40]}... (Alpha={alpha})")
            print(f"RV: {rv_nat:.2f} -> {rv_steered:.2f} (Delta: {rv_nat - rv_steered:.2f})")
            print(f"Steered: {text_steered[:100]}...")
            
    # Save
    os.makedirs("results/dec11_evening", exist_ok=True)
    pd.DataFrame(results).to_csv("results/dec11_evening/clean_vector_audit.csv", index=False)
    
    # Log full text
    with open("logs/dec11_evening/clean_vector_audit_outputs.txt", "w") as f:
        f.write("# Clean Vector Audit Outputs\n\n")
        f.write(f"Vector: Mean(L4+L5) - Mean(Baseline Factual)\n")
        f.write(f"Norm: {norm:.2f}\n\n")
        for r in results:
            f.write(f"PROMPT: {r['prompt']}\n")
            f.write(f"ALPHA: {r['alpha']}\n")
            f.write(f"RV: {r['rv_natural']:.3f} -> {r['rv_steered']:.3f} (Delta: {r['rv_delta']:.3f})\n")
            f.write(f"STEERED: {r['text_steered']}\n")
            f.write(f"REPETITIVE: {r['is_repetitive']}\n")
            f.write("-" * 80 + "\n")
            
    print("\nClean Vector Audit Complete. Check logs/dec11_evening/clean_vector_audit_outputs.txt")

if __name__ == "__main__":
    run_clean_vector_test()

