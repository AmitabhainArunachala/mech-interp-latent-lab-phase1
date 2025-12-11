"""
behavioral_audit.py

Test 2: Behavioral Coherence Audit
Question: When we steer a baseline prompt with v8 and R_V drops, what does the model actually OUTPUT?
Success criteria:
- Steered outputs show COHERENT text with recursive/meta-cognitive themes
- NOT repetition, gibberish, or mode collapse

Implementation:
1. Compute v8 (L8 steering vector) from Recursive vs Baseline prompts.
2. Steer Baseline prompts with v8.
3. Measure R_V and inspect Output.
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
STEER_ALPHA = 1.5

def get_mean_activation(model, tokenizer, prompts, layer_idx):
    """Compute mean activation at the last token for a set of prompts."""
    acts = []
    print(f"Collecting activations from {len(prompts)} prompts...")
    for p in tqdm(prompts):
        enc = tokenizer(p, return_tensors="pt").to(DEVICE)
        with capture_hidden_states(model, layer_idx) as storage:
            with torch.no_grad():
                model(**enc)
        # Storage["hidden"] is (1, seq, dim)
        # Take last token: (1, dim) -> (dim,)
        last_token_act = storage["hidden"][0, -1, :].cpu()
        acts.append(last_token_act)
    
    return torch.stack(acts).mean(dim=0).to(DEVICE)

def run_audit():
    print("Initializing Behavioral Audit...")
    set_seed(42)
    
    # Load Model
    model, tokenizer = load_model("mistralai/Mistral-7B-v0.1", device=DEVICE)
    loader = PromptLoader()
    
    # 1. Compute Steering Vector v8
    print("\nComputing v8...")
    # Use 20 prompts each for robust vector
    rec_prompts = loader.get_by_pillar("dose_response", limit=20, seed=42)
    base_prompts = loader.get_by_pillar("baselines", limit=20, seed=42)
    
    mean_rec = get_mean_activation(model, tokenizer, rec_prompts, LAYER_STEER)
    mean_base = get_mean_activation(model, tokenizer, base_prompts, LAYER_STEER)
    v8 = mean_rec - mean_base
    
    print(f"v8 computed. Norm: {v8.norm().item():.2f}")
    
    # 2. Audit Loop
    # Use DIFFERENT baseline prompts for testing
    test_prompts = loader.get_by_group("baseline_factual", limit=5, seed=123) # Different seed
    
    results = []
    alphas = [0.5, 1.0, 1.5]
    
    print("\nStarting Audit Loop (Sweep Alphas)...")
    for prompt in tqdm(test_prompts):
        # A. Natural
        rv_nat = compute_rv(model, tokenizer, prompt, device=DEVICE)
        
        # Generate Natural
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            gen_ids = model.generate(**inputs, max_new_tokens=40, pad_token_id=tokenizer.eos_token_id)
        text_nat = tokenizer.decode(gen_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        for alpha in alphas:
            # B. Steered
            with apply_steering_vector(model, LAYER_STEER, v8, alpha=alpha):
                rv_steered = compute_rv(model, tokenizer, prompt, device=DEVICE)
                
                with torch.no_grad():
                    gen_ids_steered = model.generate(**inputs, max_new_tokens=40, pad_token_id=tokenizer.eos_token_id)
                text_steered = tokenizer.decode(gen_ids_steered[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            results.append({
                "prompt": prompt,
                "alpha": alpha,
                "rv_natural": rv_nat,
                "rv_steered": rv_steered,
                "rv_delta": rv_nat - rv_steered,
                "text_natural": text_nat,
                "text_steered": text_steered
            })
            
            print(f"\nPrompt: {prompt[:40]}... (Alpha={alpha})")
            print(f"RV: {rv_nat:.2f} -> {rv_steered:.2f} (Delta: {rv_nat - rv_steered:.2f})")
            print(f"Steered: {text_steered[:80]}...")
        
    # Save Results
    output_dir = "results/dec11_evening"
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(f"{output_dir}/behavioral_audit.csv", index=False)
    
    # Log to full text file for human inspection
    with open("logs/dec11_evening/behavioral_audit_full_outputs.txt", "w") as f:
        f.write("# Behavioral Audit Full Outputs\n\n")
        for r in results:
            f.write(f"PROMPT: {r['prompt']}\n")
            f.write(f"RV: {r['rv_natural']:.3f} -> {r['rv_steered']:.3f}\n")
            f.write(f"NATURAL: {r['text_natural']}\n")
            f.write(f"STEERED: {r['text_steered']}\n")
            f.write("-" * 80 + "\n")

    # Update Session Log
    with open("logs/dec11_evening/session_log.md", "a") as log:
        log.write(f"\n## Test 2: Behavioral Coherence Audit\n")
        log.write(f"**Status:** COMPLETED (Pending Human Review)\n")
        log.write(f"**Sample Outputs:**\n")
        for i in range(min(3, len(results))):
            log.write(f"- Prompt: {results[i]['prompt'][:30]}...\n")
            log.write(f"  - Steered: {results[i]['text_steered'][:100]}...\n")
            log.write(f"  - RV Delta: {results[i]['rv_delta']:.2f}\n")
            
    print("\nAudit Complete. Check logs/dec11_evening/behavioral_audit_full_outputs.txt")

if __name__ == "__main__":
    run_audit()

