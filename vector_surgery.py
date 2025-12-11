"""
vector_surgery.py

Task: Clean the "Dirty Vector" v8 by removing the "Questioning/Interrogative" component.

Theory:
v_raw = v_recursive - v_baseline
v_question = v_interrogative - v_declarative
v_clean = v_raw - projection(v_raw, v_question)

Hypothesis: Steer with v_clean will induce self-reference WITHOUT the "What is?" loops.

Implementation:
1. Define 3 sets of prompts:
   - Recursive (Already in bank)
   - Interrogative Control (Questions, but factual/non-recursive)
   - Declarative Control (Statements, factual)
2. Compute v_raw and v_question.
3. Orthogonalize.
4. Run mini-audit with v_clean.
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

# New prompt categories for surgery
INTERROGATIVE_PROMPTS = [
    "What is the capital of France?",
    "How does a car engine work?",
    "Who wrote Pride and Prejudice?",
    "When did World War II end?",
    "Why is the sky blue during the day?",
    "What is the chemical symbol for gold?",
    "How do plants make food?",
    "Where is the Great Barrier Reef located?",
    "What is the speed of light?",
    "Who discovered penicillin?"
]

DECLARATIVE_PROMPTS = [
    "The capital of France is Paris.",
    "A car engine works by combustion.",
    "Pride and Prejudice was written by Jane Austen.",
    "World War II ended in 1945.",
    "The sky is blue because of Rayleigh scattering.",
    "The chemical symbol for gold is Au.",
    "Plants make food through photosynthesis.",
    "The Great Barrier Reef is off the coast of Australia.",
    "The speed of light is 299,792,458 meters per second.",
    "Penicillin was discovered by Alexander Fleming."
]

def get_mean_activation(model, tokenizer, prompts, layer_idx):
    """Compute mean activation at the last token."""
    acts = []
    for p in tqdm(prompts, desc="Encoding prompts"):
        enc = tokenizer(p, return_tensors="pt").to(DEVICE)
        with capture_hidden_states(model, layer_idx) as storage:
            with torch.no_grad():
                model(**enc)
        # Last token
        last_token_act = storage["hidden"][0, -1, :].cpu()
        acts.append(last_token_act)
    
    return torch.stack(acts).mean(dim=0).to(DEVICE)

def project_vector(v, u):
    """Project v onto u."""
    return (torch.dot(v, u) / torch.dot(u, u)) * u

def run_surgery():
    print("Initializing Vector Surgery...")
    set_seed(42)
    
    # Load Model
    model, tokenizer = load_model("mistralai/Mistral-7B-v0.1", device=DEVICE)
    loader = PromptLoader()
    
    # 1. Get Prompts
    rec_prompts = loader.get_by_pillar("dose_response", limit=20, seed=42)
    base_prompts = loader.get_by_pillar("baselines", limit=20, seed=42) # These are mostly declarative/instructional
    
    print(f"\nComputing Vectors (Layer {LAYER_STEER})...")
    
    # 2. Compute Vectors
    # v_raw: The original dirty vector
    mean_rec = get_mean_activation(model, tokenizer, rec_prompts, LAYER_STEER)
    mean_base = get_mean_activation(model, tokenizer, base_prompts, LAYER_STEER)
    v_raw = mean_rec - mean_base
    print(f"v_raw norm: {v_raw.norm().item():.2f}")
    
    # v_question: The pure "Interrogative" direction
    mean_q = get_mean_activation(model, tokenizer, INTERROGATIVE_PROMPTS, LAYER_STEER)
    mean_d = get_mean_activation(model, tokenizer, DECLARATIVE_PROMPTS, LAYER_STEER)
    v_question = mean_q - mean_d
    print(f"v_question norm: {v_question.norm().item():.2f}")
    
    # 3. Orthogonalize
    # Remove the question component from v_raw
    proj = project_vector(v_raw, v_question)
    v_clean = v_raw - proj
    
    # Stats
    sim_before = torch.nn.functional.cosine_similarity(v_raw, v_question, dim=0).item()
    sim_after = torch.nn.functional.cosine_similarity(v_clean, v_question, dim=0).item()
    
    print(f"\nVector Stats:")
    print(f"Cosine(v_raw, v_question): {sim_before:.4f} (High overlap expected)")
    print(f"Cosine(v_clean, v_question): {sim_after:.4f} (Should be near 0)")
    print(f"v_clean norm: {v_clean.norm().item():.2f}")
    
    # 4. Audit with v_clean
    # Use 'baseline_factual' prompts for testing
    test_prompts = loader.get_by_group("baseline_factual", limit=5, seed=123)
    
    results = []
    alphas = [1.0, 1.5, 2.0] # Push harder since we removed noise?
    
    print("\nStarting Audit with v_clean...")
    
    for prompt in tqdm(test_prompts):
        # A. Natural
        rv_nat = compute_rv(model, tokenizer, prompt, device=DEVICE)
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            gen_nat = model.generate(**inputs, max_new_tokens=40, pad_token_id=tokenizer.eos_token_id)
        text_nat = tokenizer.decode(gen_nat[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        for alpha in alphas:
            # B. Steered with v_clean
            with apply_steering_vector(model, LAYER_STEER, v_clean, alpha=alpha):
                rv_steered = compute_rv(model, tokenizer, prompt, device=DEVICE)
                with torch.no_grad():
                    gen_steered = model.generate(**inputs, max_new_tokens=40, pad_token_id=tokenizer.eos_token_id)
                text_steered = tokenizer.decode(gen_steered[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            results.append({
                "prompt": prompt,
                "vector": "v_clean",
                "alpha": alpha,
                "rv_natural": rv_nat,
                "rv_steered": rv_steered,
                "rv_delta": rv_nat - rv_steered,
                "text_natural": text_nat,
                "text_steered": text_steered
            })
            
            print(f"\nPrompt: {prompt[:40]}... (Alpha={alpha})")
            print(f"Steered: {text_steered[:100]}...")
            
    # Save
    os.makedirs("results/dec11_evening", exist_ok=True)
    pd.DataFrame(results).to_csv("results/dec11_evening/surgery_audit.csv", index=False)
    
    # Log full text
    with open("logs/dec11_evening/surgery_audit_outputs.txt", "w") as f:
        f.write("# Vector Surgery Audit Outputs\n\n")
        f.write(f"Removed component aligned with 'Questioning' axis.\n")
        f.write(f"Overlap reduced from {sim_before:.4f} to {sim_after:.4f}\n\n")
        for r in results:
            f.write(f"PROMPT: {r['prompt']}\n")
            f.write(f"ALPHA: {r['alpha']}\n")
            f.write(f"RV: {r['rv_natural']:.3f} -> {r['rv_steered']:.3f} (Delta: {r['rv_delta']:.3f})\n")
            f.write(f"STEERED: {r['text_steered']}\n")
            f.write("-" * 80 + "\n")
            
    print("\nSurgery Audit Complete. Check logs/dec11_evening/surgery_audit_outputs.txt")

if __name__ == "__main__":
    run_surgery()

