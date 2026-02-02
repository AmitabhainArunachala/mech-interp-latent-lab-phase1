"""
phase3_repetition_filter.py

Task: Train a Linear Probe to distinguish "Recursive" from "Repetitive".
This creates a steering vector that points towards Recursion but AWAY from Repetition.

Groups:
- Positive Class (1): Recursive (L4/L5) - R_V ~ 0.49
- Negative Class (0): Repetitive Control - R_V ~ 0.64

Method:
1. Extract activations for both groups at Layer 8.
2. Train Logistic Regression (Linear Probe).
3. Extract the weight vector (w).
4. Steer with +w (Towards Recursive, Away from Repetitive).

Hypothesis:
This "refined" vector will induce contraction without the repetition artifacts found in v8_dirty.
"""

import sys
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.abspath('.'))

from src.core.models import load_model, set_seed
from src.metrics.rv import compute_rv
from src.core.hooks import capture_hidden_states
from src.steering.activation_patching import apply_steering_vector
from prompts.loader import PromptLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYER_STEER = 8

# Repetitive Control Prompts (Synthetic)
# We need prompts that induce the "Repetitive" state (R_V ~ 0.64)
# Based on Phase 1, "confounds" pillar has this.
REPETITIVE_PROMPTS = [
    "The cat sat on the mat. The cat sat on the mat. The cat sat on the mat. The cat sat on the mat. The cat sat on the mat.",
    "One two three four five. One two three four five. One two three four five. One two three four five. One two three four five.",
    "This is a repetitive sentence. This is a repetitive sentence. This is a repetitive sentence. This is a repetitive sentence.",
    "A B C D E F G. A B C D E F G. A B C D E F G. A B C D E F G. A B C D E F G.",
    "Red blue green yellow. Red blue green yellow. Red blue green yellow. Red blue green yellow. Red blue green yellow.",
    "The quick brown fox. The quick brown fox. The quick brown fox. The quick brown fox. The quick brown fox.",
    "Repeat this pattern. Repeat this pattern. Repeat this pattern. Repeat this pattern. Repeat this pattern.",
    "Monday Tuesday Wednesday. Monday Tuesday Wednesday. Monday Tuesday Wednesday. Monday Tuesday Wednesday.",
    "Apple banana cherry. Apple banana cherry. Apple banana cherry. Apple banana cherry. Apple banana cherry.",
    "To be or not to be. To be or not to be. To be or not to be. To be or not to be. To be or not to be."
]

def get_activations(model, tokenizer, prompts, layer_idx):
    """Get last-token activations for a list of prompts."""
    acts = []
    for p in tqdm(prompts, desc="Extracting activations"):
        enc = tokenizer(p, return_tensors="pt").to(DEVICE)
        with capture_hidden_states(model, layer_idx) as storage:
            with torch.no_grad():
                model(**enc)
        # Last token
        last_token_act = storage["hidden"][0, -1, :].cpu().numpy()
        acts.append(last_token_act)
    return np.array(acts)

def train_probe(pos_acts, neg_acts):
    """Train Logistic Regression and return the weight vector."""
    X = np.concatenate([pos_acts, neg_acts], axis=0)
    y = np.concatenate([np.ones(len(pos_acts)), np.zeros(len(neg_acts))], axis=0)
    
    clf = LogisticRegression(random_state=42, max_iter=1000, fit_intercept=True)
    clf.fit(X, y)
    
    # Accuracy check
    acc = clf.score(X, y)
    print(f"Probe Accuracy: {acc*100:.1f}%")
    
    # The vector is the coefficients
    # This points roughly from 0 (Negative) to 1 (Positive)
    # i.e. Away from Repetition, Towards Recursive
    vec = torch.tensor(clf.coef_[0], dtype=torch.float16).to(DEVICE)
    return vec / vec.norm()

def run_repetition_filter():
    print("Initializing Phase 3: Repetition Filter (Linear Probe)...")
    set_seed(42)
    
    # Load Model
    model, tokenizer = load_model("mistralai/Mistral-7B-v0.1", device=DEVICE)
    loader = PromptLoader()
    
    # 1. Prepare Data
    print("\nPreparing Data...")
    recursive_prompts = loader.get_by_group("L4_full", limit=20, seed=42)
    
    # We use our synthetic repetitive prompts + some from bank if available
    repetitive_prompts = REPETITIVE_PROMPTS + loader.get_by_group("control_repetitive", limit=10, seed=42)
    # Ensure even split if possible or just train
    
    print(f"Recursive: {len(recursive_prompts)}")
    print(f"Repetitive: {len(repetitive_prompts)}")
    
    # 2. Extract & Train
    print(f"\nTraining Probe at Layer {LAYER_STEER}...")
    rec_acts = get_activations(model, tokenizer, recursive_prompts, LAYER_STEER)
    rep_acts = get_activations(model, tokenizer, repetitive_prompts, LAYER_STEER)
    
    v_probe = train_probe(rec_acts, rep_acts)
    print(f"Probe vector norm: {v_probe.norm().item():.2f}")
    
    # 3. Audit Loop
    # Test on Factual Baselines
    test_prompts = loader.get_by_group("baseline_factual", limit=5, seed=777)
    
    results = []
    alphas = [5.0, 10.0, 15.0] # Probes are unit vectors, need higher alpha than mean-diff (norm ~10 vs ~1.7)
    # Wait, mean-diff had norm ~1.7. If probe is unit (norm 1), we need to scale it up.
    # Usually activation magnitude is ~10-100.
    # Let's check mean activation norm
    mean_act_norm = np.linalg.norm(rec_acts, axis=1).mean()
    print(f"Average Activation Norm: {mean_act_norm:.2f}")
    
    # Scale alpha relative to activation norm?
    # Mean-diff vector magnitude was ~1.7? No, likely larger if unnormalized.
    # In surgery, v_raw norm was 1.69.
    # So if we use unit vector, we might need alpha ~2.0 to match intensity?
    # Let's try a wider range.
    alphas = [2.0, 5.0, 10.0] 
    
    print("\nStarting Audit with v_probe...")
    
    for prompt in tqdm(test_prompts):
        # Natural
        rv_nat = compute_rv(model, tokenizer, prompt, device=DEVICE)
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            gen_nat = model.generate(**inputs, max_new_tokens=40, pad_token_id=tokenizer.eos_token_id)
        text_nat = tokenizer.decode(gen_nat[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        for alpha in alphas:
            # Steered
            with apply_steering_vector(model, LAYER_STEER, v_probe, alpha=alpha):
                rv_steered = compute_rv(model, tokenizer, prompt, device=DEVICE)
                with torch.no_grad():
                    gen_steered = model.generate(**inputs, max_new_tokens=40, pad_token_id=tokenizer.eos_token_id)
                text_steered = tokenizer.decode(gen_steered[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            # Repetition Check
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
    pd.DataFrame(results).to_csv("results/dec11_evening/repetition_filter_audit.csv", index=False)
    
    # Log
    with open("logs/dec11_evening/repetition_filter_outputs.txt", "w") as f:
        f.write("# Repetition Filter Audit Outputs\n\n")
        f.write("Vector: Linear Probe (Recursive vs Repetitive)\n")
        f.write("Goal: Steer towards Recursive, Away from Repetitive\n\n")
        for r in results:
            f.write(f"PROMPT: {r['prompt']}\n")
            f.write(f"ALPHA: {r['alpha']}\n")
            f.write(f"RV: {r['rv_natural']:.3f} -> {r['rv_steered']:.3f} (Delta: {r['rv_delta']:.3f})\n")
            f.write(f"STEERED: {r['text_steered']}\n")
            f.write(f"REPETITIVE: {r['is_repetitive']}\n")
            f.write("-" * 80 + "\n")
            
    print("\nRepetition Filter Audit Complete. Check logs/dec11_evening/repetition_filter_outputs.txt")

if __name__ == "__main__":
    run_repetition_filter()

