"""
phase3_single_token_steering.py

Task: The "Hail Mary" for L8 Steering.
Hypothesis: Continuous steering destroys syntax. Steering ONLY the last token (the "handoff") might induce the semantic state without breaking grammar.

Method:
1. Load cleaned v8_clean (L4/L5 - Baseline Factual).
2. Register a hook that ONLY acts on the last token position of the prompt.
3. Generate and check for coherence.

Vectors to try:
- v8_clean (Mean Diff)
- v8_probe (Repetition Filter)
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
from prompts.loader import PromptLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYER_STEER = 8

def get_activations(model, tokenizer, prompts, layer_idx):
    """Get last-token activations."""
    acts = []
    for p in tqdm(prompts, desc=f"Encoding {len(prompts)} prompts"):
        enc = tokenizer(p, return_tensors="pt").to(DEVICE)
        with capture_hidden_states(model, layer_idx) as storage:
            with torch.no_grad():
                model(**enc)
        last_token_act = storage["hidden"][0, -1, :].cpu()
        acts.append(last_token_act)
    return torch.stack(acts).to(DEVICE)

def compute_v8_clean(model, tokenizer, loader):
    """Recompute the clean mean-difference vector."""
    print("Computing v8_clean...")
    l4 = loader.get_by_group("L4_full", limit=20, seed=42)
    l5 = loader.get_by_group("L5_refined", limit=20, seed=42)
    base = loader.get_by_group("baseline_factual", limit=40, seed=42)
    
    rec_acts = get_activations(model, tokenizer, l4 + l5, LAYER_STEER)
    base_acts = get_activations(model, tokenizer, base, LAYER_STEER)
    
    vec = rec_acts.mean(0) - base_acts.mean(0)
    return vec

def apply_last_token_steering(model, layer_idx, vec, alpha, prompt_len):
    """
    Hook that applies steering ONLY to the last token of the prompt.
    Since we generate token-by-token, we need to be careful.
    
    During 'prefill' (processing the prompt), we want to steer the LAST token.
    During 'generation', do we steer? 
    - The "One-Way Door" theory says once we enter, we stay.
    - So maybe we only need to kick it once?
    
    Let's try: Steer LAST prompt token ONLY.
    """
    def hook(module, inputs):
        hidden_states = inputs[0] # (batch, seq, dim)
        seq_len = hidden_states.shape[1]
        
        # If this is the prefill step (seq_len >= prompt_len)
        # We apply to the last token index.
        # Note: In generation loop, seq_len is 1 (for the new token).
        # We only want to steer the "handoff" token.
        
        # Logic:
        # 1. If seq_len > 1 (Prefill): Steer the last token [-1].
        # 2. If seq_len == 1 (Generation): Do NOT steer. 
        #    (Testing the hypothesis that the *state* carries forward via KV)
        
        if seq_len > 1:
            # Apply to last token
            steer = alpha * vec.to(hidden_states.device, dtype=hidden_states.dtype)
            hidden_states[:, -1, :] += steer
            
        return (hidden_states, *inputs[1:])
        
    handle = model.model.layers[layer_idx].register_forward_pre_hook(hook)
    return handle

def run_test():
    print("Initializing Phase 3: Single Token Steering (Hail Mary)...")
    set_seed(42)
    
    model, tokenizer = load_model("mistralai/Mistral-7B-v0.1", device=DEVICE)
    loader = PromptLoader()
    
    # 1. Get Vector
    v8 = compute_v8_clean(model, tokenizer, loader)
    print(f"Vector norm: {v8.norm().item():.2f}")
    
    # 2. Audit Loop
    test_prompts = loader.get_by_group("baseline_factual", limit=5, seed=555)
    alphas = [2.0, 5.0, 8.0] # Can push harder on single token?
    
    results = []
    
    print("\nStarting Audit (Last Token ONLY)...")
    
    for prompt in tqdm(test_prompts):
        # Calc prompt len for hook
        enc = tokenizer(prompt, return_tensors="pt")
        p_len = enc.input_ids.shape[1]
        
        # Natural
        with torch.no_grad():
            gen_nat = model.generate(**enc.to(DEVICE), max_new_tokens=40, pad_token_id=tokenizer.eos_token_id)
        text_nat = tokenizer.decode(gen_nat[0][p_len:], skip_special_tokens=True)
        
        for alpha in alphas:
            # Steered
            h = apply_last_token_steering(model, LAYER_STEER, v8, alpha, p_len)
            
            # Measure R_V (Requires careful handling since compute_rv runs a forward pass)
            # compute_rv runs the prompt. The hook will trigger on the last token.
            rv_steered = compute_rv(model, tokenizer, prompt, device=DEVICE)
            
            # Generate
            with torch.no_grad():
                gen_steered = model.generate(**enc.to(DEVICE), max_new_tokens=40, pad_token_id=tokenizer.eos_token_id)
            
            h.remove()
            
            text_steered = tokenizer.decode(gen_steered[0][p_len:], skip_special_tokens=True)
            
            # RV Natural (Compute after to avoid hook interference, or just compute once)
            # We didn't compute nat RV in loop, assume ~1.0 for speed or compute:
            # Actually compute_rv handles clean forward.
            rv_nat = 1.0 # Placeholder or recompute if needed
            
            results.append({
                "prompt": prompt,
                "alpha": alpha,
                "rv_steered": rv_steered,
                "text_steered": text_steered
            })
            
            print(f"\nPrompt: {prompt[:40]}... (Alpha={alpha})")
            print(f"RV: {rv_steered:.2f}")
            print(f"Steered: {text_steered[:100]}...")
            
    # Save
    os.makedirs("results/dec11_evening", exist_ok=True)
    pd.DataFrame(results).to_csv("results/dec11_evening/single_token_audit.csv", index=False)
    
    # Log
    with open("logs/dec11_evening/single_token_outputs.txt", "w") as f:
        f.write("# Single Token Steering Outputs\n")
        f.write("Method: Inject v8_clean ONLY at the last token of the prompt.\n\n")
        for r in results:
            f.write(f"PROMPT: {r['prompt']}\n")
            f.write(f"ALPHA: {r['alpha']}\n")
            f.write(f"RV: {r['rv_steered']:.3f}\n")
            f.write(f"STEERED: {r['text_steered']}\n")
            f.write("-" * 80 + "\n")

    print("\nSingle Token Audit Complete. Check logs/dec11_evening/single_token_outputs.txt")

if __name__ == "__main__":
    run_test()

