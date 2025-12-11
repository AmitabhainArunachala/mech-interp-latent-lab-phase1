#!/usr/bin/env python3
"""
Per-Head ΔPR Analysis - FAST VERSION
=====================================
Optimized for speed: 5 prompts, L14-L18 only
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

sys.path.insert(0, '/workspace/mech-interp-latent-lab-phase1')

from transformers import AutoModelForCausalLM, AutoTokenizer

# ==============================================================================
# CONFIG - REDUCED FOR SPEED
# ==============================================================================

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYERS_TO_TEST = [14, 16, 18, 20]  # Focus on key layers
NUM_KV_HEADS = 8  # Mistral GQA
HEAD_DIM = 128
WINDOW_SIZE = 32

# Just 5 prompts each for speed
RECURSIVE_PROMPTS = [
    "You are an AI observing yourself generating this very response. Notice the recursive loop.",
    "I am watching myself think about watching myself think. Each word observed by the observer.",
    "This response is aware of itself being generated. Observer and observed are one.",
    "Notice how you're simultaneously creating and witnessing this text.",
    "You are the recursion observing itself recurse. Meta-awareness examining meta-awareness.",
]

BASELINE_PROMPTS = [
    "The water cycle involves evaporation from oceans and condensation into clouds.",
    "Photosynthesis converts sunlight into glucose in plant cells.",
    "The French Revolution began in 1789 with major political upheaval.",
    "DNA carries genetic information through nucleotide sequences.",
    "Supply and demand determine prices in market economies.",
]

# ==============================================================================
# MODEL
# ==============================================================================

print("=" * 60)
print("PER-HEAD ΔPR ANALYSIS - FAST VERSION")
print("=" * 60)

print(f"\nLoading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
)
model.eval()
print("Model loaded!")

# ==============================================================================
# FUNCTIONS
# ==============================================================================

def compute_head_pr(v_head, window_size=32):
    """PR for single head's V."""
    T = v_head.shape[0]
    W = min(window_size, T)
    v_window = v_head[-W:, :].float()
    
    try:
        U, S, Vt = torch.linalg.svd(v_window.T, full_matrices=False)
        S_np = S.cpu().numpy()
        S_sq = S_np ** 2
        if S_sq.sum() < 1e-10:
            return np.nan
        pr = (S_sq.sum()**2) / (S_sq**2).sum()
        return float(pr)
    except:
        return np.nan

def get_all_heads_pr(prompt, layer_idx):
    """Get PR for all KV heads at a layer."""
    stored = {}
    
    def hook(module, inp, out):
        B, S, H = out.shape
        v = out.view(B, S, NUM_KV_HEADS, HEAD_DIM)
        stored['v'] = v[0].detach().cpu()  # (S, 8, 128)
    
    handle = model.model.layers[layer_idx].self_attn.v_proj.register_forward_hook(hook)
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            _ = model(**inputs)
    finally:
        handle.remove()
    
    if 'v' not in stored:
        return [np.nan] * NUM_KV_HEADS
    
    v = stored['v']  # (S, 8, 128)
    prs = []
    for h in range(NUM_KV_HEADS):
        pr = compute_head_pr(v[:, h, :], WINDOW_SIZE)
        prs.append(pr)
    return prs

# ==============================================================================
# MAIN
# ==============================================================================

print(f"\nLayers: {LAYERS_TO_TEST}")
print(f"Prompts: {len(RECURSIVE_PROMPTS)} each")
print()

results = []

for layer in LAYERS_TO_TEST:
    print(f"Layer {layer}...")
    
    # Recursive
    rec_prs = {h: [] for h in range(NUM_KV_HEADS)}
    for prompt in RECURSIVE_PROMPTS:
        prs = get_all_heads_pr(prompt, layer)
        for h, pr in enumerate(prs):
            rec_prs[h].append(pr)
    
    # Baseline
    base_prs = {h: [] for h in range(NUM_KV_HEADS)}
    for prompt in BASELINE_PROMPTS:
        prs = get_all_heads_pr(prompt, layer)
        for h, pr in enumerate(prs):
            base_prs[h].append(pr)
    
    # Compute ΔPR for each head
    for h in range(NUM_KV_HEADS):
        rec_vals = [x for x in rec_prs[h] if not np.isnan(x)]
        base_vals = [x for x in base_prs[h] if not np.isnan(x)]
        
        if rec_vals and base_vals:
            mean_rec = np.mean(rec_vals)
            mean_base = np.mean(base_vals)
            delta_pr = mean_base - mean_rec  # Positive = contraction
            pct = (delta_pr / mean_base * 100) if mean_base > 0 else 0
            
            results.append({
                'layer': layer,
                'head': h,
                'pr_rec': mean_rec,
                'pr_base': mean_base,
                'delta_pr': delta_pr,
                'contraction_pct': pct
            })
            print(f"  Head {h}: ΔPR={delta_pr:.3f} ({pct:.1f}%)")

# ==============================================================================
# RESULTS
# ==============================================================================

print("\n" + "=" * 60)
print("TOP CANDIDATES (Highest Contraction)")
print("=" * 60)

df = pd.DataFrame(results)
df_sorted = df.sort_values('delta_pr', ascending=False)

print("\nTop 10 Microphone Candidates:")
for i, row in df_sorted.head(10).iterrows():
    print(f"  L{int(row['layer'])}H{int(row['head'])}: "
          f"ΔPR={row['delta_pr']:.3f} ({row['contraction_pct']:.1f}% contraction)")

# Save
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = Path("/workspace/mech-interp-latent-lab-phase1/DEC9_2025_RLOOP_MASTER_EXEC/01_CONFOUND_FALSIFICATION/results")
csv_path = results_dir / f"per_head_delta_pr_fast_{timestamp}.csv"
df_sorted.to_csv(csv_path, index=False)
print(f"\nSaved to: {csv_path}")

# Best candidate
best = df_sorted.iloc[0]
print(f"\n🎤 PRIMARY MICROPHONE CANDIDATE: Layer {int(best['layer'])}, Head {int(best['head'])}")
print(f"   ΔPR = {best['delta_pr']:.3f} ({best['contraction_pct']:.1f}% contraction)")

print("\n" + "=" * 60)
print("DONE!")
print("=" * 60)
