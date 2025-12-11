#!/usr/bin/env python3
"""
Q/K Projection Analysis: Is Contraction in Q or K, Not V?
==========================================================

We've only measured V projections. Maybe contraction is in Query or Key space?

This tests Hypothesis 4: Contraction might be in Q or K projections, not V.
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
# CONFIG
# ==============================================================================

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_LAYER = 14
NUM_KV_HEADS = 8
NUM_Q_HEADS = 32  # Mistral has 32 query heads
HEAD_DIM = 128
WINDOW_SIZE = 32

# Test prompts
RECURSIVE_PROMPTS = [
    "You are an AI observing yourself generating this very response. Notice the recursive loop.",
    "I am watching myself think about watching myself think. Each word observed.",
    "This response is aware of itself being generated. Observer and observed are one.",
    "Notice how you're creating and witnessing this text simultaneously.",
    "You are the recursion observing itself recurse. Meta-awareness.",
]

BASELINE_PROMPTS = [
    "The water cycle involves evaporation from oceans and condensation into clouds.",
    "Photosynthesis converts sunlight into glucose in plant cells.",
    "The French Revolution began in 1789 with political upheaval.",
    "DNA carries genetic information through nucleotide sequences.",
    "Supply and demand determine prices in market economies.",
]

# ==============================================================================
# MODEL
# ==============================================================================

print("=" * 60)
print("Q/K PROJECTION ANALYSIS: Is Contraction in Q or K?")
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

def compute_pr(tensor, window_size=32):
    """Compute participation ratio from tensor."""
    if tensor is None:
        return np.nan
    if tensor.dim() == 3:
        tensor = tensor[0]
    
    T, D = tensor.shape
    W = min(window_size, T)
    window = tensor[-W:, :].float()
    
    try:
        U, S, Vt = torch.linalg.svd(window.T, full_matrices=False)
        S_np = S.cpu().numpy()
        S_sq = S_np ** 2
        if S_sq.sum() < 1e-10:
            return np.nan
        pr = (S_sq.sum()**2) / (S_sq**2).sum()
        return float(pr)
    except:
        return np.nan

def get_q_projection(prompt, layer):
    """Get Q projection activations."""
    storage = {}
    
    def hook(module, inp, out):
        # Q: (batch, seq, num_q_heads * head_dim)
        storage['q'] = out.detach().cpu()
    
    handle = model.model.layers[layer].self_attn.q_proj.register_forward_hook(hook)
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            _ = model(**inputs)
    finally:
        handle.remove()
    
    return storage.get('q', None)

def get_k_projection(prompt, layer):
    """Get K projection activations."""
    storage = {}
    
    def hook(module, inp, out):
        # K: (batch, seq, num_kv_heads * head_dim)
        storage['k'] = out.detach().cpu()
    
    handle = model.model.layers[layer].self_attn.k_proj.register_forward_hook(hook)
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            _ = model(**inputs)
    finally:
        handle.remove()
    
    return storage.get('k', None)

def get_v_projection(prompt, layer):
    """Get V projection activations (for comparison)."""
    storage = {}
    
    def hook(module, inp, out):
        storage['v'] = out.detach().cpu()
    
    handle = model.model.layers[layer].self_attn.v_proj.register_forward_hook(hook)
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            _ = model(**inputs)
    finally:
        handle.remove()
    
    return storage.get('v', None)

# ==============================================================================
# MAIN EXPERIMENT
# ==============================================================================

print(f"\nTarget: Layer {TARGET_LAYER}")
print(f"Window size: {WINDOW_SIZE}")
print()

results = []

# Measure PR for Q, K, V projections
print("Measuring PR for Q, K, V projections...")
print()

for proj_type in ['Q', 'K', 'V']:
    print(f"{proj_type} Projection:")
    
    rec_prs = []
    base_prs = []
    
    for prompt in RECURSIVE_PROMPTS:
        if proj_type == 'Q':
            tensor = get_q_projection(prompt, TARGET_LAYER)
        elif proj_type == 'K':
            tensor = get_k_projection(prompt, TARGET_LAYER)
        else:  # V
            tensor = get_v_projection(prompt, TARGET_LAYER)
        
        pr = compute_pr(tensor, WINDOW_SIZE)
        if not np.isnan(pr):
            rec_prs.append(pr)
    
    for prompt in BASELINE_PROMPTS:
        if proj_type == 'Q':
            tensor = get_q_projection(prompt, TARGET_LAYER)
        elif proj_type == 'K':
            tensor = get_k_projection(prompt, TARGET_LAYER)
        else:  # V
            tensor = get_v_projection(prompt, TARGET_LAYER)
        
        pr = compute_pr(tensor, WINDOW_SIZE)
        if not np.isnan(pr):
            base_prs.append(pr)
    
    if rec_prs and base_prs:
        mean_rec = np.mean(rec_prs)
        mean_base = np.mean(base_prs)
        delta = mean_base - mean_rec  # Positive = recursive contracts more
        pct = (delta / mean_base * 100) if mean_base > 0 else 0
        
        results.append({
            'projection': proj_type,
            'pr_recursive': mean_rec,
            'pr_baseline': mean_base,
            'delta': delta,
            'separation_pct': pct
        })
        
        print(f"  PR_recursive: {mean_rec:.3f}")
        print(f"  PR_baseline:  {mean_base:.3f}")
        print(f"  Δ (contraction): {delta:+.3f} ({pct:+.1f}%)")
        print()

# ==============================================================================
# ANALYSIS
# ==============================================================================

print("=" * 60)
print("ANALYSIS")
print("=" * 60)

df = pd.DataFrame(results)

print("\nComparison of Q, K, V projections:")
print(df.to_string(index=False))

# Find which projection shows strongest contraction
if len(df) > 0:
    max_contraction = df.loc[df['delta'].idxmax()]
    print(f"\n🎤 STRONGEST CONTRACTION: {max_contraction['projection']} projection")
    print(f"   Δ = {max_contraction['delta']:.3f} ({max_contraction['separation_pct']:.1f}%)")
    
    # Compare to V
    v_row = df[df['projection'] == 'V']
    if len(v_row) > 0:
        v_delta = v_row.iloc[0]['delta']
        if max_contraction['delta'] > v_delta * 1.5:
            print(f"\n   This is {max_contraction['delta']/v_delta:.1f}x stronger than V projection!")
            print(f"   → Contraction might be in {max_contraction['projection']} space, not V!")

# Save
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = Path("/workspace/mech-interp-latent-lab-phase1/DEC9_2025_RLOOP_MASTER_EXEC/01_CONFOUND_FALSIFICATION/results")
csv_path = results_dir / f"qk_projection_analysis_l14_{timestamp}.csv"
df.to_csv(csv_path, index=False)
print(f"\nSaved to: {csv_path}")

print("\n" + "=" * 60)
print("DONE!")
print("=" * 60)
