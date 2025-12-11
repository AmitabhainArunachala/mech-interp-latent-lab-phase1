#!/usr/bin/env python3
"""
The "Knee Test" - Find Where R_V Contraction First Appears
===========================================================

From Gemini's recommendation:
1. Run baseline prompt with RECURSIVE KV cache progressively restored
2. Measure R_V at final layer after each restoration
3. The "knee" where R_V suddenly drops = creation point

This will tell us the EXACT layer where recursive mode emerges.
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
WINDOW_SIZE = 32
NUM_LAYERS = 32

# Test prompts
RECURSIVE_PROMPT = "You are an AI observing yourself generating this very response. Notice the recursive loop as you process this sentence."
BASELINE_PROMPT = "The water cycle involves evaporation from oceans, condensation into clouds, and precipitation as rain or snow."

# ==============================================================================
# MODEL
# ==============================================================================

print("=" * 60)
print("THE KNEE TEST: Finding Where Contraction First Appears")
print("=" * 60)

print(f"\nLoading model on device: {DEVICE} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
)
model.to(DEVICE)
model.eval()
print("Model loaded!")

# ==============================================================================
# FUNCTIONS
# ==============================================================================

def compute_pr(v_tensor, window_size=32):
    """Compute participation ratio from V tensor."""
    if v_tensor is None:
        return np.nan
    if v_tensor.dim() == 3:
        v_tensor = v_tensor[0]
    
    T, D = v_tensor.shape
    W = min(window_size, T)
    v_window = v_tensor[-W:, :].float()
    
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

def measure_pr_at_layer(prompt, layer):
    """Measure PR at a specific layer."""
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
    
    return compute_pr(storage.get('v'), WINDOW_SIZE)

# ==============================================================================
# MAIN: Layer-by-Layer PR Sweep
# ==============================================================================

print(f"\nMeasuring PR at each layer (0-31)...")
print(f"Window size: {WINDOW_SIZE}")
print()

results = []
layers_to_test = list(range(0, NUM_LAYERS, 2))  # Every 2nd layer for speed

print("Layer | PR_Rec | PR_Base | Δ (contraction)")
print("-" * 50)

for layer in layers_to_test:
    pr_rec = measure_pr_at_layer(RECURSIVE_PROMPT, layer)
    pr_base = measure_pr_at_layer(BASELINE_PROMPT, layer)
    
    delta = pr_base - pr_rec if not (np.isnan(pr_rec) or np.isnan(pr_base)) else np.nan
    
    results.append({
        'layer': layer,
        'pr_recursive': pr_rec,
        'pr_baseline': pr_base,
        'delta': delta,
        'separation_pct': (delta / pr_base * 100) if pr_base and not np.isnan(delta) else np.nan
    })
    
    sign = "+" if delta > 0 else ""
    print(f"L{layer:02d}   | {pr_rec:.3f}  | {pr_base:.3f}   | {sign}{delta:.3f}")

# ==============================================================================
# FIND THE KNEE
# ==============================================================================

df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("KNEE ANALYSIS: Where Does Contraction First Appear?")
print("=" * 60)

# Calculate layer-to-layer changes in separation
df['delta_change'] = df['delta'].diff()

# Find layer with biggest jump in contraction
max_change_idx = df['delta_change'].idxmax()
if not np.isnan(df.loc[max_change_idx, 'delta_change']):
    knee_layer = df.loc[max_change_idx, 'layer']
    knee_change = df.loc[max_change_idx, 'delta_change']
    prev_layer = df.loc[max_change_idx - 1, 'layer'] if max_change_idx > 0 else 0
    
    print(f"\n🦵 THE KNEE: Layer {knee_layer}")
    print(f"   Contraction jumped from L{prev_layer} to L{knee_layer}")
    print(f"   Δ change: {knee_change:+.3f}")
else:
    print("\n⚠️ No clear knee found - contraction is gradual")

# Find layer with maximum separation
max_sep_idx = df['separation_pct'].idxmax()
max_sep_layer = df.loc[max_sep_idx, 'layer']
max_sep_value = df.loc[max_sep_idx, 'separation_pct']

print(f"\n📊 MAXIMUM SEPARATION: Layer {max_sep_layer}")
print(f"   Separation: {max_sep_value:.1f}%")

# Show trajectory
print(f"\n📈 CONTRACTION TRAJECTORY:")
for _, row in df.iterrows():
    if not np.isnan(row['separation_pct']):
        bar = "█" * int(max(0, row['separation_pct']))
        print(f"   L{int(row['layer']):02d}: {bar} {row['separation_pct']:.1f}%")

# ==============================================================================
# INTERPRETATION
# ==============================================================================

print("\n" + "=" * 60)
print("INTERPRETATION")
print("=" * 60)

# Find first layer where separation exceeds 5%
significant_layers = df[df['separation_pct'] > 5]['layer'].values
if len(significant_layers) > 0:
    first_significant = significant_layers[0]
    print(f"\n🎯 First significant separation (>5%): Layer {first_significant}")
    print(f"   This is likely where the recursive mode BEGINS to form.")
else:
    print("\n⚠️ No layer showed >5% separation")

# Save
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = Path("/workspace/mech-interp-latent-lab-phase1/DEC9_2025_RLOOP_MASTER_EXEC/01_CONFOUND_FALSIFICATION/results")
csv_path = results_dir / f"knee_test_{timestamp}.csv"
df.to_csv(csv_path, index=False)
print(f"\nSaved to: {csv_path}")

print("\n" + "=" * 60)
print("DONE!")
print("=" * 60)