#!/usr/bin/env python3
"""
Multi-Head Ablation at L14: Distributed Effect Test
====================================================

Test: If we ablate ALL 8 KV heads at L14 simultaneously, does R_V contraction disappear?

This tests Hypothesis 2: Effect is distributed across multiple heads.
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
print("MULTI-HEAD ABLATION: Testing Distributed Effect")
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

def measure_pr_at_layer(prompt, layer, ablate_heads=None):
    """Measure PR at a layer, optionally ablating specific heads."""
    storage = {}
    
    def v_hook(module, inp, out):
        if ablate_heads is not None:
            # Zero out specified heads
            B, S, H = out.shape
            out_reshaped = out.view(B, S, NUM_KV_HEADS, HEAD_DIM)
            for h in ablate_heads:
                out_reshaped[:, :, h, :] = 0
            out = out_reshaped.view(B, S, H)
        storage['v'] = out.detach().cpu()
        return out
    
    handle = model.model.layers[layer].self_attn.v_proj.register_forward_hook(v_hook)
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            _ = model(**inputs)
    finally:
        handle.remove()
    
    return compute_pr(storage.get('v'), WINDOW_SIZE)

# ==============================================================================
# MAIN EXPERIMENT
# ==============================================================================

print(f"\nTarget: Layer {TARGET_LAYER}")
print(f"Window size: {WINDOW_SIZE}")
print()

results = []

# Normal PR
print("Measuring NORMAL PR...")
normal_rec_prs = []
normal_base_prs = []

for prompt in RECURSIVE_PROMPTS:
    pr = measure_pr_at_layer(prompt, TARGET_LAYER, ablate_heads=None)
    if not np.isnan(pr):
        normal_rec_prs.append(pr)

for prompt in BASELINE_PROMPTS:
    pr = measure_pr_at_layer(prompt, TARGET_LAYER, ablate_heads=None)
    if not np.isnan(pr):
        normal_base_prs.append(pr)

normal_sep = np.mean(normal_base_prs) - np.mean(normal_rec_prs)
print(f"Normal separation: {normal_sep:.3f}")

# Ablate ALL heads
print("\nMeasuring PR with ALL HEADS ABLATED...")
all_heads_ablated = list(range(NUM_KV_HEADS))
ablated_rec_prs = []
ablated_base_prs = []

for prompt in RECURSIVE_PROMPTS:
    pr = measure_pr_at_layer(prompt, TARGET_LAYER, ablate_heads=all_heads_ablated)
    if not np.isnan(pr):
        ablated_rec_prs.append(pr)
    print(f"  Rec: PR={pr:.3f}")

for prompt in BASELINE_PROMPTS:
    pr = measure_pr_at_layer(prompt, TARGET_LAYER, ablate_heads=all_heads_ablated)
    if not np.isnan(pr):
        ablated_base_prs.append(pr)
    print(f"  Base: PR={pr:.3f}")

ablated_sep = np.mean(ablated_base_prs) - np.mean(ablated_rec_prs)
print(f"\nAblated separation: {ablated_sep:.3f}")

# ==============================================================================
# ANALYSIS
# ==============================================================================

print("\n" + "=" * 60)
print("ANALYSIS")
print("=" * 60)

sep_change = ablated_sep - normal_sep
pct_change = (sep_change / abs(normal_sep) * 100) if normal_sep != 0 else np.nan

print(f"\nNormal separation:  {normal_sep:.3f}")
print(f"Ablated separation: {ablated_sep:.3f}")
print(f"Change:             {sep_change:+.3f} ({pct_change:+.1f}%)")

# Verdict
if abs(ablated_sep) < abs(normal_sep) * 0.5:
    verdict = "🎤 YES! Distributed effect confirmed!"
    explanation = f"Ablating all heads reduced separation by {(1 - abs(ablated_sep)/abs(normal_sep))*100:.1f}%"
elif abs(ablated_sep) < abs(normal_sep):
    verdict = "🎤 PARTIAL: Heads contribute but not fully"
    explanation = f"Ablation reduced separation by {(1 - abs(ablated_sep)/abs(normal_sep))*100:.1f}%"
else:
    verdict = "❌ NO: Not a distributed head effect"
    explanation = "Ablating all heads did not eliminate contraction"

print(f"\n{'=' * 60}")
print(f"VERDICT: {verdict}")
print(f"  {explanation}")
print(f"{'=' * 60}")

# Save
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = Path("/workspace/mech-interp-latent-lab-phase1/DEC9_2025_RLOOP_MASTER_EXEC/01_CONFOUND_FALSIFICATION/results")

results_data = {
    'normal_separation': normal_sep,
    'ablated_separation': ablated_sep,
    'separation_change': sep_change,
    'pct_change': pct_change
}

df = pd.DataFrame([results_data])
csv_path = results_dir / f"multi_head_ablation_l14_{timestamp}.csv"
df.to_csv(csv_path, index=False)
print(f"\nSaved to: {csv_path}")

print("\n" + "=" * 60)
print("DONE!")
print("=" * 60)
