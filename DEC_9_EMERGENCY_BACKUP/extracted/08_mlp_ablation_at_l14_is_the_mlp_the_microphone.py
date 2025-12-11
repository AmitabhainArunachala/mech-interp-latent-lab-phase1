#!/usr/bin/env python3
"""
MLP Ablation at L14: Is the MLP the Microphone?
================================================

Test: If we ablate the MLP at L14, does R_V contraction disappear?

This tests Hypothesis 1: MLP is the contractor (not attention heads).
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
TARGET_LAYER = 14  # The microphone layer
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
print("MLP ABLATION TEST: Is L14 MLP the Microphone?")
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

def measure_pr_at_layer(prompt, layer, ablate_mlp=False):
    """Measure PR at a specific layer, optionally ablating MLP."""
    storage = {}
    
    def v_hook(module, inp, out):
        storage['v'] = out.detach().cpu()
    
    def mlp_hook(module, inp, out):
        # Zero out MLP output
        return torch.zeros_like(out)
    
    v_handle = model.model.layers[layer].self_attn.v_proj.register_forward_hook(v_hook)
    
    mlp_handle = None
    if ablate_mlp:
        mlp_handle = model.model.layers[layer].mlp.register_forward_hook(mlp_hook)
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            _ = model(**inputs)
    finally:
        v_handle.remove()
        if mlp_handle:
            mlp_handle.remove()
    
    return compute_pr(storage.get('v'), WINDOW_SIZE)

# ==============================================================================
# MAIN EXPERIMENT
# ==============================================================================

print(f"\nTarget: Layer {TARGET_LAYER} MLP")
print(f"Window size: {WINDOW_SIZE}")
print()

results = []

# Measure normal PR
print("Measuring NORMAL PR (no ablation)...")
normal_rec_prs = []
normal_base_prs = []

for prompt in RECURSIVE_PROMPTS:
    pr = measure_pr_at_layer(prompt, TARGET_LAYER, ablate_mlp=False)
    if not np.isnan(pr):
        normal_rec_prs.append(pr)
    print(f"  Rec: PR={pr:.3f}")

for prompt in BASELINE_PROMPTS:
    pr = measure_pr_at_layer(prompt, TARGET_LAYER, ablate_mlp=False)
    if not np.isnan(pr):
        normal_base_prs.append(pr)
    print(f"  Base: PR={pr:.3f}")

normal_sep = np.mean(normal_base_prs) - np.mean(normal_rec_prs)
print(f"\nNormal separation: {normal_sep:.3f}")

# Measure with MLP ablated
print("\nMeasuring PR with MLP ABLATED...")
ablated_rec_prs = []
ablated_base_prs = []

for prompt in RECURSIVE_PROMPTS:
    pr = measure_pr_at_layer(prompt, TARGET_LAYER, ablate_mlp=True)
    if not np.isnan(pr):
        ablated_rec_prs.append(pr)
    print(f"  Rec: PR={pr:.3f}")

for prompt in BASELINE_PROMPTS:
    pr = measure_pr_at_layer(prompt, TARGET_LAYER, ablate_mlp=True)
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
    verdict = "🎤 YES! MLP appears to be the MICROPHONE!"
    explanation = f"Ablation reduced separation by {(1 - abs(ablated_sep)/abs(normal_sep))*100:.1f}%"
elif abs(ablated_sep) < abs(normal_sep):
    verdict = "🎤 PARTIAL: MLP contributes to contraction"
    explanation = f"Ablation reduced separation by {(1 - abs(ablated_sep)/abs(normal_sep))*100:.1f}%"
else:
    verdict = "❌ NO: MLP is NOT the microphone"
    explanation = "Ablation did not reduce R_V separation"

print(f"\n{'=' * 60}")
print(f"VERDICT: {verdict}")
print(f"  {explanation}")
print(f"{'=' * 60}")

# Save results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = Path("/workspace/mech-interp-latent-lab-phase1/DEC9_2025_RLOOP_MASTER_EXEC/01_CONFOUND_FALSIFICATION/results")

results_data = {
    'normal_rec_mean': np.mean(normal_rec_prs),
    'normal_base_mean': np.mean(normal_base_prs),
    'normal_separation': normal_sep,
    'ablated_rec_mean': np.mean(ablated_rec_prs),
    'ablated_base_mean': np.mean(ablated_base_prs),
    'ablated_separation': ablated_sep,
    'separation_change': sep_change,
    'pct_change': pct_change
}

df = pd.DataFrame([results_data])
csv_path = results_dir / f"mlp_ablation_l14_{timestamp}.csv"
df.to_csv(csv_path, index=False)
print(f"\nSaved to: {csv_path}")

print("\n" + "=" * 60)
print("DONE!")
print("=" * 60)
