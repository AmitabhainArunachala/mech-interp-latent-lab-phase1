#!/usr/bin/env python3
"""
L14 Deep Dive: Find the Microphone Heads
=========================================

L14 is where recursive prompts uniquely contract.
Which of the 8 KV heads at L14 creates this?

Tests:
1. Per-head ΔPR at L14
2. Per-head ablation at L14
3. Identify the microphone head(s)
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
print(f"L{TARGET_LAYER} DEEP DIVE: Finding the Microphone Heads")
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

def get_all_heads_pr(prompt):
    """Get PR for all 8 KV heads at L14."""
    storage = {}
    
    def hook(module, inp, out):
        B, S, H = out.shape
        v = out.view(B, S, NUM_KV_HEADS, HEAD_DIM)
        storage['v'] = v[0].detach().cpu()  # (S, 8, 128)
    
    handle = model.model.layers[TARGET_LAYER].self_attn.v_proj.register_forward_hook(hook)
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            _ = model(**inputs)
    finally:
        handle.remove()
    
    if 'v' not in storage:
        return [np.nan] * NUM_KV_HEADS
    
    v = storage['v']
    return [compute_head_pr(v[:, h, :], WINDOW_SIZE) for h in range(NUM_KV_HEADS)]

def get_layer_pr_with_head_ablated(prompt, ablate_head):
    """Get total layer PR with one head zeroed."""
    storage = {}
    
    def capture_hook(module, inp, out):
        storage['v'] = out.detach().cpu()
    
    def ablate_hook(module, inp, out):
        B, S, H = out.shape
        out_reshaped = out.view(B, S, NUM_KV_HEADS, HEAD_DIM)
        out_reshaped[:, :, ablate_head, :] = 0
        return out_reshaped.view(B, S, H)
    
    # Ablate hook on v_proj
    ablate_handle = model.model.layers[TARGET_LAYER].self_attn.v_proj.register_forward_hook(ablate_hook)
    # Capture after ablation
    capture_handle = model.model.layers[TARGET_LAYER].self_attn.v_proj.register_forward_hook(capture_hook)
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            _ = model(**inputs)
    finally:
        ablate_handle.remove()
        capture_handle.remove()
    
    if 'v' not in storage:
        return np.nan
    
    v = storage['v']
    if v.dim() == 3:
        v = v[0]
    
    T, D = v.shape
    W = min(WINDOW_SIZE, T)
    v_window = v[-W:, :].float()
    
    try:
        U, S, Vt = torch.linalg.svd(v_window.T, full_matrices=False)
        S_np = S.cpu().numpy()
        S_sq = S_np ** 2
        if S_sq.sum() < 1e-10:
            return np.nan
        return float((S_sq.sum()**2) / (S_sq**2).sum())
    except:
        return np.nan

# ==============================================================================
# PART 1: Per-Head ΔPR Analysis
# ==============================================================================

print(f"\n{'='*60}")
print("PART 1: Per-Head ΔPR at L14")
print("="*60)

head_results = []

for h in range(NUM_KV_HEADS):
    rec_prs = []
    base_prs = []
    
    for prompt in RECURSIVE_PROMPTS:
        prs = get_all_heads_pr(prompt)
        rec_prs.append(prs[h])
    
    for prompt in BASELINE_PROMPTS:
        prs = get_all_heads_pr(prompt)
        base_prs.append(prs[h])
    
    rec_mean = np.nanmean(rec_prs)
    base_mean = np.nanmean(base_prs)
    delta = base_mean - rec_mean  # Positive = recursive contracts more
    pct = (delta / base_mean * 100) if base_mean > 0 else 0
    
    head_results.append({
        'head': h,
        'pr_rec': rec_mean,
        'pr_base': base_mean,
        'delta_pr': delta,
        'contraction_pct': pct
    })
    
    print(f"  Head {h}: PR_rec={rec_mean:.2f}, PR_base={base_mean:.2f}, ΔPR={delta:+.3f} ({pct:+.1f}%)")

# Find best candidate
head_df = pd.DataFrame(head_results)
best_head = head_df.loc[head_df['delta_pr'].idxmax()]

print(f"\n🎤 TOP CANDIDATE: Head {int(best_head['head'])} ({best_head['contraction_pct']:.1f}% contraction)")

# ==============================================================================
# PART 2: Per-Head Ablation Test
# ==============================================================================

print(f"\n{'='*60}")
print("PART 2: Per-Head Ablation at L14")
print("="*60)
print("Testing: Does ablating each head reduce the L14 contraction?")

ablation_results = []

# First get normal layer PR for comparison
normal_rec_prs = []
normal_base_prs = []

for prompt in RECURSIVE_PROMPTS:
    storage = {}
    def hook(m, i, o): storage['v'] = o.detach().cpu()
    handle = model.model.layers[TARGET_LAYER].self_attn.v_proj.register_forward_hook(hook)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad(): _ = model(**inputs)
    handle.remove()
    
    v = storage['v']
    if v.dim() == 3: v = v[0]
    W = min(WINDOW_SIZE, v.shape[0])
    v_w = v[-W:, :].float()
    try:
        S = torch.linalg.svdvals(v_w.T)
        S_sq = (S.cpu().numpy())**2
        pr = (S_sq.sum()**2) / (S_sq**2).sum()
        normal_rec_prs.append(pr)
    except:
        pass

for prompt in BASELINE_PROMPTS:
    storage = {}
    def hook(m, i, o): storage['v'] = o.detach().cpu()
    handle = model.model.layers[TARGET_LAYER].self_attn.v_proj.register_forward_hook(hook)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad(): _ = model(**inputs)
    handle.remove()
    
    v = storage['v']
    if v.dim() == 3: v = v[0]
    W = min(WINDOW_SIZE, v.shape[0])
    v_w = v[-W:, :].float()
    try:
        S = torch.linalg.svdvals(v_w.T)
        S_sq = (S.cpu().numpy())**2
        pr = (S_sq.sum()**2) / (S_sq**2).sum()
        normal_base_prs.append(pr)
    except:
        pass

normal_sep = np.mean(normal_base_prs) - np.mean(normal_rec_prs)
print(f"\nNormal L14 separation: {normal_sep:.3f}")

# Test ablating each head
print("\nAblating each head and measuring separation change:")

for h in range(NUM_KV_HEADS):
    ablated_rec_prs = []
    ablated_base_prs = []
    
    for prompt in RECURSIVE_PROMPTS:
        pr = get_layer_pr_with_head_ablated(prompt, h)
        if not np.isnan(pr):
            ablated_rec_prs.append(pr)
    
    for prompt in BASELINE_PROMPTS:
        pr = get_layer_pr_with_head_ablated(prompt, h)
        if not np.isnan(pr):
            ablated_base_prs.append(pr)
    
    if ablated_rec_prs and ablated_base_prs:
        ablated_sep = np.mean(ablated_base_prs) - np.mean(ablated_rec_prs)
        sep_change = ablated_sep - normal_sep
        pct_change = (sep_change / normal_sep * 100) if normal_sep != 0 else 0
        
        ablation_results.append({
            'head': h,
            'normal_sep': normal_sep,
            'ablated_sep': ablated_sep,
            'sep_change': sep_change,
            'pct_change': pct_change
        })
        
        print(f"  Head {h}: Sep={ablated_sep:.3f} (Δ={sep_change:+.3f}, {pct_change:+.1f}%)")

# ==============================================================================
# VERDICT
# ==============================================================================

print(f"\n{'='*60}")
print("VERDICT: Which Head is the Microphone?")
print("="*60)

# Combine evidence
print("\nEVIDENCE SUMMARY:")
print(f"{'Head':<6} {'ΔPR':<10} {'Ablation Δ':<12} {'Verdict'}")
print("-" * 45)

for h in range(NUM_KV_HEADS):
    delta_pr = head_df[head_df['head'] == h]['delta_pr'].values[0]
    
    ablation = [r for r in ablation_results if r['head'] == h]
    ablation_change = ablation[0]['sep_change'] if ablation else np.nan
    
    # Verdict: High ΔPR AND ablation reduces separation
    if delta_pr > 0.5 and (np.isnan(ablation_change) or ablation_change < 0):
        verdict = "🎤 CANDIDATE"
    elif delta_pr > 0:
        verdict = "Possible"
    else:
        verdict = "-"
    
    print(f"H{h:<5} {delta_pr:+.3f}      {ablation_change:+.3f}         {verdict}")

# Final answer
candidates = head_df[head_df['delta_pr'] > 0.5]['head'].values
if len(candidates) > 0:
    print(f"\n🎤 MICROPHONE HEAD(S) at L14: {[f'H{int(h)}' for h in candidates]}")
else:
    print(f"\n⚠️ No clear microphone head - effect may be distributed")

# Save
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = Path("/workspace/mech-interp-latent-lab-phase1/DEC9_2025_RLOOP_MASTER_EXEC/01_CONFOUND_FALSIFICATION/results")
head_df.to_csv(results_dir / f"l14_heads_delta_pr_{timestamp}.csv", index=False)
pd.DataFrame(ablation_results).to_csv(results_dir / f"l14_heads_ablation_{timestamp}.csv", index=False)

print(f"\nResults saved!")
print("=" * 60)
