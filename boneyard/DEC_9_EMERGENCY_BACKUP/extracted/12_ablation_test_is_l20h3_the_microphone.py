#!/usr/bin/env python3
"""
Ablation Test: Is L20H3 the Microphone?
========================================

Test: If we ablate L20H3, does R_V contraction disappear?
- Measure R_V at L27 with normal model
- Measure R_V at L27 with L20H3 zeroed out
- If R_V goes UP (less contraction), L20H3 is the source!
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager

sys.path.insert(0, '/workspace/mech-interp-latent-lab-phase1')

from transformers import AutoModelForCausalLM, AutoTokenizer

# ==============================================================================
# CONFIG
# ==============================================================================

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Target: L20H3 (our microphone candidate)
ABLATE_LAYER = 20
ABLATE_HEAD = 3

# Measure R_V at L27 (where we see contraction)
MEASURE_LAYER_EARLY = 5
MEASURE_LAYER_LATE = 27

# GQA params
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
print("ABLATION TEST: Is L20H3 the Microphone?")
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
# HOOKS AND FUNCTIONS
# ==============================================================================

def compute_pr(v_tensor, window_size=32):
    """Compute participation ratio from V tensor."""
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

@contextmanager
def capture_v_at_layer(layer_idx, storage):
    """Capture V activations at a layer."""
    def hook(module, inp, out):
        storage['v'] = out.detach().cpu()
    
    handle = model.model.layers[layer_idx].self_attn.v_proj.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()

@contextmanager
def ablate_kv_head(layer_idx, head_idx):
    """Zero out a specific KV head's output."""
    def hook(module, inp, out):
        # out shape: (batch, seq, num_kv_heads * head_dim)
        B, S, H = out.shape
        out_reshaped = out.view(B, S, NUM_KV_HEADS, HEAD_DIM)
        out_reshaped[:, :, head_idx, :] = 0  # Zero this head
        return out_reshaped.view(B, S, H)
    
    handle = model.model.layers[layer_idx].self_attn.v_proj.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()

def measure_rv(prompt, ablate=False):
    """Measure R_V = PR_late / PR_early for a prompt."""
    storage_early = {}
    storage_late = {}
    
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        # Set up capture hooks
        hook_early = model.model.layers[MEASURE_LAYER_EARLY].self_attn.v_proj.register_forward_hook(
            lambda m, i, o: storage_early.update({'v': o.detach().cpu()})
        )
        hook_late = model.model.layers[MEASURE_LAYER_LATE].self_attn.v_proj.register_forward_hook(
            lambda m, i, o: storage_late.update({'v': o.detach().cpu()})
        )
        
        # Optionally add ablation hook
        ablate_hook = None
        if ablate:
            def ablate_fn(module, inp, out):
                B, S, H = out.shape
                out_reshaped = out.view(B, S, NUM_KV_HEADS, HEAD_DIM)
                out_reshaped[:, :, ABLATE_HEAD, :] = 0
                return out_reshaped.view(B, S, H)
            ablate_hook = model.model.layers[ABLATE_LAYER].self_attn.v_proj.register_forward_hook(ablate_fn)
        
        try:
            _ = model(**inputs)
        finally:
            hook_early.remove()
            hook_late.remove()
            if ablate_hook:
                ablate_hook.remove()
    
    # Compute PRs
    pr_early = compute_pr(storage_early.get('v'), WINDOW_SIZE)
    pr_late = compute_pr(storage_late.get('v'), WINDOW_SIZE)
    
    if np.isnan(pr_early) or np.isnan(pr_late) or pr_early == 0:
        return np.nan
    
    rv = pr_late / pr_early
    return rv

# ==============================================================================
# MAIN EXPERIMENT
# ==============================================================================

print(f"\nTarget: Layer {ABLATE_LAYER}, Head {ABLATE_HEAD}")
print(f"Measuring R_V at L{MEASURE_LAYER_LATE}/L{MEASURE_LAYER_EARLY}")
print(f"Window size: {WINDOW_SIZE}")
print()

results = []

# Test recursive prompts
print("Testing RECURSIVE prompts...")
for i, prompt in enumerate(RECURSIVE_PROMPTS):
    rv_normal = measure_rv(prompt, ablate=False)
    rv_ablated = measure_rv(prompt, ablate=True)
    
    delta = rv_ablated - rv_normal if not (np.isnan(rv_normal) or np.isnan(rv_ablated)) else np.nan
    pct_change = (delta / rv_normal * 100) if rv_normal and not np.isnan(delta) else np.nan
    
    results.append({
        'type': 'recursive',
        'prompt_idx': i,
        'rv_normal': rv_normal,
        'rv_ablated': rv_ablated,
        'delta': delta,
        'pct_change': pct_change
    })
    print(f"  Rec {i+1}: Normal={rv_normal:.3f}, Ablated={rv_ablated:.3f}, Δ={delta:+.3f} ({pct_change:+.1f}%)")

# Test baseline prompts
print("\nTesting BASELINE prompts...")
for i, prompt in enumerate(BASELINE_PROMPTS):
    rv_normal = measure_rv(prompt, ablate=False)
    rv_ablated = measure_rv(prompt, ablate=True)
    
    delta = rv_ablated - rv_normal if not (np.isnan(rv_normal) or np.isnan(rv_ablated)) else np.nan
    pct_change = (delta / rv_normal * 100) if rv_normal and not np.isnan(delta) else np.nan
    
    results.append({
        'type': 'baseline',
        'prompt_idx': i,
        'rv_normal': rv_normal,
        'rv_ablated': rv_ablated,
        'delta': delta,
        'pct_change': pct_change
    })
    print(f"  Base {i+1}: Normal={rv_normal:.3f}, Ablated={rv_ablated:.3f}, Δ={delta:+.3f} ({pct_change:+.1f}%)")

# ==============================================================================
# ANALYSIS
# ==============================================================================

df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("ABLATION ANALYSIS")
print("=" * 60)

rec_df = df[df['type'] == 'recursive']
base_df = df[df['type'] == 'baseline']

print(f"\nRECURSIVE PROMPTS:")
print(f"  Mean R_V (normal):  {rec_df['rv_normal'].mean():.3f}")
print(f"  Mean R_V (ablated): {rec_df['rv_ablated'].mean():.3f}")
print(f"  Mean Δ:             {rec_df['delta'].mean():+.3f}")

print(f"\nBASELINE PROMPTS:")
print(f"  Mean R_V (normal):  {base_df['rv_normal'].mean():.3f}")
print(f"  Mean R_V (ablated): {base_df['rv_ablated'].mean():.3f}")
print(f"  Mean Δ:             {base_df['delta'].mean():+.3f}")

# Key question: Does ablating L20H3 reduce recursive contraction?
normal_separation = base_df['rv_normal'].mean() - rec_df['rv_normal'].mean()
ablated_separation = base_df['rv_ablated'].mean() - rec_df['rv_ablated'].mean()

print(f"\nR_V SEPARATION (baseline - recursive):")
print(f"  Normal model:  {normal_separation:.3f}")
print(f"  After ablation: {ablated_separation:.3f}")

if ablated_separation < normal_separation * 0.5:
    verdict = "🎤 YES! L20H3 appears to be the MICROPHONE!"
    explanation = "Ablation reduced R_V separation by >50%"
elif ablated_separation < normal_separation:
    verdict = "🎤 PARTIAL: L20H3 contributes to contraction"
    explanation = f"Ablation reduced separation by {(1 - ablated_separation/normal_separation)*100:.1f}%"
else:
    verdict = "❌ NO: L20H3 is NOT the microphone"
    explanation = "Ablation did not reduce R_V separation"

print(f"\n{'=' * 60}")
print(f"VERDICT: {verdict}")
print(f"  {explanation}")
print(f"{'=' * 60}")

# Save results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = Path("/workspace/mech-interp-latent-lab-phase1/DEC9_2025_RLOOP_MASTER_EXEC/01_CONFOUND_FALSIFICATION/results")
csv_path = results_dir / f"ablate_l20h3_{timestamp}.csv"
df.to_csv(csv_path, index=False)
print(f"\nSaved to: {csv_path}")
