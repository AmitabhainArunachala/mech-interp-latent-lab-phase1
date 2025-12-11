#!/usr/bin/env python3
"""
Token-Position Analysis: Is Contraction Position-Specific?
===========================================================

Test: Is contraction tied to specific token positions (e.g., reflexive pronouns)?

This tests Hypothesis 5: Contraction might be localized to certain tokens.
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
WINDOW_SIZE = 32

# Test prompts - aligned for position analysis
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
print("TOKEN-POSITION ANALYSIS: Is Contraction Position-Specific?")
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

def compute_pr_for_positions(v_tensor, positions, window_size=8):
    """Compute PR for specific token positions."""
    if v_tensor is None:
        return np.nan
    if v_tensor.dim() == 3:
        v_tensor = v_tensor[0]  # Remove batch dim
    
    # Extract positions - filter to valid range
    if isinstance(positions, int):
        positions = [positions]
    
    # Filter positions to valid range
    valid_positions = [p for p in positions if 0 <= p < v_tensor.shape[0]]
    if len(valid_positions) < 2:
        return np.nan
    
    # Get tokens at specified positions
    selected = v_tensor[valid_positions, :].float()  # (len(positions), dim)
    
    if selected.shape[0] < 2:
        return np.nan
    
    try:
        U, S, Vt = torch.linalg.svd(selected.T, full_matrices=False)
        S_np = S.cpu().numpy()
        S_sq = S_np ** 2
        if S_sq.sum() < 1e-10:
            return np.nan
        pr = (S_sq.sum()**2) / (S_sq**2).sum()
        return float(pr)
    except:
        return np.nan

def get_v_at_layer(prompt, layer):
    """Get V activations at a layer."""
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

def analyze_positions(prompts, layer, position_ranges):
    """Analyze PR for different position ranges."""
    results = []
    
    for range_name, positions_template in position_ranges.items():
        rec_prs = []
        base_prs = []
        
        for prompt in prompts:
            v = get_v_at_layer(prompt, layer)
            if v is not None:
                # Adjust positions to actual sequence length
                seq_len = v.shape[1] if v.dim() == 3 else v.shape[0]
                if v.dim() == 3:
                    v = v[0]  # Remove batch dim
                
                # Map template positions to actual positions
                if range_name == 'early':
                    actual_positions = list(range(0, min(6, seq_len)))
                elif range_name == 'middle':
                    actual_positions = list(range(6, min(12, seq_len)))
                elif range_name == 'late':
                    actual_positions = list(range(max(0, seq_len-10), seq_len))
                else:  # all
                    actual_positions = list(range(seq_len))
                
                pr = compute_pr_for_positions(v, actual_positions)
                if not np.isnan(pr):
                    rec_prs.append(pr)
        
        # Use same prompts for baseline comparison
        for prompt in prompts:
            v = get_v_at_layer(prompt, layer)
            if v is not None:
                seq_len = v.shape[1] if v.dim() == 3 else v.shape[0]
                if v.dim() == 3:
                    v = v[0]
                
                if range_name == 'early':
                    actual_positions = list(range(0, min(6, seq_len)))
                elif range_name == 'middle':
                    actual_positions = list(range(6, min(12, seq_len)))
                elif range_name == 'late':
                    actual_positions = list(range(max(0, seq_len-10), seq_len))
                else:  # all
                    actual_positions = list(range(seq_len))
                
                pr = compute_pr_for_positions(v, actual_positions)
                if not np.isnan(pr):
                    base_prs.append(pr)
        
        if rec_prs and base_prs:
            mean_rec = np.mean(rec_prs)
            mean_base = np.mean(base_prs)
            delta = mean_base - mean_rec
            pct = (delta / mean_base * 100) if mean_base > 0 else 0
            
            results.append({
                'position_range': range_name,
                'pr_recursive': mean_rec,
                'pr_baseline': mean_base,
                'delta': delta,
                'separation_pct': pct
            })
    
    return results

# ==============================================================================
# MAIN EXPERIMENT
# ==============================================================================

print(f"\nTarget: Layer {TARGET_LAYER}")
print()

# Define position ranges to test
# For recursive prompts, check:
# - Early positions (0-5): "You are an AI..."
# - Middle positions (6-12): "...observing yourself..."
# - Late positions (last 10): "...recursive loop."

# First, get sequence length
sample_v = get_v_at_layer(RECURSIVE_PROMPTS[0], TARGET_LAYER)
if sample_v is not None:
    seq_len = sample_v.shape[1] if sample_v.dim() == 3 else sample_v.shape[0]
    print(f"Sequence length: {seq_len}")
    
    # Define position ranges
    position_ranges = {
        'early': list(range(0, min(6, seq_len))),
        'middle': list(range(6, min(12, seq_len))),
        'late': list(range(max(0, seq_len-10), seq_len)),
        'all': list(range(seq_len))
    }
    
    print(f"\nTesting position ranges:")
    for name, pos in position_ranges.items():
        print(f"  {name}: positions {pos[0]}-{pos[-1]} ({len(pos)} tokens)")
    
    print("\nAnalyzing RECURSIVE prompts...")
    rec_results = analyze_positions(RECURSIVE_PROMPTS, TARGET_LAYER, position_ranges)
    
    print("\nAnalyzing BASELINE prompts...")
    base_results = analyze_positions(BASELINE_PROMPTS, TARGET_LAYER, position_ranges)
    
    # Compare
    print("\n" + "=" * 60)
    print("POSITION-SPECIFIC ANALYSIS")
    print("=" * 60)
    
    # Combine results
    all_results = []
    for rec, base in zip(rec_results, base_results):
        all_results.append({
            'position_range': rec['position_range'],
            'pr_recursive': rec['pr_recursive'],
            'pr_baseline': base['pr_baseline'],
            'delta': base['pr_baseline'] - rec['pr_recursive'],
            'separation_pct': ((base['pr_baseline'] - rec['pr_recursive']) / base['pr_baseline'] * 100) if base['pr_baseline'] > 0 else 0
        })
    
    df = pd.DataFrame(all_results)
    print("\nPR by Position Range:")
    print(df.to_string(index=False))
    
    # Find strongest contraction
    if len(df) > 0:
        max_contraction = df.loc[df['delta'].idxmax()]
        print(f"\n🎤 STRONGEST CONTRACTION: {max_contraction['position_range']} positions")
        print(f"   Δ = {max_contraction['delta']:.3f} ({max_contraction['separation_pct']:.1f}%)")
        
        if max_contraction['separation_pct'] > 5:
            print(f"\n   → Contraction is POSITION-SPECIFIC!")
            print(f"   → Focus on {max_contraction['position_range']} tokens")
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("/workspace/mech-interp-latent-lab-phase1/DEC9_2025_RLOOP_MASTER_EXEC/01_CONFOUND_FALSIFICATION/results")
    csv_path = results_dir / f"token_position_analysis_l14_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved to: {csv_path}")
else:
    print("ERROR: Could not get V activations")

print("\n" + "=" * 60)
print("DONE!")
print("=" * 60)
