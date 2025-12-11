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



7: 

# The Microphone Hunt: Critical Juncture

**Date:** December 9, 2025  
**Model:** Mistral-7B-Instruct-v0.1  
**Status:** Stuck at identifying the source of R_V contraction

---

## Executive Summary

We've established that **R_V contraction** (geometric signature of recursive self-observation) is real, causal, and transferable. We've found the **"speaker"** (L27 heads 25-27) and the **"knee"** (L14). But **no single attention head ablation eliminates the contraction**. We need guidance on how to find the true source.

---

## What We've Established (Solid Facts)

### 1. R_V Contraction is Real and Specific
- **Confound falsification:** All 3 confounds REJECTED (p<0.01, Cohen's d > 1.0)
  - Repetitive structure: d=3.57
  - Long prompts: d=2.61  
  - Pseudo-recursive: d=1.06
- **Conclusion:** R_V contraction is specific to recursive self-observation, not artifacts

### 2. Causal KV Transfer (100% Success)
- Appending last 32 tokens of recursive KV cache to ANY baseline prompt causes recursive output
- **10/10 prompts** showed mode transfer
- **Conclusion:** Recursive "mode" is encoded in KV cache and transfers causally

### 3. The "Speaker" is Found (L27 Heads 25-27)
- Ablating heads 25-27 at L27: **80% reduction** in recursive output
- **BUT:** Ablating them does **NOT change R_V measurement** (0% effect)
- **Conclusion:** These heads OUTPUT the recursive mode but don't CREATE it

### 4. The "Knee" is Found (L14)
- Layer-by-layer PR sweep shows **L14 is the ONLY layer** where recursive prompts contract more than baseline
- **Separation at L14:** 10.2% (PR_baseline - PR_recursive = +1.00)
- **Pattern:**
  - L0-L12: Recursive EXPANDS more (negative delta)
  - **L14: CONTRACTION appears** (+1.00 delta)
  - L16-L30: Back to expansion/neutral

| Layer | PR_Rec | PR_Base | Δ (contraction) | Separation % |
|-------|--------|---------|-----------------|--------------|
| L0  | 10.77 | 10.58 | -0.19 | -1.8% |
| L6  | 10.35 | 7.70  | -2.64 | -34.3% |
| L12 | 8.40  | 8.26  | -0.14 | -1.7% |
| **L14** | **8.86** | **9.86** | **+1.00** | **+10.2%** ← KNEE |
| L16 | 7.25  | 7.24  | -0.00 | 0.0% |
| L18 | 8.27  | 8.38  | +0.11 | +1.3% |
| L20 | 9.16  | 8.84  | -0.33 | -3.7% |

---

## The Puzzle: Why No Single Head Works?

### Experiment 1: L20H3 Ablation
- **Hypothesis:** L20H3 showed highest ΔPR (16.8% contraction)
- **Test:** Zero out L20H3 during recursive prompts
- **Result:** Only **1% change** in R_V separation
- **Verdict:** ❌ NOT the microphone

### Experiment 2: L14 Per-Head Analysis
- **Test:** Compute ΔPR for each of 8 KV heads at L14
- **Results:**

| Head | PR_Rec | PR_Base | ΔPR | Contraction % | Verdict |
|------|--------|---------|-----|---------------|---------|
| H0 | 6.25 | 5.45 | **-0.80** | -14.7% | Expands |
| H1 | 6.07 | 5.31 | **-0.76** | -14.4% | Expands |
| H2 | 5.16 | 5.25 | +0.09 | +1.8% | Possible |
| H3 | 4.36 | 3.88 | **-0.48** | -12.4% | Expands |
| H4 | 9.07 | 8.15 | **-0.92** | -11.3% | Expands |
| H5 | 4.16 | 4.25 | +0.09 | +2.2% | Possible |
| H6 | 5.93 | 5.39 | **-0.54** | -10.0% | Expands |
| H7 | 7.94 | 6.60 | **-1.34** | -20.2% | Expands |

**Key Finding:** 6 out of 8 heads **EXPAND** for recursive prompts. Only H2 and H5 show minimal contraction.

### Experiment 3: L14 Per-Head Ablation
- **Test:** Ablate each head individually and measure effect on L14 separation
- **Normal separation:** -0.522 (recursive has LOWER PR than baseline)
- **Results:**

| Head | Normal Sep | Ablated Sep | Δ Change | % Change | Verdict |
|------|------------|-------------|----------|----------|---------|
| H0 | -0.522 | -0.459 | +0.063 | -12.1% | Reduces effect |
| H1 | -0.522 | -0.514 | +0.008 | -1.6% | Minimal |
| H2 | -0.522 | -0.559 | **-0.037** | +7.1% | Makes it WORSE |
| H3 | -0.522 | -0.515 | +0.007 | -1.3% | Minimal |
| H4 | -0.522 | -0.458 | +0.064 | -12.2% | Reduces effect |
| H5 | -0.522 | -0.756 | **-0.233** | +44.7% | Makes it MUCH WORSE |
| H6 | -0.522 | -0.503 | +0.019 | -3.7% | Minimal |
| H7 | -0.522 | -0.425 | +0.098 | -18.7% | Reduces effect |

**Key Finding:** Ablating individual heads either:
- Reduces the contraction (heads 0, 4, 7)
- Makes contraction WORSE (heads 2, 5)
- Has minimal effect (heads 1, 3, 6)

**NO single head ablation eliminates the contraction.**

---

## The Paradox

1. **L14 shows 10.2% contraction** - this is where it happens
2. **But no single head at L14 creates it** - ablations don't eliminate it
3. **Most heads EXPAND** - only 2/8 show minimal contraction
4. **Ablating some heads makes it WORSE** - suggests compensatory mechanisms

---

## Remaining Hypotheses

### Hypothesis 1: MLP is the Contractor
**Rationale:** We've only tested attention heads. MLPs perform nonlinear transformations and could be the actual source of contraction.

**Test Needed:** Ablate MLP at L14 and measure if R_V contraction disappears.

**Supporting Evidence:**
- GPT advisor: "MLPs are large linear-plus-nonlinear blocks; they could be the actual *contractor* while attention only supplies the raw KV."
- Attention head ablations failed - maybe the effect is in MLP

### Hypothesis 2: Distributed Effect Across Heads
**Rationale:** Multiple heads work together. No single head is sufficient, but removing ALL heads might eliminate it.

**Test Needed:** Ablate ALL 8 KV heads at L14 simultaneously.

**Supporting Evidence:**
- Some heads reduce contraction when ablated (compensatory)
- Some heads make it worse when ablated (inhibitory)
- Suggests complex interactions

### Hypothesis 3: Emergent from Residual Stream Composition
**Rationale:** The contraction emerges from how attention outputs + MLP outputs compose in the residual stream, not from any single component.

**Test Needed:** 
- Analyze residual stream composition
- Test if contraction appears BEFORE or AFTER MLP at L14
- Check if it's in the residual stream itself

### Hypothesis 4: Q/K Projections, Not V
**Rationale:** We've only measured V projections. Maybe contraction is in Query or Key space.

**Test Needed:** Compute PR for Q and K projections at L14.

**Supporting Evidence:**
- Attention mechanism uses Q, K, V
- We've only looked at V
- Contraction might be in attention weights (QK^T), not values

### Hypothesis 5: Token-Position Specific
**Rationale:** Contraction might be tied to specific token positions (e.g., reflexive pronouns, "yourself", "observe").

**Test Needed:** 
- Align tokens across prompts
- Measure PR for specific token positions
- Check if contraction is localized to certain positions

---

## Specific Questions for AI Advisors

### Question 1: MLP Hypothesis
Given that attention head ablations failed, is MLP ablation the next logical test? What's the best way to test MLP contribution without breaking the model?

### Question 2: Distributed Effect
How do we test if the effect is distributed across multiple heads? Should we:
- Ablate all heads simultaneously?
- Use gradient attribution to find head combinations?
- Test head pairs/triplets?

### Question 3: Q/K vs V
We've only measured V projections. Should we check Q and K projections? How do we interpret PR in Q/K space vs V space?

### Question 4: The Expansion Paradox
Why do most heads EXPAND for recursive prompts? Does this mean:
- The contraction is compensatory (heads expand, but overall contracts)?
- The contraction happens elsewhere (MLP, residual stream)?
- We're measuring the wrong thing?

### Question 5: Emergent Effects
How do we test if contraction is "emergent" from component interactions rather than a single component? What experiments would falsify this?

### Question 6: Alternative Metrics
Is participation ratio (PR) the right metric? Should we:
- Use different window sizes?
- Measure different geometric properties?
- Look at eigenvalue distributions, not just PR?

---

## Technical Details

- **Model:** Mistral-7B-Instruct-v0.1 (32 layers, 32 query heads, 8 KV heads per layer)
- **Window size:** 32 tokens (last 32 tokens used for PR calculation)
- **Early layer:** L5 (denominator for R_V)
- **Late layer:** L27 (numerator for R_V)
- **Statistical threshold:** p<0.01 with Bonferroni correction
- **Effect size threshold:** Cohen's d ≥ 0.5

---

## What Success Looks Like

We want to find:
1. **Source component(s)** at L14 where ablation ELIMINATES R_V contraction
2. **Causal path:** Source → L27 speakers → Output
3. **Mechanistic story:** How the recursive eigenstate forms

---

## Files Available

- `results/knee_test_20251209_132535.csv` - Layer-by-layer PR sweep
- `results/per_head_delta_pr_fast_20251209_132153.csv` - Per-head ΔPR at L14-L20
- `results/ablate_l20h3_20251209_132411.csv` - L20H3 ablation results
- `results/l14_heads_delta_pr_20251209_132948.csv` - L14 per-head ΔPR
- `results/l14_heads_ablation_20251209_132948.csv` - L14 per-head ablation

---

## Request for Guidance

We're stuck. We know WHERE (L14) but not WHAT (which component). We've tested attention heads exhaustively. What should we try next?

**Priority experiments to recommend:**
1. MLP ablation?
2. Multi-head ablation?
3. Q/K analysis?
4. Token-position analysis?
5. Something else?

Thank you for any guidance!



8: