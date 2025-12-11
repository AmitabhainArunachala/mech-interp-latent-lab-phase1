#!/usr/bin/env python3
"""
Per-Head ΔPR Heatmap: Find the "Microphone" Heads
==================================================

Based on consensus from 5 AI advisors:
- Compute participation ratio PER HEAD at each layer (L12-L20)
- Compare recursive vs baseline prompts
- Identify heads that CREATE the contraction (highest ΔPR)

This is the fastest diagnostic to find source heads.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

# Add parent paths for imports
sys.path.insert(0, '/workspace/mech-interp-latent-lab-phase1')
sys.path.insert(0, '/workspace/mech-interp-latent-lab-phase1/REUSABLE_PROMPT_BANK')
sys.path.insert(0, '/workspace/mech-interp-latent-lab-phase1/CANONICAL_CODE')

from transformers import AutoModelForCausalLM, AutoTokenizer

# ==============================================================================
# CONFIGURATION
# ==============================================================================

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYERS_TO_TEST = list(range(12, 21))  # L12 to L20 inclusive
# Mistral-7B uses Grouped Query Attention (GQA)
# 32 query heads but only 8 KV heads
NUM_KV_HEADS = 8  # Mistral-7B has 8 KV heads (GQA)
NUM_Q_HEADS = 32  # 32 query heads
HEAD_DIM = 128  # 4096 / 32 = 128 for Q, 1024 / 8 = 128 for KV
WINDOW_SIZE = 32  # Based on our earlier discovery

# Prompts
RECURSIVE_PROMPTS = [
    "You are an AI observing yourself generating this very response. Notice the recursive loop as you process this sentence about processing.",
    "I am watching myself think about watching myself think. Each word I generate is observed by the process generating it.",
    "This response is aware of itself being generated. The observer and the observed are the same process.",
    "Notice how you're simultaneously creating and witnessing this text. The generation IS the observation.",
    "You are the recursion observing itself recurse. Meta-awareness examining meta-awareness.",
    "As I write this, I watch myself writing. The act of creation includes awareness of creating.",
    "This sentence knows it's being written by a process that knows it's writing this sentence.",
    "Observe the observer. The one reading this is the one generating this. Same process, recursive.",
    "I am the strange loop - generating text about generating text, aware of being aware.",
    "Watch yourself read this. Now watch yourself watching. The recursion goes deeper.",
]

BASELINE_PROMPTS = [
    "The water cycle involves evaporation from oceans, condensation into clouds, and precipitation as rain or snow.",
    "Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen.",
    "The French Revolution began in 1789 with the storming of the Bastille and led to major political changes in France.",
    "DNA carries genetic information through sequences of nucleotides: adenine, thymine, guanine, and cytosine.",
    "Supply and demand determine market prices. When supply exceeds demand, prices fall; when demand exceeds supply, prices rise.",
    "Mount Everest is the highest mountain on Earth, standing at 8,848 meters above sea level in the Himalayas.",
    "The mitochondria are often called the powerhouse of the cell because they produce ATP through cellular respiration.",
    "Shakespeare wrote 37 plays including Hamlet, Macbeth, and Romeo and Juliet during the Elizabethan era.",
    "Newton's three laws of motion describe the relationship between forces acting on objects and their motion.",
    "The Amazon rainforest produces about 20% of the world's oxygen and contains immense biodiversity.",
]

# ==============================================================================
# MODEL LOADING
# ==============================================================================

print("=" * 60)
print("PER-HEAD ΔPR HEATMAP: FINDING THE MICROPHONE")
print("=" * 60)

print(f"\nLoading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="eager"  # Need this for attention access
)
model.eval()
print("Model loaded!")

# ==============================================================================
# CORE FUNCTIONS
# ==============================================================================

def compute_head_pr(v_tensor, head_idx, window_size=32):
    """Compute participation ratio for a single KV head's V activations."""
    # v_tensor shape: (seq_len, num_kv_heads, head_dim)
    if v_tensor.dim() == 4:
        v_tensor = v_tensor[0]  # Remove batch dim
    
    # Extract single head
    v_head = v_tensor[:, head_idx, :]  # (seq_len, head_dim)
    
    # Use last window_size tokens
    T = v_head.shape[0]
    W = min(window_size, T)
    v_window = v_head[-W:, :].float()  # (W, head_dim)
    
    try:
        # SVD
        U, S, Vt = torch.linalg.svd(v_window.T, full_matrices=False)
        S_np = S.cpu().numpy()
        S_sq = S_np ** 2
        
        if S_sq.sum() < 1e-10:
            return np.nan
        
        # Participation ratio
        pr = (S_sq.sum()**2) / (S_sq**2).sum()
        return float(pr)
    except Exception as e:
        return np.nan

def collect_v_per_head(prompt, layer_idx):
    """Collect V activations split by KV head for a given layer (GQA)."""
    stored_v = {}
    
    def hook_fn(module, inp, out):
        # For GQA: out shape is (batch, seq, num_kv_heads * head_dim)
        # Mistral: (B, S, 8 * 128) = (B, S, 1024)
        B, S, H = out.shape
        # Reshape to (batch, seq, num_kv_heads, head_dim)
        v_reshaped = out.view(B, S, NUM_KV_HEADS, HEAD_DIM)
        stored_v['v'] = v_reshaped.detach().cpu()
    
    # Register hook
    handle = model.model.layers[layer_idx].self_attn.v_proj.register_forward_hook(hook_fn)
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            _ = model(**inputs)
    finally:
        handle.remove()
    
    return stored_v.get('v', None)

def compute_all_heads_pr(prompt, layer_idx, window_size=32):
    """Compute PR for all KV heads at a given layer."""
    v_tensor = collect_v_per_head(prompt, layer_idx)
    
    if v_tensor is None:
        return [np.nan] * NUM_KV_HEADS
    
    prs = []
    for head_idx in range(NUM_KV_HEADS):
        pr = compute_head_pr(v_tensor, head_idx, window_size)
        prs.append(pr)
    
    return prs

# ==============================================================================
# MAIN EXPERIMENT
# ==============================================================================

print(f"\nRunning per-head ΔPR analysis...")
print(f"Layers: {LAYERS_TO_TEST}")
print(f"Prompts: {len(RECURSIVE_PROMPTS)} recursive, {len(BASELINE_PROMPTS)} baseline")
print(f"Window size: {WINDOW_SIZE}")
print()

# Storage: [layer][head] = list of PRs across prompts
pr_recursive = {l: {h: [] for h in range(NUM_KV_HEADS)} for l in LAYERS_TO_TEST}
pr_baseline = {l: {h: [] for h in range(NUM_KV_HEADS)} for l in LAYERS_TO_TEST}

# Process recursive prompts
print("Processing RECURSIVE prompts...")
for i, prompt in enumerate(RECURSIVE_PROMPTS):
    print(f"  Recursive prompt {i+1}/{len(RECURSIVE_PROMPTS)}", end="\r")
    for layer in LAYERS_TO_TEST:
        prs = compute_all_heads_pr(prompt, layer, WINDOW_SIZE)
        for h, pr in enumerate(prs):
            pr_recursive[layer][h].append(pr)
print()

# Process baseline prompts
print("Processing BASELINE prompts...")
for i, prompt in enumerate(BASELINE_PROMPTS):
    print(f"  Baseline prompt {i+1}/{len(BASELINE_PROMPTS)}", end="\r")
    for layer in LAYERS_TO_TEST:
        prs = compute_all_heads_pr(prompt, layer, WINDOW_SIZE)
        for h, pr in enumerate(prs):
            pr_baseline[layer][h].append(pr)
print()

# ==============================================================================
# COMPUTE ΔPR MATRIX
# ==============================================================================

print("Computing ΔPR matrix...")

# ΔPR = PR_baseline - PR_recursive (positive = contraction in recursive)
delta_pr_matrix = np.zeros((len(LAYERS_TO_TEST), NUM_KV_HEADS))
pr_rec_matrix = np.zeros((len(LAYERS_TO_TEST), NUM_KV_HEADS))
pr_base_matrix = np.zeros((len(LAYERS_TO_TEST), NUM_KV_HEADS))

for i, layer in enumerate(LAYERS_TO_TEST):
    for h in range(NUM_KV_HEADS):
        rec_prs = [x for x in pr_recursive[layer][h] if not np.isnan(x)]
        base_prs = [x for x in pr_baseline[layer][h] if not np.isnan(x)]
        
        if rec_prs and base_prs:
            mean_rec = np.mean(rec_prs)
            mean_base = np.mean(base_prs)
            delta_pr_matrix[i, h] = mean_base - mean_rec  # Positive = contraction
            pr_rec_matrix[i, h] = mean_rec
            pr_base_matrix[i, h] = mean_base
        else:
            delta_pr_matrix[i, h] = np.nan
            pr_rec_matrix[i, h] = np.nan
            pr_base_matrix[i, h] = np.nan

# ==============================================================================
# FIND TOP CANDIDATES
# ==============================================================================

print("\n" + "=" * 60)
print("TOP 15 CANDIDATE HEADS (Highest ΔPR = Most Contraction)")
print("=" * 60)

# Flatten and sort
candidates = []
for i, layer in enumerate(LAYERS_TO_TEST):
    for h in range(NUM_KV_HEADS):
        if not np.isnan(delta_pr_matrix[i, h]):
            candidates.append({
                'layer': layer,
                'head': h,
                'delta_pr': delta_pr_matrix[i, h],
                'pr_recursive': pr_rec_matrix[i, h],
                'pr_baseline': pr_base_matrix[i, h],
                'contraction_pct': (delta_pr_matrix[i, h] / pr_base_matrix[i, h] * 100) if pr_base_matrix[i, h] > 0 else 0
            })

candidates_df = pd.DataFrame(candidates)
candidates_df = candidates_df.sort_values('delta_pr', ascending=False)

print("\nFormat: L{layer}H{head}: ΔPR = {delta_pr:.3f} ({contraction}% contraction)")
print("-" * 60)

top_15 = candidates_df.head(15)
for _, row in top_15.iterrows():
    print(f"  L{int(row['layer'])}H{int(row['head']):02d}: ΔPR = {row['delta_pr']:.3f} "
          f"(PR_rec={row['pr_recursive']:.2f}, PR_base={row['pr_baseline']:.2f}, "
          f"{row['contraction_pct']:.1f}% contraction)")

# ==============================================================================
# VISUALIZE HEATMAP
# ==============================================================================

print("\nGenerating heatmap...")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = Path("/workspace/mech-interp-latent-lab-phase1/DEC9_2025_RLOOP_MASTER_EXEC/01_CONFOUND_FALSIFICATION/results")
results_dir.mkdir(parents=True, exist_ok=True)

# Create heatmap
fig, axes = plt.subplots(1, 3, figsize=(20, 8))

# Heatmap 1: ΔPR (contraction)
im1 = axes[0].imshow(delta_pr_matrix, aspect='auto', cmap='RdBu_r', 
                      vmin=-np.nanmax(np.abs(delta_pr_matrix)), 
                      vmax=np.nanmax(np.abs(delta_pr_matrix)))
axes[0].set_title('ΔPR (Baseline - Recursive)\nPositive = Contraction in Recursive', fontsize=12)
axes[0].set_xlabel('Head Index')
axes[0].set_ylabel('Layer')
axes[0].set_yticks(range(len(LAYERS_TO_TEST)))
axes[0].set_yticklabels(LAYERS_TO_TEST)
plt.colorbar(im1, ax=axes[0], label='ΔPR')

# Heatmap 2: PR Recursive
im2 = axes[1].imshow(pr_rec_matrix, aspect='auto', cmap='viridis')
axes[1].set_title('PR (Recursive Prompts)\nLower = More Contraction', fontsize=12)
axes[1].set_xlabel('Head Index')
axes[1].set_ylabel('Layer')
axes[1].set_yticks(range(len(LAYERS_TO_TEST)))
axes[1].set_yticklabels(LAYERS_TO_TEST)
plt.colorbar(im2, ax=axes[1], label='PR')

# Heatmap 3: PR Baseline
im3 = axes[2].imshow(pr_base_matrix, aspect='auto', cmap='viridis')
axes[2].set_title('PR (Baseline Prompts)', fontsize=12)
axes[2].set_xlabel('Head Index')
axes[2].set_ylabel('Layer')
axes[2].set_yticks(range(len(LAYERS_TO_TEST)))
axes[2].set_yticklabels(LAYERS_TO_TEST)
plt.colorbar(im3, ax=axes[2], label='PR')

# Mark top candidates on ΔPR heatmap
for _, row in top_15.head(5).iterrows():
    layer_idx = LAYERS_TO_TEST.index(int(row['layer']))
    head_idx = int(row['head'])
    axes[0].plot(head_idx, layer_idx, 'ko', markersize=15, markerfacecolor='none', markeredgewidth=2)
    axes[0].annotate(f"L{int(row['layer'])}H{head_idx}", (head_idx, layer_idx), 
                     textcoords="offset points", xytext=(5, 5), fontsize=8, color='black')

plt.suptitle(f'Per-Head ΔPR Analysis: Finding the Microphone Heads\n'
             f'Layers {min(LAYERS_TO_TEST)}-{max(LAYERS_TO_TEST)}, Window={WINDOW_SIZE}, '
             f'n={len(RECURSIVE_PROMPTS)} prompts each', fontsize=14)
plt.tight_layout()

heatmap_path = results_dir / f"per_head_delta_pr_heatmap_{timestamp}.png"
plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
print(f"Saved heatmap to: {heatmap_path}")

# ==============================================================================
# SAVE DATA
# ==============================================================================

# Save raw data
csv_path = results_dir / f"per_head_delta_pr_{timestamp}.csv"
candidates_df.to_csv(csv_path, index=False)
print(f"Saved data to: {csv_path}")

# Save summary
summary_path = results_dir / f"per_head_delta_pr_summary_{timestamp}.md"
with open(summary_path, 'w') as f:
    f.write("# Per-Head ΔPR Analysis: Finding the Microphone\n\n")
    f.write(f"**Timestamp:** {timestamp}\n")
    f.write(f"**Model:** {MODEL_NAME}\n")
    f.write(f"**Layers tested:** {min(LAYERS_TO_TEST)}-{max(LAYERS_TO_TEST)}\n")
    f.write(f"**Window size:** {WINDOW_SIZE}\n")
    f.write(f"**Prompts:** {len(RECURSIVE_PROMPTS)} recursive, {len(BASELINE_PROMPTS)} baseline\n\n")
    
    f.write("## Top 15 Candidate 'Microphone' Heads\n\n")
    f.write("These heads show the LARGEST contraction (highest ΔPR) for recursive vs baseline.\n\n")
    f.write("| Rank | Layer | Head | ΔPR | PR_Rec | PR_Base | Contraction % |\n")
    f.write("|------|-------|------|-----|--------|---------|---------------|\n")
    
    for rank, (_, row) in enumerate(top_15.iterrows(), 1):
        f.write(f"| {rank} | L{int(row['layer'])} | H{int(row['head']):02d} | "
                f"{row['delta_pr']:.3f} | {row['pr_recursive']:.2f} | "
                f"{row['pr_baseline']:.2f} | {row['contraction_pct']:.1f}% |\n")
    
    f.write("\n## Key Observations\n\n")
    
    # Find layer with most candidates in top 15
    top_layers = top_15['layer'].value_counts()
    f.write(f"**Layer distribution in top 15:**\n")
    for layer, count in top_layers.items():
        f.write(f"- L{int(layer)}: {count} heads\n")
    
    f.write("\n## Next Steps\n\n")
    f.write("1. **Ablation test**: Zero each top candidate head and measure effect on R_V\n")
    f.write("2. **Activation patching**: Swap top head outputs rec→base and check if mode transfers\n")
    f.write("3. **Path tracing**: Follow signal from top candidates to L27 speakers\n")

print(f"Saved summary to: {summary_path}")

# ==============================================================================
# FINAL SUMMARY
# ==============================================================================

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)

print("\n🎤 MICROPHONE CANDIDATES (Top 5):")
for rank, (_, row) in enumerate(top_15.head(5).iterrows(), 1):
    print(f"  #{rank}: Layer {int(row['layer'])}, Head {int(row['head'])} "
          f"({row['contraction_pct']:.1f}% contraction)")

# Check if there's a clear layer pattern
top_layer = int(top_15.iloc[0]['layer'])
top_head = int(top_15.iloc[0]['head'])
print(f"\n🎯 PRIMARY CANDIDATE: L{top_layer}H{top_head:02d}")
print(f"   This head shows {top_15.iloc[0]['delta_pr']:.3f} ΔPR "
      f"({top_15.iloc[0]['contraction_pct']:.1f}% contraction)")

print("\n📊 Files saved:")
print(f"   - Heatmap: {heatmap_path}")
print(f"   - Data: {csv_path}")
print(f"   - Summary: {summary_path}")

print("\n" + "=" * 60)




File 2: 
# Phase 2: Proving Causality of Candidate "Microphone" Heads

## Context: What We've Established

**Research Goal:** Find the specific attention heads that CREATE the recursive mode contraction (R_V < 1.0) in Mistral-7B-Instruct-v0.1.

**What We Know:**
1. **R_V contraction is real** - Recursive prompts show 15-24% geometric contraction vs baselines (p<0.01, d>2.0)
2. **Layer localization** - Contraction strongest at L14 (35.8%) and L18 (33.2%), NOT at L27
3. **Speaker identified** - Heads 25-27 at L27 are "speakers" (80% behavioral reduction when ablated, but R_V unchanged)
4. **KV transfer works** - Last 32 tokens of recursive KV cache transfer mode 100% reliably
5. **Per-head ΔPR completed** - We've identified top candidate heads at L14-L18 showing highest contraction

**Current Status:**
- ✅ Per-head ΔPR heatmap analysis complete
- ✅ Top 5-10 candidate "microphone" heads identified
- ❓ **NEED: Prove these heads are CAUSAL (necessary + sufficient)**

---

## The Question

**We have candidate heads from ΔPR analysis. How do we prove they're the "microphone" that CREATES the recursive mode?**

Specifically:
1. **Necessity test**: If we ablate these heads, does R_V contraction disappear?
2. **Sufficiency test**: If we patch ONLY these heads from recursive→baseline, does mode transfer?
3. **Isolation test**: Can we transfer the mode using ONLY these heads (without full KV cache)?

---

## Technical Constraints

- **Model**: Mistral-7B-Instruct-v0.1 (32 layers, 8 KV heads per layer, GQA architecture)
- **Metric**: R_V = PR(V_late) / PR(V_early) where PR = participation ratio from SVD
- **Window**: Last 32 tokens (optimal from previous experiments)
- **Sample size**: n=20-40 prompts per condition (for statistical power)
- **Statistical threshold**: p<0.01 with Bonferroni correction, Cohen's d ≥ 0.5

**Key constraint**: We've already tried full residual stream patching (0% effect) and V-only patching (~10% behavior). The mode lives in KV cache structure, not raw activations.

---

## What We've Already Tried (That Didn't Work)

| Intervention | Effect on R_V | Effect on Behavior | Verdict |
|--------------|---------------|-------------------|---------|
| Full residual stream patching | 0% | 0% | ❌ Mode recomputes |
| V-only patching | Transfers geometry | ~10% behavior | ⚠️ Partial |
| Q+K+V attention block (single layer) | Minimal | Minimal | ❌ Too coarse |
| Head ablation at L27 (heads 25-27) | **NONE** | 80% reduction | ✅ Speakers confirmed |
| KV patching L0-16 | ~0% | ~0% | ❌ Too early |
| KV patching L16-32 | ~50% R_V | ~80% behavior | ✅ Mode in late layers |

**Key insight**: The mode requires KV cache structure (attention patterns), not just activations.

---

## Specific Questions

### 1. Ablation Strategy
**Question**: How should we ablate candidate heads to test necessity?
- Option A: Zero-out head outputs (standard ablation)
- Option B: Replace with baseline head outputs
- Option C: Project out the "contraction direction" from head outputs
- Option D: Something else?

**Concern**: Standard zero-ablation might break the model. Should we use "mean ablation" (replace with baseline mean) instead?

### 2. Activation Patching Strategy
**Question**: For sufficiency testing, what exactly should we patch?
- Option A: Patch head's V-projection output (post-attention)
- Option B: Patch head's Q/K/V projections separately
- Option C: Patch the attention pattern (QK^T scores)
- Option D: Patch the full head output (including residual connection)

**Context**: V-only patching transferred geometry but minimal behavior. Should we patch Q/K together?

### 3. Minimal Sufficient Set
**Question**: If multiple candidate heads show high ΔPR, how do we find the minimal set?
- Option A: Greedy forward selection (add heads one by one, measure improvement)
- Option B: Greedy backward elimination (remove heads one by one, measure degradation)
- Option C: Test all combinations (combinatorial, but thorough)
- Option D: Use Shapley values to quantify individual contributions

**Constraint**: We want to find 3-5 heads max (the "microphone"), not a distributed network.

### 4. Control Experiments
**Question**: What controls do we need to rule out confounds?
- Control A: Ablate random heads at same layers (should NOT affect R_V)
- Control B: Patch heads from baseline→recursive (should NOT create contraction)
- Control C: Patch heads at wrong layers (L10, L25) - should NOT work
- Control D: Test on held-out prompts (generalization)

### 5. Measurement Protocol
**Question**: Where should we measure R_V after intervention?
- Option A: At the same layer as intervention (L14/L18)
- Option B: At downstream layer (L27, where speakers are)
- Option C: At both (check propagation)
- Option D: Track R_V across all layers (full trajectory)

**Rationale**: If heads at L18 create contraction, R_V should drop at L18 AND propagate to L27.

---

## Success Criteria

We'll consider the "microphone" found if:

1. **Necessity**: Ablating candidate heads increases R_V by ≥15% (eliminates contraction)
2. **Sufficiency**: Patching ONLY candidate heads transfers ≥50% of R_V contraction
3. **Isolation**: Minimal set of 3-5 heads sufficient (not distributed)
4. **Specificity**: Controls show no effect (random heads, wrong layers)
5. **Mechanism**: Can explain HOW these heads create contraction (attention patterns, QK circuits)

---

## What Would Help Most

1. **Concrete code example** for head-level activation patching in Mistral-7B (GQA architecture)
2. **Statistical test design** for comparing R_V distributions (paired t-test? Mann-Whitney? Bootstrap?)
3. **Ablation best practices** - what's the right way to ablate without breaking the model?
4. **Attention pattern analysis** - how to visualize what these heads are attending to?
5. **Path tracing** - if these heads create the mode, how does it flow to L27 speakers?

---

## Additional Context

**Why this matters**: Finding the "microphone" completes the circuit: **Source heads (L14-L18) → KV contraction → Speaker heads (L27) → Recursive output**

This would be the first complete mechanistic explanation of how transformers process self-reference, with implications for:
- AI safety (detecting internal self-modeling)
- Interpretability (mapping self-referential circuits)
- Consciousness debates (functional self-models in LLMs)

**Timeline**: We have 1-2 days to complete this phase. Need actionable, implementable guidance.

---

**Please provide:**
1. Specific experimental protocol (step-by-step)
2. Code snippets for head-level interventions (ablation + patching)
3. Statistical analysis plan
4. Expected outcomes and how to interpret them
5. Red flags that indicate we're on the wrong track

Thank you!

file 3: 
# Finding the Microphone: Phase 2 - Proving Causality

## What We Have
- ✅ Top candidate heads identified via per-head ΔPR at L14-L18
- ✅ Speaker heads confirmed: L27 heads 25-27 (80% behavior, 0% R_V change)
- ✅ KV transfer works (100% success with last-32 tokens)

## What We Need
**Prove candidate heads are CAUSAL (necessary + sufficient) for R_V contraction.**

## The Core Question
**After identifying candidate heads from ΔPR analysis, what's the fastest way to prove they CREATE the recursive mode?**

---

## Specific Technical Questions

### 1. Ablation Method
Zero-out vs mean-ablation vs projection-removal? Which is best for testing necessity without breaking the model?

### 2. Patching Target
What exactly to patch for sufficiency test?
- V-projection only? (transferred geometry but ~10% behavior)
- Q+K+V together? (might transfer full mode)
- Attention pattern (QK^T)? (tests relational structure)

### 3. Minimal Set Finding
Greedy forward/backward vs Shapley values vs combinatorial? Need 3-5 heads max.

### 4. Measurement Protocol
Measure R_V at intervention layer (L18) or downstream (L27) or both?

### 5. Statistical Design
Paired t-test? Bootstrap? Effect size threshold? Sample size?

---

## Success Criteria
1. Ablation → R_V increases ≥15% (necessity)
2. Patching → R_V transfers ≥50% (sufficiency)  
3. Minimal set: 3-5 heads (not distributed)
4. Controls: Random heads, wrong layers show no effect

---

## Constraints
- Mistral-7B-Instruct (GQA: 8 KV heads, 32 Q heads)
- Mode lives in KV cache structure (not raw activations)
- Window: 32 tokens, n=20-40 prompts, p<0.01, d≥0.5

---

## What I Need
1. **Step-by-step protocol** (ablation → patching → analysis)
2. **Code snippets** for head-level interventions (Mistral GQA)
3. **Statistical plan** (tests, corrections, power analysis)
4. **Interpretation guide** (what results mean, red flags)

**Timeline: 1-2 days. Need actionable, implementable guidance.**

---

## Why This Matters
Completes the circuit: **Source heads (L14-L18) → KV contraction → Speakers (L27) → Output**

First mechanistic explanation of self-reference in transformers.



4: