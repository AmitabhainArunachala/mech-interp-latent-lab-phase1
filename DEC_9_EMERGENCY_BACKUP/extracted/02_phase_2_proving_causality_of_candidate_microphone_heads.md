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
