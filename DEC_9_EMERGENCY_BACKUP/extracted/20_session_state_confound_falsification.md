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
