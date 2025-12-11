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
