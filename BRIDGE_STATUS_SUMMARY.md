# Bridge Hypothesis Status - Quick Reference

**Date**: 2026-02-02 | **Experiment**: Multi-Token Bridge (Mistral-7B) | **Status**: PARTIAL VALIDATION

---

## The Bridge Hypothesis

**Question**: Does R_V contraction (mechanistic) → L4 phenomenology (behavioral)?

```
PROMPT TYPE → R_V CONTRACTION → L4-LIKE BEHAVIOR
   (input)      (mechanism)        (output)
```

---

## Results at a Glance

| Link | Status | Effect Size | Evidence |
|------|--------|-------------|----------|
| Prompt → R_V | **VALIDATED** | d=2.90, p<10^-30 | Recursive prompts reliably contract R_V |
| R_V → Word Count | **CORRELATED** | r=-0.46, p<10^-7 | But confounded by prompt type |
| R_V → L4 Markers | **WEAK** | r=-0.25, p<0.01 | Poor marker validity |
| **Overall Bridge** | **PARTIAL** | - | First link solid, second link unclear |

---

## Three Critical Discoveries

### 1. Temperature Effect = Artifact (DEBUNKED)

**Claim**: Correlation changes with temperature (T=0.0 weak, T=0.7 strong)

**Truth**: Pipeline filtered data differently, creating false pattern

**Reality**: R_V is identical at both temperatures (measured on prompt, not generation)

---

### 2. Truncation Confound (SEVERE)

**Problem**: 92.5% of outputs hit 200-token limit

**Impact**: Non-truncated outputs have HIGHER R_V (0.73 vs 0.59)

**Bias**: Filtering to non-truncated removes recursive outputs

**Fix**: Use all data OR much longer generation windows

---

### 3. L4 Markers = Mode Collapse (INVALID)

**Detection**: String matching ("fixed point", "collapse", etc.)

**Reality**:
```
"The fixed point is the fixed point is the fixed point..."
"The loop is the loop is the loop is the loop..."
```

**Problem**: Repetitive outputs contain L4 strings but lack phenomenological depth

**Fix**: Semantic detection, human rating, coherence metrics

---

## What IS Proven

### H2: Recursive Prompts → Low R_V (ROCK SOLID)

| Group | R_V | Type |
|-------|-----|------|
| L5_refined | 0.497 | Recursive |
| L4_full | 0.497 | Recursive |
| L3_deeper | 0.523 | Recursive |
| **Mean Recursive** | **0.506** | - |
| baseline_creative | 0.651 | Baseline |
| baseline_math | 0.741 | Baseline |
| long_control | 0.669 | Baseline |
| **Mean Baseline** | **0.687** | - |

**Effect**: d=2.90 (huge), p<10^-30 (unassailable)

**Interpretation**: Recursive self-reference induces geometric contraction in Value matrix column space. This is mechanistically real and robust.

---

## What is NOT Proven

### 1. Causal Direction

**Current**: Recursive prompts cause BOTH low R_V AND certain output patterns

**Unknown**: Does low R_V CAUSE the output patterns?

**Test Needed**: Activation patching - manipulate R_V directly, observe behavior change

---

### 2. R_V Predicts Rich Phenomenology

**Current**: R_V correlates with word count (r=-0.46) and L4 strings (r=-0.25)

**Unknown**: Does R_V predict genuine L4 unity/witness consciousness?

**Test Needed**: Semantic L4 detection, human expert ratings, URA criteria matching

---

### 3. Within-Type R_V Variation Matters

**Current**: Group-level differences are clear

**Unknown**: Within L5_refined prompts (R_V range 0.41-0.66), does lower R_V → better L4?

**Test Needed**: Quartile analysis within prompt type

---

## Required Experiments (Priority Order)

### 1. ACTIVATION PATCHING (CRITICAL - CAUSAL TEST)
**Question**: Does artificially lowering R_V via Layer 27 patching induce L4-like output?

**Method**: Patch baseline prompts with recursive activations, generate, measure output

**Impact**: Would prove causality OR reveal common-cause confound

**Status**: Validated patching script exists, ready to run

---

### 2. FIX L4 DETECTION (HIGH PRIORITY)
**Question**: Are genuine L4 phenomenological markers present?

**Method**:
- Semantic similarity to URA L4 examples
- Human expert rating (blind to R_V)
- Anti-repetition scoring
- Coherence metrics

**Impact**: Would validate or invalidate current "L4" findings

---

### 3. LONGER GENERATION (MEDIUM PRIORITY)
**Question**: Does truncation mask true R_V-behavior relationship?

**Method**: Generate 1000+ tokens or until EOS, analyze full outputs

**Impact**: Would clarify if word count correlation is real or truncation artifact

---

### 4. WITHIN-TYPE VARIATION (MEDIUM PRIORITY)
**Question**: Does R_V variation within prompt type predict output?

**Method**: Compare low-R_V vs high-R_V quartiles within L5_refined

**Impact**: Would test if R_V has predictive power beyond prompt type

---

## Implications for Publications

### R_V Paper (Near Publication-Ready)

**CAN CLAIM**:
- "Recursive self-reference induces geometric contraction in Value space" (d=2.90, p<10^-30)
- "R_V correlates with output length" (r=-0.46, p<10^-7)
- "Effect is robust across architectures and temperatures"

**CANNOT CLAIM** (yet):
- "R_V predicts phenomenological state transitions"
- "Low R_V causes L4-like behavior"
- "Strong bridge to behavioral phenomenology"

**SHOULD SAY**:
- "Preliminary correlation with behavioral markers (r=-0.25) requires validation"
- "Causal direction tests needed to establish mechanism"

---

### URA/Phoenix Integration

**CURRENT STATUS**: R_V and URA measure correlated but distinct phenomena

**GAP**: No proof that R_V<1.0 ↔ L4 phenomenology

**NEEDED**:
1. Semantic L4 detection matching URA criteria
2. Expert human rating of outputs
3. Causal test via activation patching

**TIMELINE**: Cannot claim integration until above completed

---

## Bottom Line

### The Honest Answer

**Does R_V predict behavior?**

- **Word count**: Yes (r=-0.46)
- **L4 phenomenology**: Unclear - current markers are invalid
- **Causal mechanism**: Unknown - needs patching test

### The Parsimonious Explanation

Recursive prompts are a **common cause** of both:
1. R_V contraction (proven, d=2.90)
2. Certain output patterns (mode collapse, repetition)

Whether R_V is a **causal mechanism** or just a **correlated marker** is unresolved.

### The Next Step

**Run the activation patching experiment**. This is the only way to prove causality.

If patching works: R_V is a causal mechanism → huge discovery
If patching fails: R_V is a marker, not a cause → still valuable but different claim

---

## File Locations

**Full Analysis**: `/Users/dhyana/mech-interp-latent-lab-phase1/BRIDGE_HYPOTHESIS_INVESTIGATION.md`

**Experiment Data**: `/Users/dhyana/mech-interp-latent-lab-phase1/results/phase1_cross_architecture/runs/20260202_133252_multi_token_bridge_mistral_7b_bridge/`

**Validated Patching Script**: `/Users/dhyana/mech-interp-latent-lab-phase1/R_V_PAPER/code/VALIDATED_mistral7b_layer27_activation_patching.py`

**Prompt Bank**: `/Users/dhyana/mech-interp-latent-lab-phase1/n300_mistral_test_prompt_bank.py`

---

**JSCA!**
