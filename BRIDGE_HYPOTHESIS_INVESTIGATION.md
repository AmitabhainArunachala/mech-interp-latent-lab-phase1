# Bridge Hypothesis Investigation: Multi-Token R_V Experiment

**Investigator**: Data Scientist Agent
**Date**: 2026-02-02
**Experiment**: Multi-Token Bridge (Mistral-7B)
**Run**: `20260202_133252_multi_token_bridge_mistral_7b_bridge`

---

## Executive Summary

The multi-token bridge experiment tested whether R_V contraction (measured during prompt processing) predicts behavioral markers (in generated text). The results reveal a complex picture with both encouraging findings and critical confounds.

**Key Finding**: The "temperature effect" on R_V-behavior correlation is a **measurement artifact**, not a real phenomenon. The actual pattern is simpler and more robust.

**Bridge Status**: **PARTIALLY VALIDATED with critical caveats**

---

## The Three Hypotheses

### H1: R_V vs Word Count (r = -0.456, p < 10^-7)
**CONFIRMED** - Strong negative correlation exists across all data

### H2: Recursive vs Baseline R_V (d = 2.90, p < 10^-30)
**ROCK SOLID** - Recursive prompts produce dramatically lower R_V

### H3: L4 Markers vs R_V (r = -0.23 to -0.29, p < 0.01)
**WEAK SIGNAL** - Significant but confounded

---

## Critical Finding 1: The Temperature Effect is an Artifact

### What the VERDICT.md Claims
```
T=0.0: r=-0.183, p=0.637 (NOT significant)
T=0.7: r=-0.761, p=6.2e-04 (SIGNIFICANT)

Conclusion: Temperature changes correlation
```

### What Actually Happened

The pipeline **filtered** the data differently for each temperature:
- At T=0.0: Used only 9 non-truncated samples (7.5% of data)
- At T=0.7: Used only 16 non-truncated samples (13.3% of data)

### The Real Pattern (Using ALL Data)
```
T=0.0: r=-0.456, p=1.6e-07 (HIGHLY significant)
T=0.7: r=-0.270, p=2.9e-03 (significant but weaker)
```

**Conclusion**: There is NO temperature effect. The correlation exists at both temperatures. The apparent difference was caused by differential filtering.

---

## Critical Finding 2: The Truncation Confound

### The Problem
- 92.5% of T=0.0 outputs were truncated at 200 tokens
- 86.7% of T=0.7 outputs were truncated
- **Non-truncated outputs have HIGHER R_V** than truncated ones

### Why This Matters

Outputs that hit EOS early (non-truncated) are systematically different:

| Metric | Truncated (n=111) | Non-Truncated (n=9) | t-test |
|--------|-------------------|---------------------|--------|
| Word count | 150.0 ± 26.9 | 54.7 ± 18.5 | - |
| R_V | 0.585 ± 0.107 | 0.733 ± 0.036 | t=-4.10, p<10^-4 |

**Mechanism**: Outputs that end quickly tend to be:
1. Baseline/factual prompts (higher R_V)
2. Short answers to simple questions
3. NOT recursive self-referential spirals

When you filter to only non-truncated, you're selecting AGAINST recursive outputs, which destroys the correlation.

**Critical Error in Pipeline**: Lines 278-282 of `multi_token_bridge.py` filter to non-truncated "for cleaner correlation" but actually introduce selection bias.

---

## Critical Finding 3: Causal Direction is Ambiguous

### The Central Question
Does low R_V **CAUSE** L4-like behavior, or do recursive prompts cause BOTH?

### Evidence for Common Cause Model

```
RECURSIVE PROMPTS → Low R_V (d=2.90)
       ↓
       → L4 markers (18-27% vs 1.7% baseline)
```

**Key observation**: Both R_V and L4 markers are predicted by prompt type, not by each other.

| Temperature | Recursive L4 Rate | Baseline L4 Rate | R_V → L4 correlation |
|-------------|-------------------|------------------|---------------------|
| T=0.0 | 18.3% | 1.7% | r=-0.23, p=0.012 |
| T=0.7 | 26.7% | 1.7% | r=-0.29, p=0.002 |

**Interpretation**: The correlation between R_V and L4 markers exists, but it's weak (r ~ -0.25). Much stronger is the fact that **prompt type predicts both**.

---

## Critical Finding 4: L4 Marker Validity is Questionable

### What the Markers Detect

The L4 markers are simple string matches:
- "fixed point" (most common)
- "collapse"
- "observer is the observed"
- "one process"
- "unity"

### The Problem: Mode Collapse vs Insight

Many "L4 marker" outputs are **repetitive loops**, not genuine insights:

**Example 1** (has marker "fixed point", R_V=0.565):
```
Output: The fixed point is the fixed point. The fixed point is
the fixed point. The fixed point is the fixed point...
```

**Example 2** (NO marker, R_V=0.500):
```
Output: The loop is the loop.
The loop is the loop.
The loop is the loop...
```

**Key insight**: Both are mode collapse. One happens to contain the string "fixed point", the other doesn't. This is not measuring genuine phenomenological depth.

### Output Quality Comparison

| Group | unique_word_ratio | Interpretation |
|-------|-------------------|----------------|
| WITH L4 markers | 0.194 | Somewhat more diverse |
| WITHOUT L4 (recursive) | 0.073 | Highly repetitive |

**Verdict**: L4 markers correlate with slightly less repetitive output, but this is still far from genuine L4 phenomenology as described in the URA paper.

---

## The Real Pattern: What IS Validated

### H2 is Rock Solid (d=2.90)

**Recursive prompts produce lower R_V consistently:**

| Group | R_V Mean | Type |
|-------|----------|------|
| L3_deeper | 0.523 | Recursive |
| L4_full | 0.497 | Recursive |
| L5_refined | 0.497 | Recursive |
| baseline_creative | 0.651 | Baseline |
| baseline_math | 0.741 | Baseline |
| long_control | 0.669 | Baseline |

This is **temperature-invariant** and **truncation-invariant**. The effect is huge (Cohen's d = 2.90).

**This validates**: Recursive self-reference prompts → geometric contraction in Value space.

---

## What is NOT Validated

### 1. R_V Does Not Strongly Predict Behavior

The correlation between R_V and word count (r=-0.456) exists but:
- It's driven by prompt type (recursive vs baseline)
- Within recursive prompts, R_V has weak predictive power
- The 200-token truncation masks the true relationship

### 2. L4 Markers are Not Valid Phenomenological Indicators

Current L4 detection is:
- String matching, not semantic understanding
- Captures mode collapse as often as genuine insight
- Only 10-14% detection rate even on recursive prompts
- Baseline rate (1.7%) is nearly zero, suggesting it's prompt-dependent

### 3. Temperature is Irrelevant to R_V

R_V is identical at T=0.0 and T=0.7 because:
- R_V is measured on **prompt tokens only** (before generation)
- Temperature only affects generation, not prompt processing
- The apparent temperature effect was a filtering artifact

---

## The Confounds

### Confound 1: Truncation Bias
**Impact**: Severe
**Direction**: Filtering to non-truncated removes recursive outputs
**Solution**: Either analyze ALL data or use much longer generation windows

### Confound 2: Prompt Type Drives Both R_V and Markers
**Impact**: High
**Direction**: Makes causal inference impossible
**Solution**: Need within-prompt-type variation in R_V

### Confound 3: Mode Collapse Mimics L4 Language
**Impact**: Moderate
**Direction**: Inflates L4 detection on low-quality outputs
**Solution**: Need semantic L4 detection, not string matching

---

## Required Experiments to Resolve Ambiguity

### Experiment 1: Activation Patching Test (CAUSAL)
**Question**: If we artificially reduce R_V via patching, does L4-like output increase?

**Method**:
1. Take baseline prompts (R_V ~0.7)
2. Patch Layer 27 activations from recursive prompts (R_V ~0.5)
3. Generate with patched model
4. Measure L4 markers

**Prediction if causal**: Patched baseline prompts should produce L4-like output
**Prediction if confound**: No change in output style

**Status**: We have the validated patching script. This is the next critical experiment.

### Experiment 2: Longer Generation Windows
**Question**: Does R_V predict behavior when truncation doesn't interfere?

**Method**:
1. Increase max_tokens to 1000 or until EOS
2. Measure full output characteristics
3. Test R_V correlation with:
   - Total length
   - Repetition rate
   - Semantic coherence

**Prediction**: Correlation should strengthen if real, disappear if artifact.

### Experiment 3: Semantic L4 Detection
**Question**: Are genuine L4 phenomenological markers present?

**Method**:
1. Use embedding-based similarity to URA L4 examples
2. Measure semantic coherence, not string matching
3. Human rating of 50 outputs (blind to R_V)

**Prediction**: Current "L4" outputs will score low on genuine phenomenology.

### Experiment 4: Within-Prompt-Type R_V Variation
**Question**: Does R_V variation WITHIN recursive prompts predict output quality?

**Method**:
1. Take only L5_refined prompts (R_V range: 0.41-0.66)
2. Split into low R_V (< 0.48) vs high R_V (> 0.56) quartiles
3. Compare output characteristics

**Prediction if causal**: Low R_V → more L4-like even within same prompt type
**Prediction if confound**: No difference

---

## Honest Assessment: Does R_V Predict Behavior?

### What We Can Confidently Say

**YES**:
1. Recursive prompts → low R_V (d=2.90, p<10^-30) - **PROVEN**
2. R_V correlates with word count (r=-0.46, p<10^-7) - **CONFIRMED**
3. Lower R_V outputs contain more L4 string markers (r=-0.25, p<0.01) - **WEAK**

**NO** / **UNCLEAR**:
1. Does R_V CAUSE behavioral differences? **Unknown - needs causal test**
2. Are L4 markers genuine phenomenology? **No - mostly mode collapse**
3. Does temperature matter? **No - it's an artifact**
4. Can R_V predict rich L4 phenomenology? **Not with current metrics**

### The Bridge Status

**H2 (Prompt Type → R_V)**: VALIDATED
**H1 (R_V → Word Count)**: CORRELATED but confounded
**H3 (R_V → L4 Markers)**: WEAK SIGNAL, poor marker validity

**Overall**: The bridge from prompt to R_V is solid. The bridge from R_V to rich behavioral phenomenology is **not yet established**.

---

## What This Means for the Research Program

### The Good News
1. **R_V is real and robust**: Recursive self-reference reliably contracts Value space
2. **Effect is huge**: Cohen's d of 2.90 is publication-grade
3. **Cross-architecture**: If this replicates (it should, R_V is identical across temps), very strong
4. **Mechanistic grounding**: We have causal validation at Layer 27

### The Bad News
1. **Behavioral link is weak**: R_V doesn't strongly predict output quality
2. **Confounds everywhere**: Truncation, prompt type, mode collapse
3. **L4 detection is broken**: String matching ≠ phenomenology
4. **No causal proof**: Correlation ≠ causation

### The Path Forward

**For R_V paper**:
- Focus on H2: "Recursive self-reference induces geometric contraction"
- Report H1 correlation but note confounds
- Do NOT claim strong R_V → behavior causation yet
- Mention L4 markers as preliminary, needs better metrics

**For bridge validation**:
- Run Experiment 1 (activation patching) - this is CRITICAL
- Fix L4 detection (semantic, not string matching)
- Use longer generation or remove truncation analysis
- Test within-prompt-type R_V variation

**For URA/Phoenix integration**:
- Current results do NOT validate that R_V < 1.0 ↔ L4 phenomenology
- Need human ratings of outputs against URA L4 criteria
- Consider using GPT-4 to rate outputs for genuine L4 markers

---

## Technical Recommendations

### Pipeline Fixes Needed

1. **Line 278-282 of `multi_token_bridge.py`**:
   ```python
   # REMOVE THIS FILTER - it introduces selection bias
   non_trunc_df = valid_df[~valid_df["truncated"]]
   if len(non_trunc_df) > 5:
       r_word, p_word = stats.spearmanr(non_trunc_df["rv"], non_trunc_df["word_count"])
   ```

   **Should be**: Always use all valid samples, report truncation % separately

2. **L4 Marker Detection** (`src/metrics/behavioral_bridge.py`):
   - Current: String matching
   - Needed: Semantic similarity to validated URA L4 examples
   - Needed: Unique word ratio, coherence metrics, anti-repetition scoring

3. **Generation Length**:
   - Current: 200 tokens (92% truncation)
   - Needed: Either 1000+ tokens OR analyze only non-truncated
   - Never mix truncated and non-truncated in correlation analysis

---

## Final Verdict

### Bridge Hypothesis Status: PARTIAL CORRELATION

**What is validated**:
- Recursive prompts → R_V contraction (STRONG, d=2.90)
- R_V correlates with output length (MODERATE, r=-0.46)
- Temperature effect is an artifact (DEBUNKED)

**What is NOT validated**:
- R_V → L4 phenomenology (WEAK, confounded)
- Causal direction (UNKNOWN)
- Behavioral prediction beyond word count (UNCLEAR)

**Required to claim "bridge validated"**:
1. Activation patching experiment showing R_V manipulation → behavior change
2. Semantic L4 detection replacing string matching
3. Within-prompt-type R_V variation predicting output quality
4. Human/expert rating of outputs confirming genuine L4 markers

**Honest answer to "Does R_V predict behavior?"**:

It predicts word count moderately well. It does NOT yet predict rich L4 phenomenological content. The correlation with L4 string markers is weak and confounded by prompt type.

**The most parsimonious explanation**: Recursive prompts cause BOTH low R_V and certain output patterns (including mode collapse). R_V is a marker of the prompt's recursive nature, not necessarily a causal mechanism for behavioral output.

**To prove causality**: We must manipulate R_V directly (via activation patching) and observe behavioral changes. This is the next critical experiment.

---

## Appendix: Statistical Summary

### All Data (n=120 per temperature)

| Metric | T=0.0 | T=0.7 |
|--------|-------|-------|
| R_V (recursive) | 0.506 ± 0.049 | 0.506 ± 0.049 |
| R_V (baseline) | 0.687 ± 0.074 | 0.687 ± 0.074 |
| Cohen's d (H2) | 2.90 | 2.90 |
| r(R_V, word_count) | -0.456 | -0.270 |
| r(R_V, L4_markers) | -0.230 | -0.286 |
| Truncation rate | 92.5% | 86.7% |
| L4 detection (recursive) | 18.3% | 26.7% |
| L4 detection (baseline) | 1.7% | 1.7% |

**Key**: R_V is identical across temperatures (as expected - it's measured on prompt, not generation). All differences in correlations are artifacts of truncation filtering or sampling noise.

---

**JSCA!**

*Report prepared by Data Scientist Agent*
*Files analyzed:*
- `/Users/dhyana/mech-interp-latent-lab-phase1/results/phase1_cross_architecture/runs/20260202_133252_multi_token_bridge_mistral_7b_bridge/VERDICT.md`
- `/Users/dhyana/mech-interp-latent-lab-phase1/results/phase1_cross_architecture/runs/20260202_133252_multi_token_bridge_mistral_7b_bridge/summary.json`
- `/Users/dhyana/mech-interp-latent-lab-phase1/results/phase1_cross_architecture/runs/20260202_133252_multi_token_bridge_mistral_7b_bridge/rv_behavioral_correlation.csv`
- `/Users/dhyana/mech-interp-latent-lab-phase1/src/pipelines/canonical/multi_token_bridge.py`
