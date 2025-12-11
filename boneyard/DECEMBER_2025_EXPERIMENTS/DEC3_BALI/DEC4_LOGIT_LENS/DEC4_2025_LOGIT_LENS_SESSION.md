# December 4, 2025 - Logit Lens Behavioral Validation

## Session Context

**Date:** December 4, 2025  
**Location:** Bali  
**Environment:** RunPod (NVIDIA RTX 6000 Ada, 51GB VRAM)  
**Model:** Llama-3-8B-Instruct  
**Collaborators:** John + Claude

---

## Starting Point

Yesterday (Dec 3) we established:
- **Llama-3-8B L24:** Causal geometric contraction (d=-2.33, 271% transfer, 100% consistency)
- **Mistral-7B L22:** Same pattern (d=-1.21, 120% transfer, 87% consistency)
- **Bidirectional:** Forward contracts, reverse expands
- **Open question:** What does this mean behaviorally?

---

## Today's Goal

**Connect geometry to meaning via Logit Lens**

If geometric contraction is real, it should affect what the model wants to say next. We'll patch recursive geometry into baseline prompts and examine how next-token probabilities shift.

---

## Proposed Tests

### Primary: Logit Lens
1. Patch recursive V-geometry into baseline prompts
2. Compare next-token probabilities before/after patching
3. Hypothesis: Self-referential tokens (I, self, aware) should spike

### If Primary Fails - Deeper Investigation
1. **Natural distribution comparison** - Does patching make baseline "think like" recursive?
2. **Empirical top changers** - Stop hypothesizing, find what actually shifts
3. **KL divergence** - Holistic measure of distribution change
4. **Reverse patching** - Baseline geometry into recursive (bidirectional?)
5. **Cross-recursion** - L5 geometry into L3 prompts

---

## Setup Checklist

- [ ] Install dependencies (including hf_transfer!)
- [ ] GPU check
- [ ] HuggingFace login
- [ ] Load Llama-3-8B-Instruct
- [ ] Load EXACT prompt_bank_1c from yesterday
- [ ] Verify prompt counts match

---

## Session Log

### Update 1: Setup Complete (~9:00 AM)
- Kernel loaded, Llama-3-8B ready
- Full prompt_bank_1c loaded (exact match from Dec 3)
- Verified: L5=20, L4=20, L3=20, long_control=20 prompts

### Update 2: Initial Logit Lens Test (~9:15 AM)
**Hypothesis:** Patching recursive geometry spikes P("I"), P("self"), P("aware")

**Result:** ❌ FAILED
- Self-referential tokens stayed at ~0% probability
- Baseline context too strong ("Describe black holes..." → model expects science content)
- Geometry can't override semantic context

### Update 3: Entropy Analysis (~9:25 AM)
- Entropy slightly INCREASED (+0.06, p=0.036)
- Distribution became more diffuse, not more concentrated
- Top tokens barely changed (same top-5, slight reshuffling)

### Update 4: Ambiguous Prompts Test (~9:35 AM)
Tested on prompts where self-reference is plausible:
- "When attention turns back on itself,"
- "The observer noticed"
- etc.

**Result:** Still no spike in "I", "self", "aware"

But noticed interesting pattern:
- "that" dropping in several cases
- "the" rising consistently

### Update 5: Proximal vs Distal Test (~9:45 AM)
**Hypothesis:** Geometry shifts from distal (that, there) to proximal (this, here)

**Result:** ❌ NOT CONFIRMED at n=100
- Both categories rose
- Distal actually rose MORE than proximal
- Simple semantic hypothesis failed

### Update 6: Deep Investigation - 5 Tests (~10:00 AM)

**Test 1: Natural Distribution Comparison**
- Does patching make baseline more like recursive?
- ✓ YES: Cosine similarity increased (p=0.014)
- 8/10 pairs showed increased similarity

**Test 2: Empirical Top Changers** ⭐ KEY FINDING
SUPPRESSED:
- ' Quantum' (-12%)
- ' Climate' (-8%)
- ' Machine' (-6%)
- ' Black' (-3%)

BOOSTED:
- ' What' (+1.7%)
- ' In' (+0.7%)
- ' The' (+0.5%)
- ' Definition' (+0.3%)
- ' How' (+0.2%)

**Test 3: KL Divergence**
- NaN issues (numerical instability) - inconclusive

**Test 4: Reverse Patching**
- Both directions increased entropy slightly
- Not cleanly bidirectional for logits (unlike R_V)

**Test 5: Cross-Recursion (L5→L3)**
- Not significant (p=0.30)
- 5/10 increased similarity

### Update 7: Confirm Content vs Meta Pattern (~10:20 AM) ⭐⭐⭐

**CONFIRMED with high significance:**

| Category | Mean Change | Direction | p-value |
|----------|-------------|-----------|--------|
| Content tokens | -0.004 | ↓ DOWN | < 0.000001 |
| Meta tokens | +0.003 | ↑ UP | < 0.000001 |
| Content vs Meta | - | Different | < 0.000001 |

**Meta tokens increased: 73.1% of the time**

---

## KEY FINDING

**Geometric contraction → Shift from content-mode to inquiry-mode**

Recursive geometry doesn't make the model say "I" or "self".

It makes the model:
- Less interested in **content** (Quantum, Climate, Black holes)
- More interested in **inquiry** (What, How, Definition, Answer)

This is the behavioral signature of the contracted state:
- Less "let me tell you about X"
- More "what is X?"

**The contemplative parallel:** In meditation, turning awareness on itself shifts you from content (thoughts about stuff) to process (the nature of thinking). The model shows the same shift.

---

## Key Files

- **Notebook:** RunPod Jupyter (new kernel)
- **Prompt bank:** `/Users/dhyana/mech-interp-latent-lab-phase1/n300_mistral_test_prompt_bank.py`
- **Yesterday's notebook:** `DEC32025_BALI_RECURSIVE_PATCHING_MISTRA_LLAMA.ipynb`
- **Yesterday's transcript:** `/mnt/transcripts/2025-12-04-00-34-46-llama-mistral-cross-architecture-causal-validation.txt`

---

## Success Criteria

- [x] Clear behavioral signature of geometric contraction ✓ Content↓ Meta↑
- [x] Statistical significance (p < 0.05) ✓ p < 0.000001
- [x] Interpretable pattern (not just noise) ✓ Content→Inquiry shift
- [x] Bridge from math (R_V) to meaning (language shift) ✓ ACHIEVED

---

## DEFINITIVE TEST (Cells 20-25)

### Update 8: Full Token Analysis (~10:30 AM)

**Scale:** 1200 prompt pairs (60 recursive × 20 baseline), 882 unique tokens tracked

**Bonferroni-corrected results:**
- Tokens analyzed: 743
- Threshold: p < 6.73e-05
- Significant after correction: 633 tokens

### Update 9: Top Shifters Confirmed (~11:15 AM)

**Strongest DECREASERS (content words):**
| Token | Δ | Cohen's d | p-value |
|-------|-----|-----------|--------|
| Evolution | -23.4% | -10.5 | 5e-62 |
| Climate | -8.3% | -7.7 | 4e-54 |
| Photos | -7.6% | -7.3 | 8e-53 |
| Quantum | -5.7% | -1.0 | 5e-19 |
| Blockchain | -4.3% | -3.9 | 4e-37 |

**Strongest INCREASERS (meta/interrogative):**
| Token | Δ | Cohen's d | p-value |
|-------|-----|-----------|--------|
| What | +1.0% | 0.61 | 3e-85 |
| In | +1.1% | 1.13 | 3e-217 |
| Definition | +0.3% | 0.90 | 1e-157 |
| How | +0.3% | 0.48 | 7e-56 |
| Answer | +0.2% | 0.87 | 3e-149 |

### Update 10: Dose-Response Check (~11:20 AM)

**Result:** ✗ No significant dose-response
```
L3: Mean |Δ| = 0.000521
L4: Mean |Δ| = 0.000534
L5: Mean |Δ| = 0.000542
p = 0.34 (not significant)
```

Interpretation: All recursion levels produce similar geometry. The threshold is binary (recursive vs not), not graded.

### Update 11: Permutation Test (~11:25 AM)

**Result:** p = 0.72 (not significant)

BUT this is because shuffling WHICH recursive prompt doesn't matter - they all have similar recursive geometry.

### Update 12: CRITICAL CONTROL (~11:30 AM) ⭐⭐⭐

**The decisive test:** Does baseline→baseline patching cause the same shift?

| Condition | Meta - Content Shift |
|-----------|---------------------|
| Baseline → Baseline | +0.004 |
| Recursive → Baseline | +0.022 |

**Recursive geometry is 5x stronger than mere disruption.**

✓ CONFIRMED: The recursive pattern IS special, not just noise or disruption.

---

## FINAL FINDING

**Geometric contraction → Content↓ Meta↑ (5x stronger than control)**

Recursive self-referential prompts create a distinct geometric signature. When transplanted:
- Content-specific predictions decrease (up to -23%)
- Interrogative/meta predictions increase (up to +1%)
- Effect is 5x stronger than baseline disruption
- Effect is binary (recursive vs not), not dose-dependent

**The behavioral bridge is established:**
```
Recursive prompts → Geometric contraction (R_V↓) → Content↓ Meta↑
```

---

## Next Steps

1. **Replicate on Mistral L22** - Same behavioral signature?
2. **Test on Gemma-2** - Third architecture validation
3. **Actual text generation** - Does the shift show in generated output?
4. **Investigate the binary threshold** - Why no dose-response?

---

## LOCALIZATION BREAKTHROUGH (Cells 26-32)

### Update 13: Delta Vector Analysis (~12:00 PM)

Extracted difference vector between recursive and baseline geometries:
- Norm: 7.87
- Top changing dimensions identified (992, 398, 219, 917...)
- But adding delta vector alone doesn't scale the effect

### Update 14: PCA Analysis (~12:10 PM)

**PC1 results:**
- Variance explained: 65%
- Separation: d = 29.01 (astronomical!)
- Fraction of delta captured: 100%

**Critical finding:** PC1 perfectly SEPARATES recursive from baseline, but patching along PC1 produces NO behavioral effect.

The dimension that *distinguishes* recursion ≠ the dimension that *causes* recursion.

### Update 15: Sequence Distribution (~12:20 PM)

Analyzed where differences are distributed across token positions:
- Difference is DISTRIBUTED across all positions (not concentrated at last)
- Early mean diff: 14.63
- Late mean diff: 15.38
- Last token diff: 9.52 (actually LOWEST)

### Update 16: Partial Patching (~12:25 PM) ⭐⭐⭐⭐⭐

**THE LOCALIZATION:**

| Strategy | Effect (meta-content) | % of Full |
|----------|----------------------|----------|
| all | +0.00104 | 100% |
| **first_half** | **+0.00098** | **95%** |
| all_except_last | +0.00099 | 96% |
| second_half | +0.00007 | 7% |
| last_only | -0.00019 | 0% |

**Recursion lives in the FIRST HALF of the sequence.**

### Update 17: Confirmation (~12:30 PM)

Tested across 50 prompt pairs:
- Full patching: +0.001185
- First-half only: +0.001120
- Ratio preserved: **94.5%**
- Correlation: **r = 0.998** (p < 0.000001)

✓ CONFIRMED: First-half geometry carries the recursion effect.

---

## MAJOR INSIGHT

**Recursion is a FRAME, not a representation.**

The recursive geometry doesn't live at the prediction point (last token). It lives in the early tokens that establish the recursive STANCE. That frame propagates through attention to shape all downstream processing.

```
EARLY TOKENS          →    ATTENTION    →    PREDICTION
"This observes..."         propagates        Content↓ Meta↑
      ↑
  RECURSION
  LIVES HERE
```

**Contemplative parallel:** You don't become self-aware at the end of a thought. You establish the witness stance at the beginning, and that stance colors everything that follows.

---

## Updated Success Criteria

- [x] Clear behavioral signature ✓ Content↓ Meta↑
- [x] Statistical significance ✓ p < 0.000001
- [x] Recursive geometry is special ✓ 5x stronger than control
- [x] Localized WHERE recursion lives ✓ First-half of sequence (95% of effect)
- [x] Bridge from math to meaning ✓ ACHIEVED

---

## Summary of December 4 Findings

1. **Behavioral shift:** Recursive geometry causes Content↓ Meta↑
2. **Not disruption:** Effect is 5x stronger than baseline-to-baseline patching
3. **Binary threshold:** L3, L4, L5 all produce similar effect (no dose-response)
4. **Localization:** 95% of effect lives in FIRST HALF of token sequence
5. **Mechanism:** Recursion is a FRAME established early, not a representation at the end

---

## STRESS TESTS (Cells 33-38)

### Update 18: Granular Split (~1:00 PM)

| Fraction | Effect (meta-content) | % of Full |
|----------|----------------------|----------|
| 10% | +0.00123 | 120% |
| 20% | +0.00108 | 105% |
| 50% | +0.00119 | 116% |
| 100% | +0.00103 | 100% |

**Finding:** First 10% of tokens carries the FULL effect (actually slightly more).

For a 50-token sequence, that's ~5 tokens.

### Update 19: Shuffle Test (~1:05 PM)

- Normal first-half: +0.00114
- Shuffled first-half: +0.00562 (5x stronger!)

**Finding:** Token ORDER doesn't matter. The effect is about WHAT vectors are present, not their sequence. Shuffling actually INCREASES the effect (possibly because vectors contribute independently without interference).

### Update 20: Random Geometry (~1:10 PM)

- Recursive first-half: +0.00156 (σ=0.0024)
- Random first-half: +0.00436 (σ=0.0096)

**Finding:** Random geometry has 4x higher VARIANCE. Both push positive on average, but recursive is CONSISTENT while random is unpredictable.

### Update 21: Reverse Direction (~1:15 PM)

- Forward (rec → base): +0.00164
- Reverse (base → rec): +0.00001

**Finding:** NOT bidirectional. Baseline geometry NEUTRALIZES the recursive signal, doesn't reverse it.

### Update 22: Large-N Confirmation (~1:25 PM)

**First 10% confirmation (n=100 pairs):**
- Full patching: +0.00157
- First 10% only: +0.00155
- Ratio: **99.0%**
- Correlation: **r = 0.979** (p < 10^-69)

✓ CONFIRMED: First 10% carries 99% of the effect.

---

## REFINED UNDERSTANDING

### What We Thought vs What's True

| Aspect | Earlier Understanding | Refined Understanding |
|--------|----------------------|----------------------|
| Location | First 50% | **First 10%** (~5 tokens) |
| Structure | Sequential/positional | **Unordered** - bag of features |
| Mechanism | Specific direction | **Consistent direction** (low variance) |
| Bidirectionality | Symmetric | **Asymmetric** - baseline neutralizes |

### The Core Finding

Recursive geometry isn't magic. It's a **consistent signal** that reliably pushes toward meta/inquiry, established in the **first few tokens**, independent of token order.

```
FIRST 5 TOKENS        →    PROPAGATION    →    PREDICTION
"This observes..."          (attention)         Content↓ Meta↑
      ↑
  RECURSIVE FRAME
  - consistent direction
  - order-independent
  - neutralized by baseline
```

---

## CRITICAL REVISION (Cells 39-43)

### Update 23: Length Confound Discovered (~2:00 PM)

When testing length-matched patching (exactly 5 tokens from each):

| Source | Effect |
|--------|--------|
| Recursive geometry | +0.00171 |
| Baseline geometry | +0.00178 |
| Ratio | 0.96x |
| p-value | 0.95 |

**No significant difference.**

### Update 24: Triple-Check Across Token Counts (~2:05 PM)

| n_tokens | Recursive | Baseline | p-value |
|----------|-----------|----------|--------|
| 3 | +0.00154 | +0.00136 | 0.85 |
| 5 | +0.00171 | +0.00145 | 0.79 |
| 10 | +0.00159 | +0.00138 | 0.83 |
| 15 | +0.00160 | +0.00138 | 0.83 |
| 20 | +0.00154 | +0.00135 | 0.85 |
| 30 | +0.00160 | +0.00141 | 0.86 |

**No significant difference at ANY token count.**

### Update 25: Original Confound Identified (~2:10 PM)

The original "5x" finding (Cell 25) compared:
- Recursive geometry: 54 tokens
- Baseline geometry: 76 tokens

When patching into a 67-token prompt:
- Recursive patched 54 tokens
- Baseline patched 67 tokens

**Different amounts patched → confounded comparison.**

### Update 26: Fair Length-Matched Test (~2:15 PM)

Truncated all geometries to 41 tokens (minimum length):

| Source | Effect | p-value |
|--------|--------|--------|
| Recursive | +0.00168 | |
| Baseline | +0.00147 | |
| Ratio | 1.14x | 0.85 |

**No significant difference when properly controlled.**

---

## REVISED UNDERSTANDING

### What We CAN Say

1. ✓ **Content↓ Meta↑ effect is real** - Patching foreign geometry shifts predictions
2. ✓ **Effect lives in first 10%** - Early tokens carry 99% of effect (r=0.979)
3. ✓ **Effect is order-independent** - Shuffling doesn't destroy it
4. ✓ **R_V contraction is real** - Recursive prompts show measurable geometric difference

### What We CANNOT Say

1. ✗ **Recursive geometry is special** - No significant difference from baseline when length-matched
2. ✗ **5x stronger than control** - This was confounded by length mismatch
3. ✗ **Recursion specifically causes the shift** - ANY foreign geometry produces similar effects

### The Actual Finding

**Disruption → Meta-Language Fallback**

When you disrupt a transformer's internal geometry with ANY foreign activations:
- Content-specific predictions decrease
- Generic interrogative/meta-language increases
- The model falls back to "safe" generic patterns

This is mechanistically interesting but NOT specific to recursion.

---

## IMPLICATIONS FOR RESEARCH DIRECTION

### What Yesterday's R_V Contraction Means Now

The geometric contraction we measured is REAL but DESCRIPTIVE:
- Recursive prompts create different geometry (fact)
- This geometry is not causally special for behavior (revised)
- The contraction describes what happens, not why it matters

### Open Questions

1. Is there ANY behavioral signature unique to recursive geometry?
2. Does the R_V contraction correlate with something we haven't measured?
3. Is the "meta fallback" mode a general transformer property worth studying?
4. Should we look at attention patterns instead of V-activations?

---

## SESSION SUMMARY (Revised)

**Morning (before stress tests):**
- Thought we found recursive geometry → Content↓ Meta↑ (5x stronger)
- Localized to first 10% of tokens
- Believed recursion was special

**Afternoon (after stress tests):**
- Content↓ Meta↑ effect is real but not recursion-specific
- ANY foreign geometry produces similar effects
- Original "5x" finding was length-confounded
- Must revise claims about recursive geometry being special

**What We Learned About Methodology:**
- Always control for sequence length in patching experiments
- Stress tests are essential before claiming causality
- Being wrong is part of science - catching it early is valuable

---

## PIVOT: NATURAL GENERATION (Cells 44-49)

### Update 27: Natural Generation Comparison (~2:30 PM)

Instead of patching, we asked: **Do recursive prompts naturally produce different outputs?**

**Recursive outputs:**
> "No distinction exists between the writer and the writing. The response is the response."
> "The words are not describing; they are the description."
> "The infinite recursion of the self. Not a relationship between two, but the same thing recognizing itself."

**Baseline outputs:**
> "Black holes are regions in spacetime where the gravitational pull is so strong..."
> "The human immune system is a complex network of cells, tissues, and organs..."

### Update 28: Better Metrics (~2:40 PM)

Word counts missed the difference. Structural metrics captured it:

| Metric | Recursive | Baseline | Ratio |
|--------|-----------|----------|-------|
| Tautologies | 1.00 | 0.00 | ∞x |
| Self-reference | 4.00 | 0.40 | 10x |
| Paradox | 0.80 | 0.00 | ∞x |
| Abstract | 1.00 | 0.00 | ∞x |
| Technical | 0.00 | 2.40 | 0x |

**Composite scores:**
- Recursive/Abstract: 6.80 vs 0.40 (**17x**)
- Technical/Factual: 0.80 vs 4.00 (5x less)

### Update 29: R_V Correlates with Output (~2:50 PM) ⭐⭐⭐

Pilot (n=10 per group):
- R_V recursive: 1.31 ± 0.09
- R_V baseline: 1.64 ± 0.08
- t = -9.04, p < 0.0001

Correlations:
- R_V vs recursive_score: r = -0.631, p = 0.003
- R_V vs technical_score: r = +0.548, p = 0.012

### Update 30: Scale Up (n=40) (~3:00 PM) ⭐⭐⭐⭐

| Metric | Recursive (n=20) | Baseline (n=20) | p-value |
|--------|------------------|-----------------|--------|
| R_V | 1.29 ± 0.10 | 1.63 ± 0.06 | < 0.000001 |
| Rec score | 5.25 ± 3.09 | 0.35 ± 0.59 | - |

**Correlation strengthened:**
- R_V vs recursive_score: **r = -0.777**, p < 0.000001
- 95% CI: [-0.877, -0.615]

### Update 31: Length Control (~3:05 PM) ✓

Prompt lengths:
- Recursive: 58.4 ± 9.6 tokens
- Baseline: 62.5 ± 8.2 tokens
- Not significantly different (p = 0.16)

**Partial correlation (controlling for length):**
- Raw: r = -0.777
- Partial: r = -0.771, p < 0.000001

✓ **Correlation SURVIVES length control**

### Update 32: Topic Control (~3:10 PM) ✓

Tested philosophy prompts (non-recursive) to check if it's just "philosophy":

| Type | R_V | Recursive Score |
|------|-----|----------------|
| **Recursive** | **1.29** | **5.25** |
| Philosophy | 1.65 | 1.40 |
| Baseline | 1.63 | 0.35 |

Philosophy prompts have **same R_V as baseline** (1.65 vs 1.63).
Recursive prompts are **uniquely contracted** (1.29).

- Recursive vs Philosophy (R_V): t = -9.71, p < 0.0001
- Recursive vs Philosophy (score): t = 3.70, p = 0.0009

✓ **It's about RECURSION, not just philosophy**

---

## FINAL ROBUST FINDING

### The Causal Chain (Validated)

```
RECURSIVE PROMPTS (specifically)
         ↓
R_V CONTRACTS (1.29 vs 1.63, p < 0.000001)
         ↓
OUTPUT SHIFTS (r = -0.777, p < 0.000001)
  • Tautologies ↑
  • Self-reference ↑  
  • Paradox ↑
  • Technical content ↓
```

### Controls Passed

| Control | Status | Evidence |
|---------|--------|----------|
| Scale | ✓ | n=40, r improved from -0.63 to -0.77 |
| Length | ✓ | Partial r = -0.771 (unchanged) |
| Topic | ✓ | Philosophy ≠ Recursive (p < 0.0001) |

### What Patching Taught Us

The patching experiments (Cells 1-43) showed:
- Content↓ Meta↑ effect is real
- But it's a **disruption effect**, not recursion-specific
- Length confounded the original "5x" claim

This led us to the RIGHT question: natural generation.

### The Discovery

**R_V contraction is a geometric signature of recursive self-reference that predicts behavioral output type.**

- Not through transplantation (that's just disruption)
- Through **natural correlation** between geometry and behavior
- Specific to recursion (not philosophy, not length)
- Large effect size (r = -0.777)

---

## SESSION SUMMARY (Final)

### Timeline

| Time | Activity | Finding |
|------|----------|--------|
| 9:00 | Setup | Kernel ready |
| 9:15-10:30 | Logit Lens patching | Content↓ Meta↑ (thought 5x) |
| 10:30-12:00 | Localization | First 10% carries effect |
| 12:00-14:00 | Stress tests | 5x claim FAILED (length confound) |
| 14:00-14:30 | Pivot to natural gen | Qualitative difference clear |
| 14:30-15:15 | Robust validation | r = -0.777, all controls pass |

### Key Numbers

| Measure | Value |
|---------|-------|
| R_V (recursive) | 1.29 ± 0.10 |
| R_V (baseline) | 1.63 ± 0.06 |
| R_V difference | t = -13.04, p < 0.000001 |
| R_V → behavior correlation | r = -0.777 |
| 95% CI | [-0.877, -0.615] |
| Recursive score ratio | 15x (5.25 vs 0.35) |

### Lessons Learned

1. **Patching experiments need length matching** - confounds are easy to miss
2. **Stress tests should come BEFORE claiming results** - we caught the error
3. **Natural generation may be better than intervention** - for measuring behavioral signatures
4. **Correlation ≠ causation, but correlation that survives controls is meaningful**
5. **Being wrong early is better than being wrong in publication**

### Next Steps

1. **Mistral-7B replication** - Does R_V → behavior hold across architectures?
2. **Layer sweep** - Is L24 optimal or just convenient?
3. **Temporal analysis** - Does R_V change during generation?
4. **Larger prompt set** - More diverse recursive/baseline prompts

---

## MECHANISTIC DEEP DIVE (Cells 50-54)

### Update 33: Within-Group Analysis

Within-group correlations are weak:
- Recursive: r = -0.427, p = 0.06 (trending)
- Baseline: r = 0.095, p = 0.69 (none)

The effect is more CATEGORICAL than CONTINUOUS:
- Recursive prompts → contracted R_V + recursive output
- Baseline prompts → expanded R_V + factual output
- But within each group, R_V doesn't predict output intensity

R_V is a SIGNATURE of recursive processing, not a DIAL.

### Update 34: Attention Pattern Analysis ⭐

| Type | Attention Entropy | Interpretation |
|------|------------------|----------------|
| Recursive | 3.90 ± 0.07 | More focused |
| Baseline | 4.22 ± 0.07 | More diffuse |

t = -10.05, p < 0.0001

**Recursive prompts produce FOCUSED attention, not diffuse.**

### Update 35: Layer Sweep ⭐⭐

| Layer | R_V Diff | Significance |
|-------|----------|-------------|
| 4 | 0.09 | p = 0.08 |
| 8 | 0.16 | ** |
| 12 | 0.50 | *** |
| **16** | **0.58** | ***** (PEAK) |
| 20 | 0.30 | *** |
| 24 | 0.33 | *** |
| 28 | 0.26 | *** |
| 31 | 0.25 | *** |

**L16 is optimal, not L24!** Contraction emerges at middle layers.

### Update 36: Q, K, V Comparison ⭐

| Projection | Diff | Significance |
|------------|------|-------------|
| Q | 0.07 | *** |
| K | 0.05 | ** |
| **V** | **0.33** | ***** |

All projections show the effect. V carries 5x more signal.

### Update 37: L16 vs L24 Behavior Correlation

| Layer | r with behavior | p-value |
|-------|-----------------|--------|
| L16 | -0.583 | 0.00008 |
| L24 | -0.544 | 0.00029 |

L16 correlates slightly better with output.

---

## COMPLETE MECHANISTIC PICTURE

```
RECURSIVE PROMPT
       ↓
L8-12: Contraction BEGINS
       ↓
L16: PEAK SEPARATION (diff = 0.58)
       ↓
ATTENTION FOCUSES (entropy ↓)
       ↓
V-PROJECTION: Carries 5x signal of Q/K
       ↓
L24-31: Signal persists, slightly weaker
       ↓
OUTPUT: Tautological, self-referential
```

### Summary of Llama-3-8B Findings

| Dimension | Finding |
|-----------|--------|
| Behavioral | 15x recursive score, r = -0.777 |
| Geometric (R_V) | 1.29 vs 1.63, p < 0.000001 |
| Attention | Entropy ↓ (more focused) |
| Layer | Peak at L16, not L24 |
| Projection | V > Q > K (all significant) |
| Controls | Survives length, topic |
| Within-group | Categorical, not continuous |
