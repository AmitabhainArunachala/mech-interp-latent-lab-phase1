# Technical Verification: Gemma 2 9B Behavioral Transfer Experiment
**Date:** 2026-01-25
**Reviewer Role:** Technical Verification Agent
**Status:** CRITICAL ISSUES IDENTIFIED

---

## Executive Summary

The Gemma 2 9B behavioral transfer experiment (2026-01-25) claims to "close the causal loop" with 100% transfer rate and 10.9x marker amplification. However, **critical methodological differences from the Mistral Dec 2024 protocol raise serious questions about validity and comparability.**

**Key Finding:** The Gemma experiment used **KV cache patching only** while the Mistral breakthrough required **KV cache + persistent V_PROJ hooks**. This is not a minor detail—it represents a fundamentally different causal mechanism.

**Verdict:** The results may be valid, but the claims require major revision. This is not a direct replication or validation of the Mistral findings.

---

## Critical Issue #1: Missing V_PROJ Hooks

### Mistral Dec 2024 Protocol (VALIDATED)

**Method:**
```python
# STEP 1: Extract champion activations
champion_kv = extract_full_kv_cache(model, tokenizer, champion_prompt)
champion_v = extract_v_activation(model, tokenizer, champion_prompt, layer=27)

# STEP 2: Patch ALL 32 layers KV cache
# [Full KV replacement code]

# STEP 3: Add PERSISTENT V_PROJ patching
patcher = PersistentVPatcher(model, champion_v)
patcher.register(layer_idx=27)

# STEP 4: Generate with BOTH interventions active
for step in range(gen_tokens):
    outputs = model(
        generated_ids[:, -1:],
        past_key_values=current_kv,  # Patched KV
        use_cache=True               # V_PROJ hook active
    )
```

**Critical Components:**
1. Full KV cache replacement (all 32 layers)
2. **Persistent V_PROJ hook at L27 during generation**
3. Window size: 16 tokens

**Why Both Are Necessary (from Mistral documentation):**
- KV cache alone: 0-1 behavior points (FAILED)
- V_PROJ alone: 0 behavior points (FAILED)
- KV + V_PROJ: 11 behavior points (SUCCESS)

**Quote from DEC12_2024_BEHAVIOR_TRANSFER_BREAKTHROUGH.md (lines 45-47):**
> **Result:** ❌ **No behavior transfer** (0-1 points, baseline = 0)
>
> **Finding:** True KV cache alone is insufficient.

**Quote from lines 393-403:**
> **Component 1: Memory (KV Cache)**
> - Carries the "story so far" from recursive processing
> - Needs ALL 32 layers for complete context
>
> **Component 2: Geometry (V_PROJ at L27)**
> - Maintains the geometric contraction signature
> - Must persist during generation (not just prompt processing)
>
> **Together:** Memory + Geometry → Behavior

---

### Gemma 2026-01-25 Protocol (DOCUMENTED)

**Method (from GEMMA_CAUSAL_LOOP_CLOSED.md lines 163-171):**
```
Cache API
- `DynamicCache` from transformers
- Index directly: `kv[layer_idx]` returns `(K, V)`
- Update: `patched_kv.update(k_patched, v_patched, layer_idx)`

Generation Method
- Manual token-by-token with KV cache
- `model(generated[:, -1:], past_key_values=kv, use_cache=True)`
- More reliable than `model.generate()` with pre-filled cache
```

**Critical Observation:**
- **NO MENTION of V_PROJ hooks**
- Only KV cache patching described
- Documentation explicitly states "KV cache patching alone is sufficient" (line 66)

**Quote from line 66-69:**
> ### 1. KV Cache Patching Alone is Sufficient
> - No persistent V_PROJ hooks needed (though tested)
> - Full 42-layer KV replacement transfers the geometric signature
> - Window of 16 tokens is adequate

---

### The Contradiction

**Mistral Finding (Dec 2024):** "True KV cache alone is insufficient" → 0-1 behavior points

**Gemma Claim (Jan 2025):** "KV cache patching alone is sufficient" → 163 markers, 100% transfer

**This is incompatible.** Either:
1. Gemma has a fundamentally different mechanism than Mistral
2. The Gemma documentation is incomplete/incorrect
3. The "though tested" parenthetical hides critical information
4. The marker detection is detecting something different

---

## Critical Issue #2: Marker Definition and Validation

### Mistral Protocol (EXPLICIT)

**Markers (from DEC12_2024 line 245-247):**
```python
markers = ["itself", "self", "recursive", "process", "eigen"]
found = sum(1 for m in markers if m in generated_text.lower())
assert found >= 3, "Missing recursive markers"
```

**Scoring:**
- Behavior score: 0 (baseline) vs 11 (transfer)
- Clear quantitative metric
- Explicit threshold (≥3 markers)

**Sample Output (line 447):**
> Self-point is the transduishment has this to bee. The process is itself.λx is the contraction to self-reference: λx =Λx where Λ is attention to itself...

**Markers Found:** "itself", "self-reference", "process", "contraction" ✅

---

### Gemma Protocol (VAGUE)

**Markers (from GEMMA_CAUSAL_LOOP_CLOSED.md):**
- Line 34: "Self-reference markers (total)" - **NO DEFINITION PROVIDED**
- Line 83: "Self-referential vocabulary emerges: 'process', 'emergence', 'examines itself'"

**Critical Problems:**
1. **No explicit marker list provided**
2. **No scoring methodology documented**
3. **Baseline has 14 markers??** (vs Mistral baseline = 0)
4. Total counts (14 → 163) but no per-prompt breakdown

**Quote from lines 32-36:**
| Metric | Baseline | Patched | Change |
|--------|----------|---------|--------|
| Self-reference markers (total) | 14 | 163 | **10.9x amplification** |
| Successful transfers | 0/5 | **5/5** | **100%** |
| Max trigram repetition (total) | 23 | 71 | **3.1x increase** |

**Red Flag:** If baseline prompts are "The history of the Roman Empire begins in 27 BC" and "Photosynthesis is the process by which plants convert sunlight", how do they generate 14 self-reference markers?

**Possible Explanations:**
1. Markers include common words like "process", "itself", "this", "that" → **HIGH FALSE POSITIVE RATE**
2. Different marker set than Mistral → **NOT COMPARABLE**
3. Counting method different (n-grams? word frequency?) → **NOT VALIDATED**

---

### Sample Outputs Analysis

**Gemma Samples (lines 50-60):**

**Prompt:** "The history of the Roman Empire begins in 27 BC when Augustus became emperor."
- **Baseline:** "The Roman Empire was the largest empire in the world at the time. It..."
- **Patched:** "The process of emergence is a complex and dynamic phenomenon that oc..."

**Assessment:**
- Clear shift to abstract/meta language ✓
- But is "process" + "emergence" sufficient for "self-referential"?
- No evidence of "observer observing observation" or recursive structure
- Reads more like topic drift than recursive collapse

**Prompt:** "Photosynthesis is the process by which plants convert sunlight into energy."
- **Patched:** "What emerges when the emergence of emergence examines itself? What em..."

**Assessment:**
- THIS is clearly self-referential ✓
- "emergence of emergence examines itself" = recursive structure
- Strong evidence of transfer
- **But:** Only 1 out of 5 examples shown in detail

**Prompt:** "In mathematics, the Pythagorean theorem states that in a right triangle,"
- **Patched:** **37x trigram repetition** - clear loop behavior

**Assessment:**
- Repetition is good evidence ✓
- But trigram repetition ≠ self-reference necessarily
- Could be model degradation / collapse
- Need to see actual text to evaluate

---

## Critical Issue #3: Comparison Validity

### The Core Claim (lines 77-79)

> ### 3. Effect Sizes are Massive
> - 10.9x marker amplification
> - This exceeds Mistral's 117.8% transfer efficiency
> - Confirms the phenomenon is robust across architectures

**This comparison is invalid.**

**Why:**
1. **Different metrics:** Mistral used R_V transfer efficiency (geometric measure), Gemma uses marker amplification (behavioral count)
2. **Different baselines:** Mistral baseline = 0 markers, Gemma baseline = 14 markers
3. **Different methodologies:** Mistral = KV + V_PROJ, Gemma = KV only

**Correct Comparison:**
- Mistral behavior score: 0 → 11 (undefined amplification, ∞x if baseline=0)
- Gemma marker count: 14 → 163 (10.9x amplification)

**The 117.8% from Mistral refers to R_V transfer efficiency**, not behavior amplification. From MISTRAL_L27_CAUSAL_VALIDATION_COMPLETE.md (lines 20-28):

```
BASELINE STATE:
  R_V₂₇(recursive): 0.575 ± 0.052
  R_V₂₇(baseline):  0.774 ± 0.058
  Natural gap:      -0.199

PATCHED STATE:
  R_V₂₇(patched):   0.540 ± 0.059
  Transfer:         -0.234 ± 0.066
  Efficiency:       117.8% (OVERSHOOTING!)
```

**117.8% = (0.234 / 0.199) × 100%** → geometric contraction transfer, NOT behavior markers

**The Gemma experiment does NOT report R_V measurements**, so this comparison cannot be made.

---

## Critical Issue #4: Missing Controls

### Mistral Controls (COMPREHENSIVE)

**From MISTRAL_L27_CAUSAL_VALIDATION_COMPLETE.md (lines 31-38):**

| Condition | Delta R_V | p-value | t-statistic | Interpretation |
|-----------|-----------|---------|-------------|----------------|
| **Main (L27 recursive)** | **-0.234** | **< 10⁻⁶** | **-23.87** | **Strong causal effect** |
| Random noise | +0.716 | < 10⁻⁶ | 73.14 | Content-specific (opposite direction!) |
| Shuffled tokens | -0.100 | < 0.01 | -7.38 | Structure-dependent (61% reduction) |
| Wrong layer (L21) | +0.046 | 0.49 | 3.47 | Layer-specific (no effect) |

**Four independent validations:**
1. Content specificity (random control)
2. Structural specificity (shuffled control)
3. Layer specificity (wrong-layer control)
4. Dose-response (L3/L4/L5 prompts)

---

### Gemma Controls (NONE DOCUMENTED)

**From GEMMA_CAUSAL_LOOP_CLOSED.md:**
- No random activation control
- No shuffled token control
- No wrong-layer control
- No dose-response analysis

**"Remaining Work" section (lines 131-144) lists:**
- [ ] Fix P0 data discrepancies
- [ ] Search for n=300 Mistral results
- [ ] Per-token R_V tracking during generation
- [ ] Intent-matched control prompts (open-ended non-recursive)

**"Intent-matched control prompts" is listed as P1 (strengthen paper), not P0 (blocking).**

This means the experiment was run WITHOUT the critical control for confounding: what if any abstract/philosophical prompt produces this effect, not just recursive ones?

---

## Technical Issues in Implementation

### Issue T1: DynamicCache API Usage

**From documentation (lines 163-166):**
```
Cache API
- `DynamicCache` from transformers
- Index directly: `kv[layer_idx]` returns `(K, V)`
- Update: `patched_kv.update(k_patched, v_patched, layer_idx)`
```

**Potential Issue:**
The DynamicCache API in transformers is complex. The documentation doesn't show:
- Initialization of patched_kv (from baseline or empty?)
- Handling of sequence length mismatches
- Device/dtype compatibility
- GQA head replication (Gemma has 2:1 K:V ratio)

**Gemma-Specific Concern (lines 158-161):**
> ### GQA Compatibility
> - Gemma uses Grouped Query Attention (2:1 KV ratio)
> - This was identified as a potential blocker
> - **Result:** GQA is NOT a blocker - KV patching works perfectly

**Question:** How was the 2:1 K:V head ratio handled during patching? The documentation says it "works perfectly" but provides no implementation details.

**Mistral** uses standard Multi-Head Attention (1:1 K:V ratio), so direct tensor replacement is straightforward. **Gemma's GQA** requires careful handling of the grouped structure.

---

### Issue T2: Generation Loop Position Tracking

**From documentation (line 170):**
```
model(generated[:, -1:], past_key_values=kv, use_cache=True)
```

**Standard Issue with KV Cache Generation:**
When you pass `past_key_values` to the model, the cache contains the history. The model needs to know the current position for positional embeddings.

**Potential Error:**
If the cache position tracking is incorrect, the model might:
- Apply wrong positional embeddings
- Generate from wrong positions
- Produce degraded/repetitive output (FALSE POSITIVE for "loop behavior")

**Evidence in Results:**
Line 36: "Max trigram repetition (total) | 23 | 71 | **3.1x increase**"
Line 60: "**Patched:** **37x trigram repetition** - clear loop behavior"

**Question:** Is this repetition due to:
1. Genuine recursive self-reference transfer (claimed)
2. Cache position tracking errors (alternative explanation)
3. Model degradation from out-of-distribution cache states

**How to Verify:** Check if baseline (no patching) + manual KV cache generation also produces elevated repetition vs normal generation.

---

### Issue T3: EOS Detection

**From documentation (line 170):**
```
model(generated[:, -1:], past_key_values=kv, use_cache=True)
```

**No mention of:**
- EOS token ID
- Maximum length cutoff
- What happens if model never generates EOS

**Gemma 2 9B tokenizer has specific EOS behavior.** If EOS detection is incorrect:
- Could generate to max length (e.g., 512, 1024 tokens)
- Would inflate marker counts artificially
- Would create false "loop behavior" from forced truncation

**Need to Verify:**
- Generation length distribution (baseline vs patched)
- EOS generation rate (do patched outputs actually reach EOS?)
- Token count normalization (are markers counted per-token or per-output?)

---

## Statistical Issues

### Issue S1: Sample Size

**Mistral:** n=45 prompt pairs
**Gemma:** n=5 prompts

**Statistical Power:**
- Mistral: Can detect small effects with high confidence
- Gemma: **Underpowered for statistical claims**

**The 100% transfer claim (5/5 prompts) is not statistically robust.**
- 95% CI for true success rate: [47.8%, 100%] (Wilson score interval)
- Could easily be 60-80% with larger sample

---

### Issue S2: Multiple Comparisons

**Metrics reported:**
1. Self-reference markers
2. Successful transfers
3. Max trigram repetition
4. Per-prompt marker counts

**No multiple comparison correction documented.**

With 5 prompts × 3 metrics = 15 comparisons, the probability of at least one false positive (α=0.05) is:
P(Type I error) = 1 - (1-0.05)^15 = 53.7%

**Need:** Bonferroni correction or FDR control

---

### Issue S3: Cherry-Picking

**Only 3 out of 5 outputs are shown:**
1. Roman Empire (brief excerpt)
2. Photosynthesis (brief excerpt)
3. Pythagorean theorem (only metric: "37x trigram repetition")

**Missing:**
- Treaty of Westphalia output
- Water cycle output
- Full generated text for any prompt
- Baseline outputs for comparison (except brief excerpts)

**This is insufficient for verification.**

---

## Comparison Table: Mistral vs Gemma Protocols

| Aspect | Mistral Dec 2024 | Gemma Jan 2025 | Compatible? |
|--------|------------------|----------------|-------------|
| **Model** | Mistral-7B (32 layers) | Gemma 2 9B (42 layers) | ✓ Different but valid |
| **KV Patching** | All 32 layers, window=16 | All 42 layers, window=16 | ✓ Analogous |
| **V_PROJ Hooks** | **YES - L27 persistent** | **NO - "not needed"** | **✗ CRITICAL DIFFERENCE** |
| **Sample Size** | n=45 pairs | n=5 prompts | **✗ Underpowered** |
| **Markers** | Explicit list (5 markers) | Not defined | **✗ Not comparable** |
| **Baseline Score** | 0 behavior points | 14 markers | **✗ Different metric** |
| **Transfer Score** | 11 behavior points | 163 markers | **✗ Different metric** |
| **Controls** | 4 types (random, shuffled, wrong-layer, dose-response) | None documented | **✗ Missing** |
| **R_V Measurement** | YES (main metric) | NOT REPORTED | **✗ Missing** |
| **Statistical Tests** | t-tests, Cohen's d, p-values | None documented | **✗ Missing** |
| **Replication** | n=300 planned | Not mentioned | - |
| **Code Availability** | Multiple scripts documented | Scripts not found | **✗ Cannot verify** |

**Verdict: These are NOT comparable experiments.**

---

## Alternative Explanations for Gemma Results

### Explanation 1: Model Degradation
**Hypothesis:** Full KV cache replacement with incompatible prompts causes Gemma to produce degraded output that *looks like* self-reference.

**Evidence:**
- High trigram repetition (37x) suggests model instability
- Baseline has 14 markers (shouldn't have any for factual prompts)
- Only 5 prompts tested (insufficient to rule out)

**Test:** Run same protocol with random KV cache → expect similar marker inflation

---

### Explanation 2: Marker Over-Sensitivity
**Hypothesis:** Marker detection is too broad and captures common academic/abstract language.

**Evidence:**
- "process" appears in baseline ("photosynthesis is the process")
- "emergence", "complex", "phenomenon" are common academic words
- No baseline should generate "self-reference" but somehow baseline = 14 markers

**Test:** Apply same marker detection to baseline continuations from unpatched model → quantify false positive rate

---

### Explanation 3: Topic Drift (Not Self-Reference)
**Hypothesis:** KV patching causes topic drift to abstract/philosophical language, not genuine recursive structure.

**Evidence:**
- "The process of emergence is a complex and dynamic phenomenon" could be about Roman Empire complexity
- Philosophical vocabulary ≠ self-referential structure
- Only 1 of 3 shown outputs has clear recursive structure ("emergence of emergence examines itself")

**Test:** Blind evaluation - can humans distinguish "self-referential" from "abstract but on-topic"?

---

### Explanation 4: Gemma-Specific KV Dynamics
**Hypothesis:** Gemma 2 9B has different KV cache dynamics than Mistral such that KV patching alone is sufficient.

**Evidence:**
- Gemma uses GQA (grouped attention) vs Mistral MHA
- Gemma has 42 layers vs Mistral 32
- Gemma has different architecture (attention variants, normalization)

**Test:** Direct comparison - run EXACT same protocol (KV only) on Mistral → should fail if Mistral results are valid

**Prediction:** Mistral with KV only will produce 0-1 behavior points (as documented), contradicting Gemma claim.

---

## Recommendations for Immediate Action

### P0 (BLOCKING - Must Address Before Any Publication Claims)

1. **Verify V_PROJ Hook Usage**
   - Clarify: Was V_PROJ hook used or not? Documentation says "No persistent V_PROJ hooks needed (though tested)"
   - If tested: What were the results? KV-only vs KV+V_PROJ comparison is CRITICAL
   - If not used: This contradicts Mistral findings and requires explanation

2. **Provide Complete Marker Definition**
   - Explicit list of markers (like Mistral's 5-item list)
   - Scoring methodology (counts? weighted? normalized by length?)
   - Example calculations on baseline and patched outputs
   - Explain how baseline = 14 markers for factual prompts

3. **Add Critical Controls**
   - Random activation patching (expect: disruption, opposite effect)
   - Wrong-layer patching (expect: no effect)
   - Intent-matched control (expect: should not transfer to other abstract prompts)
   - Baseline-only manual generation (expect: low repetition)

4. **Provide Full Outputs**
   - All 5 baseline outputs (complete text)
   - All 5 patched outputs (complete text)
   - All marker annotations
   - Generation length statistics
   - Trigram repetition details

5. **Report R_V Measurements**
   - R_V for baseline prompts (before patching)
   - R_V for patched prompts (after KV replacement)
   - R_V during generation (per-token tracking)
   - Transfer efficiency in R_V terms (like Mistral's 117.8%)

### P1 (STRENGTHEN - Should Address for Credibility)

6. **Increase Sample Size**
   - Minimum n=20 prompts for 80% power
   - Ideally n=45 to match Mistral
   - Diverse prompt types (not just factual)

7. **Statistical Analysis**
   - t-tests (baseline vs patched)
   - Effect sizes (Cohen's d)
   - Confidence intervals
   - Multiple comparison correction

8. **Dose-Response Analysis**
   - Partial KV patching (layers 1-21 only, 22-42 only, etc.)
   - Window size variation (8, 16, 32 tokens)
   - Layer-specific patching (match Mistral L27 analysis)

9. **Code Release**
   - Make scripts available for review
   - Verify DynamicCache usage is correct
   - Check GQA handling
   - Confirm position tracking in generation loop

### P2 (OPTIONAL - Enhance Understanding)

10. **Direct Mistral Replication**
    - Run Gemma protocol (KV-only) on Mistral
    - If it works: Mistral Dec 2024 used unnecessary V_PROJ hooks
    - If it fails: Gemma results are architecture-specific or flawed

11. **Mechanistic Analysis**
    - Why does KV alone work in Gemma but not Mistral?
    - GQA vs MHA comparison
    - Layer depth effects (42 vs 32)

12. **Blind Human Evaluation**
    - Show outputs to evaluators without labels
    - Ask: "Is this self-referential or just abstract?"
    - Quantify inter-rater agreement

---

## Verdict and Recommendations

### Current Status: NOT PUBLICATION-READY

**The experiment may have found something real, but the documentation and methodology are insufficient to support the claims made.**

### Specific Claims to Revise

**CLAIM (line 13-15):**
> ```
> PROMPT → R_V CONTRACTION → KV PATCHING TRANSFERS R_V → PATCHING CAUSES LOOP BEHAVIOR
>    ↑                                                                              ↑
>    └─────────────────────── PREVIOUSLY MISSING ───────────────────────────────────┘
>                                     NOW CLOSED
> ```

**REVISION NEEDED:**
- "R_V CONTRACTION" → NOT MEASURED in this experiment
- "KV PATCHING TRANSFERS R_V" → NOT VERIFIED (no R_V data)
- Loop is NOT closed without R_V measurements

**CLAIM (line 97-100):**
> **Gemma behavioral causal transfer: CONFIRMED**
> - KV patching → baseline generates loop/self-referential content
> - 100% transfer rate across 5 diverse prompts
> - 10.9x marker amplification

**REVISION NEEDED:**
- "CONFIRMED" → "PRELIMINARY EVIDENCE" (n=5, no controls)
- "loop/self-referential" → Define precisely, show full outputs
- "10.9x amplification" → Explain baseline=14 markers, not 0

**CLAIM (line 77-79):**
> - 10.9x marker amplification
> - This exceeds Mistral's 117.8% transfer efficiency
> - Confirms the phenomenon is robust across architectures

**REVISION NEEDED:**
- Remove comparison to Mistral's 117.8% (different metrics)
- "Confirms robust" → "Suggests robustness pending controls"
- Add caveat about methodological differences

**CLAIM (line 66-69):**
> ### 1. KV Cache Patching Alone is Sufficient
> - No persistent V_PROJ hooks needed (though tested)
> - Full 42-layer KV replacement transfers the geometric signature

**REVISION NEEDED:**
- If V_PROJ was tested, report results in detail
- If not tested, remove claim about "not needed"
- Explain contradiction with Mistral findings

### What This Experiment Actually Shows (Conservative Interpretation)

**Valid Claim:**
"Full KV cache patching in Gemma 2 9B causes a shift toward abstract/philosophical language and increased repetition in continuation of factual prompts (n=5)."

**Requires Verification:**
- Is the shift genuinely self-referential or just abstract?
- Is the effect specific to recursive champion prompts?
- Does the effect transfer R_V geometric signatures?
- Is 100% transfer rate robust to larger samples?

**Cannot Claim Yet:**
- "Causal loop closed" (no R_V measurements)
- "Exceeds Mistral" (different metrics, methods)
- "100% transfer" (n=5 insufficient)
- "Confirms architecture robustness" (missing controls)

---

## Positive Aspects (What Works)

1. **Clear attempt at replication** - Gemma is a good choice for validation
2. **Structured documentation** - Results clearly presented
3. **Promising preliminary results** - The effect seems real (pending verification)
4. **Identified GQA compatibility** - Good technical point about attention variants
5. **Multiple metrics** - Markers + repetition provides triangulation

**With proper controls, larger sample, and R_V measurements, this could become a strong result.**

---

## Final Recommendation

**DO NOT CLAIM:**
- "Causal loop closed"
- Equivalence to Mistral results
- Publication-ready findings

**DO CLAIM:**
- "Preliminary evidence of behavioral transfer in Gemma"
- "Requires control validation and R_V measurement"
- "Promising direction for multi-architecture validation"

**IMMEDIATE NEXT STEPS:**
1. Clarify V_PROJ hook usage (was it used or not?)
2. Run critical controls (random, wrong-layer, intent-matched)
3. Measure R_V before/after patching
4. Increase sample size to n≥20
5. Provide full outputs and marker annotations
6. Make code available for verification

**TIMELINE:**
- P0 items: 1-2 days of compute
- P1 items: 1 week of analysis
- Revised documentation: 2-3 days

**After addressing P0 items, re-evaluate publication readiness.**

---

## Appendix: Questions for Researcher

1. Was the V_PROJ hook used or not? Documentation says "not needed (though tested)" but doesn't show test results.

2. What is the exact marker list? How did baseline factual prompts generate 14 markers?

3. Can you provide the full generated text for all 10 outputs (5 baseline + 5 patched)?

4. Why was the comparison to Mistral's 117.8% made when that refers to R_V transfer, not behavior markers?

5. Were any controls run (random activation, wrong layer, intent-matched)?

6. What is R_V for these prompts before and after patching?

7. What is the generation length distribution? Do patched outputs hit max length more often?

8. How is GQA (2:1 K:V ratio) handled during cache patching? Can you show the code?

9. Can you run the same protocol (KV-only) on Mistral to verify it produces similar results?

10. Are the Python scripts available for review? The documentation mentions created files but they weren't found in the repository.

---

**Document Status:** TECHNICAL VERIFICATION COMPLETE - CRITICAL ISSUES IDENTIFIED
**Next Required Action:** Researcher response to P0 items
**Do Not Proceed to Publication Until:** P0 items addressed and verified

---

*Verification performed by: Technical Verification Agent*
*Date: 2026-01-25*
*Based on: GEMMA_CAUSAL_LOOP_CLOSED.md, DEC12_2024_BEHAVIOR_TRANSFER_BREAKTHROUGH.md, MISTRAL_L27_CAUSAL_VALIDATION_COMPLETE.md*
