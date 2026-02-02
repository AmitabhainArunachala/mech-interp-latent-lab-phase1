# MLP Steering Experimental Hygiene Audit
**Date:** January 4, 2025  
**Auditor:** Composer AI  
**Purpose:** Verify experimental integrity and protocol compliance

---

## Executive Summary

**Status: ⚠️ PARTIAL COMPLIANCE**

Several protocol violations were identified that may compromise experimental integrity:
1. ✅ Used prompt bank (via PromptLoader)
2. ✅ Used champion recursive groups (L3_deeper, L4_full, L5_refined)
3. ❌ **Sample size too small** (n_pairs=5-10, protocol requires ≥80)
4. ❌ **No statistical testing** (no p-values, Bonferroni correction)
5. ⚠️ **R_V measured on wrong text** (baseline prompt, not generated text)
6. ⚠️ **Generation parameters** (temperature=0.7, do_sample=True) may introduce variance

---

## 1. Prompt Selection ✅ COMPLIANT

### What We Did:
```python
loader = PromptLoader()
pairs = loader.get_balanced_pairs(n_pairs=n_pairs, seed=42)
```

### What `get_balanced_pairs()` Does:
- **Default recursive groups:** `["L3_deeper", "L4_full", "L5_refined"]` ✅
- **Default baseline groups:** `["baseline_math", "baseline_factual", "baseline_creative"]` ✅
- **Source:** `prompts/bank.json` (canonical source) ✅
- **Seed:** 42 (fixed for reproducibility) ✅

### Verification:
- ✅ Used prompt bank (not hardcoded prompts)
- ✅ Used champion recursive groups (L3_deeper, L4_full, L5_refined)
- ✅ Used standard baseline groups
- ✅ Fixed seed for reproducibility

**Verdict: COMPLIANT** - Prompt selection follows protocol.

---

## 2. Sample Size ❌ VIOLATION

### Protocol Standard (from repo rules):
- **Minimum n_pairs:** 80 pairs for statistical power
- **Statistical threshold:** p < 0.01 with Bonferroni correction
- **Effect size threshold:** |d| ≥ 0.5 for meaningful effects

### What We Did:
- **Initial sweep:** n_pairs = 10
- **Full layer sweep:** n_pairs = 5
- **Alpha sweep:** n_pairs = 5
- **Random control:** n_pairs = 10

### Impact:
- **Underpowered:** 5-10 pairs is insufficient for statistical significance
- **No p-values:** Cannot determine if effects are statistically significant
- **No Bonferroni correction:** Multiple comparisons not accounted for
- **High variance:** Small sample size increases uncertainty

**Verdict: VIOLATION** - Sample sizes 8-16x smaller than protocol minimum.

---

## 3. Statistical Testing ❌ MISSING

### Protocol Requirements:
- Statistical threshold: p < 0.01 with Bonferroni correction
- Effect size: |d| ≥ 0.5
- Multiple comparisons correction

### What We Did:
- **No statistical tests:** Only reported means and standard deviations
- **No p-values:** Cannot determine significance
- **No effect size calculations:** Only raw deltas
- **No multiple comparisons correction:** Tested 32 layers without correction

### Impact:
- **Cannot distinguish signal from noise:** Effects may be random
- **False positives likely:** Multiple comparisons without correction
- **No confidence intervals:** Uncertainty not quantified

**Verdict: VIOLATION** - No statistical testing performed.

---

## 4. R_V Measurement ⚠️ POTENTIAL ISSUE

### What We Did:
```python
# Compute steered R_V (on generated text) - use original prompt, not full generated text
rv_steered = compute_rv(model, tokenizer, base_text, early=5, late=27, window=window_size, device=device)
```

### Issue:
- **R_V measured on baseline prompt** (`base_text`), not generated text
- **Comment says:** "use original prompt, not full generated text"
- **But:** We're measuring R_V on the INPUT, not the OUTPUT

### Protocol Standard:
- R_V should measure geometry of the **actual generated text**
- Or measure R_V during generation (on-the-fly)
- Current approach measures geometry of input, not steered output

### Impact:
- **R_V delta may not reflect steering effect:** Measuring input geometry, not output
- **May miss actual geometric changes:** Steering happens during generation, not on input

**Verdict: POTENTIAL ISSUE** - R_V measured on wrong text (input vs output).

---

## 5. Generation Parameters ⚠️ VARIANCE SOURCE

### What We Did:
```python
outputs_steered = model.generate(
    **inputs_gen,
    max_new_tokens=max_new_tokens,
    temperature=0.7,        # ⚠️ Non-zero temperature
    do_sample=True,         # ⚠️ Sampling enabled
    pad_token_id=tokenizer.eos_token_id
)
```

### Protocol Standard:
- **Temperature:** Should be 0.0 for deterministic results
- **do_sample:** Should be False for deterministic results
- **Seed:** Should be set for reproducibility

### Impact:
- **Non-deterministic:** Sampling introduces variance
- **Harder to reproduce:** Results may vary between runs
- **Confounds effect:** Steering effect mixed with sampling variance

**Verdict: VARIANCE SOURCE** - Non-deterministic generation may confound results.

---

## 6. Model Configuration ✅ COMPLIANT

### What We Did:
- **Model:** `mistralai/Mistral-7B-v0.1` ✅
- **Device:** CUDA ✅
- **Model.eval():** Called ✅
- **torch.no_grad():** Used ✅
- **Seed:** 42 (fixed) ✅

**Verdict: COMPLIANT** - Model configuration follows protocol.

---

## 7. R_V Parameters ✅ COMPLIANT

### What We Did:
```python
rv_base = compute_rv(model, tokenizer, base_text, early=5, late=27, window=16, device=device)
```

### Protocol Standard:
- **Early layer:** 5 ✅
- **Late layer:** 27 (num_layers - 5) ✅
- **Window:** 16 tokens ✅

**Verdict: COMPLIANT** - R_V parameters match protocol.

---

## 8. Controls ⚠️ INCOMPLETE

### What We Did:
- ✅ Random direction control (L2) - **COMPLETED**
- ⏳ Random direction control (L3) - **INCOMPLETE** (server disconnect)
- ⏳ Random direction control (L4) - **INCOMPLETE** (server disconnect)

### Protocol Requirements:
- **4 controls required:** random, shuffled, wrong-layer, opposite
- **Current:** Only random vectors tested
- **Missing:** Shuffled, wrong-layer, opposite controls

**Verdict: INCOMPLETE** - Only 1 of 4 required controls completed.

---

## 9. Mode Score Measurement ⚠️ ISSUES

### What We Did:
```python
mode_base = mode_metric.compute_score(out_base.logits, baseline_logits=out_base.logits)
mode_steered = mode_metric.compute_score(out_steered.logits, baseline_logits=out_base.logits)
```

### Issues Observed:
- **Many NaN values:** Mode score computation failed for most pairs
- **Tensor size mismatches:** Errors like "The size of tensor a (210) must match the size of tensor b (34)"
- **Truncation issues:** Generated text truncated before mode score computation

### Impact:
- **Missing data:** Most mode scores are NaN
- **Cannot assess behavior transfer:** Primary behavioral metric unavailable
- **Results incomplete:** Cannot fully evaluate steering effectiveness

**Verdict: ISSUES** - Mode score measurement unreliable.

---

## 10. Coherence Measurement ✅ COMPLIANT

### What We Did:
```python
behavior_score = score_behavior_strict(generated_text)
coherence = behavior_score.coherence_score
```

### Protocol:
- Uses `StrictBehaviorScore` ✅
- Measures output quality ✅
- Proper error handling ✅

**Verdict: COMPLIANT** - Coherence measurement follows protocol.

---

## Summary of Violations

| Issue | Severity | Impact |
|-------|----------|--------|
| **Sample size too small** (5-10 vs 80) | 🔴 HIGH | Underpowered, cannot detect effects |
| **No statistical testing** | 🔴 HIGH | Cannot determine significance |
| **R_V measured on input** (not output) | 🟡 MEDIUM | May miss actual steering effects |
| **Non-deterministic generation** | 🟡 MEDIUM | Introduces variance, harder to reproduce |
| **Mode score failures** | 🟡 MEDIUM | Missing behavioral data |
| **Incomplete controls** | 🟡 MEDIUM | Only 1/4 controls completed |

---

## Recommendations

### Critical (Must Fix):
1. **Increase sample size to n_pairs ≥ 80**
2. **Add statistical testing** (p-values, Bonferroni correction)
3. **Fix R_V measurement** (measure on generated text, not input)

### Important (Should Fix):
4. **Use deterministic generation** (temperature=0.0, do_sample=False)
5. **Fix mode score computation** (resolve tensor size mismatches)
6. **Complete control experiments** (L3-L4 random controls)

### Nice to Have:
7. **Add confidence intervals** for all metrics
8. **Report effect sizes** (Cohen's d)
9. **Add power analysis** to justify sample sizes

---

## Files to Review

### Code Files:
- `src/pipelines/mlp_steering_sweep.py` - Main pipeline (lines 231, 244-250, 259)
- `src/pipelines/random_direction_control.py` - Control pipeline

### Config Files:
- `configs/mlp_steering_sweep.json` - n_pairs=10
- `configs/mlp_steering_sweep_full.json` - n_pairs=5
- `configs/mlp_steering_alpha_sweep.json` - n_pairs=5

### Prompt Loading:
- `prompts/loader.py` - Lines 147-194 (get_balanced_pairs implementation)
- Uses champion groups: L3_deeper, L4_full, L5_refined ✅

---

## Conclusion

**Experimental Integrity: ⚠️ COMPROMISED**

While prompt selection follows protocol (champion groups from prompt bank), several critical violations were identified:

1. **Sample sizes are 8-16x too small** (5-10 vs 80 minimum)
2. **No statistical testing** (cannot determine significance)
3. **R_V measured incorrectly** (on input, not output)
4. **Non-deterministic generation** (introduces variance)

**These violations may explain:**
- Why L2 steering appears to be an artifact (underpowered, no stats)
- Why results are inconsistent (small sample, high variance)
- Why we cannot determine if effects are real (no p-values)

**Recommendation:** Re-run experiments with proper sample sizes (n_pairs ≥ 80) and statistical testing before drawing conclusions.

---

**Audit completed:** January 4, 2025  
**Next steps:** Fix violations and re-run critical experiments


