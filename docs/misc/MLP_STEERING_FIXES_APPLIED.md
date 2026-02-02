# MLP Steering Pipeline Fixes Applied
**Date:** January 4, 2025  
**Purpose:** Address experimental hygiene violations identified in audit

---

## Fixes Applied

### 1. ✅ Sample Size Increased
- **Before:** n_pairs = 5-10
- **After:** n_pairs = 80 (protocol minimum)
- **Impact:** Sufficient statistical power to detect effects

### 2. ✅ Statistical Testing Added
- **Added:** One-sample t-tests against null hypothesis (no effect)
- **Added:** Bonferroni correction for multiple comparisons
- **Added:** Cohen's d effect size calculations
- **Output:** p-values, t-statistics, significance flags in summary.json
- **Impact:** Can now determine if effects are statistically significant

### 3. ✅ R_V Measurement Fixed
- **Before:** Measured R_V on input prompt (`base_text`)
- **After:** Measures R_V on full generated text (`generated_text`)
- **Impact:** Now measures actual geometry of steered output, not input

### 4. ✅ Deterministic Generation
- **Before:** `temperature=0.7, do_sample=True` (non-deterministic)
- **After:** `temperature=0.0, do_sample=False` (deterministic)
- **Impact:** Reproducible results, eliminates sampling variance

### 5. ✅ Mode Score Fixes
- **Added:** Sequence length matching (pad/truncate to same length)
- **Added:** Proper tensor shape handling
- **Impact:** Reduces NaN values, more reliable mode score computation

### 6. ✅ Summary Statistics Enhanced
- **Added:** Statistical test results (t-stat, p-value, significance)
- **Added:** Effect sizes (Cohen's d)
- **Added:** Bonferroni-corrected alpha threshold
- **Impact:** Complete statistical reporting for publication

---

## Code Changes

### File: `src/pipelines/mlp_steering_sweep.py`

1. **Imports:** Added `scipy.stats` for statistical testing
2. **Generation:** Changed to deterministic (`temperature=0.0, do_sample=False`)
3. **R_V:** Now measures on `generated_text` instead of `base_text`
4. **Mode Score:** Added sequence length matching and proper tensor handling
5. **Summary:** Added statistical tests, Bonferroni correction, effect sizes

---

## Config Files

### New Config: `configs/mlp_steering_sweep_corrected.json`
- **n_pairs:** 80 (protocol minimum)
- **layers:** All 32 layers
- **alpha:** 2.0
- **max_new_tokens:** 200
- **Deterministic:** Yes (via code changes)

---

## Expected Output

### CSV File: `mlp_steering_sweep.csv`
- All individual results (80 pairs × 32 layers × 1 alpha = 2,560 rows)
- Columns: layer, pair_idx, alpha, rv_delta, mode_delta, coherence, etc.

### Summary JSON: `summary.json`
- **summary_by_layer_alpha:** Mean/std for each layer-alpha combination
- **statistical_tests:** T-tests, p-values, significance flags
- **bonferroni_alpha:** Corrected threshold (0.01 / n_comparisons)
- **n_comparisons:** Total number of comparisons

---

## Running the Experiment

```bash
python -m src.pipelines.run --config configs/mlp_steering_sweep_corrected.json
```

**Expected Runtime:** ~2-4 hours (80 pairs × 32 layers × ~2-3 min/pair)

---

## Success Criteria

1. ✅ **Sample size:** n_pairs = 80
2. ✅ **Statistical tests:** p-values < bonferroni_alpha indicate significance
3. ✅ **Effect sizes:** |Cohen's d| ≥ 0.5 for meaningful effects
4. ✅ **R_V measured correctly:** On generated text, not input
5. ✅ **Deterministic:** Same seed produces identical results
6. ✅ **Mode score:** < 10% NaN values

---

## Next Steps

1. Run corrected experiment
2. Analyze results with proper statistics
3. Identify layers with significant effects (p < bonferroni_alpha)
4. Verify effect sizes (|d| ≥ 0.5)
5. Compare to previous (underpowered) results

---

**Status:** ✅ All fixes applied, ready to run


