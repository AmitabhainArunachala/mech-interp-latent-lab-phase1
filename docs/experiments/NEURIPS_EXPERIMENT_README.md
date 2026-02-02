# NeurIPS n=300 Robust Experiment

## Overview

This experiment reproduces the 100% behavior transfer finding at scale (n=300) with proper controls and statistical analysis suitable for NeurIPS submission.

## Method

**Winning Strategy:**
- Full KV cache replacement (all 32 layers)
- Persistent V_PROJ patching at L27 during generation

**Controls:**
1. **Baseline:** No patching
2. **Random:** Random V activation (control for structure)
3. **Wrong Layer:** Patch at L5 instead of L27 (control for layer specificity)

## Experimental Design

- **N:** 300 prompt pairs
- **Recursive prompts:** L3_deeper, L4_full, L5_refined
- **Baseline prompts:** baseline_factual, baseline_creative, baseline_math, baseline_instructional, baseline_personal, long_control
- **Metrics:** Behavior score (marker-based) + R_V (geometric)
- **Generation:** 150 tokens, temperature=0.8

## Statistical Analysis

- **T-tests:** One-sample t-tests for each condition vs. zero
- **Effect sizes:** Cohen's d
- **Confidence intervals:** 95% CI for mean effects
- **Comparisons:** Independent t-tests (transfer vs. controls)

## Expected Results

Based on pilot (n=1):
- **Transfer:** Behavior score ~11 (baseline ~0)
- **Transfer efficiency:** ~100%
- **Controls:** Should show minimal/no effect

## Files

- `neurips_n300_robust_experiment.py` - Main experiment script
- `neurips_n300_results.csv` - Full results (pair-level data)
- `neurips_n300_summary.md` - Statistical summary
- `neurips_n300_output.log` - Execution log

## Running

```bash
python neurips_n300_robust_experiment.py
```

**Estimated runtime:** ~2-3 hours for n=300 pairs

## Status

Experiment is running in background. Check `neurips_n300_output.log` for progress.

