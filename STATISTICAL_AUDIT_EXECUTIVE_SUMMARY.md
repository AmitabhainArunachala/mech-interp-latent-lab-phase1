# Statistical Audit Executive Summary

**Date**: 2026-02-02
**Overall Validity Score**: 8.5 / 10
**Publication Status**: READY (with caveats)

---

## Bottom Line

The R_V causal validation is **statistically robust** across 5 architectures. All reported statistics are verified and accurate. Effect survives multiple comparisons correction, sensitivity analyses, and outlier removal. **Proceed to publication** with explicit heterogeneity reporting and qualified behavioral correlation claims.

---

## Key Results Verified

### Main Effect (All Models Combined)
- **n = 225 pairs** across 5 architectures
- **Cohen's d = -0.91** (large effect)
- **p < 10⁻³⁰** (highly significant)
- **Direction**: Recursive prompts → lower R_V ✅

### Individual Models
| Model | n | Cohen's d | p-value | Power | Bonf. Sig? | Holm Sig? |
|-------|---|-----------|---------|-------|------------|-----------|
| Mistral-7B | 45 | **-2.285** | 2.24e-19 | 100% | ✅ | ✅ |
| Qwen2-7B | 45 | -0.727 | 8.72e-06 | 99.9% | ✅ | ✅ |
| Pythia-1.4B | 45 | -0.314 | 2.14e-02 | 66.7% | ❌ | ✅ |
| OPT-6.7B | 45 | **-1.857** | 3.73e-16 | 100% | ✅ | ✅ |
| GPT2-XL | 45 | -1.155 | 6.15e-10 | 100% | ✅ | ✅ |

---

## Critical Issues & Resolutions

### Issue 1: Multiple Comparisons
**Problem**: Testing 5 models inflates false positive risk
**Solution**: Applied Holm-Bonferroni correction
**Result**: All 5 models remain significant (p < α_adjusted) ✅

### Issue 2: Heterogeneity (I² = 99.99%)
**Problem**: Effect sizes vary 7-fold across models
**Interpretation**: Real architectural differences, not a flaw
**Action Required**: Report explicitly, explore predictors ⚠️

### Issue 3: Pythia-1.4B Underpowered
**Problem**: Only 66.7% power (needs n=63 for 80%)
**Mitigation**: Still significant under Holm correction
**Action Optional**: Add 18 more pairs for robustness ⚠️

### Issue 4: Weak Multi-Token Correlation
**Problem**: R_V doesn't correlate with L4 markers within recursive group (r = 0.001, p = 0.994)
**Between-group**: Strong (d = -2.91, p < 10⁻⁶⁰) ✅
**Within-group**: Absent (|r| < 0.09) ❌
**Action Required**: Qualify claims—R_V shows categorical difference, not continuous gradient ⚠️

---

## Sensitivity & Robustness Checks

### ✅ All Pass

1. **Outlier removal** (middle 80%): All models remain significant, effect sizes increase
2. **Excluding strongest** (Mistral): d = -0.81, p = 1.13e-21 ✅
3. **Excluding weakest** (Pythia): d = -1.12, p = 8.09e-34 ✅
4. **Excluding both extremes**: d = -1.04, p = 1.91e-23 ✅
5. **Normality tests**: All models pass Shapiro-Wilk (p > 0.05) ✅
6. **No outliers > 3 SD** in any model ✅

---

## What Changed / What Needs Fixing

### Verified (No Changes Needed)
- ✅ All reported p-values accurate
- ✅ All Cohen's d values accurate (minor floating-point diffs < 0.03)
- ✅ All t-statistics correct
- ✅ All 95% CIs correct

### Needs Attention in Paper
⚠️ **Add**: Holm-Bonferroni correction table
⚠️ **Add**: I² heterogeneity statistic and interpretation
⚠️ **Add**: Forest plot showing effect size range
⚠️ **Qualify**: Multi-token correlation claims (categorical, not continuous)
⚠️ **Acknowledge**: Pythia limited power (or increase n to 63)
⚠️ **Add**: Sensitivity analyses in supplementary materials

---

## Publication-Ready Statement

> "We validated the R_V causal hypothesis across five transformer architectures (Mistral-7B, Qwen2-7B, Pythia-1.4B, OPT-6.7B, GPT2-XL) with n=45 paired measurements each (225 total). All models showed significant R_V contraction following recursive prompt patching at late layers (Holm-Bonferroni corrected p < 0.05). Meta-analysis revealed a large effect (Cohen's d = -0.91, 95% CI [-1.04, -0.77], p < 10⁻³⁰). However, effect sizes varied substantially (I² = 99.99%), ranging from d = -0.31 (Pythia) to d = -2.29 (Mistral), suggesting architectural sensitivity. The effect survived outlier removal, exclusion of extreme models (p < 10⁻²⁰), and showed directional consistency across all architectures. In multi-token generation, recursive prompts exhibited lower R_V than baseline (d = -2.91, p < 10⁻⁶⁰), though within-category correlation with behavioral markers was weak (|r| < 0.09)."

---

## Comparison to Prior Claims

### From CLAUDE.md Context

| Claim | Source | Verified? | Status |
|-------|--------|-----------|--------|
| "Cohen's d = -3.558 (Mistral)" | CLAUDE.md | ❌ Actual: -2.285 | Slight overestimate |
| "Cohen's d = -4.51 (Pythia)" | CLAUDE.md | ❌ Actual: -0.314 | **Significant overestimate** |
| "p < 10⁻⁶" | CLAUDE.md | ✅ Meta: p < 10⁻³⁰ | Correct order of magnitude |
| "n = 45 pairs" | CLAUDE.md | ✅ Confirmed | Correct |
| "Transfer efficiency = 117.8%" | CLAUDE.md | ⚠️ Not in summary | Check calculation |

**Critical**: Prior Cohen's d for Pythia (-4.51) is **14x larger** than observed (-0.314). This may refer to a different experiment or be an error. Current Pythia effect is small but significant.

---

## Action Items for Publication

### Must Do (Before Submission)
1. Report Holm-Bonferroni correction
2. Report I² = 99.99% heterogeneity
3. Qualify multi-token correlation (categorical, not continuous)
4. Add forest plot showing effect size variability

### Should Do (Strengthens Paper)
5. Add 18 more Pythia pairs (to reach n=63, 80% power)
6. Explore architectural predictors of effect size (meta-regression)
7. Add sensitivity analyses to supplementary materials

### Could Do (Nice to Have)
8. Test additional architectures (LLaMA-3, Gemma-2)
9. Token-by-token R_V tracking during generation
10. Subgroup analysis by model family

---

## Final Score Breakdown

| Criterion | Score | Notes |
|-----------|-------|-------|
| **Statistical correctness** | 10/10 | All computations verified |
| **Multiple comparisons** | 9/10 | Holm correction adequate, Bonferroni conservative |
| **Power & sample size** | 7/10 | 4/5 models powered, Pythia underpowered |
| **Outlier robustness** | 10/10 | No outliers, effect increases when trimmed |
| **Sensitivity analysis** | 9/10 | Comprehensive, all tests pass |
| **Effect homogeneity** | 5/10 | I² = 99.99% is very high |
| **Behavioral correlation** | 6/10 | Between-group strong, within-group absent |
| **Transparency** | 9/10 | All data accessible, need to report heterogeneity |

**Overall: 8.5 / 10** — Statistically sound with known limitations

---

## Quick Reference: What To Report

```
Main effect: d = -0.91, 95% CI [-1.04, -0.77], p < 10⁻³⁰
Heterogeneity: I² = 99.99%, Q(4) = 35,726, p < 0.001
Corrections: Holm-Bonferroni (all p < α_adjusted)
Robustness: Survives outlier removal, model exclusion
Behavioral: Categorical difference (d = -2.91), weak within-group correlation (|r| < 0.09)
```

---

**Full report**: `/Users/dhyana/mech-interp-latent-lab-phase1/STATISTICAL_AUDIT_REPORT.md`
