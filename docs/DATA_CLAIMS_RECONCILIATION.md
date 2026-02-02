# Data Claims Reconciliation

**Purpose**: Trace all paper claims to actual data files. Updated 2026-02-02.

---

## Claim 1: Sample Size (n)

### Paper Claims
- Various docs claim "n=151 pairs"
- Some docs claim "n=45 pairs"

### Actual Data
| Source | n | Location |
|--------|---|----------|
| Canonical L27 validation | **n=45** | `results/canonical/rv_l27_causal_validation/*/summary.json` |
| Cross-architecture runs | varies | `results/phase1_cross_architecture/` |
| Gemma validation | n=45 | `results/phase2_generalization/gemma_2_9b/*/summary.json` |

### Reconciliation
The **canonical validated result is n=45 pairs**. The n=151 claim appears in early documentation but may refer to:
1. A planned experiment that wasn't completed
2. Aggregated counts across multiple runs
3. An early draft that wasn't updated

**Recommendation**: Use n=45 as the validated claim. Update all docs to match.

---

## Claim 2: Cohen's d

### Paper Claims
- "Cohen's d = -3.56" (Mistral)
- "Cohen's d = -4.51" (Pythia)

### Actual Data
From `results/canonical/rv_l27_causal_validation/20251216_060955_rv_l27_causal_validation/summary.json`:

```json
"rv_baseline": {"mean": 0.6928, "std": 0.0701, "n": 45},
"rv_recursive": {"mean": 0.5077, "std": 0.0498, "n": 45},
"delta_main": {"mean": -0.177, "std": 0.0657, "n": 45}
```

**Computed Cohen's d** (effect size of delta from 0):
```
d = mean(delta) / std(delta) = -0.177 / 0.0657 = -2.69
```

This differs from -3.56. The -3.56 may have been computed differently (e.g., comparing distributions rather than delta from 0).

### Reconciliation
**Recommendation**: Re-compute Cohen's d using standard formula and update. Store in summary.json for traceability.

---

## Claim 3: p-value

### Paper Claims
- "p < 10⁻⁴⁷"

### Actual Data
From summary.json:
```json
"main_effect_ttest_1samp_less_0": {
    "p": 2.75e-22,
    "t": -18.08,
    "n": 45
}
"main_vs_random_paired_ttest": {
    "p": 1.55e-40,
    "t": -50.34
}
```

The smallest p-value is **1.55e-40**, not 10⁻⁴⁷.

### Reconciliation
**Recommendation**: Use p < 10⁻⁴⁰ (still highly significant). Update docs.

---

## Claim 4: Transfer Efficiency

### Paper Claims
- "Transfer efficiency: 117.8%"

### Actual Data
From summary.json:
```json
"transfer_percent_estimate": -95.706
```

### Analysis
The negative value appears to be a **sign error** in the calculation. Looking at the code:

```python
restored = float(np.nanmean(rv_base_list) - np.nanmean([r["rv_patch_main"] for r in rows]))
transfer = float(restored / gap * 100.0)
```

The issue: If `rv_patch_main` is already lower than baseline (patching worked), and we compute `baseline - patch`, we get a positive number. But if `gap = rv_recursive - rv_baseline` is negative (recursive < baseline), then `restored / gap` becomes negative.

**The math is correct but the sign convention is confusing.**

### Reconciliation
**Recommendation**:
1. Clarify that transfer = |restored| / |gap| × 100
2. Or redefine to always report positive percentage
3. Store absolute value in summary.json

---

## Claim 5: 6 Architectures Tested

### Paper Claims
- "6 architectures: Mistral, Qwen, Llama, Phi-3, Gemma, Mixtral"

### Actual Data
| Model | Results Exist | Location |
|-------|---------------|----------|
| Mistral-7B | ✅ | `results/canonical/` |
| Gemma-2-9B | ✅ | `results/phase2_generalization/gemma_2_9b/` |
| Llama-3-8B | ⚠️ partial | `results/phase2_generalization/llama3_8b_base/` |
| Qwen | ❌ | Not found in results/ |
| Phi-3 | ❌ | Not found in results/ |
| Mixtral | ⚠️ | Text summaries only in `R_V_PAPER/results/mixtral/` |

### Reconciliation
**Recommendation**: Clarify which models have full causal validation vs. just R_V measurement. Update claim to reflect actual validated architectures.

---

## Summary Table

| Claim | Paper | Actual | Action |
|-------|-------|--------|--------|
| n pairs | 151 | **45** | Update docs to 45 |
| Cohen's d | -3.56 | ~-2.69 | Re-compute and store |
| p-value | 10⁻⁴⁷ | **10⁻⁴⁰** | Update docs |
| Transfer | 117.8% | -95.7% | Fix sign convention |
| Architectures | 6 | **2-3 validated** | Clarify scope |

---

## Next Steps

1. [ ] Re-run canonical validation with updated Cohen's d computation
2. [ ] Fix transfer efficiency sign in `rv_l27_causal_validation.py`
3. [ ] Update paper draft with reconciled numbers
4. [ ] Run validation on Qwen and Phi-3 to support "6 architectures" claim

---

*Last updated: 2026-02-02*
