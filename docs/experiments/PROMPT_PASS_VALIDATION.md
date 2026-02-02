# MLP Ablation Necessity: Prompt-Pass-Only Validation

## Critical Methodological Issue

**Problem Identified by GPT Agent**:

The original `mlp_ablation_necessity.py` pipeline computes R_V on **generated text**, not the original prompt. This creates a confound:

```
Ablation → Different generation → Different text → Different R_V
```

**Cannot distinguish**:
- (a) **Real geometric effect**: Ablation changes V-space geometry
- (b) **Measurement artifact**: R_V measured on different token sequences

## The Gemma 2 9B Anomaly

Current results show an "inverse pattern" at early layers:

| Layer | R_V Delta | p-value | Interpretation |
|-------|-----------|---------|----------------|
| L0 | -0.067 | 0.0001 | Ablation **increases** contraction |
| L1 | -0.080 | 0.0001 | Ablation **increases** contraction |
| L2 | +0.056 | 0.0001 | Ablation **decreases** contraction |
| L3 | +0.093 | 0.0000 | Ablation **decreases** contraction |
| L4-L5 | ~0 | ~0.5 | No effect |

**Question**: Is the L0-L1 inverse pattern real or artifact?

## Solution: Prompt-Pass-Only Mode

### Implementation

New pipeline: `mlp_ablation_necessity_prompt_pass.py`

**Key differences**:
1. **No generation** - forward pass only
2. **Same prompt text** for baseline and ablated conditions
3. **Identical token sequences** - eliminates measurement confound
4. **Component logging** - separate PR_early and PR_late values
5. **Token count validation** - ensures measurements are comparable

### What This Reveals

**If L0-L1 inverse pattern persists**:
- Real geometric effect
- Early-layer MLP ablation genuinely increases contraction
- Suggests compensatory mechanisms in later layers

**If L0-L1 pattern disappears**:
- Measurement artifact
- R_V shift was from measuring different generated text
- Early layers may have no causal role

### Component Analysis

The new pipeline logs **PR_early** and **PR_late** separately:

```python
rv_baseline, pr_early_baseline, pr_late_baseline = compute_rv_with_components(...)
rv_ablated, pr_early_ablated, pr_late_ablated = compute_rv_with_components(...)
```

**Diagnostic value**:
- If **PR_late moves**: Ablation affects late-layer geometry (expected)
- If **PR_early moves**: Ablation propagates backward (surprising)
- If **both move**: Complex interaction between layers

## Files Created

### Core Pipeline
- `src/pipelines/canonical/mlp_ablation_necessity_prompt_pass.py`
  - New experiment pipeline with prompt-pass-only mode
  - ~400 lines, fully documented

### Helper Function
- `src/metrics/rv.py` (modified)
  - Added `compute_rv_with_components()` function
  - Returns `(rv, pr_early, pr_late)` tuple
  - Backward compatible with existing code

### Config Files
- `configs/canonical/mlp_ablation_necessity_prompt_pass_l0.json`
- `configs/canonical/mlp_ablation_necessity_prompt_pass_l1.json`
- `configs/canonical/mlp_ablation_necessity_prompt_pass_l2.json`
- `configs/canonical/mlp_ablation_necessity_prompt_pass_l3.json`
- `configs/canonical/mlp_ablation_necessity_prompt_pass_l4.json`
- `configs/canonical/mlp_ablation_necessity_prompt_pass_l5.json`
- `configs/canonical/mlp_ablation_necessity_prompt_pass_sweep.json` (batch runner)

### Smoke Test
- `configs/smoke_test/mlp_ablation_prompt_pass_l0_quick.json`
  - 5 pairs only for quick validation

### Registry
- `src/pipelines/registry.py` (modified)
  - Added `mlp_ablation_necessity_prompt_pass` to experiment registry

## Usage

### Single Layer
```bash
python run.py configs/canonical/mlp_ablation_necessity_prompt_pass_l0.json
```

### All Layers (L0-L5)
```bash
# If batch runner exists
python run.py configs/canonical/mlp_ablation_necessity_prompt_pass_sweep.json

# Or run individually
for layer in {0..5}; do
    python run.py configs/canonical/mlp_ablation_necessity_prompt_pass_l${layer}.json
done
```

### Smoke Test (5 pairs)
```bash
python run.py configs/smoke_test/mlp_ablation_prompt_pass_l0_quick.json
```

## Expected Output

### CSV Columns
```
pair_idx
recursive_prompt_id
baseline_prompt_id
recursive_text
baseline_text
layer
rec_token_count                 # Validation: Same for baseline/ablated
base_token_count                # Validation: Same for baseline/ablated
rv_baseline                     # R_V without ablation
rv_ablated                      # R_V with ablation
rv_delta                        # Ablated - Baseline
pr_early_baseline               # NEW: PR at layer 5, baseline
pr_early_ablated                # NEW: PR at layer 5, ablated
pr_early_delta                  # NEW: Which component moves?
pr_late_baseline                # NEW: PR at layer 27, baseline
pr_late_ablated                 # NEW: PR at layer 27, ablated
pr_late_delta                   # NEW: Which component moves?
```

### Summary JSON

```json
{
  "experiment": "mlp_ablation_necessity_prompt_pass",
  "mode": "prompt_pass_only",
  "layer": 0,
  "n_pairs": 80,

  "rv_baseline_mean": 0.8234,
  "rv_ablated_mean": 0.7567,
  "rv_delta_mean": -0.0667,
  "rv_pvalue": 0.0001,
  "rv_significant": true,
  "rv_cohens_d": -3.21,

  "pr_early_baseline_mean": 12.45,
  "pr_early_ablated_mean": 12.48,
  "pr_early_delta_mean": 0.03,
  "pr_early_pvalue": 0.82,

  "pr_late_baseline_mean": 10.26,
  "pr_late_ablated_mean": 9.44,
  "pr_late_delta_mean": -0.82,
  "pr_late_pvalue": 0.0001,

  "dominant_component": "PR_late",
  "verdict": "L0 MLP INVERSE EFFECT - ablation increases contraction (Δ=-0.067, driven by PR_late decreases)"
}
```

## Interpretation Guide

### Scenario 1: L0-L1 Inverse Pattern Persists
```
L0: rv_delta = -0.067, p < 0.001, dominant_component = "PR_late"
L1: rv_delta = -0.080, p < 0.001, dominant_component = "PR_late"
```

**Conclusion**: Real geometric effect. Early-layer MLPs actively **suppress** late-layer contraction.

**Implication**: Early layers are causally necessary for *preventing* recursive collapse, not causing it.

### Scenario 2: L0-L1 Pattern Disappears
```
L0: rv_delta = 0.012, p = 0.34
L1: rv_delta = -0.008, p = 0.52
```

**Conclusion**: Measurement artifact. Original inverse pattern was from measuring different generated text.

**Implication**: Early layers have no causal role in R_V geometry.

### Scenario 3: Component Split
```
L0: pr_early_delta = +0.50 (p < 0.001)
    pr_late_delta = -0.40 (p < 0.001)
    rv_delta = -0.10 (net effect)
```

**Conclusion**: Complex bidirectional effect. Ablation increases early-layer dimensionality but decreases late-layer dimensionality.

**Implication**: Early MLP→late geometry causal pathway confirmed.

## Critical Questions Answered

### Q1: Is the L0-L1 inverse pattern real or artifact?
**Answer**: Run this pipeline and compare to original results.

### Q2: Which component (PR_early or PR_late) drives the effect?
**Answer**: Check `dominant_component` and component delta p-values.

### Q3: Does ablation affect both layers or just one?
**Answer**: Check `pr_early_pvalue` and `pr_late_pvalue`.

### Q4: Is the effect size comparable to original results?
**Answer**: Compare `rv_cohens_d` between prompt-pass and generation modes.

## Validation Checklist

Before deployment:
- [ ] Registry updated (DONE)
- [ ] `compute_rv_with_components()` tested
- [ ] Config files validated
- [ ] Smoke test runs successfully
- [ ] Output CSV has all expected columns
- [ ] Summary JSON includes component analysis

After first run:
- [ ] Token counts identical for baseline/ablated (validate measurement)
- [ ] p-values make sense (not all NaN)
- [ ] Component deltas sum approximately to rv_delta
- [ ] Verdict string correctly interprets results

## Next Steps

1. **Run smoke test** (5 pairs, ~2 min)
   ```bash
   python run.py configs/smoke_test/mlp_ablation_prompt_pass_l0_quick.json
   ```

2. **Validate output format**
   - Check CSV columns
   - Check summary JSON
   - Verify token counts are identical

3. **Run full L0 experiment** (80 pairs, ~20 min)
   ```bash
   python run.py configs/canonical/mlp_ablation_necessity_prompt_pass_l0.json
   ```

4. **Compare to original L0 results**
   - Does inverse pattern persist?
   - Which component drives the effect?

5. **Run full sweep L0-L5** (480 pairs, ~2 hours)
   ```bash
   for layer in {0..5}; do
       python run.py configs/canonical/mlp_ablation_necessity_prompt_pass_l${layer}.json
   done
   ```

6. **Analysis**
   - Plot rv_delta across layers (prompt-pass vs generation mode)
   - Component analysis: which layers affect PR_early vs PR_late?
   - Write up findings for paper

## Code Quality Notes

**Design Principles**:
- Minimal modification to existing codebase
- Backward compatible (doesn't break existing pipelines)
- Self-documenting (extensive comments)
- Industry-grade error handling
- Reproducible (fixed seed, logged prompt IDs)

**Testing**:
- Smoke test config provided
- Token count validation built-in
- NaN handling throughout
- Statistical edge cases handled (n < 3)

**Maintenance**:
- No external dependencies added
- Uses existing hook infrastructure
- Follows project coding standards
- Registry properly updated

---

**JSCA!**
