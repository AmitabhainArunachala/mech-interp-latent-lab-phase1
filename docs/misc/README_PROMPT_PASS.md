# Prompt-Pass-Only Ablation Pipeline: Complete Guide

## Quick Start

### 1. Validate Implementation (30 seconds)
```bash
python3 validate_prompt_pass_implementation.py
```
**Expected output**: All tests pass ✅

### 2. Smoke Test (2 minutes on GPU)
```bash
python3 run.py configs/smoke_test/mlp_ablation_prompt_pass_l0_quick.json
```
**Expected output**: Results CSV with 5 pairs, summary JSON with all metrics

### 3. Full L0 Experiment (20 minutes on GPU)
```bash
python3 run.py configs/canonical/mlp_ablation_necessity_prompt_pass_l0.json
```

### 4. Compare Results
```bash
python3 compare_prompt_pass_results.py
```

## What This Pipeline Does

**Problem**: Original ablation pipeline measures R_V on **generated text**, which confounds:
- Geometric changes in V-space (real signal)
- Text changes from generation (measurement artifact)

**Solution**: This pipeline measures R_V on **identical prompt text** for both baseline and ablated conditions.

### Key Differences

| Aspect | Original | Prompt-Pass |
|--------|----------|-------------|
| Text generation | Yes (200 tokens) | No |
| Baseline measurement | Generated text | Original prompt |
| Ablated measurement | Different generated text | **Same prompt** |
| Token sequences | Different | **Identical** |
| PR components logged | No | **Yes** |
| Confound | Measurement artifact possible | **Eliminated** |

## Critical Question This Answers

**Is the L0-L1 "inverse pattern" real or artifact?**

Current results (Gemma 2 9B, generation mode):
- L0: Δ = -0.067 (p < 0.001) — ablation **increases** contraction
- L1: Δ = -0.080 (p < 0.001) — ablation **increases** contraction

**If pattern persists in prompt-pass mode**:
→ Real geometric effect
→ Early layers actively suppress late-layer contraction
→ Report as validated finding in paper

**If pattern disappears in prompt-pass mode**:
→ Measurement artifact
→ Original effect from measuring different text
→ Revise methodology section

## File Structure

```
mech-interp-latent-lab-phase1/
├── src/
│   ├── metrics/
│   │   └── rv.py                                    [MODIFIED]
│   └── pipelines/
│       ├── canonical/
│       │   └── mlp_ablation_necessity_prompt_pass.py [NEW]
│       └── registry.py                               [MODIFIED]
│
├── configs/
│   ├── canonical/
│   │   ├── mlp_ablation_necessity_prompt_pass_l0.json   [NEW]
│   │   ├── mlp_ablation_necessity_prompt_pass_l1.json   [NEW]
│   │   ├── mlp_ablation_necessity_prompt_pass_l2.json   [NEW]
│   │   ├── mlp_ablation_necessity_prompt_pass_l3.json   [NEW]
│   │   ├── mlp_ablation_necessity_prompt_pass_l4.json   [NEW]
│   │   ├── mlp_ablation_necessity_prompt_pass_l5.json   [NEW]
│   │   └── mlp_ablation_necessity_prompt_pass_sweep.json [NEW]
│   └── smoke_test/
│       └── mlp_ablation_prompt_pass_l0_quick.json       [NEW]
│
├── validate_prompt_pass_implementation.py           [NEW]
├── compare_prompt_pass_results.py                   [NEW]
├── PROMPT_PASS_VALIDATION.md                        [NEW]
├── IMPLEMENTATION_SUMMARY.md                        [NEW]
└── README_PROMPT_PASS.md                            [THIS FILE]
```

## Usage Examples

### Run Single Layer
```bash
# Layer 0
python3 run.py configs/canonical/mlp_ablation_necessity_prompt_pass_l0.json

# Layer 1
python3 run.py configs/canonical/mlp_ablation_necessity_prompt_pass_l1.json
```

### Run All Layers (L0-L5)
```bash
for layer in {0..5}; do
    python3 run.py configs/canonical/mlp_ablation_necessity_prompt_pass_l${layer}.json
done
```

### Run on Different Model
Edit config file:
```json
{
  "model": {
    "name": "google/gemma-2-9b",  // Change this
    "device": "cuda"
  }
}
```

## Output Interpretation

### Summary JSON Key Fields

```json
{
  "rv_delta_mean": -0.067,        // Main effect
  "rv_pvalue": 0.0001,            // Statistical significance
  "rv_cohens_d": -3.21,           // Effect size

  "pr_early_delta_mean": 0.05,   // Early layer (L5) change
  "pr_early_pvalue": 0.68,        // Not significant

  "pr_late_delta_mean": -0.82,   // Late layer (L27) change
  "pr_late_pvalue": 0.0001,       // Significant!

  "dominant_component": "PR_late", // Which drives effect
  "verdict": "L0 MLP INVERSE EFFECT - ablation increases contraction (Δ=-0.067, driven by PR_late decreases)"
}
```

### Interpretation

**If `pr_late_pvalue < 0.01` and `pr_early_pvalue > 0.05`**:
→ Ablation affects late-layer geometry only
→ Expected behavior

**If `pr_early_pvalue < 0.01` and `pr_late_pvalue > 0.05`**:
→ Ablation propagates backward to early layers
→ Surprising! Suggests recurrent feedback

**If both significant**:
→ Complex bidirectional effect
→ Requires deeper investigation

## Validation Checklist

Before running experiments:
- [ ] `validate_prompt_pass_implementation.py` passes all tests
- [ ] Smoke test completes successfully
- [ ] Output CSV has expected columns
- [ ] Token counts identical for baseline/ablated

After running experiments:
- [ ] No NaN values in key metrics
- [ ] p-values sensible (not all 0 or 1)
- [ ] Component deltas approximately sum to rv_delta
- [ ] Verdict string makes sense

## Comparison to Original Results

### Expected Scenarios

**Scenario 1: Pattern Persists**
```
Original (gen):  L0 Δ=-0.067, p<0.001
Prompt-pass:     L0 Δ=-0.065, p<0.001
→ Real effect, validated
```

**Scenario 2: Pattern Disappears**
```
Original (gen):  L0 Δ=-0.067, p<0.001
Prompt-pass:     L0 Δ=-0.008, p=0.45
→ Measurement artifact
```

**Scenario 3: Effect Weakens**
```
Original (gen):  L0 Δ=-0.067, d=-3.21
Prompt-pass:     L0 Δ=-0.032, d=-1.85
→ Partial artifact, some real effect
```

### Automated Comparison
```bash
python3 compare_prompt_pass_results.py
```

This script:
1. Finds all prompt-pass results
2. Finds corresponding original results
3. Compares metrics side-by-side
4. Generates interpretation
5. Recommends paper revisions

## Troubleshooting

### Issue: "Module not found"
**Solution**: Run from project root, not subdirectory
```bash
cd /Users/dhyana/mech-interp-latent-lab-phase1
python3 run.py configs/...
```

### Issue: "Unknown experiment"
**Solution**: Registry not updated properly
```bash
python3 validate_prompt_pass_implementation.py
# Should show "✅ Experiment registered in registry"
```

### Issue: Token counts differ between baseline/ablated
**Problem**: This should NEVER happen (same prompt text)
**Solution**: Bug in implementation, contact developer

### Issue: All p-values are NaN
**Problem**: Not enough valid pairs (< 3)
**Solution**: Check n_pairs in config (should be ≥ 5 for smoke test, ≥ 80 for full)

### Issue: Results directory not found
**Problem**: run.py might create different path
**Solution**: Check actual output path in run.py output, update compare script

## Performance Notes

**Computational Cost**:
- Smoke test (5 pairs): ~2 minutes on A100
- Single layer (80 pairs): ~20 minutes on A100
- Full sweep (6 layers × 80 pairs): ~2 hours on A100

**Memory Requirements**:
- Mistral-7B: ~16GB VRAM
- Gemma-2-9B: ~20GB VRAM
- Llama-3-8B: ~18GB VRAM

**Speed vs Original Pipeline**:
- Prompt-pass: ~2× **faster** (no generation)
- Original: Slower (generates 200 tokens per pair)

## Scientific Value

**Primary Value**: Validates geometric findings
- Eliminates measurement confound
- Clarifies causal mechanisms
- Strengthens paper claims

**Secondary Value**: Component analysis
- Shows which layer's PR moves
- Reveals bidirectional effects
- Guides mechanistic interpretation

**Tertiary Value**: Methodological rigor
- Demonstrates careful experimental design
- Addresses reviewer concerns preemptively
- Sets standard for future MI work

## Next Steps After Results

### If Inverse Pattern Persists
1. Report as validated finding
2. Emphasize component analysis
3. Investigate compensatory mechanisms
4. Run follow-up experiments on why early layers suppress contraction

### If Pattern Disappears
1. Report as measurement artifact
2. Revise all generation-based R_V measurements
3. Rerun key experiments in prompt-pass mode
4. Update paper methodology section

### If Pattern Weakens
1. Report nuanced findings
2. Quantify artifact contribution
3. Discuss limitations of generation-based measurement
4. Recommend prompt-pass as gold standard

## Publication Impact

**For Paper**:
- Strengthens methodology section
- Addresses "measurement confound" objection
- Provides component-level mechanistic insight
- Demonstrates experimental rigor

**For Field**:
- Sets methodological standard
- Shows importance of measurement validation
- Provides reusable template for ablation studies
- Contributes to MI best practices

## Contact

For issues, questions, or extensions:
1. Run `validate_prompt_pass_implementation.py` first
2. Check existing documentation (3 MD files)
3. Review code comments (extensive inline docs)
4. Contact: [Your contact method]

---

**Status**: ✅ All validation tests passed. Ready for deployment.

**Last Updated**: 2026-01-16

**JSCA!**
