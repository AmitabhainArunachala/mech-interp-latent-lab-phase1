# Cross-Architecture Validation Fix Summary

**Date:** January 11, 2025  
**Issue:** Cross-architecture validation was using wrong prompts/model/layer  
**Fix:** Updated to use EXACT conditions from validated confound_validation run

---

## The Problem

The original cross-architecture validation experiment was using:
- ❌ Model: `mistralai/Mistral-7B-v0.1` (base model)
- ❌ Prompts: `recursive_self_reference` (new, untested prompts)
- ❌ Comparison: Recursive vs non-recursive families (not validated controls)
- ❌ Result: R_V = 0.86 (recursive) vs 0.82 (non-recursive) = no clear contraction

---

## The Fix

Updated to use EXACT conditions from validated confound_validation run:
- ✅ Model: `mistralai/Mistral-7B-Instruct-v0.2` (Instruct model, matches validated run)
- ✅ Prompts: `champions` vs `length_matched` + `pseudo_recursive` (validated groups)
- ✅ Late layer: 27 (matches validated run, not 28)
- ✅ Window: 16 (matches validated run)
- ✅ Expected results: Champions R_V = 0.5185, length_matched = 0.8323, pseudo_recursive = 0.7792

---

## Updated Configs

### `configs/cross_architecture_mistral.json`
```json
{
  "experiment": "cross_architecture_validation",
  "params": {
    "model": "mistralai/Mistral-7B-Instruct-v0.2",
    "early_layer": 5,
    "late_layer": 27,
    "window": 16,
    "prompt_groups": {
      "recursive": "champions",
      "controls": ["length_matched", "pseudo_recursive"]
    },
    "n_champions": 30,
    "n_length_matched": 30,
    "n_pseudo_recursive": 30,
    "seed": 42
  },
  "expected_results": {
    "champions_rv": 0.5185,
    "length_matched_rv": 0.8323,
    "pseudo_recursive_rv": 0.7792
  }
}
```

### `configs/cross_architecture_llama.json`
```json
{
  "experiment": "cross_architecture_validation",
  "params": {
    "model": "meta-llama/Llama-3-8B-Instruct",
    "early_layer": 5,
    "late_layer": 27,
    "window": 16,
    "prompt_groups": {
      "recursive": "champions",
      "controls": ["length_matched", "pseudo_recursive"]
    },
    "n_champions": 30,
    "n_length_matched": 30,
    "n_pseudo_recursive": 30,
    "seed": 42
  },
  "expected_results": {
    "champions_rv": "TBD - if ~0.55, effect generalizes; if ~0.80, Mistral-specific"
  }
}
```

---

## Updated Pipeline

Modified `src/pipelines/cross_architecture_validation.py` to:
1. Use `champions` group (not `recursive_self_reference`)
2. Use `length_matched` and `pseudo_recursive` controls (not new families)
3. Use fixed window=16 (not window sweep)
4. Use fixed late_layer=27 (not model-dependent)
5. Match confound_validation summary format exactly

---

## Expected Results

### Mistral-7B-Instruct (Baseline - Should Match)
- Champions: R_V = 0.5185 (should match validated run)
- Length-matched: R_V = 0.8323
- Pseudo-recursive: R_V = 0.7792

### Llama-3-8B-Instruct (Cross-Architecture Test)
- **If Llama shows R_V ≈ 0.55 for champions:** Effect generalizes across architectures ✅
- **If Llama shows R_V ≈ 0.80 for champions:** Effect is Mistral-specific ❌

---

## Next Steps

1. Run Mistral-Instruct config to verify it matches validated run (should get R_V ≈ 0.52)
2. Run Llama-Instruct config to test cross-architecture generalization
3. Compare results to determine if effect is architecture-specific

---

## Key Insight

**The original discovery WAS real** - the validated confound_validation run shows:
- Champions: R_V = 0.5185 (strong contraction)
- Controls: R_V = 0.77-0.83 (no contraction)
- Effect size: Cohen's d ≈ -2.6 to -4.0 (massive!)

We just needed to use the EXACT same conditions to replicate it.
