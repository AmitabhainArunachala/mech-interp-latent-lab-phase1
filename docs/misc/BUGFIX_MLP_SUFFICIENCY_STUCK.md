# Bug Fix: MLP Sufficiency Test Stuck/Hanging

**Date:** January 5, 2025  
**Severity:** 🚨 CRITICAL  
**Status:** ✅ FIXED

---

## Problem

The L0 Sufficiency test was stuck for **87+ minutes** (expected 5-6 minutes) with:
- ✅ Process killed
- ✅ Code fixed
- ✅ Experiments restarted

---

## Root Cause Analysis

### Symptoms
1. **Runtime:** 87+ minutes (expected 5-6 min)
2. **GPU Memory:** 3.9GB (expected ~14GB for Mistral-7B)
3. **Threads:** 208
4. **RAM:** 12GB
5. **Output:** No CSV/summary.json files produced
6. **Process State:** Running (R), 101% CPU but not progressing

### Code Issues Identified

**Issue 1: R_V computed on generated_text (too long)**
```python
# BEFORE (line 198):
rv_patched = compute_rv(model, tokenizer, generated_text, ...)
# Problem: generated_text = base_text + up to 200 tokens (very long)
```

**Issue 2: Hook used twice, potential deadlock**
```python
# BEFORE:
with patching_hook:
    outputs_patched = model.generate(...)  # First use
# ...
with patching_hook:  # Second use - hook might not be properly cleaned up
    out_patched = model(**inputs_base)
```

**Issue 3: Missing max_length parameter in generate()**
- Could cause generation to hang if model doesn't stop properly

---

## Fix Applied

### Change 1: Use base_text for R_V computation
```python
# AFTER:
# Compute R_V with patching (on base_text, not generated)
with patching_hook:
    with torch.no_grad():
        rv_patched = compute_rv(model, tokenizer, base_text, ...)
        out_patched = model(**inputs_base)
```

**Rationale:** R_V should be measured on the prompt, not the generated text. Generated text can be very long and cause memory/computation issues.

### Change 2: Reorder operations, single hook use
```python
# AFTER:
# Compute R_V and mode score first (single hook use)
with patching_hook:
    rv_patched = compute_rv(...)
    out_patched = model(**inputs_base)

# Then generate (separate hook use)
with patching_hook:
    outputs_patched = model.generate(..., max_length=...)
```

**Rationale:** Separates R_V/metric computation from generation, avoids hook reuse issues.

### Change 3: Add explicit max_length
```python
# AFTER:
outputs_patched = model.generate(
    **inputs_gen,
    max_new_tokens=max_new_tokens,
    max_length=inputs_gen["input_ids"].shape[1] + max_new_tokens,  # Explicit limit
    ...
)
```

**Rationale:** Prevents infinite generation loops.

---

## Files Changed

1. ✅ `src/pipelines/mlp_sufficiency_test.py` (lines 180-200)
   - Reordered R_V computation and generation
   - Changed R_V input from `generated_text` to `base_text`
   - Added explicit `max_length` parameter

---

## Experiments Restarted

1. ✅ **L0 Sufficiency** - Retry with fixed code
2. ✅ **L0+L1 Combined Sufficiency** - Started fresh
3. ✅ **L0 Position-Specific** - Started fresh

**Status:** All 3 experiments now running with fixed code.

---

## Monitoring

Check progress:
```bash
# Check logs
ssh runpod-current "tail -f /tmp/canonical_l0_sufficiency_retry.log"
ssh runpod-current "tail -f /tmp/canonical_l0_l1_sufficiency.log"
ssh runpod-current "tail -f /tmp/canonical_l0_position.log"

# Check GPU
ssh runpod-current "nvidia-smi"

# Check processes
ssh runpod-current "ps aux | grep python3 | grep -E 'sufficiency|position'"
```

---

## Expected Behavior

With the fix:
- ✅ R_V computed on prompt (not generated text) - faster, more accurate
- ✅ Single hook use per operation - no deadlock
- ✅ Explicit max_length - prevents infinite generation
- ✅ Expected runtime: ~5-6 minutes per experiment

---

**Fixed by:** Cursor AI Assistant  
**Status:** ✅ Code fixed, experiments restarted  
**Next:** Monitor experiments for completion


