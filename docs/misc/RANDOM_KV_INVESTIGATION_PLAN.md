# Random KV Effect Investigation Plan

**Status:** ⏳ Waiting for disk space / model loading fix

## Problem

In Experiment 2 (KV-Only Control), we found that:
- **KV-only:** Expression rate = 12.00%
- **Random KV:** Expression rate = 12.00% (same as KV-only!)

This suggests the effect might not be content-specific.

## Original "Random KV" Method

Looking at `experiment_kv_only_control.py`, the "random KV" was actually:
- KV cache from a **different recursive prompt** (pair_idx + 1)
- Not truly random - still structured KV cache from recursive content

## Investigation Plan

Created `experiment_random_kv_investigation.py` to test:

### Conditions:
1. **Control** - No KV replacement
2. **Recursive KV** - KV from recursive prompt (positive control)
3. **Gaussian KV** - Truly random Gaussian noise matching reference shape
4. **Shuffled KV** - Same tokens as reference, but shuffled sequence order
5. **Baseline KV** - KV from baseline prompt (original "random" method)

### Hypotheses:

**If Gaussian KV shows same effect as recursive KV:**
- → Effect is NOT content-specific (any KV replacement works)
- → Suggests mechanism is about KV cache replacement itself, not content

**If Gaussian KV shows NO effect but baseline KV does:**
- → Effect requires structured KV cache (from real prompts)
- → Suggests mechanism is about KV cache structure/content

**If shuffled KV shows same effect as recursive KV:**
- → Effect is about KV cache structure, not token order
- → Suggests mechanism doesn't depend on sequence ordering

## Expected Runtime

- N=50 pairs
- 5 conditions per pair
- ~1.5 seconds per condition
- **Total: ~6-7 minutes**

## Next Steps

1. ✅ Script created
2. ⏳ Fix disk space issue
3. ⏳ Run experiment
4. ⏳ Analyze results

## Files

- `experiment_random_kv_investigation.py` - Main investigation script
- Results will be saved to `results/path_b_validation/runs/TIMESTAMP_random_kv_investigation/`









