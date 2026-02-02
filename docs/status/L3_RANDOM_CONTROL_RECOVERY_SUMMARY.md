# L3 Random Control Experiment - Recovery Summary
**Date:** January 5, 2025  
**Status:** Process killed after 3h 53m (stuck on pair 16/30 of random_3)

---

## What We Know

### Completed Conditions (from log messages):
1. **TRUE steering:** 30/30 pairs ✅
2. **Orthogonal:** 30/30 pairs ✅
3. **Random 1:** 30/30 pairs ✅
4. **Random 2:** 30/30 pairs ✅
5. **Random 3:** 16/30 pairs ⚠️ (stuck here)

**Total completed:** 136/150 pairs (91%)

---

## What Was Lost

**All actual results data is lost:**
- Results were accumulated in memory (`results` list)
- Only saved to CSV at the very end (line 260-262)
- Process was killed before saving
- No intermediate saves or checkpoints

**Lost data includes:**
- R_V deltas for all 136 completed pairs
- Mode score deltas
- Coherence scores
- Generated text outputs
- All statistical comparisons

---

## Why It Got Stuck

**Issues identified:**
1. **R_V computation bug:** Computes R_V on `base_text` instead of `generated_text` (line 201)
2. **Non-deterministic generation:** `temperature=0.7, do_sample=True` (slower, less reproducible)
3. **Mode score errors:** Frequent tensor size mismatches
4. **No intermediate saves:** All results lost if process crashes

**Stuck on:** Random 3, pair 16/30 - likely hung in R_V computation or generation

---

## What We Can Recover

**Nothing - all data was in memory only.**

**What we know:**
- 4 conditions completed fully (120 pairs)
- 1 condition partially completed (16 pairs)
- Total: 136/150 pairs processed before getting stuck

**What we don't know:**
- Actual R_V deltas
- Mode score deltas  
- Coherence scores
- Whether TRUE steering was different from random controls
- Any statistical comparisons

---

## Recommendations

1. **Fix the code issues:**
   - R_V on generated text (not base_text)
   - Deterministic generation (temperature=0.0)
   - Fix mode score tensor size handling
   - Add intermediate saves (save after each condition)

2. **Restart with fixes:**
   - Should complete in ~20-30 minutes with fixes
   - Will have proper results this time

3. **Add checkpointing:**
   - Save results after each condition completes
   - Don't lose everything if process crashes

---

**Status:** No recoverable results - all data lost when process was killed.


