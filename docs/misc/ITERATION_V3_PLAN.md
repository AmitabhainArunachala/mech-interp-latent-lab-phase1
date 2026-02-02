# Iteration V3: Improve Consistency

**Goal:** Get all pairs to show behavior transfer (not just 4/20)

---

## Current Status

- ✅ **Perfect matches:** 2/20 pairs show 100% transfer
- ⚠️ **Consistency issue:** 16/20 pairs score 0.0
- ✅ **Mechanism confirmed:** Top samples prove it works

---

## V3 Improvements

### 1. Enhanced Patching Verification

**Problem:** May not be patching correctly for all sequence lengths.

**Fix:**
- Add logging to verify patching is applied
- Check window size handling for edge cases
- Ensure device/dtype compatibility

### 2. Multi-Layer Patching

**Hypothesis:** Dec 12 showed L18+L27 works better than L27 alone.

**Action:**
- Add L18 RESIDUAL patching alongside L27 V_PROJ
- Test if this improves consistency

### 3. Prompt Filtering

**Problem:** Some prompt pairs may not be suitable for transfer.

**Action:**
- Pre-filter by geometric signature strength (R_V < 0.9)
- Match prompt semantic similarity
- Ensure length compatibility (both >= 16 tokens)

### 4. Improved Error Handling

**Problem:** Silent failures may be causing 0.0 scores.

**Action:**
- Add try/except with detailed logging
- Track which pairs fail and why
- Verify V_PROJ extraction succeeded

---

## Expected Outcome

**Before V3:**
- Mean Transfer: 0.1250
- Samples > 0: 4/20 (20%)

**After V3 (Target):**
- Mean Transfer: 0.25-0.30
- Samples > 0: 15-18/20 (75-90%)
- Perfect matches: 5-10 pairs

---

## Implementation Priority

1. **Add patching verification/logging** (Quick win)
2. **Test multi-layer patching** (L18+L27)
3. **Add prompt filtering** (Prevent bad pairs)
4. **Improve error handling** (Catch failures)

---

## Status

🔄 **READY TO IMPLEMENT**









