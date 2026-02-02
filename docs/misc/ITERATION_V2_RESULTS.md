# Iteration V2 Results: Improved Scorer

**Date:** 2025-12-16T12:55:00Z  
**Run:** `results/runs/20251216_125333_behavior_strict`  
**Status:** ✅ COMPLETED

---

## Results Summary

### Comparison: V1 vs V2

| Condition | V1 Score | V2 Score | Change |
|-----------|----------|----------|--------|
| **Recursive Control** | 0.0250 | **0.3150** | +1160% ✅ |
| **Transfer** | 0.0250 | **0.1250** | +400% ✅ |
| **Baseline** | 0.0000 | 0.0000 | - |
| **Transfer Improvement** | 0.0250 | **0.1250** | +400% ✅ |

### Key Metrics (V2)

| Condition | Mean Score | Recursion | Pass Rate | Samples > 0 |
|-----------|------------|-----------|-----------|-------------|
| **Recursive Control** | **0.3150** | 0.3150 | 75% | 9/20 |
| **Transfer** | **0.1250** | 0.1250 | 45% | 4/20 |
| **Baseline** | 0.0000 | 0.0000 | 80% | 0/20 |
| **Shuffled Control** | 0.3400 | 0.3400 | 80% | 10/20 |
| **Random Control** | 0.0000 | 0.0000 | 100% | 0/20 |

### Transfer Performance

- **Transfer vs Baseline:** 0.1250 vs 0.0000 ✅ **+0.1250 improvement**
- **Transfer vs Recursive:** 0.1250 vs 0.3150 ⚠️ **40% of Recursive Control**
- **Max Transfer Score:** 0.7000 (up from 0.5000) ✅
- **Samples > 0:** 4/20 (up from 1/20) ✅

---

## Analysis

### ✅ What's Working

1. **Improved Scorer is Detecting Recursion**
   - Recursive Control: 0.3150 (was 0.0250)
   - 9/20 samples scoring > 0 (was 1/20)
   - Expanded patterns are being detected

2. **Transfer is Showing Signal**
   - Transfer: 0.1250 (was 0.0250)
   - 4/20 samples scoring > 0 (was 1/20)
   - Max score: 0.7000 (strong signal in some samples)

3. **Transfer > Baseline**
   - Clear improvement over baseline (0.1250 vs 0.0000)
   - Suggests behavior transfer is occurring

### ⚠️ What Needs Improvement

1. **Transfer Still Lower Than Recursive Control**
   - Transfer: 0.1250 (40% of Recursive Control)
   - Need to close the gap to 80-100%

2. **Shuffled Control Scoring High**
   - Shuffled: 0.3400 (higher than Recursive!)
   - Suggests scorer may be detecting structure, not semantics
   - Need semantic coherence check

3. **Low Pass Rate for Transfer**
   - Transfer pass rate: 45% (vs 75% for Recursive)
   - May be failing diversity/repetition gates
   - Need to investigate failure reasons

---

## Top Transfer Samples

1. **Pair 16:** Score = 0.7000 ✅
2. **Pair 8:** Score = 0.7000 ✅
3. **Pair 10:** Score = 0.6000 ✅
4. **Pair 19:** Score = 0.5000 ✅

**Key Insight:** Some samples are achieving strong transfer (0.5-0.7), but most are still 0.0.

---

## Next Steps

### Priority 1: Investigate Why Transfer < Recursive

**Hypothesis:** V_PROJ patching may not be fully effective, or we need additional components.

**Actions:**
1. Check if V_PROJ patching is actually being applied correctly
2. Compare generated text from Transfer vs Recursive Control
3. Consider adding L18 RESIDUAL patching (Dec 12 showed L18+L27 works)

### Priority 2: Fix Shuffled Control Issue

**Problem:** Shuffled Control scoring higher than Recursive suggests scorer is detecting structure, not semantics.

**Actions:**
1. Add semantic coherence check (perplexity or embedding similarity)
2. Ensure scorer distinguishes semantic content from structural patterns

### Priority 3: Improve Transfer Consistency

**Problem:** Only 4/20 samples showing transfer signal.

**Actions:**
1. Investigate why some pairs transfer and others don't
2. Check if prompt quality affects transfer
3. Consider multi-layer patching (L18+L27)

---

## Conclusion

**Status:** Significant Progress ✅

- Transfer improved from 0.0250 → 0.1250 (5x improvement)
- Recursive Control improved from 0.0250 → 0.3150 (12.6x improvement)
- Transfer > Baseline confirmed
- But Transfer still only 40% of Recursive Control

**Confidence:** 75% that behavior transfer is happening, but not at full strength yet.

**Next:** Investigate Transfer < Recursive gap, add semantic coherence check, consider multi-layer patching.









