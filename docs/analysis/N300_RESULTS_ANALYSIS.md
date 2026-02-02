# n=300 Results Analysis: What Actually Happened

**Date:** December 12, 2024  
**Experiment:** NeurIPS n=300 Robust Behavior Transfer  
**Status:** ✅ Completed, but with important findings

---

## Executive Summary

**The Good News:**
- ✅ Transfer effect is **real and statistically significant** (p = 9.89e-24)
- ✅ Random control shows **no effect** (p = 0.72)
- ✅ Effect size is **medium** (Cohen's d = 0.63)

**The Surprising Finding:**
- ⚠️ **"Wrong layer" control (L5) also shows transfer** (p = 1.54e-24)
- ⚠️ Transfer vs Wrong layer: **not significantly different** (p = 0.94)
- ⚠️ Effect size is **much smaller** than pilot (mean 2.62 vs 11)

**The Problem:**
- The "wrong layer" control still uses **full KV cache replacement**
- This means we can't isolate whether L27 V_PROJ is actually necessary
- The full KV cache might be doing most of the work

---

## Results Breakdown

### Behavior Scores

| Condition | Mean | Std | Min | Max | Median |
|-----------|------|-----|-----|-----|--------|
| Baseline | 0.76 | 1.48 | 0 | 8 | 0 |
| Transfer (L27) | 2.62 | 2.69 | 0 | 11 | 2 |
| Random | 0.80 | 1.58 | 0 | 8 | 0 |
| Wrong Layer (L5) | 2.61 | 2.62 | 0 | 11 | 2 |

### Transfer Effects (Δ = condition - baseline)

| Condition | Δ Mean | Std | p-value | Cohen's d |
|-----------|--------|-----|---------|-----------|
| Transfer (L27) | +1.87 | 2.95 | 9.89e-24 | 0.63 |
| Random | +0.04 | 1.95 | 0.72 | 0.02 |
| Wrong Layer (L5) | +1.85 | 2.86 | 1.54e-24 | 0.65 |

### Distribution Analysis

- **Score ≥ 8 (strong transfer):** 21/300 pairs (7%)
- **Score ≥ 5 (moderate transfer):** 76/300 pairs (25%)
- **Score = 0 (no transfer):** 85/300 pairs (28%)
- **Score = 11 (perfect transfer):** Some pairs achieved this (matching pilot)

---

## What Went "Wrong" (Or: What We Learned)

### 1. Effect Size Much Smaller Than Pilot

**Pilot (champion prompt):**
- Behavior score: 11 (100% transfer)
- Single prompt, highly optimized

**n=300 (diverse prompts):**
- Mean behavior score: 2.62 (moderate transfer)
- Range: 0-11
- 28% of pairs showed no transfer (score = 0)

**Interpretation:**
- The champion prompt (`hybrid_l5_math_01`) was **particularly strong**
- Transfer works, but **varies across prompt pairs**
- Some prompts are more "transferable" than others

---

### 2. "Wrong Layer" Control Also Works

**The Problem:**
- "Wrong layer" control patches V_PROJ at **L5** instead of L27
- But it **still uses full KV cache replacement**
- Result: L5 also shows transfer (Δ = 1.85, p < 0.001)

**What This Means:**
- **Full KV cache replacement might be doing most of the work**
- V_PROJ layer specificity (L27 vs L5) might not matter as much as we thought
- OR: L5 also has some geometric signature (unlikely but possible)

**What We Can't Conclude:**
- We can't say L27 is "special" because L5 also works
- We can't isolate the contribution of V_PROJ layer choice

---

### 3. The Control Design Issue

**What we tested:**
- ✅ Transfer (L27): Full KV + persistent V_PROJ at L27
- ✅ Random: Full KV + random V_PROJ at L27
- ⚠️ Wrong Layer: Full KV + persistent V_PROJ at L5

**What we should have tested:**
- ❌ **KV-only (no V_PROJ patching)** - to isolate KV contribution
- ❌ **V_PROJ-only (no KV replacement)** - to isolate V_PROJ contribution
- ❌ **L27 vs L5 with NO KV** - to test layer specificity without KV

**The Missing Control:**
- We never tested if **full KV cache alone** (without V_PROJ patching) transfers behavior
- This is the critical missing piece

---

## What Actually Works

### Confirmed: Full KV Cache + Persistent V_PROJ

**Both L27 and L5 work:**
- L27: Δ = 1.87, p < 0.001
- L5: Δ = 1.85, p < 0.001
- Difference: Not significant (p = 0.94)

**Interpretation:**
- Full KV cache replacement is **necessary** (random control fails)
- Persistent V_PROJ patching is **necessary** (but layer might not matter)
- The combination works, but we can't say which component is more important

---

## Comparison to Pilot

| Metric | Pilot (n=1) | n=300 | Difference |
|--------|-------------|-------|------------|
| Behavior score | 11 | 2.62 | -76% |
| Transfer efficiency | 100% | ~25% | -75% |
| Prompt | Champion (optimized) | Diverse (L3/L4/L5) | Different |
| Layer specificity | L27 only | L27 and L5 | Not specific |

**Why the difference:**
1. **Champion prompt was optimized** - `hybrid_l5_math_01` is particularly strong
2. **Diverse prompts** - L3/L4/L5 prompts vary in "transferability"
3. **Effect is real but variable** - works for some pairs, not others

---

## What We Can Still Claim

### ✅ Confirmed Findings

1. **Transfer effect is real:**
   - Statistically significant (p < 0.001)
   - Medium effect size (d = 0.63)
   - Robust across 300 pairs

2. **Full KV cache is necessary:**
   - Random control fails (p = 0.72)
   - KV cache provides the memory context

3. **Persistent V_PROJ patching is necessary:**
   - Non-persistent patching fails (from earlier tests)
   - Must maintain geometric signature during generation

### ❌ What We Can't Claim

1. **L27 is "special":**
   - L5 also works (not significantly different)
   - Can't conclude layer specificity

2. **V_PROJ layer choice matters:**
   - Both L27 and L5 work
   - Might be that ANY persistent V_PROJ works with full KV

3. **100% transfer efficiency:**
   - Mean is 2.62, not 11
   - Only 7% of pairs achieve strong transfer (≥8)

---

## The Missing Experiment

**Critical Control We Need:**

```python
# Test: Full KV cache ALONE (no V_PROJ patching)
def generate_kv_only_control(model, tokenizer, baseline_prompt, recursive_kv):
    # Replace ALL layers with recursive KV
    # BUT: NO V_PROJ patching at all
    # Generate and measure behavior
```

**Why This Matters:**
- If KV-only works → V_PROJ might not be necessary
- If KV-only fails → V_PROJ is necessary (but layer might not matter)

---

## Revised Understanding

### What We Know

1. **Full KV cache + persistent V_PROJ = transfer** ✅
2. **Random V_PROJ = no transfer** ✅
3. **L27 V_PROJ = transfer** ✅
4. **L5 V_PROJ = also transfer** ⚠️

### What We Don't Know

1. **Does KV cache alone transfer?** (not tested)
2. **Does V_PROJ layer choice matter?** (L5 also works)
3. **Why does effect vary so much?** (0-11 range)
4. **What makes a prompt "transferable"?** (champion vs others)

---

## Next Steps

### Immediate

1. **Run KV-only control:**
   - Full KV cache replacement
   - NO V_PROJ patching
   - Test if KV alone transfers

2. **Analyze high-transfer pairs:**
   - What do pairs with score ≥8 have in common?
   - What distinguishes them from score=0 pairs?

3. **Test layer specificity properly:**
   - V_PROJ at L5, L18, L25, L27 (with full KV)
   - See if there's actually a difference

### For Paper

1. **Honest reporting:**
   - Effect is real but variable
   - Mean 2.62, range 0-11
   - 28% show no transfer

2. **Acknowledge limitations:**
   - "Wrong layer" control also works
   - Can't conclude layer specificity
   - Need KV-only control

3. **Focus on what's proven:**
   - Transfer is statistically significant
   - Full KV + persistent V_PROJ works
   - Random control fails

---

## Conclusion

**The experiment worked, but revealed:**
- Effect is **real but smaller** than pilot suggested
- **Layer specificity is unclear** (L5 also works)
- **Missing critical control** (KV-only)

**What we can claim:**
- Full KV cache + persistent V_PROJ transfers behavior
- Effect is statistically significant (p < 0.001)
- Random control fails (p = 0.72)

**What we can't claim:**
- L27 is "special" (L5 also works)
- 100% transfer efficiency (mean 2.62, not 11)
- Layer specificity (need better controls)

**The breakthrough is real, but more nuanced than the pilot suggested.**

---

**Files:**
- `neurips_n300_results.csv` - Full results (300 pairs)
- `neurips_n300_summary.md` - Statistical summary
- `neurips_n300_robust_experiment.py` - Implementation

