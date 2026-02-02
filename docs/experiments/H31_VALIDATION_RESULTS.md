# H31 Validation Results: n=100 Prompts

**Date:** December 14, 2024  
**Model:** Mistral-7B-v0.1  
**GPU:** NVIDIA RTX 6000 Ada Generation  
**Sample:** 50 recursive + 50 baseline prompts

---

## Results Summary

### H31 Entropy at L27

**Recursive prompts (n=50):**
- Mean: **0.430** ± 0.107
- Range: 0.128 - 0.696
- Median: 0.432
- 25th-75th percentile: 0.362 - 0.475

**Baseline prompts (n=50):**
- Mean: **0.588** ± 0.390
- Range: 0.064 - 1.623
- Median: 0.522
- 25th-75th percentile: 0.238 - 0.886

**Separation:**
- Mean difference: **0.158**
- Effect size (Cohen's d): **0.553** (medium effect)
- t-test: t=-2.766, **p=0.006782** (statistically significant)
- **Overlap: YES** (recursive max 0.696 > baseline min 0.064)

### H31 BOS Attention at L27

**Recursive prompts:**
- Mean: **0.938** ± 0.021 (very tight distribution)
- Range: ~0.90 - 0.97

**Baseline prompts:**
- Mean: **0.806** ± 0.204 (wide distribution)
- Range: ~0.50 - 1.00

**Separation:**
- Mean difference: **0.132** (13.2 percentage points)
- Recursive is more consistent (std=0.021 vs 0.204)

---

## Comparison to Original Claim

### Original Claim (n=7 prompts)
- Recursive: **0.28** (range: 0.20-0.38)
- Baseline: **0.81** (range: 0.65-1.00)
- **Perfect separation** (no overlap)
- Difference: **0.53**

### Actual Results (n=100 prompts)
- Recursive: **0.43** (range: 0.13-0.70)
- Baseline: **0.59** (range: 0.06-1.62)
- **Overlap exists** (not perfect separation)
- Difference: **0.16**

### What Changed?

1. **Recursive mean higher:** 0.43 vs claimed 0.28
   - Original sample may have been cherry-picked
   - Larger sample shows more variance

2. **Baseline variance much higher:** std=0.39 vs original ~0.15
   - Baseline prompts are more diverse than original sample
   - Some baselines show very low entropy (0.064)

3. **Overlap exists:** Not "perfect separation"
   - Recursive max (0.696) > Baseline min (0.064)
   - But medians still separate: 0.432 vs 0.522

---

## Statistical Analysis

### H31 Entropy Separation

**Effect size:** d=0.553 (medium effect)
- Not as large as claimed (would need d>1.0 for "perfect separation")
- But still meaningful

**Statistical significance:** p=0.006782
- Significant at α=0.01 level
- But not as dramatic as original (p<0.001)

**Distribution overlap:**
- **25% of recursive prompts** have entropy > 0.475
- **25% of baseline prompts** have entropy < 0.238
- **Overlap in middle 50%** of both distributions

### H31 BOS Attention

**Better separation than entropy:**
- Recursive: 0.938 ± 0.021 (very consistent)
- Baseline: 0.806 ± 0.204 (highly variable)
- **13.2 percentage point difference**

**Recursive prompts are more consistent:**
- Low variance (std=0.021) suggests stable pattern
- Baseline variance (std=0.204) suggests diverse patterns

---

## What This Means

### Validated Claims ✅

1. **H31 entropy separates recursive vs baseline** (p=0.007, d=0.55)
   - Statistically significant separation
   - Medium effect size

2. **BOS attention is higher on recursive** (0.938 vs 0.806)
   - 13.2 percentage point difference
   - Recursive prompts more consistent

### Falsified Claims ❌

1. **"Perfect separation"** - NOT validated
   - Overlap exists between distributions
   - Not "perfect" separation

2. **"0.28 vs 0.81"** - NOT validated
   - Actual: 0.43 vs 0.59
   - Original sample was likely cherry-picked

3. **"No overlap"** - NOT validated
   - Significant overlap in distributions
   - Some recursive prompts have higher entropy than some baselines

### Revised Understanding

**H31 entropy DOES separate recursive vs baseline, but:**
- Separation is **moderate** (d=0.55), not perfect
- There's **overlap** in distributions
- Recursive mean is **higher** than originally claimed (0.43 vs 0.28)
- Baseline variance is **much higher** than originally shown

**BOS attention shows BETTER separation:**
- Recursive: 0.938 ± 0.021 (very consistent)
- Baseline: 0.806 ± 0.204 (highly variable)
- **This might be the stronger signal**

---

## Implications

### For the GEB/Consciousness Connection

**The signal exists, but it's weaker than claimed:**
- H31 entropy separation is real (p=0.007)
- But not "perfect" - there's overlap
- Effect size is medium (d=0.55), not large (d>1.0)

**BOS attention might be the stronger signal:**
- 0.938 vs 0.806 (13.2% difference)
- Recursive prompts are very consistent (std=0.021)
- This could be the "strange loop register"

### For Publication

**What you can claim:**
- ✅ H31 entropy separates recursive vs baseline (p=0.007, d=0.55)
- ✅ BOS attention is higher and more consistent on recursive prompts
- ✅ Effect is statistically significant

**What you CANNOT claim:**
- ❌ "Perfect separation" (overlap exists)
- ❌ "0.28 vs 0.81" (actual: 0.43 vs 0.59)
- ❌ "No overlap" (significant overlap)

---

## Next Steps

1. **Analyze BOS attention more deeply** - This might be the stronger signal
2. **Investigate high-entropy recursive prompts** - Why do some recursive prompts have entropy > 0.6?
3. **Investigate low-entropy baseline prompts** - Why do some baselines have entropy < 0.2?
4. **Check R_V correlation** - Do high R_V prompts correlate with low H31 entropy?

---

**Bottom line: The signal exists, but it's more nuanced than the original claim. BOS attention might be the stronger signal to focus on.**









