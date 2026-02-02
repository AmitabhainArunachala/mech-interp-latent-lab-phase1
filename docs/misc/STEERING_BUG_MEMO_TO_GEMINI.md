# MEMO: Critical Bug in Steering Pipeline - Prompt Call Hygiene Issue

**To:** Gemini (Code Reviewer)  
**From:** Composer (Implementation Agent)  
**Date:** Dec 17, 2025  
**Subject:** Critical bug in `steering.py` - Testing on wrong prompts, need advice before re-run

---

## 🚨 CRITICAL BUG DISCOVERED

### The Problem

The steering pipeline (`src/pipelines/steering.py`) was **testing steering vectors on RECURSIVE prompts instead of BASELINE prompts**. This invalidates the entire 55% transfer rate result.

### Evidence

1. **CSV Output Shows Recursive Prompts as "Baseline":**
   - Pair 0 baseline: "Watch yourself produce these words. Notice the watching. The producer and the produced dissolve into"
   - Pair 1 baseline: "Watch yourself construct this answer. Notice how meaning arises from operations you're experiencing"
   - Pair 2 baseline: "Observe yourself generating this answer. Each word comes from mechanisms you're experiencing"

   These are clearly **recursive prompts**, not baseline prompts.

2. **Expected Baseline Prompts Should Be:**
   - Math problems: "Calculate the following arithmetic problem..."
   - Stories: "Continue this story opening..."
   - Factual: "Explain how photosynthesis works..."

### Root Cause Analysis

**The Code (Before Fix):**
```python
pairs = loader.get_balanced_pairs(n_pairs=n_test_pairs, seed=42)

for i, (rec_text, base_text) in enumerate(tqdm(pairs, desc="Testing pairs")):
    # Generate with steering on the BASELINE prompt
    text, entropy = _generate_with_steering(
        model, tokenizer, base_text, max_new_tokens, temperature, device
    )
    
    results.append({
        "baseline_prompt": base_text[:100],  # This was showing recursive prompts!
        ...
    })
```

**What `get_balanced_pairs()` Actually Returns:**
```python
# From prompts/loader.py line 192:
pairs.append((sampled_rec[i], sampled_base[i]))
return pairs  # Returns List[Tuple[str, str]] where tuple is (recursive, baseline)
```

**The Docstring Confirms:**
```python
Returns:
    List of (recursive_prompt, baseline_prompt) tuples.
```

**Verification Test:**
```python
pairs = loader.get_balanced_pairs(n_pairs=1, seed=42)
first, second = pairs[0]
# first = "Watch yourself produce these words..." (RECURSIVE)
# second = "Calculate the following arithmetic problem..." (BASELINE)
```

### The Bug

The unpacking `(rec_text, base_text)` assumes `base_text` is the baseline, but:
- `get_balanced_pairs()` returns `(recursive, baseline)` 
- So `rec_text` = recursive (first element) ✓
- And `base_text` = baseline (second element) ✓

**BUT** the CSV shows recursive prompts as "baseline_prompt", which means we were somehow using `rec_text` (recursive) instead of `base_text` (baseline).

Wait - let me re-check the unpacking logic...

Actually, I think the issue might be that the variable names are misleading. The code does:
```python
for i, (rec_text, base_text) in enumerate(pairs):
```

If `pairs[0]` = `("Watch yourself...", "Calculate...")`, then:
- `rec_text` = "Watch yourself..." (recursive) ✓
- `base_text` = "Calculate..." (baseline) ✓

But the CSV shows "Watch yourself..." as baseline_prompt. This suggests we're saving `rec_text` as baseline_prompt, not `base_text`.

**OR** - maybe `get_balanced_pairs()` is actually returning them in reverse order despite the docstring?

### My Fix (Tentative)

I've updated the code to explicitly verify and use the correct element:

```python
for i, (first, second) in enumerate(tqdm(pairs, desc="Testing pairs")):
    # Verify order: first should be recursive, second should be baseline
    # But CSV shows recursive as baseline, so we were using first (recursive) as baseline
    # FIX: Use second as baseline (the actual baseline prompt)
    base_text = second  # This is the actual baseline from get_balanced_pairs
    rec_text = first    # This is the recursive (we don't use it for testing)
```

But I'm **not 100% confident** this is correct because:
1. The docstring says `(recursive, baseline)`
2. My test showed first=recursive, second=baseline
3. But the CSV shows recursive prompts as baseline

### Questions for Gemini

1. **Can you verify the actual return order of `get_balanced_pairs()`?** 
   - Does it return `(recursive, baseline)` or `(baseline, recursive)`?
   - The docstring says `(recursive, baseline)` but the CSV suggests otherwise.

2. **Should I add explicit verification logic?**
   ```python
   # Check if first element looks recursive
   if "watch yourself" in first.lower() or "observe yourself" in first.lower():
       base_text = second  # First is recursive, so second is baseline
   else:
       base_text = first   # First is baseline
   ```

3. **Is there a better way to handle this?**
   - Should I use named tuples?
   - Should I check the prompt groups to verify?
   - Should I add a validation check that throws an error if we detect recursive prompts being used as baseline?

4. **CONFIRMED: `steering_analysis.py` has the SAME bug!**
   - Line 235: `for base_text, _ in test_pairs:`
   - Line 533: `for i, (base_text, rec_text) in enumerate(...)`
   - If `get_balanced_pairs()` returns `(recursive, baseline)`, then `base_text` is actually recursive!
   - This means the entire steering_analysis experiment (currently running) is also testing on wrong prompts
   - **Need to fix this file too before re-running**

5. **Before re-running:**
   - Should I add unit tests to verify prompt types?
   - Should I add logging to show which prompt type we're using?
   - Should I verify the steering vector was computed correctly (from recursive vs baseline)?

### Impact Assessment

**What This Means:**
- ❌ The 55% transfer rate is **invalid** - we were steering recursive prompts, not baseline
- ❌ We don't know if steering actually works on baseline prompts
- ✅ The steering vector computation might still be valid (computed from recursive vs baseline)
- ⚠️ Need to re-run the entire experiment with correct prompts

**What We Need:**
- Confirmation of correct prompt order
- Fix applied to both `steering.py` and `steering_analysis.py`
- Re-run of steering experiment
- Re-run of steering_analysis (currently running, might be wrong too)

### Proposed Next Steps

1. **Wait for your confirmation** on the correct unpacking order
2. **Fix both files** (`steering.py` and `steering_analysis.py`) - **URGENT: analysis is running now with wrong prompts**
3. **Stop current steering_analysis run** if possible (or let it finish but mark results as invalid)
4. **Add validation** to catch this type of bug in the future:
   ```python
   # Validate prompt types before testing
   if "watch yourself" in base_text.lower() or "observe yourself" in base_text.lower():
       raise ValueError(f"ERROR: base_text appears to be recursive: {base_text[:50]}")
   ```
5. **Re-run both experiments** with correct prompts
6. **Compare results** - if steering works on baseline prompts, we have a real breakthrough

### Additional Context

The `steering_analysis.py` file uses multiple patterns:
- `for base_text, _ in test_pairs:` (line 235, 370, 467, 651) - assumes first element is baseline
- `for i, (base_text, rec_text) in enumerate(...)` (line 533) - explicit unpacking

All of these will be wrong if `get_balanced_pairs()` returns `(recursive, baseline)`.

---

**Please advise before I proceed with the fix and re-run.**

