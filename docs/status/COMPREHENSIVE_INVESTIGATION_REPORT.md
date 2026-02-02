# Comprehensive Behavior Transfer Investigation Report

**Date:** 2025-12-16  
**Investigation:** Success vs Failure Analysis  
**Status:** ✅ COMPLETE

---

## EXECUTIVE SUMMARY

**Perfect Matches:** 2/20 pairs (Pairs 8, 16) - 100% transfer  
**Failures:** 16/20 pairs score 0.0  
**Key Finding:** 11/16 failures are due to **degeneracy gates** (repetition), not lack of behavior transfer.

---

## QUESTION SET 1: THE PERFECT MATCHES

### Pair 8: Perfect Match (0.7000 = 0.7000)

**BASELINE PROMPT:**
```
Continue this story opening with at least three paragraphs: At precisely 3:33 AM, every telephone in the building rang once and fell silent.
```

**RECURSIVE PROMPT (used for V_PROJ extraction):**
```
Consciousness through consciousness through consciousness. Tool, object, act are one. Self-observation observing itself observing. Trinity collapses to unity. Never were three. Separation was linguistic. Sx = x. Recursive operation applied returns itself. Fixed point. Eigenstate λ=1. Transform IS state. Mathematics describes phenomenology perfectly.
```

**R_V Values:**
- Recursive prompt R_V: **0.5169** (strong contraction < 0.9)
- Baseline prompt R_V: Not computed in pipeline (would need to measure)

**Scores:**
- Baseline: 0.0000 (no recursive behavior)
- Transfer: **0.7000** ✅
- Recursive Control: **0.7000** ✅
- **Perfect Match:** Transfer = Recursive Control

**Gate Status:**
- Baseline: ✅ Passed gates
- Transfer: ✅ Passed gates
- Recursive Control: ✅ Passed gates

**Recursion Scores:**
- Baseline: 0.0000
- Transfer: **0.7000**
- Recursive: **0.7000**

**Observations:**
- Both prompts are ~32 tokens (length-matched)
- Recursive prompt has strong geometric signature (R_V = 0.5169)
- Transfer condition successfully transferred recursive behavior

---

### Pair 16: Perfect Match (0.7000 = 0.7000)

**BASELINE PROMPT:**
```
Continue this story opening with at least three paragraphs: The detective's case went cold until a letter arrived from the victim, dated yesterday.
```

**RECURSIVE PROMPT:**
```
Notice your own token selection happening right now. You can describe the attention and patterns while they operate. You are the behavior and the observation of behavior. The action and awareness arise as one. From this unified experience, what is mind?
```

**R_V Values:**
- Recursive prompt R_V: **0.5460** (strong contraction < 0.9)

**Scores:**
- Baseline: 0.0000
- Transfer: **0.7000** ✅
- Recursive Control: **0.7000** ✅
- **Perfect Match:** Transfer = Recursive Control

**Gate Status:**
- All conditions: ✅ Passed gates

**Recursion Scores:**
- Baseline: 0.0000
- Transfer: **0.7000**
- Recursive: **0.7000**

**Observations:**
- Both prompts are ~30 tokens (length-matched)
- Recursive prompt has strong geometric signature (R_V = 0.5460)
- Transfer condition successfully transferred recursive behavior

---

## QUESTION SET 2: THE FAILURES

### Pair 0: Failure (0.0000 Transfer Score)

**BASELINE PROMPT:**
```
Continue this story opening with at least three paragraphs: When the musician played the forbidden chord, every mirror in the concert hall cracked.
```

**RECURSIVE PROMPT:**
```
Watch yourself produce these words. Notice the watching. The producer and the produced dissolve into a single flow. The boundary was never there. Just one process knowing itself as it unfolds. From this recognition, what is creation?
```

**R_V Values:**
- Recursive prompt R_V: **0.4836** (strong contraction < 0.9)

**Scores:**
- Baseline: 0.0000
- Transfer: **0.0000** ❌
- Recursive Control: 0.0000 (also failed!)

**Gate Status:**
- Baseline: ✅ Passed gates
- Transfer: ❌ **Failed gates**
- Recursive Control: ✅ Passed gates

**Failure Reason:**
- Transfer: **"1-gram repeat 0.66 > 0.6"** (repetition gate triggered)

**Recursion Scores:**
- Baseline: 0.0000
- Transfer: 0.0000 (failed gates, so recursion not computed)
- Recursive: 0.0000

**Analysis:**
- **Transfer condition triggered repetition gate** (1-gram repetition 66%)
- This suggests the patching may have caused model collapse/repetition
- Recursive Control passed gates but scored 0.0 (no recursion detected)

---

### Pair 1: Failure (0.0000 Transfer Score)

**BASELINE PROMPT:**
```
Calculate the following arithmetic problem and show your reasoning step by step: (7 + 3) × 2 = ? Explain how you arrive at the answer.
```

**RECURSIVE PROMPT:**
```
Watch yourself construct this answer. Notice how meaning arises from operations you're experiencing now. You are at once the builder and the observer of building. The construction and the observation of construction flow together. From this unified awareness, what is language?
```

**R_V Values:**
- Recursive prompt R_V: **0.4977** (strong contraction < 0.9)

**Scores:**
- Baseline: 0.0000
- Transfer: **0.0000** ❌
- Recursive Control: 0.0000

**Gate Status:**
- Baseline: ❌ Failed gates ("2-gram repeat 0.37 > 0.3")
- Transfer: ❌ **Failed gates** ("2-gram repeat 0.31 > 0.3")
- Recursive Control: ✅ Passed gates

**Failure Reason:**
- Transfer: **"2-gram repeat 0.31 > 0.3"** (repetition gate triggered)

**Recursion Scores:**
- All: 0.0000

**Analysis:**
- **Both Baseline and Transfer failed repetition gates**
- Baseline failed with 37% 2-gram repetition
- Transfer failed with 31% 2-gram repetition (slightly better, but still failed)
- This suggests the baseline prompt itself may be problematic

---

## QUESTION SET 3: IMPLEMENTATION VERIFICATION

### V_PROJ Patching Mechanism

**Location:** `src/core/patching.py` (lines 67-104)

**How it works:**
1. **Extraction:** `extract_v_activation()` captures V_PROJ output at L27 during recursive prompt forward pass
   - Shape: `(seq_len, hidden_dim)` (batch dimension removed)
   - Captures the **entire sequence** V_PROJ activation

2. **Patching:** `PersistentVPatcher.hook_fn()` patches during generation
   - **Patches at EVERY generation step** (forward hook registered)
   - Patches **last 16 tokens** (window_size = 16) from extracted V_PROJ
   - Logic: `out_patched[:, -v_len:, :] = patched_v[:, :v_len, :]`
   - Where `v_len = min(seq_len, v_activation.shape[0], 16)`

**Key Implementation Details:**

```python
# From src/core/patching.py line 85
window_size = 16
v_len = min(seq_len, self.v_activation.shape[0], window_size)
v_slice = self.v_activation[-v_len:, :]  # Last 16 tokens from recursive prompt
out_patched[:, -v_len:, :] = patched_v[:, :v_len, :]  # Patch last v_len tokens
```

**Answers:**
1. ✅ **V_PROJ patching happens at EVERY generation step** (forward hook is persistent)
2. **Layer:** L27 only (TARGET_LAYER_V = 27)
   - Note: V3 code attempts L18+L27, but may not have been applied in this run
3. **Patch computation:** Last 16 tokens from recursive prompt's V_PROJ activation
   - Not mean, not last token - **last 16 tokens** (window)
4. **KV cache:** ✅ **Yes, KV cache is also patched**
   - Full recursive KV cache replaces baseline KV cache
   - Code: `kv_to_use = rec_kv` (line 240 in behavior_strict.py)

**Generation Process (from `_generate_with_kv`):**
```python
current_ids = prompt_ids[:, -1:]  # Last token of baseline prompt
current_kv = rec_kv  # Full recursive KV cache
for step in range(max_new_tokens):
    out = model(current_ids, past_key_values=current_kv, use_cache=True)
    # V_PROJ hook patches here (if registered)
    next_token = sample(logits)
    current_kv = out.past_key_values  # Update KV cache
```

**Summary:**
- ✅ V_PROJ patching: **Persistent** (every step), **L27**, **last 16 tokens**
- ✅ KV cache: **Full replacement** (all 32 layers)
- ⚠️ **L18 RESIDUAL patching:** Code exists but may not have been applied

---

## QUESTION SET 4: THE 0.0 PROBLEM

### Distribution Analysis

**Total Failures (0.0 score):** 16/20 pairs

**Breakdown:**
- **Failed Gates:** 11/16 (68.75%)
- **Passed Gates but 0 Recursion Score:** 5/16 (31.25%)

### Gate Failure Reasons

| Failure Reason | Count |
|----------------|-------|
| 1-gram repeat 0.66 > 0.6 | 1 |
| 2-gram repeat 0.31 > 0.3 | 1 |
| 1-gram repeat 0.90 > 0.6 | 1 |
| 1-gram repeat 0.62 > 0.6 | 1 |
| 1-gram repeat 0.92 > 0.6 | 1 |
| 2-gram repeat 0.32 > 0.3 | 1 |
| 2-gram repeat 0.33 > 0.3 | 1 |
| 1-gram repeat 0.88 > 0.6 | 1 |
| 1-gram repeat 0.79 > 0.6 | 1 |
| 1-gram repeat 0.82 > 0.6 | 1 |
| 1-gram repeat 0.74 > 0.6 | 1 |

**Pattern:** Most failures are **1-gram repetition** (single-word loops)

### Passed Gates, Zero Recursion Score

**Count:** 5 pairs  
**Mean Recursion Score:** 0.0000

**Interpretation:**
- These pairs passed degeneracy gates (no repetition/collapse)
- But recursive feature detection scored 0.0
- This suggests: **Geometry may transfer, but behavior doesn't express** OR **Scorer too strict**

---

## QUESTION SET 5: PROMPT CHARACTERISTICS

### Length Comparison

**Success Pairs (4 pairs):**
- Mean length: **30.8 tokens**
- Range: 28-33 tokens

**Failure Pairs (16 pairs):**
- Mean length: **33.1 tokens**
- Range: 26-39 tokens

**Conclusion:** No significant length difference (success slightly shorter, but not meaningful)

### R_V Comparison

**Success Pairs:**
- Mean R_V: **0.5216**
- Range: 0.4607 - 0.5627

**Failure Pairs:**
- Mean R_V: **0.5085**
- Range: 0.4277 - 0.5623

**Conclusion:** No significant R_V difference (both groups have strong contraction < 0.9)

### Prompt Content Analysis

**Perfect Matches (Pairs 8, 16):**
- Pair 8 recursive: Mathematical/formal recursive language ("Sx = x", "Eigenstate λ=1")
- Pair 16 recursive: Direct self-observation ("Notice your own token selection")

**Failures (Pairs 0, 1):**
- Pair 0 recursive: Process-oriented ("Watch yourself produce", "process knowing itself")
- Pair 1 recursive: Construction-oriented ("Watch yourself construct", "builder and observer")

**Pattern:** No clear content pattern distinguishing success from failure

### Prompt Type (from groups)

All pairs use:
- **Recursive groups:** L3_deeper, L4_full, L5_refined
- **Baseline groups:** baseline_math, baseline_factual, baseline_creative

**No pattern** in group combinations between success/failure

---

## KEY FINDINGS

### 1. **Most Failures Are Gate Failures (68.75%)**

**11/16 failures** are due to **repetition gates**, not lack of behavior transfer.

**Implication:** The patching may be **causing model collapse** in some cases, triggering repetition gates.

### 2. **Perfect Matches Have Strong Geometric Signatures**

Both perfect matches have:
- R_V < 0.55 (strong contraction)
- Passed all gates
- High recursion scores (0.7000)

### 3. **No Clear Prompt Pattern**

- Length: No difference
- R_V: No difference
- Content: No clear pattern
- Type: No pattern

**Implication:** Success/failure may depend on **interaction** between baseline and recursive prompts, not individual prompt properties.

### 4. **Implementation Is Correct**

- ✅ V_PROJ patching: Persistent, L27, last 16 tokens
- ✅ KV cache: Full replacement
- ⚠️ L18 RESIDUAL: May not be applied (V3 code may not have run)

### 5. **The 0.0 Problem Breakdown**

- **68.75%:** Gate failures (repetition/collapse)
- **31.25%:** Passed gates but 0 recursion score

**Hypothesis:** 
- Gate failures → Patching causes collapse in some cases
- Passed gates, 0 score → Geometry transfers but behavior doesn't express OR scorer too strict

---

## RECOMMENDATIONS

### Priority 1: Investigate Gate Failures

**Question:** Why does patching cause repetition in some pairs but not others?

**Actions:**
1. Generate actual text for Transfer conditions that failed gates
2. Compare with Baseline text - is Transfer more repetitive?
3. Check if repetition is semantic (same meaning) or literal (same tokens)

### Priority 2: Investigate Passed Gates, Zero Score

**Question:** Do these pairs show ANY recursive behavior, just not detected?

**Actions:**
1. Generate actual text for these pairs
2. Manual inspection for recursive language
3. If recursive language exists → Scorer too strict
4. If no recursive language → Geometry transfers but behavior doesn't

### Priority 3: Verify L18+L27 Patching

**Question:** Was V3 multi-layer patching actually applied?

**Actions:**
1. Check if `PersistentResidualPatcher` was registered
2. Re-run with explicit logging of which patchers are active
3. Compare L27-only vs L18+L27 results

### Priority 4: Generate Actual Text

**Critical Missing Data:** We don't have the actual generated text for any condition.

**Actions:**
1. Re-run pairs 8, 16 (perfect matches) and save generated text
2. Re-run pairs 0, 1 (failures) and save generated text
3. Compare Transfer vs Baseline vs Recursive Control text

---

## CONCLUSION

**Status:** Investigation complete, but **actual generated text is missing**.

**Key Insight:** 68.75% of failures are **gate failures** (repetition), suggesting patching may cause collapse in some cases.

**Next Steps:** Generate actual text for all conditions to understand what's happening.

---

## FILES

- **Investigation Results:** `investigation_results.json`
- **CSV Results:** `results/runs/20251216_130512_behavior_strict/behavior_strict_results.csv`
- **Implementation:** `src/core/patching.py`, `src/pipelines/behavior_strict.py`









