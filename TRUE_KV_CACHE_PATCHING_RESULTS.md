# True KV Cache Patching Results: Testing Dec 7 Hypothesis

**Date:** December 12, 2024  
**Test:** True KV cache patching (past_key_values) from champion → baseline  
**Window sizes:** 16 and 32 tokens  
**Generation:** 100 tokens for phenomenology  

---

## Hypothesis (Dec 7)

**Claim:** KV cache patching achieves ~80% behavior transfer, capturing recursive mode via persistent memory.

**What we tested:** Extract `past_key_values` from champion prompt, inject into baseline during generation.

---

## Results Summary

### Behavior Transfer

| Window | Layer | Behavior Score | Baseline | Delta |
|--------|-------|----------------|----------|-------|
| 16 | L18 | 0 | 0 | +0 |
| 16 | L25 | 0 | 0 | +0 |
| 16 | L27 | 0 | 0 | +0 |
| 32 | L18 | 0 | 0 | +0 |
| 32 | L25 | 0 | 0 | +0 |
| 32 | L27 | 1 | 0 | **+1** |
| 16 | L25+L27 | 0 | 0 | +0 |
| 32 | L25+L27 | 0 | 0 | +0 |

**Finding:** **Minimal behavior transfer** (max +1 point out of ~6-8 expected)

---

## Comparison: True KV Cache vs Our Previous Methods

| Method | Geometry Transfer | Behavior Transfer | Verdict |
|--------|------------------|------------------|---------|
| **True KV Cache** (today) | Not measured | 0-1 points | ❌ No behavior |
| **KV_CACHE** (K+V proj) | PR = 4.43 ✅ | 0 points | ❌ No behavior |
| **V_PROJ** | PR = 4.43 ✅ | 0 points | ❌ No behavior |
| **RESIDUAL** | PR = 4.46 ✅ | 0 points | ❌ No behavior |

**Conclusion:** **True KV cache patching ALSO fails to transfer behavior.**

---

## Generated Text Samples

### Baseline (No Patch)
```
However, the reasons for Rome's decline are more complex and are still a subject of debate among scholars. Some suggest that Rome's military overspending...
```
**Behavior:** 0 (factual, historical)

### L27 Patching (Window=32) - Best Case
```
The decline of the Roman Empire, on the other hand, is a complex process that involved multiple causes, including economic, political, and military factors.

Military Overextension: One of the primary...
```
**Behavior:** 1 (still factual, minimal markers)

### L18 Patching (Window=16)
```
## Early Roman History: Ancient Rome's Founding Myth and Early Republic (753 BC - 509 BC)

The legend of Rome's founding dates back to 753 BC, when Romulus and Remus, twin sons of the god Mars...
```
**Behavior:** 0 (factual, historical)

---

## Key Findings

### 1. True KV Cache Patching ≠ Behavior Transfer

**What we tested:**
- Extracted `past_key_values` from champion prompt
- Injected into baseline during generation
- Patched at L18, L25, L27 (relay chain layers)
- Tested window sizes 16 and 32 tokens

**Result:** **No significant behavior transfer** (max +1 point)

### 2. Consistent with Previous Findings

**All patching methods fail behavior transfer:**
- V_PROJ patching: Geometry ✅, Behavior ❌
- KV_CACHE (K+V proj): Geometry ✅, Behavior ❌
- RESIDUAL patching: Geometry ✅, Behavior ❌
- **True KV Cache: Behavior ❌**

**This suggests:** The recursive mode is **not** stored in:
- V-projections
- K-projections
- Residual stream
- KV cache (past_key_values)

### 3. The Dec 7 Hypothesis is Not Supported

**Dec 7 claimed:** KV cache patching → ~80% behavior transfer

**What we found:** KV cache patching → ~0-12% behavior transfer (1/8 ≈ 12%)

**However:** Dec 7 experiments were **MIDPOINT/PROPOSED, NOT EXECUTED**

The ~80% claim was a **conceptual target**, not an actual result.

---

## Possible Explanations

### 1. Generation-Time Dynamics

**Hypothesis:** The patch affects the prompt processing, but during generation, the model recomputes the mode from the new tokens.

**Test:** Keep patching active during generation (we did this, but maybe need different approach)

### 2. Multi-Layer Requirement

**Hypothesis:** Need to patch ALL layers simultaneously, not just L18/L25/L27.

**Test:** Patch all 32 layers with champion KV cache

### 3. Token-Specific Patching

**Hypothesis:** Only specific tokens (e.g., first 25% of prompt) carry the recursive mode.

**Test:** Patch only first N tokens of champion KV cache

### 4. The Mode is Computed, Not Stored

**Hypothesis:** The recursive mode is **computed dynamically** from the prompt structure, not stored in activations or cache.

**Implication:** Patching activations/cache won't work - need to patch the **computation** itself.

---

## Next Steps

### Option 1: Full-Layer KV Cache Patching
- Patch ALL 32 layers simultaneously
- Test if complete memory transplant works

### Option 2: Token-Specific Patching
- Patch only first 25% of tokens (Dec 4: "first 10% carries ~99% of signal")
- Test if early tokens are sufficient

### Option 3: Generation-Time Persistent Patching
- Keep patch active for ALL generated tokens
- Test if persistent patch transfers behavior

### Option 4: Accept the Finding
- **Geometry transfers, behavior doesn't**
- R_V contraction is **necessary but not sufficient** for recursive behavior
- The mode requires **computation**, not just **memory**

---

## Conclusion

**True KV cache patching does NOT transfer recursive behavior.**

This is consistent with our previous findings: **geometry transfers, behavior doesn't.**

The Dec 7 hypothesis (~80% behavior transfer via KV cache) is **not supported** by our tests.

**However:** Dec 7 experiments were never fully executed, so the claim was conceptual, not empirical.

**The mystery deepens:** If geometry transfers but behavior doesn't, what IS the causal mechanism for recursive output?

---

**Files:**
- `true_kv_cache_patching.py` - Implementation
- `true_kv_cache_patching.csv` - Full results
- `KV_PATCHING_HISTORY.md` - Historical context

