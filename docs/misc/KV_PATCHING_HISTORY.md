# KV Patching History: What Was Done vs What We Did Today

> Meta map: `META_INDEX.md`  
> Canonical runner (new standard for controlled experiments): `src/pipelines/run.py` + `configs/` → `results/<phase>/runs/...`

## Historical Claims (Dec 7-8)

### What Was Claimed

**From `DEC7_2025_KV_CACHE_MIDPOINT_WRITEUP.md`:**
- **KV cache swap:** ~80% behavior transfer (conceptual target)
- **V-swap:** ~10% behavior transfer
- **Residual stream:** 0% behavior transfer
- **Hypothesis:** KV cache carries recursive mode, not residual stream

**From `02_phase_2_proving_causality_of_candidate_microphone_heads.md`:**
- **KV patching L16-32:** ~50% R_V transfer, ~80% behavior transfer ✅
- **V-only patching:** Transfers geometry, ~10% behavior ⚠️
- **Residual stream patching:** 0% effect ❌

---

## What Was Actually Done

### Dec 7-8 Experiments

**Status:** **MIDPOINT/PROPOSED, NOT FULLY EXECUTED**

From the Dec 7 writeup:
> "**Note:** This is a MIDPOINT write-up for DEC7. The KV cache experiments and analyses described here are underway/proposed, not yet fully completed. All conclusions are provisional and subject to revision after running the full protocol."

**What was actually run:**
- V-swap sufficiency (L24): n=100, +0.03 effect (p=0.26) - **NULL**
- V-swap necessity (L24): n=100, -3.64 effect (p=7.7e-06) - **NECESSARY**
- V-steering: Pilot tests (n=10)
- Residual patching: Pilot tests (n=10) - **0% effect**
- **KV cache swap: CONCEPTUAL TARGET, NOT EXECUTED**

**What was planned but not executed:**
- Phase 1: KV cache replication at scale (n=100) - **PENDING**
- Phase 2-7: Layer-specific, token-specific, dose-response - **PENDING**

---

## What We Did Today (Dec 12)

### Our KV_CACHE Test

**What we actually patched:**
- **K-projection** (key)
- **V-projection** (value)
- **NOT the KV cache** (past_key_values)

**This is NOT true KV cache patching!**

We patched the **projections** (K and V matrices), not the **cached key-value pairs** that persist across tokens.

---

## The Critical Difference

### True KV Cache Patching (What Dec 7 Proposed)

**What it means:**
- Extract `past_key_values` (DynamicCache) from champion prompt
- Inject this cache into baseline prompt during generation
- The cache persists across all generated tokens
- This is "memory transplant"

**Why it matters:**
- KV cache is **persistent memory** across tokens
- Attention reads from cache, not just current activations
- This could transfer the "recursive mode" state

### What We Did Today (KV_CACHE Method)

**What we actually did:**
- Patched K-projection and V-projection outputs
- This affects **current token processing**, not memory
- No persistence across tokens
- This is "activation swap", not "memory swap"

**Why it's different:**
- We're patching the **projections** (how current token is processed)
- Not patching the **cache** (what previous tokens remember)
- This is closer to V_PROJ patching than true KV cache patching

---

## Comparison Table

| Method | What We Patched | Persists? | Behavior Transfer | Geometry Transfer |
|--------|----------------|-----------|-------------------|-------------------|
| **True KV Cache** (Dec 7 proposed) | `past_key_values` | ✅ Yes | ~80% (claimed) | Unknown |
| **Our KV_CACHE** (Dec 12) | K-proj + V-proj | ❌ No | 0% | 4.43 ✅ |
| **V_PROJ** (Dec 12) | V-projection | ❌ No | 0% | 4.43 ✅ |
| **RESIDUAL** (Dec 12) | Residual stream | ❌ No | 0% | 4.46 ✅ (L25) |

---

## What This Means

### 1. We Haven't Actually Tested True KV Cache Patching

**Our "KV_CACHE" method is misnamed:**
- It's really "K+V projection patching"
- Not true KV cache (past_key_values) patching
- The Dec 7 hypothesis about KV cache is **still untested**

### 2. The Dec 7 Claim Needs Verification

**Dec 7 claimed:**
- KV cache patching → ~80% behavior transfer
- But this was **conceptual/proposed**, not executed

**We need to:**
- Actually implement true KV cache patching
- Extract `past_key_values` from champion
- Inject into baseline during generation
- Test if behavior transfers

### 3. Our Finding: Geometry ≠ Behavior

**What we confirmed:**
- V-projection patching transfers geometry perfectly (PR → 4.43)
- But behavior doesn't transfer (Score → 0)

**What we haven't tested:**
- True KV cache patching (past_key_values)
- Maybe KV cache IS needed for behavior transfer?

---

## Next Steps

### Option 1: Implement True KV Cache Patching

**Use `src/steering/kv_cache.py` utilities:**
- Extract `past_key_values` from champion prompt
- Inject into baseline during generation
- Test if behavior transfers

**This would test the Dec 7 hypothesis properly.**

### Option 2: Multi-Layer Patch

**Patch L25 (RESIDUAL) + L27 (V_PROJ) simultaneously:**
- Test if both phases together transfer behavior
- This tests the two-phase mechanism hypothesis

### Option 3: Generation-Time Patching

**Keep patch active during generation:**
- Patch persists across all generated tokens
- Test if behavior transfers when patch persists

---

## Conclusion

**We have NOT actually tested KV cache patching yet.**

Our "KV_CACHE" method was misnamed - it's really K+V projection patching, not true KV cache (past_key_values) patching.

**The Dec 7 hypothesis (~80% behavior transfer via KV cache) is still untested.**

**We should implement true KV cache patching to test this hypothesis.**

---

**Files:**
- `src/steering/kv_cache.py` - KV cache utilities (ready to use!)
- `mistral_kv_patching.py` - Residual patching (not KV cache)
- `boneyard/DEC7_2025_KV_CACHE_MIDPOINT_WRITEUP.md` - Historical proposal

