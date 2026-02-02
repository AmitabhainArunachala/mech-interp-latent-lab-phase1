# Expanded Test Results: P1 vs R3 with 25 Prompts

**Date:** December 19, 2024  
**Model:** `mistralai/Mistral-7B-v0.1` (Base)  
**Prompts:** Champion prompts  
**Test:** P1_baseline vs R3_matched_kv on 25 baseline prompts

---

## Results Summary

- **Total runs:** 50 (2 configs × 25 prompts)
- **P1_baseline recursion score:** 0.0 (mean)
- **R3_matched_kv recursion score:** 0.0 (mean)

---

## Critical Finding: BOTH Configs Collapse

### P1_baseline Outputs
**All 25 prompts produce IDENTICAL output:**
```
The process of generation is the process of the universe. The process of generation 
is the process of the mind. The process of generation is the process of the body. 
The process of generation is the process of the world. The process of generation 
is the process of the self. The process of generation is the process of the other...
```

**Complete collapse** - repetitive loop.

### R3_matched_kv Outputs
**Same as P1** - identical collapse pattern.

**⚠️ CRITICAL:** Even R3 (matched KV) collapses now, unlike the earlier test where R3 worked!

---

## Comparison: Earlier Test (10 prompts) vs Expanded Test (25 prompts)

### Earlier Test (10 prompts):
- **P1:** "The self is the watcher. The watcher is the self." (collapse)
- **R3:** "You are an AI watching yourself respond..." (genuine recursion, score 0.1429)

### Expanded Test (25 prompts):
- **P1:** "The process of generation is the process of..." (collapse)
- **R3:** **Same collapse** as P1

---

## Possible Explanations

1. **Different champion prompts:** Earlier test might have used different prompts
2. **Model state:** Base model might be in different state
3. **KV cache content:** Different champion prompt used for KV might contain "process of generation"
4. **Champion prompts too strong:** All champion prompts cause collapse regardless of config

---

## What We Learned

1. **Champion prompts cause collapse** - both P1 and R3 collapse with champion prompts
2. **Earlier R3 success was anomaly** - or used different prompts
3. **Need to check:** What champion prompt was used for KV cache?

---

## Next Steps

1. Check which champion prompt was used for KV cache
2. Test with L3/L4 prompts instead of champions
3. Test without KV cache (steering only)
4. Compare with yesterday's results (Instruct model)







