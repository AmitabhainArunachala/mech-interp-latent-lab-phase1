# Dual Investigation: Launched Successfully

**Date:** 2025-12-16T13:52:00Z  
**Status:** ✅ BOTH RUNNING IN PARALLEL

---

## ✅ PART A: Extract Generated Text

**Script:** `extract_generated_text.py`  
**Status:** 🔄 Running (model loading, processing pairs)  
**Log:** `/tmp/extract_text.log`

**Target Pairs:** 10 pairs total
- Perfect matches: [8, 16]
- Gate failures: [0, 1, 2]
- Passed gates, zero score: [3, 6, 13, 15, 18]

**Expected Outputs:**
- `generated_text_comparison.csv` - Full data
- `text_samples.md` - Human-readable comparison

**Progress:** Model loaded, starting pair processing

---

## ✅ PART B: V_PROJ Only Experiment

**Script:** `behavior_strict_vproj_only.py`  
**Status:** 🔄 Running  
**Log:** `/tmp/vproj_only.log`

**Key Difference:**
- ✅ V_PROJ patching at L27: YES
- ❌ KV cache replacement: NO (uses baseline KV)

**Expected Outputs:**
- `results/runs/[timestamp]_behavior_strict_vproj_only/vproj_only_results.csv`
- `results/runs/[timestamp]_behavior_strict_vproj_only/vproj_only_summary.json`

**Progress:** Starting...

---

## Monitoring Commands

```bash
# Check process status
ssh root@157.157.221.30 -p 53751 "ps aux | grep -E 'extract|vproj' | grep python3"

# Check logs
ssh root@157.157.221.30 -p 53751 "tail -50 /tmp/extract_text.log"
ssh root@157.157.221.30 -p 53751 "tail -50 /tmp/vproj_only.log"

# Check for completion
ssh root@157.157.221.30 -p 53751 "ls -lh generated_text_comparison.csv text_samples.md 2>/dev/null"
ssh root@157.157.221.30 -p 53751 "ls -ld results/runs/*vproj_only 2>/dev/null | tail -1"
```

---

## Expected Completion Times

- **Part A (extract text):** ~10-15 minutes (10 pairs × 3 conditions = 30 generations)
- **Part B (vproj only):** ~30-40 minutes (20 pairs × 3 conditions = 60 generations)

---

## Next Steps After Completion

1. Download results from RunPod
2. Analyze generated text:
   - What does Transfer text look like for gate failures?
   - Do passed-gates-zero pairs show recursive language?
   - Compare perfect matches vs failures
3. Compare V_PROJ-only vs KV+V_PROJ:
   - Does removing KV reduce collapse?
   - Does transfer still work?
4. Answer analysis questions from user request

---

## Files Created

- ✅ `extract_generated_text.py` - Text extraction script
- ✅ `behavior_strict_vproj_only.py` - V_PROJ-only experiment (already existed, updated)
- ✅ `DUAL_INVESTIGATION_STATUS.md` - Status tracking
- ✅ `DUAL_INVESTIGATION_LAUNCHED.md` - This file









