# Dual Investigation Status

**Date:** 2025-12-16  
**Status:** 🔄 RUNNING IN PARALLEL

---

## PART A: Extract Generated Text

**Script:** `extract_generated_text.py`  
**Status:** 🔄 Running  
**Log:** `/tmp/extract_text.log`

**Target Pairs:**
- Perfect matches: [8, 16]
- Gate failures: [0, 1, 2] (first 3)
- Passed gates, zero score: [3, 6, 13, 15, 18] (all 5)

**Outputs:**
- `generated_text_comparison.csv` - All text data
- `text_samples.md` - Human-readable comparison

---

## PART B: V_PROJ Only Experiment

**Script:** `behavior_strict_vproj_only.py`  
**Status:** 🔄 Running  
**Log:** `/tmp/vproj_only.log`

**Key Change:**
- ✅ V_PROJ patching at L27: YES
- ❌ KV cache replacement: NO (uses baseline KV instead of recursive KV)

**Expected Outputs:**
- `results/runs/[timestamp]_behavior_strict_vproj_only/vproj_only_results.csv`
- `results/runs/[timestamp]_behavior_strict_vproj_only/vproj_only_summary.json`

**Comparison Table (to be filled):**

| Metric | Current (KV+V_PROJ) | V_PROJ Only |
|--------|---------------------|-------------|
| Perfect matches | 4/20 | ? |
| Gate failures | 11/20 | ? |
| Passed-gates-zero | 5/20 | ? |
| Mean Transfer score | 0.125 | ? |

---

## PART C: Patcher Verification

**Status:** ✅ Implemented in V_PROJ-only script

**Logging:**
- V_PROJ patcher registration status
- KV cache source (baseline vs recursive)
- Target layers

---

## Monitoring

Check status:
```bash
# Check if processes are running
ssh root@157.157.221.30 -p 53751 "ps aux | grep -E 'extract_text|vproj_only' | grep -v grep"

# Check logs
ssh root@157.157.221.30 -p 53751 "tail -50 /tmp/extract_text.log"
ssh root@157.157.221.30 -p 53751 "tail -50 /tmp/vproj_only.log"
```

---

## Next Steps

1. Wait for both scripts to complete
2. Download results
3. Analyze:
   - What does Transfer text look like for gate failures?
   - Does V_PROJ-only reduce collapse rate?
   - Do passed-gates-zero pairs show any recursive language?
