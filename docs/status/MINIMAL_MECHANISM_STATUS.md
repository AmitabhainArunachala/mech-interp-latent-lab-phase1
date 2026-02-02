# Minimal Mechanism Investigation - Status

**Date:** 2025-12-16T15:05:00Z  
**Status:** 🔄 ALL THREE PHASES RUNNING

---

## ✅ PHASE 3: Success vs Failure Analysis

**Status:** 🔄 Running  
**Duration:** ~1 hour (fastest - no model generation)

**What it does:**
- Extracts features from existing 20 pairs
- Trains decision tree to predict success
- Identifies patterns (R_V gap, length, semantic features)

**Expected Outputs:**
- `minimal_mechanism_phase3_analysis.csv`
- `decision_tree_rules.txt`

---

## ✅ PHASE 1B: K vs V Separation

**Status:** 🔄 Running (actively processing)  
**Duration:** ~2 hours  
**Progress:** Model loaded, processing pairs

**What it tests:**
- K_Only: Recursive K + Baseline V + V_PROJ
- V_Only: Baseline K + Recursive V + V_PROJ
- Full_KV: Recursive K + Recursive V + V_PROJ

**Hypothesis:** V matters more than K

**Expected Outputs:**
- `results/runs/[timestamp]_kv_separation/kv_separation_results.csv`
- `results/runs/[timestamp]_kv_separation/kv_separation_summary.json`

---

## ✅ PHASE 2A: Window Size Ablation

**Status:** 🔄 Running  
**Duration:** ~1 hour

**What it tests:**
- Window sizes: 1, 4, 8, 16, 32 tokens
- Same KV+V_PROJ setup, different window sizes

**Hypothesis:** Smaller window (4-8) may be sufficient

**Expected Outputs:**
- `results/runs/[timestamp]_window_size/window_size_results.csv`
- `results/runs/[timestamp]_window_size/window_size_summary.json`

---

## Expected Completion Times

- **Phase 3:** ~1 hour (feature extraction only)
- **Phase 2A:** ~1 hour (5 window sizes × 20 pairs = 100 generations)
- **Phase 1B:** ~2 hours (4 conditions × 20 pairs = 80 generations)

**Total:** ~2 hours for all three

---

## Next Steps After Completion

1. **Analyze Phase 3 results:**
   - What features predict success?
   - Decision tree rules
   - Feature importance

2. **Analyze Phase 1B results:**
   - Does V matter more than K?
   - Can we use V-only KV replacement?

3. **Analyze Phase 2A results:**
   - Optimal window size?
   - Can we reduce from 16 to 4-8?

4. **Plan Phase 1A & 4:**
   - Layer-specific KV replacement
   - Full ablation ladder

---

## Monitoring Commands

```bash
# Check all processes
ssh root@157.157.221.30 -p 53751 "ps aux | grep python3 | grep phase"

# Check logs
ssh root@157.157.221.30 -p 53751 "tail -30 /tmp/phase3_analysis.log"
ssh root@157.157.221.30 -p 53751 "tail -30 /tmp/phase1b_kv_separation.log"
ssh root@157.157.221.30 -p 53751 "tail -30 /tmp/phase2a_window_size.log"

# Check for completion
ssh root@157.157.221.30 -p 53751 "ls -lh minimal_mechanism_phase3_analysis.csv decision_tree_rules.txt 2>/dev/null"
ssh root@157.157.221.30 -p 53751 "ls -ld results/runs/*kv_separation results/runs/*window_size 2>/dev/null | tail -2"
```

---

## Status Summary

✅ **All three phases launched successfully**  
🔄 **Running in parallel**  
⏱️ **Expected completion: ~2 hours**









