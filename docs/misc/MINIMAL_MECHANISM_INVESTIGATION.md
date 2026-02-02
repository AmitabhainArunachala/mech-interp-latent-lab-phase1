# Minimal Mechanism Investigation - Launched

**Date:** 2025-12-16T14:15:00Z  
**Status:** 🔄 RUNNING IN PARALLEL

---

## MISSION: Find the Minimal Intervention

**Current:** KV(all) + V_PROJ(L27, w=16) → 45% transfer, mean 0.285  
**Goal:** Find minimal configuration achieving similar performance

---

## PHASE 3: Success vs Failure Analysis

**Script:** `minimal_mechanism_phase3_analysis.py`  
**Status:** 🔄 Running  
**Log:** `/tmp/phase3_analysis.log`  
**Duration:** ~1 hour

**Outputs:**
- `minimal_mechanism_phase3_analysis.csv` - Feature matrix
- `decision_tree_rules.txt` - What predicts success

**Analysis:**
- Extract features from existing 20 pairs
- Train decision tree to predict success
- Identify patterns distinguishing success from failure

---

## PHASE 1B: K vs V Separation

**Script:** `minimal_mechanism_phase1b_kv_separation.py`  
**Status:** 🔄 Running  
**Log:** `/tmp/phase1b_kv_separation.log`  
**Duration:** ~2 hours

**Test Conditions:**
- Baseline_Control: Baseline KV, no patching
- K_Only: Recursive K + Baseline V + V_PROJ patch
- V_Only: Baseline K + Recursive V + V_PROJ patch
- Full_KV: Recursive K + Recursive V + V_PROJ patch

**Hypothesis:** V in KV matters more than K (aligns with V_PROJ finding)

**Outputs:**
- `results/runs/[timestamp]_kv_separation/kv_separation_results.csv`
- `results/runs/[timestamp]_kv_separation/kv_separation_summary.json`

---

## PHASE 2A: Window Size Ablation

**Script:** `minimal_mechanism_phase2a_window_size.py`  
**Status:** 🔄 Running  
**Log:** `/tmp/phase2a_window_size.log`  
**Duration:** ~1 hour

**Window Sizes Tested:**
- 1 (last token only)
- 4 (last 4 tokens)
- 8 (last 8 tokens)
- 16 (current)
- 32 (last 32 tokens)

**Hypothesis:** Smaller window (4-8) may be sufficient, reducing collapse

**Outputs:**
- `results/runs/[timestamp]_window_size/window_size_results.csv`
- `results/runs/[timestamp]_window_size/window_size_summary.json`

---

## Expected Findings

### Phase 3 (Analysis)
- Decision rules: "If R_V gap > X AND length_diff < Y, transfer succeeds"
- Feature importance ranking
- Failure mode classification

### Phase 1B (K vs V)
- V_Only > K_Only (if V matters more)
- V_Only ≈ Full_KV (if K doesn't matter)
- Or K_Only > V_Only (if K matters more - surprising!)

### Phase 2A (Window)
- Window 8 ≈ Window 16 (if smaller works)
- Window 4 < Window 8 (if too small)
- Optimal window size identified

---

## Next Phases (After Results)

### Phase 1A: Layer-Specific KV Replacement
- Test KV replacement at specific layers only
- Hypothesis: Only L24-27 matters

### Phase 4: Ablation Ladder
- Systematic removal of components
- Find minimal configuration

---

## Monitoring

```bash
# Check status
ssh root@157.157.221.30 -p 53751 "ps aux | grep -E 'phase3|phase1b|phase2a' | grep python3"

# Check logs
ssh root@157.157.221.30 -p 53751 "tail -30 /tmp/phase3_analysis.log"
ssh root@157.157.221.30 -p 53751 "tail -30 /tmp/phase1b_kv_separation.log"
ssh root@157.157.221.30 -p 53751 "tail -30 /tmp/phase2a_window_size.log"
```

---

## Status

🔄 **ALL THREE PHASES RUNNING IN PARALLEL**

Expected completion:
- Phase 3: ~1 hour
- Phase 1B: ~2 hours
- Phase 2A: ~1 hour









