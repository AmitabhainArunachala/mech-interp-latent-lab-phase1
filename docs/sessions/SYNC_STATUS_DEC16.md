# Sync Status - Dec 16, 2025

**Last Sync:** 2025-12-16T15:15:00Z  
**Status:** ✅ KEY FILES SAVED LOCALLY

## Critical Files Saved Locally

### Investigation Results
- ✅ `MINIMAL_MECHANISM_RESULTS.md` - Full results summary
- ✅ `MINIMAL_MECHANISM_INVESTIGATION.md` - Investigation plan
- ✅ `MINIMAL_MECHANISM_STATUS.md` - Status tracking
- ✅ `minimal_mechanism_phase3_analysis.csv` - Feature matrix
- ✅ `decision_tree_rules.txt` - Decision tree rules
- ✅ `IMPROVED_SCORER_FINAL_RESULTS.md` - Scorer improvements (45% transfer!)

### Experiment Scripts
- ✅ `minimal_mechanism_phase3_analysis.py`
- ✅ `minimal_mechanism_phase1b_kv_separation.py`
- ✅ `minimal_mechanism_phase2a_window_size.py`

### Results Directories (Local)
- ✅ `results/runs/20251216_140553_behavior_strict/` - Improved scorer run
- ✅ `results/runs/20251216_150743_kv_separation/` - K vs V experiment
- ✅ `results/runs/20251216_150825_window_size/` - Window size experiment

## Key Findings

### Phase 1B: K vs V Separation
- **Finding:** Both K and V needed (Full_KV 3.2x better than V_Only)
- **Full_KV:** 0.5650 mean score, 14/20 transfer (70%)
- **V_Only:** 0.1750 mean score, 5/20 transfer (25%)
- **K_Only:** 0.1200 mean score, 4/20 transfer (20%)

### Phase 2A: Window Size
- **Finding:** Window 1 performs best (surprising!)
- **Window 1:** 0.3950 mean score, 10/20 transfer (50%)
- **Window 16:** 0.1200 mean score, 3/20 transfer (15%)
- **Note:** Needs verification against proper baseline

### Phase 3: Success Prediction
- **Finding:** Success is predictable!
- **Top predictors:** token_overlap (0.344), rec_rv (0.321)
- **Decision rule:** Lower R_V (≤0.47) predicts success for story prompts

## Next Steps

1. **Phase 1A:** Layer-specific KV replacement (L24-27 only)
2. **Phase 4:** Full ablation ladder
3. **Verify:** Window 1 result with proper baseline

## Status✅ **All critical files saved locally**  
⚠️ **Full sync interrupted - will resume when connection restored**