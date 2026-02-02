# Stage 1 Smoke Test — Complete Summary

**Date:** January 5, 2025  
**Status:** ✅ **INFRASTRUCTURE VALIDATED**

---

## Test Results

### Test 1: L0 Necessity (Ablation) — ✅ PASSED

**Run:** `results/phase1_mechanism/runs/20260105_131922_l0_necessity_smoke_test/`

**Success Criteria:** 7/7 ✅
- ✅ CSV has `recursive_prompt_id`, `baseline_prompt_id`
- ✅ `summary.json` has all required keys
- ✅ `metadata.json` exists
- ✅ `RUN_INDEX.jsonl` created

**Results:**
- R_V Delta: +0.669 (L0 MLP is NECESSARY)
- Prompt IDs stored: 5 recursive + 5 baseline

---

### Test 2: L0 Sufficiency (Patch) — ✅ PASSED

**Run:** `results/phase1_mechanism/runs/20260105_133626_l0_sufficiency_smoke_test/`

**Success Criteria:** 7/7 ✅
- ✅ CSV has `recursive_prompt_id`, `baseline_prompt_id`
- ✅ `summary.json` has all required keys
- ✅ `metadata.json` exists
- ✅ `RUN_INDEX.jsonl` updated (2 entries)

**Results:**
- R_V Restoration: (check summary.json)
- Mode Score Delta: (check summary.json)

---

## Infrastructure Validation

### ✅ All Systems Operational

1. **PromptLoader with IDs**
   - `get_balanced_pairs_with_ids()` working
   - Prompt IDs correctly stored in CSV

2. **Run Metadata Collection**
   - `get_run_metadata()` working
   - Git commit, prompt bank version, prompt IDs all captured

3. **Standardized Metric Contract**
   - `mode_score_m` (PRIMARY) in summary.json
   - `rv`, `rv_restoration_pct` (SECONDARY) in summary.json
   - `eval_window`, `intervention_scope`, `behavior_metric` standardized

4. **Run Index Tracking**
   - `RUN_INDEX.jsonl` created and updated
   - Centralized tracking of all runs

---

## Issues Fixed During Smoke Test

1. ✅ **summary.json corruption** - Fixed numpy bool_ serialization
2. ✅ **RUN_INDEX.jsonl path issue** - Fixed relative/absolute path handling

---

## Stage 1 Verdict: ✅ PASSED

**Both smoke tests passed all success criteria.**

**Ready for Stage 2: Canonical Suite (13 experiments)**

---

## Next Steps: Stage 2

**Canonical Suite Experiments:**

1. **Necessity (4):** L0, L1, L2, L3 zero ablation
2. **Sufficiency (2):** L0 patch, L0+L1 patch
3. **Position (1):** L0 position-specific (BOS, first-4, last-16, all)
4. **Windowed Denoising (4):** L0-L2, L0-L4, L0-L8, L0-L12
5. **KV Interaction (2):** KV-only, KV + L0-L4 window

**Output:** `results/canonical_suite_v1_0/` with `consolidated_results.csv`

---

**Status:** ✅ **STAGE 1 COMPLETE - READY FOR STAGE 2**


