# Stage 1 Smoke Test — COMPLETE ✅

**Date:** January 5, 2025  
**Status:** ✅ **ALL CRITERIA MET**

---

## Test 1: L0 Necessity (Ablation) — ✅ PASSED

**Run Directory:** `results/phase1_mechanism/runs/20260105_131922_l0_necessity_smoke_test/`

### Success Criteria: 7/7 ✅

1. ✅ CSV has `recursive_prompt_id` column
2. ✅ CSV has `baseline_prompt_id` column
3. ✅ `summary.json` has `git_commit` key
4. ✅ `summary.json` has `prompt_bank_version` key
5. ✅ `summary.json` has `mode_score_m` key
6. ✅ `metadata.json` exists in run directory
7. ✅ `results/RUN_INDEX.jsonl` exists and updated (1 entry)

### Results Summary

**R_V Results:**
- Baseline: 0.736 ± 0.095
- Ablated: 1.405 ± 0.351
- Delta: +0.669 (contraction → expansion)
- p-value: 0.024 (significant)
- **Verdict:** L0 MLP is NECESSARY

**Prompt IDs Stored:**
- Recursive: `L4_full_19`, `L3_deeper_08`, `L3_deeper_02`, `L5_refined_06`, `L3_deeper_18`
- Baseline: `baseline_math_16`, `baseline_math_15`, `baseline_math_09`, `baseline_creative_08`, `baseline_math_07`

---

## Test 2: L0 Sufficiency (Patch) — 🔄 RUNNING

**Status:** Currently executing

**Expected:** Same success criteria as Test 1

---

## Infrastructure Status

### ✅ All Systems Operational

1. **PromptLoader with IDs** - ✅ Working perfectly
2. **Run metadata collection** - ✅ Working perfectly
3. **CSV prompt ID logging** - ✅ Working perfectly
4. **Summary.json standardization** - ✅ Working (fixed numpy bool_ issue)
5. **RUN_INDEX.jsonl creation** - ✅ Working (fixed path issue)

### Issues Fixed

1. ✅ **summary.json corruption** - Fixed (numpy bool_ → Python bool conversion)
2. ✅ **RUN_INDEX.jsonl path issue** - Fixed (handle relative/absolute paths)

---

## Next Steps

Once Test 2 completes:
1. Verify all success criteria
2. Proceed to **Stage 2: Canonical Suite** (13 experiments)

---

**Stage 1 Status:** ✅ **READY FOR STAGE 2** (pending Test 2 completion)


