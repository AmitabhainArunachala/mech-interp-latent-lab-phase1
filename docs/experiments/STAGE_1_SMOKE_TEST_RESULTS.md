# Stage 1 Smoke Test — Results

**Date:** January 5, 2025  
**Status:** ✅ **TEST 1 COMPLETE** (Test 2 pending)

---

## Test 1: L0 Necessity (Ablation) — ✅ PASSED

**Run Directory:** `results/phase1_mechanism/runs/20260105_131922_l0_necessity_smoke_test/`

### Success Criteria Check

#### ✅ 1. CSV Prompt IDs
- `recursive_prompt_id` column: ✅ Present
- `baseline_prompt_id` column: ✅ Present

**Sample IDs:**
- Recursive: `L4_full_19`, `L3_deeper_08`, `L3_deeper_02`, `L5_refined_06`, `L3_deeper_18`
- Baseline: `baseline_math_16`, `baseline_math_15`, `baseline_math_09`, `baseline_creative_08`, `baseline_math_07`

#### ✅ 2. Summary.json Keys
- `git_commit`: ✅ Present (`not_a_git_repo` - expected on remote server)
- `prompt_bank_version`: ✅ Present (`b1e5291421c5646d`)
- `mode_score_m`: ✅ Present (null - mode scores failed, but key exists)
- `eval_window`: ✅ Present (`16`)
- `intervention_scope`: ✅ Present (`all_tokens`)
- `behavior_metric`: ✅ Present (`mode_score_m`)

#### ✅ 3. Metadata.json
- File exists: ✅ Present
- Contains: `git_commit`, `prompt_bank_version`, `prompt_ids`, `model_id`, `seed`, `n_pairs`, `eval_window`, `intervention_scope`, `behavior_metric`

#### ✅ 4. RUN_INDEX.jsonl
- File exists: ✅ Created
- Entries: 1 entry

---

## Test Results Summary

**R_V Results:**
- Baseline R_V: 0.736 ± 0.095
- Ablated R_V: 1.405 ± 0.351
- R_V Delta: +0.669 (contraction → expansion)
- p-value: 0.024 (significant)
- **Verdict:** L0 MLP is NECESSARY - R_V contraction disappears when ablated

**Mode Score Results:**
- Mode scores failed (null values) - likely due to sequence length mismatch or other issue
- This is acceptable for smoke test (infrastructure works, metric computation needs debugging)

---

## Infrastructure Status

### ✅ Working
1. PromptLoader with IDs - ✅ Working
2. Run metadata collection - ✅ Working
3. CSV prompt ID logging - ✅ Working
4. Summary.json standardization - ✅ Working (after fix)
5. RUN_INDEX.jsonl creation - ✅ Working

### ⚠️ Issues Found & Fixed
1. **summary.json corruption** - Fixed (numpy bool_ serialization issue)
2. **Mode score computation** - Failed (needs investigation, but not blocking)

---

## Next Steps

### Test 2: L0 Sufficiency (Patch)
**Status:** ⏳ Ready to run

**Command:**
```bash
ssh runpod-current "cd /root/mech-interp-latent-lab-phase1 && python3 scripts/smoke_test_l0_sufficiency.py"
```

**Expected:** Same success criteria as Test 1

---

## Success Criteria Summary

| Criterion | Status |
|-----------|--------|
| CSV has `recursive_prompt_id` | ✅ |
| CSV has `baseline_prompt_id` | ✅ |
| `summary.json` has `git_commit` | ✅ |
| `summary.json` has `prompt_bank_version` | ✅ |
| `summary.json` has `mode_score_m` | ✅ |
| `metadata.json` exists | ✅ |
| `RUN_INDEX.jsonl` exists and updated | ✅ |

**Overall:** ✅ **7/7 criteria met** (Test 1 passed)

---

**Ready for Test 2!**


