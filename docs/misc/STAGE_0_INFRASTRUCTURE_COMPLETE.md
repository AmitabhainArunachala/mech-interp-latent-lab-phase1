# Stage 0 Infrastructure Upgrade — COMPLETE ✅

**Date:** January 5, 2025  
**Status:** All infrastructure upgrades implemented

---

## ✅ 0.1 PromptLoader with IDs — COMPLETE

**File:** `prompts/loader.py`

**Added:**
- `_find_prompt_id(text: str) -> Optional[str]` - Finds prompt ID by text
- `get_balanced_pairs_with_ids()` - Returns `(rec_id, base_id, rec_text, base_text)` tuples

**Usage:**
```python
pairs_with_ids = loader.get_balanced_pairs_with_ids(n_pairs=30, seed=42)
for rec_id, base_id, rec_text, base_text in pairs_with_ids:
    # Use IDs for reproducibility
```

---

## ✅ 0.2 Run Metadata Helper — COMPLETE

**File:** `src/utils/run_metadata.py` (NEW)

**Functions:**
- `get_git_commit()` - Gets current git commit hash
- `get_run_metadata(cfg, prompt_ids, eval_window, intervention_scope, behavior_metric)` - Standardized metadata dict
- `save_metadata(run_dir, metadata)` - Saves metadata.json
- `append_to_run_index(run_dir, summary)` - Appends to `results/RUN_INDEX.jsonl`

**Metadata Includes:**
- `git_commit` - Git commit hash
- `prompt_bank_version` - SHA256 hash of bank.json
- `prompt_ids` - List of recursive and baseline prompt IDs
- `model_id` - Model name/version
- `seed`, `n_pairs` - Experimental parameters
- `eval_window` - Measurement window (default: 16)
- `intervention_scope` - Where intervention applied ("all_tokens", "last_16", etc.)
- `behavior_metric` - Primary metric name ("mode_score_m")

---

## ✅ 0.3 Metric Contract — COMPLETE

**Standardized Summary Keys:**

**PRIMARY Metrics:**
- `mode_score_m` - Mode Score M (renamed from `mode_baseline_mean`)
- `mode_score_m_delta` - Change in Mode Score M
- `mode_t_statistic`, `mode_pvalue`, `mode_significant` - Statistical tests

**SECONDARY Metrics:**
- `rv` - R_V baseline value
- `rv_restoration_pct` - R_V restoration percentage (when applicable)
- `rv_baseline_mean`, `rv_ablated_mean`, `rv_patched_mean` - R_V values
- `rv_t_statistic`, `rv_pvalue`, `rv_significant` - Statistical tests

**METADATA:**
- `eval_window` - Measurement window size
- `intervention_scope` - Intervention scope
- `behavior_metric` - Primary behavior metric name
- All metadata from `get_run_metadata()`

**Legacy Metrics (Secondary/Non-Comparable):**
- `coherence_mean`, `recursion_score_mean` - Still logged but marked as secondary

---

## ✅ 0.4 Pipeline Updates — COMPLETE

**Updated Pipelines:**

### 1. `mlp_ablation_necessity.py`
- ✅ Uses `get_balanced_pairs_with_ids()`
- ✅ Logs `recursive_prompt_id`, `baseline_prompt_id` to CSV
- ✅ Includes metadata in summary.json
- ✅ Calls `save_metadata()` and `append_to_run_index()`
- ✅ Standardized metric contract (mode_score_m, rv, eval_window, intervention_scope)

### 2. `mlp_sufficiency_test.py`
- ✅ Uses `get_balanced_pairs_with_ids()`
- ✅ Logs `recursive_prompt_id`, `baseline_prompt_id` to CSV
- ✅ Includes metadata in summary.json
- ✅ Calls `save_metadata()` and `append_to_run_index()`
- ✅ Standardized metric contract (mode_score_m, rv_restoration_pct, eval_window, intervention_scope)

### 3. `mlp_combined_sufficiency_test.py`
- ✅ Uses `get_balanced_pairs_with_ids()`
- ✅ Logs `recursive_prompt_id`, `baseline_prompt_id` to CSV
- ✅ Includes metadata in summary.json
- ✅ Calls `save_metadata()` and `append_to_run_index()`
- ✅ Standardized metric contract (mode_score_m, rv_restoration_pct, eval_window, intervention_scope)

---

## Next Steps: Stage 1 — Smoke Test

**Run 2 experiments:**
1. L0 Necessity (ablation) — `configs/mlp_ablation_necessity_l0.json`
2. L0 Sufficiency (patch) — `configs/mlp_sufficiency_l0.json`

**Success Criteria:**
- [ ] `results.csv` has `recursive_prompt_id`, `baseline_prompt_id` columns
- [ ] `summary.json` has `git_commit`, `prompt_bank_version`, `mode_score_m` keys
- [ ] `metadata.json` exists in run directory
- [ ] `results/RUN_INDEX.jsonl` exists and updated

**DO NOT proceed to Stage 2 until smoke test passes.**

---

## Files Changed

1. `prompts/loader.py` - Added `get_balanced_pairs_with_ids()` and `_find_prompt_id()`
2. `src/utils/run_metadata.py` - NEW FILE - Metadata helpers
3. `src/pipelines/mlp_ablation_necessity.py` - Updated to use new infrastructure
4. `src/pipelines/mlp_sufficiency_test.py` - Updated to use new infrastructure
5. `src/pipelines/mlp_combined_sufficiency_test.py` - Updated to use new infrastructure

---

**Status:** ✅ **READY FOR STAGE 1 SMOKE TEST**


