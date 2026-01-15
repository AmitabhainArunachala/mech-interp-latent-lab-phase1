# Pipeline Compliance Matrix

**Date:** January 10, 2026  
**Audited By:** Cursor Agent  
**Standard:** Industry-Grade Methodology Contract v1.0

---

## Legend

| Symbol | Meaning |
|--------|---------|
| ✅ | Compliant (code implements + disk evidence exists) |
| ⚠️ | Partial (code implements, no disk evidence OR partial implementation) |
| ❌ | Non-compliant (not implemented or missing) |

---

## A. Core MLP Pipelines (Mechanism Suite)

### 1. `mlp_ablation_necessity.py`

| Requirement | Status | Evidence |
|-------------|--------|----------|
| **Prompt IDs stored** | ✅ | `pairs_with_ids = loader.get_balanced_pairs_with_ids()` (line 113) |
| **Bank hash logged** | ✅ | `prompt_bank_version.txt/json` written (line 108-109) |
| **Seed set** | ✅ | `set_seed(seed)` (line 93) |
| **Gen params logged** | ❌ | `temperature`, `max_new_tokens` used but not in metadata |
| **Intervention scope** | ✅ | `"all_tokens"` in metadata (line 305) |
| **Eval window** | ✅ | `eval_window=window_size` (line 304) |
| **mode_score_m** | ✅ | `ModeScoreMetric.compute_score()` used (line 165, 208) |
| **restore_norm(M)** | ❌ | Not computed |
| **rv computed** | ✅ | `compute_rv()` called (line 152, 190) |
| **Norm logs** | ❌ | Not logged |
| **Run index append** | ✅ | `append_to_run_index(run_dir, summary)` (line 369) |
| **Metadata saved** | ✅ | `save_metadata(run_dir, metadata)` (line 361) |

**Overall:** ⚠️ **PARTIAL** - Core metrics present, gen params and restore_norm missing

---

### 2. `mlp_sufficiency_test.py`

| Requirement | Status | Evidence |
|-------------|--------|----------|
| **Prompt IDs stored** | ✅ | `pairs_with_ids = loader.get_balanced_pairs_with_ids()` (line 122) |
| **Bank hash logged** | ✅ | `prompt_bank_version.txt/json` written (line 117-118) |
| **Seed set** | ✅ | `set_seed(seed)` (line 108) |
| **Gen params logged** | ❌ | Not in metadata |
| **Intervention scope** | ✅ | `"last_16"` in metadata (line 293) |
| **Eval window** | ✅ | `eval_window=window_size` (line 292) |
| **mode_score_m** | ✅ | `ModeScoreMetric.compute_score()` used (line 147, 205) |
| **restore_norm(M)** | ⚠️ | `rv_restoration_pct` computed, not `mode_score_m` restoration |
| **rv computed** | ✅ | `compute_rv()` called (line 143, 186, 230) |
| **Norm logs** | ❌ | Not logged |
| **Run index append** | ✅ | `append_to_run_index(run_dir, summary)` (line 346) |
| **Metadata saved** | ✅ | `save_metadata(run_dir, metadata)` (line 338) |

**Overall:** ⚠️ **PARTIAL** - Core metrics present, gen params and mode restore_norm missing

---

### 3. `mlp_combined_sufficiency_test.py`

| Requirement | Status | Evidence |
|-------------|--------|----------|
| **Prompt IDs stored** | ✅ | `pairs_with_ids = loader.get_balanced_pairs_with_ids()` (line 172) |
| **Bank hash logged** | ✅ | `prompt_bank_version.txt/json` written (line 167-168) |
| **Seed set** | ✅ | `set_seed(seed)` (line 158) |
| **Gen params logged** | ❌ | Not in metadata |
| **Intervention scope** | ✅ | `"last_16"` in metadata (line 402) |
| **Eval window** | ✅ | `eval_window=window_size` (line 401) |
| **mode_score_m** | ✅ | `ModeScoreMetric.compute_score()` used (line 198, 288) |
| **restore_norm(M)** | ⚠️ | `rv_restoration_pct` computed, not mode restoration |
| **rv computed** | ✅ | `compute_rv()` called (line 194, 280, 316) |
| **Norm logs** | ✅ | `MultiMLPPatchingHook.get_norm_logs()` (line 273) |
| **Run index append** | ✅ | `append_to_run_index(run_dir, summary)` (line 457) |
| **Metadata saved** | ✅ | `save_metadata(run_dir, metadata)` (line 449) |

**Overall:** ⚠️ **PARTIAL** - Best of MLP suite, has norm logs, missing gen params

---

### 4. `random_direction_control.py`

| Requirement | Status | Evidence |
|-------------|--------|----------|
| **Prompt IDs stored** | ❌ | Uses `get_balanced_pairs()` not `get_balanced_pairs_with_ids()` (line 111) |
| **Bank hash logged** | ✅ | `prompt_bank_version.txt/json` written (line 106-107) |
| **Seed set** | ✅ | `set_seed(seed)` (line 97) |
| **Gen params logged** | ❌ | Not in metadata |
| **Intervention scope** | ❌ | Not specified |
| **Eval window** | ❌ | Not specified |
| **mode_score_m** | ✅ | `ModeScoreMetric.compute_score()` used (line 180, 223) |
| **restore_norm(M)** | ❌ | Not computed |
| **rv computed** | ✅ | `compute_rv()` called (line 177, 205) |
| **Norm logs** | ❌ | Not logged |
| **Run index append** | ❌ | Not called |
| **Metadata saved** | ❌ | Not called |

**Overall:** ❌ **NON-COMPLIANT** - Missing prompt IDs, metadata, run index

---

### 5. `circuit_discovery.py`

| Requirement | Status | Evidence |
|-------------|--------|----------|
| **Prompt IDs stored** | ❌ | Uses `get_balanced_pairs()` not `get_balanced_pairs_with_ids()` (line 51) |
| **Bank hash logged** | ✅ | `prompt_bank_version.txt/json` written (line 46-47) |
| **Seed set** | ✅ | `set_seed(seed)` (line 37) |
| **Gen params logged** | ❌ | No generation in this pipeline |
| **Intervention scope** | ❌ | Not specified |
| **Eval window** | ❌ | Not specified |
| **mode_score_m** | ✅ | `ModeScoreMetric.compute_score()` used (line 99, 143) |
| **restore_norm(M)** | ❌ | Not applicable (discovery, not patching) |
| **rv computed** | ❌ | Not computed |
| **Norm logs** | ❌ | Not logged |
| **Run index append** | ❌ | Not called |
| **Metadata saved** | ❌ | Not called |

**Overall:** ❌ **NON-COMPLIANT** - Missing prompt IDs, metadata, run index, rv

---

## B. Summary Compliance Table

| Pipeline | Prompt IDs | Bank Hash | Seed | Gen Params | Scope | Window | mode_score_m | restore_norm | rv | Norms | Index | Metadata |
|----------|:----------:|:---------:|:----:|:----------:|:-----:|:------:|:------------:|:------------:|:--:|:-----:|:-----:|:--------:|
| **mlp_ablation_necessity** | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ | ✅ |
| **mlp_sufficiency_test** | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ⚠️ | ✅ | ❌ | ✅ | ✅ |
| **mlp_combined_sufficiency** | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ⚠️ | ✅ | ✅ | ✅ | ✅ |
| **random_direction_control** | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ |
| **circuit_discovery** | ❌ | ✅ | ✅ | N/A | ❌ | ❌ | ✅ | N/A | ❌ | ❌ | ❌ | ❌ |

---

## C. Compliance Scores

| Pipeline | Score | Grade |
|----------|-------|-------|
| **mlp_ablation_necessity** | 9/12 | **B+** (75%) |
| **mlp_sufficiency_test** | 9/12 | **B+** (75%) |
| **mlp_combined_sufficiency** | 10/12 | **A-** (83%) |
| **random_direction_control** | 4/12 | **D** (33%) |
| **circuit_discovery** | 4/11 | **D** (36%) |

---

## D. Legacy Pipelines (Not Audited in Detail)

These pipelines exist but use deprecated metrics and/or lack industry-grade metadata:

| Pipeline | Primary Issue |
|----------|---------------|
| `p1_ablation.py` | Uses inline `recursion_score` (regex), no `mode_score_m` |
| `surgical_sweep.py` | Uses inline `recursion_score`, no `mode_score_m` |
| `kv_mechanism.py` | Uses `rv` only, no `mode_score_m` |
| `layer_sweep.py` | Uses inline keyword counting, no `mode_score_m` |
| `steering_analysis.py` | Legacy metrics, no standardized metadata |

---

## E. Disk Evidence Status

### Verified On-Disk Artifacts (January 10, 2026)

| Artifact | Path | Status |
|----------|------|--------|
| `run_index.csv` | `results/run_index.csv` | ✅ Exists (36 runs indexed) |
| `RUN_INDEX.jsonl` | `results/RUN_INDEX.jsonl` | ❌ **DOES NOT EXIST** |
| `bank.json` | `prompts/bank.json` | ✅ Exists (754 prompts) |
| `metadata.json` | `results/**/runs/**/metadata.json` | ❌ **NOT FOUND** in any run dir |

### Critical Finding

Despite code implementations for:
- `append_to_run_index()` → Creates `RUN_INDEX.jsonl`
- `save_metadata()` → Creates `metadata.json`

**No disk evidence exists** that these functions have executed successfully in any production run.

---

## F. Recommendations by Pipeline

### mlp_ablation_necessity (Highest Priority)

```diff
# Line ~300: Add generation_params to metadata
+ metadata["generation_params"] = {
+     "max_new_tokens": max_new_tokens,
+     "temperature": 0.0,
+     "do_sample": False,
+ }
```

### random_direction_control (Needs Overhaul)

```diff
- pairs = loader.get_balanced_pairs(n_pairs=n_pairs, seed=seed)
+ pairs_with_ids = loader.get_balanced_pairs_with_ids(n_pairs=n_pairs, seed=seed)

# Add at end:
+ from src.utils.run_metadata import get_run_metadata, save_metadata, append_to_run_index
+ metadata = get_run_metadata(cfg, prompt_ids=pairs_with_ids, ...)
+ save_metadata(run_dir, metadata)
+ append_to_run_index(run_dir, summary_json)
```

### circuit_discovery (Needs Overhaul)

```diff
- pairs = loader.get_balanced_pairs(n_pairs=n_pairs, seed=seed)
+ pairs_with_ids = loader.get_balanced_pairs_with_ids(n_pairs=n_pairs, seed=seed)

# Add rv computation for each patched condition
+ from src.metrics.rv import compute_rv

# Add metadata/run_index calls at end
```

---

## G. Verification Protocol

After fixing any pipeline, run this verification:

```bash
# 1. Run the pipeline
python -m src.pipelines.run --config configs/<experiment>.json

# 2. Check outputs exist
ls -la results/phase*/runs/*_<experiment>/
# Expected: config.json, summary.json, metadata.json, *.csv

# 3. Check RUN_INDEX.jsonl was created/appended
cat results/RUN_INDEX.jsonl | tail -1

# 4. Validate metadata contents
cat results/phase*/runs/*_<experiment>/metadata.json | jq .prompt_ids
```

---

**Matrix Version:** 1.0  
**Last Updated:** January 10, 2026  
**Next Audit:** After pipeline fixes applied
