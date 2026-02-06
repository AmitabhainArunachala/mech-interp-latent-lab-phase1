# A_results_audit.md

## RUN_INDEX.jsonl Verification Report

**Generated:** 2026-02-05  
**Analysis Group:** A8  
**File Analyzed:** `~/mech-interp-latent-lab-phase1/results/RUN_INDEX.jsonl`

---

## Executive Summary

| Metric | Value |
|--------|-------|
| **Total Entries** | 20 |
| **Unique run_dir Paths** | 14 |
| **Duplicate Entries** | 6 directories (12 entries) |
| **Existing Directories** | 14 (100%) |
| **Missing Directories** | 0 (0%) |
| **Orphan Directories** | 159 (not in index) |

---

## 1. Total Entry Count

**Result:** 20 JSON entries in RUN_INDEX.jsonl

### Entry Breakdown by Schema Type:
- **Detailed Schema (mlp_ablation experiments):** 8 entries (40%)
  - Contains comprehensive metrics: `behavior_metric`, `pr_early_*`, `pr_late_*`, `rv_*`, `prompt_ids`
- **Metrics Summary Schema:** 12 entries (60%)
  - Simplified schema with: `rv_d`, `rv_p`, `rv_delta`, `logit_diff_*`, `success`

---

## 2. Schema Consistency Analysis

### Core Fields (100% coverage):
| Field | Count | Coverage |
|-------|-------|----------|
| `experiment` | 20/20 | 100% |
| `git_commit` | 20/20 | 100% |
| `n_pairs` | 20/20 | 100% |
| `prompt_bank_version` | 20/20 | 100% |
| `run_dir` | 20/20 | 100% |
| `timestamp` | 20/20 | 100% |

### Schema Fragmentation Issues:
| Field | Count | Coverage | Issue |
|-------|-------|----------|-------|
| `model` | 12/20 | 60% | Inconsistent with `model_id` |
| `model_id` | 8/20 | 40% | Alternative model field |
| `success` | 12/20 | 60% | Missing in legacy entries |
| `schema_version` | 12/20 | 60% | Only in newer entries |
| `rv_pvalue` | 8/20 | 40% | Duplicated as `rv_p_value` (6 entries) |

### Schema Versions Found:
- `metrics_summary_v1`: 12 entries (60%)
- **NO_SCHEMA_VERSION**: 8 entries (40%) - legacy format

### Key Schema Inconsistencies:
1. **Model field duality:** Some entries use `model`, others use `model_id`
2. **P-value field naming:** `rv_pvalue` (8 entries) vs `rv_p_value` (6 entries)
3. **Baseline mean naming:** `rv_baseline_mean` vs `rv_unablated_mean` (legacy)
4. **Missing schema_version:** 40% of entries lack version identifier

---

## 3. Experiment Type Distribution

| Experiment | Count | % |
|------------|-------|---|
| `mlp_ablation_necessity_prompt_pass` | 14 | 70% |
| `multi_token_bridge` | 3 | 15% |
| `rv_l27_causal_validation` | 1 | 5% |
| `confound_validation` | 1 | 5% |
| `gemma_head_decomposition` | 1 | 5% |

**Observation:** Heavy bias toward MLP ablation experiments (70% of index).

---

## 4. Success/Failure Status

| Status | Count | % |
|--------|-------|---|
| Success (explicit) | 5 | 25% |
| Failure (explicit) | 7 | 35% |
| No `success` field | 8 | 40% |

**⚠️ Issue:** 40% of entries lack explicit success/failure indication.

---

## 5. Directory Cross-Reference

### Referenced Directories Status:
- **Total unique run_dir paths:** 14
- **Existing directories:** 14 (100%)
- **Missing directories:** 0 (0%)

**✅ All referenced directories exist on filesystem.**

### Duplicate run_dir Entries:
The following directories appear **twice** in the index (6 directories, 12 entries total):

| Directory | Appearances |
|-----------|-------------|
| `.../20260124_112408_mlp_ablation_necessity_prompt_pass_gemma_2_9b_prompt_pass_l0` | 2 |
| `.../20260124_112500_mlp_ablation_necessity_prompt_pass_gemma_2_9b_prompt_pass_l1` | 2 |
| `.../20260124_120341_mlp_ablation_necessity_prompt_pass_gemma_2_9b_prompt_pass_l2` | 2 |
| `.../20260124_120529_mlp_ablation_necessity_prompt_pass_gemma_2_9b_prompt_pass_l3` | 2 |
| `.../20260124_120619_mlp_ablation_necessity_prompt_pass_gemma_2_9b_prompt_pass_l4` | 2 |
| `.../20260124_120712_mlp_ablation_necessity_prompt_pass_gemma_2_9b_prompt_pass_l5` | 2 |

**Duplicate Entry Analysis:**
Each duplicate pair contains:
1. **Detailed entry:** Full schema with `behavior_metric`, `pr_early_*`, `pr_late_*`, `prompt_ids`
2. **Summary entry:** Minimal schema with `success: false`, null metrics

**⚠️ Root Cause:** Likely double-logging - once with full results, once with failure status.

---

## 6. Orphan Directories (Not in Index)

**Critical Finding:** 159 run directories exist but are NOT indexed.

| Category | Count |
|----------|-------|
| Total run directories on filesystem | 173 |
| Indexed in RUN_INDEX.jsonl | 14 |
| **Orphan directories** | **159** |

### Sample Orphan Directories:
- `results/archive/superseded/runs/20251216_135425_behavior_strict_vproj_only`
- `results/archive/superseded/runs/20251216_150743_kv_separation`
- `results/archive/superseded/runs/20251216_150825_window_size`
- `results/archive/superseded/runs/20251218_070943_surgical_sweep`
- `results/champion_paraphrase_hunt/runs/20251215_081556_paraphrase_hunt`
- `results/confound_validation/runs/20251215_091017_confound_validation_mistral7b_instruct_l27_w16`
- `results/gold_standard/runs/20251216_060514_confound_validation`
- ... and 152 more

**⚠️ Data Loss Risk:** 91.9% of run directories are not indexed, making them effectively invisible to automated analysis.

---

## 7. Model Distribution

| Model | Count | % |
|-------|-------|---|
| `unknown` | 12 | 60% |
| `google/gemma-2-9b` | 8 | 40% |

**⚠️ Issue:** 60% of entries have `model: "unknown"` instead of actual model identifier.

---

## 8. Timestamp Analysis

| Timestamp | Entries | Notes |
|-----------|---------|-------|
| `20260124` | 8 | Legacy format (date only) |
| `20260124_112226` | 1 | Full timestamp |
| `20260124_112312` | 1 | Full timestamp |
| `20260124_112408` | 1 | Full timestamp |
| `20260124_112500` | 1 | Full timestamp |
| `20260124_120341` | 1 | Full timestamp |
| `20260124_120529` | 1 | Full timestamp |
| `20260124_120619` | 1 | Full timestamp |
| `20260124_120712` | 1 | Full timestamp |
| `20260124_121621` | 1 | Full timestamp |

**⚠️ Issue:** Inconsistent timestamp formats - 40% use date-only (`20260124`), 60% use full datetime.

---

## 9. Issues Summary

### 🔴 Critical Issues:
1. **159 orphan directories** (91.9% of runs) not indexed
2. **6 duplicate run_dir entries** causing data confusion

### 🟡 Warning Issues:
3. **Schema fragmentation:** 68 unique fields, inconsistent naming (`rv_pvalue` vs `rv_p_value`)
4. **Model field duality:** `model` vs `model_id` inconsistency
5. **40% entries lack `success` field**
6. **60% entries have `model: "unknown"`**
7. **40% entries lack `schema_version`**
8. **Inconsistent timestamp formats**

### 🟢 Positive Findings:
- All 14 indexed directories exist (no broken references)
- Core fields (`experiment`, `run_dir`, `timestamp`) have 100% coverage
- `prompt_bank_version` consistently tracked

---

## 10. Recommendations

### Immediate Actions:
1. **Deduplicate:** Remove duplicate run_dir entries (retain full schema versions)
2. **Re-index:** Run batch indexing to capture all 159 orphan directories
3. **Schema Migration:** Unify field naming conventions

### Schema Standardization:
4. Consolidate `model`/`model_id` into single field
5. Standardize p-value field naming (`rv_pvalue`)
6. Require `schema_version` for all new entries
7. Enforce full ISO timestamps

### Data Integrity:
8. Add validation to prevent duplicate `run_dir` entries
9. Require explicit `success`/`failure` status
10. Auto-populate model identifier from run configuration

---

## Appendix: Evidence Files

- **Source:** `~/mech-interp-latent-lab-phase1/results/RUN_INDEX.jsonl`
- **Total Lines:** 20
- **File Size:** 43,004 bytes
- **Last Modified:** 2026-02-04 14:03

---

*Report generated by OpenClaw subagent (Group A8)*

---

## n=300 Experiment Deep Verification Report

**Generated:** 2026-02-05  
**Analysis Group:** A10  
**Experiment:** NeurIPS n=300 Robust Behavior Transfer

---

### Executive Summary

| Status | Item |
|--------|------|
| ✅ **EXISTS** | `neurips_n300_summary.md` (statistical summary) |
| ✅ **EXISTS** | `N300_RESULTS_ANALYSIS.md` (detailed analysis) |
| ✅ **EXISTS** | `neurips_n300_robust_experiment.py` (implementation) |
| ❌ **MISSING** | `neurips_n300_results.csv` (raw data - 300 pairs) |
| ❌ **MISSING** | `neurips_n300_summary.csv` (summary statistics CSV) |

**Verdict:** Documentation and code exist, but raw data files are missing.

---

### 1. Files Verified

#### 1.1 EXISTS: neurips_n300_summary.md
- **Location:** `docs/misc/neurips_n300_summary.md`
- **Content:** Statistical summary of n=300 experiment
- **Last Modified:** 2025-12-12 (per document header)
- **Claims Made:**
  - N = 300 prompt pairs
  - Transfer Δ = 1.87 ± 2.95, p = 9.89e-24, Cohen's d = 0.63
  - Random control Δ = 0.04 ± 1.95, p = 0.72, Cohen's d = 0.02
  - Wrong layer Δ = 1.85 ± 2.86, p = 1.54e-24, Cohen's d = 0.65
  - Transfer vs Random: t = 8.95, p = 4.35e-18
  - Transfer vs Wrong: t = 0.07, p = 9.44e-01

#### 1.2 EXISTS: N300_RESULTS_ANALYSIS.md
- **Location:** `docs/analysis/N300_RESULTS_ANALYSIS.md`
- **Content:** Comprehensive analysis with key findings:
  - Effect is real (p < 0.001) but smaller than pilot (mean 2.62 vs 11)
  - "Wrong layer" (L5) also shows transfer (p = 0.94 vs L27)
  - 28% of pairs show no transfer (score = 0)
  - Missing critical KV-only control
  - Cannot claim layer specificity

#### 1.3 EXISTS: neurips_n300_robust_experiment.py
- **Location:** Root directory
- **Content:** Full implementation with:
  - CONFIG dict with n_pairs = 300
  - save_csv = "neurips_n300_results.csv"
  - save_summary = "neurips_n300_summary.md"
  - Proper controls (random, wrong-layer)
  - Statistical analysis (t-tests, effect sizes, CIs)

---

### 2. Files MISSING

#### 2.1 MISSING: neurips_n300_results.csv
- **Expected Location:** Root directory (per CONFIG)
- **Expected Content:** Raw data for 300 prompt pairs
- **Search Results:** File not found anywhere in repository
- **Impact:** Cannot verify statistical claims against raw data

#### 2.2 MISSING: neurips_n300_summary.csv
- **Expected Location:** Root directory
- **Expected Content:** Summary statistics CSV
- **Search Results:** File not found
- **Impact:** Secondary file, less critical than raw results

---

### 3. Claims Cross-Check

#### 3.1 Statistical Claims (from summary.md)

| Claim | Value in Doc | Status |
|-------|--------------|--------|
| N pairs | 300 | ✅ Consistent |
| Baseline mean | 0.76 ± 1.48 | ✅ Consistent with analysis.md |
| Transfer mean | 2.62 ± 2.69 | ✅ Consistent with analysis.md |
| Random mean | 0.80 ± 1.58 | ✅ Consistent with analysis.md |
| Wrong layer mean | 2.61 ± 2.62 | ✅ Consistent with analysis.md |
| Transfer p-value | 9.89e-24 | ✅ Significant |
| Random p-value | 7.22e-01 | ✅ Not significant |
| Wrong layer p-value | 1.54e-24 | ✅ Significant |
| Transfer vs Wrong p | 9.44e-01 | ✅ Not different |

#### 3.2 Key Finding: Layer Specificity Contradiction

**CONTRADICTION IDENTIFIED:**
- Documents claim L27 as target layer in CONFIG
- But results show L5 ("wrong layer") performs identically (p = 0.94)
- Analysis.md acknowledges this means "L27 is NOT special"
- This contradicts the pilot claim that L27 was the critical layer

**Status:** Documents honestly report this finding, but it contradicts initial hypotheses.

---

### 4. What EXISTS vs MISSING

#### EXISTS:
1. ✅ Statistical summary document (neurips_n300_summary.md)
2. ✅ Detailed analysis with honest limitations (N300_RESULTS_ANALYSIS.md)
3. ✅ Complete experiment implementation (neurips_n300_robust_experiment.py)
4. ✅ Configuration showing n=300 intent
5. ✅ Honest reporting of limitations (wrong layer works, effect is variable)

#### MISSING:
1. ❌ Raw data file (neurips_n300_results.csv) - **CRITICAL**
2. ❌ Summary CSV (neurips_n300_summary.csv) - minor
3. ❌ KV-only control results (mentioned as needed but not run)

#### CONTRADICTED:
1. ⚠️ "L27 is special" - Contradicted by wrong layer results
2. ⚠️ "100% transfer" - Contradicted by mean of 2.62 vs pilot 11
3. ⚠️ Layer specificity - Cannot be claimed given L5 = L27 results

---

### 5. Data Integrity Assessment

| Aspect | Status | Notes |
|--------|--------|-------|
| Raw data | ❌ MISSING | neurips_n300_results.csv not found |
| Documentation | ✅ EXISTS | Both summary and analysis present |
| Code | ✅ EXISTS | Full implementation verified |
| Statistical claims | ⚠️ UNVERIFIABLE | No raw data to cross-check |
| Limitations | ✅ ACKNOWLEDGED | Documents are honest about issues |

---

### 6. Conclusion

**The n=300 experiment is DOCUMENTED but not DATA-VERIFIABLE.**

**Strengths:**
- Comprehensive documentation exists
- Honest reporting of limitations
- Statistical claims are internally consistent
- Wrong layer finding is transparently reported

**Weaknesses:**
- Raw data file (neurips_n300_results.csv) is missing
- Cannot independently verify statistical claims
- Critical KV-only control was never run

**Recommendation:**
If this experiment is to be cited, the missing raw data file should be located or the experiment should be re-run with proper data retention.

---

*Report generated by OpenClaw subagent (Group A10)*
