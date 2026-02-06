# Comprehensive Signal Quality & Industry Standard Audit Report

**Date:** 2026-02-05  
**Auditor:** Claude (Comprehensive Analysis)  
**Version:** 1.0  
**Status:** COMPLETE

---

## Executive Summary

This audit evaluates `mech-interp-latent-lab-phase1` against industry-standard rigor requirements. **Critical finding**: The repository contains **high-signal causal validation work** (Mistral L27 causal validation, cross-architecture validation) but also **contract violations** (single-layer PR mislabeled as R_V in some results) and **incomplete artifacts** (missing hardware info, inconsistent metadata).

**Overall Assessment:**
- **Signal Quality**: 7.5/10 (strong causal findings, but some contract violations)
- **Industry Standard Compliance**: 6.5/10 (good structure, missing artifacts)
- **Reproducibility**: 7/10 (config-driven, but hardware/precision gaps)

**Key Actions Required:**
1. **CRITICAL**: Fix `final_results.json` - contains single-layer PR values mislabeled as "rv"
2. **HIGH**: Add hardware_info.json to all runs
3. **HIGH**: Verify all R_V implementations compute PR_late/PR_early ratio
4. **MEDIUM**: Archive duplicate/superseded results
5. **MEDIUM**: Ramp up promising low-N experiments to n≥50

---

## 1. KEEP_SIGNAL List

| File Path | n | Stats | Controls | Artifacts | Why High-Signal |
|-----------|---|-------|----------|-----------|-----------------|
| `results/canonical/rv_l27_causal_validation/20251216_060955_rv_l27_causal_validation/summary.json` | 45 | d=-3.56, p<1e-6 | random/shuffled/wrong_layer | ✅ config, summary, CSV | **Causal validation with perfect control separation** |
| `results/phase1_cross_architecture/runs/20260202_121604_rv_l27_causal_validation_mistral_7b/summary.json` | 45 | d=-2.26, p<1e-18 | random/shuffled/wrong_layer | ✅ config, summary, CSV | Cross-architecture validation, verified in STATISTICAL_AUDIT_REPORT |
| `results/phase1_cross_architecture/runs/20260202_115958_rv_l27_causal_validation_pythia_1_4b/summary.json` | 45 | d=-0.31, p=0.021 | random/shuffled/wrong_layer | ✅ config, summary, CSV | Cross-architecture validation (weaker but significant) |
| `results/phase1_cross_architecture/runs/20260202_120856_rv_l27_causal_validation_qwen2_7b/summary.json` | 45 | d=-0.72, p<1e-5 | random/shuffled/wrong_layer | ✅ config, summary, CSV | Cross-architecture validation |
| `results/phase1_cross_architecture/runs/20260202_125718_rv_l27_causal_validation_opt_6_7b/summary.json` | 45 | d=-1.84, p<1e-15 | random/shuffled/wrong_layer | ✅ config, summary, CSV | Cross-architecture validation |
| `results/phase1_cross_architecture/runs/20260202_130807_rv_l27_causal_validation_gpt2_xl/summary.json` | 45 | d=-1.14, p<1e-9 | random/shuffled/wrong_layer | ✅ config, summary, CSV | Cross-architecture validation |
| `RECOVERED_GOLD/MISTRAL_L27_CAUSAL_VALIDATION_COMPLETE.md` | 45 | d=-3.56, p<1e-6 | ✅ Verified | ✅ Documented | **Canonical causal proof document** |
| `src/metrics/rv.py` | N/A | N/A | N/A | ✅ Code | **Correct R_V implementation (PR_late/PR_early)** |

**Total High-Signal Results: 7 experiments + 1 canonical implementation**

---

## 2. RAMP_UP List

| File Path | Current n | Target n | Missing Controls | Missing Artifacts | Config Changes Needed |
|-----------|-----------|----------|------------------|-------------------|----------------------|
| `results/phase1_cross_architecture/runs/20260202_133252_multi_token_bridge_mistral_7b_bridge/summary.json` | 40 | 80 | None (has controls) | hardware_info.json | `{"n_prompts": 80}` in config |
| `results/discovery/behavioral_grounding/20251216_123737_behavior_strict/summary.json` | ~20-30 | 50 | wrong_layer | hardware_info.json, prompt_bank_version | Add wrong_layer control, increase n_pairs to 50 |
| `results/discovery/path_patching/20251213_055827_path_patching_mechanism_default/summary.json` | ~20-30 | 50 | wrong_layer | hardware_info.json | Add wrong_layer control, increase n_pairs to 50 |
| `results/canonical/confound_validation/20251216_060911_confound_validation/summary.json` | ~30 | 50 | None | hardware_info.json | Increase n_champions/n_length_matched to 50 each |

**Total RAMP_UP Candidates: 4 experiments**

---

## 3. ARCHIVE_ONLY List

| File Path | Reason | Evidence |
|-----------|--------|----------|
| `results/canonical/final_results.json` | **CONTRACT VIOLATION** | Contains single-layer PR values (5.279, 6.733) mislabeled as "rv". Should be PR_late/PR_early ratio. Values >1.0 contradict R_V<1.0 finding. |
| `results/canonical/session_2/` | **DUPLICATE** | Superseded by `session_2_complete/` and `session_2_final/` |
| `results/canonical/session_complete/` | **DUPLICATE** | Multiple versions exist, keep only most recent |
| `results/canonical/session_2_complete/` | **DUPLICATE** | Superseded by `session_2_final/` |
| `results/archive/superseded/` | **ALREADY ARCHIVED** | ✅ Correctly archived, keep as-is |
| `rv_toolkit/rv_toolkit/metrics.py` | **CONTRACT VIOLATION** | Computes single-layer PR only, not PR_late/PR_early ratio. Should be fixed or deprecated. |
| `results/canonical/n80_results.json` | **INCOMPLETE** | Missing controls, no hardware_info, no prompt_bank_version |
| `results/canonical/rv_ratio_results.json` | **INCOMPLETE** | Missing controls, no hardware_info |

**Total ARCHIVE_ONLY Items: 8**

---

## 4. Top 5 ROI Experiments

### Rank 1: Multi-Token R_V → Behavior Bridge
**Current State:**
- File: `results/phase1_cross_architecture/runs/20260202_133252_multi_token_bridge_mistral_7b_bridge/summary.json`
- n=40 prompts (20 recursive, 20 baseline)
- Shows R_V difference (d=-2.91) but weak within-group correlation
- Missing: hardware_info.json, n needs to be 80

**Gap to Bridge:**
- Increase n to 80 (40 recursive, 40 baseline)
- Add token-by-token R_V tracking during generation
- Add behavioral metrics (L4 markers, word count, coherence)
- Compute correlation: R_V during prompt → behavioral output

**Config Path:** `configs/phase3_bridge/gemma_2_9b/01_multi_token_bridge.json` (or create new Mistral version)

**Expected Outcome:** Bridge geometric contraction (R_V) to behavioral phase transition (L3→L4), completing the mechanistic-behavioral link.

**Priority: CRITICAL** - This is the central research question.

---

### Rank 2: Cross-Architecture Layer Sweep
**Current State:**
- Partial: 5 models validated at L27 only
- Missing: Systematic layer sweep (L20-L31) for each architecture

**Gap to Bridge:**
- Run layer sweep (L20-L31) for Mistral, Qwen, OPT, GPT-2 XL
- Verify L27 is peak effect across architectures
- Test early layer sensitivity (L3-L7)

**Config Path:** Create `configs/gold/03_layer_map_cross_arch.json` variants

**Expected Outcome:** Prove universal ~84% depth heuristic across architectures.

**Priority: HIGH** - Validates core finding.

---

### Rank 3: Confound Validation at High-N
**Current State:**
- File: `results/canonical/confound_validation/20251216_060911_confound_validation/summary.json`
- n=30 per condition (champions, length-matched, pseudo-recursive)
- Missing: hardware_info.json, needs n=50

**Gap to Bridge:**
- Increase to n=50 per condition
- Add hardware_info.json
- Verify all confounds ruled out at high-N

**Config Path:** `configs/gold/01_existence.json` (update n_champions/n_length_matched to 50)

**Expected Outcome:** Definitive proof that R_V contraction is recursion-specific, not confounded.

**Priority: HIGH** - Required for publication.

---

### Rank 4: Behavioral Grounding with Strict Gates
**Current State:**
- File: `results/discovery/behavioral_grounding/20251216_123737_behavior_strict/summary.json`
- n=20-30, has degeneracy gates
- Missing: wrong_layer control, hardware_info.json, needs n=50

**Gap to Bridge:**
- Increase n to 50
- Add wrong_layer control
- Add hardware_info.json
- Verify degeneracy gates work correctly

**Config Path:** `configs/gold/05_behavior_strict.json` (update n_pairs to 50, add wrong_layer)

**Expected Outcome:** Prove geometric intervention causes genuine behavioral change (not artifacts).

**Priority: MEDIUM** - Important but secondary to R_V→behavior bridge.

---

### Rank 5: MLP Ablation Necessity (High-N Validation)
**Current State:**
- Files: `results/RUN_INDEX.jsonl` shows MLP ablation runs with n=60
- Status: ✅ Already at high-N, but verify artifacts complete

**Gap to Bridge:**
- Verify all MLP ablation runs have complete artifacts
- Add hardware_info.json if missing
- Consolidate findings into single report

**Config Path:** Already exists: `configs/canonical/mlp_ablation_necessity_prompt_pass_l*.json`

**Expected Outcome:** Definitive proof of which MLP layers are necessary for R_V contraction.

**Priority: MEDIUM** - Already high-N, just needs artifact verification.

---

## 5. Claims vs Data Audit

| Claim Location | Claim | Data Location | Verification | Status |
|---------------|-------|---------------|--------------|--------|
| `RECOVERED_GOLD/MISTRAL_L27_CAUSAL_VALIDATION_COMPLETE.md` | "Cohen's d = -3.56" | `results/canonical/rv_l27_causal_validation/.../summary.json` | ✅ Verified: d=-3.558 (matches) | **VALID** |
| `RECOVERED_GOLD/MISTRAL_L27_CAUSAL_VALIDATION_COMPLETE.md` | "p < 10⁻⁶" | `results/canonical/rv_l27_causal_validation/.../summary.json` | ✅ Verified: p=2.75e-22 (matches) | **VALID** |
| `STATISTICAL_AUDIT_REPORT.md` | "Mistral-7B: d=-2.26" | `results/phase1_cross_architecture/.../mistral_7b/summary.json` | ✅ Verified: d=-2.259 (matches) | **VALID** |
| `STATISTICAL_AUDIT_REPORT.md` | "All 5 models significant (Holm-Bonferroni)" | `results/phase1_cross_architecture/runs/*/summary.json` | ✅ Verified: All p<0.05 after correction | **VALID** |
| `results/canonical/final_results.json` | "baseline_rv=5.279" | N/A | ❌ **CONTRACT VIOLATION**: This is single-layer PR, not R_V ratio | **INVALID** |
| `QUALITY_CONTROL_REPORT.md` | "rv_toolkit computes PR at single layer, not R_V ratio" | `rv_toolkit/rv_toolkit/metrics.py:133` | ✅ Verified: Returns `pr` only, not ratio | **VALID CLAIM** |
| `README.md` | "R_V = PR_late / PR_early" | `src/metrics/rv.py:164` | ✅ Verified: Correct implementation | **VALID** |
| `docs/misc/GOLD_STANDARD_SUITE.md` | "Pipeline 5 (Behavior) - Not implemented" | `src/pipelines/` | ✅ Verified: No `behavior_validation_strict.py` | **VALID CLAIM** |

**Summary:**
- **Valid Claims**: 7/8 (87.5%)
- **Invalid Claims**: 1/8 (12.5%) - `final_results.json` contract violation
- **Untraceable Claims**: 0/8 (0%)

---

## 6. Critical Gaps Summary

### Gap 1: Contract Violation - Single-Layer PR Mislabeled as R_V
**Severity: CRITICAL**

**Location:** `results/canonical/final_results.json`

**Issue:** Contains values like `baseline_rv=5.279`, `recursive_rv=6.733` which are single-layer PR values at L27, NOT R_V ratios (PR_late/PR_early).

**Evidence:**
- R_V ratios should be <1.0 for recursive prompts (contraction)
- Values >1.0 indicate expansion, contradicting core finding
- `src/metrics/rv.py` correctly implements ratio, but this file uses wrong metric

**Fix Required:**
1. Re-compute using `src/metrics/rv.py` (PR_late/PR_early)
2. Archive old `final_results.json` as `final_results_SINGLE_LAYER_PR.json` (mislabeled)
3. Create new `final_results.json` with correct R_V ratios

**Impact:** High - This file may be referenced in papers/documents.

---

### Gap 2: Missing Hardware Info in Run Artifacts
**Severity: HIGH**

**Issue:** No `hardware_info.json` in any run directory.

**Required Fields:**
```json
{
  "gpu_name": "NVIDIA L40S",
  "cuda_version": "12.1",
  "torch_version": "2.1.2",
  "torch_dtype": "float16",
  "device": "cuda"
}
```

**Fix Required:**
1. Add `get_hardware_info()` function to `src/utils/run_metadata.py`
2. Call in all pipeline functions
3. Save to `hardware_info.json` in run directory

**Impact:** Medium - Prevents hardware-specific reproducibility verification.

---

### Gap 3: rv_toolkit Contract Violation
**Severity: HIGH**

**Location:** `rv_toolkit/rv_toolkit/metrics.py:133`

**Issue:** `compute_rv()` returns single-layer PR, not PR_late/PR_early ratio.

**Fix Required:**
1. Option A: Fix `rv_toolkit` to compute ratio (requires early+late layer inputs)
2. Option B: Deprecate `rv_toolkit` and use `src/metrics/rv.py` exclusively
3. Option C: Rename function to `compute_pr()` to avoid confusion

**Impact:** High - `rv_toolkit` is publishable package, must be correct.

---

### Gap 4: Incomplete Artifacts in Legacy Results
**Severity: MEDIUM**

**Issue:** Many results in `results/canonical/` missing:
- `hardware_info.json`
- `prompt_bank_version.json`
- `metadata.json`
- Per-sample CSV files

**Fix Required:**
1. Archive incomplete results to `results/archive/incomplete/`
2. Re-run critical experiments with complete artifact generation
3. Document which results are "legacy" vs "industry-grade"

**Impact:** Medium - Reduces reproducibility but doesn't invalidate findings.

---

### Gap 5: Duplicate Results Directories
**Severity: LOW**

**Issue:** Multiple versions of same experiments:
- `results/canonical/session_2/`
- `results/canonical/session_2_complete/`
- `results/canonical/session_2_final/`

**Fix Required:**
1. Keep only `session_2_final/`
2. Archive others to `results/archive/duplicates/`
3. Document which version is canonical

**Impact:** Low - Confusing but doesn't affect science.

---

## 7. Recommendations

### Immediate Actions (This Week)

1. **Fix `final_results.json` contract violation**
   - Re-compute R_V using `src/metrics/rv.py`
   - Archive old version with clear label
   - Update any documentation referencing this file

2. **Add hardware_info.json to all new runs**
   - Implement `get_hardware_info()` function
   - Update all pipeline functions to log hardware info
   - Test on next run

3. **Fix or deprecate rv_toolkit**
   - Decide: fix to compute ratio, or deprecate
   - Update documentation accordingly
   - Add warning if keeping as-is

### Short-Term Actions (This Month)

4. **Ramp up multi-token bridge experiment**
   - Increase n to 80
   - Add token-by-token R_V tracking
   - Complete R_V→behavior correlation analysis

5. **Archive duplicate/incomplete results**
   - Move duplicates to `results/archive/duplicates/`
   - Move incomplete to `results/archive/incomplete/`
   - Document canonical versions

6. **Complete cross-architecture layer sweep**
   - Run L20-L31 sweep for 3-4 architectures
   - Verify L27 peak effect
   - Document layer selection heuristic

### Long-Term Actions (Next Quarter)

7. **Implement Pipeline 5 (Behavior Strict)**
   - Complete `behavior_validation_strict.py`
   - Add degeneracy gates
   - Validate with high-N

8. **Create comprehensive replication protocol**
   - Document hardware requirements
   - Create step-by-step replication guide
   - Test on fresh environment

9. **Automated artifact validation**
   - Add pre-commit hook to verify artifacts
   - Add CI/CD check for artifact completeness
   - Fail builds if artifacts missing

---

## 8. Conclusion

The `mech-interp-latent-lab-phase1` repository contains **strong causal validation work** with proper controls and statistical rigor. However, **contract violations** (single-layer PR mislabeled as R_V) and **missing artifacts** (hardware info) prevent full industry-standard compliance.

**Key Strengths:**
- ✅ Causal validation with perfect control separation (d=-3.56, p<1e-6)
- ✅ Cross-architecture replication (5 models, all significant)
- ✅ Correct R_V implementation in `src/metrics/rv.py`
- ✅ Config-driven experiments with good structure

**Key Weaknesses:**
- ❌ Contract violation in `final_results.json` (single-layer PR)
- ❌ Missing hardware_info.json in all runs
- ❌ rv_toolkit computes single-layer PR, not ratio
- ❌ Some duplicate/incomplete results

**Overall Verdict:** **7.5/10** - Strong science, needs artifact cleanup and contract fixes.

**Recommendation:** Fix critical contract violations immediately, then proceed with ramp-up experiments. The core findings are sound and publication-ready after artifact fixes.

---

**Audit Completed:** 2026-02-05  
**Next Review:** After contract violation fixes (estimated 1 week)
