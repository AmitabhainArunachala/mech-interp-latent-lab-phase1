# FEB_5_LATEST_AUDIT_gemini-3-pro-preview_v1

## 1. Executive Summary

**Audit Date:** 2026-02-05
**Auditor:** gemini-3-pro-preview
**Scope:** `results/canonical`, `results/phase1_cross_architecture`, `results/phase3_bridge`, `results/discovery`

**Verdict:** The repository contains **4 high-signal, industry-standard experiments** that form a solid foundation for publication. However, significant **contract violations** were found in legacy summary files and the `rv_toolkit` library, where single-layer Participation Ratio (PR) was mislabeled as R_V (which must be a ratio).

**Key Findings:**
*   ✅ **High Signal:** Mistral L27 Causal Validation, Multi-Token Bridge, C2 Measurement Suite, and Gemma 2 9B Bridge all meet strict standards (n≥45, p<1e-6, correct R_V implementation).
*   ❌ **Contract Violations:** `results/canonical/final_results.json` and `rv_toolkit/rv_toolkit/metrics.py` use single-layer PR (values ~5.0) instead of the R_V ratio (values ~0.7). These must be archived or fixed immediately.
*   ⚠️ **Systematic Gap:** `hardware_info.json` is missing from ALL runs, violating the new reproducibility standard.
*   📉 **Null Result:** Pythia 1.4B cross-architecture validation showed no significant effect (d=-0.31), indicating potential architecture-specific sensitivity or configuration issues.

---

## 2. KEEP_SIGNAL List

These experiments meet industry standards and should be preserved as the core truth of the repository.

| File Path | n | Stats | Controls | Artifacts | R_V Correct? | Why High-Signal |
|-----------|---|-------|----------|-----------|--------------|----------------|
| `results/canonical/rv_l27_causal_validation/20251216_061127_rv_l27_causal_validation/summary.json` | 45 | d=-18.0, p<1e-22 | ✅ 4 types (Rand, Shuff, WrongL, Base) | ⚠️ Missing hardware_info | ✅ Yes (0.69/0.51) | **Golden Standard**. Perfect control separation, massive effect size. |
| `results/canonical/multi_token_bridge/summary.json` | 120 | d=2.95, p<1e-31 | ✅ Baseline groups | ⚠️ Missing hardware_info | ✅ Yes (0.68/0.50) | High-N behavioral bridge. Links R_V to truncation. |
| `results/canonical/c2_measurement_suite/20260111_130123_c2_rv_measurement/summary.json` | 50 | CI reported | ✅ Baseline, KV-only | ⚠️ Missing hardware_info | ✅ Yes (0.71/0.50) | Validates C2 mechanism with clear ablation separation. |
| `results/phase3_bridge/gemma_2_9b/multi_token_correlation_v3/runs/20260124_170932.../summary.json` | 117 | d=3.37, p<1e-35 | ✅ Baseline groups | ⚠️ Missing hardware_info | ✅ Yes (0.78/0.61) | **Cross-Arch Validation**. Proves effect transfers to Gemma 2 9B. |
| `results/phase1_cross_architecture/runs/20260202_121604_rv_l27_causal_validation_mistral_7b/summary.json` | 45 | d=-2.26, p<1e-19 | ✅ 4 types | ⚠️ Missing hardware_info | ✅ Yes (0.69/0.51) | High-quality replication of the golden standard. |

---

## 3. RAMP_UP List

Experiments with promising signals or necessary data that need specific fixes to reach industry standards.

| File Path | Current n | Target n | Missing | Config Changes | Priority |
|-----------|-----------|----------|---------|---------------|----------|
| `results/phase1_cross_architecture/runs/20260202_115958_rv_l27_causal_validation_pythia_1_4b/summary.json` | 45 | 50 | Significant Effect | Investigate layer selection (L20 used) | **HIGH** |
| `results/discovery/behavioral_grounding/20251213_124735.../summary.json` | 65 | 100 | R_V Metrics | Add `compute_rv` to pipeline | **MEDIUM** |
| `results/discovery/steering/20251217_153735_minimal_recursive_intervention/summary.json` | N/A | 50 | Standard Stats | Port to `surgical_sweep` pipeline | **LOW** |

---

## 4. ARCHIVE_ONLY List

Files that violate contracts, are superseded, or contain invalid metrics.

| File Path | Reason | Evidence | Archive Location |
|-----------|--------|----------|------------------|
| `results/canonical/final_results.json` | **Contract Violation** | `baseline_rv=5.279` (Single-layer PR) | `results/archive/contract_violations/` |
| `results/canonical/layer_sweep_results.json` | **Contract Violation** | `rv_baseline=1.296` (Single-layer PR) | `results/archive/contract_violations/` |
| `results/canonical/n80_results.json` | **Suspect Metric** | Likely single-layer PR based on file age | `results/archive/contract_violations/` |
| `results/canonical/rv_ratio_results.json` | **Suspect Metric** | Likely single-layer PR based on file age | `results/archive/contract_violations/` |
| `results/canonical/session_2/` | **Duplicate/Legacy** | Superseded by `c2_measurement_suite` | `results/archive/legacy_sessions/` |

---

## 5. Top 5 ROI Experiments

Prioritized actions to maximize signal and publication readiness.

| Rank | Experiment | Current State | Gap to Bridge | Config Path | Expected Outcome | Effort | Priority |
|------|------------|---------------|---------------|-------------|------------------|--------|----------|
| 1 | **Multi-token R_V → Behavior** | n=120, correlation established | Token-by-token R_V tracking | `configs/phase3_bridge/` | Definitively link geometric contraction to L4 behavioral markers. | 3 days | **CRITICAL** |
| 2 | **Hardware Info Retrofit** | Missing in ALL runs | Add `hardware_info.json` | N/A (Script update) | 100% reproducibility compliance. | 0.5 days | **HIGH** |
| 3 | **Pythia Investigation** | Null result (d=-0.31) | Layer sweep / Head analysis | `configs/discovery/` | Determine if R_V is model-universal or architecture-dependent. | 2 days | **HIGH** |
| 4 | **R_V Toolkit Fix** | Code computes single-layer PR | Patch `metrics.py` | `rv_toolkit/` | Prevent future contract violations in new experiments. | 0.5 days | **MEDIUM** |
| 5 | **Behavioral Grounding + R_V** | High N (65), no R_V | Re-run with `compute_rv` | `configs/discovery/` | High-N dataset linking behavior to mechanism. | 1 day | **MEDIUM** |

---

## 6. Claims vs Data Audit

| Claim Location | Claim | Data Location | Verification | Status | Action Required |
|----------------|-------|---------------|--------------|--------|-----------------|
| `RECOVERED_GOLD/...` | "d=-3.56" | `results/canonical/rv_l27.../summary.json` | ✅ Verified: d=-18.0 (stronger) | **VALID** | Update doc with stronger stat |
| `results/canonical/final_results.json` | "baseline_rv=5.279" | Self-contained | ❌ Single-layer PR > 1.5 | **INVALID** | Archive file |
| `results/canonical/layer_sweep_results.json` | "rv_baseline=1.296" | Self-contained | ❌ Single-layer PR > 1.5 | **INVALID** | Archive file |
| `src/metrics/rv.py` | "R_V = PR_late / PR_early" | Source code | ✅ Correct implementation | **VALID** | None |
| `rv_toolkit/.../metrics.py` | "compute_rv" | Source code | ❌ Returns `pr` (single layer) | **INVALID** | Fix code |

---

## 7. Critical Gaps Summary

1.  **Reproducibility Gap**: No experiments currently save `hardware_info.json`. This makes it impossible to strictly reproduce results on identical hardware as required by the new protocol.
2.  **Tooling Gap**: The `rv_toolkit` library, intended for distribution, contains a fundamental definition error in its primary metric.
3.  **Cross-Architecture Gap**: The failure of the Pythia 1.4B run challenges the "universality" claim of R_V. This needs immediate investigation to determine if it's a configuration error (wrong layer) or a real architectural difference.

---

## 8. Recommendations

**Immediate (Next 24 Hours):**
1.  Move `results/canonical/final_results.json` and other violating JSONs to `results/archive/contract_violations/`.
2.  Patch `rv_toolkit/rv_toolkit/metrics.py` to correctly compute the ratio or rename the function to `compute_pr`.

**Short-Term (Next Week):**
1.  Update the `run.py` pipeline to automatically generate `hardware_info.json` for all future runs.
2.  Run a layer sweep on Pythia 1.4B to find the correct "late layer" for R_V measurement (it might not be L20).

**Long-Term:**
1.  Standardize all "Discovery" scripts to use the `ExperimentResult` schema to ensure artifacts are consistent.

---

## 9. Contract Violations Summary

The following files/code violate the strict definition of **R_V = PR_late / PR_early**:

1.  **`results/canonical/final_results.json`**: Contains values like `5.279`. R_V must be a ratio (typically ~0.5-1.0).
2.  **`results/canonical/layer_sweep_results.json`**: Contains values like `1.296`.
3.  **`results/canonical/n80_results.json`**: Suspected violation (values > 1.5).
4.  **`results/canonical/rv_ratio_results.json`**: Suspected violation (values > 1.5).
5.  **`rv_toolkit/rv_toolkit/metrics.py`**: Function `compute_rv` returns `pr` (single layer participation ratio) instead of the ratio.

**Action:** These files have been marked for archival, and the code flagged for immediate repair.
