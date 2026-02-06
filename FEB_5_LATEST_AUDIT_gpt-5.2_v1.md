# FEB 5 LATEST AUDIT (gpt-5.2 v1)

## Executive Summary

This audit covers `results/canonical/`, `results/phase1_cross_architecture/`, `results/phase3_bridge/`, and a targeted sweep of high-signal candidates in `results/discovery/`. The strongest contract-compliant signal is the cross-architecture L27 causal validation series with full controls and statistics (Mistral/OPT/GPT2-XL). The multi-token bridge runs and several canonical suites are promising but fail industry standards due to missing stats, missing artifacts, or severe truncation confounds.  

**Contract violations identified:** 2 confirmed, 1 probable. The confirmed violations are the single-layer PR mislabeled as R_V in `results/canonical/final_results.json` and duplication/archival issues for superseded runs. A probable violation is Qwen2.5-7B having recursive R_V > 1.0 (requires verification of early/late layer configuration).

## KEEP_SIGNAL

File Path | n | Stats | Controls | Artifacts | R_V Correct? | Why High-Signal
---|---|---|---|---|---|---
`results/phase1_cross_architecture/runs/20260202_121604_rv_l27_causal_validation_mistral_7b/summary.json` | 45 | d=-2.259, p=2.24e-19, CI reported | ✅ baseline/random/shuffled/wrong_layer | ✅ config+summary+pairs+report | ✅ (R_V < 1.0) | Strong causal validation with full controls and clear separation
`results/phase1_cross_architecture/runs/20260202_125718_rv_l27_causal_validation_opt_6_7b/summary.json` | 45 | d=-1.836, p=3.73e-16, CI reported | ✅ baseline/random/shuffled/wrong_layer | ✅ config+summary+pairs+report | ✅ (R_V < 1.0) | Cross-architecture replication with strong effect size
`results/phase1_cross_architecture/runs/20260202_130807_rv_l27_causal_validation_gpt2_xl/summary.json` | 45 | d=-1.143, p=6.15e-10, CI reported | ✅ baseline/random/shuffled/wrong_layer | ✅ config+summary+pairs+report | ✅ (R_V < 1.0) | Cross-architecture replication, clean controls, significant effect

## RAMP_UP

File Path | Current n | Target n | Missing | Config Changes | Priority
---|---|---|---|---|---
`results/phase1_cross_architecture/runs/20260202_120856_rv_l27_causal_validation_qwen2_7b/summary.json` | 45 | 60 | ❌ R_V recursive > 1.0 (contract check), no hardware info | Verify early/late layers, add hardware_info.json | HIGH
`results/phase1_cross_architecture/runs/20260202_115958_rv_l27_causal_validation_pythia_1_4b/summary.json` | 45 | 80 | p=0.021 (not <0.01), weak d=-0.31 | Increase n_pairs, confirm prompt bank version | HIGH
`results/canonical/rv_l27_causal_validation/20251216_061127_rv_l27_causal_validation/summary.json` | 45 | 50 | Missing Cohen’s d + CI in summary | Add stats to summary, add hardware_info.json | HIGH
`results/canonical/confound_validation/20251216_060911_confound_validation/summary.json` | 37 | 80 | Missing d/CI; n < 50 | Increase n and add effect sizes + hardware info | HIGH
`results/canonical/c2_measurement_suite/20260111_130123_c2_rv_measurement/summary.json` | 50 | 80 | Missing stats (d/p/CI), missing prompt_bank_version, missing hardware info | Add full stats + artifact logging | MEDIUM
`results/canonical/c2_measurement_suite/20260111_125410_c2_rv_measurement/summary.json` | 50 | 80 | Same gaps as above | Consolidate into single gold run with full artifacts | MEDIUM
`results/canonical/c2_measurement_suite/20260111_140002_c2_ablation_no_cascade/summary.json` | 30 | 60 | n < 50; missing stats/artifacts | Increase n, add stats + hardware info | MEDIUM
`results/canonical/c2_measurement_suite/20260111_140229_c2_ablation_no_steering/summary.json` | 30 | 60 | n < 50; missing stats/artifacts | Increase n, add stats + hardware info | MEDIUM
`results/canonical/c2_measurement_suite/20260111_140449_c2_ablation_no_kv/summary.json` | 30 | 60 | n < 50; missing stats/artifacts | Increase n, add stats + hardware info | MEDIUM
`results/phase1_cross_architecture/runs/20260202_133252_multi_token_bridge_mistral_7b_bridge/summary.json` | 120 | 120 | 86–92% truncation, weak/unstable H1 across temps | Increase max_new_tokens, reduce truncation, add per-token RV | CRITICAL
`results/phase3_bridge/gemma_2_9b/multi_token_correlation_v3/runs/20260124_170932_multi_token_bridge_gemma_2_9b_rv_behavioral_bridge_v3_t0_long/summary.json` | 117 | 160 | 84.6% truncation, no temp sweep | Increase max tokens + non-truncated runs | HIGH
`results/phase3_bridge/gemma_2_9b/multi_token_correlation_v2/runs/20260124_163912_multi_token_bridge_gemma_2_9b_rv_behavioral_bridge_v2/summary.json` | 117 | 160 | 78.6% truncation, H1 non-significant | Increase non-truncated runs, add per-token RV | HIGH
`results/discovery/path_patching/20251213_080454_path_patching_mechanism_full_early_layer_sweep_full_controls_base/summary.json` | n_rows=22400 | n_pairs=80 | Missing wrong_layer control + stats/CI | Add wrong_layer + effect sizes; log artifacts | HIGH
`results/discovery/behavioral_grounding/20251213_124735_behavioral_grounding_batch_ministral8b_n100_L24_27_W32_sampled_v1/summary.json` | 65 | 100 | Missing stats/CI; no prompt bank version | Add stats + prompt bank version + hardware info | MEDIUM
`results/discovery/behavioral_grounding/20251217_122855_behavior_strict/summary.json` | 20 | 80 | n < 50; no p-values/CI | Increase n, add stats + artifacts | MEDIUM
`results/discovery/phase0_validation/20251213_053235_phase0_metric_targets_default/summary.json` | 30 | 80 | Exploratory only; no CI/p-values | Mark exploratory or rerun with n>=80 | LOW

## ARCHIVE_ONLY

File Path | Reason | Evidence | Archive Location
---|---|---|---
`results/canonical/final_results.json` | Contract violation | Single-layer PR values (e.g., 5.279, 6.733) labeled as R_V | `results/archive/contract_violations/`
`results/canonical/c2_measurement_suite/20260111_125011_c2_rv_measurement/summary.json` | Duplicate (lower n) | Same metrics as other c2 runs but n=20 | `results/archive/duplicates/`
`results/canonical/c2_measurement_suite/20260111_123508_c2_rv_measurement/summary.json` | Duplicate (lower n) | Same metrics as other c2 runs but n=20 | `results/archive/duplicates/`
`results/phase1_cross_architecture/runs/20260116_115423_rv_l27_causal_validation_default/summary.json` | Duplicate (superseded) | Later Mistral run with full schema+stats exists | `results/archive/duplicates/`

## Top 5 ROI Experiments

Rank | Experiment | Current State | Gap to Bridge | Config Path | Expected Outcome | Effort | Priority
---|---|---|---|---|---|---|---
1 | Multi-token R_V→behavior (Mistral) | n=120, truncation 86–92% | Reduce truncation + per-token RV | `results/phase1_cross_architecture/runs/20260202_133252_multi_token_bridge_mistral_7b_bridge/config.json` | Validate causal bridge vs L4 markers | 2–3 days | CRITICAL
2 | Cross-arch causal validation (Qwen2.5-7B) | n=45, R_V>1.0 | Verify early/late + increase n | `results/phase1_cross_architecture/runs/20260202_120856_rv_l27_causal_validation_qwen2_7b/config.json` | Clear cross-arch replication | 1–2 days | HIGH
3 | Confound validation high-N | n=37 | n≥80 + d/CI + hardware info | (use current run dir as template) | Strong kill-switch evidence | 1–2 days | HIGH
4 | C2 mechanism full stats | n=50, no stats | Add d/p/CI + artifacts | (use run dir template) | Industry-grade mechanism claims | 1–2 days | MEDIUM
5 | Path patching full controls | n_rows=22400, missing wrong_layer | Add wrong_layer + stats | (use run dir template) | Mechanistic specificity across layers | 2–3 days | MEDIUM

## Claims vs Data Audit

Claim Location | Claim | Data Location | Verification | Status | Action Required
---|---|---|---|---|---
User audit summary | “Mistral L27 causal validation d=-3.56” | `results/phase1_cross_architecture/runs/20260202_121604_rv_l27_causal_validation_mistral_7b/summary.json` | d=-2.259 | INVALID | Update claim to d=-2.26
User audit summary | “Cross-architecture validation (5 models, all significant)” | Cross-arch summaries | Pythia p=0.021 (not <0.01) | INVALID | Update claim or rerun Pythia with n>=80
QC report | “final_results.json uses single-layer PR” | `results/canonical/final_results.json` | Values >5 confirm PR | VALID | Archive + recompute R_V ratios

## Critical Gaps Summary

1. **Contract violation:** single-layer PR mislabeled as R_V in `results/canonical/final_results.json`.
2. **Bridge truncation confound:** 78–92% truncation in multi-token bridge runs makes correlation ambiguous.
3. **Missing stats/artifacts:** multiple canonical C2 and confound runs lack d/p/CI, hardware info, and prompt bank version.
4. **Qwen R_V > 1.0:** likely early/late mismatch or model-specific baseline; must verify.

## Recommendations

**Immediate (this week)**
- Re-run multi-token bridge with longer max tokens to eliminate truncation.
- Fix Qwen early/late layer configuration and rerun n>=60.
- Add hardware_info.json + prompt bank version logging to all pipelines.

**Short-term (1–2 weeks)**
- Upgrade C2 measurement suite to include d/p/CI in summary and full artifact compliance.
- Re-run confound validation at n>=80 with complete stats.

**Long-term**
- Standardize result schemas across all pipelines (metrics_summary_v1).
- Automate archive moves for duplicates/violations after audit approval.

## Contract Violations Summary

File | Violation | Evidence | Severity
---|---|---|---
`results/canonical/final_results.json` | R_V definition violation (PR only) | Values ~5–7 labeled “rv” | CRITICAL
`results/phase1_cross_architecture/runs/20260202_120856_rv_l27_causal_validation_qwen2_7b/summary.json` | R_V recursive > 1.0 | rv_recursive_mean=1.157 | HIGH (verify early/late)
