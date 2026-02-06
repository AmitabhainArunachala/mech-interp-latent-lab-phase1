# Signal Core Index (Target: 100–200 files)

Purpose: define the minimal, high-signal set to preserve and keep clean.
This list is the authoritative “signal core” for audits and future work.

## 1) Top 10 Meta Files (read first)
See `META_TOP10_INDEX.md`.

## 2) Canonical Code (gold standard)
- `CANONICAL_CODE/causal_loop_closure_v2.py`
- `CANONICAL_CODE/mistral_L27_FULL_VALIDATION.py`
- `CANONICAL_CODE/n300_mistral_test_prompt_bank.py`

## 3) Core Prompt Infrastructure
- `prompts/bank.json`
- `prompts/loader.py`
- `prompts/README.md`

## 4) Core Metrics + Pipelines
- `src/metrics/rv.py`
- `src/metrics/baseline_suite.py`
- `src/metrics/behavioral_bridge.py`
- `src/metrics/behavior_strict.py`
- `src/metrics/mode_score.py`
- `src/metrics/logit_diff.py`
- `src/metrics/logit_lens.py`
- `src/metrics/extended.py`
- `src/metrics/behavior_states.py`
- `src/metrics/__init__.py`

- `src/core/models.py`
- `src/core/patching.py`
- `src/core/hooks.py`
- `src/core/experiment_io.py`
- `src/core/head_specific_patching.py`
- `src/core/model_physics.py`
- `src/core/logit_capture.py`
- `src/core/utils.py`
- `src/core/__init__.py`

- `src/pipelines/run.py`
- `src/pipelines/registry.py`
- `src/pipelines/canonical/__init__.py`
- `src/pipelines/canonical/rv_l27_causal_validation.py`
- `src/pipelines/canonical/confound_validation.py`
- `src/pipelines/canonical/multi_token_bridge.py`
- `src/pipelines/canonical/random_direction_control.py`
- `src/pipelines/canonical/head_ablation_validation.py`
- `src/pipelines/canonical/mlp_sufficiency_test.py`
- `src/pipelines/canonical/mlp_combined_sufficiency_test.py`
- `src/pipelines/canonical/mlp_ablation_necessity.py`
- `src/pipelines/canonical/mlp_ablation_necessity_prompt_pass.py`

## 5) Gold Configs (all)
- `configs/gold/*.json` (33 files)

## 6) Canonical Results (high-signal runs)

### Causal Validation (Mistral, canonical)
- `results/canonical/rv_l27_causal_validation/20251216_060955_rv_l27_causal_validation/config.json`
- `results/canonical/rv_l27_causal_validation/20251216_060955_rv_l27_causal_validation/summary.json`
- `results/canonical/rv_l27_causal_validation/20251216_060955_rv_l27_causal_validation/report.md`
- `results/canonical/rv_l27_causal_validation/20251216_061127_rv_l27_causal_validation/config.json`
- `results/canonical/rv_l27_causal_validation/20251216_061127_rv_l27_causal_validation/summary.json`
- `results/canonical/rv_l27_causal_validation/20251216_061127_rv_l27_causal_validation/report.md`

### Confound Validation (canonical)
- `results/canonical/confound_validation/20251216_060911_confound_validation/config.json`
- `results/canonical/confound_validation/20251216_060911_confound_validation/summary.json`
- `results/canonical/confound_validation/20251216_060911_confound_validation/report.md`

### Multi-Token Bridge (canonical v2)
- `results/canonical/multi_token_bridge_v2/config.json`
- `results/canonical/multi_token_bridge_v2/summary.json`
- `results/canonical/multi_token_bridge_v2/report.md`
- `results/canonical/multi_token_bridge_v2/VERDICT.md`
- `results/canonical/multi_token_bridge_v2/metadata.json`
- `results/canonical/multi_token_bridge_v2/prompt_bank_version.txt`

### Cross-Architecture (high-signal)
- `results/phase1_cross_architecture/runs/20260202_121604_rv_l27_causal_validation_mistral_7b/summary.json`
- `results/phase1_cross_architecture/runs/20260202_121604_rv_l27_causal_validation_mistral_7b/report.md`
- `results/phase1_cross_architecture/runs/20260202_121604_rv_l27_causal_validation_mistral_7b/config.json`
- `results/phase1_cross_architecture/runs/20260202_125718_rv_l27_causal_validation_opt_6_7b/summary.json`
- `results/phase1_cross_architecture/runs/20260202_125718_rv_l27_causal_validation_opt_6_7b/report.md`
- `results/phase1_cross_architecture/runs/20260202_125718_rv_l27_causal_validation_opt_6_7b/config.json`
- `results/phase1_cross_architecture/runs/20260202_130807_rv_l27_causal_validation_gpt2_xl/summary.json`
- `results/phase1_cross_architecture/runs/20260202_130807_rv_l27_causal_validation_gpt2_xl/report.md`
- `results/phase1_cross_architecture/runs/20260202_130807_rv_l27_causal_validation_gpt2_xl/config.json`

### Bridge Run Artifacts (canonical target run)
- `results/phase1_cross_architecture/runs/20260205_104737_multi_token_bridge_mistral_7b_bridge_truncfix/config.json`
- `results/phase1_cross_architecture/runs/20260205_104737_multi_token_bridge_mistral_7b_bridge_truncfix/summary.json`
- `results/phase1_cross_architecture/runs/20260205_104737_multi_token_bridge_mistral_7b_bridge_truncfix/report.md`

### Discovery: High-N Path Patching (preserve)
- `results/phase1_mechanism/runs/20251213_080454_path_patching_mechanism_full_early_layer_sweep_full_controls_base/summary.json`
- `results/phase1_mechanism/runs/20251213_090121_path_patching_mechanism_early_layers_deep_base/summary.json`
- `results/phase1_mechanism/runs/20251213_073754_path_patching_mechanism_full_early_layer_sweep_base/summary.json`
- `results/phase1_mechanism/runs/20251213_064141_path_patching_mechanism_layer_sweep_base/summary.json`

### Discovery: Behavioral Grounding (preserve)
- `results/discovery/behavioral_grounding/20251213_124735_behavioral_grounding_batch_ministral8b_n100_L24_27_W32_sampled_v1/summary.json`

### Gold Standard Runs (preserve)
- `results/gold_standard/runs/20251216_060955_rv_l27_causal_validation/summary.json`
- `results/gold_standard/runs/20251216_061127_rv_l27_causal_validation/summary.json`
- `results/gold_standard/runs/20251216_060911_confound_validation/summary.json`

## 7) Paper Core
- `R_V_PAPER/research/PHASE1_FINAL_REPORT.md`

## 8) Audit & Standards Core
- `docs/standards/MEASUREMENT_CONTRACT.md`
- `UNIFIED_AUDITOR_INTEGRATION.md`
- `QUALITY_CONTROL_REPORT.md`
- `REPRODUCIBILITY_AUDIT_REPORT.md`
- `STATISTICAL_AUDIT_REPORT.md`
- `STATISTICAL_AUDIT_EXECUTIVE_SUMMARY.md`
- `PUBLICATION_BLOCKERS_STATUS.md`
- `BRIDGE_STATUS_SUMMARY.md`
- `BRIDGE_HYPOTHESIS_INVESTIGATION.md`
- `ARCHITECTURE_EXECUTIVE_SUMMARY.md`
- `ARCHITECTURE_REVIEW_INDEX.md`
- `AGENT_ONBOARDING.md`
- `docs/status/RESEARCH_PROGRESS_SUMMARY.md`
- `RUN_INDEX.md`

## 9) Deprecation / Contract Notices
- `results/canonical/FINAL_RESULTS_DEPRECATED.md`
- `results/archive/CONTRACT_VIOLATIONS.md`
- `results/canonical/multi_token_bridge/CANONICAL_POINTER_NOTICE.md`

---

## Notes
- This index is a **preservation list**, not a move list.
- Any file outside this core is eligible for archive, but only after verification.
