# Config Categorization

**Created**: 2026-01-11
**Purpose**: Authoritative categorization of 54 config files
**Structure**: `configs/{canonical,discovery,archive}/`

---

## Summary

| Category | Count | Purpose |
|----------|-------|---------|
| Canonical | 14 | Drive 7 canonical pipelines |
| Discovery | 27 | Drive 12 discovery pipelines |
| Archive | 13 | Historical/orphan configs |
| **Total** | **54** | |

---

## Canonical Configs (14 files)

These configs drive the core paper findings.

| Config | Pipeline | Purpose |
|--------|----------|---------|
| `rv_l27_causal_validation.json` | rv_l27_causal_validation | L27 causal validation |
| `confound_validation.json` | confound_validation | 4-pillar control |
| `random_direction_control.json` | random_direction_control | Base random control |
| `random_direction_control_l3.json` | random_direction_control | L3 variant |
| `random_direction_control_l3_targeted.json` | random_direction_control | L3 targeted |
| `random_direction_control_l4.json` | random_direction_control | L4 variant |
| `mlp_ablation_necessity_l0.json` | mlp_ablation_necessity | L0 necessity |
| `mlp_ablation_necessity_l1.json` | mlp_ablation_necessity | L1 necessity |
| `mlp_ablation_necessity_l2.json` | mlp_ablation_necessity | L2 necessity |
| `mlp_ablation_necessity_l3.json` | mlp_ablation_necessity | L3 necessity |
| `mlp_sufficiency_l0.json` | mlp_sufficiency_test | L0 sufficiency |
| `combined_mlp_sufficiency_l0_l1.json` | combined_mlp_sufficiency_test | L0+L1 |
| `combined_mlp_sufficiency_l0_l1_l3.json` | combined_mlp_sufficiency_test | L0+L1+L3 |
| `combined_mlp_sufficiency_l0_l1_l18_l19_l20.json` | combined_mlp_sufficiency_test | Extended |

---

## Discovery Configs (27 files)

These configs drive methodology tools for new model exploration.

### C2 Measurement (4 configs)
- `c2_rv_measurement.json`
- `c2_ablation_no_cascade.json`
- `c2_ablation_no_kv.json`
- `c2_ablation_no_steering.json`

### Behavioral Grounding (7 configs)
- `behavioral_grounding.json`
- `behavioral_grounding_override.json`
- `behavioral_grounding_override_ministral8b.json`
- `behavioral_grounding_override_mixtral.json`
- `behavioral_grounding_override_nemo12b.json`
- `behavioral_grounding_n100_ministral_collapse_layers.json`
- `behavioral_grounding_batch_ministral8b_n100_L24_27_W32.json`

### Path Patching (9 configs)
- `path_patching_mechanism.json`
- `path_patching_mechanism_early_layers_deep.json`
- `path_patching_mechanism_full_early_layer_sweep.json`
- `path_patching_mechanism_full_early_layer_sweep_full_controls.json`
- `path_patching_mechanism_layer_sweep.json`
- `path_patching_mechanism_math_source_control.json`
- `path_patching_mechanism_peak_sweep_strong.json`
- `path_patching_mechanism_stress.json`
- `path_patching_full_controls_sweep.json`

### KV Mechanism (4 configs)
- `kv_sweep_l0_l8.json`
- `kv_sweep_l8_l16.json`
- `kv_sweep_l16_l24.json`
- `kv_sweep_l24_l32.json`

### Other Discovery (3 configs)
- `logit_lens_analysis.json`
- `vproj_patching_analysis.json`
- `mlp_vproj_combined_sufficiency.json`

---

## Archive Configs (13 files)

Historical configs for archived pipelines.

| Config | Original Pipeline | Status |
|--------|-------------------|--------|
| `phase0_minimal_pairs.json` | phase0_minimal_pairs | Superseded |
| `phase0_metric_targets.json` | phase0_metric_targets | Superseded |
| `phase1_existence.json` | phase1_existence | Superseded |
| `phase1_existence_ministral8b_instruct.json` | phase1_existence | Superseded |
| `l27_head_analysis.json` | l27_head_analysis | Merged |
| `kv_sufficiency_matrix.json` | kv_sufficiency_matrix | Exploratory |
| `hysteresis_patching.json` | hysteresis_patching | Superseded |
| `mlp_steering_sweep.json` | mlp_steering_sweep | Superseded |
| `mlp_steering_sweep_full.json` | mlp_steering_sweep | Superseded |
| `mlp_steering_sweep_corrected.json` | mlp_steering_sweep | Superseded |
| `mlp_steering_l2_l5_corrected.json` | mlp_steering_sweep | Superseded |
| `mlp_steering_alpha_sweep.json` | mlp_steering_sweep | Superseded |
| `position_specific_l0_ablation.json` | position_specific_ablation | Superseded |

---

*Categorization complete. All 54 configs accounted for.*
