# Pipeline Analysis

**Date**: 2026-01-11
**Phase**: 1 - Pipeline Categorization
**Total Files**: 59 in `src/pipelines/`

## Decision Criteria

- **Canonical**: Required to reproduce paper findings (core causal validations)
- **Discovery**: Methodology tools useful for exploring new models
- **Archive**: Dead ends, superseded code, exploratory experiments

**Key Question**: "If we started on Llama-7B tomorrow, would we need this code?"

---

## Infrastructure (Keep in Place)

| File | Purpose | Decision |
|------|---------|----------|
| `registry.py` | Experiment dispatch, config validation, ExperimentResult contract | **INFRASTRUCTURE** |
| `run.py` | CLI entry point, config-driven execution | **INFRASTRUCTURE** |
| `__init__.py` | Module exports | **INFRASTRUCTURE** |

---

## Canonical Pipelines (7 files)

These reproduce the paper's core findings.

| File | Purpose | Key Result | Notes |
|------|---------|------------|-------|
| `rv_l27_causal_validation.py` | L27 V-proj activation patching | d=-3.56, p<10⁻⁶, n=45 | **STRONGEST RESULT** |
| `confound_validation.py` | 4-pillar control validation | Rules out confounds | Champions vs controls |
| `random_direction_control.py` | Random direction baseline | Proves direction specificity | Not "any perturbation" |
| `mlp_ablation_necessity.py` | L0 MLP necessity test | Tests if ablation kills effect | Causal necessity |
| `mlp_sufficiency_test.py` | L0 MLP sufficiency test | Tests if patching induces effect | Causal sufficiency |
| `mlp_combined_sufficiency_test.py` | L0+L1+L3 combined test | Multi-layer sufficiency | Combined MLP |
| `head_ablation_validation.py` | KV-head ablation validation | H18/H26 specificity | Gold standard pipeline 4 |

---

## Discovery Pipelines (12 files)

Methodology tools for exploring circuits in new models.

| File | Purpose | When to Use |
|------|---------|-------------|
| `c2_rv_measurement.py` | C2 config geometry→behavior bridge | Measuring R_V during generation |
| `behavioral_grounding.py` | Geometry→behavior validation | Does patching change output? |
| `behavioral_grounding_batch.py` | Batch version of above | Large-scale behavior testing |
| `eigenstate_direction_finder.py` | PCA/SVD direction discovery | Finding steering directions |
| `logit_lens_analysis.py` | Logit lens integration | Token prediction analysis |
| `vproj_patching_analysis.py` | V_proj patching analysis | Head-level intervention |
| `mlp_vproj_combined_sufficiency_test.py` | MLP+V_proj combined | Joint sufficiency test |
| `path_patching_mechanism.py` | Path patching (Nanda-style) | Circuit tracing |
| `hysteresis.py` | Temporal stability testing | Persistence of effect |
| `temporal_stability.py` | Related stability testing | Effect over time |
| `kv_mechanism.py` | KV swap mechanism | KV-based interventions |
| `layer_sweep.py` | Layer-by-layer analysis | Finding critical layers |

---

## Archive Pipelines (35 files)

Historical/superseded/exploratory code.

### Early Phase Experiments (4 files)
| File | Reason |
|------|--------|
| `phase0_minimal_pairs.py` | Superseded by rv_l27_causal_validation |
| `phase0_metric_targets.py` | Early exploration |
| `phase1_existence.py` | Superseded by full validation |
| `mistral_L27_full_validation.py` | Merged into rv_l27_causal_validation |

### Steering Experiments (9 files)
| File | Reason |
|------|--------|
| `steering.py` | General steering - superseded by surgical_sweep |
| `steering_analysis.py` | Analysis companion - archive with steering |
| `steering_control.py` | Control variant |
| `steering_layer_matrix.py` | Matrix sweep - exploratory |
| `extended_context_steering.py` | Extended context - exploratory |
| `minimal_recursive_intervention.py` | Early minimal version |
| `mlp_steering_sweep.py` | MLP sweep - superseded |
| `sprint_head_specific_steering/pipeline.py` | Sprint experiment - exploratory |
| `p10_advanced_steering/pipeline.py` | Advanced steering - exploratory |

### Investigation Experiments (8 files)
| File | Reason |
|------|--------|
| `l27_deep_dive.py` | Deep dive - superseded by validation |
| `l27_head_analysis.py` | Head analysis - merged into head_ablation |
| `h31_investigation.py` | H31 investigation - dead end |
| `h31_ablation_causal.py` | H31 ablation - dead end |
| `anthropic_level_investigation.py` | Anthropic-style - exploratory |
| `comprehensive_circuit_analysis.py` | Comprehensive - too broad |
| `causal_mechanism_hunt.py` | Mechanism hunt - early exploration |
| `source_isolation_diagnostic.py` | Diagnostic - debugging |

### Sweep/Matrix Experiments (6 files)
| File | Reason |
|------|--------|
| `surgical_sweep.py` | Sweep experiment - exploratory |
| `verification_sweep.py` | Verification sweep - superseded |
| `importance_sweep.py` | Importance ranking - exploratory |
| `kv_sufficiency_matrix.py` | KV matrix - exploratory |
| `unified_layer_map.py` | Layer mapping - exploratory |
| `triple_system_intervention.py` | Triple intervention - complex |

### Utility/One-off Experiments (5 files)
| File | Reason |
|------|--------|
| `kitchen_sink.py` | "Ultimate stress test" - exploratory |
| `circuit_discovery.py` | Early discovery - superseded |
| `geometry_behavior.py` | Geometry bridge - superseded by c2_rv |
| `behavior_strict.py` | Strict behavior scoring - utility |
| `retrocompute_mode_score.py` | Retrocompute - utility |

### Test/Validation (3 files)
| File | Reason |
|------|--------|
| `ioi_causal_test.py` | IOI test - reference implementation |
| `prompt_bank_audit.py` | Audit utility - one-time |
| `hysteresis_patching.py` | Hysteresis variant - superseded |

### MLP Position Specific (1 file)
| File | Reason |
|------|--------|
| `mlp_ablation_position_specific.py` | Position-specific - superseded by combined |
| `p1_ablation.py` | P1 ablation - superseded |

---

## Excluded from Categorization

| Location | Reason |
|----------|--------|
| `runpod_sync_20260105_224717/src/pipelines/` | Deployment artifact (duplicate) |
| `boneyard/DEC_9_EMERGENCY_BACKUP/` | Already archived |
| `SUBSYSTEM_2D_MAP_COMPLETION/02_CODE/` | Separate subsystem |

---

## Summary Statistics

| Category | Count |
|----------|-------|
| Infrastructure | 3 |
| Canonical | 7 |
| Discovery | 12 |
| Archive | 35 |
| **Total Categorized** | **57** |
| Subdirectories (2 files) | 2 (in archive) |
| **Grand Total** | **59** |

---

## Notes

1. **rv_l27_causal_validation.py** is the crown jewel - strongest causal result
2. **c2_rv_measurement.py** bridges geometry→behavior (new, Jan 11)
3. Many steering experiments were necessary exploration but are now superseded
4. The archive preserves research history without cluttering active work
5. Discovery pipelines are model-agnostic methodology tools

---

*Analysis complete. Ready for Plan 01-02: Create PIPELINE_CATEGORIZATION.md*
