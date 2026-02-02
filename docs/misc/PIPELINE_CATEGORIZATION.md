# Pipeline Categorization

**Created**: 2026-01-11
**Purpose**: Authoritative categorization for Phase 2 reorganization
**Source**: `.planning/phases/01-pipeline-categorization/PIPELINE_ANALYSIS.md`

---

## Decision Criteria

| Category | Definition | Question |
|----------|------------|----------|
| **Canonical** | Required to reproduce paper findings | "Does this prove a claim in the paper?" |
| **Discovery** | Methodology tools for new model exploration | "Would we use this on Llama-7B?" |
| **Archive** | Historical/superseded/dead-end code | "Is this still actively useful?" |

---

## Canonical Pipelines (7 files)

**Target Location**: `src/pipelines/canonical/`

These pipelines reproduce the paper's core causal findings.

| File | Purpose | Key Result | New Location |
|------|---------|------------|--------------|
| `rv_l27_causal_validation.py` | L27 V-proj activation patching with controls | d=-3.56, p<10⁻⁶, n=45, 117.8% transfer | `canonical/rv_l27_causal_validation.py` |
| `confound_validation.py` | 4-pillar control: champions vs length-matched vs pseudo-recursive | Rules out confounds | `canonical/confound_validation.py` |
| `random_direction_control.py` | Random/orthogonal direction baseline | Proves direction specificity | `canonical/random_direction_control.py` |
| `mlp_ablation_necessity.py` | L0 MLP necessity (zero-out ablation) | Tests if ablation kills effect | `canonical/mlp_ablation_necessity.py` |
| `mlp_sufficiency_test.py` | L0 MLP sufficiency (patch into baseline) | Tests if patching induces effect | `canonical/mlp_sufficiency_test.py` |
| `mlp_combined_sufficiency_test.py` | L0+L1+L3 combined sufficiency | Multi-layer joint test | `canonical/mlp_combined_sufficiency_test.py` |
| `head_ablation_validation.py` | KV-head (H18/H26) ablation validation | Head specificity proof | `canonical/head_ablation_validation.py` |

---

## Discovery Pipelines (12 files)

**Target Location**: `src/pipelines/discovery/`

Methodology tools for finding circuits in new models.

| File | Purpose | When to Use | New Location |
|------|---------|-------------|--------------|
| `c2_rv_measurement.py` | C2 config geometry→behavior bridge | Measuring R_V during generation | `discovery/c2_rv_measurement.py` |
| `behavioral_grounding.py` | Test if geometric changes affect output | Validate geometry→behavior | `discovery/behavioral_grounding.py` |
| `behavioral_grounding_batch.py` | Batch version for large-scale testing | Scale behavior validation | `discovery/behavioral_grounding_batch.py` |
| `eigenstate_direction_finder.py` | PCA/SVD to find steering directions | Initial direction discovery | `discovery/eigenstate_direction_finder.py` |
| `logit_lens_analysis.py` | Logit lens token prediction analysis | Understand intermediate representations | `discovery/logit_lens_analysis.py` |
| `vproj_patching_analysis.py` | V_proj patching analysis | Head-level interventions | `discovery/vproj_patching_analysis.py` |
| `mlp_vproj_combined_sufficiency_test.py` | MLP + V_proj joint sufficiency | Test combined mechanisms | `discovery/mlp_vproj_combined_sufficiency_test.py` |
| `path_patching_mechanism.py` | Nanda-style path patching | Circuit tracing | `discovery/path_patching_mechanism.py` |
| `hysteresis.py` | Temporal stability testing | Test effect persistence | `discovery/hysteresis.py` |
| `temporal_stability.py` | Related stability analysis | Effect over time | `discovery/temporal_stability.py` |
| `kv_mechanism.py` | KV swap interventions | KV-based steering | `discovery/kv_mechanism.py` |
| `layer_sweep.py` | Layer-by-layer analysis | Find critical layers | `discovery/layer_sweep.py` |

---

## Archive Pipelines (35 files)

**Target Location**: `src/pipelines/archive/`

Historical record preserved but not for active use.

### Early Phase Experiments
| File | Reason for Archive |
|------|-------------------|
| `phase0_minimal_pairs.py` | Superseded by rv_l27_causal_validation |
| `phase0_metric_targets.py` | Early exploration, no longer needed |
| `phase1_existence.py` | Superseded by full validation pipelines |
| `mistral_L27_full_validation.py` | Merged into rv_l27_causal_validation |

### Steering Experiments
| File | Reason for Archive |
|------|-------------------|
| `steering.py` | Superseded by surgical_sweep |
| `steering_analysis.py` | Companion to steering.py |
| `steering_control.py` | Control variant, superseded |
| `steering_layer_matrix.py` | Matrix sweep, exploratory |
| `extended_context_steering.py` | Extended context, exploratory |
| `minimal_recursive_intervention.py` | Early minimal version |
| `mlp_steering_sweep.py` | MLP sweep, superseded by necessity/sufficiency tests |
| `sprint_head_specific_steering/pipeline.py` | Sprint experiment, exploratory |
| `p10_advanced_steering/pipeline.py` | Advanced steering, exploratory |

### Investigation Experiments
| File | Reason for Archive |
|------|-------------------|
| `l27_deep_dive.py` | Superseded by validation pipelines |
| `l27_head_analysis.py` | Merged into head_ablation_validation |
| `h31_investigation.py` | Dead end (H31 not causal) |
| `h31_ablation_causal.py` | Dead end (H31 not causal) |
| `anthropic_level_investigation.py` | Exploratory, too broad |
| `comprehensive_circuit_analysis.py` | Too comprehensive, unfocused |
| `causal_mechanism_hunt.py` | Early exploration |
| `source_isolation_diagnostic.py` | Debugging utility |

### Sweep/Matrix Experiments
| File | Reason for Archive |
|------|-------------------|
| `surgical_sweep.py` | Exploratory sweep |
| `verification_sweep.py` | Superseded by canonical tests |
| `importance_sweep.py` | Importance ranking, exploratory |
| `kv_sufficiency_matrix.py` | KV matrix, exploratory |
| `unified_layer_map.py` | Layer mapping, exploratory |
| `triple_system_intervention.py` | Complex intervention, exploratory |

### Utility/One-off
| File | Reason for Archive |
|------|-------------------|
| `kitchen_sink.py` | "Stress test", exploratory |
| `circuit_discovery.py` | Early discovery, superseded |
| `geometry_behavior.py` | Superseded by c2_rv_measurement |
| `behavior_strict.py` | Utility, not experiment |
| `retrocompute_mode_score.py` | Utility, not experiment |
| `ioi_causal_test.py` | Reference implementation |
| `prompt_bank_audit.py` | One-time audit utility |
| `hysteresis_patching.py` | Superseded by hysteresis.py |
| `mlp_ablation_position_specific.py` | Superseded by combined test |
| `p1_ablation.py` | Superseded |

---

## Infrastructure (Stays in Place)

| File | Purpose | Action |
|------|---------|--------|
| `registry.py` | Experiment dispatch, config validation | **NO MOVE** - update imports |
| `run.py` | CLI entry point | **NO MOVE** |
| `__init__.py` | Module exports | **NO MOVE** - update exports |

---

## Exclusions

| Location | Reason |
|----------|--------|
| `runpod_sync_20260105_224717/src/pipelines/` | Deployment artifact (duplicates main) |
| `boneyard/DEC_9_EMERGENCY_BACKUP/` | Already archived externally |
| `SUBSYSTEM_2D_MAP_COMPLETION/02_CODE/` | Separate subsystem |

---

## Summary

| Category | Count | Target Directory |
|----------|-------|------------------|
| Canonical | 7 | `src/pipelines/canonical/` |
| Discovery | 12 | `src/pipelines/discovery/` |
| Archive | 35 | `src/pipelines/archive/` |
| Infrastructure | 3 | `src/pipelines/` (no move) |
| Subdirs | 2 | Included in archive |
| **Total** | **59** | |

---

## Phase 2 Instructions

1. **Create directory structure**:
   ```bash
   mkdir -p src/pipelines/{canonical,discovery,archive}
   ```

2. **Move canonical pipelines first** (7 files):
   - These are the most critical; verify imports work after move

3. **Move discovery pipelines** (12 files):
   - Update any cross-references

4. **Move archive pipelines** (35 files):
   - Include subdirectories `sprint_head_specific_steering/` and `p10_advanced_steering/`

5. **Update infrastructure**:
   - `registry.py`: Update import paths
   - `__init__.py`: Update exports
   - Test `python -m src.pipelines.run --help` works

6. **Run tests**:
   - `python -c "from src.pipelines.registry import get_registry; print(len(get_registry()))"`
   - Verify all 41 registered experiments still load

---

*Categorization complete. See Phase 2 plans for execution.*
