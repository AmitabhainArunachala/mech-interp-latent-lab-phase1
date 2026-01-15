# Results Audit

**Date**: 2026-01-11
**Total Runs Analyzed**: 137

## Summary

| Category | Count | Description |
|----------|-------|-------------|
| Canonical | 10 | Paper-worthy, reproducible findings |
| Discovery | 70 | Methodology development, parameter sweeps |
| Archive | 24 | Historical explorations, superseded |
| Failed | 33 | Runs with error.txt |

## Canonical Results (Paper-Worthy)

### Gold Standard R_V Causal Validation
Best evidence for core finding: R_V contraction is layer-specific and causal.

| Run | Experiment | n | p-value | Quality |
|-----|------------|---|---------|---------|
| gold_standard/runs/20251216_061127_rv_l27_causal_validation | rv_l27_causal_validation | 45 | 2.75e-22 | **Best** |
| gold_standard/runs/20251216_060955_rv_l27_causal_validation | rv_l27_causal_validation | 45 | 2.75e-22 | Duplicate |

**Recommendation**: Keep 20251216_061127 as the canonical run.

### Recent C2 Measurements (Jan 2026)
New R_V measurement runs:

| Run | Experiment | Status |
|-----|------------|--------|
| phase1_mechanism/runs/20260111_123508_c2_rv_measurement | c2_rv_measurement | Needs review |
| phase1_mechanism/runs/20260111_125011_c2_rv_measurement | c2_rv_measurement | Needs review |
| phase1_mechanism/runs/20260111_125410_c2_rv_measurement | c2_rv_measurement | Needs review |
| phase1_mechanism/runs/20260111_130123_c2_rv_measurement | c2_rv_measurement | Needs review |
| phase1_mechanism/runs/20260111_140002_c2_ablation_no_cascade | c2_ablation | Ablation study |
| phase1_mechanism/runs/20260111_140229_c2_ablation_no_steering | c2_ablation | Ablation study |
| phase1_mechanism/runs/20260111_140449_c2_ablation_no_kv | c2_ablation | Ablation study |

### Confound Validation
| Run | Experiment | Status |
|-----|------------|--------|
| gold_standard/runs/20251216_060911_confound_validation | confound_validation | Gold standard |
| confound_validation/runs/20251215_152231_confound_validation_mistral7b_instruct_l27_w16 | confound_validation | Earlier run |
| confound_validation/runs/20251215_091017_confound_validation_mistral7b_instruct_l27_w16 | confound_validation | Earlier run |

## Discovery Results (Methodology Development)

### Behavioral Grounding (43 runs)
Parameter sweeps across layers, window sizes, seeds. Key patterns:
- Layers tested: 24, 26, 27, 30, 33, 35
- Window sizes: 16, 32
- Seeds: 9, 10, 11
- Most recent batch: 20251213_124735 (n=100)

### Steering Experiments (7 runs)
Layer matrix exploration, head-specific interventions:
- steering_layer_matrix: 20251217_135538
- minimal_recursive_intervention: 3 successful runs
- extended_context_steering: 20251217_161456
- triple_system_intervention: 20251218_063238

### Path Patching Mechanism (9 runs)
Early layer sweep explorations:
- default, stress, layer_sweep variants
- Peak: 20251213_070135

## Archive Results (Historical)

### Empty Phase Directories
These were pre-created but never used:
- phase1_cross_architecture (empty)
- phase2_eigenstate (empty)
- phase4_kv_mechanism (empty)
- phase5_steering (empty)
- phase6_alternative_selfref (empty)

### Superseded Explorations
- circuit_hunt_v2_focused
- dec13_kitchen_sink
- kitchen_sink variants
- comprehensive_circuit_test
- champion_paraphrase_hunt
- h31_validation (superseded by gold_standard)
- head_discovery (exploratory)

## Failed Runs (33)

All have error.txt - typically import errors or config issues during development.

| Location | Count | Reason |
|----------|-------|--------|
| runs/ (top-level) | 16 | Early steering experiments |
| gold_standard/ | 2 | Initial confound setup |
| phase0_metric_validation/ | 5 | Early metric tests |
| kv_sufficiency_matrix/ | 5 | Matrix exploration |
| phase1_mechanism/ | 4 | Stress tests |
| confound_validation/ | 1 | Config error |

## Reorganization Plan

### Target Structure
```
results/
├── canonical/           # Paper figures (10 runs)
│   ├── rv_l27_causal_validation/
│   ├── confound_validation/
│   └── c2_ablation_suite/
├── discovery/           # Methodology (70 runs)
│   ├── behavioral_grounding/
│   ├── steering/
│   └── path_patching/
├── archive/             # Historical (57 runs = 24 + 33 failed)
│   ├── failed/
│   └── superseded/
└── run_index.csv        # Updated index
```

### Migration Rules
1. **canonical/**: Only runs with quality_score >= 3 or explicit gold_standard
2. **discovery/**: Active methodology development
3. **archive/failed/**: All runs with error.txt
4. **archive/superseded/**: Exploratory runs not needed for paper

## Next Steps (04-02)
1. Create new directory structure
2. Move runs according to categorization
3. Update run_index.csv with new paths
4. Clean up empty phase directories
