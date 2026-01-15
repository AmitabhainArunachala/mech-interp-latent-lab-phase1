# Results Categorization

**Date**: 2026-01-11
**Phase**: 4 (Results Organization)

## Final Structure

```
results/
├── canonical/                           # Paper figures (10 direct runs, 17 total dirs)
│   ├── rv_l27_causal_validation/        # Core R_V causal validation
│   ├── confound_validation/             # Confound control experiments
│   └── c2_measurement_suite/            # C2 R_V measurements + ablations
├── discovery/                           # Methodology (70 runs)
│   ├── behavioral_grounding/            # 49 runs - behavioral output analysis
│   ├── path_patching/                   # 10 runs - layer sweep patching
│   ├── phase0_validation/               # 4 runs - metric validation
│   └── steering/                        # 7 runs - steering interventions
├── archive/                             # Historical (109 dirs)
│   ├── failed/                          # 33 runs with error.txt
│   ├── superseded/                      # Legacy directories
│   └── steering_control_outputs.txt     # Artifact
├── phase2_generalization/               # Cross-architecture validation runs
└── RUN_INDEX.jsonl                      # Run index (JSONL format)
```

## Canonical Results

### rv_l27_causal_validation/ (2 runs)
**Best run**: `20251216_061127_rv_l27_causal_validation`
- n = 45 pairs
- p = 2.75e-22 (one-sample t-test)
- Cohen's d = -3.56 (estimated)
- Model: Mistral-7B-v0.1
- Key finding: R_V contraction is layer-specific and causally mediated

### confound_validation/ (1 run)
**Run**: `20251216_060911_confound_validation`
- Controls for: complexity, attention, length confounds
- Validates R_V signal is not spurious

### c2_measurement_suite/ (7 runs)
Recent measurements (Jan 2026):
- 4 R_V measurement runs
- 3 ablation runs (no_cascade, no_steering, no_kv)

## Discovery Results

### behavioral_grounding/ (49 runs)
Layer/window/seed parameter sweeps:
- Layers: 24, 26, 27, 30, 33, 35
- Windows: 16, 32
- Seeds: 9, 10, 11
- Key batch: `20251213_124735_behavioral_grounding_batch_ministral8b_n100` (n=100)

### path_patching/ (10 runs)
Mechanism exploration:
- Layer sweep experiments
- Early layer controls
- Stress tests

### phase0_validation/ (4 runs)
Metric validation:
- `phase0_minimal_pairs` - basic pair comparisons
- `phase0_metric_targets` - target metric tests

### steering/ (7 runs)
Steering interventions:
- `steering_layer_matrix` - layer-specific effects
- `minimal_recursive_intervention` - targeted patches
- `extended_context_steering` - longer context tests
- `triple_system_intervention` - combined interventions

## Archive

### failed/ (33 runs)
Runs with error.txt - typically import errors or config issues during development.

### superseded/ (53 directories)
Legacy structure including:
- Old phase directories (phase0-6)
- Kitchen sink explorations
- Head-specific investigations
- Gold standard originals (now in canonical/)
- Mixed-organization runs/ directory

## Usage Notes

1. **Paper figures**: Use `canonical/` runs only
2. **Methodology reference**: Check `discovery/` for parameter sweep results
3. **Historical context**: `archive/superseded/` contains full research trail
4. **Failed runs**: `archive/failed/` useful for debugging patterns

## Migration from Legacy

Old paths → New paths:
- `results/gold_standard/runs/` → `results/canonical/*/`
- `results/phase1_mechanism/runs/` → `results/{canonical,discovery}/*/`
- `results/runs/` → `results/archive/superseded/runs/`
