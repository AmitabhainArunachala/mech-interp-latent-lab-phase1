# Mech-Interp Latent Lab: Project Index (2026-02-20)

## Snapshot
- Repository root: `/Users/dhyana/mech-interp-latent-lab-phase1`
- Tracked file count (via `rg --files`): `2686`
- Largest top-level areas (file count):
  - `results` (1483)
  - `docs` (313)
  - `configs` (268)
  - `archive` (159)
  - `src` (100)
  - `scripts` (64)

## Core Execution Surface
- Canonical runner: `src/pipelines/run.py`
- Experiment registry: `src/pipelines/registry.py`
- Canonical experiment implementations:
  - `src/pipelines/canonical/rv_l27_causal_validation.py`
  - `src/pipelines/canonical/rv_l27_activation_patching_bridge.py`
  - `src/pipelines/canonical/rv_l27_kv_patching_bridge.py`
  - `src/pipelines/canonical/head_ablation_validation.py`
  - `src/pipelines/canonical/mlp_sufficiency_test.py`
  - `src/pipelines/canonical/mlp_combined_sufficiency_test.py`
  - `src/pipelines/canonical/random_direction_control.py`
  - `src/pipelines/canonical/confound_validation.py`
  - `src/pipelines/canonical/multi_token_bridge.py`

## Core Mechanics
- Model loading/device policy: `src/core/models.py`
- Hooking + interventions: `src/core/hooks.py`, `src/core/patching.py`, `src/core/head_specific_patching.py`
- Geometry metric: `src/metrics/rv.py`
- Behavioral bridge metrics: `src/metrics/behavioral_bridge.py`
- Baseline/ledger utilities: `src/metrics/baseline_suite.py`, `src/utils/run_index.py`, `src/utils/run_metadata.py`

## Canonical Config Entry Points
- Base causal validation: `configs/canonical/rv_l27_causal_validation.json`
- Bridge intervention matrix:
  - `configs/canonical/rv_l27_head_specific_bridge.json`
  - `configs/canonical/rv_l27_random_head_bridge.json`
  - `configs/canonical/rv_l27_baseline_donor_bridge.json`
- Fast GPU bridge variants:
  - `configs/canonical/rv_l27_head_specific_bridge_fast.json`
  - `configs/canonical/rv_l27_random_head_bridge_fast.json`
  - `configs/canonical/rv_l27_baseline_donor_bridge_fast.json`
- Multi-token bridge:
  - `configs/canonical/multi_token_bridge_mistral_deconfound_fast.json`

## Experiment/Analysis Scripts
- Research-readiness gate: `scripts/verify_research_ready.py`
- Meta experiment orchestrator: `scripts/offline_yolo_lab.py`
- Batch runners and reporting:
  - `scripts/run_batch_and_report.py`
  - `scripts/run_and_report.py`
  - `scripts/stage2_canonical_suite.py`

## Current Strongest Finding Artifacts
- NeurIPS candidate memo: `docs/findings/NEURIPS_CANDIDATE_2026-02-20.md`
- Offline meta-yolo run:
  - `results/meta_yolo/runs/20260220_102900_offline_meta_yolo/summary.json`
  - `results/meta_yolo/runs/20260220_102900_offline_meta_yolo/report.md`
- GPU bridge fast run sync:
  - `results/remote_gpu_sync/2026-02-20/phase1_mechanism/20260220_062713_rv_l27_activation_patching_bridge_head_specific_bridge_fast/summary.json`
  - `results/remote_gpu_sync/2026-02-20/phase1_mechanism/20260220_063343_rv_l27_activation_patching_bridge_random_head_bridge_control_fast/summary.json`
  - `results/remote_gpu_sync/2026-02-20/phase1_mechanism/20260220_063955_rv_l27_activation_patching_bridge_baseline_donor_specificity_control_fast/summary.json`
  - `results/remote_gpu_sync/2026-02-20/phase1_mechanism/contrast_stats.md`

## Suggested Default Run Order
1. `scripts/verify_research_ready.py`
2. `rv_l27_causal_validation`
3. Bridge matrix (`head_specific`, `random_head`, `baseline_donor`)
4. Multi-token bridge only after low truncation is guaranteed
