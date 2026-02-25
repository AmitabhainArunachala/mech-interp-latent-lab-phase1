# Dense-Model Pilot Spec (Pythia-1.4B) — 2026-02-20

## Purpose
Test architecture dependence of the bridge specificity signal by running the same control matrix on a non-GQA dense model.

## Model
- `EleutherAI/pythia-1.4b` (dense attention)

## Matrix
- Conditions: `head_specific`, `random_head_control`, `baseline_donor_control`
- Seeds: `42, 123, 456`
- Total runs: `9`
- Configs: `configs/canonical/seed_bridge_dense_pythia_2026_02_20/`
- Runner: `industry_grade/2026-02-20/run_seed_bridge_dense_pythia_pilot.sh`

## Parameters
- `n_pairs=40` (pilot)
- `early_layer=3`, `target_layer=20`
- `window=16`
- `temperature=0.0` (deterministic tier)

## Primary readout
Within each seed:
- `head_specific - random_head_control` (`rv_delta` mean difference)
- `head_specific - baseline_donor_control`

## Interpretation hook
If the large Mistral separation is GQA-semantics-specific, dense-model effects should attenuate or shift directionality.
