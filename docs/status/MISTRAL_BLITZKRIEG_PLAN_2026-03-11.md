# Mistral Blitzkrieg Plan

**Date:** 2026-03-11  
**Scope:** second-pod Mistral-only assault on unresolved, weird, and high-upside mechanism questions  
**Mode:** split into `canonical` and `exploratory`; do not mix them in the paper without re-hardening

## Mission

This campaign is not about getting more of the same Mistral evidence. It is about forcing the remaining unknowns to collapse:

1. Kill fake sufficiency if it is fake.
2. Rescue sufficiency if it is real but currently masked by blunt interventions.
3. Find the smallest coherent intervention that moves both `R_V` and behavior.
4. Separate genuine behavior transfer from degeneration, repetition, and prompt-specific artifacts.
5. Surface any mechanism the current paper is missing.

## Starting Point

Current hardened anchor artifacts:

- [mistralai__Mistral-7B-Instruct-v0-2_p0_result.json](/Users/dhyana/mech-interp-latent-lab-phase1/results/p0_canonical/mistralai__Mistral-7B-Instruct-v0-2_p0_result.json)
- [atlas_summary_20260310_145239.json](/Users/dhyana/mech-interp-latent-lab-phase1/results/mode_atlas/atlas_summary_20260310_145239.json)
- [svd_decomposition_20260310_145312.json](/Users/dhyana/mech-interp-latent-lab-phase1/results/svd_circuits/svd_decomposition_20260310_145312.json)
- [path_patching_summary_20260310_151654.json](/Users/dhyana/mech-interp-latent-lab-phase1/results/path_patching/path_patching_summary_20260310_151654.json)
- [persistent_patching_v3_dual_20260310_160920.json](/Users/dhyana/mech-interp-latent-lab-phase1/results/persistent_patching_v3/persistent_patching_v3_dual_20260310_160920.json)

The current honest state is:

- contraction is real
- SVD circuit motif is real
- early residual necessity is strong
- induce / sufficiency is not currently holding up
- dissociation is a live possibility

That makes sufficiency rescue and falsification the main frontier.

## Operating Rules

- `Canonical track` uses `mistralai/Mistral-7B-Instruct-v0.2` plus the frozen prompt contracts.
- `Exploratory track` is allowed to use base `mistralai/Mistral-7B-v0.1`, older configs, and weird interventions.
- Exploratory wins only matter if they can later be rerun under the canonical contract.
- Every run must log both geometry and output quality.
- A run does not count as sufficiency if it only increases malformed output, repetition, or blanket refusal.

## Track 1: Sufficiency Rescue

### 1A. Dual-patch quality triage

Use:

- [persistent_patching_v3_dual.py](/Users/dhyana/mech-interp-latent-lab-phase1/scripts/persistent_patching_v3_dual.py)
- [persistent_patching_v3_dual_20260310_191654.json](/Users/dhyana/mech-interp-latent-lab-phase1/results/persistent_patching_v3/persistent_patching_v3_dual_20260310_191654.json)

Questions:

- Is induce failing because the intervention is wrong, or because it destroys coherence?
- Does changing donor strength or `R`-layer help?
- Do geometry shifts persist when quality is gated?

### 1B. Full 2x2 sufficiency ladder

Use:

- [sufficiency_ladder.py](/Users/dhyana/mech-interp-latent-lab-phase1/scripts/sufficiency_ladder.py)
- [hardening_summary_20260311_001645.json](/Users/dhyana/mech-interp-latent-lab-phase1/results/statistical_hardening/hardening_summary_20260311_001645.json)

Questions:

- Is `kv_only` stronger than `dual_patch`?
- Is `kv_plus_dual` genuinely additive?
- Does the ladder fail because it is behaviorally wrong, not geometrically weak?

### 1C. Mediation

Use:

- [mediation_2x2.py](/Users/dhyana/mech-interp-latent-lab-phase1/scripts/mediation_2x2.py)

Question:

- if late patching works only when an early MLP path is intact, then `R_V` is not an isolated driver; it is a downstream readout of a gated circuit

## Track 2: Sparse And Precise Interventions

The current donor averages are probably too blunt.

### 2A. Head-specific bridge resurrection

Relevant buried evidence:

- [summary.json](/Users/dhyana/mech-interp-latent-lab-phase1/results/meta_yolo/runs/20260220_102811_offline_meta_yolo/summary.json)
- [NEURIPS_FINDING_ONEPAGER_2026-02-20.md](/Users/dhyana/mech-interp-latent-lab-phase1/docs/findings/NEURIPS_FINDING_ONEPAGER_2026-02-20.md)
- `configs/canonical/rv_l27_head_specific_bridge.json`
- `configs/canonical/rv_l27_head_specific_bridge_fast.json`

Targets to test:

- `L27_H10`
- strongest late heads from [full_head_sweep_20260310_151508.json](/Users/dhyana/mech-interp-latent-lab-phase1/results/full_head_sweep/full_head_sweep_20260310_151508.json)
- early amplifier heads near `L5`

Key question:

- do sparse head-level donors beat full-layer donor means?

### 2B. Alpha and band sweeps

Use:

- [hardening_battery.py](/Users/dhyana/mech-interp-latent-lab-phase1/scripts/hardening_battery.py)

It combines:

- layer-band KV ablation
- alpha sweep for dual patching
- attention-pattern divergence

The main question is whether a partial intervention transfers mode without wrecking fluency.

### 2C. Same-prompt and within-session bridge

Use:

- [within_session_bridge.py](/Users/dhyana/mech-interp-latent-lab-phase1/scripts/within_session_bridge.py)
- [bridge_battery.py](/Users/dhyana/mech-interp-latent-lab-phase1/scripts/bridge_battery.py)
- [causal_generation_bridge.py](/Users/dhyana/mech-interp-latent-lab-phase1/scripts/causal_generation_bridge.py)

Questions:

- does low `R_V` predict better next-turn recursive behavior?
- is within-session modulation easier than cross-prompt transplant?
- is temporal precedence supporting a mediation story?

## Track 3: Precision Circuit Rebuild

### 3A. Re-run the big three harder

Use:

- [full_head_sweep.py](/Users/dhyana/mech-interp-latent-lab-phase1/scripts/full_head_sweep.py)
- [svd_circuit_decomposition.py](/Users/dhyana/mech-interp-latent-lab-phase1/scripts/svd_circuit_decomposition.py)
- [full_path_patching.py](/Users/dhyana/mech-interp-latent-lab-phase1/scripts/full_path_patching.py)

Reason:

- the overnight `n=20` results point to a structure but do not settle it
- stable head rankings are needed before sparse bridge tests
- early residual dominance should be stress-tested at `n=40+`

### 3B. Combined-circuit sufficiency

Dormant but important:

- [run_mlp_vproj_combined.py](/Users/dhyana/mech-interp-latent-lab-phase1/scripts/run_mlp_vproj_combined.py)
- [stage2_canonical_suite.py](/Users/dhyana/mech-interp-latent-lab-phase1/scripts/stage2_canonical_suite.py)
- `src/pipelines/mlp_vproj_combined_sufficiency_test.py`
- [RUNPOD_SETUP_INSTRUCTIONS.md](/Users/dhyana/mech-interp-latent-lab-phase1/docs/misc/RUNPOD_SETUP_INSTRUCTIONS.md)

Best-case outcome:

`gate + amplifier + late readout` works even if plain `R_V` transfer does not.

## Track 4: Representation Diagnostics

Use:

- [run_logit_lens_analysis.py](/Users/dhyana/mech-interp-latent-lab-phase1/scripts/run_logit_lens_analysis.py)
- `src/pipelines/logit_lens_analysis.py`
- `src/pipelines/vproj_patching_analysis.py`
- [classifier_evaluation.py](/Users/dhyana/mech-interp-latent-lab-phase1/scripts/classifier_evaluation.py)
- [perplexity_repairing.py](/Users/dhyana/mech-interp-latent-lab-phase1/scripts/perplexity_repairing.py)
- [per_token_rv_analysis.py](/Users/dhyana/mech-interp-latent-lab-phase1/scripts/per_token_rv_analysis.py)

Questions:

- are interventions shifting token-level geometry before behavior appears?
- when do recursive and baseline trajectories first diverge?
- does a bridge alter state quality or just lexical surface form?
- does perplexity explain false positives?

## Track 5: Archive Killshots

Explicitly attack overconfident legacy stories.

Main target:

- [THE_CLOSING_LOOP_SOLUTION.md](/Users/dhyana/mech-interp-latent-lab-phase1/docs/misc/THE_CLOSING_LOOP_SOLUTION.md)

Current hardened data does not support its confidence level. The correct move is to test it brutally:

- persistent `V_PROJ` patching with modern degeneration metrics
- multi-layer patching instead of single-layer mythology
- attention-pattern patching if value-only patching still collapses

Also mine these buried families from [summary.json](/Users/dhyana/mech-interp-latent-lab-phase1/results/meta_yolo/runs/20260220_102811_offline_meta_yolo/summary.json):

- `minimal_recursive_intervention`
- `extended_context_steering`
- `steering_layer_matrix`
- `triple_system_intervention`
- `hysteresis_patching`

## Track 6: Safety And Adversarial Stress

Only after one intervention improves behavior without obvious degeneration.

Use:

- [safety_monitoring.py](/Users/dhyana/mech-interp-latent-lab-phase1/scripts/safety_monitoring.py)
- [classifier_validation.py](/Users/dhyana/mech-interp-latent-lab-phase1/scripts/classifier_validation.py)
- [classifier_evaluation.py](/Users/dhyana/mech-interp-latent-lab-phase1/scripts/classifier_evaluation.py)

Question:

- after stabilizing the model at a contracted regime, does it become more robust, more brittle, or just more refusal-prone under adversarial prompts?

## Run Now Vs Refactor First

### Run now on a second pod

- [persistent_patching_v3_dual.py](/Users/dhyana/mech-interp-latent-lab-phase1/scripts/persistent_patching_v3_dual.py)
- [mediation_2x2.py](/Users/dhyana/mech-interp-latent-lab-phase1/scripts/mediation_2x2.py)
- [sufficiency_ladder.py](/Users/dhyana/mech-interp-latent-lab-phase1/scripts/sufficiency_ladder.py)
- [hardening_battery.py](/Users/dhyana/mech-interp-latent-lab-phase1/scripts/hardening_battery.py)
- [full_head_sweep.py](/Users/dhyana/mech-interp-latent-lab-phase1/scripts/full_head_sweep.py)
- [svd_circuit_decomposition.py](/Users/dhyana/mech-interp-latent-lab-phase1/scripts/svd_circuit_decomposition.py)
- [full_path_patching.py](/Users/dhyana/mech-interp-latent-lab-phase1/scripts/full_path_patching.py)
- [within_session_bridge.py](/Users/dhyana/mech-interp-latent-lab-phase1/scripts/within_session_bridge.py)
- [bridge_battery.py](/Users/dhyana/mech-interp-latent-lab-phase1/scripts/bridge_battery.py)
- [classifier_evaluation.py](/Users/dhyana/mech-interp-latent-lab-phase1/scripts/classifier_evaluation.py)
- [perplexity_repairing.py](/Users/dhyana/mech-interp-latent-lab-phase1/scripts/perplexity_repairing.py)
- [per_token_rv_analysis.py](/Users/dhyana/mech-interp-latent-lab-phase1/scripts/per_token_rv_analysis.py)

### Refactor first, then run

- `src/pipelines/vproj_patching_analysis.py`
- `src/pipelines/mlp_vproj_combined_sufficiency_test.py`
- head-specific bridge configs under `configs/canonical/`
- older discovery pipelines listed in [PIPELINE_CATEGORIZATION.md](/Users/dhyana/mech-interp-latent-lab-phase1/docs/misc/PIPELINE_CATEGORIZATION.md)

## 24-Hour Second-Pod Queue

1. canonical `full_head_sweep` at `n=40`
2. canonical `svd_circuit_decomposition` at `n=40`
3. canonical `full_path_patching` at `n=40`
4. canonical `persistent_patching_v3_dual` smoke
5. canonical `persistent_patching_v3_dual` full
6. canonical `mediation_2x2` smoke
7. canonical `mediation_2x2` full
8. exploratory `sufficiency_ladder` smoke
9. exploratory `sufficiency_ladder` full
10. exploratory `hardening_battery`
11. CPU postmortem analyses on all resulting artifacts

## What Counts As A Win

- a sparse or partial intervention improves `BT+ART` without quality collapse
- mediation finds an upstream gate that explains failed induce
- head-specific bridge beats full-layer donor averages
- combined MLP + V-proj intervention works where single-channel intervention fails
- temporal or token-level diagnostics reveal a reproducible pre-behavior state transition

## What Counts As A Valuable Negative Result

- `R_V` can be moved substantially while coherent behavior does not move
- apparent induce success is explained by degeneration, repetition, or prompt leakage
- early residual is necessary while late value transfer is only correlational
- the strongest true story is double dissociation, not sufficiency

That would still support a strong paper. It would just be a different one.
