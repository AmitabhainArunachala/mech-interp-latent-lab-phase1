# Nightly Summary

Generated: 2026-03-14 16:52:24Z

## Program Status
- Registry: `configs/experiment_registry/mistral_program_registry.json`
- Results index: `configs/experiment_registry/results_index.json`
- Pod leases: `configs/experiment_registry/pod_leases.json`
- Queue units: `13` total, `12` completed, `0` running, `0` queued, `1` blocked, `0` failed
- Experiments: `23` total, `17` completed, `2` running, `4` queued, `0` failed
- Claim registry: `22` locked, `5` provisional, `8` invalidated

## Active Pods
- `d08fc4e9d529` [STALE]: queue `mistral_pca_vs_mean_steering_v1`, run `20260314_020135`, step `pca_vs_mean_steering_v1`, updated `2026-03-14T02:01:36Z`

## Ready Next Queue Units
- No ready queue units. Either the queue is exhausted or dependencies are still blocked.

## Latest Results
- `subspace_component_steering_l27_v1` [completed] -> `results/subspace_component_steering_l27_v1/20260314_144647/summary.json`
- `induced_persistence_followup_v2_long` [completed] -> `results/induced_persistence_followup_v2_long/20260314_151808/summary.json`
- `induced_persistence_followup_v1` [completed] -> `results/induced_persistence_followup_v1/20260314_150405/summary.json`
- `pca_subspace_ablation_l5_v1` [completed] -> `results/pca_subspace_ablation_l5_v1/20260314_133243/summary.json`
- `anchor_bundle_v5_ordinary_baselines_confirmatory` [completed] -> `results/phase1_mechanism/runs/20260314_133516_causal_state_benchmark_v4_multisite_mistral_anchor_bundle_v5_ordinary_baselines_confirmatory/summary.json`
- `pca_subspace_ablation_l25_v1` [completed] -> `results/pca_subspace_ablation_l25_v1/20260314_115345/summary.json`
- `anchor_bundle_v4_generalization_controls` [completed] -> `results/phase1_mechanism/runs/20260314_115501_causal_state_benchmark_v4_multisite_mistral_anchor_bundle_v4_generalization_controls/summary.json`
- `anchor_bundle_v3_champion_controls` [completed] -> `results/phase1_mechanism/runs/20260314_102606_causal_state_benchmark_v4_multisite_mistral_anchor_bundle_v3_champion_controls/summary.json`
- `pca_subspace_ablation_v1` [completed] -> `results/pca_subspace_ablation_v1/20260314_102447/summary.json`
- `anchor_bundle_v2` [completed] -> `results/phase1_mechanism/runs/20260314_025048_causal_state_benchmark_v4_multisite_mistral_anchor_bundle_v2/summary.json`

## State Warnings
- Stale running lease: `d08fc4e9d529` queue `mistral_pca_vs_mean_steering_v1` last updated `2026-03-14T02:01:36Z`
- Blocked queue unit: `sufficiency_bundle_v2` waiting on `unknown`
- Result not represented in registry: `subspace_component_steering_l27_v1` -> `results/subspace_component_steering_l27_v1/20260314_144647/summary.json`
- Result not represented in registry: `induced_persistence_followup_v2_long` -> `results/induced_persistence_followup_v2_long/20260314_151808/summary.json`
- Result not represented in registry: `induced_persistence_followup_v1` -> `results/induced_persistence_followup_v1/20260314_150405/summary.json`
- Result not represented in registry: `pca_subspace_ablation_l5_v1` -> `results/pca_subspace_ablation_l5_v1/20260314_133243/summary.json`
- Result not represented in registry: `anchor_bundle_v5_ordinary_baselines_confirmatory` -> `results/phase1_mechanism/runs/20260314_133516_causal_state_benchmark_v4_multisite_mistral_anchor_bundle_v5_ordinary_baselines_confirmatory/summary.json`
- Result not represented in registry: `pca_subspace_ablation_l25_v1` -> `results/pca_subspace_ablation_l25_v1/20260314_115345/summary.json`
- Result not represented in registry: `anchor_bundle_v4_generalization_controls` -> `results/phase1_mechanism/runs/20260314_115501_causal_state_benchmark_v4_multisite_mistral_anchor_bundle_v4_generalization_controls/summary.json`

## Recommended Next Actions
- Reconcile or clear stale leases before trusting any queue status.
- Harvest remote artifacts before updating paper-facing claims.
- Treat orphan or stale state as operational debt, not as evidence.
