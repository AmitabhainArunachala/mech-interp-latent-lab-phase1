# Nightly Summary

Generated: 2026-03-14 16:52:29Z

## Program Status
- Registry: `configs/experiment_registry/mistral_program_registry.json`
- Results index: `configs/experiment_registry/results_index.json`
- Pod leases: `configs/experiment_registry/pod_leases.json`
- Queue units: `13` total, `11` completed, `0` running, `1` queued, `1` blocked, `0` failed
- Experiments: `19` total, `14` completed, `0` running, `5` queued, `0` failed
- Claim registry: `21` locked, `4` provisional, `8` invalidated

## Active Pods
- No running pod leases recorded.

## Ready Next Queue Units
- `anchor_bundle_v2`: stage `sufficiency_bundle`, queue `mistral_anchor_bundle_v2`, priority `110`, expected `3.0`h, launcher `scripts/runpod_mistral_anchor_bundle_v2_queue.sh`

## Latest Results
- `subspace_component_steering_l27_v1` [completed] -> `results/subspace_component_steering_l27_v1/20260314_144647/summary.json`
- `pca_subspace_ablation_l5_v1` [completed] -> `results/pca_subspace_ablation_l5_v1/20260314_133243/summary.json`
- `pca_subspace_ablation_l25_v1` [completed] -> `results/pca_subspace_ablation_l25_v1/20260314_115345/summary.json`
- `pca_subspace_ablation_v1` [completed] -> `results/pca_subspace_ablation_v1/20260314_102447/summary.json`
- `pca_vs_mean_steering_v2` [completed] -> `results/pca_vs_mean_steering_v2/20260314_024943/summary.json`
- `anchor_bundle_v1` [completed] -> `results/phase1_mechanism/runs/20260314_014025_causal_state_benchmark_v4_multisite_mistral_anchor_bundle_v1/summary.json`
- `pca_vs_mean_steering_v1` [completed] -> `results/pca_vs_mean_steering_v1/20260314_020333/summary.json`
- `eigenstate_subspace_v1` [completed] -> `results/phase3_attention/runs/20260314_014444_eigenstate_subspace_v1/summary.json`
- `subspace_probe_v1` [completed] -> `results/linear_probe_subspace_v1/20260314_013917/probe_analysis_20260314_014027.json`
- `scaffold_ablation_ladder_v2` [completed] -> `results/self_feeding_scaffold_ablation_v2/20260313_210159/self_feeding_summary_20260313_234114.json`

## State Warnings
- Completed without local artifact: `bridge_low_trunc_quality_n24` expected `results/phase1_cross_architecture/runs/20260313_081519_multi_token_bridge_mistral_7b_bridge_low_trunc_quality_n24/summary.json`
- Completed without local artifact: `bridge_true_longgen_quality_n18` expected `results/phase1_cross_architecture/runs/20260313_083452_multi_token_bridge_mistral_7b_bridge_true_longgen_quality_n18/summary.json`
- Completed without local artifact: `self_feeding_loop_v3` expected `results/self_feeding_loop_bundle_v3/self_feeding_summary_20260313_111543.json`
- Completed without local artifact: `sustained_gnani_v3_v3` expected `results/sustained_gnani_v3_bundle_v3/comparison_summary.json`
- Completed without local artifact: `scaffold_ablation_ladder_v1` expected `results/self_feeding_scaffold_ablation_v1/20260313_135628/self_feeding_summary_20260313_153730.json`
- Completed without local artifact: `sustained_gnani_v3_v4` expected `results/sustained_gnani_v3_bundle_v4/20260313_135628/comparison_summary.json`
- Completed without local artifact: `scaffold_ablation_ladder_v2` expected `results/self_feeding_scaffold_ablation_v2/20260313_210159/self_feeding_summary_20260313_234114.json`
- Completed without local artifact: `anchor_bundle_v1` expected `results/phase1_mechanism/runs/20260314_014025_causal_state_benchmark_v4_multisite_mistral_anchor_bundle_v1/summary.json`
- Blocked queue unit: `sufficiency_bundle_v2` waiting on `unknown`
- Result not represented in registry: `subspace_component_steering_l27_v1` -> `results/subspace_component_steering_l27_v1/20260314_144647/summary.json`
- Result not represented in registry: `pca_subspace_ablation_l5_v1` -> `results/pca_subspace_ablation_l5_v1/20260314_133243/summary.json`
- Result not represented in registry: `pca_subspace_ablation_l25_v1` -> `results/pca_subspace_ablation_l25_v1/20260314_115345/summary.json`
- Result not represented in registry: `pca_subspace_ablation_v1` -> `results/pca_subspace_ablation_v1/20260314_102447/summary.json`

## Recommended Next Actions
- Next clean launch is `anchor_bundle_v2` via `scripts/runpod_mistral_anchor_bundle_v2_queue.sh`.
- Harvest remote artifacts before updating paper-facing claims.
- Treat orphan or stale state as operational debt, not as evidence.
