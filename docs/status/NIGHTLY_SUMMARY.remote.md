# Nightly Summary

Generated: 2026-03-14 16:51:47Z

## Program Status
- Registry: `configs/experiment_registry/mistral_program_registry.json`
- Results index: `configs/experiment_registry/results_index.json`
- Pod leases: `configs/experiment_registry/pod_leases.json`
- Queue units: `13` total, `11` completed, `0` running, `1` queued, `1` blocked, `0` failed
- Experiments: `19` total, `14` completed, `0` running, `5` queued, `0` failed
- Claim registry: `21` locked, `4` provisional, `8` invalidated

## Active Pods
- `d08fc4e9d529` [STALE]: queue `mistral_pca_vs_mean_steering_v1`, run `20260314_020135`, step `pca_vs_mean_steering_v1`, updated `2026-03-14T02:01:36Z`

## Ready Next Queue Units
- `pca_vs_mean_steering_v2`: stage `subspace_probe`, queue `mistral_pca_vs_mean_steering_v2`, priority `120`, expected `2.5`h, launcher `scripts/runpod_mistral_pca_vs_mean_steering_v2_queue.sh`

## Latest Results
- `induced_persistence_followup_v2_long` [completed] -> `results/induced_persistence_followup_v2_long/20260314_151808/summary.json`
- `induced_persistence_followup_v1` [completed] -> `results/induced_persistence_followup_v1/20260314_150405/summary.json`
- `anchor_bundle_v5_ordinary_baselines_confirmatory` [completed] -> `results/phase1_mechanism/runs/20260314_133516_causal_state_benchmark_v4_multisite_mistral_anchor_bundle_v5_ordinary_baselines_confirmatory/summary.json`
- `anchor_bundle_v4_generalization_controls` [completed] -> `results/phase1_mechanism/runs/20260314_115501_causal_state_benchmark_v4_multisite_mistral_anchor_bundle_v4_generalization_controls/summary.json`
- `anchor_bundle_v3_champion_controls` [completed] -> `results/phase1_mechanism/runs/20260314_102606_causal_state_benchmark_v4_multisite_mistral_anchor_bundle_v3_champion_controls/summary.json`
- `anchor_bundle_v2` [completed] -> `results/phase1_mechanism/runs/20260314_025048_causal_state_benchmark_v4_multisite_mistral_anchor_bundle_v2/summary.json`
- `anchor_bundle_v1` [completed] -> `results/phase1_mechanism/runs/20260314_014025_causal_state_benchmark_v4_multisite_mistral_anchor_bundle_v1/summary.json`
- `pca_vs_mean_steering_v1` [completed] -> `results/pca_vs_mean_steering_v1/20260314_020333/summary.json`
- `eigenstate_subspace_v1` [completed] -> `results/phase3_attention/runs/20260314_014444_eigenstate_subspace_v1/summary.json`
- `subspace_probe_v1` [completed] -> `results/linear_probe_subspace_v1/20260314_013917/probe_analysis_20260314_014027.json`

## State Warnings
- Stale running lease: `d08fc4e9d529` queue `mistral_pca_vs_mean_steering_v1` last updated `2026-03-14T02:01:36Z`
- Completed without local artifact: `control_bundle_gate_bridge_v1` expected `results/phase1_mechanism/runs/20260313_152416_causal_state_benchmark_v4_multisite_mistral_multisite_gate_L5_bridge_L25/summary.json`
- Completed without local artifact: `l4_micro4_confirmatory_focus_v1` expected `results/phase1_mechanism/runs/20260313_210217_causal_state_benchmark_v4_multisite_mistral_multisite_l4_mlp_micro4_confirmatory_focus_v1/summary.json`
- Completed without local artifact: `subspace_probe_v1` expected `results/linear_probe_subspace_v1/20260314_013917/probe_analysis_20260314_014027.json`
- Completed without local artifact: `eigenstate_subspace_v1` expected `results/phase3_attention/runs/20260314_014444_eigenstate_subspace_v1/summary.json`
- Completed without local artifact: `pca_vs_mean_steering_v1` expected `results/pca_vs_mean_steering_v1/20260314_020333/summary.json`
- Blocked queue unit: `sufficiency_bundle_v2` waiting on `unknown`
- Result not represented in registry: `induced_persistence_followup_v2_long` -> `results/induced_persistence_followup_v2_long/20260314_151808/summary.json`
- Result not represented in registry: `induced_persistence_followup_v1` -> `results/induced_persistence_followup_v1/20260314_150405/summary.json`
- Result not represented in registry: `anchor_bundle_v5_ordinary_baselines_confirmatory` -> `results/phase1_mechanism/runs/20260314_133516_causal_state_benchmark_v4_multisite_mistral_anchor_bundle_v5_ordinary_baselines_confirmatory/summary.json`
- Result not represented in registry: `anchor_bundle_v4_generalization_controls` -> `results/phase1_mechanism/runs/20260314_115501_causal_state_benchmark_v4_multisite_mistral_anchor_bundle_v4_generalization_controls/summary.json`
- Result not represented in registry: `anchor_bundle_v3_champion_controls` -> `results/phase1_mechanism/runs/20260314_102606_causal_state_benchmark_v4_multisite_mistral_anchor_bundle_v3_champion_controls/summary.json`

## Recommended Next Actions
- Reconcile or clear stale leases before trusting any queue status.
- Harvest remote artifacts before updating paper-facing claims.
- Treat orphan or stale state as operational debt, not as evidence.
