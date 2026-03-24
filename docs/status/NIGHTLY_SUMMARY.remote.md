# Nightly Summary

Generated: 2026-03-16 17:43:42Z

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
- `anchor_layermatched_bridge_alpha_sweep_v1` [completed] -> `results/anchor_layermatched_bridge_alpha_sweep_v1/20260316_132850/summary.json`
- `induced_persistence_anchor_layermatched_long_v1` [completed] -> `results/induced_persistence_anchor_layermatched_long_v1/20260316_123052/summary.json`
- `anchor_layermatched_hybrid_protocol_v1` [completed] -> `results/anchor_layermatched_protocol_confirm_v1/20260316_105309/summary.json`
- `induced_persistence_anchor_layermatched_confirm_v1` [completed] -> `results/induced_persistence_anchor_layermatched_confirm_v1/20260316_092904/summary.json`
- `induced_persistence_anchor_layermatched_v1` [completed] -> `results/induced_persistence_anchor_layermatched_v1/20260316_040319/summary.json`
- `induced_persistence_anchor_layermatched_v1` [completed] -> `results/induced_persistence_anchor_layermatched_v1/20260316_035827/summary.json`
- `closed_loop_anchor_controller_v1` [completed] -> `results/closed_loop_anchor_controller_v1/20260316_025020/summary.json`
- `subspace_component_steering_l1_v1` [completed] -> `results/subspace_component_steering_l1_v1/20260315_184445/summary.json`
- `subspace_component_steering_l2_v1` [completed] -> `results/subspace_component_steering_l2_v1/20260315_153614/summary.json`
- `subspace_component_steering_l3_v1` [completed] -> `results/subspace_component_steering_l3_v1/20260315_144837/summary.json`

## State Warnings
- Stale running lease: `d08fc4e9d529` queue `mistral_pca_vs_mean_steering_v1` last updated `2026-03-14T02:01:36Z`
- Completed without local artifact: `control_bundle_gate_bridge_v1` expected `results/phase1_mechanism/runs/20260313_152416_causal_state_benchmark_v4_multisite_mistral_multisite_gate_L5_bridge_L25/summary.json`
- Completed without local artifact: `l4_micro4_confirmatory_focus_v1` expected `results/phase1_mechanism/runs/20260313_210217_causal_state_benchmark_v4_multisite_mistral_multisite_l4_mlp_micro4_confirmatory_focus_v1/summary.json`
- Completed without local artifact: `subspace_probe_v1` expected `results/linear_probe_subspace_v1/20260314_013917/probe_analysis_20260314_014027.json`
- Completed without local artifact: `eigenstate_subspace_v1` expected `results/phase3_attention/runs/20260314_014444_eigenstate_subspace_v1/summary.json`
- Completed without local artifact: `pca_vs_mean_steering_v1` expected `results/pca_vs_mean_steering_v1/20260314_020333/summary.json`
- Blocked queue unit: `sufficiency_bundle_v2` waiting on `unknown`
- Result not represented in registry: `anchor_layermatched_bridge_alpha_sweep_v1` -> `results/anchor_layermatched_bridge_alpha_sweep_v1/20260316_132850/summary.json`
- Result not represented in registry: `induced_persistence_anchor_layermatched_long_v1` -> `results/induced_persistence_anchor_layermatched_long_v1/20260316_123052/summary.json`
- Result not represented in registry: `anchor_layermatched_hybrid_protocol_v1` -> `results/anchor_layermatched_protocol_confirm_v1/20260316_105309/summary.json`
- Result not represented in registry: `induced_persistence_anchor_layermatched_confirm_v1` -> `results/induced_persistence_anchor_layermatched_confirm_v1/20260316_092904/summary.json`
- Result not represented in registry: `induced_persistence_anchor_layermatched_v1` -> `results/induced_persistence_anchor_layermatched_v1/20260316_040319/summary.json`
- Result not represented in registry: `induced_persistence_anchor_layermatched_v1` -> `results/induced_persistence_anchor_layermatched_v1/20260316_035827/summary.json`
- Result not represented in registry: `closed_loop_anchor_controller_v1` -> `results/closed_loop_anchor_controller_v1/20260316_025020/summary.json`
- Result not represented in registry: `subspace_component_steering_l1_v1` -> `results/subspace_component_steering_l1_v1/20260315_184445/summary.json`
- Result not represented in registry: `subspace_component_steering_l2_v1` -> `results/subspace_component_steering_l2_v1/20260315_153614/summary.json`
- Result not represented in registry: `subspace_component_steering_l3_v1` -> `results/subspace_component_steering_l3_v1/20260315_144837/summary.json`

## Recommended Next Actions
- Reconcile or clear stale leases before trusting any queue status.
- Harvest remote artifacts before updating paper-facing claims.
- Treat orphan or stale state as operational debt, not as evidence.
