# AMIROS Status Board

Generated: 2026-03-22 00:28:37Z

## Program Status
- Registry: `configs/experiment_registry/mistral_program_registry.json`
- Results index: `configs/experiment_registry/results_index.json`
- Pod leases: `configs/experiment_registry/pod_leases.json`
- Queue units: `42` total, `38` completed, `1` running, `2` queued, `0` blocked, `1` failed
- Experiments: `52` total, `47` completed, `0` running, `4` queued, `1` failed
- Claim registry: `35` locked, `6` provisional, `8` invalidated

## Active Pods
- `d08fc4e9d529` [STALE]: queue `mistral_pca_vs_mean_steering_v1`, run `20260314_020135`, step `pca_vs_mean_steering_v1`, updated `2026-03-14T02:01:36Z`
- `grotesque_beige_salmon` [STALE]: queue `mistral_soft_break_latebundle_sweep_v1`, run `20260318_103918`, step `mistral_soft_break_latebundle_sweep_v1`, updated `2026-03-18T10:39:54Z`

## Ready Next Queue Units
- `staged_anchor_handoff_confirm_v1`: stage `sufficiency_protocol`, queue `mistral_staged_anchor_handoff_confirm_v1`, priority `184`, expected `5.2`h, launcher `scripts/runpod_mistral_staged_anchor_handoff_confirm_v1_queue.sh`
- `mixtral8x7b_p0_canonical_v1`: stage `cross_architecture_replication`, queue `mixtral8x7b_p0_canonical_v1`, priority `201`, expected `4.0`h, launcher `scripts/runpod_mixtral8x7b_p0_canonical_v1_queue.sh`

## Latest Results
- `induced_persistence_unselected_reduced_drop_l25_v2` [completed] -> `results/induced_persistence_unselected_reduced_drop_l25_v2/20260321_112300/summary.json`
- `induced_persistence_unselected_reduced_late_only_v2` [completed] -> `results/induced_persistence_unselected_reduced_late_only_v2/20260321_030414/summary.json`
- `sustained_gnani_v3_recover` [completed] -> `results/mistral_reduced_late_ladder_v1_bundle/20260320_125708/sustained_gnani_v3_recover/comparison_summary.json`
- `reduced_late_structured_unselected` [completed] -> `results/mistral_reduced_late_ladder_v1_bundle/20260320_125708/reduced_late_structured_unselected/summary.json`
- `reduced_late_lowrv_24` [completed] -> `results/mistral_reduced_late_ladder_v1_bundle/20260320_125708/reduced_late_lowrv_24/summary.json`
- `reduced_late_lowrv_12` [completed] -> `results/mistral_reduced_late_ladder_v1_bundle/20260320_125708/reduced_late_lowrv_12/summary.json`
- `reduced_late_random_12` [completed] -> `results/mistral_reduced_late_ladder_v1_bundle/20260320_125708/reduced_late_random_12/summary.json`
- `self_feeding_loop_v2` [completed] -> `results/self_feeding_loop_bundle_v2/20260319_110753/self_feeding_summary_20260319_172423.json`
- `bridge_true_longgen_n18` [completed] -> `results/phase1_cross_architecture/runs/20260319_131310_multi_token_bridge_mistral_7b_bridge_true_longgen_n18/summary.json`
- `bridge_low_trunc_confirmatory_n24` [completed] -> `results/phase1_cross_architecture/runs/20260319_110757_multi_token_bridge_mistral_7b_bridge_low_trunc_confirmatory_n24/summary.json`

## State Warnings
- Stale running lease: `d08fc4e9d529` queue `mistral_pca_vs_mean_steering_v1` last updated `2026-03-14T02:01:36Z`
- Stale running lease: `grotesque_beige_salmon` queue `mistral_soft_break_latebundle_sweep_v1` last updated `2026-03-18T10:39:54Z`
- Completed without local artifact: `bridge_low_trunc_quality_n24` expected `results/phase1_cross_architecture/runs/20260313_081519_multi_token_bridge_mistral_7b_bridge_low_trunc_quality_n24/summary.json`
- Completed without local artifact: `bridge_true_longgen_quality_n18` expected `results/phase1_cross_architecture/runs/20260313_083452_multi_token_bridge_mistral_7b_bridge_true_longgen_quality_n18/summary.json`
- Result not represented in registry: `induced_persistence_unselected_reduced_drop_l25_v2` -> `results/induced_persistence_unselected_reduced_drop_l25_v2/20260321_112300/summary.json`
- Result not represented in registry: `induced_persistence_unselected_reduced_late_only_v2` -> `results/induced_persistence_unselected_reduced_late_only_v2/20260321_030414/summary.json`
- Result not represented in registry: `sustained_gnani_v3_recover` -> `results/mistral_reduced_late_ladder_v1_bundle/20260320_125708/sustained_gnani_v3_recover/comparison_summary.json`
- Result not represented in registry: `reduced_late_structured_unselected` -> `results/mistral_reduced_late_ladder_v1_bundle/20260320_125708/reduced_late_structured_unselected/summary.json`
- Result not represented in registry: `reduced_late_lowrv_24` -> `results/mistral_reduced_late_ladder_v1_bundle/20260320_125708/reduced_late_lowrv_24/summary.json`
- Result not represented in registry: `reduced_late_lowrv_12` -> `results/mistral_reduced_late_ladder_v1_bundle/20260320_125708/reduced_late_lowrv_12/summary.json`
- Result not represented in registry: `reduced_late_random_12` -> `results/mistral_reduced_late_ladder_v1_bundle/20260320_125708/reduced_late_random_12/summary.json`
- Result not represented in registry: `mistral_soft_break_latebundle_sweep_v1` -> `results/mistral_soft_break_latebundle_sweep_v1/20260318_103918/summary.json`
- Result not represented in registry: `mistral_soft_break_latebundle_v1` -> `results/mistral_soft_break_latebundle_v1/20260318_094202/summary.json`
- Result not represented in registry: `mistral_caliper_matched_promptpass_v1` -> `results/mistral_caliper_matched_promptpass_v1/20260318_093519/summary.json`

## Recommended Next Actions
- Reconcile or clear stale leases before trusting any queue status.
- Harvest remote artifacts before updating paper-facing claims.
- Treat orphan or stale state as operational debt, not as evidence.
