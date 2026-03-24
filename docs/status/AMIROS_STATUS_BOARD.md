# AMIROS Status Board

Generated: 2026-03-24 14:39:36Z

## Program Status
- Registry: `configs/experiment_registry/mistral_program_registry.json`
- Results index: `configs/experiment_registry/results_index.json`
- Pod leases: `configs/experiment_registry/pod_leases.json`
- Queue units: `42` total, `38` completed, `2` running, `1` queued, `0` blocked, `1` failed
- Experiments: `52` total, `47` completed, `0` running, `4` queued, `1` failed
- Claim registry: `35` locked, `6` provisional, `8` invalidated

## Active Pods
- No running pod leases recorded.

## Ready Next Queue Units
- `staged_anchor_handoff_confirm_v1`: stage `sufficiency_protocol`, queue `mistral_staged_anchor_handoff_confirm_v1`, priority `184`, expected `5.2`h, launcher `scripts/runpod_mistral_staged_anchor_handoff_confirm_v1_queue.sh`

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
- `positive_broad_persistence_lowrv_v3` [completed] -> `results/positive_broad_persistence_lowrv_v3/20260319_144423/summary.json`

## State Warnings
- Result not represented in registry: `induced_persistence_unselected_reduced_drop_l25_v2` -> `results/induced_persistence_unselected_reduced_drop_l25_v2/20260321_112300/summary.json`
- Result not represented in registry: `induced_persistence_unselected_reduced_late_only_v2` -> `results/induced_persistence_unselected_reduced_late_only_v2/20260321_030414/summary.json`
- Result not represented in registry: `sustained_gnani_v3_recover` -> `results/mistral_reduced_late_ladder_v1_bundle/20260320_125708/sustained_gnani_v3_recover/comparison_summary.json`
- Result not represented in registry: `reduced_late_structured_unselected` -> `results/mistral_reduced_late_ladder_v1_bundle/20260320_125708/reduced_late_structured_unselected/summary.json`
- Result not represented in registry: `reduced_late_lowrv_24` -> `results/mistral_reduced_late_ladder_v1_bundle/20260320_125708/reduced_late_lowrv_24/summary.json`
- Result not represented in registry: `reduced_late_lowrv_12` -> `results/mistral_reduced_late_ladder_v1_bundle/20260320_125708/reduced_late_lowrv_12/summary.json`
- Result not represented in registry: `reduced_late_random_12` -> `results/mistral_reduced_late_ladder_v1_bundle/20260320_125708/reduced_late_random_12/summary.json`
- Result not represented in registry: `positive_broad_persistence_lowrv_v3` -> `results/positive_broad_persistence_lowrv_v3/20260319_144423/summary.json`
- Result not represented in registry: `mistral_soft_break_latebundle_sweep_v1` -> `results/mistral_soft_break_latebundle_sweep_v1/20260318_103918/summary.json`
- Result not represented in registry: `mistral_soft_break_latebundle_v1` -> `results/mistral_soft_break_latebundle_v1/20260318_094202/summary.json`

## Recommended Next Actions
- Next clean launch is `staged_anchor_handoff_confirm_v1` via `scripts/runpod_mistral_staged_anchor_handoff_confirm_v1_queue.sh`.
- Paper-facing claims can now cite local artifacts under `results/`.
- Treat orphan or stale state as operational debt, not as evidence.
