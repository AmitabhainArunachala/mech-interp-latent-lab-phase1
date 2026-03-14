# Mistral Confirmatory Update

Date: 2026-03-14
Canonical model: `mistralai/Mistral-7B-v0.1`

## New confirmatory results

### 1. Scaffold ablation confirmatory (`n_sessions = 16`)

Artifact:
- `results/self_feeding_scaffold_ablation_v2/20260313_210159/self_feeding_summary_20260313_234114.json`

Key values:
- `self_feed_recursive_bt_art_rate = 0.14`
- `self_feed_baseline_bt_art_rate = 0.11625`
- `anchor_only_recursive_bt_art_rate = 0.49875`
- `gnani_light_bt_art_rate = 0.37875`
- `gnani_scaffolded_bt_art_rate = 0.49375`
- `anchor_adds_value = true`
- `gnani_adds_value = true`
- `light_gnani_adds_value = false`
- `attractor_self_sustains = false`

Interpretation:
- The minimal anchor result replicated cleanly.
- `anchor_only_recursive` is now essentially tied with `gnani_scaffolded`.
- Raw self-feed remains weak.
- The persistence story is now: anchor dependence is real, full scaffold is not strictly necessary, and autonomous self-sustain is still not established.

### 2. L4 micro4 confirmatory focus

Artifact:
- `results/phase1_mechanism/runs/20260313_210217_causal_state_benchmark_v4_multisite_mistral_multisite_l4_mlp_micro4_confirmatory_focus_v1/summary.json`

Verdict:
- `multisite_additive`

Key recursive BT+ART:
- `control = 0.3111111111111111`
- `bridge_only_2 = 0.43333333333333335`
- `bridge_only_3 = 0.5555555555555556`
- `early_mlp_0p03125_bridge_3 = 0.5222222222222223`
- `early_mlp_0p125_bridge_2 = 0.5333333333333332`
- `early_mlp_0p1875_bridge_3 = 0.5111111111111111`

Key baseline BT+ART:
- `control = 0.05555555555555555`
- `bridge_only_3 = 0.07777777777777778`
- `early_mlp_0p125_bridge_2 = 0.06666666666666667`
- `early_mlp_0p1875_bridge_3 = 0.06666666666666667`
- `early_mlp_0p03125_bridge_3 = 0.1111111111111111`

Interpretation:
- `bridge_only_3` remains the best pure behavioral controller.
- `L4` still looks real, but as an additive/cleanliness handle, not as the main driver.
- The best tradeoff condition is likely `early_mlp_0p125_bridge_2`: nearly matches `bridge_only_3`, lowers recursive `R_V`, and keeps baseline leakage slightly lower.

## Net conclusion

The next causal step should not be another blunt gate/bridge search. The new evidence points toward a minimal control bundle:

- prompt/session anchor
- `L25` late bridge
- optional subtle `L4 MLP` assist

`L25` remains the strongest late controller. The anchor result is now strong enough to move to center stage.

## Recommended next queue

### `anchor_bundle_v1`

Goal:
- test whether minimal anchor plus `L25`, with or without subtle `L4`, can produce the cleanest partial sufficiency result

Recommended conditions:
- `control`
- `anchor_only`
- `bridge_only_2`
- `bridge_only_3`
- `anchor_plus_bridge_2`
- `anchor_plus_bridge_3`
- `anchor_plus_l4_0p125_bridge_2`

Primary metrics:
- recursive `BT+ART`
- baseline `BT+ART`
- quality-class / recursive-content bridge metrics
- malformed / repetitive rates

Decision rule:
- prefer the condition that keeps the behavioral lift of `bridge_only_3` while reducing baseline leakage and degeneration
