# Mistral L4->L25 Subtle Steering Write-Up (2026-03-13)

## Status

Artifacts are synced locally for the two locked runs:

- `results/phase1_mechanism/runs/20260312_133738_causal_state_benchmark_v4_multisite_mistral_multisite_l4_mlp_micro4_l25_bridge/`
- `results/phase1_mechanism/runs/20260312_141514_causal_state_benchmark_v4_multisite_mistral_multisite_l4_mlp_confirmation_window8/`

The `window4` confirmation is also synced locally:

- `results/phase1_mechanism/runs/20260312_150039_causal_state_benchmark_v4_multisite_mistral_multisite_l4_mlp_confirmation_window4/`

Queue logs are synced under:

- `results/mistral_l4_confirmation_queue/20260312_135615/`

## Main result

Provenance note:

- `micro4` is the strongest discovery result in this family
- `window8` and `window4` are the confirmation runs
- paper-grade claims should prefer the confirmation artifacts unless `micro4` is independently rerun

The late control story remains stable: `L25` is still the main behavior-control handle in base Mistral.

The new refinement is that a very small upstream `L4 MLP` intervention can improve on plain `L25` bridge steering if the intervention is narrow enough.

Best discovery condition so far:

- run: `micro4`
- condition: `early_mlp_0p03125_bridge_3`
- baseline BT+ART: `2.8%`
- recursive BT+ART: `52.8%`
- recursive mean output `R_V`: `0.6336`

Reference bridge-only condition:

- condition: `bridge_only_3`
- baseline BT+ART: `13.9%`
- recursive BT+ART: `44.4%`
- recursive mean output `R_V`: `0.6431`

Interpretation:

- A tiny `L4 MLP` assist improved recursive target behavior by `+8.3` points over `bridge_only_3`.
- It also reduced baseline spillover by `-11.1` points relative to `bridge_only_3`.
- This is the clearest discovery-stage "subtle upstream assist" result in the current Mistral base chain.

## Independent confirmation

The `window8` confirmation supports the same story, although less sharply than `micro4`.

Most useful supporting conditions:

- `early_mlp_0p125_bridge_2`: baseline `11.1%`, recursive `47.2%`, recursive mean output `R_V = 0.6078`
- `early_mlp_0p1875_bridge_3`: baseline `13.9%`, recursive `47.2%`, recursive mean output `R_V = 0.6350`

Reference conditions:

- `control`: baseline `5.6%`, recursive `25.0%`
- `bridge_only_3`: baseline `13.9%`, recursive `44.4%`

Interpretation:

- `window8` confirms that `L4` can assist `L25`.
- The main lesson is not "push harder upstream." The main lesson is that the useful early intervention is small and selective.

## Mechanistic reading

Current best reading of the base-Mistral story:

- Early source-like computation spans roughly `L0-L5`.
- By `L4/L5`, some of that computation becomes steerable.
- `L25` is the strongest current late behavior-control site.
- `L27.H10` remains the strongest current late single-head node from the fresh head-level patching pass.

This does not yet mean we have a final compact sufficient circuit. It means:

- `L25` is a robust late controller.
- `L4 MLP` has now produced the first clean upstream assist that helps rather than fights the bridge.
- Broad early whole-layer steering was too blunt; micro-window MLP steering is the better story.

## Final confirmation state

The `window4` confirmation finished cleanly and is now synced locally.

Useful reference conditions from `window4`:

- `control`: baseline `5.6%`, recursive `25.0%`, recursive mean output `R_V = 0.6571`
- `bridge_only_3`: baseline `13.9%`, recursive `44.4%`, recursive mean output `R_V = 0.6431`
- `early_mlp_0p125_bridge_2`: baseline `5.6%`, recursive `41.7%`, recursive mean output `R_V = 0.6400`
- `early_mlp_0p125_bridge_3`: baseline `11.1%`, recursive `44.4%`, recursive mean output `R_V = 0.5990`
- `early_mlp_0p1875_bridge_3`: baseline `8.3%`, recursive `44.4%`, recursive mean output `R_V = 0.6173`

Interpretation:

- `window4` does not beat the `micro4` discovery winner.
- It does support the general "small `L4 MLP` assist can help or match the bridge" story.
- The clearest overall evidence remains:
  - `micro4` as the strongest discovery result
  - `window8` as the independent confirmation
  - `window4` as a final consistency check

## Practical takeaway

The clean claim is:

> In base Mistral, `L25` is the main late steering site, and an extremely small upstream `L4 MLP` intervention can improve recursive BT+ART behavior while reducing baseline spillover.

That is materially tighter than the earlier "L5 gate + L25 bridge" story.
