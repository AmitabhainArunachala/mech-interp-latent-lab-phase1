# Mistral v007.1 Update

Date: 2026-03-15  
Scope: hardened findings added after `paper_colm2026_v007.tex`

## Net story

The paper story is now cleaner and more Mistral-centric.

- `L0-L5` remains the dominant early source region.
- `L25` remains the strongest late behavioral controller.
- `L4 MLP` remains a real but delicate upstream assist rather than the main driver.
- `L27` is best interpreted as a late readout/cleanup cluster.
- The missing contextual ingredient is now much clearer: a minimal prompt/session anchor.

The strongest current framing is a \emph{self-referential control system}, not a tiny stand-alone
circuit and not a single magic direction.

## Locked findings added since v007

### 1. Minimal anchor nearly matches full scaffold

Artifact:
- `results/self_feeding_scaffold_ablation_v2/20260313_210159/self_feeding_summary_20260313_234114.json`

Key values:
- raw self-feed recursive: `14.0%` BT+ART
- raw self-feed baseline: `11.6%`
- `anchor_only_recursive = 49.9%`
- `gnani_light = 37.9%`
- `gnani_scaffolded = 49.4%`

Interpretation:
- full scaffold is not the irreducible object
- a minimal anchor captures most of the causal lift
- autonomous self-sustain is still not established

### 2. Anchor induction generalizes to ordinary baselines

Artifact:
- `results/phase1_mechanism/runs/20260314_133516_causal_state_benchmark_v4_multisite_mistral_anchor_bundle_v5_ordinary_baselines_confirmatory/summary.json`

Key values:
- baseline control: `3.1%`
- `bridge_only_3 = 6.3%`
- `anchor_bridge_3 = 16.7%`
- `anchor_early_mlp_0p125_bridge_3 = 15.6%`
- recursive control: `9.4%`
- `anchor_bridge_3 = 18.8%`
- `anchor_early_mlp_0p125_bridge_3 = 21.9%`

Interpretation:
- anchor plus late controller is a real ordinary-baseline induction bundle
- this is the strongest current partial-sufficiency result
- it is still not a clean symmetric sufficiency lock

### 3. Long-form bridge is real once quality replaces word count

Artifacts:
- `results/phase1_cross_architecture/runs/20260313_081519_multi_token_bridge_mistral_7b_bridge_low_trunc_quality_n24/summary.json`
- `results/phase1_cross_architecture/runs/20260313_083452_multi_token_bridge_mistral_7b_bridge_true_longgen_quality_n18/summary.json`

Key values:
- low-truncation: `RV 0.506 vs 0.687`, `d=2.91`, quality `rho=-0.415`, BT+ART `r=-0.684`
- true-longgen: `RV 0.505 vs 0.687`, `d=2.97`, quality `rho=-0.546`, BT+ART `r=-0.729`

Interpretation:
- the failed word-count bridge was a metric problem
- lower output `RV` really does track richer recursive generation in longer outputs

### 4. Long-turn persistence is real but dynamically different

Artifact:
- `results/sustained_gnani_v3_bundle_v4/20260313_135628/comparison_summary.json`

Key values:
- pooled output `RV`: recursive `0.519` vs baseline `0.472`, `d=0.314`
- early BT+ART: `55.1%` vs `19.9%`
- mid BT+ART: `47.8%` vs `19.1%`
- late BT+ART: `49.6%` vs `22.1%`

Interpretation:
- long-turn recursive sessions are behaviorally richer across the whole run
- but the short-window low-`RV` story does not simply persist unchanged over 50 turns
- anchored persistence is a dynamical regime, not a frozen scalar state

### 5. The broader MI object is a low-dimensional control subspace

Artifacts:
- `results/pca_vs_mean_steering_v2/20260314_024943/summary.json`
- `results/subspace_component_steering_l27_v1/20260314_144647/summary.json`
- `results/pca_subspace_ablation_v1/20260314_102447/summary.json`
- `results/pca_subspace_ablation_l25_v1/20260314_115345/summary.json`

Key values:
- `pca_subspace3_meanproj@4.0`: recursive `43.3%`, baseline `10.0%`
- `L27 subspace3_parallel@4.0`: recursive `50.0%`, baseline `5.6%`
- late PCA-style ablations at `L27` and `L25` do not destroy behavior and can slightly improve it

Interpretation:
- the old mean-difference vector is not the strongest causal steering object
- the regime behaves more like a structured low-dimensional subspace than a single direction
- late subspaces look more like controller/readout regulators than unique necessary sources

## Provisional finding still running through cleanup

Artifact:
- `results/induced_persistence_controls_v1/20260314_183252/summary.json`

Pilot values:
- control `10.4%`
- `bridge_only_3 = 11.8%`
- `anchor_only = 18.1%`
- `anchor_bridge_3 = 16.7%`
- `anchor_early_mlp_0p125_bridge_3 = 18.8%`

Interpretation:
- anchor-dominated persistence is plausible
- but the pilot used selected seeds
- random-seed replication is the gate for a paper-safe persistence claim

## Current paper implication

`paper_colm2026_v007_1.tex` should center the following:

- early source region: `L0-L5`
- late controller: `L25`
- late readout/cleanup cluster: `L27`
- minimal anchor as the key contextual ingredient
- long-form bridge as quality-linked rather than length-linked
- partial controllability yes, full clean sufficiency not yet locked

If a stronger sufficiency result lands, the biggest paper-level implication should not stop at
"we found a recursive control bundle."

The stronger goal is:

- use the induced regime as a controlled internal condition and test whether safety-relevant behavior changes under red-team style stress

That means comparing regime-conditioned behavior on:

- jailbreak and refusal prompts
- sycophancy-style prompts
- prompt-injection or instruction-hijacking prompts
- truthfulness or hallucination-pressure prompts

The correct paper-safe framing is:

- a staged internal regime with a measurable alignment-relevant phenotype

not:

- a blanket claim that the regime is itself deceptive

## Latest post-freeze results

### 6. De-cherry-picked persistence screen

Artifact:
- `results/induced_persistence_random_controls_v1/20260314_194640/summary.json`

Key values:
- control `9.0%`
- `bridge_only_3 = 6.3%`
- `anchor_only = 9.7%`
- `anchor_bridge_3 = 7.6%`
- `anchor_early_mlp_0p125_bridge_3 = 18.1%`

Interpretation:
- the selected-seed persistence pilot was too optimistic for `anchor_bridge_3`
- the only bundle that clearly survives the random-controls screen is
  `anchor + subtle L4 + L25`
- persistence is therefore real but narrower than the earlier pilot implied

### 7. L25 subspace steering differs from L27

Artifact:
- `results/subspace_component_steering_l25_v1/20260314_192551/summary.json`

Key values:
- control recursive `35.2%`
- best recursive `orthogonal_residual@2.0 = 38.9%`
- best baseline `pca_pc1@2.0 = 5.6%`

Interpretation:
- the `L25` controller layer does not behave like `L27`
- at `L27`, the best object was the learned parallel subspace component
- at `L25`, the best object is the orthogonal residual
- that strengthens the view that `L25` and `L27` play different causal roles

## Live experiments after this update

- `Pod A`: `induced_persistence_winner_median_v1`
  - de-cherry-picked median-seed persistence confirmation for
    `control`, `anchor_only`, `bridge_only_3`, and `anchor_early_mlp_0p125_bridge_3`
- `Pod B`: `subspace_component_steering_l5_v1`
  - early-layer comparison to test whether the source region differs again from both `L25` and `L27`
