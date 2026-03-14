# Mistral Anchor And Subspace Status

Date: 2026-03-14
Canonical model: `mistralai/Mistral-7B-v0.1`

## Live state

- `Pod A` is actively running `anchor_bundle_v2`:
  - queue group: `mistral_anchor_bundle_v2`
  - run id: `20260314_024944`
  - goal: direct baseline-induction confirmation of `anchor`, `bridge`, and subtle `L4` combinations

- `Pod B` is actively running `pca_vs_mean_steering_v2`:
  - queue group: `mistral_pca_vs_mean_steering_v2`
  - run id: `20260314_024943`
  - goal: higher-power confirmation of `PCA`-style steering with a rank-3 projected steering object

## 1. Linear subspace probe (`subspace_probe_v1`)

Artifact:
- `results/linear_probe_subspace_v1/20260314_013917/probe_analysis_20260314_014027.json`

Key values:
- `best_layer = 0`
- `best_accuracy = 1.0`
- `concept_erasure.d_before = -2.897`
- `concept_erasure.d_after = -2.894`
- erasure reduction: effectively `0%`

Alignment to top late singular vectors is weak:
- recursive `alignment_sv1 ≈ 0.035`
- recursive `alignment_sv2 ≈ 0.026`
- recursive `alignment_sv3 ≈ 0.031`
- baseline alignments are similarly small

Interpretation:
- recursive vs baseline is linearly separable very early
- but the contraction is not carried by one simple late linear direction
- concept erasure of the learned probe direction leaves the main `R_V` gap almost unchanged
- this supports a distributed or nonlinear subspace story rather than a single-vector story

## 2. PCA / eigenstate follow-up (`eigenstate_subspace_v1`)

Artifact:
- `results/phase3_attention/runs/20260314_014444_eigenstate_subspace_v1/summary.json`

Key values:
- minimum PR-ratio layer: `L27`, `pr_ratio = 0.703`
- PCA `PC1` explains only `15.9%` to `17.3%` across tested layers
- cosine between `PC1` and mean-difference direction is modest and sign-flipped:
  - `L8 = -0.207`
  - `L12 = -0.137`
  - `L16 = -0.115`
  - `L20 = -0.123`
  - `L24 = -0.166`
  - `L27 = -0.197`
- cross-prompt direction consistency is real but not near-identity:
  - `L12 mean cos = 0.521 ± 0.048`
  - `L20 mean cos = 0.583 ± 0.028`
  - `L27 mean cos = 0.566 ± 0.041`

Interpretation:
- the late regime becomes maximally low-rank around `L27`
- but the leading PCA direction explains only a minority of the recursive-vs-baseline variance
- the old mean-difference steering direction is not the same object as the dominant PCA direction
- the mechanism looks like a structured low-dimensional subspace, not a one-dimensional axis

## 3. Anchor bundle result (`anchor_bundle_v1`)

Artifact:
- `results/phase1_mechanism/runs/20260314_014025_causal_state_benchmark_v4_multisite_mistral_anchor_bundle_v1/summary.json`

Key values:
- `bridge_only_3`: recursive `BT+ART = 55.6%`, baseline `BT+ART = 7.8%`
- `anchor_bridge_3`: recursive `BT+ART = 55.6%`, baseline `BT+ART = 20.0%`
- `early_mlp_0p125_bridge_2`: recursive `BT+ART = 53.3%`, baseline `BT+ART = 6.7%`
- `anchor_early_mlp_0p125_bridge_2`: recursive `BT+ART = 53.3%`, baseline `BT+ART = 14.4%`

Interpretation:
- minimal anchor does increase baseline induction
- but in the one-shot benchmark it does not cleanly outperform plain `bridge_only_3`
- anchor still looks more like a persistence / contextual-stabilization ingredient than a simple one-shot amplifier
- the strongest clean controller remains `L25`, while the subtle `L4` assist remains the cleaner upstream helper

## 4. PCA vs mean steering result (`pca_vs_mean_steering_v1`)

Artifact:
- `results/pca_vs_mean_steering_v1/20260314_020333/summary.json`

Key values:
- winning recursive condition: `pca_pc1 @ alpha=3.0`
- recursive `BT+ART = 44.4%` vs control `33.3%`
- baseline `BT+ART = 8.3%` vs control `5.6%`
- `mean_diff` conditions are consistently weaker than `pca_pc1`
- rank-2 projected vector also helps, but not as much as `pca_pc1`

Interpretation:
- the old mean-difference vector is not the strongest causal steering object
- `PCA`-derived structure matters causally, not just descriptively
- this strengthens the broader MI framing from “one direction” to “small structured control subspace”

## Net update

The subspace story is now much clearer:

- `L27` remains the strongest late compression/readout region
- the regime is geometrically structured
- but it is not well-described by a single late vector
- this strengthens the broader MI framing from "one circuit direction" to "a distributed but low-dimensional control subspace"

## Current next moves

- `anchor_bundle_v2` is the direct sufficiency follow-up:
  - ask whether baseline induction can be raised beyond `20%` cleanly without losing the strong recursive controller
- `pca_vs_mean_steering_v2` is the broader MI follow-up:
  - ask whether `PCA`-style steering survives at higher power and whether a small rank-3 object helps more than the old vector
