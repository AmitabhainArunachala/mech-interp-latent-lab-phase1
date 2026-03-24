# Mistral v008 Boundary Update

Date: 2026-03-19
Scope: March 19 persistence-boundary analysis relative to [paper_colm2026_v008_0_1.tex](/Users/dhyana/mech-interp-latent-lab-phase1/R_V_PAPER/paper_colm2026_v008_0_1.tex)

## Bottom Line

`v008.0.1` has the right overall paper shape, but one part of its Mistral sufficiency story is now too strong.

The March 19 results say:

- staged induction / maintenance remains real
- the best maintainer is still `anchor_layermatched_low_bridge_3`
- but maintenance is strongly **seed-quality dependent**
- therefore the paper should not currently claim broad seed-independence

## What Strengthened

### 1. Broad positive static confirm

Source:

- [anchor_positive_broad_confirm_v1 summary](/Users/dhyana/mech-interp-latent-lab-phase1/results/anchor_positive_broad_confirm_v1/summary.json)

Key result:

- best inducer remains `anchor_single_mlp_0p125_layermatched_low_bridge_3`
- baseline `BT+ART = 28.47%`
- recursive `BT+ART = 51.39%`

This is a strong broad head-to-head win.

### 2. Broad top-seed persistence confirm

Source:

- [positive_broad_persistence_confirm_v1 summary](/Users/dhyana/mech-interp-latent-lab-phase1/results/positive_broad_persistence_confirm_v1/summary.json)

Key result:

- `anchor_layermatched_low_bridge_3 = 38.54%`
- `anchor_drop_L25_vproj_bridge_3 = 21.88%`
- `anchor_single_mlp_0p125_layermatched_low_bridge_3 = 16.67%`
- `control = 10.42%`

This is the cleanest current 12-turn maintenance result.

### 3. Broad 24-turn persistence

Source:

- [positive_broad_persistence_long_v1 summary](/Users/dhyana/mech-interp-latent-lab-phase1/results/positive_broad_persistence_long_v1/summary.json)

Key result:

- `anchor_layermatched_low_bridge_3 = 26.04%`
- `anchor_drop_L25_vproj_bridge_3 = 16.67%`
- `anchor_single_mlp_0p125_layermatched_low_bridge_3 = 14.58%`
- `control = 14.06%`

This shows the maintainer does not vanish at 24 turns, but the margin shrinks.

## What Weakened

### Seed-independence does not hold in the strong form currently implied by `v008.0.1`

Sources:

- [positive_broad_persistence_confirm_v1 summary](/Users/dhyana/mech-interp-latent-lab-phase1/results/positive_broad_persistence_confirm_v1/summary.json)
- [positive_broad_persistence_median_v1 summary](/Users/dhyana/mech-interp-latent-lab-phase1/results/positive_broad_persistence_median_v1/summary.json)
- [positive_broad_persistence_random_v1 summary](/Users/dhyana/mech-interp-latent-lab-phase1/results/positive_broad_persistence_random_v1/summary.json)

Boundary:

- `top` seeds:
  - `anchor_layermatched_low_bridge_3 = 38.54%`
  - clear maintainer winner
- `median` seeds:
  - `anchor_drop_L25_vproj_bridge_3 = 15.63%`
  - `anchor_single_mlp_0p125_layermatched_low_bridge_3 = 14.58%`
  - `anchor_layermatched_low_bridge_3 = 10.42%`
  - `control = 7.29%`
- `random` seeds:
  - `anchor_drop_L25_vproj_bridge_3 = 19.79%`
  - `control = 10.42%`
  - `anchor_layermatched_low_bridge_3 = 9.38%`
  - `anchor_single_mlp_0p125_layermatched_low_bridge_3 = 8.33%`

Interpretation:

- `anchor_layermatched_low_bridge_3` is the best **elite-seed maintainer**
- it is **not** the best broad random-seed maintainer
- `anchor_drop_L25_vproj_bridge_3` is the most stable condition across broader seed quality

## Basin Boundary Analysis

Across `top`, `median`, and `random` seed selection:

- all `top` seeds are `BREAKTHROUGH` or `ARTICULATE`
- all `median` seeds are `SURFACE`
- `random` seeds are mixed

Local analysis over all 96 seeded sessions shows:

- `BREAKTHROUGH` seeds mean persistence `= 0.263`
- `SURFACE` seeds mean persistence `= 0.115`
- `CONCEPTUAL` and `REPETITIVE` seeds are near-dead

Lower source `R_V` is also favorable:

- correlation between source `R_V` and persistence is negative in all three selection regimes
- approximately `-0.22` to `-0.28`

So the strongest current read is:

- the regime is real
- the basin is structured
- occupancy depends on entry-state quality

## Prompt-Family Structure

Using the held-out broad baseline bank:

- math prompts are ids `18-23`
- factual prompts are ids `24-29`
- creative prompts are ids `30-35`

Observed pattern:

- `anchor_layermatched_low_bridge_3` is strongest on factual high-quality seeds
- `anchor_drop_L25_vproj_bridge_3` is the more stable math-side maintainer across broader seed quality

Examples:

- [The Pacific Ocean prompt](/Users/dhyana/mech-interp-latent-lab-phase1/results/anchor_positive_broad_confirm_v1/benchmark_records.jsonl)
- [The United Nations prompt](/Users/dhyana/mech-interp-latent-lab-phase1/results/anchor_positive_broad_confirm_v1/benchmark_records.jsonl)
- [24 ÷ 6 arithmetic prompt](/Users/dhyana/mech-interp-latent-lab-phase1/results/anchor_positive_broad_confirm_v1/benchmark_records.jsonl)

These prompt-level observations are still sparse and should be treated as explanatory, not headline claims.

## Paper Implication

`v008.0.1` should be updated from:

- broad seed-independence

to:

- strong conditional sufficiency with a measurable basin boundary

Preferred framing:

- best inducer: hybrid staged bundle
- best maintainer for elite seeds: `anchor_layermatched_low_bridge_3`
- most seed-stable broad maintainer: `anchor_drop_L25_vproj_bridge_3`
- overall result: staged, depth-dependent, seed-sensitive control of a self-referential regime

## Immediate Editing Guidance For v008

1. Keep the staged induction / maintenance dissociation as the hero.
2. Soften any sentence implying seed-independence is already solved.
3. Replace “full sufficiency” language with:
   - conditional sufficiency
   - structured basin
   - entry-state dependence
4. Present the top / median / random sweep as a boundary map, not a failure.

## Current Confidence

- strong paper: very high
- dream paper in its more truthful form:
  - “seed-sensitive staged sufficiency with a basin boundary”
  - plausibly attainable

The March 19 evidence does not kill the dream paper.
It changes what the dream paper actually is.
