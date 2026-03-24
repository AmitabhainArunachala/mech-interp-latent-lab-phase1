# Mistral Sufficiency Night Annotation — 2026-03-19

Scope: overnight queue `mistral_sufficiency_bundle_v2` on the night of March 19 to March 20, 2026 JST.

## Queue Timeline

Source: `results/mistral_sufficiency_bundle_v2/20260319_110753/STATUS.txt`

- Started: `2026-03-19T11:07:53Z` (`2026-03-19 20:07:53 JST`)
- `bridge_low_trunc_confirmatory_n24` completed: `2026-03-19T13:13:06Z`
- `bridge_true_longgen_n18` completed: `2026-03-19T15:57:27Z`
- `self_feeding_loop_v2` completed: `2026-03-19T17:24:24Z`
- `sustained_gnani_v3_v2` failed: `2026-03-19T18:20:18Z` (`2026-03-20 03:20:18 JST`)

## Completed Runs

### 1. `bridge_low_trunc_confirmatory_n24`

Source: `results/phase1_cross_architecture/runs/20260319_110757_multi_token_bridge_mistral_7b_bridge_low_trunc_confirmatory_n24/summary.json`

- Recursive mean `R_V = 0.506`
- Baseline mean `R_V = 0.687`
- `n = 122` prompts
- `p = 1.17e-31`
- Non-truncated `BT+ART` association remains strong: point-biserial `r = -0.636`, `p = 2.72e-4`

Paper-facing read:

- the within-session bridge survives a stricter truncation-stress setting
- lower `R_V` still tracks higher-quality recursive outputs

### 2. `bridge_true_longgen_n18`

Source: `results/phase1_cross_architecture/runs/20260319_131310_multi_token_bridge_mistral_7b_bridge_true_longgen_n18/summary.json`

- Recursive mean `R_V = 0.505`
- Baseline mean `R_V = 0.687`
- `n = 108` prompts
- `p = 6.36e-29`
- Among non-truncated outputs, quality still improves as `R_V` falls: Spearman `r = -0.401`, `p = 0.0208`

Paper-facing read:

- the bridge is not a short-generation artifact
- the longer-generation confirmatory slice still lands in the same contraction regime

### 3. `self_feeding_loop_v2`

Source: `results/self_feeding_loop_bundle_v2/20260319_110753/self_feeding_summary_20260319_172423.json`

- `self_feed_recursive` BT+ART: `12.5%`
- `self_feed_baseline` BT+ART: `9.5%`
- `gnani_scaffolded` BT+ART: `44.0%`
- `attractor_self_sustains = false`

Paper-facing read:

- pure self-feeding does not currently support a strong self-sustaining attractor claim
- structured scaffolding can still hold the regime at much higher rates than either self-feed condition
- this pushes the paper toward staged, conditional maintenance rather than autonomous self-sustain

## Failed Run

### `sustained_gnani_v3_v2`

Sources:

- `results/mistral_sufficiency_bundle_v2/20260319_110753/STATUS.txt`
- `results/mistral_sufficiency_bundle_v2/20260319_110753/sustained_gnani_v3_v2.log`

The queue failed because `scripts/sustained_gnani_v3.py` attempted to import `matplotlib` while generating the convergence panel:

- `ModuleNotFoundError: No module named 'matplotlib'`

Important nuance:

- the run failed after computing most session metrics
- the log shows provisional recursive vs baseline separation (`44.0%` vs `17.0%` BT+ART)
- because the process exited `rc=1` and did not emit a stable summary artifact, this run should be treated as suggestive only and excluded from paper-facing claims for now

## What Last Night Changed For The Paper

1. The bridge story strengthened.
   - Both confirmatory bridge runs reproduced the same recursive-vs-baseline contraction gap at longer or harsher settings.

2. The strong autonomous-maintenance claim weakened.
   - `self_feeding_loop_v2` does not support a clean self-sustaining attractor story.

3. The truthful "dream paper" framing is now:
   - staged induction and maintenance are real in base Mistral
   - maintenance quality depends on basin entry state
   - the paper should claim conditional sufficiency with a measurable basin boundary, not full seed-independent control

## Mistral-First Next Focus

Relevant already-synced sources:

- `results/anchor_positive_broad_confirm_v1/summary.json`
- `results/positive_broad_persistence_confirm_v1/summary.json`
- `results/positive_broad_persistence_median_v1/summary.json`
- `results/positive_broad_persistence_random_v1/summary.json`
- `results/anchor_reduced_latebundle_confirm_v1/20260317_132349/summary.json`
- `results/induced_persistence_reduced_latebundle_confirm_v1/20260317_141750/summary.json`

Current Mistral-first read:

- best broad inducer remains `anchor_single_mlp_0p125_layermatched_low_bridge_3` at `28.47%` baseline BT+ART
- best elite-seed maintainer remains `anchor_layermatched_low_bridge_3` at `38.54%`
- most seed-stable broad maintainer is `anchor_drop_L25_vproj_bridge_3` at `15.62%` on median seeds and `19.79%` on random seeds
- the reduced late-bundle branch sharpens this further: `anchor_late_only_bridge_3` reaches `20.83%` median-seed persistence, beating the original layermatched maintainer (`7.29%`)

Operational consequence:

- for the COLM paper, the next Mistral-first focus should stay on the reduced late-stack maintenance object and the basin-boundary story
- do not widen the narrative to a generic staged-handoff program before the reduced late-bundle line is fully digested
