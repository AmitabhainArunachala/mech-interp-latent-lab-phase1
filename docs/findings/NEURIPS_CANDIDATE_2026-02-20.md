# NeurIPS Candidate (2026-02-20)

## Candidate Claim
`gqa_headspace_specificity_bridge`

In Mistral-7B bridge interventions, correcting patching to GQA headspace (`v4_gqa_headspace`) separates mechanism-specific effects from control effects. Head-specific recursive donor patching decreases `rv_delta`, while random-head and baseline-donor controls increase it.

## Evidence (Offline Meta-Experiment)

Source run: `results/meta_yolo/runs/20260220_102900_offline_meta_yolo/summary.json`

1. `v4_head_specific_vs_v4_random_head`
- delta mean (`head - random`): `-0.03534277183091872`
- Welch p-value: `3.979575959566492e-07`
- Cohen's d: `-1.2539166694831796`

2. `v4_head_specific_vs_v4_baseline_donor`
- delta mean (`head - baseline_donor`): `-0.05172129187593341`
- Welch p-value: `3.9784014976513625e-08`
- Cohen's d: `-1.3701282954126641`

3. Control implementation flip (`v2 -> v4`)
- `random_head` mean in `v2_head_specific`: `-0.0339909016422978`
- `random_head` mean in `v4_gqa_headspace`: `0.011257600205766652`
- Sign flip: `True`
- Welch p-value (`v2 random` vs `v4 random`): `3.562916560633351e-11`

## Evidence (GPU Fast Bridge Replication, 2026-02-20)

Source runs:
- `results/remote_gpu_sync/2026-02-20/phase1_mechanism/20260220_062713_rv_l27_activation_patching_bridge_head_specific_bridge_fast/summary.json`
- `results/remote_gpu_sync/2026-02-20/phase1_mechanism/20260220_063343_rv_l27_activation_patching_bridge_random_head_bridge_control_fast/summary.json`
- `results/remote_gpu_sync/2026-02-20/phase1_mechanism/20260220_063955_rv_l27_activation_patching_bridge_baseline_donor_specificity_control_fast/summary.json`
- Pairwise tests: `results/remote_gpu_sync/2026-02-20/phase1_mechanism/contrast_stats.md`

Single-run deltas:
- `head_specific`: `rv_delta_mean = -0.02763009919286047`, `rv_p_value = 0.016671622344567236`, `rv_cohens_d = -1.1056815575720984`
- `random_head`: `rv_delta_mean = 0.023465900592108666`, `rv_p_value = 0.0069035501353712425`, `rv_cohens_d = 1.3359894032191306`
- `baseline_donor`: `rv_delta_mean = 0.029746171572397354`, `rv_p_value = 0.1427864936650304`, `rv_cohens_d = 0.583606973754712`

Cross-condition contrasts:
- `head_specific - random_head`: `-0.051096`, Welch `p = 0.000430594`, Cohen's d `= -2.3657`
- `head_specific - baseline_donor`: `-0.057376`, Welch `p = 0.0167042`, Cohen's d `= -1.4294`
- `random_head - baseline_donor`: `-0.006280`, Welch `p = 0.749631`, Cohen's d `= -0.1647`

## Evidence (Seed-Matrix Triad, Seed 42 Complete)

Source: `industry_grade/2026-02-20/evidence/seed_bridge_analysis.json`

Paired contrasts on matched prompt pairs (`n_overlap=60`):
- `head_specific - random_head_control`: `-0.038566`, paired `p = 1.786e-08`, `d = -0.841`
- `head_specific - baseline_donor_control`: `-0.055016`, paired `p = 2.029e-16`, `d = -1.461`
- `random_head_control - baseline_donor_control`: `-0.016450`, paired `p = 0.0284`, `d = -0.290`

Condition-level means (same seed):
- `head_specific`: `rv_delta_mean = -0.026541`
- `random_head_control`: `rv_delta_mean = +0.012025`
- `baseline_donor_control`: `rv_delta_mean = +0.028475`

Interpretation:
- The direction split required by the claim is already present in a full triad with strong paired significance for both primary contrasts.

## Parallel Signals

1. Cross-architecture contraction remains robust:
- 6/6 latest model runs have negative `rv_delta_mean`
- Sign test p-value: `0.015625`
- Random-effects pooled delta: `-0.1570030654050248`
- 95% CI: `[-0.25449181851806024, -0.059514312291989335]`

2. Multi-token bridge remains truncation-confounded:
- Spearman(`pct_truncated`, `h3_r`) = `-0.6063348416289592`, p = `0.021521097008774158`
- `h3` significance rate high truncation: `0.875`
- `h3` significance rate low truncation: `0.3333333333333333`

3. Fresh GPU multi-token rerun completed:
- Run: `results/remote_gpu_sync/2026-02-20/phase1_cross_architecture/20260220_071457_multi_token_bridge_mistral_7b_bridge_deconfound_fast/summary.json`
- `temp=0.0`: truncation `88.9%`, `h1 r=-0.6498 p=1.80e-05`, `h2 d=3.536 p=2.52e-12`
- `temp=0.7`: truncation `69.4%`, `h1 r=-0.4091 p=0.2115`, `h2 d=3.536 p=2.52e-12`
- Readout: recursive-vs-baseline `R_V` separation is very strong, but behavior-correlation evidence remains truncation-sensitive.

4. Embedding-based semantic scorer implemented and executed:
- Artifacts:
  - `industry_grade/2026-02-20/evidence/semantic_behavior_analysis.json`
  - `industry_grade/2026-02-20/evidence/semantic_bridge_scores_seed_bridge.csv`
  - `industry_grade/2026-02-20/evidence/semantic_bridge_scores_c2.csv`
- Method:
  - `all-MiniLM-L6-v2`, max cosine to 5 fixed `L5_refined` exemplars, threshold `> 0.4`.
- Seed-bridge (current synced runs): thresholded rate floor, but continuous signal present:
  - Spearman(`rv_patch`, semantic_score) `rho=-0.128`, `p=0.0266`, `n=300`
- C2 transfer: strong semantic signal:
  - `c2_full` semantic_recursive_rate `0.20` vs `baseline` `0.00`
  - Spearman(`rv_mean`, semantic_score) `rho=-0.652`, `p=1.44e-92`, `n=755`

## Why This Is Submission-Relevant

This points to a concrete mechanistic methodology result:
"Causal claims from head-level intervention can invert under incorrect headspace semantics; GQA-aligned intervention restores specificity."

That is a defensible, falsifiable claim with clear controls and direct implications for mechanistic interpretability methodology.

## Immediate Run Plan (when model access is available)

1. Re-run `v4_gqa_headspace` bridge conditions with independent seeds (`>=5`) and disjoint prompt subsets.
2. Run same control matrix on one non-GQA dense model to test architecture dependence of the effect.
3. Enforce low-truncation multi-token runs (`pct_truncated < 20%`) before using `h3` as evidence.
4. Add preregistered pass/fail thresholds for the above two bridge contrasts.
