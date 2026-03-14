Title: Cross-Model Numeric Deep Dive
Date: 2026-03-11
Purpose: Deep provenance pass across model families, pipelines, and headline numeric stories after `paper_colm2026_v006.tex` contamination.

## Executive Findings

1. The paper is numerically contaminated right now.
   - `R_V_PAPER/paper_colm2026_v006.tex:289` says `Mistral-7B-v0.1` is the primary model throughout.
   - `R_V_PAPER/paper_colm2026_v006.tex:523-535` uses `Mistral-7B-Instruct-v0.2` dual-layer numbers from `results/persistent_patching_v3/persistent_patching_v3_dual_20260310_204100.json`.

2. The strongest cross-model story that survives direct file checks is:
   - `Mistral-7B` and `Qwen2.5-7B`: contraction in both cross-arch and power-up.
   - `OPT-6.7B` and `GPT-2 XL`: genuine sign reversal across pipelines.
   - `Pythia-1.4B`: null in power-up; cross-arch summary is internally inconsistent.
   - `Gemma-2-9B`: raw cross-arch-style summaries support strong contraction, but not the paper's `-3.37`.
   - `Mixtral-8x7B`: no comparable raw summary located in this pass.

3. The strongest Mistral mechanistic story that survives direct file checks is:
   - early residual BREAK effects are stable and strong
   - the `L27H10` contractive / `L5H29` expansive SVD motif is stable
   - head-sweep headline identities are not stable across family or sample size

4. There is no local raw base-family `p0_canonical n100` artifact at the moment.
   - Located local `p0_canonical` files are only for `mistralai/Mistral-7B-Instruct-v0.2`.

## Lane 1: Mistral Family Split

### Base family currently anchored by
- `results/phase1_cross_architecture/runs/20260202_121604_rv_l27_causal_validation_mistral_7b/summary.json`
  - `model = mistralai/Mistral-7B-v0.1`
  - `rv_cohens_d = -2.259009432780737`
  - `rv_recursive_mean = 0.5080480927874367`
  - `rv_baseline_mean = 0.6938362806665048`
  - `n_pairs = 45`
- `results/power_up/mistral-7b_n80_result.json`
  - `cohens_d = -1.6564878536967445`
  - `rv_recursive_mean = 0.685975896685412`
  - `rv_baseline_mean = 0.8550126067144158`
  - `n_recursive = 75`
  - `n_baseline = 77`

### Instruct family currently anchored by
- `results/p0_canonical/mistralai__Mistral-7B-Instruct-v0-2_p0_result.json`
  - `hedges_g = -1.417503`
  - `95% CI = [-1.766751, -1.127628]`
  - `n_selfref_valid = 96`
  - `n_baseline_valid = 100`
- `results/persistent_patching_v3/persistent_patching_v3_dual_20260311_130612.json`
  - `recursive_clean BT+ART = 52.0%`
  - `recursive_dual_patched BT+ART = 0.0%`
  - `baseline_clean BT+ART = 3.67%`
  - `baseline_dual_patched BT+ART = 0.0%`
  - `break session d = 3.8838508623093597`
  - `induce session p = 0.2105305888274869`
  - `recursive_dual_patched repetitive_rate = 1.0`
  - `baseline_dual_patched repetitive_rate = 1.0`

### Hard conclusion
- The paper must pick a family or explicitly separate them.
- Current v006 does neither.

## Lane 2: Cross-Model Pipeline Story

| Model | Cross-arch raw file | Cross-arch result | Power-up raw file | Power-up result | Surviving story |
| --- | --- | --- | --- | --- | --- |
| Mistral-7B | `results/phase1_cross_architecture/runs/20260202_121604_rv_l27_causal_validation_mistral_7b/summary.json` | `d=-2.2590`, rec `0.5080` < base `0.6938`, `n_pairs=45` | `results/power_up/mistral-7b_n80_result.json` | `d=-1.6565`, rec `0.6860` < base `0.8550`, `75/77` | stable contraction |
| Qwen2.5-7B | `results/phase1_cross_architecture/runs/20260202_120856_rv_l27_causal_validation_qwen2_7b/summary.json` | `d=-0.7185`, rec `1.1574` < base `1.2562`, `n_pairs=45` | `results/power_up/qwen2.5-7b_n80_result.json` | `d=-2.3181`, rec `0.9031` < base `1.3292`, `61/63` | stable contraction |
| OPT-6.7B | `results/phase1_cross_architecture/runs/20260202_125718_rv_l27_causal_validation_opt_6_7b/summary.json` | `d=-1.8359`, rec `0.9400` < base `1.2003`, `n_pairs=45` | `results/power_up/opt-6.7b_n80_result.json` | `d=+1.6825`, rec `1.1150` > base `0.7892`, `72/66` | real sign reversal |
| GPT-2 XL | `results/phase1_cross_architecture/runs/20260202_130807_rv_l27_causal_validation_gpt2_xl/summary.json` | `d=-1.1425`, rec `0.7671` < base `0.8510`, `n_pairs=45` | `results/power_up/gpt2-xl_n80_result.json` | `d=+1.5163`, rec `0.8723` > base `0.7110`, `69/56` | real sign reversal |
| Pythia-1.4B | `results/phase1_cross_architecture/runs/20260202_115958_rv_l27_causal_validation_pythia_1_4b/summary.json` | `d=-0.3109`, but rec `0.4193` > base `0.3796`, `n_pairs=45` | `results/power_up/pythia-1.4b_n80_result.json` | `d=-0.0057`, rec `0.6329` approx base `0.6331`, `66/54` | power-up null; cross-arch file needs review |

## Lane 3: Gemma, Mixtral, And Scaling Edge Cases

### Gemma-2-9B

Located raw summaries:
- `results/phase2_generalization/gemma_2_9b/11_causal_validation_champion/runs/20260124_112226_rv_l27_causal_validation_gemma_2_9b_causal_validation_champion/summary.json`
  - `rv_cohens_d = -1.7356227754504767`
  - `rv_recursive_mean = 0.5930448668476814`
  - `rv_baseline_mean = 0.7684291904438101`
  - `n_pairs = 60`
- `results/phase2_generalization/gemma_2_9b/08_causal_validation_n45/runs/20260124_100037_rv_l27_causal_validation_gemma_2_9b_causal_validation_n45/summary.json`
  - `rv_cohens_d = -1.9080790173305877`
  - `n_pairs = 45`

What did not verify:
- the paper's `Gemma-2-9B d=-3.37`

Where `3.37` actually appears:
- session docs such as `docs/sessions/2026-01-24_gemma_multi_token_bridge_v3.md:17`
- that appears to be a different Gemma behavioral/correlation result, not the raw cross-architecture summary used in Table 1

### Mixtral-8x7B

What was found:
- `results/phase2_generalization/mixtral_8x7b_v0_1/01_baseline_rv/runs/20260115_232500_cross_architecture_validation_mixtral_8x7b_v0_1_baseline_rv/config.json`

What was not found:
- a comparable raw `summary.json` with `rv_cohens_d` suitable for the paper's cross-model table

### Qwen2.5-3B scaling row

Raw file:
- `results/scaling_gap/qwen2.5-3b_result.json`
  - `cohens_d = +1.6019301588133141`
  - `rv_recursive_mean = 1.1967844614753727`
  - `rv_baseline_mean = 0.9871670566391431`
  - `n_recursive = 19`
  - `n_baseline = 18`

Hard conclusion:
- `paper_colm2026_v006.tex:435-438` currently uses a positive expansion result to motivate a threshold for reliable contraction.

## Lane 4: Mistral Head Sweep Drift

| File | Family | n | Entropy sig | Rank sig | Either sig | Top entropy | Top rank |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `results/full_head_sweep/full_head_sweep_20260302_074757.json` | `BASE_V01` | `20/20` | `606` | `169` | `681` | `L10H20 d=+3.9011` | `L0H5 d=-0.9321` |
| `results/full_head_sweep/full_head_sweep_20260310_151508.json` | `INSTRUCT_V02` | `20/20` | `630` | `165` | `691` | `L22H21 d=-7.2263` | `L0H5 d=-3.0035` |
| `results/full_head_sweep/full_head_sweep_20260311_120236.json` | `INSTRUCT_V02` | `100/100` | `865` | `190` | `899` | `L26H30 d=+3.5984` | `L0H3 d=+0.7598` |

Surviving story:
- a large fraction of heads separate on entropy
- headline head identity is not stable enough to treat one top head as canonical across family/sample-size changes

## Lane 5: Mistral SVD Drift

| File | Family | n_prompts | Strongest contractive head | Strongest expansive head |
| --- | --- | --- | --- | --- |
| `results/svd_circuits/svd_decomposition_20260304_122437.json` | `BASE_V01` | `20` | `L27H10 d=-1.5437` | `L5H29 d=+2.9281` |
| `results/svd_circuits/svd_decomposition_20260310_145312.json` | `INSTRUCT_V02` | `20` | `L27H10 d=-3.0197` | `L5H29 d=+1.8882` |
| `results/svd_circuits/svd_decomposition_20260311_120339.json` | `INSTRUCT_V02` | `100` | `L27H10 d=-1.3272` | `L5H29 d=+1.1208` |

Surviving story:
- the `L27H10` contractive / `L5H29` expansive motif is robust
- the exact effect sizes are not stable enough to move numbers across families casually

## Lane 6: Mistral Path-Patching Drift

| File | Family | n_prompts | Max absolute effect |
| --- | --- | --- | --- |
| `results/path_patching/path_patching_summary_20260227_080128.json` | `BASE_V01` | `20` | `L4 residual d=+1.9616` |
| `results/path_patching/path_patching_summary_20260310_151654.json` | `INSTRUCT_V02` | `20` | `L1 residual d=+1.9439` |
| `results/path_patching/path_patching_summary_20260311_121417.json` | `INSTRUCT_V02` | `100` | `L4 residual d=+3.9529` |

Important `INSTRUCT_V02 n=100` entries from `results/path_patching/path_patching_summary_20260311_121417.json`:
- `L4 residual break d=+3.9529`
- `L5 residual break d=+3.6330`
- `L3 residual break d=+3.5793`
- `L2 residual break d=+3.5564`
- `L5 v_proj break d=+2.4985`
- `L27 residual break d=-1.9553`
- `L27 v_proj break d=-1.9552`

Surviving story:
- strongest BREAK effects live in early residual layers
- the largest V-proj effect seen here is early `L5`, not late `L27`
- late `L27` remains interesting, but it is not the dominant path-patching headline

## Lane 7: Dual-Layer Drift Inside Instruct Family

Three instruct-family files already disagree in exact percentages:

| File | Recursive clean | Recursive patched | Baseline clean | Baseline patched | Notes |
| --- | --- | --- | --- | --- | --- |
| `results/persistent_patching_v3/persistent_patching_v3_dual_20260310_160920.json` | `40.7%` | `0.0%` | `3.3%` | `2.7%` | older generation/metric path |
| `results/persistent_patching_v3/persistent_patching_v3_dual_20260310_191654.json` | `50.0%` | `0.0%` | `0.0%` | `0.0%` | single-session smoke |
| `results/persistent_patching_v3/persistent_patching_v3_dual_20260310_204100.json` | `54.7%` | `0.0%` | `2.0%` | `0.0%` | current best instruct artifact; repetitive collapse logged |
| `results/persistent_patching_v3/persistent_patching_v3_dual_20260311_130612.json` | `52.0%` | `0.0%` | `3.7%` | `0.0%` | completed `n100` queue tail on pod 2; same asymmetry, different exact rates |

Surviving story:
- BREAK survives
- INDUCE fails
- patched generations degenerate
- exact percentages are pipeline-version-sensitive and cannot be mixed

## Lane 8: Steering / Gate Search

Base-family targeted scan:
- `results/phase1_mechanism/runs/20260311_055109_causal_state_targeted_scan_v1_mistral_targeted_scan_v1/summary.json`
  - `verdict = ready_for_confirmatory_v3`
  - `best_candidate = L25_W32_A3`
  - `source_layer = 25`
  - `window = 32`
  - `alpha = 3.0`

This is promising, but it is still base-family steering work, not a replacement for the instruct-family dual-layer artifact.

## Lane 9: What Is Stable Enough To Say Today

Stable enough:
- Mistral and Qwen contract across both available pipelines
- OPT and GPT-2 genuinely reverse sign across pipelines
- the strongest path-patching story is early residual BREAK, not late V-proj sufficiency
- the SVD motif is real: early expansive head(s), late contractive head(s)
- dual-patch BREAK works; naive INDUCE does not

Not stable enough:
- one canonical top attention head for Mistral
- one exact Mistral dual-layer percentage unless family/version is named
- `Gemma=-3.37` as a cross-model table cell
- any Mixtral cross-model headline without a raw summary artifact
- the current scaling-threshold sentence

## Lane 10: Required Paper Actions

1. Pick `BASE_V01` or explicitly split `BASE_V01` and `INSTRUCT_V02`.
2. Replace every contaminated Mistral line using `docs/status/CLAIM_REGISTRY.md`.
3. Remove or rewrite `Gemma=-3.37` until a raw matching summary is located.
4. Remove or rewrite the Mixtral table row until a raw matching summary is located.
5. Rewrite the scaling-threshold sentence so it does not cite `Qwen2.5-3B d=+1.60` as evidence for contraction.
6. Do not use a local base-family `p0_canonical n100` claim unless a real base artifact lands in `results/p0_canonical/`.
