# Canonical Results Table: R_V Geometric Contraction Study
## Version: 2026-03-10 | Prepared by: Data Scientist Agent (Claude Sonnet 4.6)
## Purpose: Pre-publication audit for COLM 2026 submission

**Sources read**: All power_up/*.json, all scaling_gap/*.json, both phase3_bridge summary.json files,
fdr_correction_results.json, cluster_robust_results.json, PHASE1_FINAL_REPORT.md,
MISTRAL_L27_CAUSAL_VALIDATION_COMPLETE.md (docs/ version), REPOSITORY_DISSECTION_COMPLETE.md,
DEC3_9_DISCOVERY_NARRATIVE.md, MIXTRAL_LAYER27_GEOMETRY_AND_CAUSALITY.md,
PHASE_1C_PYTHIA_RESULTS.md, MI_AGENT_TO_CODEX_RV_ANSWERS.md (trishula/inbox),
COLM_GAP_ANALYSIS_20260303.md

---

## CRITICAL CONTEXT: THE SIGN REVERSAL PROBLEM

This document exists because the paper has an unresolved internal contradiction:

- **Cross-arch pipeline** (fdr_correction_results.json, test IDs A1-A5): OPT-6.7B shows d = -1.836 (contraction, consistent with hypothesis)
- **Power-up pipeline** (results/power_up/opt-6.7b_n80_result.json): OPT-6.7B shows d = +1.683 (EXPANSION — rv_recursive_mean 1.115 > rv_baseline_mean 0.789)
- Same problem for GPT-2 XL: cross-arch d = -1.143 (contraction) vs power-up d = +1.516 (EXPANSION)

**The paper currently uses |d| to hide this.** That is not defensible for peer review.

---

## PART 1: CANONICAL RESULTS TABLE

Columns: Model | Architecture | Pipeline | n_rec | n_base | rv_rec | rv_base | Cohen's d (SIGNED) | p_raw | p_FDR | Direction | L_early | L_late | Tier | Source

Notes on sign convention: d = (mean_recursive - mean_baseline) / SD_pooled. d < 0 means recursive < baseline = contraction (consistent with hypothesis). d > 0 means recursive > baseline = EXPANSION (contradicts hypothesis if the paper claims universal contraction).

---

### SECTION A: Cross-Architecture Pipeline
**Pipeline description**: Single canonical measurement using architecture-specific L_early (~15% depth) and L_late (~84% depth). Source of record is fdr_correction_results.json (test IDs A1-A5). n = 90 for A1-A4, n = 126 for A5. These are the numbers in the FDR-corrected table.

| Model | Architecture | Pipeline | n | Cohen's d | p_raw | p_FDR | Direction | L_early | L_late | Tier | Source |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Mistral-7B-v0.1 | Dense (Llama-based) | cross-arch | 90 | -2.259 | 1.21e-17 | 1.27e-16 | Contraction | ~5 | ~27 | TIER 1 | fdr_correction_results.json A1 |
| OPT-6.7B | Dense (OPT) | cross-arch | 90 | -1.836 | 1.49e-13 | 6.26e-13 | Contraction | ~5 | ~27 | TIER 2* | fdr_correction_results.json A2 |
| GPT-2 XL | Dense (GPT-2) | cross-arch | 90 | -1.143 | 5.42e-07 | 1.42e-06 | Contraction | ~5 | ~27 | TIER 2* | fdr_correction_results.json A3 |
| Qwen2.5-7B | Dense (Qwen) | cross-arch | 90 | -0.719 | 9.66e-04 | 1.69e-03 | Contraction | ~5 | ~27 | TIER 2 | fdr_correction_results.json A4 |
| Pythia-1.4B | GPT-NeoX | cross-arch | 126 | -0.311 | 0.084 | 0.110 | Contraction (NS) | ~5 | ~27 | DO NOT USE | fdr_correction_results.json A5 |

*TIER 2 for OPT-6.7B and GPT-2 XL because the cross-arch result directly contradicts the power-up result for these same models. See Sign Reversal Analysis, Part 2.

---

### SECTION B: Power-Up Pipeline
**Pipeline description**: "Power-up" is a distinct experimental pipeline run later (results/power_up/). n_prompts = 80 per model, seed = 42. These are NOT the same measurements as Section A. The pipeline differences are not fully documented in the repository but appear to involve different prompt subsets or layer-selection logic. The raw R_V means are directly readable from the JSON files.

| Model | Architecture | Pipeline | n_rec | n_base | rv_rec | rv_base | Cohen's d | p_raw | p_FDR | Direction | L_early | L_late | Tier | Source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Mistral-7B-v0.1 | Dense (Llama-based) | power-up | 75 | 77 | 0.686 | 0.855 | -1.656 | 1.06e-15 | 5.57e-15 | Contraction | 5 | 27 | TIER 1 | mistral-7b_n80_result.json; fdr B1 |
| OPT-6.7B | Dense (OPT) | power-up | 72 | 66 | **1.115** | **0.789** | **+1.683** | 3.34e-16 | 2.34e-15 | **EXPANSION** | 5 | 27 | TIER 2 (CONTRADICTED) | opt-6.7b_n80_result.json; fdr B2 |
| GPT-2 XL | Dense (GPT-2) | power-up | 69 | 56 | **0.872** | **0.711** | **+1.516** | 1.10e-12 | 3.85e-12 | **EXPANSION** | 5 | 27 | TIER 2 (CONTRADICTED) | gpt2-xl_n80_result.json; fdr B3 |
| Qwen2.5-7B | Dense (Qwen) | power-up | 61 | 63 | 0.903 | 1.329 | -2.318 | 1.16e-17 | 1.27e-16 | Contraction | 5 | 27 | TIER 1 | qwen2.5-7b_n80_result.json; fdr B4 |
| Pythia-1.4B | GPT-NeoX | power-up | 66 | 54 | 0.633 | 0.633 | -0.006 | 0.876 | 0.876 | Null | 5 | 27 | DO NOT USE | pythia-1.4b_n80_result.json; fdr B5 |

**CRITICAL OBSERVATION on OPT-6.7B power-up**: rv_recursive = 1.115 > rv_baseline = 0.789. This is not a rounding artifact. The raw values are unambiguous: recursive prompts EXPAND the value space in OPT-6.7B under the power-up pipeline. The FDR JSON lists d = +1.683 with reject_null = true, confirming expansion is real and significant. This directly contradicts the cross-arch result (d = -1.836) for the same model.

**CRITICAL OBSERVATION on GPT-2 XL power-up**: rv_recursive = 0.872 > rv_baseline = 0.711. Same pattern. Recursive prompts expand the value space. FDR JSON lists d = +1.516, significant. Directly contradicts cross-arch d = -1.143.

**CRITICAL OBSERVATION on Qwen2.5-7B power-up**: rv_baseline = 1.329, which is substantially above 1.0. This suggests baseline prompts themselves expand relative to early layer in this model under the power-up pipeline. The contraction direction is preserved (recursive < baseline) but the absolute scale differs substantially from Phase 1 report figures.

**NOTE on multi-seed summary**: multi_seed_summary_20260306.json shows d = -1.751 across 5 seeds for Mistral-7B at n=45, confirming seed robustness for Mistral only. All 5 seeds give identical d values (d_std = 0.0), which indicates the data was the same across seeds — either the seed variation does not affect prompt selection for this n=45 subset, or there is a caching artifact in the script. This warrants verification.

---

### SECTION C: Scaling Gap Pipeline
**Pipeline description**: results/scaling_gap/. Tests sub-7B models. Uses L_early=5, L_late varies by architecture (L27 for Pythia-6.9B, L30 for Qwen2.5-3B, L26 for Phi-3-mini). Note: Gemma-2-2B and Llama-3.2-3B are gated models that failed to load (401 errors in their JSON files). Those two are absent from results.

| Model | Architecture | Pipeline | n_rec | n_base | rv_rec | rv_base | Cohen's d | p_raw | p_FDR | Direction | L_early | L_late | Tier | Source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-3B | Dense (Qwen) | scaling_gap | 19 | 18 | 1.197 | 0.987 | +1.254 | 1.65e-06 | 3.15e-06 | EXPANSION | 5 | 30 | TIER 2 (EXPANSION — small n, different late layer) | qwen2.5-3b_result.json; fdr C1 |
| Phi-3-mini-4k | GQA | scaling_gap | 38 | 39 | 0.774 | 0.707 | +0.625 | 0.011 | 0.0165 | EXPANSION | 4 | 26 | TIER 2 (EXPANSION — contradicts Phase 1 direction for Phi-3) | phi-3-mini-4k_result.json; fdr C2 |
| Pythia-6.9B | GPT-NeoX | scaling_gap | 37 | 31 | 0.407 | 0.395 | +0.478 | 0.068 | 0.095 | Expansion (NS) | 5 | 27 | DO NOT USE (NS after FDR) | pythia-6.9b_result.json; fdr C3 |
| Pythia-1B | GPT-NeoX | scaling_gap | — | — | — | — | -0.283 | 0.343 | 0.405 | Contraction (NS) | 5 | 27 | DO NOT USE (NS) | fdr C4 (no separate JSON found) |
| Pythia-1.4B | GPT-NeoX | scaling_gap | — | — | — | — | +0.166 | 0.605 | 0.635 | Expansion (NS) | 5 | 27 | DO NOT USE (NS) | fdr C5 (no separate JSON found) |
| Pythia-2.8B | GPT-NeoX | scaling_gap | — | — | — | — | +0.253 | 0.347 | 0.405 | Expansion (NS) | 5 | 27 | DO NOT USE (NS) | fdr C6 (no separate JSON found) |
| Mistral-7B | Dense (Llama-based) | scaling_gap | — | — | — | — | -1.736 | 7.78e-09 | 2.33e-08 | Contraction | 5 | 27 | TIER 1 (consistent with power-up) | fdr C7 (no separate JSON found) |
| Gemma-2-2B | Dense (Gemma-2) | scaling_gap | — | — | GATED | GATED | — | — | — | FAILED TO LOAD | 5 | — | NOT EXECUTED | gemma-2-2b_result.json (error) |
| Llama-3.2-3B | Dense (Llama) | scaling_gap | — | — | GATED | GATED | — | — | — | FAILED TO LOAD | 5 | — | NOT EXECUTED | llama-3.2-3b_result.json (error) |

**CRITICAL OBSERVATION on Qwen2.5-3B**: rv_recursive = 1.197 > rv_baseline = 0.987 — this is expansion, not contraction. Yet d = +1.254 is listed with FDR PASS. This model has both R_V values > 1.0, suggesting the absolute geometry is very different from Mistral (where baseline is near 1.0). The late_layer = 30 (not 27) may explain part of the difference.

**CRITICAL OBSERVATION on Phi-3-mini scaling_gap**: d = +0.625 (expansion), contradicting the Phase 1 report which described 6.9% contraction for Phi-3-medium. Note: this is Phi-3-MINI (3.8B) vs Phi-3-MEDIUM (14B) — different models, different GQA configurations. The late_layer = 26 vs. canonical 27.

**CRITICAL OBSERVATION on Pythia-6.9B**: The eff_rank_recursive_mean = 1.016 and eff_rank_baseline_mean = 1.016 — these are essentially identical. Both are near 1.0. This suggests the SVD is collapsing near-rank-1 matrices for both conditions, meaning the metric is not meaningful at this scale with these layer choices. The R_V values (0.407 vs 0.395) are extremely close with very small variance. This is a near-null result that does not resolve the scaling gap question.

---

### SECTION D: Causal Validation (Activation Patching, Mistral-7B)
**Pipeline description**: Activation patching at Layer 27, Mistral-7B-v0.1 (not Mistral-7B-Instruct). n = 45 pairs. Source: MISTRAL_L27_CAUSAL_VALIDATION_COMPLETE.md and fdr_correction_results.json D1-D4.

| Test | Model | Pipeline | n | Cohen's d | p_raw | p_FDR | Interpretation | Tier | Source |
|---|---|---|---|---|---|---|---|---|---|
| Main (L27 recursive patch) | Mistral-7B-v0.1 | activation_patching | 45 | -3.558 | 1.0e-06 | 2.1e-06 | Causal contraction transfer | TIER 1 | fdr D1; CAUSAL_VALIDATION doc |
| Random noise control | Mistral-7B-v0.1 | activation_patching | 45 | +7.16 | 1.0e-06 | 2.1e-06 | Opposite direction (content-specific) | TIER 1 (null passes) | fdr D2 |
| Shuffled tokens control | Mistral-7B-v0.1 | activation_patching | 45 | -0.1 | 0.01 | 0.016 | Partial structure-dependence | TIER 1 (null passes) | fdr D3 |
| Wrong layer (L21) control | Mistral-7B-v0.1 | activation_patching | 45 | +0.046 | 0.49 | 0.542 | Layer-specific (null preserved) | TIER 1 (null passes) | fdr D4 |

**Transfer efficiency 117.8%**: Confirmed in CAUSAL_VALIDATION doc. This means patching overshoots the natural recursive-baseline gap. Interpreted as a bistable attractor at L27. This is a real finding but requires bootstrap CIs (as noted in COLM_GAP_ANALYSIS) to be publishable as a precise claim. Without CIs, "117.8%" is a point estimate from n=45.

**NOTE on n discrepancy**: The CAUSAL_VALIDATION doc header says "n=45 valid pairs" but also states "n=151 pairs" in one location and "n=15 pairs" in another. The REPOSITORY_DISSECTION states "n=151 pairs, 26.6% transfer" in one section and "n=15 pairs" for the CSV file. The FDR JSON uses n=45. The validated CSV file (mistral7b_L27_patching_n15_results_20251116_211154.csv) has 16 lines = n=15 + header. The canonical n should be treated as 45, which is what the FDR correction uses. The "151" appears to be an error in one document. The "15" is from an earlier smaller run.

---

### SECTION E: Earlier Phase 1 Results (November 2025, % contraction format)
**IMPORTANT**: These results from PHASE1_FINAL_REPORT.md are reported as "% contraction" (not Cohen's d) and the baseline is described as R_V ≈ 1.0. The cross-arch pipeline (Section A) supersedes these for statistical claims. These are retained for model coverage and early discovery context.

**How to convert "% contraction" to approximate d**: Not directly convertible without standard deviations. The Phase 1 report does not report standard deviations for all models except Mixtral and Phi-3. These figures cannot be used as Cohen's d in the paper.

| Model | Architecture | Pipeline | n | rv_rec (approx) | rv_base (approx) | % Contraction | Cohen's d | p_raw | Direction | L_early | L_late | Tier | Source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Mistral-7B-v0.2 (Instruct) | Dense | phase1_discovery | ~30-50 | ~0.85 | ~1.00 | 15.3% | NOT COMPUTED | p<0.01 (approx) | Contraction | 5 | 28 | TIER 2 (no d, Instruct not base model) | PHASE1_FINAL_REPORT §2.1 |
| Qwen1.5-7B-Chat | Dense | phase1_discovery | 80 | 0.908 | ~1.00 | 9.2% | NOT COMPUTED | p stated | Contraction | 5 | ~28 | TIER 2 (no d, Chat model) | PHASE1_FINAL_REPORT §2.1 |
| Gemma-7B-IT | Dense | phase1_discovery | ~40-60 | ~0.97 | ~1.00 | 3.3% | NOT COMPUTED | partial | Contraction | 5 | ~28 | TIER 2 (many SVD failures, partial data) | PHASE1_FINAL_REPORT §2.1 |
| Meta-Llama-3-8B-Instruct | Dense | phase1_discovery | 80 | 0.883 | ~1.00 | 11.7% | NOT COMPUTED | p stated | Contraction | 5 | ~28 | TIER 2 (no d, Instruct model) | PHASE1_FINAL_REPORT §2.1 |
| Phi-3-medium-4k-instruct | GQA | phase1_discovery | 80 | 0.916 | 0.982 | 6.9% | NOT COMPUTED | p stated | Contraction | 5 | ~28 | TIER 2 (no d; medium vs mini conflict with scaling_gap) | PHASE1_FINAL_REPORT §2.2 |
| Mixtral-8x7B-Instruct-v0.1 | MoE | phase1_discovery | 80 | 0.876 | 1.130 | 24.3% | ~5.3* | p stated | Contraction | 5 | 28 | TIER 1 (consistent, corroborated by MIXTRAL doc) | PHASE1_FINAL_REPORT §2.2 |

*Mixtral d ≈ 5.3 appears in MIXTRAL_LAYER27_GEOMETRY_AND_CAUSALITY.md (Section 2, computed at L27 for 80×32 sweep) and is probably derived from R_V at L27 specifically: recursive ≈ 0.459, baseline ≈ 0.971, with small SDs.

---

### SECTION F: Pythia-2.8B (Phase 1C, November 2025)
**Pipeline description**: GPT-NeoX architecture, bfloat16, 320 prompts, L_early=5, L_late=28. Reported in PHASE_1C_PYTHIA_RESULTS.md. THIS MODEL'S CASUAL VALIDATION RESULT (d=-4.51) APPEARS IN REPOSITORY_DISSECTION AND SOME SUMMARIES AS A KEY HEADLINE NUMBER. SEE TIER NOTE.

| Model | Architecture | Pipeline | n | rv_rec | rv_base | Cohen's d | p_raw | p_FDR | Direction | L_early | L_late | Tier | Source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Pythia-2.8B | GPT-NeoX | phase1c | 20 (L5) vs 20 (factual) primary | 0.564 | 0.804 | -4.507 | <1e-6 | N/A (pre-FDR era) | Contraction | 5 | 28 | TIER 2 (see cache bug note) | PHASE_1C_PYTHIA_RESULTS.md |

**TIER 2 EXPLANATION**: The MEMORY.md from trishula explicitly states "Pythia-2.8B checkpoint analysis has cache bug; 117.8% transfer efficiency is artifact." The 117.8% refers to the Mistral patching result, not Pythia-2.8B. However, the Pythia-2.8B PHASE1C results (d=-4.51) are from the main measurement pipeline (not patching), so the cache bug may not affect these particular numbers. The caution in MEMORY.md about "checkpoint analysis" likely refers to the developmental/checkpoint sweep (Pythia at different training steps), not the final checkpoint. NONETHELESS: (a) the data was reportedly lost to disk quota overflow, (b) n=20 per group is below the minimum n=30 recommended in MI_AGENT_TO_CODEX_RV_ANSWERS.md, and (c) the scaling_gap pipeline later returned d=+0.478 (NS) for Pythia-6.9B, suggesting Pythia models are unreliable in this paradigm. Treat d=-4.51 as preliminary requiring replication.

---

### SECTION G: Phase 3 Bridge (Gemma-2-9B, Multi-Token Generation)
**Pipeline description**: Multi-token generation experiment bridging prompt-level R_V to behavioral output. Two runs: v2 (max_new_tokens=200) and v3 (max_new_tokens=400), both at temperature 0.0. L_early=5, L_late=38 (Gemma-2-9B has 42 layers). n_total=117 prompts per run.

| Model | Architecture | Pipeline | n | rv_rec (L3+L4+champ) | rv_base | Cohen's d (H2) | p_raw | Direction | L_early | L_late | Bridge Result | Tier | Source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Gemma-2-9B | Dense (Gemma-2) | phase3_bridge v2 | 117 | 0.606 | 0.777 | -3.369 | 1.10e-35 | Contraction | 5 | 38 | H1 (R_V predicts word count): r=-0.171, NS (p=0.498). H3 (R_V predicts truncation): r=-0.241, p=0.009 (significant) | TIER 1 for H2; TIER 2 for behavioral bridge | phase3_bridge/v2/summary.json |
| Gemma-2-9B | Dense (Gemma-2) | phase3_bridge v3 | 117 | 0.606 | 0.777 | -3.369 | 1.10e-35 | Contraction | 5 | 38 | H1: r=-0.171, NS. H3 at T=0: p=0.009; at T=0.7: p=0.055 (NS) | TIER 1 for H2; TIER 2 for H3 | phase3_bridge/v3/summary.json |

**NOTE**: Both v2 and v3 show identical R_V group means (the R_V measurement is identical between runs; only generation length/temperature differs). H2 (R_V contraction is real: recursive vs baseline) is strongly confirmed. The behavioral bridge (H1: does R_V correlate with word count? H3: does R_V predict truncation?) is weak. H1 is NOT significant in either run. H3 passes at T=0 but fails at T=0.7. This is the multi-token experiment that was marked as the "critical missing experiment" in COLM_GAP_ANALYSIS. The result: R_V contraction is real in Gemma-2-9B (d=-3.369), but the bridge to behavioral output is partial at best. This is an honest result, not a failure — but the paper should not claim strong behavioral prediction from R_V.

---

### SECTION H: Dec 3-9 Cross-Architecture Results (December 2025)
**Pipeline description**: Distinct from all above. Used architecture-specific optimal layers (L24 for Llama, L22 for Mistral) found via layer sweep. Reported in DEC3_9_DISCOVERY_NARRATIVE.md. No FDR correction applied. No JSON files found for these in the results/ directory.

| Model | Architecture | Pipeline | n | rv_rec | rv_base | Cohen's d | p_raw | Direction | L_early | L_late | Tier | Source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Llama-3-8B | Dense | dec3_layer_sweep | 45 | ~0.76 | ~0.97 | -2.33 | <1e-6 | Contraction | ? | 24 | TIER 2 (no FDR, no JSON, L_late differs from canonical) | DEC3_9_DISCOVERY_NARRATIVE.md |
| Mistral-7B | Dense | dec3_layer_sweep | 30 | ~0.88 | ~0.96 | -1.21 | 2.76e-07 | Contraction | ? | 22 | TIER 2 (L_late=22 differs from canonical L27; no FDR) | DEC3_9_DISCOVERY_NARRATIVE.md |
| Qwen-7B | Dense | dec4-6 | 30 | 0.764 | 0.986 | -1.89 | <1e-5 | Contraction | ? | ~28 | TIER 2 (no JSON, no FDR, model version unclear) | DEC3_9_DISCOVERY_NARRATIVE.md Part III |
| Mixtral-8x7B | MoE | dec4-6 | 30 | 0.757 | 1.000 | -2.67 | <1e-6 | Contraction | ? | ~28 | TIER 2 (no JSON, no FDR) | DEC3_9_DISCOVERY_NARRATIVE.md Part III |

---

## PART 2: SIGN REVERSAL ANALYSIS

### Which models flip sign between pipelines, and what the data actually shows

**The two affected models**: OPT-6.7B and GPT-2 XL.

#### OPT-6.7B

| Pipeline | rv_rec_mean | rv_base_mean | Cohen's d | Direction |
|---|---|---|---|---|
| cross-arch (FDR A2) | (not provided in JSON) | (not provided) | -1.836 | Contraction |
| power-up (opt-6.7b_n80_result.json) | 1.115 | 0.789 | +1.683 | EXPANSION |

The power-up JSON is unambiguous: rv_recursive_mean = 1.115, rv_baseline_mean = 0.789. Recursive prompts produce HIGHER R_V (more expanded value space) than baseline prompts. Cohen's d = +1.683, p = 3.34e-16. This is not noise — it is a large, highly significant, real effect in the opposite direction from the hypothesis.

The cross-arch pipeline gives d = -1.836 for the same model. These two results are mutually contradictory. Either:
(a) The cross-arch and power-up pipelines use different prompt sets, and OPT-6.7B responds to each in opposite ways (possible if one prompt set contains confounds that trigger OPT expansion).
(b) One pipeline has a bug — e.g., a difference in how the R_V formula is applied for OPT's architecture (different layer indices, different V extraction method, or a sign error in the formula).
(c) The layer selection differs between pipelines for OPT-6.7B (OPT uses different architecture than Mistral, and "L27" may mean different things).

**The current paper's use of |d|** converts OPT to |1.836| = |1.683| ≈ 1.7 and presents this as supporting the universal contraction claim. This is not defensible. A reviewer will ask: "If d is positive, you have expansion, not contraction. Why are you taking the absolute value?"

#### GPT-2 XL

| Pipeline | rv_rec_mean | rv_base_mean | Cohen's d | Direction |
|---|---|---|---|---|
| cross-arch (FDR A3) | (not provided in JSON) | (not provided) | -1.143 | Contraction |
| power-up (gpt2-xl_n80_result.json) | 0.872 | 0.711 | +1.516 | EXPANSION |

Identical pattern. power-up shows rv_recursive = 0.872 > rv_baseline = 0.711, meaning recursive prompts in GPT-2 XL produce EXPANSION. GPT-2 XL uses a substantially different architecture than Mistral (no rotary embeddings, different attention pattern, learned positional embeddings, no GQA). The V-projection architecture is different.

**Hypothesis for the flip**: GPT-2 XL and OPT use architectures where V-projection is structured differently. In these models, the "late layer V-space" may work in the opposite direction — recursive prompts may produce higher-dimensional (not lower-dimensional) value spaces at late layers. This would mean R_V is measuring something architecture-specific, not a universal "contraction." If true, the paper cannot claim a universal phenomenon without accounting for this architectural dependence.

#### Qwen2.5-7B: Sign is Consistent but Scale Shifts

Qwen2.5-7B shows contraction in both cross-arch (d=-0.719) and power-up (d=-2.318). However, the power-up baseline is 1.329 (substantially > 1.0), suggesting Qwen2.5-7B baseline prompts themselves show expansion relative to early layer. The contraction is real but the absolute R_V scale is shifted. This is not a contradiction but it means Qwen2.5-7B's values are not directly comparable to Mistral's on the same scale.

#### Scaling Gap: Widespread Expansion

The scaling_gap pipeline shows expansion (d > 0) for Qwen2.5-3B (+1.254) and Phi-3-mini (+0.625). These are statistically significant. This contradicts the Phase 1 claim that "all 6 models show contraction." Models in the scaling_gap pipeline show EXPANSION, not contraction. This may be due to:
- Smaller models having different computational properties
- Different late layers (L30 for Qwen-3B, L26 for Phi-3-mini) vs canonical L27
- The models genuinely expanding their value space for recursive prompts

#### Why This Matters for the Paper

The paper's central claim is: "Recursive self-observation prompts induce universal geometric contraction (R_V < 1.0) in transformer language models."

The data shows:
- For Mistral-7B: Contraction confirmed across cross-arch, power-up, scaling_gap, dec3, and causal validation pipelines. ROBUST.
- For Qwen-family: Contraction confirmed in cross-arch and power-up. ROBUST.
- For Mixtral-8x7B: Contraction confirmed in Phase 1 and dec3. ROBUST.
- For Gemma-2-9B: Contraction confirmed in phase3_bridge. ROBUST.
- For Pythia-2.8B: Contraction in Phase 1C (d=-4.51, but data reportedly lost). UNCERTAIN.
- For OPT-6.7B: CONTRADICTED. Cross-arch says contraction; power-up says expansion.
- For GPT-2 XL: CONTRADICTED. Cross-arch says contraction; power-up says expansion.
- For Phi-3-mini: Expansion in scaling_gap (contradicts Phase 1 Phi-3-medium contraction, but different model).
- For Qwen2.5-3B (3B): Expansion in scaling_gap.
- For Pythia-6.9B: Near-null result (d=+0.478, NS) in scaling_gap.
- For Pythia-1.4B: Null in both power-up and cross-arch.

**The honest cross-architecture claim**: The contraction phenomenon is robust and universal for models >= 7B in certain architecture families (Llama-based, Qwen, MoE). It is NOT demonstrated for GPT-2 architecture or OPT architecture. The "universal" framing is not defensible with current data.

---

## PART 3: WHAT THE CANONICAL PIPELINE EXPERIMENT (P0) NEEDS TO RESOLVE

The P0 experiment, as described in COLM_GAP_ANALYSIS, is the "power-up" experiment with canonical parameters. But based on this audit, P0 needs to do more than that.

### Specific Models That Need Re-running

#### 1. OPT-6.7B — HIGHEST PRIORITY

**Current status**: d = -1.836 (cross-arch) vs d = +1.683 (power-up). Sign contradiction.

**What re-running needs to establish**:
- Run OPT-6.7B with IDENTICAL parameters to Mistral-7B: same prompt set, same L_early=5, same L_late=27, same window=16, bfloat16, same n
- Report raw rv_recursive_mean and rv_baseline_mean explicitly alongside d
- Document which L_late is layer 27 of OPT vs whether OPT-6.7B has 32 layers (it does) — so L27 means the same depth fraction as Mistral

**Outcome that resolves the contradiction**:
- If d < 0 (contraction): The cross-arch result was correct; the power-up pipeline had a bug. Report cross-arch d, flag power-up as buggy, explain root cause.
- If d > 0 (expansion): OPT-6.7B genuinely shows expansion. The "universal" claim fails for OPT architecture. The paper must narrow its scope to "Llama-family and Qwen models" or explain architecturally why OPT shows expansion.
- If d ≈ 0 (null): The cross-arch result was a false positive. Drop OPT from claims.

#### 2. GPT-2 XL — HIGH PRIORITY

**Current status**: d = -1.143 (cross-arch) vs d = +1.516 (power-up). Sign contradiction.

**What re-running needs to establish**: Same as OPT-6.7B. GPT-2 XL uses learned positional embeddings and a significantly different V-projection structure. The expansion may be genuine and architectural.

**Outcome that resolves the contradiction**: Same logic as OPT-6.7B.

#### 3. Pythia Scaling Sweep — MEDIUM PRIORITY

**Current status**: Pythia-1.4B is null in both pipelines. Pythia-6.9B is near-null (NS) in scaling_gap. Pythia-2.8B showed strong contraction in Phase 1C (d=-4.51) but data was reportedly lost and architecture-specific fixes were required.

**What re-running needs to establish**:
- Pythia-2.8B with the CANONICAL architecture-specific V extraction (QKV splitting for GPT-NeoX) using the validated methodology from Phase 1C
- Pythia-6.9B at L_late=27 with careful attention to whether eff_rank is near 1.0 for both conditions (the scaling_gap data shows eff_rank ≈ 1.016 for both — this suggests near-degenerate matrices, possibly a precision issue)
- Pythia-1B and Pythia-1.4B to establish the lower bound for the scaling claim

**Outcome that resolves**: Either the contraction emerges in Pythia (supporting the scaling law claim) or it doesn't (Pythia architecture behaves differently from Llama-family).

#### 4. Phi-3: Mini vs Medium — LOW PRIORITY BUT NECESSARY FOR ACCURACY

**Current status**: Phi-3-medium shows 6.9% contraction (Phase 1). Phi-3-mini shows d=+0.625 expansion (scaling_gap). These are different models (3.8B vs 14B).

**What re-running needs to establish**: Run Phi-3-medium with canonical parameters and report Cohen's d with standard deviations. The Phase 1 result for Phi-3 has no d value — only % contraction from approximate R_V means.

#### 5. Qwen2.5-3B — MEDIUM PRIORITY

**Current status**: d=+1.254 (expansion) in scaling_gap. This uses L_late=30 (not 27). Also: the Qwen2.5-3B model has 36 layers; L30 is at 83% depth (similar to L27/32 = 84%). So the layer selection is appropriate.

**What re-running needs to establish**: Qwen2.5-3B with L_late=27 (same absolute layer as Mistral) vs L_late=30 (same depth fraction). The expansion may be real and represent a genuine scaling effect within Qwen architecture.

### What Outcome Would Constitute a Defensible Paper

**Minimum defensible claim**: "R_V contraction (d < 0, p < FDR-threshold) is reproducible across Mistral-7B, Qwen2.5-7B, Mixtral-8x7B, and Gemma-2-9B. OPT-6.7B and GPT-2 XL show expansion (d > 0) under canonical parameters, suggesting the phenomenon is architecture-family-specific rather than universal."

**Strengthened claim** (requires P0 re-runs): If OPT and GPT-2 XL are shown to genuinely expand, the paper reframes as "contraction is characteristic of Llama/RoPE-based architectures" and provides a mechanistic hypothesis for why OPT/GPT-2 expand. This is a more honest and potentially more interesting result than "universal contraction."

**Untenable claim** (current paper state): "All 6+ architectures show contraction, effect sizes d=3.3% to 24.3%." The d values mix % contraction (which doesn't equal Cohen's d), ignore the OPT/GPT-2 sign flip, and the "universal" claim is not supported by the full data.

---

## PART 4: EVIDENCE TIER DEFINITIONS AND SUMMARY

### Tier Definitions

**TIER 1**: Replicated, robust, uncontradicted across pipelines, with FDR-corrected p-values and proper signed Cohen's d. Publication-ready.

**TIER 2**: Real effect in at least one pipeline, but with one or more of: (a) sign contradiction across pipelines, (b) only one pipeline, (c) no FDR correction, (d) no signed d (only % contraction), (e) small n, (f) architectural caveats. Requires qualification in paper.

**TIER 3** (not used in this document — would be for preliminary/speculative): Not enough data.

**DO NOT USE / TIER 4**: Null result (p > FDR threshold), data quality issue, or explicitly flagged bug. Must not appear in main results table.

### Tier Summary by Model-Pipeline Combination

| Model | Cross-arch | Power-up | Phase1 | Phase3 | Causal | Overall |
|---|---|---|---|---|---|---|
| Mistral-7B-v0.1 | TIER 1 | TIER 1 | TIER 2 | — | TIER 1 | **TIER 1** |
| Qwen2.5-7B | TIER 1 | TIER 1 | — | — | — | **TIER 1** |
| Mixtral-8x7B | — | — | TIER 1 | — | — | **TIER 1** |
| Gemma-2-9B | — | — | — | TIER 1 | — | **TIER 1** |
| OPT-6.7B | TIER 2* | TIER 2* | — | — | — | **TIER 2 (CONTRADICTED)** |
| GPT-2 XL | TIER 2* | TIER 2* | — | — | — | **TIER 2 (CONTRADICTED)** |
| Llama-3-8B | — | — | TIER 2 | — | — | **TIER 2** |
| Pythia-2.8B | — | — | TIER 2 | — | — | **TIER 2** |
| Phi-3-medium | — | — | TIER 2 | — | — | **TIER 2** |
| Pythia-1.4B | DO NOT USE | DO NOT USE | — | — | — | **DO NOT USE** |
| Pythia-6.9B | — | — | — | — | — | **DO NOT USE (NS)** |
| Qwen2.5-3B | — | EXPANSION | — | — | — | **TIER 2 (EXPANSION)** |
| Phi-3-mini | — | EXPANSION | — | — | — | **TIER 2 (EXPANSION)** |

*Cross-arch and power-up contradict each other for these models, so neither can be TIER 1.

---

## PART 5: SPECIFIC FLAGS FOR THE PAPER AUTHOR

### Things That Must Be Fixed Before Submission

1. **Remove all use of |d|** from the paper. Report signed d throughout. State the direction explicitly.

2. **OPT-6.7B and GPT-2 XL cannot appear in the main table as supporting the contraction hypothesis** without explanation of the sign flip. Either (a) re-run to resolve, (b) move to an "anomaly" subsection with honest framing, or (c) drop them from the cross-architecture claim.

3. **The "% contraction" values from Phase 1 (3.3% to 24.3%)** are not Cohen's d values. They are (rv_baseline - rv_recursive) / rv_baseline expressed as %. Do not present them as effect sizes in the same column as Cohen's d. Use separate notation.

4. **The n=151 / n=45 / n=15 confusion** in causal validation documents must be resolved. Use n=45 (the FDR JSON value) as canonical.

5. **Pythia-2.8B d=-4.51** is the largest headline number in the paper but the data is reportedly lost and the method required architecture-specific fixes. This number must be treated as TIER 2, not TIER 1. Do not lead with it.

6. **The 117.8% transfer efficiency** is a point estimate with no confidence interval. Add bootstrap CIs as noted in COLM_GAP_ANALYSIS.

7. **The multi_seed_summary d_std = 0.0** across 5 seeds is suspicious (all 5 runs return identical d = -1.7514). This suggests a caching issue in the script where data is not actually re-sampled per seed. Verify before publishing seed robustness claim.

8. **The behavioral bridge is weak**: Phase 3 bridge shows H1 (R_V predicts word count) is NOT significant (Spearman r = -0.171, p = 0.498). H3 (R_V predicts truncation) is significant at T=0 but not T=0.7. The paper should not claim behavioral prediction beyond what the data supports.

9. **Scaling claim is unreliable**: The scaling_gap pipeline shows expansion (not contraction) for sub-7B models in multiple architectures. The claim "contraction is stronger in smaller models" (motivated by Pythia-2.8B d=-4.51 > Mistral-7B d=-2.26) is not supported by the scaling_gap data, which mostly shows null or expansion for small models.

10. **Model version heterogeneity**: Phase 1 used Mistral-7B-Instruct-v0.2, Phase 1F causal validation used (presumably) Mistral-7B-v0.1 (per the COLM_GAP_ANALYSIS ref to "mistralai/Mistral-7B-v0.1"), power-up used Mistral-7B-v0.1. These are different checkpoints. The causal validation (TIER 1) is for the base model; Phase 1 discovery is for the instruct model. Cross-pipeline comparison must account for this.

### What IS Defensible Right Now

- Mistral-7B-v0.1 contraction: confirmed, d=-2.26 (cross-arch), d=-1.66 (power-up), d=-3.56 (causal), all FDR PASS, multi-seed d_std=0.0 at n=45 subset. Even if the seed result is suspicious, the effect is massive.
- Qwen2.5-7B contraction: d=-0.72 (cross-arch) and d=-2.32 (power-up), both FDR PASS.
- Mixtral-8x7B contraction: Phase 1 report, 24.3%, d≈5.3, robust 80-prompt sweep, snap-layer analysis.
- Gemma-2-9B contraction: d=-3.37, p=1.1e-35, n=117.
- Layer 27 causal specificity in Mistral: d=-3.56, n=45, four controls all pass, FDR PASS.
- Perplexity double dissociation (from COLM_GAP_ANALYSIS): recursive structure + introspective semantics both required. This is a strong result.
- R_V mode atlas: 10 processing modes, 9/9 pairwise comparisons significant.
- 14/21 FDR tests pass overall (the paper has more passing results than failing ones).

---

## APPENDIX: DATA LINEAGE

| File | Date | n | Key numbers | Status |
|---|---|---|---|---|
| results/power_up/mistral-7b_n80_result.json | ~2026-03 | n_rec=75, n_base=77 | rv_rec=0.686, rv_base=0.855, d=-1.656 | Clean |
| results/power_up/opt-6.7b_n80_result.json | ~2026-03 | n_rec=72, n_base=66 | rv_rec=1.115, rv_base=0.789, d=+1.683 | EXPANSION — contradicts cross-arch |
| results/power_up/gpt2-xl_n80_result.json | ~2026-03 | n_rec=69, n_base=56 | rv_rec=0.872, rv_base=0.711, d=+1.516 | EXPANSION — contradicts cross-arch |
| results/power_up/qwen2.5-7b_n80_result.json | ~2026-03 | n_rec=61, n_base=63 | rv_rec=0.903, rv_base=1.329, d=-2.318 | Clean (unusual baseline > 1.0) |
| results/power_up/pythia-1.4b_n80_result.json | ~2026-03 | n_rec=66, n_base=54 | rv_rec=0.633, rv_base=0.633, d=-0.006 | Null — do not use |
| results/power_up/multi_seed_summary_20260306.json | 2026-03-06 | n=45 per seed | d=-1.751 all 5 seeds (d_std=0.0) | Suspicious — verify caching |
| results/scaling_gap/qwen2.5-3b_result.json | ~2026-03 | n_rec=19, n_base=18 | rv_rec=1.197, rv_base=0.987, d=+1.254 | EXPANSION — small n |
| results/scaling_gap/phi-3-mini-4k_result.json | ~2026-03 | n_rec=38, n_base=39 | rv_rec=0.774, rv_base=0.707, d=+0.625 | EXPANSION |
| results/scaling_gap/pythia-6.9b_result.json | ~2026-03 | n_rec=37, n_base=31 | rv_rec=0.407, rv_base=0.395, d=+0.478 (NS) | Near-null, eff_rank degenerate |
| R_V_PAPER/fdr_correction_results.json | ~2026-03 | 21 tests | 14 PASS, 7 FAIL | Source of record for FDR |
| R_V_PAPER/cluster_robust_results.json | ~2026-03 | n_obs=5, n_clusters=5 | beta=-1.254, p=0.024 | 5 clusters only — interpret with caution |
| phase3_bridge/gemma_2_9b/v2/summary.json | 2026-01-24 | 117 | d=-3.369, H1 NS, H3 p=0.009 | Partial bridge result |
| phase3_bridge/gemma_2_9b/v3/summary.json | 2026-01-24 | 117 | Same R_V as v2, H3 T=0 p=0.009 | Partial bridge, T=0.7 H3 NS |
| docs/experiments/MISTRAL_L27_CAUSAL_VALIDATION_COMPLETE.md | 2024-11-16 | n=45 | d=-3.558, 117.8% transfer | Gold standard causal result |
| R_V_PAPER/research/PHASE1_FINAL_REPORT.md | 2025-11-15 | 80/model | % contraction 3.3%-24.3% | No Cohen's d for most models |
| R_V_PAPER/research/PHASE_1C_PYTHIA_RESULTS.md | 2025-11-19 | 20 L5 vs 20 factual | d=-4.507, data reportedly lost | Preliminary — needs replication |
| trishula/inbox/MI_AGENT_TO_CODEX_RV_ANSWERS.md | 2026-03-02 | N/A | d=-2.26 Mistral, d=-1.84 OPT, d=-1.14 GPT-2 XL (all cross-arch) | Most recent integrated summary — use these as canonical cross-arch d values |

---

*End of canonical results table. Prepared 2026-03-10.*
*This document reflects the state of the data as read from disk. It does not resolve the contradictions — it documents them. Resolution requires new experimental runs targeting the specific models identified in Part 3.*
