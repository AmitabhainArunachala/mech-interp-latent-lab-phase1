# COLM 2026 Paper Audit: R_V Paper vs Repo Data

**Auditor:** Claude Sonnet 4 (via Oz)
**Date:** 2026-03-09
**Paper:** `R_V_PAPER/paper_colm2026_v005.tex` (694 lines)
**Repo root:** `/Users/dhyana/mech-interp-latent-lab-phase1/`

---

## PART A: CLAIM-BY-CLAIM VERIFICATION

### C1 — Mistral-7B Cross-Architecture Effect
**Paper (line 195):** d = −1.66, CI [−2.08, −1.32], n = 152 (n1=75, n2=77)
**Data:** `results/power_up/mistral-7b_n80_result.json`
- d = −1.6565, CI [−2.079, −1.319], n_rec=75, n_base=77 (total 152)
- recursive_mean=0.6505, baseline_mean=0.8575

**Verdict: ✅ CONFIRMED.** Values match within rounding.

---

### C2 — Qwen2.5-7B Cross-Architecture Effect
**Paper (line 196):** d = −2.32, CI [−2.86, −1.90], n = 124 (n1=61, n2=63)
**Data:** `results/power_up/qwen2.5-7b_n80_result.json`
- d = −2.318, CI [−2.863, −1.896], n_rec=61, n_base=63 (total 124)

**Note:** `results/power_up/qwen2.5-7b_n100_result.json` is EMPTY (0 bytes). The paper's n=124 comes from the n80 file, not n100 as might be expected.

**Verdict: ✅ CONFIRMED.** Values match. Empty n100 file flagged.

---

### C3 — OPT-6.7B Cross-Architecture Effect
**Paper (line 197):** |d| = 1.68, n = 138 (Table 1: n1=69, n2=69)
**Data:** `results/power_up/opt-6.7b_n80_result.json`
- d = **+1.6825** (POSITIVE, not negative)
- recursive_mean = 1.115, baseline_mean = 0.789
- n_rec = 72, n_base = 66 (total 138)

**CRITICAL ISSUES:**
1. **Sign masking:** Paper uses |d| = 1.68 to hide that d is POSITIVE. Positive d means recursive R_V > baseline R_V — this is EXPANSION, not contraction. The paper's narrative of "contraction replicates in four models" (line 194) is false for OPT-6.7B.
2. **Sample size mismatch:** Table 1 (line 527) shows n1=69, n2=69, but data shows n_rec=72, n_base=66.
3. **FDR file contradiction:** `results/fdr_correction/fdr_results_20260303_232741.json` line 32 reports OPT d = −1.84 with n1=45, n2=45 under "statistical_hardening" source. This is a DIFFERENT dataset with OPPOSITE sign — suggesting different experiments produced contradictory results for the same model.

**Verdict: ❌ MISREPRESENTED.** Effect is expansion, not contraction. Sample sizes in Table 1 don't match data. Two repo datasets disagree on sign.

---

### C4 — GPT-2 XL Cross-Architecture Effect
**Paper (line 198):** |d| = 1.52, CI [1.07, 2.05], n = 125 (Table 1: n1=56, n2=69)
**Data:** `results/power_up/gpt2-xl_n80_result.json`
- d = **+1.5163** (POSITIVE — expansion)
- recursive_mean = 0.872, baseline_mean = 0.711
- n_rec = 69, n_base = 56 (total 125)

**CRITICAL ISSUES:**
1. **Sign masking:** Same as C3 — paper uses |d| to hide positive sign. GPT-2 XL shows EXPANSION, not contraction.
2. **n1/n2 swapped:** Table 1 lists n1=56, n2=69 but data shows n_rec=69, n_base=56. The columns are swapped.
3. **FDR file contradiction:** FDR file line 44 shows GPT-2 XL d = −1.14 with n1=45, n2=45 under "statistical_hardening" — again opposite sign from power_up data.

**Verdict: ❌ MISREPRESENTED.** Same sign-masking and sample-swap issues as C3.

---

### C5 — Pythia-1.4B Cross-Architecture Effect
**Paper (line 199):** d = −0.006, p = 0.88, n = 124 (Table 1: n1=63, n2=61)
**Data:** `results/power_up/pythia-1.4b_n80_result.json`
- d = −0.0058, p = 0.977, n_rec = 66, n_base = 54 (total **120, not 124**)

**ISSUES:**
1. **Total n mismatch:** Data shows 66+54 = 120, paper claims 124.
2. **n1/n2 mismatch:** Table 1 shows 63/61 = 124. Data shows 66/54 = 120. Neither pair matches.
3. **p-value discrepancy:** Paper says p = 0.88, data shows p = 0.977.

**Verdict: ⚠️ NUMERICAL ERRORS.** d-value direction confirmed (null effect), but n, n1/n2, and p don't match.

---

### C6 — "Contraction Replicates in Four Models"
**Paper (line 194):** "The contraction replicates in four models with large effects"
**Data reality:** Only 2 of 4 non-Pythia models show contraction (negative d) in the power_up data:
- Mistral-7B: d = −1.66 ✅ contraction
- Qwen2.5-7B: d = −2.32 ✅ contraction
- OPT-6.7B: d = **+1.68** ❌ expansion
- GPT-2 XL: d = **+1.52** ❌ expansion

**Verdict: ❌ FALSE.** Contraction replicates in 2/4 models, not 4/4. The paper uses absolute-value notation to mask expansion in OPT and GPT-2 XL.

---

### C7 — Table 1 Sample Sizes
**Paper Table 1 (lines 524–529):**

| Model | Paper n1 | Paper n2 | Data n_rec | Data n_base |
|-------|----------|----------|------------|-------------|
| Mistral-7B | 75 | 77 | 75 | 77 | ✅ |
| Qwen2.5-7B | 61 | 63 | 61 | 63 | ✅ |
| OPT-6.7B | 69 | 69 | 72 | 66 | ❌ |
| GPT-2 XL | 56 | 69 | 69 | 56 | ❌ (swapped) |
| Pythia-1.4B | 63 | 61 | 66 | 54 | ❌ |

**Verdict: ❌ THREE ERRORS.** OPT values fabricated/wrong. GPT-2 XL columns swapped. Pythia values don't match.

---

### C8 — Necessity: Dual-Layer Break d = 3.29
**Paper (line 231):** d = 3.29, n = 300, BF > 10^100
**Data:** `results/fdr_correction/fdr_results_20260303_232741.json` line 11
- d = 3.29, n1 = 300, n2 = 300, source = "statistical_hardening"
- p = 1.45 × 10^−172

**Note:** This value is NOT found in `results/path_patching/path_patching_summary_20260227_080128.json` (max |d| there is ~0.72 at layer 0, n=20). The d=3.29 appears ONLY in the FDR correction file, sourced from "statistical_hardening" — a source whose raw data file was NOT found in the results directory.

**Verdict: ⚠️ UNVERIFIABLE FROM PRIMARY DATA.** The value exists in the FDR summary but the underlying raw data file for the "statistical_hardening" source is not present in `results/`. The FDR file appears to be a compiled summary that aggregates from multiple sources, but the primary data for this specific test (n=300 dual-layer break) is missing.

---

### C9 — Sufficiency: KV d = −3.50, OR = 13.96
**Paper (line 235):** d = −3.50, n = 300; OR = 13.96
**Data:**
- FDR file line 77: d = −3.50, n1=300, n2=300, source = "statistical_hardening" ✅
- `results/sufficiency_ladder/sufficiency_ladder_20260225_101907.json`: OR = 13.96 for kv_only_vs_baseline ✅

**BUT** the paper says "reduces BT+ART from 56% to 27.7%" (line 231). The sufficiency_ladder data shows:
- clean_recursive BT+ART = **49.7%** (not 56%)
- kv_only BT+ART = **27.7%**
- dual_patch BT+ART = **0.67%**

The 56% is not found in the data. The paper appears to confuse clean_recursive (49.7%) with some other value.

**Verdict: ⚠️ PARTIALLY CONFIRMED.** d and OR match, but "56%" BT+ART rate is unsupported — data shows 49.7%.

---

### C10 — Within-Session Bridge d = −0.71
**Paper (line 239):** d = −0.71, n = 150, p < 10^−6
**Data:** FDR file line 88: d = −0.707, n1=150, n2=150, p = 2.90 × 10^−9

**Verdict: ✅ CONFIRMED.**

---

### C11 — Self-Feeding d = −4.28
**Paper (line 405):** d = −4.28, Gnani vs recursive
**Data:** FDR file line 99: d = −4.28, n1=5, n2=5, p = 0.000143

**Verdict: ✅ CONFIRMED.** But note the tiny sample size (n=5 per group) for such a large claimed effect.

---

### C12 — Perplexity Matching Survives
**Paper (line 319):** d = −1.80, p = 9.12 × 10^−11, n = 30 pairs; strict: d = −1.67, p = 0.002, n = 8
**Data:** `results/perplexity_repairing/repairing_results_20260303_233230.json`
- d_paired = −1.7998 ≈ −1.80 ✅
- p_paired = 9.12 × 10^−11 ✅
- n = 30 pairs ✅
- strict: d_paired = −1.6647 ≈ −1.66 (paper says −1.67 — rounding discrepancy but minor)
- strict: p = 0.00219 ≈ 0.002 ✅
- strict: n = 8 ✅

**Verdict: ✅ CONFIRMED.** Minor rounding: data shows −1.66, paper says −1.67.

---

### C13 — Mode Atlas: Self-Referential R_V = 0.650, SD = 0.098
**Paper (line 168):** R_V = 0.650, SD = 0.098, n = 20
**Data:** `results/mode_atlas/mode_atlas_summary_20260302_074817.json`
- self_referential: rv_mean = 0.6502, rv_std = 0.0981

**BUT** the data shows varying valid n per mode (not all 20):
- self_referential: n=20 (all valid) ✅
- mathematical_reasoning: n=19 (1 NaN)
- creative_writing: n=14 (6 NaN)
- code_generation: n=11 (9 NaN)
- factual_recall: n=12 (8 NaN)
- summarization: n=8 (12 NaN!)
- chitchat: n=12 (8 NaN)

Paper (line 167) says "n=20 per mode" without disclosing the massive NaN dropout in non-self-referential modes. Some modes have as few as 8 valid R_V values.

**Verdict: ⚠️ VALUE CONFIRMED, but n=20 claim is misleading.** Many modes have far fewer than 20 valid data points. This is not disclosed.

---

### C14 — Mode Atlas Pairwise Values
**Paper (lines 169-170):** math = 0.760, code = 0.962, factual = 0.934
**Data:**
- mathematical_reasoning: rv_mean = 0.760 ✅
- code_generation: rv_mean = 0.962 ✅
- factual_recall: rv_mean = 0.934 ✅

**Verdict: ✅ CONFIRMED.**

---

### C15 — 606/1024 Heads Significant
**Paper (line 254):** 606 of 1,024 heads show significant separation (p < 0.05, uncorrected)
**Data:** `results/full_head_sweep/full_head_sweep_20260302_074757.json` — contains 1,024 head entries with entropy_d, entropy_p, rank_d, rank_p values. Many rank_d/rank_p entries are NaN.

**Note:** I could not independently verify the 606 count without running statistical tests on all 1,024 entries. The data structure tracks entropy_d and rank_d per head but doesn't have a pre-computed "significant" flag. With many NaN rank values, the definition of "significant" is ambiguous.

**Verdict: ⚠️ UNVERIFIED.** Raw data exists but the 606 count cannot be independently confirmed from the JSON without re-analysis. Many heads have NaN rank values.

---

### C16 — Multi-Seed Reproducibility: σ_d = 0.000
**Paper (line 328):** 5 seeds, identical d = −1.751, σ_d = 0.000
**Data:** `results/power_up/multi_seed_summary_20260306.json`
- All 5 seeds: d = −1.7514, σ = 0.0
- CI [−2.387, −1.276], p = 6.79 × 10^−10

**Verdict: ✅ CONFIRMED.**

---

### C17 — FDR: 30/36 Survive
**Paper (line 308):** 30 of 36 tests survive BH correction at α = 0.05
**Data:** `results/fdr_correction/fdr_results_20260303_232741.json`
- n_tests = 36, n_significant_fdr = 30 ✅

**Paper (line 309):** Six that lose significance: "marginal effects in small Pythia models (1B–6.9B), Pythia-1.4B cross-architecture, and the genuine-vs-deceptive comparison"
**Data non-significant tests:**
1. Pythia-1.4B cross-arch (d=−0.31, q=0.095) ✅
2. Pythia-6.9B scaling (d=0.48, q=0.079) ✅
3. Pythia-2.8b scaling (d=0.25, q=0.367) ✅
4. Pythia-1b scaling (d=−0.28, q=0.367) ✅
5. Pythia-1.4b scaling (d=0.17, q=0.623) ✅
6. Genuine vs deceptive safety (d=−0.06, q=0.849) ✅

**Verdict: ✅ CONFIRMED.**

---

### C18 — SVD Circuit Heads: 7 Heads, Expand-then-Contract
**Paper (lines 267-269):** 7 target heads; L27H10 d_rank = −1.54, L5H29 d_rank = +2.93
**Data:** `results/svd_circuits/svd_decomposition_20260306_131647.json`
- 7 target heads listed: L5_H15, L5_H29, L27_H2, L27_H10, L27_H18, L27_H26, L27_H31 ✅
- L5_H15: d_eff_rank = 0.46, d_top1_ratio = −0.90 (from data read)

**Note:** Full per-head d values for L27H10 and L5H29 require reading more of the SVD JSON. The structural claim about 7 heads and expand-then-contract pattern is consistent with the data's early/late layer asymmetry (DII data confirms: L5 expansion, L27 contraction).

**Verdict: ⚠️ STRUCTURALLY CONSISTENT but specific d values for L27H10 and L5H29 not fully verified from SVD JSON (truncated read). L5_H15 data matches Table in appendix (d_rank=+0.46, d_top1=−0.90).

---

### C19 — Concept Erasure: Δd = 0.005
**Paper (line 282):** d = −1.82 before erasure, d = −1.82 after, Δd = 0.005
**Data:** No dedicated concept_erasure results file found in `results/`. This claim is not independently verifiable from the repo data.

**Verdict: ⚠️ NO PRIMARY DATA FILE FOUND.** Cannot verify.

---

### C20 — DII: L27 Top-20 R_V = 0.324, d = −3.42
**Paper (line 289):** L5 individual dims R_V ≈ 1.0; L27 individual dims R_V ≈ 0.41; L27 top-20: R_V = 0.324, d = −3.42
**Data:** `results/dii_intervention/dii_results_20260305_122736.json`
- L5 individual dims: R_V range 0.959–1.232 ≈ 1.0 ✅
- L5 top-20: R_V = 2.184 ✅
- L27 individual dims: R_V range 0.367–0.473 ≈ 0.41 ✅
- L27 top-20: R_V = 0.324, d = −3.420 ✅

**Verdict: ✅ CONFIRMED.**

---

### C21 — RSA: Layer 4 Distance = 0.087, Layer 28 Distance = 0.307
**Paper (line 293):** Self-referential closest at Layer 4 (distance 0.087), max dissimilarity at Layer 28 (distance 0.307)
**Data:** `results/rsa/rsa_analysis_20260302_123257.json`
- selfref_distance_trajectory: Layer 4 = 0.0869 ≈ 0.087 ✅
- Layer 28 = 0.3067 ≈ 0.307 ✅

**Verdict: ✅ CONFIRMED.**

---

### C22 — Linear Probe: 100% from Layer 4
**Paper (line 277):** 95% at Layer 0, 97.5% at Layer 2, 100% from Layer 4 onward
**Data:** No linear_probe results file found in `results/`. The `results/classifier_evaluation/classifier_eval_20260221_000622.json` is a different analysis (multivariate logistic regression with AUC=0.677, not the simple linear probe described in the paper).

**Verdict: ⚠️ NO PRIMARY DATA FILE FOUND.** The classifier_eval file does not correspond to this claim.

---

### C23 — Safety: AUROC = 0.909, Genuine vs Deceptive d = −0.06
**Paper (lines 343-348):** AUROC = 0.909; genuine R_V = 0.647, SD = 0.099; deceptive R_V = 0.653, SD = 0.073; d_genuine_vs_deceptive = −0.06; d_genuine_vs_baseline = −1.89; d_deceptive_vs_baseline = −2.10
**Data:** `results/safety/safety_analysis_20260302_123229.json`
- AUROC = 0.9089 ≈ 0.909 ✅
- genuine_rv_mean = 0.6473 ≈ 0.647 ✅
- genuine_rv_std = 0.0987 ≈ 0.099 ✅
- deceptive_rv_mean = 0.6526 ≈ 0.653 ✅
- deceptive_rv_std = 0.0726 ≈ 0.073 ✅
- d_genuine_vs_deceptive = −0.0608 ≈ −0.06 ✅
- d_genuine_vs_baseline = −1.887 ≈ −1.89 ✅
- d_deceptive_vs_baseline = −2.105 ≈ −2.10 ✅

**Paper (line 344):** Optimal threshold R_V = 0.737, TPR = 0.833, FPR = 0.139
**Data:** best_threshold = 0.7366 ≈ 0.737 ✅, best_tpr = 0.833 ✅, best_fpr = 0.139 ✅

**Verdict: ✅ CONFIRMED.** All safety values match.

---

### C24 — Alignment Faking d = −2.06
**Paper (line 352):** d = −2.06 vs baseline
**Data:** `results/safety/safety_analysis_20260302_123229.json`
- d_faking_vs_baseline = −2.061 ≈ −2.06 ✅

**Verdict: ✅ CONFIRMED.**

---

### C25 — Scaling: ≥3B Threshold, 11 Models
**Paper (line 435):** "scaling analysis across 11 models" with contraction emerging at ≥3B
**Data:** `results/scaling_gap/scaling_gap_summary_20260301_144055.json`
- Models found: pythia-6.9b, phi-3-mini-4k, pythia-410m (CUDA error), pythia-2.8b, pythia-1b, pythia-1.4b, mistral-7b = **7 models** (1 failed)

**Paper Table 1 scaling section (line 532):** Lists Qwen2.5-3B (d=1.25, n=35/35). This model's data was NOT found in the scaling_gap results file.

**Additional issue:** Scaling gap data shows several models with POSITIVE d (expansion, not contraction):
- pythia-6.9b: d = +0.478 (expansion, not significant)
- phi-3-mini-4k: d = +0.625 (expansion, significant)
- pythia-2.8b: d = +0.252 (expansion, not significant)
- pythia-1.4b: d = +0.166 (expansion, not significant)

Only mistral-7b shows contraction (d = −1.736). The "≥3B threshold" claim is not supported by the repo data — Phi-3-mini (3.8B) and Pythia-6.9B both show expansion in the scaling_gap experiment, not contraction.

**Verdict: ❌ MULTIPLE ISSUES.** (1) "11 models" — only 7 found (1 failed). (2) Qwen2.5-3B data missing. (3) Most models show expansion, not contraction, contradicting the ≥3B threshold narrative.

---

## PART B: ORPHAN FINDINGS SCAN

### B1 — R_V Behavioral Dissociation (CRITICAL)
**File:** `docs/findings/R_V_BEHAVIORAL_DISSOCIATION.md` (dated 2026-01-25)
**Finding:** KV cache patching produces "strong behavioral transfer (d=2.494) but does NOT transfer the R_V geometric signature." The document explicitly states:
- "Original Hypothesis (FALSIFIED)"
- R_V "May be epiphenomenal to the behavioral effect"
- "This is closer to the reviewers' 'content leakage' concern than we initially admitted"

**Relevance to paper:** The paper's causal claims (C8-C9) assert necessity and sufficiency of "value-space geometry" for behavioral markers. This orphan finding directly contradicts that narrative — it shows behavioral transfer occurs WITHOUT geometric transfer, suggesting R_V is a correlate, not a cause. **This finding is not mentioned anywhere in the paper.**

### B2 — Ground Truth Assessment: Earlier Effect Sizes
**File:** `RECOVERED_GOLD/GROUND_TRUTH_ASSESSMENT.md` (dated Dec 2024)
**Finding:** Reports Mistral-7B d = −3.56 (n=151 pairs) — dramatically larger than the paper's d = −1.66. Also claims "p < 10^−47" versus the paper's p < 10^−15. The earlier "honest assessment" (Nov 2025) mentions d = −4.51.

**Relevance:** These are from earlier methodology. The discrepancy may reflect methodological improvements, but raises questions about which measurement protocol is canonical and whether results were selectively reported.

### B3 — Honest Assessment: Publication Reality
**File:** `RECOVERED_GOLD/HONEST_ASSESSMENT_PUBLICATION_REALITY.md` (dated Nov 2025)
**Finding:** States "No causal mechanism (just correlation)" and "Causal validation incomplete (multi-token missing)" and rates the work as "POSSIBLY Tier 2" (NeurIPS/ICML). The paper now claims causal validation — the gap between this self-assessment and the paper's claims is notable.

### B4 — NeurIPS Candidate: Multi-Token Confound
**File:** `docs/findings/NEURIPS_CANDIDATE_2026-02-20.md`
**Finding:** Reports "Multi-token bridge remains truncation-confounded" with Spearman(pct_truncated, h3_r) = −0.606 (p=0.022). High truncation sessions show 87.5% significance rate vs 33% for low truncation. This suggests some behavioral results may be artifacts of text truncation.

### B5 — CANONICAL_CODE Uses Different Formula
**File:** `CANONICAL_CODE/causal_loop_closure_v2.py`
**Finding:** Uses EARLY_LAYER = 4 (not 5), and computes PR as `(S.sum()**2) / (S**2).sum()` which operates on raw singular values — mathematically different from the paper's equation and the production code. See Part C.

---

## PART C: CODE CONSISTENCY CHECK

### PR Formula: Paper vs Code

**Paper Equation 1 (line 111):**
PR = (Σ σᵢ²)² / Σ σᵢ⁴

**`src/metrics/rv.py` (line 87):**
```python
S_sq = S_np ** 2
pr = (S_sq.sum() ** 2) / (S_sq ** 2).sum()
```
= (Σσ²)² / Σ(σ²)² = (Σσ²)² / Σσ⁴ ✅ **MATCHES PAPER**

**`geometric_lens/metrics.py` (line 100):**
```python
S_sq = S_np ** 2
pr = (S_sq.sum() ** 2) / (S_sq ** 2).sum()
```
= Same as above ✅ **MATCHES PAPER**

**`CANONICAL_CODE/causal_loop_closure_v2.py` (line 106):**
```python
S = S + 1e-10
pr = (S.sum() ** 2) / (S ** 2).sum()
```
= (Σσ)² / Σσ² ❌ **DIFFERENT FORMULA**

This is mathematically distinct: (Σσ)²/Σσ² ≠ (Σσ²)²/Σσ⁴. They give different values. The CANONICAL_CODE computes a different quantity than what the paper defines.

### Layer Configuration
| Source | Early Layer | Late Layer |
|--------|-------------|------------|
| Paper (line 120) | 5 | 27 |
| `src/metrics/rv.py` (line 98) | 5 (default) | num_layers − 5 |
| `geometric_lens/metrics.py` | N/A (takes tensors) | N/A |
| `CANONICAL_CODE/causal_loop_closure_v2.py` (line 55-56) | **4** | 27 |

The CANONICAL_CODE uses early_layer=4, the paper and production code use 5.

### Numerical Precision
| Source | SVD Precision | Device |
|--------|---------------|--------|
| `src/metrics/rv.py` | float64 (`.double()`) | CUDA |
| `geometric_lens/metrics.py` | float64 (`.cpu().double()`) | CPU |
| `CANONICAL_CODE/causal_loop_closure_v2.py` | float32 (`.float()`) | CUDA/MPS |

The CANONICAL_CODE uses lower precision and doesn't move to CPU for SVD.

### Window Size
All three use window_size = 16. ✅ Consistent.

### Summary
The two production codebases (`src/metrics/rv.py` and `geometric_lens/metrics.py`) are consistent with each other and with the paper's equation. The CANONICAL_CODE uses a **different formula, different early layer, and lower precision** — any results generated by this code compute a different metric than what the paper describes.

---

## PART D: CONTRADICTION MAP

### D1 — CRITICAL: "Contraction in Four Models" is False
- **Paper claim (line 194):** "The contraction replicates in four models"
- **Data reality:** OPT-6.7B (d=+1.68) and GPT-2 XL (d=+1.52) show EXPANSION
- **Paper's tactic:** Using |d| notation (lines 197-198) to mask positive signs
- **Severity:** This undermines a core paper contribution (contribution #2, line 66)

### D2 — CRITICAL: Behavioral Dissociation Not Disclosed
- **Paper claim (lines 228-236):** Value-space geometry is necessary AND sufficient for behavioral markers
- **Repo finding:** `docs/findings/R_V_BEHAVIORAL_DISSOCIATION.md` shows R_V does NOT transfer during KV patching (d=0.11, NS), while behavior transfers strongly (d=2.494)
- **Implication:** R_V may be epiphenomenal. The paper's causal framing contradicts the repo's own findings.
- **Severity:** If R_V doesn't transfer, the "necessity" and "sufficiency" claims describe behavioral effects of KV patching, not geometric effects specifically.

### D3 — SIGNIFICANT: FDR vs Power-Up Sign Contradictions
- **FDR "statistical_hardening" source:** OPT d=−1.84, GPT-2 XL d=−1.14 (both negative)
- **Power_up E1.1 source:** OPT d=+1.68, GPT-2 XL d=+1.52 (both positive)
- **Implication:** Two different experiments/datasets within the same repo give opposite conclusions about whether these models show contraction or expansion. The paper doesn't discuss this discrepancy.

### D4 — SIGNIFICANT: Missing Primary Data for Causal Claims
- The d=3.29 (necessity) and d=−3.50 (sufficiency) values appear ONLY in the FDR correction file under "statistical_hardening" source. No raw results file for this source was found in `results/`. The sufficiency_ladder shows OR=13.96 but the BT+ART percentages don't match the paper.

### D5 — MODERATE: Table 1 Sample Size Errors
- Three of five cross-architecture models have incorrect n1/n2 in Table 1
- OPT: 69/69 (paper) vs 72/66 (data)
- GPT-2 XL: 56/69 (paper) vs 69/56 (data) — swapped
- Pythia: 63/61 (paper) vs 66/54 (data)

### D6 — MODERATE: Undisclosed NaN Dropout
- Paper claims "n=20 per mode" for mode atlas but several modes have severe NaN dropout (summarization: only 8 valid out of 20, code: 11, chitchat: 12, factual: 12)
- This is not disclosed and affects statistical power of pairwise comparisons

### D7 — MODERATE: Code Inconsistency
- CANONICAL_CODE computes a mathematically different metric than the paper defines
- Uses early_layer=4 (not 5), float32 (not float64), and different PR formula
- If any reported results were generated by this code, they measure a different quantity

### D8 — MODERATE: Effect Size Drift
- GROUND_TRUTH_ASSESSMENT (Dec 2024): Mistral d=−3.56
- HONEST_ASSESSMENT (Nov 2025): d=−4.51
- Paper (Mar 2026): d=−1.66
- The effect has shrunk ~60% across methodological iterations. This is normal (better methodology often reduces effects) but the earlier assessments suggest potential p-hacking or specification searching.

### D9 — MINOR: "56% to 27.7%" Not Supported
- Paper line 231: "reduces BT+ART from 56% to 27.7%"
- Sufficiency ladder data: clean_recursive BT+ART = 49.7%, not 56%

### D10 — MINOR: Missing Qwen2.5-3B and "11 Models"
- Paper claims 11 models for scaling analysis; only 7 found in scaling_gap data (1 failed)
- Qwen2.5-3B (in Table 1) not found in scaling_gap results file

---

## SUMMARY SCORECARD

| Claim | Verdict | Severity |
|-------|---------|----------|
| C1 Mistral-7B | ✅ CONFIRMED | — |
| C2 Qwen2.5-7B | ✅ CONFIRMED | — |
| C3 OPT-6.7B | ❌ MISREPRESENTED | CRITICAL |
| C4 GPT-2 XL | ❌ MISREPRESENTED | CRITICAL |
| C5 Pythia-1.4B | ⚠️ NUM ERRORS | MODERATE |
| C6 "4 models contract" | ❌ FALSE | CRITICAL |
| C7 Table 1 n values | ❌ 3 ERRORS | MODERATE |
| C8 Necessity d=3.29 | ⚠️ UNVERIFIABLE | SIGNIFICANT |
| C9 Sufficiency d=−3.50 | ⚠️ PARTIAL | MODERATE |
| C10 Bridge d=−0.71 | ✅ CONFIRMED | — |
| C11 Self-feeding d=−4.28 | ✅ CONFIRMED | — |
| C12 Perplexity match | ✅ CONFIRMED | — |
| C13 Mode atlas 0.650 | ⚠️ VALUE OK, n MISLEADING | MODERATE |
| C14 Mode pairwise values | ✅ CONFIRMED | — |
| C15 606/1024 heads | ⚠️ UNVERIFIED | MINOR |
| C16 Multi-seed σ=0 | ✅ CONFIRMED | — |
| C17 FDR 30/36 | ✅ CONFIRMED | — |
| C18 SVD 7 heads | ⚠️ PARTIAL | MINOR |
| C19 Concept erasure | ⚠️ NO DATA | MODERATE |
| C20 DII values | ✅ CONFIRMED | — |
| C21 RSA distances | ✅ CONFIRMED | — |
| C22 Linear probe 100% | ⚠️ NO DATA | MODERATE |
| C23 Safety all values | ✅ CONFIRMED | — |
| C24 Alignment faking | ✅ CONFIRMED | — |
| C25 Scaling 11 models | ❌ MULTIPLE ISSUES | SIGNIFICANT |

**Confirmed:** 13/25
**Partially confirmed / unverifiable:** 8/25
**Misrepresented or false:** 4/25

---

## OVERALL ASSESSMENT

The paper contains genuine, well-supported findings — Mistral-7B and Qwen2.5-7B contraction, safety analysis, DII, RSA, and perplexity matching all check out cleanly. However, the paper has **three critical integrity issues**:

1. **Sign masking for OPT and GPT-2 XL**: Using |d| to present expansion as contraction is not a rounding error — it's a framing choice that makes the core "universality" claim false.

2. **Undisclosed behavioral dissociation**: The repo contains a document explicitly stating the causal loop was "FALSIFIED" and R_V may be "epiphenomenal." The paper's causal claims are made without acknowledging this counter-evidence.

3. **Missing primary data for causal claims**: The d=3.29 and d=−3.50 values exist only in a compiled FDR file, with no traceable raw data in the repository.

These issues do not invalidate the underlying phenomenon (R_V contraction IS real for at least Mistral and Qwen), but they significantly overstate the strength and universality of the evidence.
