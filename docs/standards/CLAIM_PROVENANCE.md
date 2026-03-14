# Claim Provenance Map

**Date:** 2026-03-10
**Purpose:** Every number in the paper mapped to its raw data file and field.
**Rule:** If a number cannot be traced here, it cannot appear in the paper.

---

## Cross-Architecture (paper lines 194-199)

| Paper Claim | Paper Line | Value | Raw File | Field/Path | Status |
|---|---|---|---|---|---|
| Mistral d=-1.66 | 195 | -1.6565 | `results/power_up/mistral-7b_n80_result.json` | `.cohens_d` | CORRECT |
| Mistral CI [-2.08,-1.32] | 195 | [-2.08,-1.32] | same file | `.ci_95` | CORRECT |
| Mistral n=152 | 195 | 75+77=152 | same file | `.n_recursive` + `.n_baseline` | CORRECT |
| Qwen d=-2.32 | 196 | -2.3181 | `results/power_up/qwen2.5-7b_n80_result.json` | `.cohens_d` | CORRECT (but measured at wrong layer depth) |
| Qwen n=124 | 196 | 61+63=124 | same file | `.n_recursive` + `.n_baseline` | CORRECT |
| OPT \|d\|=1.68 | 197 | d=**+1.6825** | `results/power_up/opt-6.7b_n80_result.json` | `.cohens_d` | **SIGN HIDDEN** — d is positive (expansion) |
| OPT n=138 | 197 | 72+66=138 | same file | `.n_recursive` + `.n_baseline` | CORRECT (paper Table 1 says 69/69 — wrong) |
| GPT-2 \|d\|=1.52 | 198 | d=**+1.5163** | `results/power_up/gpt2-xl_n80_result.json` | `.cohens_d` | **SIGN HIDDEN** — d is positive (expansion) |
| GPT-2 n=125 | 198 | 69+56=125 | same file | `.n_recursive` + `.n_baseline` | CORRECT (paper Table 1 says 56/69 — swapped) |
| Pythia d=-0.006 | 199 | -0.00566 | `results/power_up/pythia-1.4b_n80_result.json` | `.cohens_d` | CORRECT |
| Pythia n=124 | 199 | 66+54=120 | same file | `.n_recursive` + `.n_baseline` | **WRONG** — actual n=120, paper says 124 |

---

## Causal: Necessity (paper line 231)

| Paper Claim | Value | Raw File | Field/Path | Status |
|---|---|---|---|---|
| "breaking both V-projections at L25 and L27" | L18 residual + L27 V-proj | `results/persistent_patching_v3/persistent_patching_v3_dual_20260225_002604.json` | `.r_layer`=18, `.v_layer`=27 | **WRONG LAYERS AND COMPONENTS** |
| BT+ART 56% | 0.56 | same file | `.aggregated.recursive_clean.bt_art_rate` | CORRECT |
| BT+ART → 27.7% | **3.67%** | same file | `.aggregated.recursive_dual_patched.bt_art_rate` = 0.0367 | **WRONG** — 27.7% is from sufficiency ladder KV injection |
| d=3.29 | NOT IN FILE | same file | Not stored — only `.comparisons.break_test.or`=33.44 | **HAND-DERIVED, not in raw data** |
| n=300 | 300 turns | same file | `.aggregated.recursive_clean.total_turns` | CORRECT value, but unit is "turn" not independent sample |
| OR=33.4 | 33.44 | same file | `.comparisons.break_test.or` | CORRECT |

---

## Causal: Sufficiency (paper lines 235-236)

| Paper Claim | Value | Raw File | Field/Path | Status |
|---|---|---|---|---|
| d=-3.50 | NOT IN ANY RAW FILE | `scripts/statistical_hardening.py:253` | Hardcoded with comment "approximate from OR=13.96" | **FABRICATED** |
| OR=13.96 | 13.96 | `results/sufficiency_ladder/sufficiency_ladder_20260225_101907.json` | `.comparisons.kv_only_vs_baseline.turn_level.or` | CORRECT |
| n=300 | 10 sessions × 30 turns | same file | `.n_sessions_per_condition`=10, `.max_turns_per_session`=30 | CORRECT value, unit is "turn" |
| "sufficient to induce recursive behavior" | CONTRADICTED | `docs/findings/R_V_BEHAVIORAL_DISSOCIATION.md` | KV transfers behavior (OR=13.96) but NOT geometry (d=0.11 NS) | **CLAIM CONTRADICTED by own data** |

---

## Within-Session Bridge (paper line 239)

| Paper Claim | Value | Raw File | Field/Path | Status |
|---|---|---|---|---|
| d=-0.71 | -0.7072 | `results/within_session_bridge/within_session_bridge_20260220_201515.json` | `.pooled.recursive_only.output_rv.cohens_d` | CORRECT |
| n=150 | n1=80, n2=107 | same file | `.pooled.recursive_only.output_rv.n_bt_art` / `.n_other` | **WRONG** — paper says 150/150 |
| p < 10^-6 | 9.16e-06 | same file | `.pooled.recursive_only.output_rv.mannwhitney_p` | CORRECT (p < 10^-5, not 10^-6) |

---

## Mode Atlas (paper line 200)

| Paper Claim | Value | Raw File | Field/Path | Status |
|---|---|---|---|---|
| d=-1.67 | -1.665 | `results/mode_atlas/mode_atlas_20260301_*.json` | needs verification | APPROXIMATE MATCH |
| CI [-2.11, -1.21] | from cluster-robust SE | `results/cluster_robust_se/` | needs verification | APPROXIMATE MATCH |

---

## Head Sweep (paper Table 2)

| Paper Claim | Value | Raw File | Field/Path | Status |
|---|---|---|---|---|
| Top 10 heads | see table | `results/full_head_sweep/full_head_sweep_20260302_*.json` | per-head d values | NEEDS VERIFICATION |
| 606/1024 heads p<0.05 | 606 | same file or FDR file | BH-corrected count | NEEDS VERIFICATION |

---

## SVD Circuit (paper Table 3)

| Paper Claim | Value | Raw File | Field/Path | Status |
|---|---|---|---|---|
| L27H10 d_rank=-1.54 | -1.544 | `results/svd_circuits/svd_circuit_20260302_*.json` | per-head rank d | NEEDS VERIFICATION |
| L5H29 d_rank=+2.93 | +2.928 | same file | per-head rank d | NEEDS VERIFICATION |

---

## Safety (paper Section 4.4)

| Paper Claim | Value | Raw File | Field/Path | Status |
|---|---|---|---|---|
| AUROC=0.909 | 0.9089 | `results/safety/safety_analysis_*.json` | `.auroc` or similar | NEEDS VERIFICATION |
| genuine vs deceptive d=-0.06 | -0.0608 | same file | `.cohens_d_genuine_vs_deceptive` | NEEDS VERIFICATION |

---

## Table 1 n-value Errors (paper lines 527-529)

| Model | Paper n1/n2 | Actual n1/n2 | Source |
|---|---|---|---|
| OPT-6.7B | 69/69 | 72/66 | `results/power_up/opt-6.7b_n80_result.json` |
| GPT-2 XL | 56/69 | 69/56 | `results/power_up/gpt2-xl_n80_result.json` |
| Pythia-1.4B | 63/61 | 66/54 | `results/power_up/pythia-1.4b_n80_result.json` |

---

## Self-Feeding (paper Table 1, line 522)

| Paper Claim | Value | Raw File | Field/Path | Status |
|---|---|---|---|---|
| d=-4.28 | 4.2765 (computed) | `results/self_feeding_loop/gnani_scaffolded_*.json` + `self_feed_recursive_*.json` | `.bt_art_rate` per file, 5+5 | CORRECT (sign was wrong in paper — gnani > recursive) |
| n1=5, n2=5 | 5/5 | 5 gnani + 5 recursive files | file count | CORRECT |

---

## Scaling Gap (paper Table 1, lines 532-534)

| Model | Paper n1/n2 | Actual n1/n2 | Source |
|---|---|---|---|
| Qwen2.5-3B | 35/35 | **19/18** | `results/scaling_gap/qwen2.5-3b_result.json` |
| Phi-3-mini-4k | 38/39 | 38/39 | `results/scaling_gap/phi-3-mini-4k_result.json` |
| Pythia-6.9B | 37/31 | 37/31 | `results/scaling_gap/pythia-6.9b_result.json` |

---

## Summary of Errors Found

| Error | Severity | Paper Location |
|---|---|---|
| d=-3.50 fabricated | CRITICAL | line 235 |
| "four models contract" — only 2 do | CRITICAL | line 194 |
| OPT/GPT-2 sign hidden by \|d\| | CRITICAL | lines 197-198 |
| "sufficient" contradicted by dissociation data | CRITICAL | line 236 |
| BT+ART 27.7% → actual 3.7% | HIGH | line 231 |
| "L25 and L27" → actual L18 + L27 | HIGH | line 231 |
| Pythia n=124 → actual 120 | MODERATE | line 199 |
| OPT n=69/69 → actual 72/66 | MODERATE | Table 1 |
| GPT-2 n=56/69 → actual 69/56 | MODERATE | Table 1 |
| Qwen-3B n=35/35 → actual 19/18 | MODERATE | Table 1 |
| Bridge n=150/150 → actual 80/107 | MODERATE | line 239 |
| Bridge p < 10^-6 → actual 9.16e-06 | LOW | line 239 |
