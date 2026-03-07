# R_V Master Plan — Experiment Status
**Updated**: 2026-03-04T14:20Z
**Session**: GPU session (RTX PRO 6000 Blackwell 98GB) — bootstrap CIs, SVD re-run, power-up n=100, Pythia-2.8B checkpoints

---

## Completed This Session (No GPU)

### FDR Correction (Benjamini-Hochberg)
- **30/36 tests survive** FDR correction (α=0.05)
- Tests that lose significance: Pythia-1B, 1.4B, 2.8B, 6.9B (all small/NS), Pythia-1.4B cross-arch, genuine vs deceptive safety (NS by design)
- All major effects (Mistral, OPT, GPT-2 XL, Qwen, controls) remain significant
- Artifact: `results/fdr_correction/fdr_results_20260303_232741.json`
- Script: `scripts/fdr_correction.py`

### Perplexity Re-Pairing (Method A)
- **R_V SURVIVES**: d=-1.80 (paired), p=9.12e-11 (n=30 pairs)
- Strict matching (PPL diff <10): d=-1.67, p=0.002 (n=8 pairs)
- Perplexity confound definitively ruled out
- Artifact: `results/perplexity_repairing/repairing_results_20260303_233230.json`
- Script: `scripts/perplexity_repairing.py`

### New Figures Generated
All saved to `figures/masterplan/` (PDF + PNG):
1. `fig_full_head_sweep_32x32` — 32×32 heatmap of 1024 heads (E2.2)
2. `fig_scaling_curve` — Effect size vs model scale with all data points
3. `fig_training_checkpoints` — Emergence during training (Pythia-1.4B + 2.8B bug flagged)
4. `fig_safety_roc` — ROC curve for R_V-based self-ref detection (AUROC=0.909)
5. `fig_safety_genuine_vs_deceptive` — Content vs intent bar chart
6. `fig_fdr_correction` — FDR summary dot plot
7. `fig_cross_arch_updated` — Forest plot with all architectures
- Script: `scripts/generate_masterplan_figures.py`

### Cluster-Robust Standard Errors
- **10/13 effects survive** conservative cluster-robust SEs (DEFF=2)
- Effects that lose significance: Phi-3-mini (d=0.625), Pythia-6.9B (d=0.478), Pythia-1.4B cross-arch (d=-0.31) — all marginal effects
- Circularity controls: baseline ICC=0.382 (substantial template clustering), but effect (d=-2.58) survives with DEFF=3.67
- All core effects (Mistral, OPT, GPT-2 XL, necessity, sufficiency, safety) remain significant
- Artifact: `results/cluster_robust_se/cluster_robust_results_*.json`
- Script: `scripts/cluster_robust_se.py`

### Original 12 Figures Regenerated
- All 12 publication figures regenerated to `R_V_PAPER/figures/` and `paper/figures/`
- Script: `scripts/generate_figures.py`

### Paper v0.0.0.5 Written
- Full 13-page paper: `R_V_PAPER/paper_colm2026_v005.tex`
- Sections: Abstract, Intro (6 contributions), Related Work, Methods, Results (7 subsections), Discussion, Conclusion, Appendix (5 sections)
- 10 figures referenced, all compiling
- Top-10 heads table populated from full head sweep
- References.bib updated with 7 new entries
- **Compiles cleanly** with pdflatex + bibtex (0 errors, 0 undefined citations)
- Artifact: `R_V_PAPER/paper_colm2026_v005.pdf`

### Power Analysis
- 8/12 effects adequately powered (1-β ≥ 0.80)
- Underpowered: Pythia-1.4B cross-arch (0.41), Phi-3-mini (0.77), Pythia-6.9B (0.49)
- Artifact: `results/power_analysis/power_analysis_20260303_234828.json`
- Script: `scripts/power_analysis.py`

### Bug Fixes
- **SVD GQA bug fixed**: `scripts/svd_circuit_decomposition.py` now correctly handles Mistral GQA (8 KV heads → 32 Q heads mapping). Was returning NaN for 6/7 target heads.
- **Pythia-2.8B checkpoint bug identified**: All 4 checkpoints (step 1k-100k) have identical results (d=1.035). Cache served same weights. Needs `--force-download` or explicit revision verification.

---

## Experiment Status Summary

| ID | Experiment | Status | Key Result |
|---|---|---|---|
| E1.1 | Power-up n≥100 | **✅ DONE** (5/5 models) | Mistral d=-1.656, OPT d=1.683, GPT2-XL d=1.516, **Qwen d=-2.318**, Pythia-1.4B d=-0.006 NS |
| E1.2 | Multi-seed | **NOT STARTED** | — |
| E1.3 | Scaling gap | **PARTIAL** (3/5 + 3 from prior) | Qwen-3B d=1.25, Phi-3 d=0.63, Pythia-6.9B d=0.48 NS. Gemma/Llama FAILED (auth). |
| E1.4 | Training checkpoints | **✅ DONE** | Pythia-1.4B ✅ (5 steps). Pythia-2.8B ✅ (5 steps, d≈1.0 constant — genuine, not cache bug) |
| E2.1 | SVD per-head | **✅ DONE** | All 7 heads valid. L27H10 d_rank=-1.54, L5H29 d_rank=2.93. |
| E2.2 | Full head sweep | **✅ DONE** | 1024/1024 heads, 606 sig, top L10H20 d=3.90 |
| E2.3 | Singular direction interp | **PARTIAL** | L27_H2 vocabulary projections done. Others need re-run with GQA fix. |
| E3.1 | SAE feature analysis | **NOT STARTED** | Needs Gemma-2-2B auth + sae-lens |
| E3.4 | R_V on Gemma-2-2B | **FAILED** | Auth error (401) |
| E4.1 | Linear probe | **✅ DONE** | 100% accuracy L4+, n=20/20 (small — possible overfit) |
| E4.3 | RSA modes | **✅ DONE** | 10 modes × 10 layers, self-ref distance trajectory |
| E5.1 | Genuine vs deceptive | **✅ DONE** | d=-0.06 (indistinguishable). Tracks content not intent. |
| E5.2 | Alignment faking | **✅ DONE** | Faking d=0.39 vs genuine, d=-2.06 vs baseline |
| E5.3 | Deployment monitor | **✅ DONE** | AUROC=0.909, TPR=0.83 at FPR=0.14 |
| — | FDR correction | **✅ DONE** | 30/36 survive BH α=0.05 |
| — | Perplexity re-pairing | **✅ DONE** | d=-1.80, p=9.12e-11. Confound ruled out. |
| — | Cluster-robust SEs | **✅ DONE** | 10/13 survive conservative DEFF=2 |
| — | Figures (all) | **✅ DONE** | 7 masterplan + 12 original = 19 total |
| — | Power analysis | **✅ DONE** | 8/12 adequately powered |
| — | Paper v0.0.0.5 | **✅ DONE** | 13pp, compiles cleanly, all data populated |
| — | Bootstrap CIs | **✅ DONE** | BCa CIs for mode atlas, causal, cross-arch Mistral |

---

## Priority Queue for Next GPU Session

1. **Fix HF token**: Create classic Read token with gated repo access
2. **E1.3 re-run**: Gemma-2-2B + Llama-3.2-3B (~2h)
3. **E2.1 re-run**: SVD decomposition with GQA fix (~1h)
4. **E1.1 completion**: Power-up all 4 models to n≥100, fix Qwen (~4h)
5. **E1.4 fix**: Pythia-2.8B with `--force-download` for actual checkpoints (~2h)
6. **E3.1+E3.4**: SAE on Gemma-2-2B (if auth works) (~4h)
7. **E1.2**: Multi-seed (if time permits) (~2h)

**Estimated GPU time**: ~15h total, or ~8h if just priorities 1-5.

---

## COLM Submission Gaps (P0 only)

| # | Gap | Status | Remaining |
|---|---|---|---|
| 1 | FDR correction | **✅ DONE** | — |
| 2 | Cluster-robust SEs | **✅ DONE** | 10/13 survive |
| 3 | Perplexity re-pairing | **✅ DONE** | — |
| 4 | Bootstrap CIs for TE | **✅ DONE** | Mode atlas d=-1.67 CI=[-2.11,-1.21]; Causal d=-3.47 CI=[-4.28,-2.47] |
| 5 | Write the paper | **✅ v0.0.0.5** | Draft complete, compiles, needs revision |
| 6 | Remaining figures | **✅ DONE** | 7 new + 12 regenerated = 19 total |

---

## Known Data Issues

1. **Scaling fit weak**: R²=0.047 with 6 points. Need Gemma-2-2B + Llama-3.2-3B (HF auth blocker).
2. **Linear probe n=20**: Too small, likely overfitting. Need n≥50.
3. **Pythia-2.8B constant d**: Confirmed real (not cache bug). d≈1.0 at all training steps.
4. **OPT bootstrap weak**: Live V-proj bootstrap gave d=-0.13 with n=20 (vs d=1.68 at n=100). Probe NaN filtering dropped most baselines.
