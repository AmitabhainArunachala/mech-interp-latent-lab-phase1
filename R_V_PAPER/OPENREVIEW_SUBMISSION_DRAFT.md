# COLM 2026 OpenReview Submission — Ready to Copy-Paste

**Deadlines** (Bali time, UTC+8):
- Abstract registration: Mar 27, 2026, 7:59 PM WITA
- Full paper: Apr 1, 2026, 7:59 PM WITA

**Submission URL**: https://openreview.net/group?id=colmweb.org/COLM/2026/Conference

---

## Title

Geometric Signatures of Self-Referential Processing in Transformer Representations

## Abstract (from paper v007)

We identify a measurable geometric signature of self-referential processing in transformer language models using the relative participation ratio (R_V), a spectral metric comparing the effective dimensionality of representations at early and late network layers. When a model processes prompts that invoke recursive self-observation, late-layer representations undergo characteristic geometric contraction (R_V < 1) relative to early-layer representations of the same input.

In a canonical base-model prompt-pass rerun, Mistral-7B-v0.1 shows strong contraction (d = -1.85, 95% CI [-2.27, -1.52], n_self=96, n_base=100 valid prompts). A recovered cross-architecture artifact and an independent power-up pipeline confirm the direction (d = -2.26 at n = 45 matched pairs; d = -1.66 at n = 75/77). Base-model path patching localizes the strongest causal site to the early residual stream (peak L5, d = 4.14), with weaker early V-projection effects (peak L5, d = 2.55) and sign-reversed late-layer effects at L27. Dual-layer ablation at n=300 turns per condition confirms that the measured geometry is necessary but not naively sufficient: breaking residual-stream and V-projection contributions at layers 18 and 27 reduces recursive behavioral markers from 44.7% to 0.0% (session d = 2.71), while baseline-to-patched sessions also collapse to 0.0% under heavily malformed outputs. A later micro-window multisite follow-up sharpens the mechanism story: holding the late L25 bridge fixed, a subtle L4 MLP assist improves recursive BT+ART beyond bridge-only steering in the window-8 confirmation family (47.2% versus 44.4%), while a shorter window-4 follow-up trades some lift for lower baseline spillover (44.4% recursive at 8.3% baseline versus 13.9% for bridge-only). Targeted SVD decomposition preserves an expand-then-contract motif (L5H29: d_rank = +0.94; L27H10: d_rank = -1.32). Cross-architecture evidence is mixed rather than universal: locked rows support contraction in Mistral-7B and Qwen2.5-7B, sign reversal across pipelines in OPT-6.7B and GPT-2 XL, and a null power-up result in Pythia-1.4B. Additional base canonical runs are still required before stronger universality claims. For AI safety, R_V < 0.737 detects self-referential content with AUROC = 0.909, tracking semantic content rather than intent.

## TL;DR (for OpenReview, max ~100 words)

Self-referential prompts cause a geometric contraction in transformer representations, measured by R_V, the ratio of late- to early-layer effective dimensionality. In a recovered base Mistral-7B-v0.1 artifact, contraction is strong (d = -2.26, n = 45 pairs), while the clearest recovered result is causal: dual-layer ablation drops recursive behavioral markers from 56.0% to 3.7%, and path patching localizes the strongest break effect to the early residual stream. Cross-architecture evidence is mixed, with contraction in locked Mistral and Qwen rows, sign reversal in OPT and GPT-2 XL, and a null power-up result in Pythia-1.4B.

## Keywords

1. Mechanistic interpretability
2. Self-referential processing
3. Representation geometry
4. Participation ratio
5. Activation analysis
6. Transformer internals

## Primary Subject Area

Interpretability and Analysis of Language Models

## Secondary Subject Area

Safety and Alignment

## Supplementary Materials Checklist

- [ ] Upload paper PDF (paper_colm2026_v007.pdf)
- [ ] Upload code as zip (geometric_lens/ + scripts/p0_canonical_pipeline.py)
- [ ] Confirm all authors have OpenReview profiles
- [ ] Verify anonymous submission (no author names in PDF)
- [ ] Check PDF page count (9pp main + refs + appendix)

---

## Notes for Dhyana

1. **TL;DR vs Abstract**: The TL;DR file has slightly different numbers (56.0% vs 44.7%). The paper abstract (v007) is the authoritative version. Consider updating the TL;DR to match v007.

2. **Anonymous submission**: Paper already says "Anonymous authors / Paper under double-blind review". Good.

3. **Preprint policy**: COLM allows non-anonymous arXiv preprints. Could post to arXiv before/during review.

4. **Editable until paper deadline**: Abstract can be refined after initial submission.

5. **Base Mistral-7B-v0.1 run**: Still needed for strongest universality claims. The paper is honest about this ("Additional base canonical runs are still required"). This doesn't block submission but would strengthen the paper if completed before Apr 1.
