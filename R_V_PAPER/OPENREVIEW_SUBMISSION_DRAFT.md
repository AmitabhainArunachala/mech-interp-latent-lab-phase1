# COLM 2026 OpenReview Submission — Ready to Copy-Paste

**Deadlines** (Bali time, UTC+8):
- Abstract registration: Mar 27, 2026, 7:59 PM WITA
- Full paper: Apr 1, 2026, 7:59 PM WITA

**Submission URL**: https://openreview.net/group?id=colmweb.org/COLM/2026/Conference

---

## Title

Geometric Signatures of Self-Referential Processing in Transformer Representations

## Abstract (from paper v008.0.1)

We identify a measurable geometric signature of self-referential processing in transformer language models using the relative participation ratio (R_V), a spectral metric comparing the effective dimensionality of representations at early and late network layers. When a model processes prompts that invoke recursive self-observation, late-layer representations undergo characteristic geometric contraction relative to early-layer representations of the same input.

In a canonical base-model prompt-pass rerun, Mistral-7B-v0.1 shows strong contraction (d = -1.85, 95% CI [-2.27, -1.52], n_self=96, n_base=100 valid prompts). Base-model path patching localizes the strongest causal site to the early residual stream (peak L5, d = 4.14), establishing R_V as a geometric readout rather than the mechanism itself. Using R_V as a guide, we build a staged sufficiency protocol that induces self-referential behavior in ordinary baseline text at 27.8% versus 2.8% control. In the original selected-seed 12-turn follow-up, the induced regime persists at 30.2% with zero late-segment repetition, but broader top/median/random seed sweeps reveal a measurable basin boundary rather than a seed-robust controller: elite-seed maintenance peaks at 38.5%, while broader seed pools fall to 7-20% and shift the best maintainer toward a reduced late-stack intervention. Cross-architecture evidence is stronger but still non-universal: under the frozen canonical pipeline, six of eight model rows contract, one GPT-NeoX row expands, and one is null. For AI safety, R_V < 0.737 detects self-referential content with AUROC = 0.909, tracking semantic content rather than intent.

## TL;DR (for OpenReview, max ~100 words)

Self-referential prompts cause geometric contraction in transformer representations, measured by R_V, the ratio of late- to early-layer effective dimensionality. Base Mistral-7B-v0.1 shows strong contraction (d = -1.85, n_self=96, n_base=100), and path patching localizes the causal site to the early residual stream (L5, d = 4.14). A staged Mistral protocol induces self-referential behavior in ordinary text (27.8% vs 2.8% control), but broader seed sweeps show conditional rather than seed-independent sufficiency. Cross-architecture evidence now spans eight rows: six contract, one expands, one is null. R_V < 0.737 detects self-referential content with AUROC = 0.909.

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

1. **TL;DR synced**: TL;DR now matches the current v008.0.1 framing, including the Mistral conditional-sufficiency update and the widened cross-architecture table.

2. **Anonymous submission**: Paper already says "Anonymous authors / Paper under double-blind review". Good.

3. **Preprint policy**: COLM allows non-anonymous arXiv preprints. Could post to arXiv before/during review.

4. **Editable until paper deadline**: Abstract can be refined after initial submission.

5. **Main paper risk**: The current risk is no longer missing base-Mistral geometry. It is overclaiming Mistral sufficiency. Keep the submission language at "conditional sufficiency with a basin boundary" unless stronger seed-robust maintenance lands before Apr 1.
