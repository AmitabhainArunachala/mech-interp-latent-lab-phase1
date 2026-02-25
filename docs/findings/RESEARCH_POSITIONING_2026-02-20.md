# Research Positioning (2026-02-20)

## One-Line Position
This project contributes a methodology result for mechanistic interpretability: in GQA architectures, head-intervention conclusions are sensitive to headspace semantics, and incorrect semantics can invert causal-direction claims.

## How It Connects To Existing Work

1. Circuit-level causal intervention tradition
- Transformer Circuits established composition-level circuit analysis and head-level mechanism tracing: https://arxiv.org/abs/2209.11895
- IOI circuit work operationalized causal intervention/patching as core evidence for mechanisms: https://arxiv.org/abs/2211.00593

Connection here:
- Your bridge experiments are in this exact lineage (intervene, compare controls, quantify effect sizes), but add an explicit implementation-validity axis: intervention semantics must match attention architecture.

2. Architecture semantics (GQA) as a hidden confound
- GQA formalizes grouped query/key-value structure: https://arxiv.org/abs/2305.13245
- Mistral-7B explicitly adopts GQA: https://arxiv.org/abs/2310.06825

Connection here:
- Your `v4_gqa_headspace` correction targets this architectural fact directly. The observed sign inversion between old and corrected random-head controls fits the prediction that mismatched head semantics can produce misleading causal effects.

3. Known caution that localization and intervention can diverge
- “Does Localization Inform Editing?” highlights that localization evidence does not automatically transfer to robust intervention/editing behavior: https://arxiv.org/abs/2301.04213

Connection here:
- Your result can be read as a concrete failure mode of that gap: even with plausible localized targets, implementation choices can alter causal conclusions.

4. Current methodological consolidation and scaling
- Practical review of attribution/path patching methods and caveats: https://arxiv.org/abs/2407.02646
- AtP* (2025) focuses on scaling path patching to larger models via approximation: https://arxiv.org/abs/2511.05442

Connection here:
- Your contribution is complementary: before scaling patching further, intervention semantics must be architecture-correct. This is an upstream validity constraint for future large-scale patching pipelines.

## Current Empirical Anchor In This Repo
- Fast GPU bridge contrasts (Mistral-7B):
  - head-specific vs random-head: mean diff `-0.051096`, `p=4.31e-4`, `d=-2.366`
  - head-specific vs baseline-donor: mean diff `-0.057376`, `p=1.67e-2`, `d=-1.429`
  - random-head vs baseline-donor: `p=0.750` (no meaningful separation)
- Source: `results/remote_gpu_sync/2026-02-20/phase1_mechanism/contrast_stats.md`

Interpretation:
- The mechanism-specific intervention separates sharply from both controls, while controls do not separate from each other, consistent with a specificity claim rather than generic perturbation.

Additional status:
- Multi-token bridge rerun completed (`20260220_071457...`) with strong recursive-vs-baseline `R_V` separation but still high truncation (88.9% at `T=0.0`, 69.4% at `T=0.7`), so behavior-correlation claims remain provisional.

## Candidate NeurIPS-Grade Claim
"Causal patching claims in grouped-query transformers are not architecture-invariant: headspace-misaligned interventions can flip estimated effect direction, while GQA-aligned interventions recover control-separated specificity."

## What Would Upgrade This To Strong Submission Evidence
1. Multi-seed replication with preregistered thresholds on the two key contrasts.
2. Transfer test on one non-GQA dense attention model (expected attenuation of headspace effect).
3. Independent implementation cross-check (second codepath reproducing sign/direction).
4. Low-truncation behavioral bridge confirmation so geometry-behavior links are not generation-length artifacts.
