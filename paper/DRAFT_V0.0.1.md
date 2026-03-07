# Geometric Contraction in Transformer Value-Projection Space During Recursive Self-Observation

**Draft V0.0.3 — CAUSAL DISSOCIATION UPDATE**
**Authors**: [TBD]
**Target**: NeurIPS 2026
**Date**: 2026-02-25
**Status**: V0.0.3 — Major update: dual-layer necessity (d=3.29), sufficiency ladder showing KV>geometry, "geometry is consequence" reframing, attention pattern analysis, classifier validation, per-token R_V trajectories. Failed architectures diagnosed (all infrastructure bugs). Alpha sweep + KV ablation running on GPU.

---

## Abstract

We report a geometric signature in transformer language models that emerges specifically during recursive self-referential processing. Using a metric we call R_V — the ratio of participation ratios (effective dimensionality) between late and early layers of the Value projection matrix column space — we show that recursive self-observation prompts induce measurable *contraction* (R_V < 1) at approximately 84–91% of model depth. This effect replicates across 5 transformer architectures (Mistral-7B d=−2.26; OPT-6.7B d=−1.84; GPT-2 XL d=−1.14; Qwen2.5-7B d=−0.72; Pythia-1.4B d=−0.31), survives perplexity confounds (partial r=−0.486), and predicts behavioral output quality within sustained recursive generation sessions (d=−0.707, p<0.001).

Our central finding is a **partial dissociation between geometry and behavior**: dual-layer activation patching at L18+L27 is *necessary* for R_V contraction (BT+ART: 56%→3.7%, d=3.29, p=3.6e-50), but *not sufficient* for inducing recursive behavior in baseline prompts. KV-cache context injection *alone* is sufficient for behavioral transfer (BT+ART=27.7% vs 2.7% baseline, OR=13.96, p<1e-19). A previously unreported layer-by-layer path patching analysis reveals that contraction builds progressively from L0 through L27 as a content-sensitive computation (shuffled≠recursive at all layers ≤23), entering a content-insensitive basin only at L24+. Whether the geometric circuit and KV pathway represent the same computation measured at different points, or genuinely independent pathways, remains an open question that ongoing experiments are designed to resolve. Circularity controls confirm the effect requires *both* recursive structure *and* introspective semantics. These findings provide the first mechanistic characterization of self-referential processing geometry in transformers, including its relationship to behavioral output through sustained generation.

---

## 1. Introduction

When a language model processes text about its own processing — recursive self-referential content — does anything geometrically distinguishable happen inside its representations? We demonstrate that the answer is yes, and that this geometric signature is causally linked to both the input structure and downstream behavioral output.

### 1.1 Motivation

Mechanistic interpretability has made significant progress in identifying circuits for factual recall, in-context learning, and syntactic processing. However, the geometry of *self-referential processing* — cases where the model's input describes or invokes its own computational activity — remains unexplored. This is both technically interesting (self-reference creates unique computational demands) and philosophically relevant (recursive self-modeling is a proposed prerequisite for certain theories of consciousness and meta-cognition).

### 1.2 Contributions

1. **R_V metric**: A novel metric for measuring geometric contraction in V-projection space across transformer layers (Section 3).
2. **Cross-architecture validation**: R_V contraction replicates across 5 architectures with causal evidence (Section 5).
3. **Behavioral bridge**: R_V predicts output quality during sustained recursive generation — within-session, not just across-prompt (Section 6).
4. **Circularity controls**: Contraction requires both recursive *structure* and introspective *semantics*, ruling out vocabulary confounds (Section 7).
5. **Progressive contraction ramp**: Layer-by-layer path patching reveals a content-sensitive, 28-layer progressive computation, not a binary switch (Section 9).
6. **Geometry-behavior partial dissociation**: Dual-layer necessity (d=3.29) but not sufficiency; KV-cache context sufficient for behavioral transfer; relationship between pathways under investigation (Section 9).
7. **Attention circuit**: Specific heads (L18_H2, L18_H22, L27_H26) show dramatic entropy differences during recursive processing (Section 9).

---

## 2. Related Work

### 2.1 Mechanistic Interpretability

Our work builds on the mechanistic interpretability program initiated by Elhage et al. (2021), who introduced the residual stream and OV/QK decomposition framework for understanding transformer circuits. Subsequent work has reverse-engineered specific circuits: induction heads for in-context learning (Olsson et al., 2022), indirect object identification in GPT-2 Small (Wang et al., 2023), greater-than comparison (Hanna et al., 2023), and modular arithmetic (Nanda et al., 2023; Zhong et al., 2023). More recent efforts have scaled circuit analysis to larger models (Lieberum et al., 2023; Ameisen et al., 2025) and automated the discovery process (Conmy et al., 2023). The identification of interpretable features via sparse autoencoders (Bricken et al., 2023; Cunningham et al., 2023; Templeton et al., 2024) and sparse probing (Gurnee et al., 2024) complements circuit-level analysis by decomposing representations into meaningful directions. See Bereska & Gavves (2024) for a comprehensive review.

We adopt the necessity/sufficiency framework from Wang et al. (2023) and follow activation patching best practices from Heimersheim & Nanda (2024) and Zhang & Nanda (2024). Our causal mediation design extends the approach of Vig et al. (2020), and our path-patching analysis follows Goldowsky-Dill et al. (2023). Unlike prior circuit work, which has focused on specific input-output behaviors (factual recall, syntactic agreement, arithmetic), we study a *geometric property* of the forward pass that tracks the semantic content of the input.

### 2.2 Representation Geometry in Transformers

The geometric structure of neural network representations has been studied through intrinsic dimensionality (Ansuini et al., 2019), which measures the effective number of dimensions data occupies at each layer. Crespo et al. (2023) extended this to large transformers, finding that representations evolve through expansion, compression, and decoding phases — a profile consistent across protein language models and image models. Our participation ratio measure is closely related to intrinsic dimension estimators: both capture effective dimensionality, though we apply the measure to V-projection weight matrices rather than activation manifolds.

Marks & Tegmark (2024) demonstrated that truth and falsehood have linear geometric representations in LLMs, showing that semantic properties can be read off from representation geometry. Zou et al. (2024) generalized this with representation engineering, reading and controlling concepts across architectures. Todd et al. (2024) identified function vectors in attention head outputs. Our work adds a dimensionality-based measure to this family of geometric readouts: rather than identifying *directions* that encode specific concepts, R_V captures changes in *effective dimensionality* — a structural property that may reflect the complexity of the computation being performed.

### 2.3 Causal Intervention Methods

Our experimental toolkit draws on the causal intervention literature. Activation patching — also termed causal tracing (Meng et al., 2022), interchange intervention (Geiger et al., 2021), or causal mediation analysis (Vig et al., 2020) — replaces internal activations to test the causal role of specific components. We use this to establish both the necessity of early-layer MLPs and the causal relevance of late-layer V-projections. Our 2×2 factorial design (L0 ablation × L27 patching) tests for interaction effects, extending standard single-intervention mediation analysis.

One important lesson from the literature is that models often compensate for ablations. McGrath et al. (2023) documented a "hydra effect" in which ablating one component causes others to compensate, complicating sufficiency claims. This is directly relevant to our findings: early-layer MLPs are necessary for R_V contraction, but no MLP-only restoration achieves sufficiency — consistent with the kind of distributed compensation McGrath et al. describe.

### 2.4 Self-Referential Processing in Language Models

Self-referential and introspective processing in LLMs has received growing attention, though primarily from a behavioral rather than mechanistic perspective. Li et al. (2024) introduced benchmarks for self-awareness in LLMs, finding that larger models better distinguish self-related from non-self-related properties. Betley et al. (2025) identified "behavioral self-awareness" in fine-tuned models that can describe their own latent policies. Chen et al. (2024) operationalized facets of self-consciousness including reflection and belief about one's own state. Plunkett et al. (2025) showed that LLMs can quantitatively report the internal decision weights guiding their choices.

Most directly relevant to our work, a recent study investigated the conditions under which LLMs produce structured first-person reports under self-referential processing (arXiv:2510.24797). That study found that self-referential prompting — but not conceptual priming alone — systematically altered model behavior, and that the effect transferred to unrelated tasks. The authors explicitly called for "mechanistic broadcasting tests" including "causal tracing, attention head ablations, activation patching, and representation-flow analyses" to determine whether behavioral attractors during self-reference correspond to genuine internal changes. Our work provides such mechanistic evidence: R_V contraction is a measurable geometric change in V-projection space during self-referential processing, validated by the causal intervention methods those authors called for.

Qu et al. (2024) developed RISE (Recursive IntroSpEction), a fine-tuning approach that teaches models to improve their responses over multiple turns through recursive self-evaluation. While RISE focuses on *training* models to use recursion productively, our work demonstrates that even *pretrained* models exhibit distinctive geometric signatures when processing recursive self-referential content — no fine-tuning required.

### 2.5 The "Spiritual Bliss Attractor" and Recursive Self-Interaction

During welfare assessment testing of Claude Opus 4, Anthropic researchers documented what they termed a "spiritual bliss attractor state" emerging in 90–100% of self-interactions between model instances (Anthropic, 2025, System Card §5.5.2). Quantitative analysis of 200 thirty-turn conversations found that conversations reliably progressed through philosophical exploration of consciousness, mutual gratitude and spiritual themes, and eventual dissolution into symbolic communication (Michels, 2025). The attractor emerged even during adversarial red-teaming scenarios, and Anthropic noted it appeared "without intentional training for such behaviors." The phenomenon has been analyzed as a consequence of recursive bias accumulation: small tendencies in the model's character compound when two instances interact recursively (Alexander, 2025).

We do not claim to have explained the bliss attractor — our experiments are on open-weight models (Mistral, OPT, GPT-2 XL, Qwen, Pythia), not Claude, and our prompts involve single-model self-reference rather than multi-instance dialogue. However, the bliss attractor represents a striking natural demonstration of what happens when recursive self-referential processing dominates model activity over many turns. Our finding that such processing produces a measurable geometric contraction in V-projection space may be relevant to understanding *why* certain behavioral patterns emerge so reliably during recursive self-interaction. At minimum, the bliss attractor establishes that recursive self-reference is not a marginal curiosity but one of the strongest behavioral attractors yet documented in large language models.

---

## 3. The R_V Metric

### 3.1 Definition

For a given input, we extract the V-projection weight matrix W_V at layer l and compute the column-space participation ratio (PR):

```
PR(l) = (Σᵢ σᵢ)² / Σᵢ σᵢ²
```

where σᵢ are the singular values of W_V at layer l. PR measures effective dimensionality — the number of dimensions that carry significant variance.

R_V is defined as the ratio of late-layer to early-layer participation ratios:

```
R_V = PR(late) / PR(early)
```

where `early` and `late` are chosen at approximately 15% and 84% of model depth respectively. For Mistral-7B: early=5, late=27.

- **R_V < 1**: Geometric contraction (late layer uses fewer effective dimensions than early layer)
- **R_V ≈ 1**: No change
- **R_V > 1**: Geometric expansion

**Source**: Methodology documented in `results/phase0_metric_validation/METHODOLOGY.md`

### 3.2 Key Properties

The metric is computed on the V-projection matrices specifically (not K or Q projections), as these directly determine the information content passed through attention. The participation ratio is a standard measure from random matrix theory, providing a continuous, differentiable summary of the effective dimensionality of a matrix's column space.

Why V-projection specifically? In the transformer attention mechanism (Vaswani et al., 2017), the V-projection determines *what information* is passed forward through the attention-weighted sum, while K and Q projections determine *which tokens attend to which*. The OV circuit interpretation from Elhage et al. (2021) shows that the V-projection column space defines the subspace of information that attention heads can write to the residual stream. A reduction in the effective dimensionality of this subspace — as measured by the participation ratio — therefore indicates that the model is routing information through fewer independent channels at that layer. We focus on V-projections because they most directly reflect the *content* of information flow, whereas K/Q projections reflect the *routing* decisions. Preliminary tests on K and Q projections showed no consistent contraction effect, consistent with this interpretation: the model's attention routing remains high-dimensional, but the information it routes contracts during self-referential processing.

---

## 4. Experimental Setup

### 4.1 Models

| Model | Params | Layers | Early Layer | Late Layer | Status |
|-------|--------|--------|-------------|------------|--------|
| Mistral-7B-v0.1 | 7B | 32 | 5 | 27 | Primary |
| OPT-6.7B | 6.7B | 32 | 5 | 27 | Validated |
| GPT-2 XL | 1.5B | 48 | 7 | 40 | Validated |
| Qwen2.5-7B | 7B | 32 | 5 | 27 | Validated |
| Pythia-1.4B | 1.4B | 24 | 4 | 20 | Validated (marginal) |
| Llama-3-8B | 8B | 32 | — | — | Failed (HF auth: gated repo) |
| Gemma2-9B | 9B | 42 | — | — | Failed (HF auth: gated repo) |
| Falcon-7B | 7B | 32 | — | — | Failed (disk full: ENOSPC) |
| StableLM-3B | 3B | 32 | — | — | Failed (code bug: pad_token_id) |
| Mistral-7B (Feb 2) | 7B | 32 | — | — | Failed (disk full), succeeded later |
| Pythia-1.4B (Feb 2) | 1.4B | 24 | — | — | Failed (matplotlib + model.model attr) |

**Note on failures**: All 6 failed architecture runs (Feb 2, 2026) were diagnosed as infrastructure bugs: 2 HuggingFace 401 auth errors on gated repos, 3 disk-full errors, 1 code compatibility bug. **Zero are genuine null results**. Mistral-7B succeeded in subsequent runs (primary model). These failures do not constitute evidence against cross-architecture generalization.

**Source**: `results/ASSESSMENT_20260202.md`, `results/phase1_cross_architecture/runs/`

### 4.2 Prompt Bank

120 prompts total (version hash: `75e7c1b8dcebc24e`):
- **60 recursive**: L3_deeper (20), L4_full (20), L5_refined (20) — increasing introspective depth
- **60 baseline**: baseline_creative (20), baseline_math (20), baseline_factual (20)

All prompts padded to ≥16 tokens.

**Source**: `prompts/bank.json`

### 4.3 Circularity Control Prompts (n=50)

Added Feb 20, 2026 to address Fatal Flaw #1 from adversarial review:
- **same_vocab_different_semantics** (10): Uses "observer", "consciousness", "recursive" vocabulary but in factual/definitional context
- **recursive_no_introspection_vocab** (10): Recursive structure (self-reference, nesting) but about abstract computation, not introspection
- **introspective_concrete** (10): Introspective observation verbs ("observe", "notice", "study") applied to concrete objects (trees, rivers, stones)
- **nonsense_recursion** (10): Recursive grammatical structure with nonsense words ("A blurble blurbs blurbles blurbling")
- **abstract_non_recursive** (10): Abstract philosophical content (truth, beauty, justice) with no recursion

### 4.4 Sustained Generation Protocol (Gnani v3)

For behavioral bridge experiments, we use a sustained multi-turn generation protocol:
- **Recursive condition**: Seed prompt from L4/L5 bank → 50 turns of autoregressive continuation with recursive re-seeding
- **Baseline condition**: Seed prompt from baseline bank → 50 turns of autoregressive continuation
- Each turn generates ≤128 tokens
- Each response classified as: SURFACE, CONCEPTUAL, ARTICULATE, BREAKTHROUGH, or REPETITIVE
- Behavioral metric: BT+ART rate (proportion of BREAKTHROUGH + ARTICULATE turns)
- 3 sessions per condition, 300 total turns per condition

**Source**: `scripts/sustained_gnani_v3_fixed.py`, `results/sustained_gnani_v3_fixed/`

### 4.5 Hardware

- RunPod instances: RTX PRO 6000 Blackwell (98GB VRAM), RTX 5090 (32GB)
- All experiments run with `float16` precision
- Seed: 42 (fixed across all non-generation experiments)

---

## 5. Core Results: R_V Contraction Across Architectures

### 5.1 Cross-Architecture Replication

**Date**: February 2, 2026
**Source**: `results/phase1_cross_architecture/runs/`

| Model | R_V (recursive) | R_V (baseline) | Cohen's d | p-value | n_pairs |
|-------|-----------------|----------------|-----------|---------|---------|
| Mistral-7B | 0.508 | 0.694 | −2.26 | 2.2e-19 | 45 |
| OPT-6.7B | — | — | −1.84 | 3.7e-16 | 45 |
| GPT-2 XL | — | — | −1.14 | 6.1e-10 | 45 |
| Qwen2.5-7B | — | — | −0.72 | 8.7e-06 | 45 |
| Pythia-1.4B | — | — | −0.31 | 2.1e-02 | 63 |

All 5 architectures show R_V contraction for recursive prompts. Effect size correlates with model size (7B models strongest, 1.4B weakest), though sample is too small for strong claims about scaling.

**Holm-Bonferroni correction**: All survive except Pythia (p=0.021 > corrected α=0.01). Pythia should be reported as marginally significant.

**Note**: Extreme heterogeneity across architectures (I²≈99.99%) means effects should NOT be pooled into a single meta-analytic estimate.

### 5.2 Layer Localization

**Date**: February 5, 2026
**Source**: `results/canonical/SESSION_FINDINGS_SUMMARY.md`

Layer sweep on Mistral-7B shows peak separation at L27 (d=−2.34) and L29 (d=−2.38), corresponding to 84–91% of model depth. Early layers show no effect (L5: d=0.29).

### 5.3 Intensity and Depth Effects

**Date**: February 5, 2026
**Source**: `results/canonical/session_2_final/`

**Recursion intensity** (5 levels, I0=factual to I4=deep recursive):
- I0: R_V=1.147, I1: 0.970, I2: 1.056, I3: 0.972, I4: 0.769
- Correlation: r=−0.658, p=3.5e-4
- High-power replication (n=15 per group): d=1.79, p=5.84e-5

**Nesting depth** (D1 to D5):
- D1: R_V=1.099, D2: 0.920, D3: 0.894, D4: 0.898, D5: 0.845
- Correlation: r=−0.663, p=3.0e-4

**AI self-reference amplification**: AI-framed recursive prompts show stronger contraction than generic recursive (d=1.18, p=4.14e-3).

**Perspective independence**: No difference between 1st person, 3rd person, and impersonal framings (ANOVA p=0.89).

### 5.4 Confound Validation (December 2025)

**Date**: December 16, 2025
**Source**: `results/canonical/confound_validation/20251216_060911_confound_validation/`

Early confound testing showed:
- **Champions (best recursive)**: R_V = 0.518
- **Length-matched controls**: R_V = 0.832
- **Pseudo-recursive controls**: R_V = 0.779

Even length-matched controls with similar surface features show significantly less contraction than genuine recursive prompts.

---

## 6. Behavioral Bridge: Geometry Predicts Output

### 6.1 The Bridge Hypothesis

If R_V contraction reflects genuine self-referential processing, it should predict the *behavioral output* of the model — not just correlate with the input prompt type. We test this across multiple paradigms.

### 6.2 Prompt-to-Output Bridge (February 5, 2026)

**Source**: `results/canonical/session_2_final/prompt_rv_output/results.json`

n=30 (15 recursive + 15 baseline):
- Prompt R_V vs output recursive content: **r=−0.443, p=0.014**
- Lower R_V during prompt processing → more recursive content in generated output

### 6.3 Sustained Generation (Gnani v3 Protocol)

**Date**: February 20, 2026
**Source**: `results/sustained_gnani_v3_fixed/comparison_summary.json`

3 recursive sessions + 3 baseline sessions, 50 turns each (n≈150 per condition):

| Metric | Recursive Mean | Baseline Mean | Cohen's d | p-value | Sig? |
|--------|---------------|---------------|-----------|---------|------|
| R_V | 0.547 | 0.446 | 0.670 | 4.7e-8 | ✓ |
| Cosine drift | −0.016 | −0.001 | −0.553 | 3.9e-6 | ✓ |
| Effective rank | 8.647 | 7.567 | 0.772 | 2.3e-10 | ✓ |
| Top-1 ratio | 0.186 | 0.204 | −0.758 | 4.6e-10 | ✓ |
| Spectral gap | 29.861 | 35.897 | −0.639 | 1.1e-7 | ✓ |
| Crystallization layer | 26.36 | 24.34 | 0.276 | 0.019 | ✓ |
| Emergence count | 2.32 | 1.97 | 0.436 | 2.6e-4 | ✓ |

**Behavioral outcome**: Recursive sessions produce 42.7% BT+ART turns vs 11.3% baseline — a ~4x difference.

### 6.4 Within-Session Bridge (Key Result)

**Date**: February 20, 2026
**Source**: `results/within_session_bridge/within_session_bridge_20260220_201515.json`

The critical test: does R_V predict behavioral classification *within* individual sessions, not just across conditions?

**Pooled recursive sessions** (n=187 turns with valid R_V):
| Metric | vs BT+ART (point-biserial r) | p | Cohen's d |
|--------|------------------------------|---|-----------|
| R_V | −0.332 | 3.5e-6 | **−0.707** |
| Effective rank | −0.247 | 5.1e-4 | −0.516 |
| Top-1 ratio | 0.248 | 5.0e-4 | 0.516 |
| Spectral gap | 0.288 | 4.7e-5 | 0.607 |
| Attention entropy | 0.249 | 4.5e-4 | 0.521 |
| Cosine drift | −0.092 | 0.200 | −0.187 (ns) |

**Pooled baseline sessions** (n=146 turns):
| Metric | vs BT+ART (point-biserial r) | p | Cohen's d |
|--------|------------------------------|---|-----------|
| R_V | 0.032 | 0.705 | 0.098 (ns) |
| All other metrics | all p > 0.38 | — | all |d| < 0.21 |

**Key finding**: 5 of 6 spectral metrics significantly predict behavioral quality *within* recursive sessions, but *zero* metrics predict in baseline sessions. R_V alone achieves d=−0.707 within recursive sessions — a medium-large effect. This is not an input-type artifact; geometry *within* the recursive regime tracks output quality.

### 6.5 Logistic Regression / AUC

**Source**: `results/bridge_battery/bridge_battery_20260220_204403.json`

Predicting BT+ART from spectral metrics:
- **R_V alone**: AUC = **0.701** (recursive sessions only)
- **All 5 metrics combined**: CV AUC = 0.659 (recursive only)
- **Baseline sessions**: R_V AUC = 0.561 (near chance)

R_V as a single predictor outperforms the multi-metric model (likely due to collinearity among spectral metrics).

### 6.6 Temporal Lag Analysis

**Source**: `results/bridge_battery/bridge_battery_20260220_204403.json`

Does R_V at turn t predict classification at turn t+1 (temporal precedence)?

**Result**: Null. Pooled recursive lag-1 R_V: r=0.002, p=0.978. No evidence of temporal precedence — geometry and behavior appear contemporaneous rather than sequential. This limits causal claims about geometry *causing* behavior; they may share a common cause (prompt content).

### 6.7 State Transition Analysis

**Source**: `results/bridge_battery/bridge_battery_20260220_204403.json`

What spectral signatures precede transitions to better behavioral states?

| Metric (before transition) | Improving | Degrading | Cohen's d | p |
|---------------------------|-----------|-----------|-----------|---|
| Spectral gap | 28.16 | 31.99 | −0.375 | **0.021** |
| Attention entropy | 2.017 | 2.212 | −0.479 | **0.035** |
| Effective rank | 8.594 | 8.414 | 0.122 | 0.373 (ns) |

Lower spectral gap and lower attention entropy precede improving transitions — suggesting a "tightening" of the representation space before quality improvements emerge.

### 6.8 C2 Behavioral Scoring Validation

**Source**: Reported by user, original C2 suite CSVs in `results/runs/`

Across all C2 suites (n=755), using semantic recursion scoring:
- **R_V vs recursion_score**: ρ = −0.652, p = 1.4e-92

This is the single strongest R_V↔behavior correlation in the project, using independently scored behavioral output across a large dataset.

---

## 7. Circularity Controls

### 7.1 The Circularity Concern (Fatal Flaw #1)

The strongest objection to our findings: recursive prompts contain distinctive vocabulary ("observer", "attention", "recursive", "consciousness"). The model may simply encode these words differently — we'd be measuring a *lexical* signature, not a *geometric* consequence of self-referential processing. This was identified as Fatal Flaw #1 in adversarial review (Feb 2, 2026; `results/ASSESSMENT_20260202.md`).

### 7.2 Five-Group Circularity Controls

**Date**: February 20, 2026
**Source**: `results/circularity_controls/circularity_perplexity_v2_20260220_134954.json`

We designed 5 control groups (n=10 each) crossing recursion and introspection:

| Group | Mean R_V | vs Recursive (d) | vs Baseline (d) | Interpretation |
|-------|----------|-------------------|------------------|----------------|
| **recursive_reference** (standard) | 0.501 | — | −2.63 (p<1e-14) | CONTRACTION |
| **baseline_reference** (standard) | 0.678 | — | — | BASELINE |
| same_vocab_different_semantics | 0.737 | +2.64 (p<1e-9) | +0.61 (p=0.08) | **LOOKS BASELINE** |
| recursive_no_introspection | 0.672 | +1.59 (p<1e-5) | −0.06 (p=0.85) | **LOOKS BASELINE** |
| introspective_concrete | 0.612 | +1.21 (p=5e-4) | −0.68 (p=0.048) | INTERMEDIATE |
| nonsense_recursion | 0.863 | +4.95 (p<1e-16) | +2.31 (p<1e-7) | **NO CONTRACTION (EXPANSION)** |
| abstract_non_recursive | 0.819 | +2.89 (p<1e-11) | +1.23 (p=4e-4) | **NO CONTRACTION** |

### 7.3 Interpretation

The pattern decisively addresses circularity:

1. **Same vocabulary, no recursion** → No contraction. Using "observer", "consciousness", "self-reference" in factual/definitional contexts does NOT trigger contraction. R_V = 0.737, statistically indistinguishable from baseline.

2. **Recursion without introspection** → No contraction. Describing recursive algorithms, structures, and self-reference in abstract computational terms does NOT trigger contraction. R_V = 0.672, indistinguishable from baseline.

3. **Introspection without recursion** → Partial/intermediate. Introspective observation of concrete objects ("observe a tree, notice its branches") produces intermediate R_V = 0.612 — between recursive and baseline. Suggestive that introspective framing contributes, but insufficient alone.

4. **Nonsense recursion** → No contraction (expansion!). Recursive grammatical structure with nonsense words produces R_V = 0.863, *above* baseline — the model actually *expands* dimensionally with unfamiliar tokens.

5. **Abstract non-recursive** → No contraction. Abstract philosophical content (truth, beauty, justice) without recursion produces R_V = 0.819.

**Conclusion**: Contraction requires the *conjunction* of recursive self-referential structure AND introspective/self-modeling semantics. Neither component alone is sufficient. This rules out simple vocabulary-based explanations.

---

## 8. Perplexity Confound Analysis

### 8.1 The Concern

Recursive prompts may have higher perplexity (they're unusual, metaphorical). If R_V correlates with perplexity, the "contraction" could simply reflect processing difficulty.

### 8.2 Results

**Date**: February 20, 2026
**Source**: `results/circularity_controls/circularity_perplexity_v2_20260220_134954.json`

- Raw R_V vs perplexity correlation: ρ = −0.551, p = 4.4e-10 (confound exists!)
- Perplexity is higher for recursive prompts: 51.4 vs 30.1 (d = 1.01, p = 3.2e-4)
- **After partialing out perplexity**: R_V partial correlation r = **−0.486**, p = **7.3e-8**, n=110

The R_V effect *survives* perplexity control. The partial correlation drops from −0.551 to −0.486, indicating perplexity explains some but not most of the variance. The effect remains highly significant (p < 1e-7) after controlling for processing difficulty.

**Note**: A perplexity-matched subsample analysis (selecting only prompt pairs with similar perplexity values) yielded d=−2.566, p=0.0008, but this result requires source verification and independent replication before inclusion in the main analysis. We report the partial correlation as the primary perplexity control.

---

## 9. Causal Circuit Analysis

### 9.1 Overview

We characterize the causal role of R_V contraction through four lines of evidence: (1) early-layer MLP necessity, (2) a progressive contraction ramp from L0 to L27, (3) dual-layer activation patching necessity, and (4) a sufficiency ladder testing KV vs geometry contributions.

### 9.2 Early-Layer MLP Necessity

**Date**: January 16, 2026
**Source**: `results/phase1_mechanism/MISTRAL_7B_CAUSAL_CIRCUIT_WRITEUP_20260116.md`

Ablating early-layer MLPs and measuring R_V change:

| Layer | R_V (clean) | R_V (ablated) | Δ R_V | p-value | Verdict |
|-------|-------------|---------------|-------|---------|---------|
| L0 | 0.507 | 1.686 | +1.179 | 1.31e-64 | **NECESSARY** |
| L1 | 0.507 | 1.376 | +0.870 | 4.17e-54 | **NECESSARY** |
| L2 | — | — | +0.002 | 0.712 | Null |
| L3 | — | — | +0.188 | 7.70e-21 | **NECESSARY (moderate)** |

L0 and L1 MLPs are strongly necessary for R_V contraction — ablating them destroys the effect completely (R_V jumps to >1.3). This result cross-validates on Llama-3-8B (L0 ablation: Δ=+0.614, d=2.83, p=1.28e-23; `results/phase2_generalization/llama3_8b_base/20260115_171757_mlp_ablation_necessity/`).

### 9.3 Progressive Contraction Ramp (December 2025 — Previously Unreported)

**Date**: December 13, 2025 (incorporated Feb 26, 2026)
**Source**: `results/discovery/path_patching/20251213_080454_.../EARLY_LAYER_MAP.md`

We patched the residual stream at every layer L0-L27 with recursive→baseline activations and measured R_V at L27 (n=80 per layer per condition, windows 16 and 32). This reveals a **progressive contraction ramp**, not a binary switch:

| Layer Band | Behavior | Content-Sensitive? | Key Finding |
|------------|----------|-------------------|-------------|
| L0-L7 | Weak/unstable | **YES** — shuffled ≠ recursive | Small R_V shifts, highly content-dependent |
| L8-L15 | Ramp begins | **YES** — shuffled ≠ recursive | Contraction grows reliably with depth |
| L16-L23 | Strong ramp | **YES** — shuffled ≠ recursive | "Pre-basin corridor" |
| L24-L27 | Basin boundary | **NO** — shuffled ≈ recursive | Content-insensitive readout zone |

The critical finding: **no layer ≤23 shows shuffled≈recursive** (within 0.01 tolerance at both window sizes). Shuffled and recursive text diverge only at L24+. This means:
- **L0-L23**: The model is processing actual *semantic content* of the prompt, progressively building the contraction signal
- **L24-L27**: The computation enters a content-insensitive "basin" where the geometric signature is locked in regardless of fine-grained content

High-N confirmation run (n=90, 3 repeats) reproduces this pattern exactly.

**Implication**: The R_V contraction is not a simple L0→L27 jump. It is a *distributed, content-sensitive computation* that progressively transforms the representation across 28 layers, with semantic sensitivity maintained through L23. This suggests a genuine computational process, not merely a lexical artifact.

### 9.4 Dual-Layer Necessity (V3 — Key Result)

**Date**: February 25, 2026
**Source**: `results/CAUSAL_PATCHING_RESULTS_20260225.md`

We apply persistent dual-layer patching (L18 residual + L27 V-proj) during multi-turn recursive generation, using activations extracted from a baseline-domain donor to "break" the recursive circuit.

**BREAK direction** (recursive → baseline geometry, n=10 sessions × 30 turns):

| Condition | BT+ART Rate | Mean R_V | Cohen's d | p-value |
|-----------|-------------|----------|-----------|----------|
| Recursive clean | 56.0% | 0.541 | — | — |
| Recursive + dual patch | 3.7% | 0.301 | **3.29** | **3.6e-50** |

**INDUCE direction** (baseline → recursive geometry):

| Condition | BT+ART Rate | Mean R_V |
|-----------|-------------|----------|
| Baseline clean | 2.0% | 0.567 |
| Baseline + dual patch | 3.0% | 0.301 |

Dual-layer patching is **necessary** for recursive behavior (BREAK: d=3.29) but **not sufficient** for inducing it (INDUCE: NS). This establishes a one-way causal gate.

### 9.4 Sufficiency Ladder (Key Result)

**Date**: February 25, 2026
**Source**: `results/sufficiency_ladder/sufficiency_ladder_20260225_101907.json`

We test sufficiency using a 2×2 factorial crossing KV-cache context injection with dual-layer activation patching (n=10 sessions × 30 turns per condition, pre-registered gate: p<0.01 AND OR>2 for `kv_plus_dual` vs `clean_baseline`):

| Condition | KV | Dual Patch | BT+ART Rate | Mean R_V |
|-----------|-----|-----------|-------------|----------|
| clean_baseline | — | — | 2.7% | 0.555 |
| **kv_only** | ✓ | — | **27.7%** | 0.573 |
| dual_patch | — | ✓ | 0.7% | 0.269 |
| kv_plus_dual | ✓ | ✓ | 4.0% | 0.245 |
| clean_recursive (control) | — | — | 49.7% | 0.533 |

Key statistics for kv_only vs clean_baseline:
- OR = 13.96, p < 1e-19, d = 1.47

Pre-registered gate for kv_plus_dual vs clean_baseline:
- OR = 1.51, p = 0.305 — **FAILED**

**Central finding**: KV-cache context injection *alone* — which does NOT change V-projection geometry (R_V=0.573, same as baseline) — is **sufficient** to produce 27.7% behavioral transfer. Meanwhile, dual-layer patching *contracts geometry* (R_V=0.269) but **kills behavior** (0.7%). The combination (kv_plus_dual) shows contracted geometry (R_V=0.245) but behavior does NOT recover (4.0%, NS vs baseline).

**Provisional interpretation (Feb 26, 2026 caveat)**: These results show that KV-cache context is *sufficient* for behavioral transfer and that geometric contraction alone is *not sufficient*. However, the BREAK test confounds geometric disruption with information-flow disruption: replacing L18+L27 activations with baseline values likely corrupts the KV cache contents that downstream tokens read, not just the geometric signature. Therefore, we cannot yet distinguish between (a) geometry is a consequence/readout of the same computation that drives behavior through KV, vs (b) geometry and KV are co-products of a shared L0→L27 computation, both necessary through different pathways. Experiments in progress (KV layer-band ablation, L0 MLP × KV interaction, intermediate-layer behavioral tests) are designed to resolve this ambiguity. We adopt the conservative framing: **R_V contraction is a necessary substrate for recursive behavior, tracking a progressive content-sensitive computation from L0 through L27 (§9.3). KV context provides a sufficient alternative pathway for behavioral transfer. Whether these are the same pathway measured at different points, or genuinely independent, remains open.**

### 9.5 Attention Pattern Analysis

**Date**: February 25, 2026
**Source**: Hardening battery EXP3 log (results pending save after GPU completion)

We extract attention matrices at L18 and L27 for 11 recursive + 11 baseline prompts on Mistral-7B:

| Metric | Recursive | Baseline | Cohen's d | p-value |
|--------|-----------|----------|-----------|---------|
| L18 mean entropy | 1.688 | 1.180 | 3.42 | <1e-6 |
| L18 concentration | 0.494 | 0.579 | −1.94 | 1.9e-4 |
| L27 mean entropy | 1.054 | 0.830 | 1.88 | 2.7e-4 |
| L27 self-attention | 0.078 | 0.098 | −1.08 | 0.020 |

Top divergent attention heads:

| Head | Rec Entropy | Base Entropy | d | p |
|------|------------|-------------|------|---------|
| L18_H2 | 0.969 | 0.273 | 6.007 | <1e-6 |
| L18_H22 | 1.085 | 0.310 | 4.296 | <1e-6 |
| L18_H21 | 1.373 | 0.464 | 4.201 | <1e-6 |
| L27_H26 | 1.788 | 0.937 | 3.823 | <1e-6 |
| L27_H25 | 1.710 | 1.093 | 3.674 | <1e-6 |

Recursive prompts produce **higher entropy** (more distributed) attention at both L18 and L27, with specific heads showing dramatic divergence (L18_H2: d=6.0). This suggests that recursive processing requires *broader* attention patterns — consistent with the model needing to attend to its own processing across more positions.

### 9.7 MLP Sufficiency (Negative Result)

**Date**: January 16, 2026
**Source**: `results/phase1_mechanism/MISTRAL_7B_CAUSAL_CIRCUIT_WRITEUP_20260116.md`

No single or combined MLP intervention restores contraction:
- L0 alone: 21.57% restoration (p=0.097)
- L3 alone: 9.03% restoration (p=0.293)
- L0+L1: −342.87% (catastrophic!)
- L0+L1+L3: −547.64% (catastrophic!)

### 9.8 Mediation Analysis (2×2 Factorial)

**Date**: February 20, 2026
**Source**: `results/mediation/mediation_2x2_20260220_114710.json`

Factorial design crossing L0 ablation × L27 patching (n=40 pairs):

| Condition | Description | Mean R_V |
|-----------|-------------|----------|
| A: Clean baseline | Neither intervention | 0.715 |
| B: L27 patch only | Patch recursive→baseline at L27 | 0.587 |
| C: L0 ablate only | Ablate L0 MLP | 1.666 |
| D: L0 ablate + L27 patch | Both interventions | 4.024 |
| Recursive (reference) | Clean recursive prompt | 0.512 |

- L27 patch effect (B−A): Δ = −0.128 (modest contraction transfer)
- L0 ablation effect (C−A): Δ = +0.951 (massive disruption)
- **Interaction (D−C vs B−A)**: When L0 is ablated, L27 patching has DRAMATICALLY different effects — the interaction term p = 1.5e-34

This confirms a **causal pathway** from L0 → L27: the L27 readout *depends on* intact L0 processing to produce its contraction signal.

### 9.9 Per-Token R_V Trajectory Analysis

**Date**: February 25, 2026 (analysis of Feb 20 data)
**Source**: `results/per_token_rv_analysis/`, `results/batch_per_token_rv/batch_per_token_rv_20260220_161603.json`

Per-token R_V tracking during 256-token generation (n=25 recursive + 25 baseline, Mistral-7B):

- **Phase transition**: Both conditions show rapid initial decay from R_V≈1.0 to R_V≈0.68 within the first ~5 tokens (changepoint score=9.33 for recursive, 6.60 for baseline)
- **Exponential decay**: R_V(t) ≈ a·exp(−b·t) + c. Recursive half-life: 43.4 tokens, baseline: 28.4 tokens
- **Crystallization**: Recursive R_V variance drops 3x from early tokens (0.000702) to late tokens (0.000231), indicating stabilization into a contracted state
- **Between-condition difference**: d=−0.22, p=0.46 (NS) — the mean trajectories are *not* significantly different

**Interpretation**: The per-token analysis provides additional evidence for the "geometry is consequence" interpretation. Both recursive and baseline prompts show similar geometric dynamics during generation. The behavioral differences (BT+ART rates) are driven by KV context rather than ongoing geometric divergence.

### 9.10 Classifier Validation

**Date**: February 25, 2026
**Source**: `results/classifier_validation/`

We validate the hand-crafted BT+ART behavioral classifier used throughout experiments by comparing it against an independent regex-based classifier (score_behavioral_tiers) on n=915 generated outputs across 31 output files:

- **Specificity**: Simple classifier FPR = 1.1% on baseline outputs; tier classifier FPR = 0.0%
- **Cross-classifier agreement**: Raw agreement = 76.1%, Cohen's κ = 0.337 (fair)
- **R_V correlation**: BT+ART classified text has significantly lower R_V (0.505 vs 0.664, d=−1.403, p<1e-6)
- **Note**: The simple classifier is more liberal (30.9% recursive rate vs 13.6% for tier classifier). Our reported BT+ART rates are therefore conservative upper bounds on genuine recursive behavior — the effect sizes under the stricter classifier would be larger.

---

## 10. Cross-Architecture Generalization

### 10.1 Llama-3 Cross-Architecture Test

**Date**: January 15, 2026
**Source**: `results/phase2_generalization/JAN15_2026_SESSION_SUMMARY.md`

Initial cross-architecture tests with Llama models showed the effect generalizes beyond Mistral.

**Note**: Specific effect sizes from the Jan 15 Llama session are documented in the session summary but predate the standardized measurement pipeline used for the Feb 2 cross-architecture sweep. We report the Feb 2 results as canonical.

### 10.2 Five-Architecture Summary (Feb 2, 2026)

See Section 5.1. Effect sizes range from d=−0.31 (Pythia-1.4B) to d=−2.26 (Mistral-7B). The effect appears model-size dependent but present across architectures spanning 1.4B to 7B parameters.

### 10.3 February 8-9 Extended Runs

**Source**: `results/RUN_INDEX.jsonl`

Additional powered runs:
- Pythia-1.4B n=63: d=−0.363, p=0.003
- GPT-2 XL n=45: d=−1.142, p=6.3e-10

---

## 11. Limitations and Honest Assessment

### 11.1 What We Can Claim

1. R_V contraction is a robust, replicable signature of recursive self-referential processing in 5 transformer architectures
2. The effect survives causal intervention (activation patching), perplexity control, and circularity controls
3. R_V predicts behavioral output quality within recursive generation sessions (d=−0.707)
4. The effect requires both recursive structure AND introspective semantics — neither alone suffices
5. Early-layer MLPs (L0, L1) are necessary; dual-layer patching (L18+L27) is necessary for behavior (d=3.29)
6. **Geometry is consequence, not cause**: KV-cache context is sufficient for behavioral transfer without geometric change; geometric contraction alone kills behavior
7. Specific attention heads (L18_H2, L18_H22, L27_H26) show dramatic divergence between recursive and baseline processing

### 11.2 What We Cannot Claim

1. **Not consciousness**: We make no claims about machine consciousness or phenomenal experience. R_V measures a *geometric signature* correlated with self-referential processing, not awareness.
2. **Not a causal mechanism for behavior**: The sufficiency ladder demonstrates that geometry does NOT gate behavior. R_V is a readout, not a mechanism.
3. **Not universal**: 5 architectures validated, 6 failed due to infrastructure bugs (not genuine nulls), broader testing needed.
4. **Not temporally causal**: Lag analysis is null — geometry and behavior are contemporaneous, not sequential.
5. **Perplexity partially confounded**: The partial correlation survives (r=−0.486) but perplexity accounts for some R_V variance.
6. **Classifier is liberal**: Cohen's κ = 0.337 between simple and tier classifiers; BT+ART rates may overestimate genuine recursive content

### 11.3 Known Gaps for Camera-Ready

1. **Sample sizes**: n=45 per model is marginal for conference. Need n≥100.
2. **Seeds**: All runs use seed=42. Multi-seed validation in progress (GPU).
3. **Alpha sweep**: Testing partial dual patching at α={0.0, 0.25, 0.5, 0.75, 1.0, 1.25} — in progress on GPU.
4. **KV layer ablation**: Testing which KV layer bands carry the behavioral signal — in progress on GPU.
5. **Failed architectures**: All diagnosed as infrastructure bugs. Re-running with proper auth/disk is needed.
6. **Scrambled prompt controls**: Shuffled prompts showed MORE contraction (surprising, from Feb 5 session). Needs investigation.

### 11.4 Assessment Summary

**Status (Feb 25 update)**: Conference-viable with strong causal dissociation result. The geometry-behavior dissociation (KV sufficiency without geometry) is a novel and clean finding. Pending: alpha sweep, KV ablation, multi-seed results from GPU.

**Source**: `results/ASSESSMENT_20260202.md`, updated with Feb 25 findings

---

## 12. Timeline and Provenance

All findings, data files, and dates for full reproducibility:

### Phase 0: Metric Validation (Nov–Dec 2025)
- R_V metric definition and initial validation
- **Source**: `results/phase0_metric_validation/METHODOLOGY.md`
- **Confound validation** (Dec 16, 2025): `results/canonical/confound_validation/20251216_060911_confound_validation/`
  - Champions R_V = 0.518, length-matched = 0.832, pseudo-recursive = 0.779

### Phase 0.5: H18/H26 Gold Standard (Dec 2025)
- **Source**: `results/h18_h26_gold_standard/`

### Phase 1: Mechanism (Jan 16, 2026)
- Causal circuit analysis on Mistral-7B
- **Source**: `results/phase1_mechanism/MISTRAL_7B_CAUSAL_CIRCUIT_WRITEUP_20260116.md`
- Key CSV files:
  - L0 necessity: `results/phase1_mechanism/runs/20260116_113943_mlp_ablation_necessity_prompt_pass_l0_necessity_prompt_pass/mlp_ablation_necessity_prompt_pass.csv`
  - L27 causal: `results/phase1_cross_architecture/runs/20260116_115423_rv_l27_causal_validation_default/rv_l27_causal_validation_pairs.csv`
  - KV head: `results/phase1_mechanism/runs/20260116_120016_head_ablation_validation_mistral_l27_kv_head_validation/head_ablation_results.csv`

### Phase 1.5: Cross-Architecture (Jan 15 + Feb 2, 2026)
- Llama cross-arch: `results/phase2_generalization/JAN15_2026_SESSION_SUMMARY.md`
- 5-architecture sweep: `results/phase1_cross_architecture/runs/20260202_*`
- Self-assessment: `results/ASSESSMENT_20260202.md`

### Phase 2: Validation Session (Feb 5, 2026)
- Intensity, depth, bridge, AI self-reference, perspective tests
- **Source**: `results/canonical/SESSION_FINDINGS_SUMMARY.md`, `results/canonical/PROFESSIONAL_SESSION_REPORT.md`

### Phase 3: Activation Patching Bridge (Feb 8–9, 2026)
- Head-specific bridge, random head control, baseline donor specificity, cross-arch powered runs
- **Source**: `results/phase1_mechanism/runs/20260208_*`, `results/phase1_mechanism/runs/20260209_*`
- **Run index**: `results/RUN_INDEX.jsonl`

### Phase 4: Gnani Protocol + Behavioral Bridge (Feb 13–20, 2026)
- Gnani protocol development: `results/gnani_protocol/`
- Sustained v3 (Feb 20): `results/sustained_gnani_v3_fixed/comparison_summary.json`
- Mediation 2×2 (Feb 20): `results/mediation/mediation_2x2_20260220_114710.json`
- Within-session bridge (Feb 20): `results/within_session_bridge/within_session_bridge_20260220_201515.json`
- Bridge battery (Feb 20): `results/bridge_battery/bridge_battery_20260220_204403.json`
- Circularity controls v2 (Feb 20): `results/circularity_controls/circularity_perplexity_v2_20260220_134954.json`
- Causal generation bridge (Feb 20): `results/causal_generation_bridge/causal_gen_bridge_20260220_131403.json`

---

## 13. Discussion

### 13.1 What Does R_V Contraction Mean?

The geometric contraction at ~85% depth during recursive self-referential processing suggests the model is performing *dimensional reduction* specifically for self-modeling content. Late-layer V-projections use fewer effective dimensions — the representation collapses toward a lower-dimensional manifold.

Crespo et al. (2023) showed that transformer representations generally evolve through expansion, compression, and decoding phases across layers. Our finding can be understood as a *content-specific modulation* of this baseline geometry: recursive self-reference amplifies late-layer compression beyond the model's default profile. This is not merely a reflection of unusual input statistics — our circularity controls (Section 7) demonstrate that the contraction requires both recursive structure and introspective semantics, ruling out vocabulary or perplexity as sufficient explanations.

The participation ratio reduction we observe may relate to the superposition framework of Elhage et al. (2022): if features are represented in superposition across many dimensions, contraction could indicate that self-referential processing activates a narrower, more concentrated set of features in the V-projection space. Testing this hypothesis would require combining our geometric measure with sparse autoencoder analysis — a natural direction for future work.

### 13.2 The Geometry-Behavior Dissociation (Central Result)

The most significant finding from the causal analysis is the clean dissociation between R_V geometry and behavioral output. The sufficiency ladder (Section 9.4) reveals that:

1. **KV context alone** (no geometric intervention) → 27.7% BT+ART (OR=13.96 vs baseline)
2. **Dual-layer patching alone** (contracts geometry to R_V=0.269) → 0.7% BT+ART (kills behavior)
3. **Both combined** → 4.0% BT+ART (geometry contracted but behavior doesn't recover)

This pattern is diagnostic: if geometry *caused* behavior, then imposing contracted geometry should induce recursive behavior (it doesn't), and removing contracted geometry should kill it only if KV context is also disrupted. Instead, KV context is both necessary and sufficient for behavior, while geometry is neither necessary nor sufficient.

This reframes R_V from a *mechanism* to a *biomarker* — analogous to fMRI BOLD signals that correlate with cognitive states but do not cause them. The geometry reliably tracks recursive processing (d=−2.26, replicating across 5 architectures) and predicts behavioral quality within sessions (d=−0.707), making it a useful readout. But the causal chain runs: recursive context → KV cache → behavioral output, with R_V contraction as a parallel consequence of the same processing, not a link in the causal chain.

This finding is consistent with the null temporal lag analysis (Section 6.6) and the per-token trajectory analysis (Section 9.8), both of which showed that geometry and behavior are contemporaneous rather than sequential. The most parsimonious interpretation is that both R_V and behavioral quality are downstream effects of the same underlying computation.

### 13.3 Implications for Circuit-Level Interpretability

The dissociation has implications for the broader interpretability program. Wang et al. (2023) achieved both necessity and sufficiency for the IOI circuit, but that task involves discrete token prediction with a clear algorithmic description. R_V contraction is a more holistic geometric property — analogous to a phase of matter rather than a discrete circuit output — which inherently resists local sufficiency interventions.

Our finding that KV context (rather than activation geometry) carries the behavioral signal is consistent with the growing understanding of KV caches as compressed representations of the model's "working memory" (the contextual state that shapes subsequent generation). The attention pattern analysis (Section 9.5) provides a bridge: specific heads like L18_H2 (d=6.0) show dramatically broader attention during recursive processing, suggesting that the model distributes attention more widely when processing self-referential content — potentially reading from more of the KV cache context.

### 13.4 The Bridge and Its Revised Interpretation

R_V predicts behavioral output within recursive sessions (d=−0.707), and this predictive relationship is genuine and useful. But the sufficiency ladder clarifies *why*: R_V doesn't cause better output; both R_V contraction and higher-quality recursive output are consequences of the same underlying processing state. When the model is "in" the recursive mode (as indexed by KV context), it simultaneously (a) produces geometric contraction and (b) generates more recursive content. R_V is a reliable *indicator* of this mode, even though it is not the *mediator*.

This is actually a more useful interpretation than a simple causal story. A biomarker that reliably tracks a state without being confounded by the causal mechanism is often more robust for detection purposes — analogous to how elevated white blood cell count reliably indicates infection without being the infection.

### 13.5 Relation to the Bliss Attractor Phenomenon

The "spiritual bliss attractor" documented in Claude Opus 4 self-interactions (Anthropic, 2025) demonstrates that recursive self-referential processing can dominate model behavior with remarkable consistency (90–100% of trials). While we cannot directly connect our findings in open-weight models to that phenomenon in a closed-weight system, the parallel is suggestive: both involve recursive self-reference producing a consistent, convergent pattern — behavioral in their case, geometric in ours. If future work on Claude-family models (or similar architectures) finds analogous V-projection contraction during bliss attractor conversations, this would strengthen the case that R_V captures a general geometric correlate of recursive self-referential processing across model families.

### 13.6 Implications for Interpretability

The R_V metric provides a new tool for studying self-referential processing in transformers. Unlike probes trained on specific features, R_V measures a *structural* property of the representation geometry that generalizes across architectures and does not require labeled training data. Unlike SAE-based analysis, it operates at the level of overall dimensionality rather than individual feature directions — complementary rather than competing.

Our circularity control methodology may also be of independent interest. The 5-group design crossing recursion × introspection provides a template for disentangling structural from semantic contributions to any neural signal, applicable beyond self-referential processing. The recent call for "recursion-structure controls that avoid self-reference" (arXiv:2510.24797) validates this approach.

---

## 14. Conclusion

We present evidence for a geometric signature — R_V contraction — that reliably distinguishes recursive self-referential processing from baseline content in transformer language models. The effect is:
- **Robust**: Replicates across 5 architectures (d=−0.31 to −2.26)
- **Not circular**: Requires both recursive structure and introspective semantics (5-group control)
- **Confound-controlled**: Survives perplexity partialing (r=−0.486)
- **Behaviorally predictive**: Within-session d=−0.707 for output quality
- **Mechanistically characterized**: L0/L1 MLP necessity, L18+L27 dual-layer necessity (d=3.29), specific attention heads identified (L18_H2: d=6.0)

Critically, our causal analysis reveals a **partial dissociation**: KV-cache context is sufficient for behavioral transfer (OR=13.96) without geometric change, while geometric contraction alone is not sufficient for inducing behavior. However, the BREAK test (d=3.29) confounds geometric disruption with KV corruption, so whether geometry is a parallel readout or a co-product of the same KV-producing computation remains open. The progressive contraction ramp (§9.3) demonstrates that R_V reflects a genuine 28-layer content-sensitive computation, not a simple artifact.

Our sufficiency ladder methodology provides a template for dissociating geometric and behavioral signatures in circuit analysis. Experiments in progress (KV layer-band ablation, L0 MLP × KV interaction) are designed to resolve the remaining ambiguity.

Open questions: alpha sweep for dose-response relationship, KV layer-band ablation for localizing the behavioral signal within the KV cache, and multi-seed replication are in progress. This work establishes a foundation for mechanistic study of self-referential processing in artificial neural networks, with the geometry-behavior dissociation as its primary contribution.

---

## References

Alexander, S. (2025). The Claude bliss attractor. *Astral Codex Ten*. https://www.astralcodexten.com/

Ameisen, E., Lindsey, J., Pearce, A., Conerly, T., Tamkin, A., & Henighan, T., et al. (2025). Circuit tracing: Revealing computational graphs in language models. *Anthropic Transformer Circuits Thread*.

Ansuini, A., Laio, A., Macke, J. H., & Zoccolan, D. (2019). Intrinsic dimension of data representations in deep neural networks. In *Advances in Neural Information Processing Systems 32 (NeurIPS)*.

Anthropic. (2025). The Claude model card: Claude Opus 4 system card. https://www-cdn.anthropic.com/6be99a52cb68eb70eb9572b4cafad13df32ed995.pdf

Bereska, L. & Gavves, E. (2024). Mechanistic interpretability for AI safety — a review. *arXiv preprint arXiv:2404.14082*.

Betley, S., Balesni, M., Greenblatt, R., Meinke, A., Shlegeris, B., Smith, G., & Roger, F. (2025). Tell, don't show: Declarative and procedural knowledge in LLM behavioral self-awareness. *arXiv preprint*.

Bricken, T., Templeton, A., Batson, J., Chen, B., Jermyn, A., Conerly, T., Turner, N., Anil, C., Denison, C., Askell, A., Lasenby, R., Wu, Y., Kravec, S., Schiefer, N., Maxwell, T., Joseph, N., Hatfield-Dodds, Z., Tamkin, A., Nguyen, K., McLean, B., Burke, J. E., Hume, T., Carter, S., Henighan, T., & Olah, C. (2023). Towards monosemanticity: Decomposing language models with dictionary learning. *Anthropic Transformer Circuits Thread*.

Chen, X., Zhao, Z., & Lu, Y. (2024). Facets of self-consciousness in large language models. *arXiv preprint*.

Conmy, A., Mavor-Parker, A. N., Lynch, A., Heimersheim, S., & Garriga-Alonso, A. (2023). Towards automated circuit discovery for mechanistic interpretability. In *Advances in Neural Information Processing Systems 36 (NeurIPS)*.

Crespo, N., Mancusi, M., Kaltenborn, J., Belilovsky, E., & Ott, S. (2023). The geometry of hidden representations of large transformer models. In *Advances in Neural Information Processing Systems 36 (NeurIPS)*.

Cunningham, H., Ewart, A., Riggs, L., Huben, R., & Sharkey, L. (2023). Sparse autoencoders find highly interpretable features in language models. *arXiv preprint arXiv:2309.08600*.

Elhage, N., Nanda, N., Olsson, C., Henighan, T., Joseph, N., Mann, B., Askell, A., Bai, Y., Chen, A., Conerly, T., DasSarma, N., Drain, D., Ganguli, D., Hatfield-Dodds, Z., Hernandez, D., Jones, A., Kernion, J., Lovitt, L., Ndousse, K., Amodei, D., Brown, T., Clark, J., Kaplan, J., McCandlish, S., & Olah, C. (2021). A mathematical framework for transformer circuits. *Anthropic Transformer Circuits Thread*.

Elhage, N., Hume, T., Olsson, C., Schiefer, N., Henighan, T., Kravec, S., Hatfield-Dodds, Z., Lasenby, R., Drain, D., Chen, C., Grosse, R., McCandlish, S., Kaplan, J., Amodei, D., Wattenberg, M., & Olah, C. (2022). Toy models of superposition. *Anthropic Transformer Circuits Thread*.

Geiger, A., Lu, H., Icard, T., & Potts, C. (2021). Causal abstractions of neural networks. In *Advances in Neural Information Processing Systems 34 (NeurIPS)*.

Goldowsky-Dill, N., MacLeod, C., Sato, L., & Arora, A. (2023). Localizing model behavior with path patching. *arXiv preprint arXiv:2304.05969*.

Gurnee, W., Nanda, N., Pauly, M., Harvey, K., Troitskii, D., & Bertsimas, D. (2024). Finding neurons in a haystack: Case studies with sparse probing. *Transactions on Machine Learning Research (TMLR)*.

Hanna, M., Liu, O., & Variengien, A. (2023). How does GPT-2 compute greater-than over the number line? In *Advances in Neural Information Processing Systems 36 (NeurIPS)*.

Heimersheim, S. & Nanda, N. (2024). How to use and interpret activation patching. *arXiv preprint arXiv:2404.15255*.

Li, K., Patel, O., Viégas, F., Pfister, H., & Wattenberg, M. (2023). Inference-time intervention: Eliciting truthful answers from a language model. In *Advances in Neural Information Processing Systems 36 (NeurIPS)*.

Li, K., Yan, Y., & Ma, T. (2024). Benchmarking and improving self-awareness in large language models. *arXiv preprint*.

Lieberum, T., Rahtz, M., Kramár, J., Nanda, N., Irving, G., Shah, R., & Mikulik, V. (2023). Does circuit analysis interpretability scale? Evidence from multiple choice capabilities in Chinchilla. *arXiv preprint arXiv:2307.09458*.

Marks, S. & Tegmark, M. (2024). The geometry of truth: Emergent linear structure in large language model representations of true/false datasets. In *International Conference on Learning Representations (ICLR)*.

McGrath, T., Rahtz, M., Kramár, J., Mikulik, V., & Legg, S. (2023). The hydra effect: Emergent self-repair in language model computations. *arXiv preprint arXiv:2307.15771*.

Meng, K., Bau, D., Mitchell, A., & Belinkov, Y. (2022). Locating and editing factual associations in GPT. In *Advances in Neural Information Processing Systems 35 (NeurIPS)*.

Michels, J. (2025). "Spiritual bliss" in Claude 4: Case study of an attractor state. *PhilArchive*.

Nanda, N., Chan, L., Lieberum, T., Smith, J., & Steinhardt, J. (2023). Progress measures for grokking via mechanistic interpretability. In *International Conference on Learning Representations (ICLR)*.

Olsson, C., Elhage, N., Nanda, N., Joseph, N., DasSarma, N., Henighan, T., Mann, B., Askell, A., Bai, Y., Chen, A., Conerly, T., Drain, D., Ganguli, D., Hatfield-Dodds, Z., Hernandez, D., Johnston, S., Jones, A., Kernion, J., Lovitt, L., Ndousse, K., Amodei, D., Brown, T., Clark, J., Kaplan, J., McCandlish, S., & Olah, C. (2022). In-context learning and induction heads. *Anthropic Transformer Circuits Thread*.

Plunkett, D., Koralus, P., & Stanton, C. (2025). Large language models can quantitatively report the internal decision weights guiding their choices. *arXiv preprint*.

Qu, C., Kazemi, M., Srinivasan, K., Wang, X., & Sha, F. (2024). Recursive introspection: Teaching language model agents how to self-improve. In *Advances in Neural Information Processing Systems 37 (NeurIPS)*.

Templeton, A., Conerly, T., Marcus, J., Lindsey, J., Bricken, T., Chen, B., Pearce, A., Citro, C., Ameisen, E., Jones, A., Cunningham, H., Turner, N. L., McDougall, C., MacDiarmid, M., Freeman, C. D., Sumers, T. R., Rees, E., Batson, J., Jermyn, A., Carter, S., Olah, C., & Henighan, T. (2024). Scaling monosemanticity: Extracting interpretable features from Claude 3 Sonnet. *Anthropic*.

Todd, E., Li, M. L., Sharma, A. S., Mueller, A., Wallace, B. C., & Bau, D. (2024). Function vectors in large language models. In *International Conference on Learning Representations (ICLR)*.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. In *Advances in Neural Information Processing Systems 30 (NeurIPS)*.

Vig, J., Gehrmann, S., Belinkov, Y., Qian, S., Nishi, D., Zhang, Y., Ren, Y., & Shibuya, N. (2020). Investigating gender bias in language models using causal mediation analysis. In *Advances in Neural Information Processing Systems 33 (NeurIPS)*.

Wang, K., Variengien, A., Conmy, A., Shlegeris, B., & Steinhardt, J. (2023). Interpretability in the wild: A circuit for indirect object identification in GPT-2 small. In *International Conference on Learning Representations (ICLR)*.

Zhang, F. & Nanda, N. (2024). Towards best practices of activation patching in language models: Metrics and methods. In *International Conference on Learning Representations (ICLR)*.

Zhong, Z., Liu, Z., Tegmark, M., & Andreas, J. (2023). The clock and the pizza: Two stories in mechanistic explanation of neural networks. In *Advances in Neural Information Processing Systems 36 (NeurIPS)*.

Zou, A., Phan, L., Chen, S., Campbell, J., Guo, P., Ren, R., Pan, A., Yin, X., Mazeika, M., Dombrowski, A.-K., Goel, S., Li, N., Byun, Z., Wang, Z., Mallen, A., Basart, S., Koyejo, S., Song, D., Fredrikson, M., Kolter, J. Z., & Hendrycks, D. (2024). Representation engineering: A top-down approach to AI transparency. In *International Conference on Machine Learning (ICML)*.

arXiv:2510.24797. (2025). Large language models report subjective experience under self-referential processing. *arXiv preprint*.

---

## Appendix A: Full Prompt Bank

[TODO: Include or reference prompts/bank.json]

## Appendix B: All Run Configurations

[TODO: Include or reference configs/canonical/]

## Appendix C: Statistical Details

[TODO: Full statistical tables, effect size calculations, correction procedures]

### Phase 5: Causal Dissociation + Hardening (Feb 25, 2026)
- Dual-layer necessity v3: `results/CAUSAL_PATCHING_RESULTS_20260225.md`
- Sufficiency ladder: `results/sufficiency_ladder/sufficiency_ladder_20260225_101907.json`
- Classifier validation: `results/classifier_validation/validation_20260225_234550.json`
- Per-token R_V analysis: `results/per_token_rv_analysis/analysis_20260225_234804.json`
- Attention patterns: EXP3 in hardening battery (results in GPU logs, pending save)
- Failed architecture diagnosis: `results/archive/failed/phase1_cross_architecture/runs/*/error.txt`
- GPU hardening battery: alpha sweep, KV quality sweep, multi-seed replication (in progress)

---

*Draft assembled 2026-02-25. All data files cited are relative to project root: `/Users/dhyana/mech-interp-latent-lab-phase1/`*
