# Paper Grading Rubric: R_V Contraction Draft vs. Top 25 MI Papers

**Date**: 2026-02-20
**Purpose**: Quantitative comparison of our V0.0.1 draft against the standards set by 25 leading mechanistic interpretability papers.

---

## The 25 Reference Papers

| # | Paper | Venue | Year | Key Contribution |
|---|-------|-------|------|-----------------|
| 1 | Wang et al. "Interpretability in the Wild" (IOI) | NeurIPS | 2023 | Full circuit for indirect object identification in GPT-2 Small |
| 2 | Olsson et al. "In-context Learning and Induction Heads" | Anthropic | 2022 | Induction heads as mechanism for ICL across 22 models |
| 3 | Meng et al. "Locating and Editing Factual Associations" (ROME) | NeurIPS | 2022 | Causal tracing + rank-one model editing for facts |
| 4 | Elhage et al. "Toy Models of Superposition" | Anthropic | 2022 | Theoretical + empirical framework for superposition |
| 5 | Conmy et al. "Towards Automated Circuit Discovery" (ACDC) | NeurIPS | 2023 | Automated circuit finding via activation patching |
| 6 | Bricken et al. "Towards Monosemanticity" (SAEs) | Anthropic | 2023 | Sparse autoencoders for 1-layer transformer |
| 7 | Templeton et al. "Scaling Monosemanticity" | Anthropic | 2024 | SAEs on Claude 3 Sonnet (frontier model) |
| 8 | Geva et al. "Dissecting Recall of Factual Associations" | EMNLP | 2023 | Three-step factual recall pipeline in LMs |
| 9 | Hanna et al. "How does GPT-2 compute greater-than?" | NeurIPS | 2023 | Circuit for numerical comparison in GPT-2 |
| 10 | Nanda et al. "Progress Measures for Grokking" | ICLR | 2023 | Mechanistic explanation of grokking in modular addition |
| 11 | Zou et al. "Representation Engineering" | ICML | 2024 | Reading/controlling representations via linear probes |
| 12 | Vig et al. "Causal Mediation Analysis for NLP" | NeurIPS | 2020 | Foundational causal mediation for transformers |
| 13 | Marks & Tegmark "The Geometry of Truth" | ICLR | 2024 | Linear representations of truth in LLMs |
| 14 | Heimersheim & Nanda "How to use activation patching" | arXiv | 2024 | Best practices and pitfalls for patching |
| 15 | Zhang & Nanda "Best Practices of Activation Patching" | ICLR | 2024 | Systematic examination of patching methodology |
| 16 | Chughtai et al. "A Toy Model of Universality" | arXiv | 2023 | Group composition circuits across architectures |
| 17 | Todd et al. "Function Vectors in Large Language Models" | ICLR | 2024 | Task vectors in attention head outputs |
| 18 | Cunningham et al. "SAEs Find Highly Interpretable Features" | arXiv | 2023 | SAEs on GPT-2 with interpretability metrics |
| 19 | Gurnee et al. "Finding Neurons in a Haystack" | TMLR | 2024 | Sparse probing across Pythia models |
| 20 | Crespo et al. "Geometry of hidden representations" | NeurIPS | 2023 | Intrinsic dimension profiles across layers |
| 21 | Lieberum et al. "Does circuit analysis scale?" | arXiv | 2023 | Circuit analysis on Chinchilla (70B) |
| 22 | Goldowsky-Dill et al. "Path Patching" | arXiv | 2023 | Localization via path-level interventions |
| 23 | Zhong et al. "The Clock and the Pizza" | NeurIPS | 2023 | Two mechanistic explanations for modular arithmetic |
| 24 | McGrath et al. "The Hydra Effect" | arXiv | 2023 | Emergent self-repair after ablation |
| 25 | Li et al. "Inference-Time Intervention" | NeurIPS | 2023 | Truthfulness improvement via activation shifting |

---

## The 10-Metric Rubric

Each metric is scored 1–10 based on where the paper falls relative to the field distribution.

### M1: Architectures Tested (count of distinct model families)

**Field range**: 1–22 models

| Tier | Count | Score | Examples |
|------|-------|-------|---------|
| Bottom | 1 model | 1–2 | IOI (GPT-2 Small only), Hanna et al. (GPT-2 Small), Nanda et al. (1-layer toy) |
| Low | 2 models | 3–4 | Geva et al. (GPT-2 + GPT-J), ROME (GPT-2 + GPT-J) |
| Mid | 3–5 models | 5–6 | Marks & Tegmark (3 Llama variants), Conmy et al. (GPT-2 + tracr) |
| High | 6–10 models | 7–8 | Zou et al. (8+ models), Li et al. (6 models), Gurnee et al. (Pythia suite, 8 sizes) |
| Top | 11+ models | 9–10 | Olsson et al. (22 models from 5 families) |

**Field median**: ~2 models (most MI papers test on 1–2 models)
**Our paper**: 5 architectures validated + 4 failed (documented) = **Score: 6/10**

### M2: Primary Experiment Sample Size (n per condition)

**Field range**: 10 to 25,000+

| Tier | n | Score | Examples |
|------|---|-------|---------|
| Bottom | n < 50 | 1–2 | Many circuit papers use hand-crafted prompts (IOI: ~100 templates, but tightly scoped) |
| Low | 50–200 | 3–4 | Marks & Tegmark (~48 statements × multiple variants), Hanna et al. (~100 examples) |
| Mid | 200–1000 | 5–6 | ROME (1209 facts), Geva et al. (~500 facts), Zou et al. (several hundred per task) |
| High | 1000–5000 | 7–8 | Gurnee et al. (3000+ probing examples), Bricken et al. (large token batches) |
| Top | 5000+ | 9–10 | Olsson et al. (full training distribution), Templeton et al. (billions of tokens for SAE training) |

**Field median**: ~200–500 examples for circuit-level work
**Our paper**: 120 prompts × 45 pairs = ~5,400 total data points across experiments; sustained generation: 300 turns per condition; C2 n=755; circularity controls n=110. Total unique measurements: ~1,500+ → **Score: 5/10**

### M3: Control Conditions (count of distinct experimental controls)

**Field range**: 0–15

| Tier | Count | Score | Examples |
|------|-------|-------|---------|
| Bottom | 0–1 | 1–2 | Some early circuit papers with no formal controls |
| Low | 2–3 | 3–4 | ROME (clean vs corrupted), IOI (ABC template as control) |
| Mid | 4–6 | 5–6 | Hanna et al. (several ablation types), Marks & Tegmark (negations, unrelated statements) |
| High | 7–10 | 7–8 | Conmy et al. (multiple ablation methods compared), Zhang & Nanda (systematic method comparison) |
| Top | 11+ | 9–10 | Comprehensive ablation studies with many systematic controls |

**Field median**: ~3–4 controls
**Our paper**: 5 circularity control groups + length-matched + pseudo-recursive + shuffled + perplexity-matched + baseline types (creative/math/factual) + wrong-layer patching + random head + baseline donor = **13+ distinct controls** → **Score: 9/10**

### M4: Causal Intervention Types (count of distinct causal methods)

**Field range**: 0–6

| Tier | Count | Score | Examples |
|------|-------|-------|---------|
| Bottom | 0 | 1–2 | Purely correlational/probing work (Gurnee et al. is mostly probing) |
| Low | 1 | 3–4 | Single method: ROME (causal tracing only), Marks & Tegmark (probing + simple patching) |
| Mid | 2–3 | 5–6 | IOI (activation patching + path patching + knockout), Hanna et al. (patching + ablation + resample) |
| High | 4–5 | 7–8 | Conmy et al. (ACDC automated + manual + ablation + denoising + noising) |
| Top | 6+ | 9–10 | Comprehensive multi-method causal validation |

**Field median**: ~2 causal methods
**Our paper**: MLP ablation (necessity) + V-proj activation patching + KV head ablation + path patching (head-specific bridge) + random head control + baseline donor specificity + 2×2 factorial mediation + generation-time patching = **8 distinct interventions** → **Score: 10/10**

### M5: Statistical Rigor (effect sizes, corrections, CIs)

**Field range**: Informal to fully rigorous

| Tier | Description | Score | Examples |
|------|-------------|-------|---------|
| Bottom | No stats, qualitative only | 1–2 | Some Anthropic blog-style papers (Bricken et al. is largely qualitative) |
| Low | p-values only, no effect sizes | 3–4 | Many early MI papers |
| Mid | Effect sizes + p-values | 5–6 | ROME, IOI (logit diff as implicit effect size) |
| High | + multiple comparison correction OR cross-validation | 7–8 | Zhang & Nanda (systematic comparison), Marks & Tegmark (train/test split) |
| Top | + CIs + correction + cross-val + power analysis | 9–10 | Rare in MI field |

**Field median**: ~4–5 (p-values + some effect size measure)
**Our paper**: Cohen's d throughout + p-values + Holm-Bonferroni correction discussed + AUC/ROC + partial correlations + power analysis mentioned + Spearman + point-biserial + Mann-Whitney. Missing: confidence intervals, pre-registration. → **Score: 8/10**

### M6: Negative Results Honestly Reported (count)

**Field range**: 0–5

| Tier | Count | Score | Examples |
|------|-------|-------|---------|
| Bottom | 0 | 1–3 | Many papers don't report negatives at all |
| Low | 1 | 4–5 | IOI (mentions some heads have unclear roles) |
| Mid | 2–3 | 6–7 | Lieberum et al. (scaling difficulties honestly reported) |
| High | 4+ | 8–10 | Zhang & Nanda (systematic failure modes documented) |

**Field median**: ~0–1 (publication bias is severe in MI)
**Our paper**: MLP sufficiency fails (4 configs) + steering non-specific + temporal lag null + 4 architectures failed + causal generation bridge null + shuffled prompts anomaly + perplexity partial confound acknowledged = **7+ negative results** → **Score: 10/10**

### M7: Confound Analysis (count of confounds explicitly tested and controlled)

**Field range**: 0–5

| Tier | Count | Score | Examples |
|------|-------|-------|---------|
| Bottom | 0 | 1–2 | Most MI papers don't discuss confounds at all |
| Low | 1 | 3–4 | IOI (mentions position effects), ROME (notes limitations) |
| Mid | 2 | 5–6 | Marks & Tegmark (tests negation, unrelated) |
| High | 3–4 | 7–8 | Zou et al. (multiple control tasks) |
| Top | 5+ | 9–10 | Systematic confound elimination |

**Field median**: ~0–1 (confound analysis is rare in MI)
**Our paper**: Perplexity confound (partial correlation) + vocabulary confound (same_vocab control) + recursion-without-introspection + length matching + complexity matching + circularity (5-group design) = **6 confounds explicitly tested** → **Score: 10/10**

### M8: Behavioral Validation (does the internal metric predict downstream behavior?)

**Field range**: None to strong behavioral bridge

| Tier | Description | Score | Examples |
|------|-------------|-------|---------|
| Bottom | No behavioral link | 1–3 | Elhage et al. (pure geometry, no behavior), many SAE papers |
| Low | Qualitative behavioral examples | 4–5 | Bricken et al. (steering examples), Templeton et al. (Golden Gate Claude) |
| Mid | Correlational behavioral link | 6–7 | Marks & Tegmark (truth probes predict outputs), Li et al. (ITI improves TruthfulQA) |
| High | Causal behavioral validation | 8–9 | ROME (editing changes factual outputs), IOI (full circuit explains behavior) |
| Top | Within-subject + causal + cross-validated behavioral bridge | 10 | Extremely rare |

**Field median**: ~3–4 (most MI papers don't validate behavior)
**Our paper**: Prompt-to-output bridge (r=−0.443) + within-session bridge (d=−0.707) + logistic AUC=0.701 + C2 correlation (ρ=−0.652) + sustained generation behavioral difference (42.7% vs 11.3%). BUT: temporal lag null, causal generation bridge null. → **Score: 7/10**

### M9: Reproducibility (code, data, configs publicly available)

**Field range**: Nothing to full reproduction package

| Tier | Description | Score | Examples |
|------|-------------|-------|---------|
| Bottom | No code or data | 1–2 | Some Anthropic papers (no public SAE weights initially) |
| Low | Code only, no data | 3–4 | Many arXiv papers |
| Mid | Code + data + configs | 5–6 | IOI (TransformerLens notebook), Nanda et al. (full code) |
| High | + pip installable, well documented | 7–8 | Conmy et al. (ACDC library), Cunningham et al. (SAE training code) |
| Top | + Docker/env + deterministic reproduction | 9–10 | Rare |

**Field median**: ~4 (code available but often messy)
**Our paper**: Local repo with scripts, configs, prompt bank versioned (hash), JSON results, CSV data. BUT: not public, no pip package, no Docker, seeds not varied, hardware not fully documented. → **Score: 4/10**

### M10: References Cited (count, as proxy for literature situating)

**Field range**: 10–150

| Tier | Count | Score | Examples |
|------|-------|-------|---------|
| Bottom | < 15 | 1–2 | Short workshop papers |
| Low | 15–30 | 3–4 | Blog-style Anthropic papers (Bricken et al. ~30 refs) |
| Mid | 30–50 | 5–6 | IOI (~50 refs), Hanna et al. (~40 refs) |
| High | 50–80 | 7–8 | Conmy et al. (~60 refs), Zhang & Nanda (~70 refs) |
| Top | 80+ | 9–10 | Survey papers, Bereska & Gavves review (~200+ refs) |

**Field median**: ~40 references
**Our paper**: Currently **0 references** (all TODO). For a NeurIPS submission, need 40–60 minimum. → **Score: 1/10**

---

## Composite Scorecard

| Metric | Our Score | Field Median | Field Best | Gap to Median | Gap to Best |
|--------|-----------|--------------|------------|---------------|-------------|
| M1: Architectures tested | **6** | 4 | 10 (Olsson: 22) | +2 | −4 |
| M2: Sample size | **5** | 5 | 10 (Olsson/SAE) | 0 | −5 |
| M3: Control conditions | **9** | 4 | 9 | +5 | 0 |
| M4: Causal interventions | **10** | 5 | 8 | +5 | +2 |
| M5: Statistical rigor | **8** | 5 | 9 | +3 | −1 |
| M6: Negative results | **10** | 2 | 8 | +8 | +2 |
| M7: Confound analysis | **10** | 2 | 7 | +8 | +3 |
| M8: Behavioral validation | **7** | 4 | 9 (ROME/IOI) | +3 | −2 |
| M9: Reproducibility | **4** | 4 | 9 | 0 | −5 |
| M10: References cited | **1** | 6 | 10 | −5 | −9 |

**TOTAL: 70/100**
**Field median equivalent: ~41/100**
**Best-in-class equivalent: ~85/100**

---

## Interpretation

### Where we CRUSH the field (Score ≥ 9):
- **Controls** (M3: 9/10): Our 5-group circularity control design is more systematic than virtually any MI paper. Most papers have 1–3 controls.
- **Causal methods** (M4: 10/10): 8 distinct causal intervention types is exceptional. Most papers use 1–2.
- **Negative results** (M6: 10/10): 7+ honestly reported negatives is extremely rare. Publication bias means most papers report 0.
- **Confound analysis** (M7: 10/10): 6 explicitly tested confounds is unheard of in MI. Most papers never mention confounds.

### Where we're ABOVE median (Score 6–8):
- **Architectures** (M1: 6/10): 5 models is above the typical 1–2, but below the 10+ of the strongest cross-architecture studies.
- **Statistical rigor** (M5: 8/10): Cohen's d + corrections + multiple test types. Missing CIs and pre-registration.
- **Behavioral validation** (M8: 7/10): The within-session bridge (d=−0.707) is a strong result. Weakened by null temporal lag and null causal generation bridge.

### Where we're AT median (Score 4–5):
- **Sample size** (M2: 5/10): n=45 per model is standard for circuit work but would benefit from n≥100.
- **Reproducibility** (M9: 4/10): Local repo is well-organized but not public. No environment pinning.

### Where we're BELOW median (Score ≤ 3):
- **References** (M10: 1/10): Zero references currently. This is the single largest gap and will be immediately obvious to any reviewer.

---

## Priority Actions to Raise Score

### Critical (would change reviewer decision):
1. **Write the Related Work section** (M10: 1→7). Cite 40–60 papers. This alone adds +6 points.
2. **Make code/data public** (M9: 4→7). GitHub repo + requirements.txt + README. Adds +3.
3. **Add confidence intervals** to all primary effects (M5: 8→9). Bootstrap CIs for all Cohen's d values.

### High value (strengthen paper substantially):
4. **Increase n to 100+** per architecture (M2: 5→7). Especially for Pythia (currently marginal).
5. **Multi-seed validation** (M9 and M5 improvement). Run at least 3 seeds per primary experiment.
6. **Investigate failed architectures** (M1: 6→7). Document Llama-3, Gemma2, Falcon, StableLM failures.

### Nice to have (polish):
7. **Pre-register the behavioral bridge hypotheses** on OSF before running additional experiments.
8. **Add figures** — the draft has zero figures. Top MI papers have 5–15 figures. A figure of R_V across layers, the circularity control bar chart, and the within-session bridge scatter plot are essential.
9. **Formalize the R_V metric definition** with proper mathematical notation (currently pseudocode-level).

---

## Head-to-Head vs. Key Comparison Papers

### vs. IOI (Wang et al. 2023) — The gold standard for circuit analysis
- IOI: 1 model, ~26 heads in circuit, full necessity+sufficiency, no confound analysis, no cross-arch
- Us: 5 models, partial circuit (necessity only), excellent confound analysis, no sufficiency
- **We win on**: breadth, controls, confounds, statistics
- **They win on**: depth of circuit description, sufficiency, completeness

### vs. ROME (Meng et al. 2022) — The gold standard for causal editing
- ROME: 2 models, 1209 facts, causal tracing + editing, behavioral proof (model outputs change)
- Us: 5 models, 120 prompts, 8 causal methods, behavioral correlation but not causal editing
- **We win on**: causal method diversity, controls, cross-architecture
- **They win on**: sample size, clean behavioral proof, practical utility

### vs. Geometry of Hidden Representations (Crespo et al. 2023) — Closest methodological comparison
- Them: Intrinsic dimension across layers in ESM-2 + iGPT, no causal intervention, no behavioral link
- Us: Participation ratio across layers in 5 LLMs, causal interventions, behavioral bridge
- **We win on**: causal validation, behavioral link, controls, breadth
- **They win on**: theoretical elegance, multi-domain (protein + image + text)

### vs. Representation Engineering (Zou et al. 2023) — Cross-architecture representations
- Them: 8+ models, reading/controlling representations, behavioral intervention (truthfulness)
- Us: 5 models, reading geometry, weak behavioral intervention (generation bridge null)
- **We win on**: causal mechanism analysis, confound controls
- **They win on**: model count, practical intervention, cleaner behavioral link

---

## Bottom Line

**Our paper's unique strength** is the combination of rigorous controls + confound analysis + causal diversity that no single MI paper matches. The circularity control design (5-group crossing recursion × introspection) and the within-session behavioral bridge are genuinely novel contributions.

**Our paper's critical weakness** is presentation: zero references, zero figures, and the draft reads as a data dump rather than a narrative. The science is strong; the packaging is not yet at submission quality.

**Realistic NeurIPS assessment with current data**: If the presentation were polished (references, figures, clean narrative), the paper would be **competitive for a workshop paper** and possibly a **borderline main conference accept** — the controls and confound analysis would impress reviewers, but the null sufficiency and null temporal lag would likely draw "major revision" requests. The most likely outcome with current evidence: **6.0/10 NeurIPS score (borderline reject/weak accept)**, primarily limited by the missing sufficiency result and the incomplete circuit.
