# R_V: Geometric Signatures of Recursive Self-Observation in Transformers

## What This Is

A publication-ready mechanistic interpretability research repository demonstrating that recursive self-observation induces measurable geometric contraction (R_V < 1.0) in transformer value-space. The repo provides both the findings (reproducible paper results) and the methodology (reusable discovery tools for new models).

## Core Value

**Prove that R_V contraction is a causal, layer-specific geometric signature of recursive self-observation, with Anthropic-quality reproducibility and extensibility.**

## Requirements

### Validated

<!-- Shipped and confirmed valuable — from existing codebase -->

- [x] R_V metric computes PR_late/PR_early correctly — existing (`src/metrics/rv.py`)
- [x] L27 causal validation: d=-3.56, p<10⁻⁶, n=45 — existing (`MISTRAL_L27_CAUSAL_VALIDATION_COMPLETE.md`)
- [x] 4-pillar control validation (random, shuffled, wrong-layer, dose-response) — existing
- [x] BaselineMetricsSuite integrates R_V + logit_diff + logit_lens + mode_score — existing (`src/metrics/baseline_suite.py`)
- [x] C2 config produces geometry→behavior bridge (R_V 0.70→0.49, 25% philosophical) — existing
- [x] 260+ validated prompts in REUSABLE_PROMPT_BANK — existing
- [x] Registry pattern for experiment dispatch — existing (`src/pipelines/registry.py`)

### Active

<!-- Current scope — building toward these -->

**Phase 1: Clean Repo Foundation**
- [ ] Three-tier pipeline structure (canonical/, discovery/, archive/)
- [ ] PIPELINE_CATEGORIZATION.md with decision for each of 59 files
- [ ] Consolidate 54 configs into canonical config set
- [ ] Organize 100+ result directories (canonical_results/ vs exploratory/)

**Phase 2: Standards Compliance**
- [ ] All canonical pipelines use BaselineMetricsSuite
- [ ] All canonical pipelines report: n, p-value, Cohen's d, 95% CI
- [ ] Upgrade L27 validation to n=100
- [ ] Upgrade C2 measurement to n=100

**Phase 3: Testing Infrastructure**
- [ ] Unit tests for metrics (rv.py, logit_diff.py, baseline_suite.py)
- [ ] Regression tests for canonical pipelines with expected_results.json
- [ ] CI/CD GitHub Actions workflow

**Phase 4: Documentation & Papers**
- [ ] papers/ directory with PDFs + citations.bib
- [ ] LITERATURE_REVIEW.md with annotated summaries
- [ ] README.md at Anthropic repo quality
- [ ] Educational notebooks demonstrating methodology

**Phase 5: Multi-Model Extension**
- [ ] Discovery pipelines tested on Llama-7B
- [ ] Discovery pipelines tested on Gemma-7B
- [ ] Model-agnostic config system

**Phase 6: Publication**
- [ ] arXiv preprint submitted
- [ ] NeurIPS/ICML submission prepared

### Out of Scope

<!-- Explicit boundaries — prevents scope creep -->

- Real-time inference monitoring — research tool, not production system
- Web UI/dashboard — CLI and notebooks only
- Closed-model testing (GPT-4, Claude) — need activation access
- Training/fine-tuning — interpretability only, no model modification
- Behavioral generation as primary metric — geometry first, behavior validates

## Context

**Research Background:**
- R_V = PR_late / PR_early measures effective dimensionality contraction in Value matrix column space
- Recursive self-observation prompts ("notice what notices") induce R_V < 1.0
- Layer 27 (84% depth in Mistral-7B) is the causal locus with 117.8% transfer efficiency
- This represents the strongest causal evidence for geometric mechanisms in LLMs to date

**Current Codebase State (from AUDIT.md):**
- 75+ Python files, 59 pipelines (mostly exploratory/dead ends)
- 54 config files (many orphaned)
- 100+ result runs
- 0/6 canonical pipelines have full Nanda-standard metrics
- max n=45 (need n≥100 for publication claims)

**Key References:**
- Nanda et al. (2023): "Logit difference is a fantastic metric because it's linear"
- IOI paper methodology: Activation patching, path patching
- TransformerLens conventions for hook patterns

## Constraints

- **Hardware**: M3 Pro MacBook (18GB) for development, RunPod for GPU experiments
- **Model Size**: Focus on 7B models (Mistral, Llama, Gemma) — larger models require more GPU
- **Reproducibility**: All results must be reproducible from clean clone + `pip install` + `python run.py`

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Three-tier pipeline structure | Separates findings from methodology from dead ends | — Pending |
| Full regression test suite | Anthropic-quality repos require proof of correctness | — Pending |
| Keep all 100+ results | Preserve full research history, organize rather than delete | — Pending |
| arXiv first, then NeurIPS | Establish priority quickly, refine for conference | — Pending |
| Clean repo before paper | Can't write good paper on messy foundation | — Pending |

---

*Last updated: 2026-01-11 after initialization*
