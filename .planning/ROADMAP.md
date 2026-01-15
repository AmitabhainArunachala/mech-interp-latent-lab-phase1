# Roadmap: R_V Geometric Signatures

## Overview

Transform a research-phase codebase into a publication-ready, Anthropic-quality repository. First clean and organize the existing 59 pipelines and 100+ results, then bring all canonical work to statistical standards, add comprehensive testing, complete documentation with literature review, and finally extend to multi-model validation. The result is both a reproducible paper and a reusable methodology toolkit.

## Domain Expertise

None — custom mechanistic interpretability research project.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

- [x] **Phase 1: Pipeline Categorization** - Analyze and classify all 59 pipelines ✓
- [x] **Phase 2: Pipeline Restructure** - Move to canonical/, discovery/, archive/ ✓
- [x] **Phase 3: Config Consolidation** - Organize 54 configs into canonical set ✓
- [x] **Phase 4: Results Organization** - Structure 100+ result directories ✓
- [x] **Phase 5: Metrics Compliance** - Add BaselineMetricsSuite to canonical pipelines ✓ (SKIPPED - already fit for purpose)
- [ ] **Phase 6: Statistical Standards** - Add p-value, Cohen's d, 95% CI everywhere
- [ ] **Phase 7: Unit Tests** - Test suite for metric implementations
- [ ] **Phase 8: Regression Tests** - Expected values for canonical pipelines
- [ ] **Phase 9: Documentation** - README, literature review, papers/, notebooks
- [ ] **Phase 10: Multi-Model Extension** - Validate discovery pipelines on new models

## Phase Details

### Phase 1: Pipeline Categorization
**Goal**: Create PIPELINE_CATEGORIZATION.md with decision for each of 59 pipeline files
**Depends on**: Nothing (first phase)
**Research**: Unlikely (internal code analysis)
**Plans**: 2 plans

Plans:
- [x] 01-01: Analyze all 59 pipelines, document purpose and status ✓
- [x] 01-02: Create categorization document with canonical/discovery/archive decisions ✓

### Phase 2: Pipeline Restructure
**Goal**: Move pipeline files to three-tier structure (canonical/, discovery/, archive/)
**Depends on**: Phase 1
**Research**: Unlikely (file organization)
**Plans**: 2 plans

Plans:
- [x] 02-01: Create directory structure, move canonical pipelines ✓
- [x] 02-02: Move discovery and archive pipelines, update imports ✓

### Phase 3: Config Consolidation
**Goal**: Organize 54 config files into canonical set, archive orphans
**Depends on**: Phase 2
**Research**: Unlikely (internal cleanup)
**Plans**: 2 plans

Plans:
- [x] 03-01: Map config→pipeline relationships, identify orphans ✓
- [x] 03-02: Create configs/canonical/, consolidate and archive ✓

### Phase 4: Results Organization
**Goal**: Structure 100+ result directories into canonical_results/ vs exploratory/
**Depends on**: Phase 3
**Research**: Unlikely (file organization)
**Plans**: 2 plans

Plans:
- [x] 04-01: Audit all result directories, identify paper-worthy runs ✓
- [x] 04-02: Reorganize into canonical_results/, exploratory/, update run_index.csv ✓

### Phase 5: Metrics Compliance (SKIPPED)
**Goal**: Add BaselineMetricsSuite to all canonical pipelines
**Depends on**: Phase 4
**Research**: Unlikely (existing suite implementation)
**Status**: SKIPPED - After analysis, canonical pipelines already have purpose-specific metrics

**Rationale**: Each canonical pipeline is focused on intervention-based validation (patching, ablation, steering) and already has appropriate metrics. BaselineMetricsSuite is designed for comprehensive prompt analysis, not intervention validation. See `.planning/phases/05-metrics-compliance/PHASE_ASSESSMENT.md`.

Plans:
- [x] 05-01: Assessed rv_l27_causal_validation.py - already complete ✓
- [x] 05-02: Assessed confound_validation.py and random_direction_control.py - already complete ✓
- [x] 05-03: Assessed MLP pipelines - already use scipy.stats ✓

### Phase 6: Statistical Standards
**Goal**: All outputs include n, p-value, Cohen's d, 95% CI
**Depends on**: Phase 5
**Research**: Unlikely (scipy.stats patterns)
**Plans**: 2 plans
**Status**: COMPLETE

Plans:
- [x] 06-01: Add statistical reporting to BaselineMetricsSuite.compute_batch_statistics ✓
- [x] 06-02: Add Cohen's d + 95% CI to all canonical pipelines ✓

**Implementation Summary**:
- `baseline_suite.py`: Added `compute_ci_95()`, `compute_cohens_d()` helpers, proper pooled std
- `rv_l27_causal_validation.py`: Added 95% CI and Cohen's d to all statistical tests
- `confound_validation.py`: Added `_cohens_d()`, `_ci_95()`, integrated into t-tests and summary
- `random_direction_control.py`: Added t-tests, Cohen's d, 95% CI for true vs random comparisons
- `mlp_sufficiency_test.py`: Added 95% CI and Cohen's d with pooled std
- `mlp_combined_sufficiency_test.py`: Added 95% CI and Cohen's d with pooled std
- `mlp_ablation_necessity.py`: Added 95% CI and fixed Cohen's d to use pooled std
- `head_ablation_validation.py`: Already had 95% CI and Cohen's d (verified)

### Phase 7: Unit Tests
**Goal**: Test suite for rv.py, logit_diff.py, baseline_suite.py, mode_score.py
**Depends on**: Phase 6
**Research**: Unlikely (pytest standard patterns)
**Plans**: 2 plans

Plans:
- [ ] 07-01: Create tests/unit/ structure, test rv.py and logit_diff.py
- [ ] 07-02: Test baseline_suite.py and mode_score.py

### Phase 8: Regression Tests
**Goal**: Expected values for canonical pipelines with CI integration
**Depends on**: Phase 7
**Research**: Likely (need expected_results.json values from validated runs)
**Research topics**: Extract exact expected values from n=100 runs, tolerance thresholds
**Plans**: 3 plans

Plans:
- [ ] 08-01: Create tests/expected_results.json from validated runs
- [ ] 08-02: Write regression tests for canonical pipelines
- [ ] 08-03: Add GitHub Actions CI workflow

### Phase 9: Documentation
**Goal**: README, literature review, papers/, educational notebooks
**Depends on**: Phase 8
**Research**: Likely (literature analysis required)
**Research topics**: Nanda IOI paper methodology, TransformerLens conventions, related MI papers
**Plans**: 4 plans

Plans:
- [ ] 09-01: Create papers/ directory with PDFs and citations.bib
- [ ] 09-02: Write LITERATURE_REVIEW.md with annotated summaries
- [ ] 09-03: Write comprehensive README.md at Anthropic quality
- [ ] 09-04: Create educational notebooks demonstrating methodology

### Phase 10: Multi-Model Extension
**Goal**: Validate discovery pipelines on Llama-7B and Gemma-7B
**Depends on**: Phase 9
**Research**: Likely (new model architectures)
**Research topics**: Llama layer structure, Gemma attention patterns, model-agnostic config system
**Plans**: 3 plans

Plans:
- [ ] 10-01: Create model-agnostic config system
- [ ] 10-02: Run discovery pipelines on Llama-7B
- [ ] 10-03: Run discovery pipelines on Gemma-7B, document cross-model findings

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Pipeline Categorization | 2/2 | **COMPLETE** | 2026-01-11 |
| 2. Pipeline Restructure | 2/2 | **COMPLETE** | 2026-01-11 |
| 3. Config Consolidation | 2/2 | **COMPLETE** | 2026-01-11 |
| 4. Results Organization | 2/2 | **COMPLETE** | 2026-01-11 |
| 5. Metrics Compliance | 3/3 | **SKIPPED** | 2026-01-11 |
| 6. Statistical Standards | 2/2 | **COMPLETE** | 2026-01-11 |
| 7. Unit Tests | 0/2 | Not started | - |
| 8. Regression Tests | 0/3 | Not started | - |
| 9. Documentation | 0/4 | Not started | - |
| 10. Multi-Model Extension | 0/3 | Not started | - |

**Total Plans:** 25
