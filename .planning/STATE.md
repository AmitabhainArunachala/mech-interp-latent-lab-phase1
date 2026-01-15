# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-15)

**Core value:** Prove that R_V contraction is a causal, layer-specific geometric signature of recursive self-observation, with Anthropic-quality reproducibility and extensibility.
**Current focus:** Phase 7 — Unit Tests

## Current Position

Phase: 7 of 10 (Unit Tests)
Plan: Not started
Status: Ready to plan Phase 7
Last activity: 2026-01-15 — Phases 1-6 COMPLETE, repo restructure ready to commit

Progress: ██████░░░░ 60%

## Completed Phases

### Phase 1: Pipeline Categorization ✓
- [x] 01-01: Analyzed all 59 pipelines
- [x] 01-02: Created PIPELINE_CATEGORIZATION.md

### Phase 2: Pipeline Restructure ✓
- [x] 02-01: Created directory structure, moved canonical pipelines
- [x] 02-02: Moved discovery/archive pipelines, updated imports

**Final Structure:**
- `src/pipelines/canonical/` - 7 pipelines (+ __init__.py)
- `src/pipelines/discovery/` - 12 pipelines (+ __init__.py)
- `src/pipelines/archive/` - 35 pipelines + 2 subdirs (+ __init__.py)
- Registry loads 43 experiments successfully

### Phase 3: Config Consolidation ✓
- [x] 03-01: Mapped config→pipeline relationships, identified orphans
- [x] 03-02: Created configs/canonical/, consolidated and archived

**Final Structure:**
- `configs/canonical/` - 14 configs
- `configs/discovery/` - 27 configs
- `configs/archive/` - 13 configs

### Phase 4: Results Organization ✓
- [x] 04-01: Audited all 137 result directories
- [x] 04-02: Reorganized into canonical/discovery/archive structure

**Final Structure:**
- `results/canonical/` - 3 directories containing paper-worthy runs (rv_l27, confound, c2_suite)
- `results/discovery/` - methodology runs (behavioral, steering, path_patching)
- `results/archive/` - historical runs (failed + superseded)
- Created RESULTS_CATEGORIZATION.md

### Phase 5: Metrics Compliance ✓ (SKIPPED)
- [x] Assessed canonical pipelines — already fit for purpose
- Rationale: Each canonical pipeline has purpose-specific metrics; BaselineMetricsSuite is for prompt analysis, not intervention validation

### Phase 6: Statistical Standards ✓
- [x] 06-01: Added statistical reporting to BaselineMetricsSuite
- [x] 06-02: Added Cohen's d + 95% CI to all canonical pipelines

**Implementations:**
- `baseline_suite.py`: Added `compute_ci_95()`, `compute_cohens_d()` helpers
- All canonical pipelines now report: n, p-value, Cohen's d, 95% CI

## Performance Metrics

**Velocity:**
- Total plans completed: 12
- Average duration: ~15 min
- Total execution time: ~3 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1. Pipeline Categorization | 2/2 | ~30 min | ~15 min |
| 2. Pipeline Restructure | 2/2 | ~30 min | ~15 min |
| 3. Config Consolidation | 2/2 | ~30 min | ~15 min |
| 4. Results Organization | 2/2 | ~30 min | ~15 min |
| 5. Metrics Compliance | 3/3 | SKIPPED | - |
| 6. Statistical Standards | 2/2 | ~30 min | ~15 min |

**Recent Trend:**
- Last 12 plans: All ✓
- Trend: On track

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- Three-tier structure (canonical/discovery/archive) — validated across pipelines, configs, results
- Full regression test suite — pending implementation
- Keep all 100+ results — organized rather than deleted

### Deferred Issues

None yet.

### Blockers/Concerns

None yet.

## Session Continuity

Last session: 2026-01-15
Stopped at: Phase 6 COMPLETE, ready for Phase 7 (Unit Tests)
Resume file: None

## Pending Actions

1. **GPU**: Llama cross-architecture validation (blocked on HF_TOKEN)
2. **NEXT**: Phase 7 — Unit tests for rv.py, logit_diff.py, baseline_suite.py

**Note:** Core restructure work has been staged and is ready to commit.
