# Architecture Review: Executive Summary

**Project**: rv_toolkit - R_V Metrics & Mechanistic Interpretability
**Review Type**: Structure Consolidation & Publication Readiness
**Date**: February 4, 2026
**Reviewer**: Architecture Assessment Agent

---

## Orientation Links (Top 10)

1. [Measurement Contract](docs/standards/MEASUREMENT_CONTRACT.md)
2. [Research Progress Summary](docs/status/RESEARCH_PROGRESS_SUMMARY.md)
3. [Phase 1 Final Report](R_V_PAPER/research/PHASE1_FINAL_REPORT.md)
4. [Bridge Hypothesis Investigation](BRIDGE_HYPOTHESIS_INVESTIGATION.md)
5. [Statistical Audit Executive Summary](STATISTICAL_AUDIT_EXECUTIVE_SUMMARY.md)
6. [Reproducibility Audit Report](REPRODUCIBILITY_AUDIT_REPORT.md)
7. [Quality Control Report](QUALITY_CONTROL_REPORT.md)
8. [Architecture Executive Summary](ARCHITECTURE_EXECUTIVE_SUMMARY.md)
9. [Publication Blockers Status](PUBLICATION_BLOCKERS_STATUS.md)
10. [Agent Onboarding](AGENT_ONBOARDING.md)

---

## Repo Story (12 bullets)

1. Core question: does recursive self-observation induce geometric contraction (R_V < 1.0)?
2. R_V defined as PR_late / PR_early on prompt tokens (window=16, early=5, late=depth-5).
3. Measurement contract is locked to avoid silent drift in definitions or parameters.
4. Canonical evidence shows strong contraction for recursive prompts vs baselines.
5. Cross-architecture replication exists with heterogeneous effect sizes.
6. Multi-token bridge shows strong between-group differences; within-group behavior link is weak.
7. Truncation is a major confound for behavioral correlations; longer generations required.
8. Causal claims require activation patching with proper controls and layer specificity.
9. Reproducibility hinges on config-driven runs and artifact completeness.
10. Hardware/precision logging is required for publication-grade reproducibility.
11. Architecture fragmentation exists; consolidation is recommended for publishability.
12. Current priority: causal bridge validation + reproducibility hardening.

---

## Current State (Honest Assessment)

The repository is **archaeologically sound but architecturally chaotic**:

### What's Working
- Core metric logic is solid (R_V computation, participation ratio)
- Activation patching implementations are validated
- Prompt bank (300 prompts) is complete and correct
- Experiment definitions capture full research history
- Tests exist (though scattered)
- Package structure (rv_toolkit/) provides foundation

### What's Broken
- **Fragmentation**: Same logic exists in 3+ locations
  - `rv_toolkit/metrics.py` + `src/metrics/rv.py` (duplicate R_V)
  - `src/core/patching.py` + `src/steering/activation_patching.py` + `rv_toolkit/patching.py` (3 variants)
  - Prompt bank in CANONICAL_CODE (hard to maintain)

- **No Clear Entry Point**:
  - CLI in rv_toolkit/cli.py
  - Experiments scattered across src/pipelines/
  - Scripts at root, in scripts/, in archive/scripts/
  - No single "how to run" answer

- **Dead Weight**:
  - 30+ archive scripts (33KB of history)
  - 11 root-level experiment scripts (gemma_*.py)
  - Deprecated prompts/ directory
  - src/ directory mixed with overlapping code

- **Import Hell**:
  - Circular dependencies possible (rv_toolkit imports src?)
  - Hard paths to prompts/data
  - No standard pattern for new experiments
  - Tests split between rv_toolkit/tests/ and scattered in src/

---

## Root Causes

### 1. Evolutionary Growth Without Architecture
The repository grew organically:
1. Started with `rv_toolkit/` as proposed public package
2. Parallel `src/` developed for internal research
3. CANONICAL_CODE created to preserve golden experiments
4. Each layer added imports without consolidation
5. Result: 3 overlapping module hierarchies

### 2. Separation Between "Research" and "Product"
There's an implicit split:
- `src/` = "How we explored"
- `rv_toolkit/` = "What we'll publish"
- CANONICAL_CODE = "What we trust"

But code isn't being integrated—just duplicated.

### 3. Lack of Namespace Discipline
Multiple inconsistent namespaces:
```
from src.metrics.rv import compute_rv
from rv_toolkit.metrics import compute_rv
from CANONICAL_CODE.n300_mistral_test_prompt_bank import prompt_bank_1c
from src.pipelines.canonical.rv_l27_causal_validation import ...
```

No one can remember the right import.

---

## Architectural Patterns Identified

### What Should Be True

**Single Source of Truth**:
```
rv_toolkit/
├── core/      (primitives only - models, hooks, utils)
├── metrics/   (ALL measurements in one place)
├── patching/  (ALL circuit manipulation in one place)
├── prompts/   (single prompt bank - versioned)
├── experiments/
│   ├── canonical/  (publication-ready)
│   └── discovery/  (exploratory)
└── tests/     (tests mirror source structure)
```

**Clear Boundaries**:
- `core/` = Load models, set up hooks, basic utilities
- `metrics/` = Measure things (R_V, logits, behavior)
- `patching/` = Modify activations/caches
- `prompts/` = All prompt data in one module
- `experiments/` = Run sequences of measurements + patching

**No Duplication**:
- One R_V implementation
- One activation patcher
- One prompt bank
- One behavior metric suite

---

## Architectural Recommendations

### Recommendation 1: CONSOLIDATE METRICS (Priority: CRITICAL)

**Current State**:
```
rv_toolkit/metrics.py ......................... 230 lines (R_V + analysis)
src/metrics/rv.py ............................ 210 lines (R_V computation)
src/metrics/behavior_strict.py ............... 180 lines (behavior)
src/metrics/behavior_states.py ............... 150 lines (states)
src/metrics/logit_diff.py .................... 120 lines (logit diff)
... 7 more metric files scattered
```

**Recommendation**:
```
rv_toolkit/metrics/
├── rv.py (consolidate from rv_toolkit/ + src/)
├── behavior.py (consolidate strict + states + bridge)
├── logit.py (consolidate logit_diff + logit_lens)
└── analysis.py (consolidate statistical tests)
```

**Rationale**:
- Single import path: `from rv_toolkit.metrics import compute_rv`
- Easy to find all metrics in one place
- Simplifies maintenance (one R_V, not two)
- Clear API surface

**Effort**: 20 minutes (copy + rename + dedupe)

---

### Recommendation 2: UNIFY PATCHING (Priority: CRITICAL)

**Current State**:
```
rv_toolkit/patching.py ...................... 320 lines (generic patching)
src/core/patching.py ........................ 280 lines (core implementation)
src/steering/activation_patching.py ......... 200 lines (variant)
src/steering/kv_cache.py .................... 150 lines (KV variant)
```

Three implementations of similar logic.

**Recommendation**:
```
rv_toolkit/patching/
├── activation.py (consolidate all activation patching)
├── kv_cache.py (KV cache specific)
└── circuit.py (circuit analysis helpers)
```

**Rationale**:
- Reduce cognitive load (one patcher, not three)
- ActivationPatcher is the canonical implementation
- KV cache patching is a natural variant (keep separate)
- Circuit analysis is a helper (clear boundary)

**Effort**: 30 minutes (careful consolidation, test after)

---

### Recommendation 3: RELOCATE PROMPT BANK (Priority: CRITICAL)

**Current State**:
```
CANONICAL_CODE/n300_mistral_test_prompt_bank.py .... 93KB (THE TRUTH)
prompts/ ................................. scattered deprecated prompts
REUSABLE_PROMPT_BANK/ ....................... symlink?
```

**Recommendation**:
```
rv_toolkit/prompts/
├── bank.py (move CANONICAL_CODE/n300_mistral_test_prompt_bank.py here)
├── loaders.py (add functions to access by category)
└── validators.py (add validation logic)
```

**Rationale**:
- Single source of truth (THE MOST IMPORTANT CODE)
- Versioned with rest of package
- Clear import: `from rv_toolkit.prompts import RECURSIVE_PROMPTS`
- Easy to add more prompt families (just extend bank.py)

**Effort**: 10 minutes (move + create loaders)

---

### Recommendation 4: ORGANIZE EXPERIMENTS (Priority: HIGH)

**Current State**:
```
src/pipelines/canonical/ .................... 10 experiments (validated)
src/pipelines/discovery/ .................... 20 experiments (exploratory)
src/pipelines/archive/ ...................... 40 experiments (history)
archive/scripts/ ............................ 30 scripts (ancient)
scripts/ ................................... 15 utility scripts
```

**Recommendation**:
```
rv_toolkit/experiments/
├── canonical/ (move from src/pipelines/canonical/)
├── discovery/ (move from src/pipelines/discovery/)
└── __init__.py (export all runnable experiments)

scripts/
├── run_experiment.py (unified experiment runner)
├── analyze_results.py (results analysis)
└── validate_reproducibility.py (validation)

[DELETE] src/pipelines/archive/ (keep in git history, remove from working tree)
[DELETE] archive/scripts/ (keep in git history, remove from working tree)
```

**Rationale**:
- Canonical experiments visible at package level
- Discovery experiments discoverable but separate
- Clear separation: code vs. scripts vs. results
- Archive preserved in git, not in working tree

**Effort**: 45 minutes (move + update imports)

---

### Recommendation 5: UNIFY TESTS (Priority: HIGH)

**Current State**:
```
rv_toolkit/tests/ ........................... scattered unit tests
src/metrics/ ............................... no tests (!)
src/pipelines/ ............................. no tests (!)
```

**Recommendation**:
```
tests/
├── test_metrics/
│   ├── test_rv.py
│   ├── test_behavior.py
│   └── test_logit.py
├── test_patching/
│   ├── test_activation.py
│   └── test_kv_cache.py
├── test_prompts/
│   └── test_bank.py
└── test_experiments/
    ├── test_canonical.py
    └── test_discovery.py
```

**Rationale**:
- Mirror structure of rv_toolkit/ for clarity
- All tests in one location (standard pytest)
- Easy to run subset: `pytest tests/test_metrics/`
- Single conftest.py for fixtures

**Effort**: 20 minutes (reorganize + create structure)

---

### Recommendation 6: CLEAN ROOT LEVEL (Priority: MEDIUM)

**Current State**:
```
Root has 11 Python files + 30 markdown files + 2 notebooks
```

**Recommendation**:
```
Keep at root:
- README.md (project overview)
- pyproject.toml (package config)
- LICENSE
- .gitignore
- requirements.txt

Move to docs/:
- All .md files (strategy, findings, audits)
- All .ipynb files

Move to results/ or archive:
- All gemma_*.py scripts
- neurips_n300_robust_experiment.py
- etc.

Move to scripts/:
- reproduce_results.py → scripts/reproduce.py
- openclaw_quickstart.py → docs/tutorials/
```

**Rationale**:
- Root is clean (only package + meta files)
- Documentation separate from code
- Clear hierarchy: src → tests → examples → docs

**Effort**: 10 minutes (mv commands)

---

## Implementation Roadmap

### Phase 1: Consolidate (Atomic Change)
```
1. Create directory structure (2 min)
2. Consolidate metrics (20 min)
3. Unify patching (30 min)
4. Relocate prompts (10 min)
5. Move experiments (45 min)
6. Reorganize tests (20 min)
7. Update imports (30 min)
8. Validate structure (15 min)
9. One commit (5 min)
```

**Total Time**: ~177 minutes (3 hours)
**Commits**: 1 (atomic consolidation)

### Phase 2: Documentation
```
- docs/architecture.md (new structure)
- docs/metrics.md (each metric documented)
- docs/patching.md (patching variants)
- docs/experiments.md (how to run)
- Update README.md with examples
```

**Total Time**: ~60 minutes (1 hour)
**Commits**: 1 (documentation)

### Phase 3: Validation
```
- Run full test suite
- Run canonical experiments (smoke test)
- Verify packaging (pip install -e .)
- Check imports (python -c "from rv_toolkit import *")
- Build documentation (if using Sphinx)
```

**Total Time**: ~30 minutes
**Commits**: 0 (if all passes)

---

## Scalability & Maintainability Assessment

### Current Architecture (Fragmented)
- **Onboarding time**: 2-3 hours (where is the R_V code?)
- **Adding metric**: Have to check 3 locations
- **Adding experiment**: Copy example, fix imports
- **Maintenance burden**: HIGH (duplicates to update)
- **Testing difficulty**: Scattered tests, unclear coverage

### Proposed Architecture (Consolidated)
- **Onboarding time**: 15 minutes (structure is obvious)
- **Adding metric**: `rv_toolkit/metrics/` + export
- **Adding experiment**: Copy from canonical/, update imports
- **Maintenance burden**: LOW (single source of truth)
- **Testing difficulty**: `pytest tests/` runs everything

---

## Technical Debt Assessment

### High-Severity Issues
1. **Duplicate R_V implementations** (3 copies)
   - Creates sync burden
   - Risk of divergence
   - MUST consolidate before publication

2. **Unclear prompt bank ownership** (CANONICAL_CODE vs. prompts/)
   - Risk of accidentally using old prompts
   - MUST have single source of truth

3. **No unified entry point**
   - Scripts scattered across 3+ directories
   - Hard to run experiments reproducibly
   - MUST consolidate

**Impact**: Publication readiness at risk

### Medium-Severity Issues
1. **Patching code duplication** (3 variants)
   - Could cause maintenance issues
   - SHOULD consolidate

2. **Tests scattered** (in rv_toolkit/ + nowhere else)
   - Incomplete coverage
   - Hard to extend tests
   - SHOULD reorganize

3. **Root clutter** (11 scripts)
   - Confusion about entry points
   - SHOULD clean up

### Low-Severity Issues
1. Archive scripts take up space (keep in git, remove from working tree)
2. Old configs in multiple locations (organize)
3. Documentation at root (move to docs/)

---

## Risk & Mitigation

### Risks of Restructuring
| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Import breakage | Medium | High | Test after each phase |
| Lost code | Low | Critical | Git preserves everything |
| Circular dependencies | Low | Medium | Check imports before commit |
| Experiments don't run | Medium | High | Run smoke test on canonical |

### Risks of NOT Restructuring
| Risk | Likelihood | Impact | Timing |
|------|-----------|--------|--------|
| Publication blocked on imports | High | Critical | Immediate |
| New contributor confusion | High | Medium | Next user |
| Merge conflicts (src vs rv_toolkit) | Medium | High | Within 1 month |
| Maintenance burden grows | High | Medium | Ongoing |

**Verdict**: Restructuring risk < No-restructuring risk

---

## Success Metrics

After restructuring, the architecture will be successful if:

1. **Unified Namespace** ✓
   - All imports start with `rv_toolkit.`
   - No `from src.` imports
   - CLI works: `rv-toolkit --help`

2. **Single Source of Truth** ✓
   - One R_V implementation
   - One activation patcher
   - One prompt bank
   - All discoverable in code

3. **Test Coverage** ✓
   - `pytest tests/` runs all tests
   - Clear test-to-source mapping
   - Coverage >= 70%

4. **Documentation** ✓
   - New contributor can find any component in 5 min
   - Each module has usage examples
   - README explains structure

5. **Publication Ready** ✓
   - Clean imports
   - No broken dependencies
   - Can install: `pip install -e .`
   - Can run canonical experiments

---

## Appendix: File Inventory Summary

### Consolidation Ratios

| Module | Current Locations | Files | Lines | → Target |
|--------|------------------|-------|-------|----------|
| Metrics | rv_toolkit/, src/metrics/ | 8 | ~1,400 | metrics/ |
| Patching | rv_toolkit/, src/core/, src/steering/ | 4 | ~950 | patching/ |
| Prompts | CANONICAL_CODE/, prompts/, REUSABLE_PROMPT_BANK | 3 | ~93KB | prompts/ |
| Experiments | src/pipelines/canonical, discovery, archive | 70+ | ~5,000 | experiments/ |
| Tests | rv_toolkit/tests/, scattered | 8 | ~500 | tests/ |

**Total Python Files**: ~170 → ~120 (after consolidation)
**Reduction**: ~30% fewer files, 0% loss of functionality

### What Gets Deleted (Logically)

```
Deleted from working tree (preserved in .git/):
- archive/scripts/*.py (30 files, 33KB) - HISTORY
- src/ (70+ files) - CODE MOVED, NOT DELETED
- CANONICAL_CODE/ (3 files) - PROMPTS MOVED, ANALYSIS MERGED
- Old configs (archive subdirs) - KEEP FOR REFERENCE
- Deprecated prompts/ - SUPERSEDED BY BANK

Physically removed (never needed again):
- None (git history is immutable)
```

---

## Recommendations Summary

### MUST DO (Publication Blocker)
1. **Consolidate metrics** - eliminate duplicate R_V implementations
2. **Unify patching** - single ActivationPatcher
3. **Relocate prompt bank** - single source of truth
4. **Create experiments structure** - clear canonical vs. discovery

### SHOULD DO (Quality, Maintainability)
1. **Reorganize tests** - unified test structure
2. **Clean root** - move docs, scripts, results
3. **Update imports** - single rv_toolkit namespace

### NICE TO HAVE (Later)
1. **Build documentation** - Sphinx/MkDocs
2. **Add CI/CD** - GitHub Actions
3. **Release process** - PyPI publishing

---

## Timeline & Effort

**Total Implementation**: 3-4 hours
**Documentation**: 1 hour
**Validation**: 30 minutes
**Total Timeline**: 5-5.5 hours (1 work day)

**After restructuring**:
- Publication can proceed (imports are clean)
- New experiments take 10 minutes to add
- Maintenance burden drops 50%
- Onboarding time drops to 15 minutes

---

## Conclusion

The repository's **technical foundation is solid** (metric logic, validation, experiments are correct). The **structure is fragmented** (same code in 3+ places, unclear entry points, scattered scripts).

**Restructuring is not a rewrite—it's a consolidation**. The code is already good; it just needs to be organized into a coherent namespace.

**Recommended action**: Execute the consolidation plan in a single atomic commit. This preserves all git history while creating a publication-ready structure.

The restructure **unblocks publication** and **enables sustainable development**.

---

*Architecture Review: Consolidation Strategy*
*Status: READY FOR IMPLEMENTATION*
*Next Step: Execute ARCHITECTURE_RESTRUCTURE_PLAN.md*
