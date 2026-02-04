# Mech-Interp Repository Restructuring Plan

**Status**: Architecture Design Phase
**Target State**: Clean, publication-ready structure
**Scope**: Consolidate 70+ scattered Python files into coherent `rv_toolkit` namespace
**Timeline**: Single atomic git commit after validation

---

## Executive Summary

Current state is **archaeologically valid but architecturally incoherent**:
- Core metric logic scattered: `rv_toolkit/`, `src/metrics/`, `CANONICAL_CODE/`
- 30+ archive scripts mix validated with exploratory code
- Prompt bank in CANONICAL_CODE (hard to version/update)
- Configs in 4+ locations without clear ownership
- Entry points at root level (11 scripts)
- Test infrastructure split between `rv_toolkit/tests/` and `src/`

**Target**: Single source of truth for all research code. Everything else is documentation or results.

---

## Current State Analysis

### File Distribution

| Location | Count | Content | Status |
|----------|-------|---------|--------|
| Root | 11 | Experimental scripts (gemma_*.py, openclaw_quickstart.py) | REMOVE |
| rv_toolkit/ | 5 | Core metrics/patching/analysis | CONSOLIDATE HERE |
| CANONICAL_CODE/ | 3 | Prompt bank + validation scripts | RELOCATE |
| src/core/ | 10 | Model loading, hooks, patching | CONSOLIDATE |
| src/metrics/ | 12 | R_V, logit_diff, behavior metrics | CONSOLIDATE |
| src/pipelines/ | 70+ | Discovery, canonical, archive | ORGANIZE |
| archive/scripts/ | 30+ | Historical scripts | DELETE |
| scripts/ | 15+ | Utility/analysis scripts | EVALUATE |

**Total**: ~170 Python files across scattered locations

### Architecture Smell List

1. **Dual Patching Implementations**: `rv_toolkit/patching.py` vs `src/core/patching.py` vs `src/steering/`
2. **Metrics Fragmentation**: `rv_toolkit/metrics.py` vs 12 files in `src/metrics/`
3. **Prompt Bank Ambiguity**: CANONICAL_CODE vs prompts/ vs REUSABLE_PROMPT_BANK
4. **Test Split**: `rv_toolkit/tests/` + scattered src tests
5. **Config Chaos**: canonical/, discovery/, archive/, gold/, phase3_bridge/, smoke_test/
6. **No Clear Entry Point**: CLI in rv_toolkit/cli.py but pipelines in src/pipelines/
7. **Root Clutter**: 11 scripts at root (should be in scripts/ or results/)

### Dependency Analysis

```
rv_toolkit (current)
├── Imports from src/core/ (circular potential)
├── Uses CANONICAL_CODE prompts (hardcoded paths)
└── Tests expect flattened structure

CANONICAL_CODE
├── n300_mistral_test_prompt_bank.py (93KB - the true gold)
├── mistral_L27_FULL_VALIDATION.py (validation code)
└── causal_loop_closure_v2.py (analysis code)

src/
├── pipelines/ (70+ experiment definitions)
├── metrics/ (dual/parallel implementations)
├── core/ (model loading, patching, hooks)
└── steering/ (activation patching variants)
```

---

## IDEAL TARGET STRUCTURE

```
rv_toolkit/
├── __init__.py                           # Package entry point
├── pyproject.toml                        # [KEEP - already good]
├── README.md                             # [KEEP - already good]
├── LICENSE
│
├── rv_toolkit/                           # Main package
│   ├── __init__.py                       # Exports all public API
│   │
│   ├── core/                             # Core primitives
│   │   ├── __init__.py
│   │   ├── models.py                     # Model loading (from src/core/models.py)
│   │   ├── hooks.py                      # Hook infrastructure (from src/core/hooks.py)
│   │   └── utils.py                      # Utilities (consolidate src/core/utils.py + src/utils/)
│   │
│   ├── metrics/                          # ALL metric definitions
│   │   ├── __init__.py
│   │   ├── rv.py                         # R_V computation (from src/metrics/rv.py)
│   │   ├── participation_ratio.py        # PR helper (extract from rv.py)
│   │   ├── behavior.py                   # Behavioral metrics (consolidate src/metrics/behavior_*)
│   │   ├── logit.py                      # Logit-based metrics (consolidate src/metrics/logit_*)
│   │   └── analysis.py                   # Statistical analysis (consolidate src/metrics/ analysis)
│   │
│   ├── patching/                         # ALL patching implementations
│   │   ├── __init__.py
│   │   ├── activation.py                 # Activation patching (consolidate src/core/patching.py + src/steering/)
│   │   ├── kv_cache.py                   # KV cache patching (from src/steering/kv_cache.py)
│   │   └── circuit.py                    # Circuit analysis helpers
│   │
│   ├── prompts/                          # Prompt bank - SINGLE SOURCE OF TRUTH
│   │   ├── __init__.py
│   │   ├── bank.py                       # All 300+ prompts (from CANONICAL_CODE/n300_mistral_test_prompt_bank.py)
│   │   ├── loaders.py                    # Load by category/phase
│   │   └── validators.py                 # Validate prompt structure
│   │
│   ├── experiments/                      # Canonical & discovery pipelines
│   │   ├── __init__.py
│   │   ├── canonical/                    # Publication-ready experiments
│   │   │   ├── __init__.py
│   │   │   ├── rv_l27_causal_validation.py
│   │   │   ├── mlp_ablation_necessity.py
│   │   │   ├── confound_validation.py
│   │   │   └── [other canonical experiments...]
│   │   │
│   │   └── discovery/                    # Exploratory experiments
│   │       ├── __init__.py
│   │       ├── behavioral_grounding.py
│   │       ├── circuit_analysis.py
│   │       ├── layer_sweep.py
│   │       └── [other discovery experiments...]
│   │
│   ├── cli.py                            # Command line interface [KEEP]
│   └── version.py                        # Version constant
│
├── tests/                                # All tests
│   ├── __init__.py
│   ├── conftest.py                       # Pytest fixtures
│   ├── test_metrics/
│   │   ├── test_rv.py
│   │   ├── test_behavior.py
│   │   └── test_logit.py
│   ├── test_patching/
│   │   ├── test_activation.py
│   │   └── test_kv_cache.py
│   ├── test_prompts/
│   │   └── test_bank.py
│   └── test_experiments/
│       ├── test_canonical.py
│       └── test_discovery.py
│
├── examples/                             # Usage examples [KEEP]
│   ├── basic_rv_measurement.py
│   ├── activation_patching.py
│   └── full_pipeline.py
│
├── docs/                                 # Documentation [CREATE]
│   ├── README.md
│   ├── api.md
│   ├── architecture.md
│   ├── metrics.md
│   ├── patching.md
│   └── experiments.md
│
└── scripts/                              # User-facing scripts [CREATE]
    ├── run_experiment.py                 # Run any experiment by name
    ├── analyze_results.py                # Post-experiment analysis
    └── validate_reproducibility.py       # Verification script
```

---

## Migration Path (File Mapping)

### Core Consolidation

**rv_toolkit/core/models.py** (CREATE via merge):
```
← src/core/models.py (main)
← CANONICAL_CODE/mistral_L27_FULL_VALIDATION.py (model loading patterns)
```

**rv_toolkit/core/hooks.py**:
```
← src/core/hooks.py (as-is)
```

**rv_toolkit/core/utils.py**:
```
← src/core/utils.py (main)
← src/utils/*.py (consolidate pattern detectors, metadata)
```

### Metrics Consolidation

**rv_toolkit/metrics/rv.py**:
```
← src/metrics/rv.py (main R_V implementation)
← rv_toolkit/metrics.py (move into here)
```

**rv_toolkit/metrics/behavior.py**:
```
← src/metrics/behavior_strict.py
← src/metrics/behavior_states.py
← src/metrics/behavioral_bridge.py
← src/metrics/mode_score.py
```

**rv_toolkit/metrics/logit.py**:
```
← src/metrics/logit_diff.py
← src/metrics/logit_lens.py
```

**rv_toolkit/metrics/analysis.py**:
```
← rv_toolkit/analysis.py (move + consolidate)
← Statistical test functions from src/metrics/
```

### Patching Consolidation

**rv_toolkit/patching/activation.py**:
```
← src/core/patching.py (main)
← src/steering/activation_patching.py (merge variants)
← rv_toolkit/patching.py (move + consolidate)
← CANONICAL_CODE/causal_loop_closure_v2.py (validation patterns)
```

**rv_toolkit/patching/kv_cache.py**:
```
← src/steering/kv_cache.py (as-is, rename if needed)
```

### Prompt Bank

**rv_toolkit/prompts/bank.py**:
```
← CANONICAL_CODE/n300_mistral_test_prompt_bank.py (MOVE - this is the truth)
  (rename to bank.py, add loaders)
← prompts/deprecated/* (DROP - superseded by bank.py)
```

**rv_toolkit/prompts/__init__.py**:
```
Export: RECURSIVE_PROMPTS, BASELINE_PROMPTS, get_prompt_pairs()
(functionality from rv_toolkit/prompts.py)
```

### Experiments

**rv_toolkit/experiments/canonical/***:
```
← src/pipelines/canonical/*.py (MOVE as-is)
← Update imports to use rv_toolkit.* instead of src.*
```

**rv_toolkit/experiments/discovery/***:
```
← src/pipelines/discovery/*.py (MOVE as-is)
← Update imports to use rv_toolkit.* instead of src.*
```

### Tests

**tests/**:
```
← rv_toolkit/tests/*.py (MOVE to tests/)
← Create new test files for consolidated modules
← src/ has NO inline tests after migration
```

### Deletion (Archaeological Archive)

**DELETE entirely**:
```
archive/scripts/*.py (33KB of history - keep in git, remove from working tree)
CANONICAL_CODE/mistral_L27_FULL_VALIDATION.py (code logic moved to metrics/patching)
CANONICAL_CODE/causal_loop_closure_v2.py (merged into patching/activation.py)
rv_toolkit/prompts.py (REPLACED by prompts/bank.py + loaders)
src/pipelines/archive/*.py (historical - keep in git)
scripts/runpod/ (historical)
```

### Deprecate (Move to root-level docs/)

**mv → docs/**:
```
All markdown at root → docs/
PHASE_1C_ANALYSIS.ipynb → docs/notebooks/
THE_GEOMETRY_OF_RECURSION_MASTER*.ipynb → docs/notebooks/
All agent_reviews → docs/reviews/
```

### Root Level (Keep Only Entry Points)

**Keep at root**:
```
README.md (project overview)
CLAUDE.md (from ~/CLAUDE.md - project context)
LICENSE
pyproject.toml (package config)
requirements.txt
.gitignore
.github/ (CI/CD if added)
```

**DELETE from root**:
```
gemma_*.py (all 8 - move to results/ or archive in git)
neurips_n300_robust_experiment.py (archive)
openclaw_quickstart.py (move to docs/tutorials/)
reproduce_results.py (move to scripts/reproduce.py)
L4transmissionTEST001.1.ipynb (move to docs/notebooks/)
```

---

## Implementation Sequence

### Phase 1: Backup & Validation (10 min)
```bash
# Create branch
git checkout -b restructure/rv-toolkit-consolidation
git log --oneline | head -5  # Verify history preserved

# List all Python files
find . -name "*.py" -type f | wc -l  # Should show ~170

# Validate imports work before changes
python -m pytest rv_toolkit/tests/ --co  # Don't run, just collect
python -c "from src.pipelines.canonical import *"
```

### Phase 2: Directory Creation (2 min)
```bash
mkdir -p rv_toolkit/{core,metrics,patching,prompts,experiments/{canonical,discovery}}
mkdir -p tests/{test_metrics,test_patching,test_prompts,test_experiments}
mkdir -p docs
mkdir -p scripts
```

### Phase 3: Core Migration (20 min)

**Step 3a: Metrics**
```bash
# Move/consolidate metrics
cp src/metrics/rv.py rv_toolkit/metrics/
cp src/metrics/behavior_*.py rv_toolkit/metrics/behavior.py (consolidate)
cp src/metrics/logit_*.py rv_toolkit/metrics/logit.py (consolidate)
# ... other metrics consolidations
```

**Step 3b: Patching**
```bash
cp src/core/patching.py rv_toolkit/patching/activation.py
cp src/steering/kv_cache.py rv_toolkit/patching/
# Merge rv_toolkit/patching.py into activation.py
```

**Step 3c: Core Utilities**
```bash
cp src/core/{models,hooks,utils}.py rv_toolkit/core/
# Consolidate src/utils/*.py into core/utils.py
```

**Step 3d: Prompts (CRITICAL)**
```bash
# Move the prompt bank - THE GOLDEN FILE
cp CANONICAL_CODE/n300_mistral_test_prompt_bank.py rv_toolkit/prompts/bank.py
# Create loaders
touch rv_toolkit/prompts/loaders.py
touch rv_toolkit/prompts/validators.py
```

### Phase 4: Experiments (15 min)

```bash
# Move canonical experiments
cp -r src/pipelines/canonical/* rv_toolkit/experiments/canonical/

# Move discovery experiments
cp -r src/pipelines/discovery/* rv_toolkit/experiments/discovery/

# UPDATE ALL IMPORTS in moved files:
# Find: from src.metrics import
# Replace: from rv_toolkit.metrics import
# Find: from src.core import
# Replace: from rv_toolkit.core import
# etc.
```

### Phase 5: Tests (10 min)

```bash
# Move existing tests
cp -r rv_toolkit/tests/* tests/
rm -rf rv_toolkit/tests/

# Create test structure matching source
# tests/test_metrics/, tests/test_patching/, etc.
```

### Phase 6: Documentation & Entry Points (10 min)

```bash
# Move docs
mv *.md docs/  # Except README.md at root
mv *.ipynb docs/notebooks/

# Create scripts/ entry points
touch scripts/run_experiment.py
touch scripts/analyze_results.py
touch scripts/validate_reproducibility.py

# Create docs/README.md
touch docs/README.md
```

### Phase 7: Cleanup (5 min)

```bash
# Remove src/ entirely (pipelines, metrics, core all moved)
rm -rf src/

# Remove old locations
rm -rf CANONICAL_CODE/
rm -rf archive/scripts/
rm -rf rv_toolkit/prompts.py  # Old file, replaced by module
rm -rf scripts/runpod/

# Root cleanup
rm -f gemma_*.py neurips_*.py openclaw_quickstart.py
rm -f PHASE_1C_ANALYSIS.ipynb
```

### Phase 8: Update Imports (20 min)

**In rv_toolkit/__init__.py**:
```python
# Update to export from new locations
from .metrics import compute_rv, compute_participation_ratio, ...
from .patching import ActivationPatcher, ...
from .prompts import RECURSIVE_PROMPTS, BASELINE_PROMPTS, ...
from .experiments.canonical import rv_l27_causal_validation, ...
```

**In pyproject.toml**:
```toml
[project.scripts]
rv-toolkit = "rv_toolkit.cli:main"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

**In all moved pipeline files**:
```python
# Search-replace all:
# from src.metrics → from rv_toolkit.metrics
# from src.core → from rv_toolkit.core
# from src.steering → from rv_toolkit.patching
```

### Phase 9: Validate Structure (10 min)

```bash
# Test imports work
python -c "from rv_toolkit import compute_rv; print('OK')"
python -c "from rv_toolkit.experiments.canonical import rv_l27_causal_validation; print('OK')"
python -c "from rv_toolkit.prompts import RECURSIVE_PROMPTS; print('OK')"

# Verify package installs
pip install -e .

# Run tests
pytest tests/ -v --tb=short

# Check git status
git status  # Should show clear additions + deletions, no conflicts
git diff --stat  # Shows magnitude of changes
```

### Phase 10: Single Commit (5 min)

```bash
git add -A
git commit -m "refactor: consolidate rv_toolkit into single coherent structure

- Merge metrics from src/metrics/ and rv_toolkit/ into rv_toolkit/metrics/
- Consolidate patching implementations into rv_toolkit/patching/
- Move CANONICAL_CODE prompt bank to rv_toolkit/prompts/ (single source of truth)
- Relocate experiments from src/pipelines/ to rv_toolkit/experiments/
- Move tests to unified tests/ directory structure
- Remove archaeological artifacts (archive/scripts, old prompts/)
- Delete src/ directory (code relocated, not deleted)
- Modernize project structure for publication

This is a pure refactoring with zero functional changes. All logic is preserved.
Git history is available for archaeological purposes.

Verification:
- All imports updated to use rv_toolkit namespace
- Package installs without errors: pip install -e .
- Test suite passes: pytest tests/
- CLI works: rv-toolkit --help
- Can run canonical experiments via new structure"
```

---

## Validation Checklist

Before committing, verify:

- [ ] All Python files accounted for (no orphaned imports)
- [ ] Package imports without errors: `python -c "import rv_toolkit; print(rv_toolkit.__version__)"`
- [ ] CLI works: `rv-toolkit --help`
- [ ] Tests pass: `pytest tests/ -x` (stop on first failure)
- [ ] Prompt bank is accessible: `python -c "from rv_toolkit.prompts import RECURSIVE_PROMPTS; print(len(RECURSIVE_PROMPTS))"`
- [ ] One canonical experiment runs: `python -m rv_toolkit.experiments.canonical.rv_l27_causal_validation --dry-run`
- [ ] No broken imports in experiments: Check all imports in moved files
- [ ] Git diff shows coherent change (no weird diffs)
- [ ] No sensitive files exposed (configs, secrets stay in place)

---

## Post-Restructuring Maintenance

### New Workflow

**Adding an experiment**:
```python
# Create rv_toolkit/experiments/discovery/my_experiment.py
# Automatically importable as:
from rv_toolkit.experiments.discovery import my_experiment
```

**Adding a metric**:
```python
# Create rv_toolkit/metrics/my_metric.py
# Export in rv_toolkit/metrics/__init__.py
from rv_toolkit.metrics import my_metric
```

**Adding a prompt**:
```python
# Add to rv_toolkit/prompts/bank.py
# Access via:
from rv_toolkit.prompts import RECURSIVE_PROMPTS
prompts = rv_toolkit.prompts.loaders.get_prompts_by_category("L4")
```

### Documentation

After restructuring, create:
1. `docs/architecture.md` - Explain new structure
2. `docs/metrics.md` - Each metric documented
3. `docs/patching.md` - Each patching variant documented
4. `docs/experiments.md` - How to add/run experiments
5. Update `README.md` with new import examples

### Archival

Keep in `.git/` history:
- `archive/scripts/` (use `git log -p` to retrieve if needed)
- `src/` (use `git show HEAD~1:src/core/models.py` if needed)
- Old configs (preserved in git, not in working tree)

Nothing is lost - just reorganized for coherence.

---

## Risk Assessment

### Low Risk Areas
- Moving files (git preserves history)
- Consolidating similar code (adding to existing modules)
- Creating new directory structure (doesn't break existing code)

### Medium Risk Areas
- Import path changes (must update all references)
- Prompt bank relocation (must be absolute path-safe in loaders)
- Experiment relocation (must update relative imports)

### Mitigation Strategy
1. **Incremental validation**: Test after each phase, not at the end
2. **Automated import checking**: Script to find all `from src.*` statements
3. **Backwards compatibility**: Keep old imports working during transition (if needed)
4. **Git history preservation**: All code preserved in git, nothing deleted irrevocably

---

## Success Criteria

After restructuring, the repository should satisfy:

1. **Single Namespace**: Everything importable from `rv_toolkit.*`
2. **Clear Hierarchy**:
   - `core/` = primitives (models, hooks, utilities)
   - `metrics/` = all measurements
   - `patching/` = all circuit/activation manipulation
   - `prompts/` = single source of truth for prompts
   - `experiments/` = canonical & discovery pipelines
3. **Discoverable**: `import rv_toolkit; help(rv_toolkit)` shows full API
4. **Publishable**: No broken imports, all tests pass, CLI works
5. **Maintainable**: New experiments/metrics follow clear patterns
6. **Documented**: Each module has docstrings and examples

---

## Appendix: Import Refactoring Script

```python
#!/usr/bin/env python3
"""Find all imports that need updating."""
import re
from pathlib import Path

patterns = [
    (r"from src\.", "from rv_toolkit."),
    (r"import src\.", "import rv_toolkit."),
    (r'"src/', '"rv_toolkit/'),
    (r"'src/", "'rv_toolkit/"),
]

for pyfile in Path("rv_toolkit").rglob("*.py"):
    content = pyfile.read_text()
    updated = content
    for old, new in patterns:
        updated = re.sub(old, new, updated)
    if updated != content:
        print(f"UPDATE: {pyfile}")
        pyfile.write_text(updated)
```

---

## Timeline

- **Phase 1-10**: ~90 minutes total (can be run sequentially)
- **Validation**: ~15 minutes
- **One atomic commit**: 5 minutes

**Total wall time**: ~2 hours for clean, publication-ready structure.

---

*Architecture Review: Architecture Consolidation Phase*
*Target: Single coherent namespace for 170+ scattered Python files*
*Status: Design Complete - Ready for Implementation*
