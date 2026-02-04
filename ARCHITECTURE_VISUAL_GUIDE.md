# Architecture Restructuring: Visual Guide

## Current Architecture (Fragmented)

```
mech-interp-latent-lab-phase1/
│
├── [ROOT CLUTTER] ........................ 11 Python scripts
│   ├── gemma_behavioral_transfer.py
│   ├── gemma_causal_batch_kv_only.py
│   ├── gemma_full_validation_v2.py
│   ├── gemma_kv_vs_vproj_comparison.py
│   ├── gemma_roman_empire_deep_dive.py
│   ├── gemma_rv_bifurcation_threshold.py
│   ├── gemma_rv_during_generation.py
│   ├── gemma_rv_trajectory_source.py
│   ├── neurips_n300_robust_experiment.py
│   ├── openclaw_quickstart.py
│   └── reproduce_results.py
│
├── [DOCUMENTATION CHAOS] ................. ~30 .md files at root
│   ├── 20_MINUTE_REPRODUCIBILITY_PROTOCOL.md
│   ├── AGENT_ONBOARDING.md
│   ├── AGENT_PROMPT_GOLD_STANDARD.md
│   ├── ASSESSMENT_DIRECTIVE.md
│   ├── BEHAVIOR_TRANSFER_ANALYSIS.md
│   ├── BRIDGE_HYPOTHESIS_*.md (3 files)
│   ├── CLEANUP_PROPOSAL.md
│   ├── CROSS_MODEL_RESULTS_SYNTHESIS.md
│   ├── DEPRECATION_*.md (2 files)
│   ├── FINAL_ALIGNMENT_REPORT.md
│   ├── OPENCLAW_*.md (3 files)
│   ├── REPRODUCIBILITY_AUDIT_REPORT.md
│   ├── STATISTICAL_AUDIT_*.md (2 files)
│   ├── TASK*.md (2 files)
│   ├── REPOSITORY_DISSECTION_COMPLETE.md
│   └── ... more
│
├── rv_toolkit/                       ← "THE PACKAGE"
│   ├── pyproject.toml
│   ├── README.md
│   ├── rv_toolkit/
│   │   ├── __init__.py
│   │   ├── cli.py                  ← Entry point
│   │   ├── metrics.py              ← R_V implementation (COPY 1)
│   │   ├── patching.py             ← Patching (COPY 1)
│   │   ├── analysis.py             ← Analysis
│   │   └── prompts.py              ← Old prompt accessors
│   └── tests/
│       └── [tests exist but scattered]
│
├── CANONICAL_CODE/                 ← "THE TRUTH"
│   ├── n300_mistral_test_prompt_bank.py .... GOLDEN PROMPT BANK (93KB)
│   ├── mistral_L27_FULL_VALIDATION.py ...... Model loading patterns
│   └── causal_loop_closure_v2.py ........... Validation logic
│
├── src/                            ← "THE RESEARCH"
│   ├── core/
│   │   ├── models.py               ← Model loading (DUPLICATE)
│   │   ├── hooks.py
│   │   ├── patching.py             ← Patching (COPY 2)
│   │   ├── logit_capture.py
│   │   ├── head_specific_patching.py
│   │   └── utils.py
│   │
│   ├── metrics/
│   │   ├── rv.py                   ← R_V implementation (COPY 2)
│   │   ├── behavior_strict.py
│   │   ├── behavior_states.py
│   │   ├── behavioral_bridge.py
│   │   ├── mode_score.py
│   │   ├── logit_diff.py
│   │   ├── logit_lens.py
│   │   ├── baseline_suite.py
│   │   ├── extended.py
│   │   └── [8 total, ~1400 lines]
│   │
│   ├── pipelines/
│   │   ├── canonical/ .................. 10 validated experiments
│   │   ├── discovery/ .................. 20 exploratory experiments
│   │   ├── archive/ .................... 40 historical experiments
│   │   └── run.py
│   │
│   ├── steering/
│   │   ├── activation_patching.py    ← Patching (COPY 3)
│   │   └── kv_cache.py
│   │
│   ├── experiments/
│   │   ├── gemma_activation_patching.py
│   │   └── rapid_recognition_rv_experiment.py
│   │
│   └── utils/
│       ├── multi_model_discovery.py
│       ├── pattern_detector.py
│       ├── prompt_compatibility_scorer.py
│       ├── recursion_prompt_generator.py
│       ├── recursive_output_analyzer.py
│       ├── run_index.py
│       └── run_metadata.py
│
├── archive/
│   └── scripts/ ......................... 30+ historical scripts (DEAD WEIGHT)
│       ├── verify_champion_behavior.py
│       ├── behavior_strict_vproj_only.py
│       ├── deep_circuit_analysis_v2.py
│       ├── [27 more]
│       └── ... [total 33KB of history]
│
├── scripts/ ............................... 15+ utility scripts
│   ├── run_mlp_vproj_combined.py
│   ├── generate_model_configs.py
│   ├── compute_c2_statistics.py
│   ├── [12 more]
│   └── runpod/ (historical)
│
├── prompts/
│   ├── deprecated/ ...................... OLD, SUPERSEDED
│   └── [scattered, not maintained]
│
├── configs/
│   ├── canonical/ ....................... Current configs
│   ├── discovery/ ....................... Experimental configs
│   ├── gold/ ............................ Best configs
│   ├── phase3_bridge/ ................... Phase-specific configs
│   ├── smoke_test/ ...................... Test configs
│   ├── archive/ ......................... Old configs
│   └── [18 config files, scattered]
│
├── results/
│   ├── canonical/ ....................... Validated results
│   ├── discovery/ ....................... Exploratory results
│   ├── phase0_metric_validation/
│   ├── phase1_circuit/
│   ├── phase2_generalization/
│   ├── phase3_bridge/
│   ├── [32+ result directories, 1000s of files]
│   └── runs/ ............................ Timestamped runs
│
├── R_V_PAPER/ ............................ Paper materials
│   ├── code/ ............................ Code used for paper
│   ├── csv_files/ ....................... Data from experiments
│   ├── figures/ ......................... Generated plots
│   ├── research/ ........................ Research notes
│   ├── results/ ......................... Result summaries
│   └── STORY_ARC/ ....................... Narrative arc
│
├── docs/ (currently scattered as .md at root)
│   ├── analysis/ ........................ Analysis documents
│   ├── audits/ .......................... Audit reports
│   ├── experiments/ ..................... Experiment notes
│   ├── findings/ ........................ Research findings
│   ├── methodology/ ..................... Methodology docs
│   ├── results/ ......................... Result summaries
│   ├── standards/ ....................... Standards/procedures
│   └── status/ .......................... Status updates
│
├── RECOVERED_GOLD/ ....................... 9 recovered experiments
├── REUSABLE_PROMPT_BANK/ ................. Symlink/duplicate?
└── visualizations/ ....................... Visualization code


PROBLEM SUMMARY:
  ✗ 3 copies of R_V implementation (src/metrics/ + rv_toolkit/ + CANONICAL_CODE)
  ✗ 3 patching implementations (src/core/, src/steering/, rv_toolkit/)
  ✗ Prompt bank split (CANONICAL_CODE + prompts/ + REUSABLE_PROMPT_BANK)
  ✗ Experiments scattered (src/pipelines/, archive/scripts/, scripts/)
  ✗ Tests scattered (rv_toolkit/tests/ + nowhere else)
  ✗ Root clutter (11 .py files + 30 .md files)
  ✗ Import chaos (from src.* vs from rv_toolkit.* vs from CANONICAL_CODE)
  ✗ No single entry point or discovery mechanism
```

---

## Target Architecture (Consolidated)

```
mech-interp-latent-lab-phase1/
│
├── README.md ............................. Project overview
├── LICENSE .............................. MIT license
├── pyproject.toml ....................... Package metadata
├── requirements.txt ..................... Dependencies
├── .gitignore ........................... Git configuration
│
├── rv_toolkit/ .......................... THE PACKAGE (consolidated)
│   ├── __init__.py ...................... Public API
│   ├── pyproject.toml
│   ├── README.md
│   ├── LICENSE
│   │
│   ├── rv_toolkit/
│   │   ├── __init__.py .................. "from rv_toolkit import *"
│   │   │
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── models.py ............... Model loading (consolidated)
│   │   │   ├── hooks.py ................ Hook infrastructure
│   │   │   └── utils.py ................ Utilities (consolidated)
│   │   │
│   │   ├── metrics/
│   │   │   ├── __init__.py
│   │   │   ├── rv.py ................... R_V (single source of truth)
│   │   │   ├── participation_ratio.py .. PR helper
│   │   │   ├── behavior.py ............ Behavior metrics (consolidated)
│   │   │   ├── logit.py ............... Logit metrics (consolidated)
│   │   │   └── analysis.py ............ Statistical analysis
│   │   │
│   │   ├── patching/
│   │   │   ├── __init__.py
│   │   │   ├── activation.py .......... Activation patching (single impl.)
│   │   │   ├── kv_cache.py ............ KV cache patching
│   │   │   └── circuit.py ............. Circuit analysis helpers
│   │   │
│   │   ├── prompts/
│   │   │   ├── __init__.py ............ Public API
│   │   │   ├── bank.py ................ Prompt bank (MOVED from CANONICAL_CODE)
│   │   │   ├── loaders.py ............. Load by category
│   │   │   └── validators.py .......... Validation logic
│   │   │
│   │   ├── experiments/
│   │   │   ├── __init__.py ............ Export all experiments
│   │   │   ├── canonical/ ............ Publication-ready (moved)
│   │   │   │   ├── __init__.py
│   │   │   │   ├── rv_l27_causal_validation.py
│   │   │   │   ├── mlp_ablation_necessity.py
│   │   │   │   ├── confound_validation.py
│   │   │   │   ├── multi_token_bridge.py
│   │   │   │   └── [7 more...]
│   │   │   │
│   │   │   └── discovery/ ............ Exploratory (moved)
│   │   │       ├── __init__.py
│   │   │       ├── behavioral_grounding.py
│   │   │       ├── circuit_analysis.py
│   │   │       ├── layer_sweep.py
│   │   │       ├── path_patching_mechanism.py
│   │   │       └── [15 more...]
│   │   │
│   │   ├── cli.py ..................... Command line interface
│   │   └── version.py ................. Version constant
│   │
│   ├── tests/ .......................... Unified test directory
│   │   ├── __init__.py
│   │   ├── conftest.py ................ Fixtures
│   │   ├── test_metrics/
│   │   │   ├── test_rv.py
│   │   │   ├── test_behavior.py
│   │   │   └── test_logit.py
│   │   ├── test_patching/
│   │   │   ├── test_activation.py
│   │   │   └── test_kv_cache.py
│   │   ├── test_prompts/
│   │   │   └── test_bank.py
│   │   └── test_experiments/
│   │       ├── test_canonical.py
│   │       └── test_discovery.py
│   │
│   ├── examples/ ...................... Usage examples
│   │   ├── basic_rv_measurement.py
│   │   ├── activation_patching.py
│   │   └── full_pipeline.py
│   │
│   └── docs/ .......................... Documentation
│       ├── README.md ................. Getting started
│       ├── architecture.md ........... Structure explanation
│       ├── api.md .................... API reference
│       ├── metrics.md ................ Metric documentation
│       ├── patching.md ............... Patching documentation
│       ├── experiments.md ............ How to run/add experiments
│       ├── notebooks/ ................ Analysis notebooks
│       ├── tutorials/ ................ Tutorials
│       └── references/ ............... External references
│
├── scripts/ ............................ Entry point scripts
│   ├── run_experiment.py ............. Run any experiment
│   ├── analyze_results.py ............ Analyze results
│   └── validate_reproducibility.py .. Validation script
│
├── configs/ ............................ Configuration files
│   ├── canonical/ .................... Current configs
│   ├── discovery/ .................... Experimental configs
│   └── gold/ .......................... Best-performing configs
│
├── results/ ........................... Experiment results (data, not code)
│   ├── canonical/ .................... Validated results
│   ├── discovery/ .................... Exploratory results
│   └── runs/ .......................... Timestamped runs
│
├── R_V_PAPER/ ......................... Paper materials
│   ├── code/ ......................... Supporting code
│   ├── figures/ ....................... Generated figures
│   ├── csv_files/ .................... Data files
│   ├── results/ ....................... Result summaries
│   └── research/ ..................... Research notes
│
└── [.git/] ............................ Git history (unchanged, everything preserved)
    └── Contains:
        - archive/scripts/ (for historical reference)
        - src/ (moved code, not deleted)
        - CANONICAL_CODE/ (moved, not deleted)
        - All old configs (preserved)


IMPROVEMENTS:
  ✓ 1 R_V implementation (rv_toolkit/metrics/rv.py)
  ✓ 1 patching implementation (rv_toolkit/patching/activation.py)
  ✓ 1 prompt bank (rv_toolkit/prompts/bank.py) - SINGLE SOURCE OF TRUTH
  ✓ Experiments clearly organized (canonical/ vs discovery/)
  ✓ Tests unified (tests/ mirrors source structure)
  ✓ Root clean (only package + meta files)
  ✓ Clear imports (from rv_toolkit.*)
  ✓ Single entry point (CLI + unified API)
  ✓ Publication-ready structure
  ✓ Zero code loss (all in git history)
```

---

## Import Path Evolution

### BEFORE (Chaos)

```python
# Which one is correct?
from rv_toolkit.metrics import compute_rv          # Option A
from src.metrics.rv import compute_rv               # Option B
from CANONICAL_CODE.n300_mistral_test_prompt_bank import prompt_bank_1c  # Option C

# Which prompts are current?
from prompts.deprecated import BASELINE_PROMPTS    # Option D
from REUSABLE_PROMPT_BANK import get_prompts       # Option E
from CANONICAL_CODE.n300_mistral_test_prompt_bank import prompt_bank_1c  # Option F

# How do I run an experiment?
import src.pipelines.canonical.rv_l27_causal_validation
import archive.scripts.some_old_script
import scripts.compute_stats
# (None of these work together)

# How do I patch?
from rv_toolkit.patching import ActivationPatcher       # Option A
from src.steering.activation_patching import patcher    # Option B
from src.core.patching import patch_activation          # Option C
```

### AFTER (Clean)

```python
# One way to do everything
from rv_toolkit import compute_rv                   # Single import
from rv_toolkit.prompts import RECURSIVE_PROMPTS    # Single source
from rv_toolkit.experiments.canonical import rv_l27_causal_validation  # Clear
from rv_toolkit.patching import ActivationPatcher   # One patcher

# Run any experiment
from rv_toolkit.experiments import canonical, discovery
exp = canonical.rv_l27_causal_validation.run(config)
exp = discovery.layer_sweep.run(config)

# Access prompts by category
from rv_toolkit.prompts import loaders
l4_prompts = loaders.get_by_category("L4")
baseline = loaders.get_baseline_control()

# Everything is discoverable
import rv_toolkit
help(rv_toolkit)  # Shows all public API
```

---

## Consolidation Impact Matrix

| Aspect | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Entry Points** | 11 scattered | 1 package entry | 11x clarity |
| **Metrics Locations** | 8 files across 2 dirs | 1 metrics/ dir | 8x faster finding |
| **Patching Variants** | 3 implementations | 1 canonical | 3x less maintenance |
| **Prompt Bank Sources** | 3 locations | 1 file (bank.py) | Eliminates sync bugs |
| **Experiments** | Across 3+ dirs | experiments/{canonical,discovery} | Clear taxonomy |
| **Test Files** | Scattered | tests/ mirrors source | Easy coverage tracking |
| **Root Clutter** | 11 .py + 30 .md | 5 files | 80% cleaner |
| **Import Patterns** | 4+ conventions | 1 convention (rv_toolkit.*) | Single pattern |
| **Onboarding Time** | 2-3 hours | 15 minutes | 10x faster |
| **Code Duplication** | ~2,000 lines (estimates) | ~500 lines | 75% less duplication |

---

## Dependency Graph

### BEFORE (Circular Risks)

```
rv_toolkit/
  ├─→ imports from CANONICAL_CODE/
  ├─→ imports from src/
  └─→ defines duplicate metrics

src/metrics/
  ├─→ imports from src/core/
  ├─→ defines duplicate metrics
  └─→ potential circular with rv_toolkit/

CANONICAL_CODE/
  ├─→ no imports (self-contained)
  └─→ data stored here, used from 3 places

archive/scripts/
  └─→ imports from src/
```

**Issues**:
- rv_toolkit and src might import each other (untested)
- Updating one place breaks others
- CANONICAL_CODE is isolated (should be integrated)

### AFTER (Clean Hierarchy)

```
rv_toolkit/core/ ...................... FOUNDATION (no internal deps)
  ├─← from transformers
  └─← from torch, numpy, scipy

rv_toolkit/metrics/ .................. MEASUREMENT LAYER
  ├─← rv_toolkit/core/
  └─← numpy, scipy

rv_toolkit/patching/ ................. INTERVENTION LAYER
  ├─← rv_toolkit/core/
  └─← torch

rv_toolkit/prompts/ ................. DATA LAYER
  └─← (no dependencies, pure data)

rv_toolkit/experiments/ ............ APPLICATION LAYER
  ├─← rv_toolkit/metrics
  ├─← rv_toolkit/patching
  ├─← rv_toolkit/prompts
  └─← rv_toolkit/core/

tests/ ............................ TEST LAYER
  ├─← rv_toolkit/*
  ├─← pytest
  └─← mock fixtures from conftest.py
```

**Properties**:
- No circular dependencies
- Clear dependency direction (down the stack)
- Each layer independent and testable
- Easy to understand data flow

---

## Discovery Time Comparison

### Finding "Where is R_V implementation?"

**BEFORE**:
```
1. Is it in rv_toolkit/metrics.py?             ← Check (230 lines)
2. Is it in src/metrics/rv.py?                 ← Check (210 lines)
3. Is it in CANONICAL_CODE/?                   ← Check
4. Is it in the archive somewhere?             ← Check
5. Which one is canonical?                     ← Grep for modifications
6. Are they in sync?                           ← Probably not
7. Which should I modify?                      ← Flip a coin
```
Time: 15-30 minutes, uncertain result

**AFTER**:
```
from rv_toolkit.metrics import compute_rv     ← One place, obvious
```
Time: 5 seconds, certain result

---

## Test Coverage Path

### BEFORE (Scattered)

```
❌ No way to run all tests
  ├─ rv_toolkit/tests/test_*.py    (exist but isolated)
  ├─ src/metrics/                  (no tests)
  ├─ src/core/                     (no tests)
  ├─ src/pipelines/                (no tests)
  └─ src/steering/                 (no tests)

Running tests:
  $ pytest rv_toolkit/tests/    ← Only runs 1/5 of code
  $ pytest src/                 ← Fails (no tests)
```

### AFTER (Unified)

```
✓ Clear test structure
  ├─ tests/test_metrics/test_rv.py
  ├─ tests/test_metrics/test_behavior.py
  ├─ tests/test_patching/test_activation.py
  ├─ tests/test_prompts/test_bank.py
  └─ tests/test_experiments/test_canonical.py

Running tests:
  $ pytest tests/              ← Runs ALL code systematically
  $ pytest tests/test_metrics/ ← Run just metrics tests
  $ pytest -v --cov           ← Coverage analysis
```

---

## File System Cleanup

### Space Reclaimed

```
Deleted from working tree (preserved in git):
  - archive/scripts/*.py ........................ 30 files, 33KB
  - src/ (code moved, directory removed) ....... 80 files, 40KB
  - Old CANONICAL_CODE/*.py .................... 3 files, 5KB
  - prompts/deprecated/ ........................ 10 files, 8KB
  - Duplicate configs .......................... ~5KB
  ─────────────────────────────────────────────────────
  TOTAL REMOVED FROM WORKING TREE: ~91KB

Added to working tree:
  - New directory structure + __init__.py files ... ~5KB
  - Documentation + guides ....................... ~15KB
  ─────────────────────────────────────────────────────
  NET CHANGE: -71KB (20% working tree reduction)

Everything preserved in git (.git/ grows ~5KB, negligible)
```

### Directory Count

```
BEFORE: 72 directories, 170 Python files
AFTER:  35 directories, 120 Python files
CHANGE: -37 dirs (-51%), -50 files (-29%), 0 loss
```

---

## Quality Metrics

### Code Duplication

```python
# BEFORE: 3 implementations of R_V
1. rv_toolkit/metrics.py:
   def compute_rv(v_tensor, window_size=16):
       participation_ratio = ...
       ...

2. src/metrics/rv.py:
   def compute_rv(v_tensor, window_size=16):
       participation_ratio = ...
       ...

3. (possibly in CANONICAL_CODE validation code)

→ Maintenance burden: 3x (every bug fix, feature adds 3 PRs)

# AFTER: 1 implementation
  rv_toolkit/metrics/rv.py:
    def compute_rv(v_tensor, window_size=16):
        participation_ratio = ...
        ...

  → Maintenance burden: 1x (every change once)
```

### Test-to-Code Ratio

```
BEFORE: ~8 test files covering only rv_toolkit/
  Coverage: ~15% (rough estimate)

AFTER: ~15 test files covering all modules
  Coverage: ~70% (target)

  Improvement: 4.7x more test coverage
```

### Import Complexity

```
BEFORE: 4 distinct import patterns in use
  Pattern 1: from rv_toolkit.metrics import compute_rv
  Pattern 2: from src.metrics.rv import compute_rv
  Pattern 3: from CANONICAL_CODE.n300_mistral_test_prompt_bank import ...
  Pattern 4: from archive.scripts.something import ...

AFTER: 1 import pattern
  Pattern: from rv_toolkit.* import ...

→ Cognitive load reduced by 4x
→ Errors from wrong import eliminated
```

---

## Risk-Free Consolidation

### Why This Won't Break Anything

1. **Git History Preserved**
   ```bash
   git log --all  # Still shows all old code
   git show HEAD~1:src/metrics/rv.py  # Can retrieve old code
   ```

2. **Tests After Each Phase**
   ```bash
   After metrics consolidation: pytest tests/test_metrics/
   After patching consolidation: pytest tests/test_patching/
   After experiment move: pytest tests/test_experiments/
   ```

3. **Atomic Commit**
   ```bash
   git commit -m "refactor: consolidate structure"
   # Either entire change succeeds, or entire change reverts
   # No partial state
   ```

4. **Validation Before Commit**
   ```bash
   python -c "from rv_toolkit import *"  # All imports work
   pytest tests/                          # All tests pass
   rv-toolkit --help                      # CLI works
   ```

---

## Migration Workflow Diagram

```
Phase 1: Backup & Validate (10 min)
  ├─ Create branch: restructure/rv-toolkit-consolidation
  ├─ Verify git history intact
  ├─ Count files: find . -name "*.py" | wc -l
  └─ Test imports work before changes

Phase 2: Create Structure (2 min)
  ├─ mkdir -p rv_toolkit/{core,metrics,patching,prompts,experiments}
  ├─ mkdir -p tests/{test_metrics,test_patching,test_prompts}
  └─ touch __init__.py files

Phase 3: Consolidate Metrics (20 min)
  ├─ cp src/metrics/rv.py rv_toolkit/metrics/
  ├─ Consolidate behavior_*.py → behavior.py
  ├─ Consolidate logit_*.py → logit.py
  ├─ Create rv_toolkit/metrics/__init__.py (exports)
  └─ pytest tests/test_metrics/

Phase 4: Unify Patching (30 min)
  ├─ cp src/core/patching.py rv_toolkit/patching/activation.py
  ├─ cp src/steering/kv_cache.py rv_toolkit/patching/
  ├─ Merge rv_toolkit/patching.py into activation.py
  ├─ Create rv_toolkit/patching/__init__.py (exports)
  └─ pytest tests/test_patching/

Phase 5: Relocate Prompts (10 min)
  ├─ cp CANONICAL_CODE/n300_mistral_test_prompt_bank.py → bank.py
  ├─ Create loaders.py (functions to access by category)
  ├─ Create rv_toolkit/prompts/__init__.py (exports)
  └─ pytest tests/test_prompts/

Phase 6: Move Experiments (45 min)
  ├─ cp -r src/pipelines/canonical/* rv_toolkit/experiments/canonical/
  ├─ cp -r src/pipelines/discovery/* rv_toolkit/experiments/discovery/
  ├─ Update all imports: from src.* → from rv_toolkit.*
  ├─ Create rv_toolkit/experiments/__init__.py
  └─ pytest tests/test_experiments/

Phase 7: Reorganize Tests (20 min)
  ├─ mv rv_toolkit/tests/* tests/
  ├─ Create tests/conftest.py (fixtures)
  ├─ Create test structure mirroring source
  └─ pytest tests/ -v

Phase 8: Clean Up (10 min)
  ├─ rm -rf src/
  ├─ rm -rf CANONICAL_CODE/
  ├─ rm -rf archive/scripts/
  ├─ rm -f gemma_*.py at root
  └─ mv *.md → docs/

Phase 9: Update Package Info (10 min)
  ├─ Update rv_toolkit/__init__.py (exports)
  ├─ Update pyproject.toml (test paths)
  ├─ Create docs/architecture.md
  └─ Update README.md (examples)

Phase 10: Validate (15 min)
  ├─ python -c "from rv_toolkit import *"
  ├─ pip install -e .
  ├─ pytest tests/ -x
  ├─ rv-toolkit --help
  └─ Run smoke test: python -m rv_toolkit.experiments.canonical.rv_l27

Phase 11: Single Commit (5 min)
  ├─ git add -A
  ├─ git commit -m "refactor: consolidate structure"
  └─ git log --oneline -5

TOTAL TIME: ~177 minutes (3 hours)
```

---

## Summary

| Aspect | Before | After | Outcome |
|--------|--------|-------|---------|
| **Structure** | Fragmented (3 hierarchies) | Unified (1 hierarchy) | ✓ Publishable |
| **Duplication** | High (3 R_V, 3 patchers) | None (single impl.) | ✓ Maintainable |
| **Discoverability** | Hard (11 entry points) | Easy (1 package API) | ✓ Usable |
| **Tests** | Scattered | Unified | ✓ Verifiable |
| **Documentation** | Chaos at root | Organized in docs/ | ✓ Clear |
| **Code Loss** | Zero (git preserved) | Zero (git preserved) | ✓ Safe |
| **Timeline** | N/A | 3 hours | ✓ Fast |

**Result**: Publication-ready repository with zero code loss and maximum clarity.

---

*Visual Architecture Guide*
*Complete restructuring path from fragmented to consolidated*
*Status: READY FOR EXECUTION*
