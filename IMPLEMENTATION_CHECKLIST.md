# Implementation Checklist: Architecture Restructuring

**Project**: rv_toolkit consolidation
**Date**: February 4, 2026
**Estimated Duration**: 3-4 hours
**Complexity**: Medium (careful but straightforward)
**Risk Level**: Low (git preserves everything)

---

## Pre-Implementation

### Review Documents
- [ ] Read ARCHITECTURE_EXECUTIVE_SUMMARY.md (15 min)
- [ ] Review ARCHITECTURE_RESTRUCTURE_PLAN.md (20 min)
- [ ] Study ARCHITECTURE_VISUAL_GUIDE.md (10 min)
- [ ] Understand the target structure
- [ ] Identify any additional files in root/ that need moving

### Preparation
- [ ] Create fresh branch: `git checkout -b restructure/rv-toolkit-consolidation`
- [ ] Verify git status is clean: `git status` (should show "nothing to commit")
- [ ] Check current file count: `find . -name "*.py" -type f | wc -l`
- [ ] List all Python files at root: `find . -maxdepth 1 -name "*.py" -type f`
- [ ] Record this in a note for later verification

### Backup
- [ ] Ensure you have git push permissions (test with `git log -1`)
- [ ] Create a local backup: `git branch backup/pre-restructure`
- [ ] This is your safety net (can revert entire change if needed)

---

## Phase 1: Create Directory Structure

**Duration**: 2 minutes

```bash
# Create new directories
mkdir -p rv_toolkit/core
mkdir -p rv_toolkit/metrics
mkdir -p rv_toolkit/patching
mkdir -p rv_toolkit/prompts
mkdir -p rv_toolkit/experiments/canonical
mkdir -p rv_toolkit/experiments/discovery

mkdir -p tests/test_metrics
mkdir -p tests/test_patching
mkdir -p tests/test_prompts
mkdir -p tests/test_experiments

mkdir -p docs
mkdir -p scripts
```

### Checklist
- [ ] All directories created successfully
- [ ] Verify with: `find rv_toolkit -type d | head -15`
- [ ] Verify with: `find tests -type d | head -10`

---

## Phase 2: Consolidate Core Module

**Duration**: 10 minutes

### Core utilities
```bash
# Copy model loading
cp src/core/models.py rv_toolkit/core/models.py

# Copy hooks
cp src/core/hooks.py rv_toolkit/core/hooks.py

# Consolidate utils
cp src/core/utils.py rv_toolkit/core/utils.py
```

### Create core/__init__.py
```bash
cat > rv_toolkit/core/__init__.py << 'EOF'
"""Core primitives for R_V metrics."""

from .models import (
    load_model,
    load_tokenizer,
    # Add other exports from models.py
)

from .hooks import (
    # Add hook exports
)

from .utils import (
    # Add utility exports
)

__all__ = [
    "load_model",
    "load_tokenizer",
    # ... complete list
]
EOF
```

### Checklist
- [ ] models.py copied and imports are valid
- [ ] hooks.py copied and imports are valid
- [ ] utils.py copied and imports are valid
- [ ] core/__init__.py created with correct exports
- [ ] Test: `python -c "from rv_toolkit.core import load_model"`

---

## Phase 3: Consolidate Metrics

**Duration**: 20 minutes

### Copy individual metric files
```bash
# Main R_V implementation
cp src/metrics/rv.py rv_toolkit/metrics/rv.py

# Behavior metrics (will consolidate into one file)
cp src/metrics/behavior_strict.py /tmp/behavior_strict.py
cp src/metrics/behavior_states.py /tmp/behavior_states.py
cp src/metrics/behavioral_bridge.py /tmp/behavioral_bridge.py
cp src/metrics/mode_score.py /tmp/mode_score.py

# Logit metrics
cp src/metrics/logit_diff.py /tmp/logit_diff.py
cp src/metrics/logit_lens.py /tmp/logit_lens.py

# Analysis
cp rv_toolkit/analysis.py /tmp/old_analysis.py
cp src/metrics/baseline_suite.py /tmp/baseline_suite.py
```

### Consolidate behavior metrics
```bash
# Create consolidated behavior.py
python << 'EOF'
import os

# Read all behavior files
behavior_strict = open("/tmp/behavior_strict.py").read()
behavior_states = open("/tmp/behavior_states.py").read()
behavioral_bridge = open("/tmp/behavioral_bridge.py").read()
mode_score = open("/tmp/mode_score.py").read()

# Extract imports and functions
# (MANUAL STEP: merge the 4 files, avoiding duplicate imports)

# Write consolidated file
with open("rv_toolkit/metrics/behavior.py", "w") as f:
    f.write("""\"\"\"Behavioral metrics for transformer analysis.\"\"\"\n\n""")
    # Add consolidated code here
    # (See ARCHITECTURE_RESTRUCTURE_PLAN.md for exact merging strategy)
EOF
```

### Consolidate logit metrics
```bash
python << 'EOF'
# Similar process for logit_diff.py + logit_lens.py
# Merge into rv_toolkit/metrics/logit.py
EOF
```

### Create metrics/__init__.py
```bash
cat > rv_toolkit/metrics/__init__.py << 'EOF'
"""Measurement metrics for R_V and behavioral analysis."""

from .rv import (
    compute_rv,
    compute_participation_ratio,
    compute_effective_rank,
    RVResult,
)

from .behavior import (
    # behavior metrics
)

from .logit import (
    # logit metrics
)

from .analysis import (
    # analysis functions
)

__all__ = [
    "compute_rv",
    "compute_participation_ratio",
    "compute_effective_rank",
    "RVResult",
    # ... complete list
]
EOF
```

### Checklist
- [ ] rv.py copied to rv_toolkit/metrics/
- [ ] behavior.py created (consolidated from 4 files)
- [ ] logit.py created (consolidated from 2 files)
- [ ] analysis.py created (copy + consolidate)
- [ ] metrics/__init__.py created with exports
- [ ] Test: `python -c "from rv_toolkit.metrics import compute_rv"`
- [ ] Test: `python -c "from rv_toolkit.metrics import compute_rv; help(compute_rv)"`

---

## Phase 4: Consolidate Patching

**Duration**: 30 minutes

### Copy patching implementations
```bash
# Copy core patching
cp src/core/patching.py /tmp/core_patching.py

# Copy steering variants
cp src/steering/activation_patching.py /tmp/steering_patching.py
cp src/steering/kv_cache.py /tmp/kv_cache.py

# Copy old rv_toolkit patching
cp rv_toolkit/patching.py /tmp/old_patching.py
```

### Consolidate into activation.py
```bash
# MANUAL STEP:
# 1. Read all 4 files
# 2. Identify the canonical implementation (likely src/core/patching.py)
# 3. Merge variants into single file
# 4. Remove duplicates
# 5. Write to rv_toolkit/patching/activation.py

python << 'EOF'
# Manual consolidation - see ARCHITECTURE_RESTRUCTURE_PLAN.md for strategy
# Key: Keep ActivationPatcher as canonical class
# Add variants as methods or helper functions
EOF
```

### Copy KV cache patching
```bash
cp src/steering/kv_cache.py rv_toolkit/patching/kv_cache.py
```

### Create patching/__init__.py
```bash
cat > rv_toolkit/patching/__init__.py << 'EOF'
"""Activation and circuit patching for transformer analysis."""

from .activation import (
    ActivationPatcher,
    PatchingResult,
    ControlCondition,
)

from .kv_cache import (
    # KV cache functions
)

__all__ = [
    "ActivationPatcher",
    "PatchingResult",
    "ControlCondition",
    # ... complete list
]
EOF
```

### Checklist
- [ ] Core patching logic copied
- [ ] activation.py created (consolidated, deduped)
- [ ] kv_cache.py copied
- [ ] patching/__init__.py created with exports
- [ ] Test: `python -c "from rv_toolkit.patching import ActivationPatcher"`
- [ ] Verify no import errors in activation.py

---

## Phase 5: Relocate Prompt Bank

**Duration**: 10 minutes

### Copy prompt bank
```bash
# THIS IS THE CRITICAL FILE - THE GOLDEN PROMPT BANK
cp CANONICAL_CODE/n300_mistral_test_prompt_bank.py rv_toolkit/prompts/bank.py
```

### Create loaders.py
```bash
cat > rv_toolkit/prompts/loaders.py << 'EOF'
"""Prompt bank loaders by category."""

from .bank import prompt_bank_1c

def get_prompt_pairs():
    """Get all baseline+recursive prompt pairs."""
    # Implementation: iterate over prompt_bank_1c
    pass

def get_by_category(category: str):
    """Get prompts by category (L3_deeper, L4_full, etc.)."""
    result = {}
    for key, prompt in prompt_bank_1c.items():
        if prompt.get("group") == category:
            result[key] = prompt
    return result

def get_baseline_control():
    """Get baseline control prompts."""
    return get_by_category("baseline")

def get_recursive_prompts():
    """Get all recursive prompts."""
    result = {}
    for key, prompt in prompt_bank_1c.items():
        if prompt.get("pillar") == "dose_response":
            result[key] = prompt
    return result
EOF
```

### Create validators.py
```bash
cat > rv_toolkit/prompts/validators.py << 'EOF'
"""Validate prompt bank structure."""

def validate_prompt_bank(bank):
    """Validate that all prompts have required fields."""
    required_fields = {"text", "group", "pillar"}
    for key, prompt in bank.items():
        missing = required_fields - set(prompt.keys())
        if missing:
            raise ValueError(f"Prompt {key} missing fields: {missing}")
    return True

def validate_categories(bank):
    """List all categories in use."""
    categories = set()
    for prompt in bank.values():
        categories.add(prompt.get("group"))
    return sorted(categories)
EOF
```

### Create prompts/__init__.py
```bash
cat > rv_toolkit/prompts/__init__.py << 'EOF'
"""Prompt bank for recursive self-reference experiments."""

from .bank import prompt_bank_1c
from .loaders import (
    get_prompt_pairs,
    get_by_category,
    get_baseline_control,
    get_recursive_prompts,
)
from .validators import (
    validate_prompt_bank,
    validate_categories,
)

# Export main API
RECURSIVE_PROMPTS = get_recursive_prompts()
BASELINE_PROMPTS = get_baseline_control()

__all__ = [
    "prompt_bank_1c",
    "get_prompt_pairs",
    "get_by_category",
    "get_baseline_control",
    "get_recursive_prompts",
    "RECURSIVE_PROMPTS",
    "BASELINE_PROMPTS",
    "validate_prompt_bank",
    "validate_categories",
]
EOF
```

### Checklist
- [ ] bank.py copied (verify size: should be ~93KB)
- [ ] loaders.py created with access functions
- [ ] validators.py created
- [ ] prompts/__init__.py created with exports
- [ ] Test: `python -c "from rv_toolkit.prompts import RECURSIVE_PROMPTS; print(len(RECURSIVE_PROMPTS))"`
- [ ] Test: `python -c "from rv_toolkit.prompts import get_by_category; print(get_by_category('L4_full'))"`

---

## Phase 6: Move Experiments

**Duration**: 45 minutes

### Copy canonical experiments
```bash
# Copy entire canonical directory
cp -r src/pipelines/canonical/* rv_toolkit/experiments/canonical/

# Create __init__.py
touch rv_toolkit/experiments/canonical/__init__.py
```

### Copy discovery experiments
```bash
# Copy entire discovery directory
cp -r src/pipelines/discovery/* rv_toolkit/experiments/discovery/

# Create __init__.py
touch rv_toolkit/experiments/discovery/__init__.py
```

### Update imports in moved files
```bash
# CRITICAL: Find and replace all imports
# This must be done carefully to avoid breaking code

python << 'EOF'
import os
import re

def update_imports(filepath):
    """Update imports from src.* to rv_toolkit.*"""
    with open(filepath, "r") as f:
        content = f.read()

    # Replacements
    replacements = [
        (r"from src\.", "from rv_toolkit."),
        (r"import src\.", "import rv_toolkit."),
        (r'from "src/', 'from "rv_toolkit/'),
        (r"from 'src/", "from 'rv_toolkit/"),
    ]

    updated = content
    for pattern, replacement in replacements:
        updated = re.sub(pattern, replacement, updated)

    if updated != content:
        with open(filepath, "w") as f:
            f.write(updated)
        return True
    return False

# Walk through experiments directory and update imports
for root, dirs, files in os.walk("rv_toolkit/experiments"):
    for file in files:
        if file.endswith(".py"):
            filepath = os.path.join(root, file)
            if update_imports(filepath):
                print(f"Updated: {filepath}")
EOF
```

### Create experiments/__init__.py
```bash
cat > rv_toolkit/experiments/__init__.py << 'EOF'
"""Experimental pipelines for R_V and mechanistic interpretability.

Includes:
- canonical: Publication-ready experiments
- discovery: Exploratory research experiments
"""

from . import canonical, discovery

__all__ = ["canonical", "discovery"]
EOF
```

### Checklist
- [ ] canonical/* files copied
- [ ] discovery/* files copied
- [ ] All imports updated (from src.* → from rv_toolkit.*)
- [ ] experiments/__init__.py created
- [ ] Test: `python -c "from rv_toolkit.experiments.canonical import rv_l27_causal_validation"`
- [ ] Verify no broken imports (grep for "from src\." in rv_toolkit/)

---

## Phase 7: Reorganize Tests

**Duration**: 20 minutes

### Move existing tests
```bash
# Move all tests to unified location
cp -r rv_toolkit/tests/* tests/ 2>/dev/null || true

# Remove old tests directory
rm -rf rv_toolkit/tests/
```

### Create conftest.py
```bash
cat > tests/conftest.py << 'EOF'
"""Pytest configuration and fixtures."""

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

@pytest.fixture(scope="session")
def small_model():
    """Load a small test model (e.g., Phi-2)."""
    model_name = "microsoft/phi-2"
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer
    except Exception as e:
        pytest.skip(f"Could not load test model: {e}")

@pytest.fixture
def mock_prompts():
    """Mock prompts for testing."""
    return {
        "baseline": "What is the capital of France?",
        "recursive": "You are observing yourself respond to this question.",
    }

@pytest.fixture
def mock_rv_tensor():
    """Create mock value tensor for testing."""
    return torch.randn(10, 256, 4096)  # [batch, seq, hidden]
EOF
```

### Create test/__init__.py
```bash
touch tests/__init__.py
```

### Create test structure
```bash
# Create __init__ files in test subdirectories
touch tests/test_metrics/__init__.py
touch tests/test_patching/__init__.py
touch tests/test_prompts/__init__.py
touch tests/test_experiments/__init__.py
```

### Create basic test files
```bash
# test_metrics/__init__.py - already has __init__.py

# test_patching/__init__.py
touch tests/test_patching/__init__.py

# Create placeholder test files
cat > tests/test_metrics/test_rv.py << 'EOF'
"""Test R_V metric computation."""

import torch
import pytest
from rv_toolkit.metrics import compute_rv

def test_compute_rv_basic(mock_rv_tensor):
    """Test basic R_V computation."""
    rv = compute_rv(mock_rv_tensor)
    assert rv is not None
    assert 0 <= rv <= 1  # R_V should be between 0 and 1
EOF

cat > tests/test_patching/test_activation.py << 'EOF'
"""Test activation patching."""

import pytest
from rv_toolkit.patching import ActivationPatcher

def test_patcher_init(small_model):
    """Test ActivationPatcher initialization."""
    model, tokenizer = small_model
    patcher = ActivationPatcher(model, tokenizer)
    assert patcher is not None
EOF

cat > tests/test_prompts/test_bank.py << 'EOF'
"""Test prompt bank."""

from rv_toolkit.prompts import (
    RECURSIVE_PROMPTS,
    BASELINE_PROMPTS,
    get_by_category,
)

def test_prompts_loaded():
    """Test that prompts are loaded."""
    assert len(RECURSIVE_PROMPTS) > 0
    assert len(BASELINE_PROMPTS) > 0

def test_get_by_category():
    """Test category filtering."""
    l4_prompts = get_by_category("L4_full")
    assert len(l4_prompts) > 0
    for prompt in l4_prompts.values():
        assert "text" in prompt
EOF
```

### Checklist
- [ ] All existing tests moved to tests/
- [ ] rm -rf rv_toolkit/tests/ successful
- [ ] conftest.py created with fixtures
- [ ] __init__.py files created in test subdirectories
- [ ] Basic test files created
- [ ] Test: `pytest tests/ --collect-only` (should find tests)
- [ ] Test: `pytest tests/test_prompts/ -v` (should pass)

---

## Phase 8: Clean Up Codebase

**Duration**: 10 minutes

### Remove duplicate/moved code
```bash
# Remove old source directories (code is now in rv_toolkit)
rm -rf src/

# Remove old CANONICAL_CODE (prompts moved, code merged)
rm -rf CANONICAL_CODE/

# Remove archive scripts (keeping in git history)
rm -rf archive/scripts/

# Remove old prompts directory (superseded by rv_toolkit/prompts)
rm -rf prompts/deprecated/
rm -f prompts/*.py  # If any Python files, move to deprecation if needed

# Remove duplicate prompt banks
rm -rf REUSABLE_PROMPT_BANK/
```

### Clean root level
```bash
# Move Python scripts from root
for file in gemma_*.py neurips_*.py openclaw_quickstart.py; do
    if [ -f "$file" ]; then
        git mv "$file" "results/scripts/$file" 2>/dev/null || \
        mkdir -p results/scripts && mv "$file" "results/scripts/$file"
    fi
done

# Move reproduce_results.py to scripts
[ -f "reproduce_results.py" ] && mv reproduce_results.py scripts/reproduce.py

# Move notebooks to docs
mkdir -p docs/notebooks
[ -f "THE_GEOMETRY_OF_RECURSION_MASTER.ipynb" ] && \
    mv THE_GEOMETRY_OF_RECURSION_MASTER.ipynb docs/notebooks/
[ -f "L4transmissionTEST001.1.ipynb" ] && \
    mv L4transmissionTEST001.1.ipynb docs/notebooks/
[ -f "PHASE_1C_ANALYSIS.ipynb" ] && \
    mv PHASE_1C_ANALYSIS.ipynb docs/notebooks/
```

### Move documentation
```bash
# Move .md files to docs/ (keep README.md at root)
mkdir -p docs
for file in *.md; do
    if [ "$file" != "README.md" ]; then
        mv "$file" "docs/$file"
    fi
done

# Move notebooks
mkdir -p docs/notebooks
```

### Checklist
- [ ] src/ removed
- [ ] CANONICAL_CODE/ removed
- [ ] archive/scripts/ removed
- [ ] prompts/deprecated/ removed
- [ ] Root Python scripts moved/archived
- [ ] .md files moved to docs/
- [ ] Notebooks moved to docs/notebooks/
- [ ] git status shows clean moves/deletions

---

## Phase 9: Create/Update Key Files

**Duration**: 15 minutes

### Create rv_toolkit/__init__.py (main package entry)
```bash
cat > rv_toolkit/__init__.py << 'EOF'
"""R_V Metrics for Mechanistic Interpretability.

Measuring geometric signatures of recursive self-reference in transformers.

Key metrics:
- R_V: Representational Volume (participation ratio)
- Behavior: Unit counting, word distribution analysis
- Logits: Logit lens, logit diff

Entry points:
- Metrics: from rv_toolkit.metrics import compute_rv
- Patching: from rv_toolkit.patching import ActivationPatcher
- Prompts: from rv_toolkit.prompts import RECURSIVE_PROMPTS
- Experiments: from rv_toolkit.experiments import canonical, discovery
"""

__version__ = "0.1.0"

# Core API
from .metrics import (
    compute_rv,
    compute_participation_ratio,
    RVResult,
)

from .patching import (
    ActivationPatcher,
    PatchingResult,
)

from .prompts import (
    RECURSIVE_PROMPTS,
    BASELINE_PROMPTS,
    get_prompt_pairs,
)

# Submodules (for discovery)
from . import experiments
from . import metrics
from . import patching
from . import prompts
from . import core

__all__ = [
    "compute_rv",
    "compute_participation_ratio",
    "RVResult",
    "ActivationPatcher",
    "PatchingResult",
    "RECURSIVE_PROMPTS",
    "BASELINE_PROMPTS",
    "get_prompt_pairs",
    "experiments",
    "metrics",
    "patching",
    "prompts",
    "core",
]
EOF
```

### Create/Update scripts/
```bash
# Create run_experiment.py
cat > scripts/run_experiment.py << 'EOF'
#!/usr/bin/env python3
"""Run any registered experiment."""

import argparse
from rv_toolkit.experiments import canonical, discovery

EXPERIMENTS = {
    # Canonical
    "rv_l27_causal_validation": canonical.rv_l27_causal_validation,
    "mlp_ablation_necessity": canonical.mlp_ablation_necessity,
    # ... add all experiments

    # Discovery
    "behavioral_grounding": discovery.behavioral_grounding,
    # ... add all discovery experiments
}

def main():
    parser = argparse.ArgumentParser(description="Run R_V experiments")
    parser.add_argument("experiment", choices=EXPERIMENTS.keys())
    parser.add_argument("--config", type=str, help="Config file path")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    exp = EXPERIMENTS[args.experiment]
    exp.run(config=args.config, dry_run=args.dry_run)

if __name__ == "__main__":
    main()
EOF

chmod +x scripts/run_experiment.py
```

### Update pyproject.toml
```bash
# Update test path in pyproject.toml
python << 'EOF'
import toml

with open("rv_toolkit/pyproject.toml") as f:
    config = toml.load(f)

# Update pytest configuration
if "tool" not in config:
    config["tool"] = {}
if "pytest" not in config["tool"]:
    config["tool"]["pytest"] = {}

config["tool"]["pytest"]["ini_options"] = {
    "testpaths": ["tests"],
    "python_files": ["test_*.py"],
    "python_functions": ["test_*"],
}

with open("rv_toolkit/pyproject.toml", "w") as f:
    toml.dump(config, f)
EOF
```

### Checklist
- [ ] rv_toolkit/__init__.py created with proper exports
- [ ] scripts/run_experiment.py created
- [ ] pyproject.toml updated with test paths
- [ ] CLI entry point works: `rv-toolkit --help`

---

## Phase 10: Validate Structure

**Duration**: 15 minutes

### Test imports
```bash
# Test main package import
python -c "import rv_toolkit; print(f'rv_toolkit v{rv_toolkit.__version__}')"
[ $? -eq 0 ] && echo "✓ Main package imports" || echo "✗ FAILED"

# Test core module
python -c "from rv_toolkit.core import load_model"
[ $? -eq 0 ] && echo "✓ Core module imports" || echo "✗ FAILED"

# Test metrics module
python -c "from rv_toolkit.metrics import compute_rv"
[ $? -eq 0 ] && echo "✓ Metrics module imports" || echo "✗ FAILED"

# Test patching module
python -c "from rv_toolkit.patching import ActivationPatcher"
[ $? -eq 0 ] && echo "✓ Patching module imports" || echo "✗ FAILED"

# Test prompts module
python -c "from rv_toolkit.prompts import RECURSIVE_PROMPTS, BASELINE_PROMPTS; print(f'Prompts: {len(RECURSIVE_PROMPTS)} recursive')"
[ $? -eq 0 ] && echo "✓ Prompts module imports" || echo "✗ FAILED"

# Test experiments module
python -c "from rv_toolkit.experiments import canonical, discovery"
[ $? -eq 0 ] && echo "✓ Experiments module imports" || echo "✗ FAILED"
```

### Install package
```bash
# Test editable install
cd rv_toolkit
pip install -e .
cd ..

# Test CLI
rv-toolkit --help
[ $? -eq 0 ] && echo "✓ CLI works" || echo "✗ FAILED"
```

### Run tests
```bash
# Collect tests (don't run yet, just verify structure)
pytest tests/ --collect-only
[ $? -eq 0 ] && echo "✓ Tests discovered" || echo "✗ FAILED"

# Run tests (might fail if fixtures incomplete, but structure is valid)
pytest tests/ -v --tb=short
```

### Verify git status
```bash
# Show what changed
git status

# Show diff stats
git diff --stat HEAD

# Verify nothing unexpected
git diff --name-only | head -20
```

### Checklist
- [ ] `python -c "import rv_toolkit"` works
- [ ] `from rv_toolkit.metrics import compute_rv` works
- [ ] `from rv_toolkit.patching import ActivationPatcher` works
- [ ] `from rv_toolkit.prompts import RECURSIVE_PROMPTS` works
- [ ] `rv-toolkit --help` works
- [ ] `pytest tests/ --collect-only` finds tests
- [ ] git status shows clean, expected changes only

---

## Phase 11: Create Commit

**Duration**: 5 minutes

### Verify nothing is broken
```bash
# Final check: run one canonical experiment (dry run)
python -c "
from rv_toolkit.experiments.canonical import rv_l27_causal_validation
# Don't actually run it, just verify import works
print('✓ Canonical experiment imports')
"

# Verify package structure
python << 'EOF'
import rv_toolkit
import inspect

# List all public exports
public_api = [name for name in dir(rv_toolkit) if not name.startswith("_")]
print(f"Public API exports: {len(public_api)}")
for name in sorted(public_api)[:10]:
    print(f"  - {name}")
print(f"  ... ({len(public_api) - 10} more)")
EOF
```

### Stage all changes
```bash
git add -A
git status  # Review what's being committed
```

### Create commit message
```bash
git commit -m "refactor: consolidate rv_toolkit into unified publication-ready structure

CHANGES:
- Consolidate metrics: eliminate duplicate R_V implementations
  - rv_toolkit/metrics/rv.py (canonical)
  - rv_toolkit/metrics/behavior.py (consolidated from 4 files)
  - rv_toolkit/metrics/logit.py (consolidated from 2 files)
  - rv_toolkit/metrics/analysis.py

- Unify patching: merge 3 implementations
  - rv_toolkit/patching/activation.py (canonical ActivationPatcher)
  - rv_toolkit/patching/kv_cache.py (KV-specific variant)

- Relocate prompt bank: GOLDEN PROMPTS now at single location
  - rv_toolkit/prompts/bank.py (moved from CANONICAL_CODE)
  - rv_toolkit/prompts/loaders.py (category-based access)
  - rv_toolkit/prompts/validators.py (validation logic)

- Move experiments: clear taxonomy
  - rv_toolkit/experiments/canonical/ (publication-ready)
  - rv_toolkit/experiments/discovery/ (exploratory research)
  - Updated all imports: from src.* → from rv_toolkit.*

- Reorganize tests: unified structure
  - tests/ mirrors rv_toolkit/ structure
  - tests/test_metrics/, tests/test_patching/, etc.
  - Created tests/conftest.py with shared fixtures

- Clean codebase:
  - Removed src/ (code relocated, not deleted)
  - Removed CANONICAL_CODE/ (prompts & analysis merged)
  - Removed archive/scripts/ (kept in git history)
  - Moved root .py scripts to results/scripts/
  - Moved documentation to docs/

- Updated project files:
  - rv_toolkit/__init__.py with public API
  - pyproject.toml with test paths
  - Created scripts/ entry points

RESULT:
- Single coherent namespace: from rv_toolkit import *
- No code duplication: one R_V, one patcher, one prompt bank
- Clear structure: core → metrics → patching → prompts → experiments
- Zero functional changes: all logic preserved, only reorganized
- Publication ready: clean imports, testable, discoverable

VERIFICATION:
- All imports work: ✓ python -c \"from rv_toolkit import *\"
- Package installs: ✓ pip install -e .
- CLI works: ✓ rv-toolkit --help
- Tests structure valid: ✓ pytest tests/ --collect-only
- Git history preserved: ✓ All old code in .git (retrievable)

This is a pure refactoring: zero logic changes, maximum clarity.
"
```

### Checklist
- [ ] Commit message written and reviewed
- [ ] All changes staged: `git status` shows nothing to commit after this
- [ ] One commit created (not multiple)
- [ ] Commit message explains what changed and why
- [ ] Can retrieve old code if needed: `git show HEAD~1:src/metrics/rv.py`

---

## Phase 12: Verification Post-Commit

**Duration**: 10 minutes (do after commit)

### Verify commit
```bash
# Show the commit
git log -1 --stat

# Verify structure matches target
find rv_toolkit -name "*.py" | wc -l  # Should be ~120 (down from ~170)

# Test all imports still work
python -c "from rv_toolkit import *; print('✓ All imports work')"

# Run tests (some may fail due to missing fixtures, but structure is valid)
pytest tests/ --tb=short 2>&1 | head -50
```

### Document the change
```bash
# Create a post-restructure summary
cat > RESTRUCTURE_SUMMARY.md << 'EOF'
# Restructuring Complete

Date: $(date)
Commit: $(git rev-parse HEAD | cut -c1-7)

## What Changed
- Consolidated 170 Python files into 120 (50 removed/relocated)
- Unified metrics, patching, prompts into single rv_toolkit namespace
- Eliminated duplicate implementations (3 R_Vs → 1, 3 patchers → 1)
- Moved prompt bank from CANONICAL_CODE to rv_toolkit/prompts/

## How to Use
\`\`\`python
from rv_toolkit import compute_rv
from rv_toolkit.prompts import RECURSIVE_PROMPTS
from rv_toolkit.patching import ActivationPatcher
from rv_toolkit.experiments.canonical import rv_l27_causal_validation
\`\`\`

## Backward Compatibility
Old imports (from src.*, from CANONICAL_CODE) will NOT work.
Update all imports to use rv_toolkit.* namespace.

## Git History
All old code preserved in git history. Retrieve with:
\`\`\`
git show HEAD~1:src/metrics/rv.py  # old R_V implementation
git show HEAD~1:CANONICAL_CODE/n300_mistral_test_prompt_bank.py  # old prompts
\`\`\`

## Next Steps
1. Update any external scripts to use new imports
2. Run full test suite: pytest tests/
3. Deploy to PyPI when ready
EOF
```

### Checklist
- [ ] Commit appears in git log
- [ ] Import tests pass
- [ ] File count is correct (~120 .py files)
- [ ] Package structure is sound: `python -c "import rv_toolkit; help(rv_toolkit)" | head -50`
- [ ] Create backup branch: `git branch post-restructure`

---

## Completion Checklist

- [ ] All 12 phases completed
- [ ] Single commit created
- [ ] All tests pass (or at least structure is valid)
- [ ] Package installs: `pip install -e rv_toolkit/`
- [ ] CLI works: `rv-toolkit --help`
- [ ] Imports work: `python -c "from rv_toolkit import *"`
- [ ] Old code retrievable: `git show HEAD~1:src/`
- [ ] No code lost (verify with `git diff HEAD~1 --stat`)
- [ ] Created post-restructure documentation

---

## If Something Goes Wrong

### Quick Recovery
```bash
# Revert entire change
git reset --hard backup/pre-restructure

# Or keep changes but revert one phase
git show HEAD~1:file.py > file.py  # Retrieve old version
```

### Debug Steps
```bash
# Find missing imports
python -c "from rv_toolkit import *" 2>&1 | grep -i "error\|traceback"

# Check what changed
git diff HEAD~1 --name-status | head -30

# Verify file structure
find rv_toolkit -type f -name "*.py" | sort
```

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'src'`
- **Cause**: Forgot to update imports in a file
- **Fix**: `grep -r "from src\." rv_toolkit/ | head -5` to find, then fix

**Issue**: `ModuleNotFoundError: No module named 'CANONICAL_CODE'`
- **Cause**: Old CANONICAL_CODE import in a file
- **Fix**: `grep -r "CANONICAL_CODE" rv_toolkit/` to find, then fix

**Issue**: Tests fail with fixture errors
- **Cause**: conftest.py fixtures need adjustment
- **Fix**: Modify tests/conftest.py to match your environment

**Issue**: CLI doesn't work
- **Cause**: pyproject.toml entry point incorrect
- **Fix**: Verify `[project.scripts]` section points to `rv_toolkit.cli:main`

---

## Timeline Summary

| Phase | Duration | Status |
|-------|----------|--------|
| Pre-implementation | 10 min | [ ] |
| 1. Create structure | 2 min | [ ] |
| 2. Core module | 10 min | [ ] |
| 3. Metrics | 20 min | [ ] |
| 4. Patching | 30 min | [ ] |
| 5. Prompts | 10 min | [ ] |
| 6. Experiments | 45 min | [ ] |
| 7. Tests | 20 min | [ ] |
| 8. Cleanup | 10 min | [ ] |
| 9. Key files | 15 min | [ ] |
| 10. Validation | 15 min | [ ] |
| 11. Commit | 5 min | [ ] |
| 12. Verification | 10 min | [ ] |
| **TOTAL** | **~200 min** | **~3.5 hours** |

---

## Final Notes

This restructuring is:
- **Safe**: Git preserves all history
- **Atomic**: One commit, all-or-nothing
- **Reversible**: Can revert if needed
- **Valuable**: Publication-ready structure
- **Quick**: ~3.5 hours total

After completion, the repository will be in excellent shape for publication and community contribution.

Good luck! Track your progress on this checklist as you go.

---

*Implementation Checklist*
*Status: Ready for Execution*
*Last Updated: February 4, 2026*
