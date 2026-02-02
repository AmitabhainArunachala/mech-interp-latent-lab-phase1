# Deprecation Enforcement: mlp_ablation_necessity

**Date**: 2026-01-24
**Task**: TASK 3 - Make deprecated mlp_ablation_necessity impossible to run accidentally

## Problem Statement

The `mlp_ablation_necessity` pipeline violates the R_V measurement contract by measuring R_V on generated text instead of prompt-only processing. The correct version is `mlp_ablation_necessity_prompt_pass`.

Prior state:
- Deprecation WARNING exists in code
- Pipeline still runnable through configs
- Registry had already removed it (comments only)
- Configs were mostly updated, but 1 smoke test config remained

## Changes Made

### 1. Registry Enforcement (`src/pipelines/registry.py`)

Added runtime block in `run_from_config()` function:

```python
def run_from_config(cfg: Dict[str, Any], run_dir: Path) -> ExperimentResult:
    exp, _params = _validate_top_level(cfg)

    # Enforce deprecation: block mlp_ablation_necessity completely
    if exp == "mlp_ablation_necessity":
        raise ConfigError(
            f"Experiment 'mlp_ablation_necessity' is deprecated (measures R_V on generated text). "
            f"Use 'mlp_ablation_necessity_prompt_pass' instead. "
            f"Update your config's 'experiment' field."
        )
    
    reg = get_registry()
    ...
```

**Effect**: Any config with `"experiment": "mlp_ablation_necessity"` will now raise a ConfigError with clear migration instructions.

### 2. Runner Script Update (`src/pipelines/run.py`)

Updated `CANONICAL_EXPERIMENTS` set:

```python
CANONICAL_EXPERIMENTS = {
    "rv_l27_causal_validation",
    "confound_validation",
    "random_direction_control",
    "mlp_ablation_necessity_prompt_pass",  # Replaces mlp_ablation_necessity
    "mlp_sufficiency_test",
    "combined_mlp_sufficiency_test",
    "head_ablation_validation",
}
```

**Effect**: Documentation now correctly reflects the canonical pipeline name.

### 3. Config Updates

Updated 1 smoke test config:
- `configs/smoke_test/l0_ablation.json`: Changed to use `mlp_ablation_necessity_prompt_pass`, added `early_layer` and `late_layer` params

Left unchanged (already in archive/orphaned):
- `configs/archive/orphaned/mlp_ablation_necessity_l0.json`
- `configs/archive/orphaned/mlp_ablation_necessity_l1.json`
- `configs/archive/orphaned/mlp_ablation_necessity_l2.json`
- `configs/archive/orphaned/mlp_ablation_necessity_l3.json`

**Note**: All other configs (canonical + discovery) were already updated to use `mlp_ablation_necessity_prompt_pass`.

## Verification

Tested with Python script:
```python
from src.pipelines.registry import run_from_config

# This now raises ConfigError:
test_config = {"experiment": "mlp_ablation_necessity", "params": {}}
run_from_config(test_config, Path("/tmp"))
# ConfigError: Experiment 'mlp_ablation_necessity' is deprecated...
```

## Result

**IMPOSSIBLE to accidentally run the violated pipeline** through:
1. Normal runner (`python -m src.pipelines.run`)
2. Registry-based execution
3. Any config-driven workflow

The deprecated code remains in `src/pipelines/canonical/mlp_ablation_necessity.py` for historical reference, with deprecation warning at import time.

## Migration Path

Users encountering the block error should:
1. Update config's `"experiment"` field to `"mlp_ablation_necessity_prompt_pass"`
2. Add `"early_layer": 5, "late_layer": 27` to params (for R_V measurement)
3. Re-run

All active configs already migrated.
