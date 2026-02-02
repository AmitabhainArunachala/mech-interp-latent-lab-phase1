# TASK 3: Deprecation Enforcement - mlp_ablation_necessity

**Date**: 2026-01-24  
**Status**: COMPLETE  
**Goal**: Make deprecated `mlp_ablation_necessity` impossible to run accidentally

---

## Summary

Successfully enforced deprecation of contract-violating pipeline. The deprecated version measured R_V on generated text; the correct version (`mlp_ablation_necessity_prompt_pass`) measures only on prompt processing.

**Result**: Runtime error with clear migration path. No active configs reference deprecated version.

---

## Changes Made

### 1. Registry Enforcement
**File**: `/Users/dhyana/mech-interp-latent-lab-phase1/src/pipelines/registry.py`

Added runtime block in `run_from_config()`:
```python
if exp == "mlp_ablation_necessity":
    raise ConfigError(
        f"Experiment 'mlp_ablation_necessity' is deprecated (measures R_V on generated text). "
        f"Use 'mlp_ablation_necessity_prompt_pass' instead. "
        f"Update your config's 'experiment' field."
    )
```

### 2. Runner Documentation Update
**File**: `/Users/dhyana/mech-interp-latent-lab-phase1/src/pipelines/run.py`

Updated `CANONICAL_EXPERIMENTS` set (line 29):
- Removed: `"mlp_ablation_necessity"`
- Added: `"mlp_ablation_necessity_prompt_pass"`

### 3. Config Updates
**File**: `/Users/dhyana/mech-interp-latent-lab-phase1/configs/smoke_test/l0_ablation.json`

Changed:
- `"experiment"`: `"mlp_ablation_necessity"` → `"mlp_ablation_necessity_prompt_pass"`
- Added required params: `"early_layer": 5, "late_layer": 27`

**Archived configs** (left unchanged, already in `archive/orphaned/`):
- `mlp_ablation_necessity_l0.json`
- `mlp_ablation_necessity_l1.json`
- `mlp_ablation_necessity_l2.json`
- `mlp_ablation_necessity_l3.json`

---

## Verification

```python
from src.pipelines.registry import run_from_config
from pathlib import Path

# This raises ConfigError:
test_config = {"experiment": "mlp_ablation_necessity", "params": {}}
run_from_config(test_config, Path("/tmp"))
# ConfigError: Experiment 'mlp_ablation_necessity' is deprecated...

# This succeeds:
from src.pipelines.registry import get_registry
reg = get_registry()
assert "mlp_ablation_necessity" not in reg  # Confirmed
assert "mlp_ablation_necessity_prompt_pass" in reg  # Confirmed
```

---

## Contract Violation Context

**Original issue**: `mlp_ablation_necessity.py` measured R_V on generated tokens (lines 149-152), violating the established contract that R_V should only be measured on prompt processing.

**Correct version**: `mlp_ablation_necessity_prompt_pass.py` measures R_V during the forward pass of the prompt, before any generation occurs.

---

## Migration Path for Users

If you encounter the deprecation error:

1. **Update config file**:
   ```json
   {
     "experiment": "mlp_ablation_necessity_prompt_pass",
     "params": {
       "layer": 0,
       "n_pairs": 80,
       "early_layer": 5,
       "late_layer": 27,
       ...
     }
   }
   ```

2. **Re-run**: The pipeline will now use contract-compliant measurement

---

## Preserved for Historical Reference

The deprecated code remains at:
- `/Users/dhyana/mech-interp-latent-lab-phase1/src/pipelines/canonical/mlp_ablation_necessity.py`

With deprecation warning at module level (lines 27-32).

---

## Rationale

**Minimal, focused changes**:
- Registry block prevents accidental execution through normal workflows
- Runner documentation updated for clarity
- Single smoke test config updated
- Archive configs left untouched (historical preservation)
- No experiment execution performed

**Preserved flexibility**:
- Old code still exists for reference
- Archive configs preserved
- Clear error messages guide migration

**Contract integrity enforced**:
- Impossible to run violated version through config system
- All active configs use compliant version
