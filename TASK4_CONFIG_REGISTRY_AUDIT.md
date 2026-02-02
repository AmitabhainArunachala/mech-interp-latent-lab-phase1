# TASK 4: Config/Registry Consistency Audit

## Executive Summary
**Status**: ✅ CLEAN - All configs reference valid registered experiments

## Findings

### 1. Cross-Architecture Validation Status
- **Pipeline**: Already removed from registry (line 214 in `src/pipelines/registry.py`)
- **Comment in registry**: `# "cross_architecture_validation" removed - file deleted during cleanup`
- **Orphaned configs**: Already archived in `configs/archive/orphaned/`
  - `cross_architecture_llama.json`
  - `cross_architecture_mistral.json`
  - `discovery_cross_arch/01_baseline_rv.json`
  - `cross_architecture_validation.json`

### 2. Registry Analysis
**Total registered experiments**: 46

**Tiers**:
- Canonical (7): Core paper findings
- Discovery (12): Methodology tools
- Archive (27+): Historical/superseded

**Intentionally removed**:
- `mlp_ablation_necessity` - Contract violation (measures R_V on generated text, not prompt-pass only)
- `cross_architecture_validation` - File deleted during cleanup

### 3. Config Inventory

#### Active Configs (All Valid)
Scanned directories:
- `configs/canonical/` ✅
- `configs/discovery/` ✅
- `configs/smoke_test/` ✅
- `configs/gold/` ✅

**Result**: All 40+ active configs reference registered experiments

#### Archived Configs
- `configs/archive/orphaned/` - 4 cross_architecture_validation configs
- `configs/archive/orphaned/smoke_test/` - 1 deprecated mlp_ablation_necessity config
- `configs/archive/orphaned/gold/` - 1 old p10_advanced_steering config

### 4. False Positive Resolved
- 40+ configs in `configs/discovery/*/02_source_hunt_mlp_ablation_*.json`
- These use `mlp_ablation_necessity_prompt_pass` (valid, registered)
- Initial grep matched substring "mlp_ablation_necessity" but full experiment name is valid

### 5. Verification Script Enhancements

Updated `scripts/verify_research_ready.py` to include:

```python
# 4. Check config files reference valid experiments
# Known exceptions: meta-experiments or intentionally removed pipelines
KNOWN_EXCEPTIONS = {
    "batch_run",  # Meta-runner, not a pipeline
    "mlp_ablation_necessity",  # Removed from registry - contract violation
    "p10_advanced_steering",  # Old experiment name, archived
}
```

**New checks**:
- Scans all active config directories
- Validates experiment name against registry
- Handles known exceptions (warns instead of errors)
- Reports orphaned configs clearly

**Test output**:
```
4. Config validation...
   [OK] All configs reference valid experiments
```

## Actions Taken

### 1. Enhanced verify_research_ready.py
- Added config/registry validation (check #4)
- Added KNOWN_EXCEPTIONS set for batch_run, deprecated pipelines
- Scans canonical/, discovery/, smoke_test/, gold/
- Clear error vs warning distinction

### 2. Archived Orphaned Configs
Moved to `configs/archive/orphaned/`:
- `smoke_test/l0_ablation.json` (mlp_ablation_necessity - deprecated)
- `gold/10_advanced_steering.json` (p10_advanced_steering - old name)

### 3. Verified Clean State
```bash
$ python3 scripts/verify_research_ready.py
============================================================
RESEARCH READY VERIFICATION (v2)
============================================================

1. Core imports...
   [OK] Core imports successful

2. Registry validation...
   [OK] Registry: 46 experiments loaded
   [OK] Deprecated mlp_ablation_necessity removed from registry

3. Prompt bank...
   [OK] Prompt bank: 754 prompts (version: 75e7c1b8dcebc24e)

4. Config validation...
   [OK] All configs reference valid experiments

5. Model physics...
   [OK] Mistral-7B: early=5, late=27

6. SVD precision check...
   [OK] rv.py uses float64 (double precision)

7. Requirements pinning...
   [OK] Uses exact version pins (==)

============================================================
RESULT: RESEARCH READY
============================================================
```

## Registry Integrity Report

### ✅ Validated
- `cross_architecture_validation` properly removed from registry
- Orphaned configs already archived
- All active configs reference valid experiments
- Discovery configs (40+) correctly use `mlp_ablation_necessity_prompt_pass`

### ✅ Contracts Enforced
- `mlp_ablation_necessity` removed (violates prompt-pass-only contract)
- Replacement `mlp_ablation_necessity_prompt_pass` properly registered
- All source hunt configs updated to use valid pipeline

### ✅ Verification Enhanced
- Config/registry mismatches now detected automatically
- Known exceptions handled gracefully
- Clear error vs warning distinction

## Rationale

### Why Not Register batch_run?
- It's a meta-experiment runner, not a pipeline
- Would require special handling in registry
- Better to keep it as a known exception
- Only 1 config uses it (sweep runner)

### Why Archive Instead of Delete?
- Preserves historical context
- Allows recovery if needed
- Clear separation: archive/ = not active

### Why Enhanced verify_research_ready.py?
- Prevents future config/registry drift
- Catches orphaned configs early
- Automates consistency checking
- Part of research-ready gate

## Files Modified

1. `/Users/dhyana/mech-interp-latent-lab-phase1/scripts/verify_research_ready.py`
   - Added config/registry validation (check #4)
   - Added KNOWN_EXCEPTIONS handling
   - Enhanced reporting

2. `/Users/dhyana/mech-interp-latent-lab-phase1/configs/archive/orphaned/`
   - Added `smoke_test/l0_ablation.json`
   - Added `gold/10_advanced_steering.json`

## Summary

**Config/Registry Consistency**: ✅ ENFORCED

- All orphaned `cross_architecture_validation` configs already archived
- Pipeline properly removed from registry with clear comment
- All active configs validated against registry
- 2 additional orphaned configs archived
- Verification script enhanced to prevent future drift
- System is research-ready with full contract integrity

**No further action needed** - registry is clean and protected.
