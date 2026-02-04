# Archive Recovery Checklist

## Files to Recover (13 total)

### Priority 1: Publication-Ready (Copy immediately)
- [ ] `archive/rv_paper_code/VALIDATED_mistral7b_layer27_activation_patching.py`
  → `rv_toolkit/methodologies/patching/validated_layer27_mistral.py`
  **GOLD**: Causal validation, locked parameters, publication-ready

### Priority 2: Critical Experiments (Copy this week)
- [ ] `archive/scripts/experiment_multi_token_generation.py`
  → `rv_toolkit/experiments/generation_dynamics.py`
  **KEY**: Addresses multi-token generation gap in current paper

- [ ] `archive/scripts/comprehensive_head_discovery.py`
  → `rv_toolkit/experiments/head_discovery.py`
  **LARGE**: 829 lines, most complete circuit discovery

- [ ] `archive/scripts/comprehensive_circuit_test.py`
  → `rv_toolkit/experiments/circuit_validation.py`
  **STRUCTURED**: Well-organized multi-condition harness

- [ ] `archive/scripts/aggressive_behavior_transfer.py`
  → `rv_toolkit/experiments/aggressive_behavior_transfer.py`
  **COMPREHENSIVE**: All transfer combinations tested

### Priority 3: Transfer Validation (Copy week 2)
- [ ] `archive/scripts/ultimate_transfer.py`
  → `rv_toolkit/experiments/transfer_validation.py`

- [ ] `archive/scripts/refined_nuclear_transfer.py`
  → `rv_toolkit/experiments/refined_transfer.py`

- [ ] `archive/scripts/investigate_transfer.py`
  → `rv_toolkit/experiments/transfer_investigation.py`

- [ ] `archive/scripts/investigate_transfer_efficient.py`
  → `rv_toolkit/experiments/transfer_efficient_remote.py`

### Priority 4: Supporting Methodologies (Copy week 2)
- [ ] `archive/scripts/advanced_activation_patching.py`
  → `rv_toolkit/methodologies/patching/advanced_sweeps.py`

- [ ] `archive/scripts/experiment_causal_sweep.py`
  → `rv_toolkit/experiments/causal_parameter_sweep.py`

- [ ] `archive/scripts/analyze_comprehensive_circuit_test_part_a.py`
  → `rv_toolkit/analysis/circuit_analysis.py`

- [ ] `archive/scripts/analyze_existing_csv.py`
  → `rv_toolkit/analysis/csv_analysis_framework.py`

- [ ] `archive/scripts/experiment_random_kv_investigation.py`
  → `rv_toolkit/experiments/control_kv_investigation.py`

---

## After Recovery: Refactoring Tasks

### Extract Utilities (Week 2)
- [ ] Create `rv_toolkit/core/patching_utils.py`
  - Activate hook pattern from validated patching
  - Window-based V projection capture
  - Shared patching logic

- [ ] Create `rv_toolkit/core/metrics_utils.py`
  - R_V computation (participation ratio)
  - Statistical functions
  - Result aggregation

### Add Documentation (Week 2)
- [ ] Add docstrings to all recovered files
- [ ] Create experiment README for each
- [ ] Document dependencies and runtimes
- [ ] Add usage examples

### Testing (Week 3)
- [ ] Test imports for all 13 recovered files
- [ ] Validate R_V computation consistency
- [ ] Run quick smoke tests

---

## Files to Keep Archived (97 files)

**In `/archive/scripts/` and `/archive/outputs/`**

These stay but create INDEX:
- Circuit discovery evolution (7 files)
- Reproduction attempts (12+ files)
- Phase ablation progression (25+ files)
- Validation tests (5 files)
- Control conditions (3 files)
- Analysis utilities (5+ files)
- Model-specific tests (3 files)
- Other methodological explorations (30+ files)

See `ARCHIVE_AUDIT_REPORT.md` for detailed list.

---

## Files to Delete (20 files)

**Safe to remove entirely:**
- `debug_local.py`, `debug_path_patching.py`
- `test_*.py` (11 test files)
- `kitchen_sink_prompts.py`
- `experiment_kitchen_sink.py`
- `grand_unified_test_original.py`
- `unified_test_head_level.py`

---

## Validation Checklist

After recovery, verify:
- [ ] All imports resolve
- [ ] Config sections match new file locations
- [ ] Output directories can be created
- [ ] torch and transformers versions compatible
- [ ] No hardcoded paths fail

---

## Success Criteria

Recovery complete when:
- [x] All 13 files copied to new locations
- [x] File structure created
- [x] Imports tested
- [x] Archive index created
- [x] Cleanup plan documented

Expected time: **2-4 hours to execute**
