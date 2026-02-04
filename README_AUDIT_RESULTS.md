# Archive Audit Results

**Date**: February 4, 2026  
**Location**: `/Users/dhyana/mech-interp-latent-lab-phase1/archive/`  
**Total Files Reviewed**: 130 Python files  

## Quick Summary

| Category | Count | Lines | Action |
|----------|-------|-------|--------|
| **RECOVER** | 13 | ~3,500 | Move to rv_toolkit/ |
| **KEEP_ARCHIVED** | 97 | ~20,000 | Reference only |
| **DELETE** | 20 | ~3,000 | Remove safely |

## Key Findings

### Gold-Tier Assets
**1 validated, publication-ready methodology:**
- `VALIDATED_mistral7b_layer27_activation_patching.py` - Core causal validation with locked parameters

### Critical Experiments (4 files)
Addressing research gaps:
- `experiment_multi_token_generation.py` - Reviewer question on R_V during generation
- `comprehensive_head_discovery.py` - 829 lines, most complete circuit discovery
- `comprehensive_circuit_test.py` - Multi-condition test harness
- `aggressive_behavior_transfer.py` - All transfer combinations tested

### Transfer Validation (4 files)
- `ultimate_transfer.py`, `refined_nuclear_transfer.py`, `investigate_transfer.py`, `investigate_transfer_efficient.py`

### Supporting Methodologies (4 files)
- Advanced patching sweeps, causal parameter sweep, circuit analysis, CSV analysis framework

## Quality Assessment

✅ **Code Quality**: EXCELLENT
- 100% proper config sections
- 92% have function docstrings
- 100% device/seed handling
- No security issues

⚠️ **Areas for Improvement**
- High code duplication (same R_V logic repeated)
- Metric utilities scattered across files
- Mixed path handling (string vs Path)

## Documentation Provided

Four comprehensive reports created:

1. **ARCHIVE_AUDIT_SUMMARY.txt** - Executive summary with categories
2. **ARCHIVE_AUDIT_REPORT.md** - Detailed report with recommendations
3. **ARCHIVE_RECOVER_CHECKLIST.md** - Implementation checklist
4. **ARCHIVE_AUDIT_DETAILED_FINDINGS.txt** - Complete findings

## Recommended Recovery Plan

### Week 1: Copy 13 Files
```
archive/rv_paper_code/VALIDATED_mistral7b_layer27_activation_patching.py
archive/scripts/experiment_multi_token_generation.py
archive/scripts/comprehensive_head_discovery.py
archive/scripts/comprehensive_circuit_test.py
archive/scripts/aggressive_behavior_transfer.py
archive/scripts/ultimate_transfer.py
archive/scripts/refined_nuclear_transfer.py
archive/scripts/investigate_transfer.py
archive/scripts/investigate_transfer_efficient.py
```

To:
```
rv_toolkit/methodologies/patching/validated_layer27_mistral.py
rv_toolkit/experiments/generation_dynamics.py
rv_toolkit/experiments/head_discovery.py
rv_toolkit/experiments/circuit_validation.py
rv_toolkit/experiments/aggressive_behavior_transfer.py
rv_toolkit/experiments/transfer_validation.py
rv_toolkit/experiments/refined_transfer.py
rv_toolkit/experiments/transfer_investigation.py
rv_toolkit/experiments/transfer_efficient_remote.py
```

Plus 4 more supporting files (see ARCHIVE_RECOVER_CHECKLIST.md)

### Week 2: Refactor
- Extract common utilities (R_V, hooks, metrics)
- Add type hints and docstrings
- Create experiment documentation

### Week 3: Testing
- Test imports
- Validate R_V consistency
- Cross-model validation

## Risk Assessment

**Low Risk**: Validated methodology + comprehensive experiments
**Medium Risk**: Advanced patching, head discovery, circuit tests (depends on hook patterns)
**No High-Risk Files**

## Files to Delete Safely

20 debug/test files:
- SSH debugging scripts (6 files)
- Test stubs and stress tests (8 files)
- Kitchen-sink utilities (6 files)

No research value, no lasting impact.

## Expected Impact

- Stronger toolkit for next experiments
- 3,500 lines of production-ready code
- Clear pathway to publication
- ~10-18 hours to full integration

## Time Estimate

- Recovery: 2-4 hours
- Refactoring: 4-8 hours
- Testing: 4-6 hours
- **Total: 10-18 hours**

## Contact & Questions

See detailed reports for:
- File-by-file analysis
- Code quality metrics
- Dependency information
- Usage recommendations
- Cross-cutting patterns

---

**Status**: Ready for implementation  
**Recommendation**: Proceed with recovery plan
