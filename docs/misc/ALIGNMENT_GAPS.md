# Alignment Gaps Report

**Date:** January 10, 2026  
**Auditor:** Cursor Agent  
**Scope:** Full repository methodology alignment audit

---

## Executive Summary

This report identifies **17 specific gaps** between the documented industry-grade methodology standard and the actual implementation/execution state of the repository.

**Critical Gaps:** 5  
**Major Gaps:** 7  
**Minor Gaps:** 5

---

## Critical Gaps (Must Fix Before Any Publication Claims)

### GAP-001: RUN_INDEX.jsonl Does Not Exist

**Severity:** 🔴 CRITICAL

**Evidence:**
- `src/utils/run_metadata.py:96` implements `append_to_run_index()` which writes to `results/RUN_INDEX.jsonl`
- File does NOT exist at `results/RUN_INDEX.jsonl`
- Multiple documentation files claim it exists

**Root Cause:**
The function opens the file in append mode (`"a"`) but the parent directory check doesn't guarantee the file will be created if it doesn't exist and no data is appended.

**Fix:**
```python
# src/utils/run_metadata.py:96-98
def append_to_run_index(run_dir: Path, summary: Dict[str, Any]) -> None:
    index_path = Path(__file__).parent.parent.parent / "results" / "RUN_INDEX.jsonl"
    index_path.parent.mkdir(parents=True, exist_ok=True)
+   # Ensure file exists before append
+   if not index_path.exists():
+       index_path.touch()
```

**Verification:** After fix, run any pipeline and confirm `results/RUN_INDEX.jsonl` exists with content.

---

### GAP-002: metadata.json Not Found in Any Run Directory

**Severity:** 🔴 CRITICAL

**Evidence:**
- `save_metadata(run_dir, metadata)` is called in mlp_ablation_necessity.py (line 361)
- No `metadata.json` files found under `results/`

**Root Cause:**
Either:
1. Pipelines were run before `save_metadata()` was added, OR
2. `save_metadata()` is being called but silently failing, OR
3. Runs completed without reaching the `save_metadata()` call

**Fix:** Re-run canonical pipelines and verify metadata.json is created.

**Verification:**
```bash
find results -name "metadata.json" -type f
# Should return at least one path
```

---

### GAP-003: restore_norm(M) Not Implemented

**Severity:** 🔴 CRITICAL

**Evidence:**
- INDUSTRY_GRADE_SPINE_AUDIT.md requires: `restore_norm = (M_patched - M_corrupt)/(M_clean - M_corrupt)`
- No implementation of `restore_norm` for `mode_score_m` exists in any pipeline
- Only `rv_restoration_pct` is computed (for R_V, not mode score)

**Impact:** Cannot make normalized restoration claims for behavioral metric.

**Fix:**
```python
# Add to sufficiency tests (mlp_sufficiency_test.py, mlp_combined_sufficiency_test.py)

# Compute mode restoration norm
mode_clean = mode_rec  # Recursive mode score (clean target)
mode_corrupt = mode_base  # Baseline mode score (corrupted baseline)
mode_patched = mode_patched  # After patching

mode_restore_norm = (mode_patched - mode_corrupt) / (mode_clean - mode_corrupt) if (mode_clean - mode_corrupt) != 0 else float("nan")

# Add to summary
summary["mode_restore_norm"] = float(mode_restore_norm)
```

---

### GAP-004: Prompt IDs Not Stored in random_direction_control.py

**Severity:** 🔴 CRITICAL

**Evidence:**
- Line 111: `pairs = loader.get_balanced_pairs(n_pairs=n_pairs, seed=seed)`
- Does NOT use `get_balanced_pairs_with_ids()`
- No prompt IDs in CSV output

**Impact:** Results cannot be reproduced if `bank.json` changes.

**Fix:**
```python
# Line 111 change:
- pairs = loader.get_balanced_pairs(n_pairs=n_pairs, seed=seed)
+ pairs_with_ids = loader.get_balanced_pairs_with_ids(n_pairs=n_pairs, seed=seed)
+ pairs = [(rec_text, base_text) for _, _, rec_text, base_text in pairs_with_ids]

# Add to results dict:
+ "recursive_prompt_id": rec_id,
+ "baseline_prompt_id": base_id,
```

---

### GAP-005: Prompt IDs Not Stored in circuit_discovery.py

**Severity:** 🔴 CRITICAL

**Evidence:**
- Line 51: `pairs = loader.get_balanced_pairs(n_pairs=n_pairs, seed=seed)`
- Does NOT use `get_balanced_pairs_with_ids()`
- No prompt IDs in CSV output

**Fix:** Same as GAP-004.

---

## Major Gaps (Should Fix Before Publication)

### GAP-006: Generation Parameters Not Logged in Metadata

**Severity:** 🟠 MAJOR

**Affected Pipelines:** All

**Evidence:**
- `max_new_tokens`, `temperature`, `do_sample`, `top_p` used in generation
- Not included in `metadata.json` or `summary.json`

**Fix:**
```python
# Add to get_run_metadata() or summary:
metadata["generation_params"] = {
    "max_new_tokens": params.get("max_new_tokens", 200),
    "temperature": params.get("temperature", 0.0),
    "do_sample": params.get("do_sample", False),
    "top_p": params.get("top_p", 1.0),
}
```

---

### GAP-007: No Run Index Append in circuit_discovery.py

**Severity:** 🟠 MAJOR

**Evidence:**
- No call to `append_to_run_index()` in file
- Runs not tracked in centralized ledger

**Fix:**
```python
# Add at end of run_circuit_discovery_from_config():
from src.utils.run_metadata import get_run_metadata, save_metadata, append_to_run_index

metadata = get_run_metadata(cfg, prompt_ids=pairs_with_ids, ...)
save_metadata(run_dir, metadata)
append_to_run_index(run_dir, summary)
```

---

### GAP-008: No Run Index Append in random_direction_control.py

**Severity:** 🟠 MAJOR

**Evidence:** Same as GAP-007.

---

### GAP-009: rv Not Computed in circuit_discovery.py

**Severity:** 🟠 MAJOR

**Evidence:**
- Pipeline does attribution patching sweep
- Does NOT compute `rv` for any patched condition
- Missing secondary geometric metric

**Fix:**
```python
# Add after patching each component:
from src.metrics.rv import compute_rv

rv_patched = compute_rv(model, tokenizer, base_text, early=5, late=27, window=16, device=device)

results.append({
    ...
    "rv_patched": rv_patched,
})
```

---

### GAP-010: Norm Logs Missing in mlp_ablation_necessity.py

**Severity:** 🟠 MAJOR

**Evidence:**
- Ablation zeros out MLP output
- No logging of before/after activation norms
- Cannot diagnose norm collapse artifacts

**Fix:** Add norm capture similar to `mlp_combined_sufficiency_test.py`.

---

### GAP-011: Norm Logs Missing in mlp_sufficiency_test.py

**Severity:** 🟠 MAJOR

**Evidence:** Same as GAP-010.

---

### GAP-012: Intervention Scope Not Specified in Legacy Pipelines

**Severity:** 🟠 MAJOR

**Affected:**
- random_direction_control.py
- circuit_discovery.py
- p1_ablation.py
- surgical_sweep.py
- kv_mechanism.py

**Evidence:** No `intervention_scope` field in metadata.

**Fix:** Add `intervention_scope` parameter to all pipelines (e.g., "all_tokens", "last_16", "BOS_only").

---

## Minor Gaps (Nice to Have)

### GAP-013: model_revision Not Logged

**Severity:** 🟡 MINOR

**Evidence:**
- HuggingFace models have revision hashes
- Not captured in metadata

**Fix:**
```python
# In get_run_metadata():
try:
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_name)
    metadata["model_revision"] = getattr(config, "_commit_hash", "unknown")
except:
    metadata["model_revision"] = "unknown"
```

---

### GAP-014: Logits Not Stored for Reproducibility

**Severity:** 🟡 MINOR

**Evidence:**
- REPRODUCIBILITY_AND_CANONICAL_SUITE.md recommends storing logits
- No `.pt` files with logits found in run directories

**Fix:**
```python
# Add to pipelines:
torch.save(out.logits.cpu(), run_dir / "logits.pt")
```

---

### GAP-015: 95% Confidence Intervals Not Reported

**Severity:** 🟡 MINOR

**Evidence:**
- GOLD_STANDARD_RESEARCH_DIRECTIVE.md requires 95% CI
- Only mean ± std reported in summaries

**Fix:**
```python
from scipy import stats

ci_low, ci_high = stats.t.interval(
    0.95,
    df=len(values)-1,
    loc=np.mean(values),
    scale=stats.sem(values)
)
summary["rv_delta_ci_95"] = [float(ci_low), float(ci_high)]
```

---

### GAP-016: Docs Claim "Stage Complete" Without Disk Evidence

**Severity:** 🟡 MINOR

**Evidence:**
- STAGE_1_FINAL_REPORT.md, STAGE_2_FINAL_REPORT.md claim completion
- Disk artifacts don't support all claims (see GAP-001, GAP-002)

**Fix:** Add "Evidence Level" markers to documentation (Disk/Code/Doc).

---

### GAP-017: Seed Not Configurable in All Pipelines

**Severity:** 🟡 MINOR

**Evidence:**
- Some pipelines hardcode seed=42
- Others read from config but default to 42

**Fix:** Ensure all pipelines read seed from config: `seed = int(params.get("seed", 42))`

---

## Prioritized Fix Order

### Immediate (Block Publication)

1. **GAP-001:** Fix RUN_INDEX.jsonl creation
2. **GAP-002:** Verify metadata.json creation
3. **GAP-003:** Implement restore_norm(M)
4. **GAP-004:** Add prompt IDs to random_direction_control.py
5. **GAP-005:** Add prompt IDs to circuit_discovery.py

### Short-Term (Before Submission)

6. **GAP-006:** Add generation_params to all pipelines
7. **GAP-007:** Add run index append to circuit_discovery.py
8. **GAP-008:** Add run index append to random_direction_control.py
9. **GAP-009:** Add rv computation to circuit_discovery.py
10. **GAP-010:** Add norm logs to mlp_ablation_necessity.py
11. **GAP-011:** Add norm logs to mlp_sufficiency_test.py
12. **GAP-012:** Add intervention_scope to legacy pipelines

### Nice-to-Have

13-17: Minor gaps (can be deferred)

---

## Verification Script

After fixing gaps, run this verification:

```bash
#!/bin/bash
# verification.sh

echo "=== Gap Verification ==="

# GAP-001: RUN_INDEX.jsonl exists
echo -n "GAP-001 (RUN_INDEX.jsonl): "
[ -f "results/RUN_INDEX.jsonl" ] && echo "✅ PASS" || echo "❌ FAIL"

# GAP-002: metadata.json exists somewhere
echo -n "GAP-002 (metadata.json): "
find results -name "metadata.json" -type f | grep -q . && echo "✅ PASS" || echo "❌ FAIL"

# GAP-003: restore_norm in code
echo -n "GAP-003 (restore_norm): "
grep -r "mode_restore_norm" src/pipelines/*.py > /dev/null && echo "✅ PASS" || echo "❌ FAIL"

# GAP-004/005: prompt IDs in control pipelines
echo -n "GAP-004/005 (prompt IDs): "
grep "get_balanced_pairs_with_ids" src/pipelines/random_direction_control.py > /dev/null && echo "✅ PASS" || echo "❌ FAIL"

# GAP-006: generation_params in metadata
echo -n "GAP-006 (gen_params): "
grep "generation_params" src/utils/run_metadata.py > /dev/null && echo "✅ PASS" || echo "❌ FAIL"

echo "=== End Verification ==="
```

---

**Report Version:** 1.0  
**Last Updated:** January 10, 2026  
**Next Review:** After gaps fixed
