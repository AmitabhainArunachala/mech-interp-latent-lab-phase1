# Industry-Grade Metrics & Reproducibility Plan

**Date:** January 15, 2026  
**Status:** PROPOSAL  
**Goal:** Publication-grade consistency without SAE training

---

## 1. Core Metrics Stack (REQUIRED)

Every canonical pipeline MUST emit these metrics:

| Metric | Module | Rationale | Compute Cost |
|--------|--------|-----------|--------------|
| **R_V** | `rv.py` | Our novel geometric signature (d = -3.56) | Low |
| **logit_diff** | `logit_diff.py` | Nanda-standard linear metric for attribution | Low |
| **mode_score_m** | `mode_score.py` | Behavioral mode classifier | Low |
| **activation_norms** | `baseline_suite.py` | Diagnostic for intervention effects | Low |

**Why these four:**
- R_V: Core claim (geometric contraction)
- logit_diff: Nanda (2023) gold standard for causal attribution
- mode_score_m: Behavioral validation (does output change?)
- activation_norms: Sanity check (are we breaking the model?)

---

## 2. Extended Metrics Stack (RECOMMENDED)

For publication-grade claims, add:

| Metric | Module | Rationale | When to Use |
|--------|--------|-----------|-------------|
| **pr_early / pr_late** | `rv.py` | Decomposed R_V for debugging | Always |
| **crystallization_layer** | `logit_lens.py` | When prediction "locks in" | Intervention studies |
| **cosine_early_late** | `extended.py` | Directional alignment (complements R_V) | Cross-arch validation |
| **spectral_gap** | `extended.py` | Eigenvalue distribution shape | Mechanism studies |
| **attention_entropy** | `extended.py` | Head focus at readout layer | Head ablation |

**NOT included (complexity without value):**
- ❌ Lyapunov exponents (requires trajectory, expensive)
- ❌ Banach fixed-point metrics (theoretical, hard to interpret)
- ❌ Full attention pattern matrices (too much data)

---

## 3. Reporting Schema (JSON Keys)

### 3.1 Summary Schema (summary.json)

Every run MUST emit `summary.json` with this structure:

```json
{
  // === IDENTITY ===
  "experiment": "string",           // REQUIRED: experiment name from registry
  "model": "string",                // REQUIRED: HuggingFace model ID
  "timestamp": "YYYYMMDD_HHMMSS",   // REQUIRED: run timestamp
  
  // === CORE METRICS ===
  "rv": {
    "mean": 0.0,                    // REQUIRED
    "std": 0.0,                     // REQUIRED
    "ci_95": [0.0, 0.0]             // REQUIRED
  },
  "logit_diff": {
    "mean": 0.0,                    // REQUIRED
    "std": 0.0,                     // RECOMMENDED
    "ci_95": [0.0, 0.0]             // RECOMMENDED
  },
  "mode_score_m": {
    "mean": 0.0,                    // REQUIRED (or null if not applicable)
    "std": 0.0
  },
  "activation_norms": {
    "early_mean": 0.0,              // REQUIRED
    "late_mean": 0.0                // REQUIRED
  },
  
  // === STATISTICS ===
  "n_samples": 0,                   // REQUIRED: sample size
  "cohens_d": 0.0,                  // REQUIRED for comparisons
  "p_value": 0.0,                   // REQUIRED for comparisons
  "t_statistic": 0.0,               // RECOMMENDED
  
  // === REPRODUCIBILITY ===
  "seed": 42,                       // REQUIRED
  "prompt_bank_version": "hash",    // REQUIRED
  "git_commit": "hash",             // REQUIRED (or "not_a_git_repo")
  
  // === PARAMS ===
  "params": {
    "early_layer": 5,               // REQUIRED
    "late_layer": 27,               // REQUIRED
    "window": 16                    // REQUIRED
  },
  
  // === ARTIFACTS ===
  "artifacts": {
    "csv": "path/to/results.csv"    // REQUIRED
  }
}
```

### 3.2 Backward Compatibility

Legacy keys still accepted (mapped internally):
- `rv_baseline_mean` → `rv.mean`
- `rv_delta_ci_95` → `rv.ci_95`
- `mean_rv.champions` → `rv.mean` (for cross-arch)

---

## 4. Run Ledger Design (RUN_INDEX.jsonl)

### 4.1 Required Fields

```json
{
  "timestamp": "20260115",
  "run_dir": "results/runs/20260115_171531_...",
  "experiment": "cross_architecture_validation",
  "model_id": "meta-llama/Meta-Llama-3-8B",
  "seed": 42,
  "prompt_bank_version": "75e7c1b8dcebc24e",
  "git_commit": "abc123...",
  "n_samples": 37,
  "rv_mean": 0.72,
  "cohens_d": -1.34,
  "p_value": 0.0087,
  "status": "success"
}
```

### 4.2 Index Query Examples

```bash
# Find all runs with strong effect
jq 'select(.cohens_d < -1.0)' results/RUN_INDEX.jsonl

# Find all Llama runs
jq 'select(.model_id | contains("Llama"))' results/RUN_INDEX.jsonl

# Find runs with specific prompt bank version
jq 'select(.prompt_bank_version == "75e7c1b8dcebc24e")' results/RUN_INDEX.jsonl
```

---

## 5. Compliance Plan

### 5.1 Enforcement Points

| Location | What to Enforce | How |
|----------|-----------------|-----|
| `registry.py` | ExperimentResult validation | `_validate_baseline_metrics()` already exists |
| `baseline_suite.py` | Metric computation | `validate_results()` method |
| `run_metadata.py` | Ledger entry | `append_to_run_index()` already exists |
| `run.py` | Summary schema | Add `validate_summary()` call |

### 5.2 Validation Flow

```
Config → Registry → Pipeline → BaselineMetricsSuite → Summary → Ledger
           ↓                          ↓                  ↓        ↓
     validate_config()         compute_all()     validate_summary()  append_to_run_index()
```

### 5.3 Warning vs Error

| Condition | Action |
|-----------|--------|
| Missing REQUIRED metric | **WARNING** (allow run to complete) |
| Missing RECOMMENDED metric | **INFO** (silent) |
| Invalid summary schema | **WARNING** |
| Missing ledger entry | **ERROR** (fail run) |

---

## 6. Implementation Steps

### Phase 1: Schema Enforcement (2 hours)

**File: `src/pipelines/run.py`**
```python
# Add after line ~80 (after summary is created)
from src.metrics.baseline_suite import BaselineMetricsSuite

def validate_summary_schema(summary: Dict[str, Any]) -> List[str]:
    """Validate summary against required schema."""
    missing = BaselineMetricsSuite.validate_summary(summary)
    if missing:
        warnings.warn(f"Summary missing: {missing}", UserWarning)
    return missing
```

### Phase 2: Ledger Guarantee (1 hour)

**File: `src/utils/run_metadata.py`**
- Already implemented ✅
- Ensure `append_to_run_index()` is called in ALL pipelines

**File: `src/pipelines/run.py`**
```python
# Add at end of main()
from src.utils.run_metadata import append_to_run_index
append_to_run_index(run_dir, summary)  # Already there, verify
```

### Phase 3: Pipeline Audit (3 hours)

Audit each canonical pipeline for compliance:

| Pipeline | Has BaselineMetricsSuite? | Has Ledger Entry? | Action |
|----------|---------------------------|-------------------|--------|
| `confound_validation.py` | ❌ | ✅ | Add suite |
| `mlp_ablation_necessity.py` | ❌ | ✅ | Add suite |
| `mlp_sufficiency_test.py` | ❌ | ✅ | Add suite |
| `head_ablation_validation.py` | ❌ | ✅ | Add suite |
| `rv_l27_causal_validation.py` | ❌ | ✅ | Add suite |
| `cross_architecture_validation.py` | ❌ | ✅ | Add suite |

### Phase 4: Integration Pattern (Template)

Add to each canonical pipeline:

```python
from src.metrics.baseline_suite import BaselineMetricsSuite

# After model load
suite = BaselineMetricsSuite(model, tokenizer, device=device)

# In measurement loop
metrics = suite.compute_all(prompt)
row["rv"] = metrics.rv
row["logit_diff"] = metrics.logit_diff
row["mode_score_m"] = metrics.mode_score_m
row["residual_norm_early"] = metrics.residual_norm_early
row["residual_norm_late"] = metrics.residual_norm_late

# In summary
summary["rv"] = {"mean": rv_mean, "std": rv_std, "ci_95": rv_ci}
summary["logit_diff"] = {"mean": ld_mean, "std": ld_std}
```

---

## 7. Risks & Tradeoffs

### 7.1 Compute Cost

| Metric | Cost per Prompt | Mitigation |
|--------|-----------------|------------|
| R_V | 2 forward passes | Already optimized |
| logit_diff | 0 extra (uses existing logits) | None needed |
| mode_score_m | 0 extra (uses existing logits) | None needed |
| logit_lens | N forward passes (N = layers) | Make optional |
| extended metrics | 2 extra forward passes | Make optional |

**Recommendation:** Core metrics add ~10% overhead. Extended metrics add ~50%. Make extended opt-in.

### 7.2 Breaking Changes

| Change | Impact | Migration |
|--------|--------|-----------|
| New summary schema | Old summaries won't validate | Add `schema_version` field |
| Ledger required | Runs without ledger fail | Backfill existing runs |
| BaselineMetricsSuite required | Pipelines need update | Gradual rollout |

### 7.3 Complexity vs Value

| Metric | Complexity | Value | Include? |
|--------|------------|-------|----------|
| R_V | Low | High (core claim) | ✅ REQUIRED |
| logit_diff | Low | High (Nanda standard) | ✅ REQUIRED |
| mode_score_m | Low | Medium (behavioral) | ✅ REQUIRED |
| activation_norms | Low | Medium (diagnostic) | ✅ REQUIRED |
| logit_lens | Medium | Medium | ⚠️ RECOMMENDED |
| spectral_stats | Medium | Low | ❌ OPTIONAL |
| attention_entropy | Medium | Low | ❌ OPTIONAL |

---

## 8. Definition of Done

### PR Checklist

- [ ] All canonical pipelines use `BaselineMetricsSuite`
- [ ] All runs append to `RUN_INDEX.jsonl`
- [ ] Summary schema validation in `run.py`
- [ ] `docs/METRICS_REFERENCE.md` updated
- [ ] One test run per pipeline passes validation
- [ ] No new warnings in CI

### Verification Command

```bash
# Verify all recent runs have required metrics
jq 'select(.rv == null or .cohens_d == null)' results/RUN_INDEX.jsonl | wc -l
# Should output: 0
```

---

## 9. Current State Assessment

### What's Working ✅
- `BaselineMetricsSuite` exists and computes all metrics
- `RUN_INDEX.jsonl` exists and is being populated
- `run_metadata.py` handles ledger entries
- `ExperimentResult` validates baseline metrics

### What's Missing ❌
- Canonical pipelines don't use `BaselineMetricsSuite`
- Summary schema not enforced
- Extended metrics not consistently computed
- No schema version for backward compatibility

### Priority Order
1. **HIGH:** Add `BaselineMetricsSuite` to canonical pipelines
2. **MEDIUM:** Enforce summary schema in `run.py`
3. **LOW:** Add extended metrics as opt-in

---

*This plan designed for mech-interp-latent-lab-phase1 repo, January 2026*
