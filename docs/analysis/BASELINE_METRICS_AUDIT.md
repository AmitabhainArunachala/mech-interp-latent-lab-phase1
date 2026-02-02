# Baseline Metrics Suite Audit

**Date:** January 11, 2026  
**Auditor:** Cursor Agent  
**Standard:** Nanda (2023) + R_V (Novel Geometric Metric)

---

## Executive Summary

**Finding:** The repository has individual metric implementations (`rv.py`, `logit_diff.py`, `logit_lens.py`, `mode_score.py`) but they are **NOT consistently applied** across all pipelines.

**Critical Gap:** Only **1 out of 39 registered pipelines** (`logit_lens_analysis.py`) computes the full Nanda-standard baseline suite.

---

## Required Baseline Metrics (Per Nanda 2023)

| Metric | Purpose | Linear in Residual? | Implementation |
|--------|---------|---------------------|----------------|
| **logit_diff** | Causal attribution (Nanda-standard) | ✅ YES | `src/metrics/logit_diff.py` |
| **logit_lens** | Crystallization point detection | N/A | `src/metrics/logit_lens.py` |
| **rv** | Geometric contraction (NOVEL) | ❌ No (nonlinear) | `src/metrics/rv.py` |
| **mode_score_m** | Behavioral mode classifier | ❌ No (logsumexp) | `src/metrics/mode_score.py` |
| **activation_norms** | Intervention effect diagnostic | ✅ YES | Ad-hoc (not centralized) |

### Why Both Are Needed

From Nanda (2023): *"Logit difference is a fantastic metric because it's a mostly linear function of the residual stream which makes it easy to directly attribute logit difference to individual components."*

**R_V is nonlinear** (SVD → Participation Ratio → Ratio), so it cannot be used for direct component attribution. We need **BOTH**:
- **R_V** for detecting geometric contraction (our novel finding)
- **logit_diff** for attributing effects to specific components (Nanda-standard)

---

## Pipeline Compliance Audit

### Legend

| Symbol | Meaning |
|--------|---------|
| ✅ | Metric is computed and logged |
| ⚠️ | Metric is imported but not systematically logged |
| ❌ | Metric is not computed |

### Core MLP Pipelines (Priority 1)

| Pipeline | R_V | Logit Diff | Logit Lens | Mode Score | Norms | Compliant |
|----------|:---:|:----------:|:----------:|:----------:|:-----:|:---------:|
| `mlp_ablation_necessity.py` | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ |
| `mlp_sufficiency_test.py` | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ |
| `mlp_combined_sufficiency_test.py` | ✅ | ❌ | ❌ | ✅ | ⚠️ | ❌ |
| `mlp_steering_sweep.py` | ✅ | ❌ | ❌ | ✅ | ⚠️ | ❌ |
| `mlp_ablation_position_specific.py` | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `random_direction_control.py` | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ |

### Discovery Pipelines (Priority 2)

| Pipeline | R_V | Logit Diff | Logit Lens | Mode Score | Norms | Compliant |
|----------|:---:|:----------:|:----------:|:----------:|:-----:|:---------:|
| `circuit_discovery.py` | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |
| `logit_lens_analysis.py` | ✅ | ✅ | ✅ | ❌ | ❌ | ⚠️ |
| `l27_head_analysis.py` | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `importance_sweep.py` | ⚠️ | ❌ | ❌ | ❌ | ❌ | ❌ |

### Steering Pipelines (Priority 3)

| Pipeline | R_V | Logit Diff | Logit Lens | Mode Score | Norms | Compliant |
|----------|:---:|:----------:|:----------:|:----------:|:-----:|:---------:|
| `steering.py` | ✅ | ❌ | ❌ | ⚠️ | ⚠️ | ❌ |
| `steering_analysis.py` | ✅ | ❌ | ❌ | ⚠️ | ⚠️ | ❌ |
| `steering_layer_matrix.py` | ⚠️ | ❌ | ❌ | ⚠️ | ⚠️ | ❌ |
| `surgical_sweep.py` | ❌ | ❌ | ❌ | ❌ | ⚠️ | ❌ |
| `p1_ablation.py` | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

### KV/Phase Pipelines (Priority 4)

| Pipeline | R_V | Logit Diff | Logit Lens | Mode Score | Norms | Compliant |
|----------|:---:|:----------:|:----------:|:----------:|:-----:|:---------:|
| `kv_mechanism.py` | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `kv_sufficiency_matrix.py` | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `phase0_minimal_pairs.py` | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `phase1_existence.py` | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `behavior_strict.py` | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |

---

## Compliance Summary

| Category | Total | R_V | Logit Diff | Logit Lens | Mode Score | Full Suite |
|----------|-------|-----|------------|------------|------------|------------|
| Core MLP | 6 | 6/6 | 0/6 | 0/6 | 5/6 | 0/6 |
| Discovery | 4 | 2/4 | 1/4 | 1/4 | 1/4 | 0/4 |
| Steering | 5 | 2/5 | 0/5 | 0/5 | 0/5 | 0/5 |
| KV/Phase | 5 | 5/5 | 0/5 | 0/5 | 0/5 | 0/5 |
| **TOTAL** | **20** | **15/20 (75%)** | **1/20 (5%)** | **1/20 (5%)** | **6/20 (30%)** | **0/20 (0%)** |

---

## Critical Finding

**0 out of 20 audited pipelines compute the full Nanda-standard baseline suite.**

The repository's causal claims (e.g., "L0 MLP is necessary") cannot be properly attributed to specific components because we lack `logit_diff` measurements.

---

## New Module Created: `BaselineMetricsSuite`

**Location:** `src/metrics/baseline_suite.py`

**Features:**
- Unified interface for all baseline metrics
- Lazy initialization of component metrics
- Validation of required vs recommended metrics
- Batch statistics with effect sizes and p-values

**Usage:**
```python
from src.metrics.baseline_suite import BaselineMetricsSuite

suite = BaselineMetricsSuite(model, tokenizer, device)

# Single prompt
metrics = suite.compute_all(prompt)

# Comparison (recursive vs baseline)
comparison = suite.compute_comparison(recursive_prompt, baseline_prompt)

# Batch statistics
stats = suite.compute_batch_statistics(recursive_prompts, baseline_prompts)
```

---

## Retrofit Plan

### Phase 1: Immediate (Core MLP Pipelines)

Add `BaselineMetricsSuite` to these pipelines:

1. **mlp_ablation_necessity.py** - Add logit_diff to each ablation condition
2. **mlp_sufficiency_test.py** - Add logit_diff to patched condition
3. **mlp_combined_sufficiency_test.py** - Add logit_diff
4. **circuit_discovery.py** - Already computes attribution, add logit_diff baseline

### Phase 2: Discovery Pipelines

1. **l27_head_analysis.py** - Add logit_diff trajectory
2. **importance_sweep.py** - Add logit_diff for attribution

### Phase 3: Steering Pipelines

1. **steering.py** - Add logit_diff for steering effect measurement
2. **surgical_sweep.py** - Add R_V AND logit_diff

### Integration Template

```python
# Add to imports
from src.metrics.baseline_suite import BaselineMetricsSuite

# In run function, after model load
suite = BaselineMetricsSuite(model, tokenizer, device)

# Replace individual metric calls with:
comparison = suite.compute_comparison(recursive_prompt, baseline_prompt)

# Add to results
results.append({
    **comparison.to_dict(),
    # existing fields...
})

# Add to summary
summary["baseline_metrics"] = suite.compute_batch_statistics(
    recursive_prompts, baseline_prompts
)
```

---

## Registry Enforcement (Proposed)

Update `src/pipelines/registry.py` to validate baseline metrics:

```python
@dataclass(frozen=True)
class ExperimentResult:
    summary: Dict[str, Any]
    baseline_metrics: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.baseline_metrics:
            from src.metrics.baseline_suite import BaselineMetricsSuite
            missing = BaselineMetricsSuite.validate_summary(self.baseline_metrics)
            if missing:
                import warnings
                warnings.warn(f"Missing baseline metrics: {missing}")
```

---

## Pre-Commit Check (Future)

Add `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: local
    hooks:
      - id: baseline-metrics-check
        name: Check baseline metrics in pipelines
        entry: python scripts/check_baseline_metrics.py
        language: python
        files: src/pipelines/.*\.py$
```

---

## Action Items

| Priority | Task | Est. Time |
|----------|------|-----------|
| 1 | ✅ Create `BaselineMetricsSuite` | DONE |
| 2 | Update `mlp_ablation_necessity.py` with full suite | 30 min |
| 3 | Update `mlp_sufficiency_test.py` with full suite | 30 min |
| 4 | Update `circuit_discovery.py` with logit_diff | 30 min |
| 5 | Add `baseline_metrics` to `ExperimentResult` validation | 15 min |
| 6 | Re-run canonical suite with baseline metrics | 2 hr |
| 7 | Create pre-commit hook | 30 min |

---

## References

- Nanda et al. (2023): "Logit difference is a fantastic metric because it's linear"
- nostalgebraist (2020): "Interpreting GPT: the logit lens"
- Our R_V metric: Geometric contraction (nonlinear, complementary)

---

**Audit Completed:** January 11, 2026  
**Next Review:** After retrofit of core pipelines
