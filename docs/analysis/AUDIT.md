# Repository Audit: mech-interp-latent-lab-phase1

**Date:** January 11, 2026
**Auditor:** Claude Code (Opus)
**Purpose:** Comprehensive inventory, gap analysis, and remediation plan

---

## 1. File Inventory

### 1.1 Source Code Structure (`src/`)

#### Metrics (`src/metrics/`) - 8 files
| File | Purpose | Status | Notes |
|------|---------|--------|-------|
| `rv.py` | R_V metric (PR_late / PR_early) | CANONICAL | Novel geometric contraction metric |
| `logit_diff.py` | Logit difference (Nanda-standard) | CANONICAL | Linear metric for attribution |
| `logit_lens.py` | Layer-wise prediction probing | CANONICAL | Crystallization detection |
| `mode_score.py` | Behavioral mode classifier | CANONICAL | logsumexp-based |
| `baseline_suite.py` | Unified metrics interface | CANONICAL | Combines all metrics |
| `__init__.py` | Module exports | SUPPORT | - |
| `pr_trajectory.py` | Per-layer PR tracking | EXPLORATORY | Not in suite |
| `activation_norms.py` | Norm diagnostics | AD-HOC | Needs centralization |

#### Core (`src/core/`) - 8 files
| File | Purpose | Status | Notes |
|------|---------|--------|-------|
| `hooks.py` | V-projection capture hooks | CANONICAL | Used by rv.py |
| `activations.py` | Activation patching utilities | CANONICAL | - |
| `head_specific_patching.py` | H18/H26 V_proj patching | CANONICAL | - |
| `kv_cache.py` | KV cache manipulation | CANONICAL | - |
| `steering.py` | Steering vector application | CANONICAL | - |
| `model_loader.py` | Model loading utilities | SUPPORT | - |
| `utils.py` | General utilities | SUPPORT | - |
| `__init__.py` | Module exports | SUPPORT | - |

#### Pipelines (`src/pipelines/`) - 59 files

**TIER 1: Canonical (Produce publication data)**
| Pipeline | Experiment | Sample Size | Has Full Suite | Notes |
|----------|------------|-------------|----------------|-------|
| `rv_l27_causal_validation.py` | L27 activation patching | n=45 | NO (R_V only) | **STRONGEST RESULT** d=-3.56 |
| `c2_rv_measurement.py` | C2 config R_V + behavior | n=20 | PARTIAL | Recently updated with BaselineMetricsSuite |
| `confound_validation.py` | 4 control conditions | n=45 | NO | Random/shuffled/wrong-layer |
| `random_direction_control.py` | Random direction control | n=20 | NO | Directional specificity |
| `mlp_ablation_necessity.py` | MLP necessity tests | n=20 | R_V+mode | Missing logit_diff |
| `mlp_sufficiency_test.py` | MLP sufficiency tests | n=20 | R_V+mode | Missing logit_diff |

**TIER 2: Exploratory (Need validation)**
| Pipeline | Purpose | Status |
|----------|---------|--------|
| `logit_lens_analysis.py` | Crystallization tracking | ONLY ONE WITH LOGIT_DIFF |
| `circuit_discovery.py` | Circuit hunting | Mode score only |
| `l27_head_analysis.py` | Head-level analysis | R_V only |
| `kv_mechanism.py` | KV swap experiments | R_V only |
| `kv_sufficiency_matrix.py` | KV sufficiency matrix | R_V only |
| `steering.py` | Basic steering | R_V + mode |
| `steering_analysis.py` | Steering analysis | R_V + mode |
| `mlp_combined_sufficiency_test.py` | Combined MLP tests | R_V + mode |

**TIER 3: Experimental/Deprecated**
| Pipeline | Status | Notes |
|----------|--------|-------|
| `kitchen_sink.py` | DEPRECATED | Superseded by focused pipelines |
| `unified_layer_map.py` | EXPLORATORY | Broad sweep |
| `h31_investigation.py` | ABANDONED | H31 hypothesis disproven |
| `h31_ablation_causal.py` | ABANDONED | - |
| `anthropic_level_investigation.py` | EXPLORATORY | - |
| `comprehensive_circuit_analysis.py` | EXPLORATORY | - |
| `eigenstate_direction_finder.py` | EXPLORATORY | - |
| `hysteresis_patching.py` | EXPLORATORY | - |
| `causal_mechanism_hunt.py` | EXPLORATORY | - |

**TIER 4: Support/Utility**
| Pipeline | Purpose |
|----------|---------|
| `registry.py` | Experiment dispatch |
| `run.py` | CLI runner |
| `prompt_bank_audit.py` | Prompt validation |
| `retrocompute_mode_score.py` | Backfill mode scores |
| `verification_sweep.py` | Verification utilities |

### 1.2 Configurations (`configs/`) - 54 files

**Active Configs (Used in recent runs)**
| Config | Pipeline | Status |
|--------|----------|--------|
| `c2_rv_measurement.json` | C2 R_V measurement | ACTIVE |
| `c2_ablation_no_cascade.json` | C2 ablation test | NEW |
| `c2_ablation_no_kv.json` | C2 ablation test | NEW |
| `c2_ablation_no_steering.json` | C2 ablation test | NEW |
| `rv_l27_causal_validation.json` | L27 validation | CANONICAL |
| `confound_validation.json` | Confound tests | CANONICAL |
| `random_direction_control_l3_targeted.json` | Random control | ACTIVE |
| `mlp_ablation_necessity_l0.json` | L0 MLP necessity | CANONICAL |
| `mlp_sufficiency_l0.json` | L0 sufficiency | CANONICAL |

**Orphaned Configs (No recent results)**
| Config | Notes |
|--------|-------|
| `behavioral_grounding_*.json` (9 files) | Dec 2025, not rerun |
| `path_patching_mechanism_*.json` (8 files) | Superseded |
| `hysteresis_patching.json` | Exploratory, not used |
| `l27_head_analysis.json` | Needs update |
| `kv_sweep_*.json` (4 files) | Broad sweeps |

### 1.3 Results Directories

**Active Results (Have summary.json)**
| Directory | Runs | Last Modified | Notes |
|-----------|------|---------------|-------|
| `results/phase1_mechanism/runs/` | 75+ | Jan 2026 | Core experiments |
| `results/gold_standard/runs/` | 3 | Dec 2025 | Gold standard runs |
| `results/confound_validation/runs/` | 2 | Dec 2025 | Control validation |
| `results/kv_sufficiency_matrix/runs/` | 3 | Dec 2025 | KV experiments |
| `results/runs/` | 15+ | Dec 2025 | Mixed experiments |
| `runpod_sync_20260105_*/` | 10+ | Jan 2026 | GPU sync results |

**Summary.json Count**: 100+ results files found

### 1.4 Prompt Bank (`REUSABLE_PROMPT_BANK/`)

| Module | Prompts | Purpose |
|--------|---------|---------|
| `dose_response.py` | ~80 | L1→L5 recursion levels |
| `baselines.py` | ~60 | Math, factual, creative |
| `confounds.py` | ~60 | Complexity without recursion |
| `generality.py` | ~40 | Cross-domain tests |
| `kill_switch.py` | ~20 | Anti-recursion tests |
| `sampling.py` | - | Sampling utilities |
| **TOTAL** | **~260** | - |

---

## 2. Canonical Pipelines (The 6 That Matter)

Based on the audit, these are the pipelines that should be brought to publication standard:

### 2.1 Primary (Must have n≥100)

| # | Pipeline | Current n | Target n | Missing Metrics |
|---|----------|-----------|----------|-----------------|
| 1 | `rv_l27_causal_validation.py` | 45 | 100 | logit_diff, norms |
| 2 | `confound_validation.py` | 45 | 100 | logit_diff, norms |
| 3 | `c2_rv_measurement.py` | 20 | 100 | Currently being updated |

### 2.2 Secondary (Must have n≥30)

| # | Pipeline | Current n | Target n | Missing Metrics |
|---|----------|-----------|----------|-----------------|
| 4 | `mlp_ablation_necessity.py` | 20 | 30 | logit_diff |
| 5 | `mlp_sufficiency_test.py` | 20 | 30 | logit_diff |
| 6 | `random_direction_control.py` | 20 | 30 | logit_diff |

---

## 3. Gap Analysis

### 3.1 Metric Coverage (Per BASELINE_METRICS_AUDIT.md)

| Pipeline | R_V | Logit Diff | Logit Lens | Mode Score | Norms | Full Suite |
|----------|:---:|:----------:|:----------:|:----------:|:-----:|:----------:|
| rv_l27_causal_validation | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| confound_validation | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| c2_rv_measurement | ✅ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ |
| mlp_ablation_necessity | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ |
| mlp_sufficiency_test | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ |
| random_direction_control | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ |
| **TOTAL COMPLIANT** | 6/6 | 1/6 | 1/6 | 4/6 | 1/6 | **0/6** |

### 3.2 Statistical Rigor

| Requirement | Current State | Gap |
|-------------|---------------|-----|
| n ≥ 100 for solid claims | max n=45 | Need 55+ more pairs |
| p-value reported | Some runs | Not systematic |
| Cohen's d reported | rv_l27_causal only | Add to all |
| 95% CI reported | None | Add to all |
| Effect size interpretation | Informal | Standardize |

### 3.3 Documentation Gaps

| Document | Status | Gap |
|----------|--------|-----|
| MISTRAL_L27_CAUSAL_VALIDATION_COMPLETE.md | EXISTS | Needs n=100 update |
| BASELINE_METRICS_AUDIT.md | EXISTS | Good, needs updates |
| PHASE1_FINAL_REPORT.md | EXISTS | Needs refresh |
| Per-pipeline README | MISSING | Add for each canonical |
| Reproducibility guide | MISSING | Add setup + run instructions |

---

## 4. Cleanup Candidates

### 4.1 Files to Archive (Move to `archive/`)

**Deprecated Pipelines**
- `src/pipelines/kitchen_sink.py` - Superseded
- `src/pipelines/h31_investigation.py` - Abandoned
- `src/pipelines/h31_ablation_causal.py` - Abandoned
- `src/pipelines/unified_layer_map.py` - Superseded by focused sweeps
- `src/pipelines/hysteresis_patching.py` - Not in registry

**Deprecated Prompt Bank**
- `n300_mistral_test_prompt_bank.py` - Points to REUSABLE_PROMPT_BANK

**Old Results**
- `results/dec13_kitchen_sink/` - Old exploratory
- `results/circuit_hunt_v2_focused/` - Superseded
- `results/h31_validation/` - Abandoned hypothesis

### 4.2 Files to Delete

- None recommended (keep for reproducibility)

### 4.3 Files to Consolidate

| Current | Consolidate Into |
|---------|------------------|
| `configs/path_patching_mechanism_*.json` (8 files) | Single `configs/gold/rv_l27_validation.json` |
| `configs/behavioral_grounding_*.json` (9 files) | Archive or delete |
| Multiple `mlp_*` configs | Consolidate into 3-4 canonical configs |

---

## 5. Standards Compliance Matrix

### 5.1 Per Nanda (2023) + Extended Standards

| Standard | Description | Compliant Pipelines | Gap |
|----------|-------------|---------------------|-----|
| **Logit Diff** | Linear metric for attribution | 1/6 (logit_lens_analysis) | 5 pipelines |
| **Controls** | Random/shuffled/wrong-layer | 1/6 (confound_validation) | Need in all |
| **Sample Size** | n≥100 for claims | 0/6 | All need increase |
| **Effect Size** | Cohen's d reported | 1/6 | 5 pipelines |
| **P-value** | p < 0.001 threshold | 1/6 | 5 pipelines |
| **95% CI** | Confidence intervals | 0/6 | All pipelines |

### 5.2 Proposed Standards (Higher than Field)

| Standard | Requirement | Rationale |
|----------|-------------|-----------|
| **OBSERVATION** | n ≥ 20 | Initial exploration |
| **SUGGESTION** | n ≥ 30 | Preliminary finding |
| **SOLID CLAIM** | n ≥ 100 | Publication-grade |
| **Metrics** | R_V + logit_diff + mode_score | Both geometric and linear |
| **Statistics** | t-test, Cohen's d, 95% CI | Full reporting |
| **Controls** | Random + shuffled + wrong-layer | 3-pillar validation |

---

## 6. TODO List (Prioritized)

### IMMEDIATE (This Week)

| Priority | Task | Est. Effort | Blocked By |
|----------|------|-------------|------------|
| 1 | Add `BaselineMetricsSuite` to `rv_l27_causal_validation.py` | 1 hr | None |
| 2 | Add `BaselineMetricsSuite` to `confound_validation.py` | 1 hr | None |
| 3 | Run n=100 L27 validation on GPU | 4 hr | GPU access |
| 4 | Run n=100 confound validation on GPU | 4 hr | GPU access |
| 5 | Update `c2_rv_measurement.py` suite integration | Done | - |

### SHORT-TERM (This Sprint)

| Priority | Task | Est. Effort | Blocked By |
|----------|------|-------------|------------|
| 6 | Add logit_diff to `mlp_ablation_necessity.py` | 30 min | None |
| 7 | Add logit_diff to `mlp_sufficiency_test.py` | 30 min | None |
| 8 | Run n=30 MLP necessity tests | 2 hr | GPU |
| 9 | Run n=30 MLP sufficiency tests | 2 hr | GPU |
| 10 | Create archive/ directory and move deprecated files | 30 min | None |

### MEDIUM-TERM (Next 2 Weeks)

| Priority | Task | Est. Effort | Blocked By |
|----------|------|-------------|------------|
| 11 | Consolidate configs into gold/ directory | 2 hr | None |
| 12 | Write reproducibility guide | 2 hr | None |
| 13 | Create per-pipeline README files | 3 hr | None |
| 14 | Add pre-commit hook for baseline metrics | 1 hr | None |
| 15 | Update PHASE1_FINAL_REPORT.md with n=100 results | 2 hr | Runs complete |

### LONG-TERM (Publication Prep)

| Priority | Task | Est. Effort | Blocked By |
|----------|------|-------------|------------|
| 16 | Multi-token generation experiment | 8 hr | n=100 runs |
| 17 | Cross-model replication (Qwen-7B) | 8 hr | Mistral complete |
| 18 | Write R_V paper Methods section | 4 hr | All data |
| 19 | Create publication figures | 4 hr | All data |
| 20 | Paper submission | - | All above |

---

## 7. Registry Validation

### 7.1 Registered vs Actual Pipelines

**In Registry (44 entries)**:
```python
get_registry() returns:
- phase0_minimal_pairs, phase0_metric_targets
- phase1_existence, rv_l27_causal_validation
- l27_head_analysis, path_patching_mechanism
- behavioral_grounding, behavioral_grounding_batch
- confound_validation, kv_sufficiency_matrix
- ... (34 more)
```

**Missing from Registry (Should add)**:
- `c2_rv_measurement.py` - Needs registration

**In Registry but Deprecated**:
- `kitchen_sink` - Consider removal
- `hysteresis` - Consider removal

---

## 8. Key Findings Summary

### Strengths

1. **R_V metric validated**: Cohen's d = -3.56, p < 10⁻⁶ (n=45)
2. **4-pillar control structure**: Random, shuffled, wrong-layer, dose-response
3. **BaselineMetricsSuite exists**: Unified metrics interface ready
4. **Prompt bank well-organized**: ~260 prompts with clear taxonomy
5. **Registry pattern**: Clean experiment dispatch system

### Critical Gaps

1. **0/6 canonical pipelines have full metric suite**
2. **No pipeline has n ≥ 100** (max is 45)
3. **logit_diff missing from 5/6 pipelines**
4. **No 95% confidence intervals reported**
5. **~20 orphaned configs** need cleanup

### Action Required

1. **Immediate**: Add BaselineMetricsSuite to 2 canonical pipelines
2. **This week**: Run n=100 on GPU for main results
3. **This sprint**: Cleanup configs, add documentation
4. **Publication**: Multi-token experiment + cross-model replication

---

## Appendix A: File Counts

| Directory | Python Files | JSON Configs | Result Dirs |
|-----------|-------------|--------------|-------------|
| `src/metrics/` | 8 | - | - |
| `src/core/` | 8 | - | - |
| `src/pipelines/` | 59 | - | - |
| `configs/` | - | 54 | - |
| `results/` | - | - | 25+ |
| **TOTAL** | **75+** | **54** | **100+ runs** |

## Appendix B: Strongest Results to Date

| Experiment | n | Cohen's d | p-value | Transfer |
|------------|---|-----------|---------|----------|
| L27 Causal Validation | 45 | -3.56 | < 10⁻⁶ | 117.8% |
| Random Control | 45 | +71.6% | < 10⁻⁶ | (opposite) |
| Shuffled Control | 45 | -61% reduction | < 0.01 | - |
| Wrong Layer (L21) | 45 | 0 | 0.49 | None |

---

**Audit Complete**: January 11, 2026
**Next Review**: After n=100 runs complete
