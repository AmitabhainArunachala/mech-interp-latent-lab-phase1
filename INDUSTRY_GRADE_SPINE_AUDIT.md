# Industry-Grade Spine Audit: Full Repository Assessment

**Date:** January 5, 2026  
**Version:** 1.1  
**Status:** VERIFIED-AWARE (Code vs Disk vs Docs reconciled)

---

## Executive Summary

This audit reconstructs the true end-to-end experimental pipeline ("central spine") of the mechanistic interpretability research repository, identifies compliance gaps against industry-grade standards, and provides actionable remediation steps.

**Key Findings (v1.1 verified):**
1. **Pipeline Spine (in code):** A broad “spine” exists in `src/pipelines/` (phase0 → mechanism → steering/KV/late-layer) but **on-disk run artifacts do not currently support “stage-complete” claims** for the industry-grade suite.
2. **Metric Contract:** Strong drift across phases: many legacy runs log **text-space** metrics (`recursion_score`, `behavior_strict`) while the industry contract requires **logit-space** `mode_score_m`.
3. **Run Ledger Reality:** `results/run_index.csv` exists, but `results/RUN_INDEX.jsonl` is **absent on disk** in this workspace (despite multiple docs claiming it exists). `append_to_run_index()` exists in code, but **there is no on-disk evidence it has been executed successfully** (no `metadata.json`, no `RUN_INDEX.jsonl`).
4. **Docs vs Disk Contradictions:** Several “Stage 0/1 complete” markdown reports claim artifacts (e.g., `metadata.json`, `summary.json`, `results/RUN_INDEX.jsonl`) that are **not present** under `results/` at audit time.

### Evidence Levels Used in This Audit

- **Disk (✅ strongest)**: File(s) exist under `results/` now.
- **Code (⚠️ plausible)**: Implementation exists in `src/`, but no confirming artifacts on disk.
- **Docs (⚠️ weakest)**: Claimed in markdown, but not confirmed by disk or code.

---

## 1. Pipeline Spine Map

### Phase Dependency Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 0: METRIC VALIDATION (Foundation) ⚠️ CODE EXISTS / RUNS UNVERIFIED     │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. phase0_metric_targets (R_V computation validation)                       │
│ 2. phase0_minimal_pairs (R_V separation: recursive vs baseline)             │
│ Dependencies: None                                                          │
│ Output (code intent): Validated R_V metric, PR computation at L5/L27         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 1A: CAUSAL DISCOVERY (Core Mechanism) ⚠️ CODE PRESENT / ARTIFACTS MIXED│
├─────────────────────────────────────────────────────────────────────────────┤
│ 3. ⚠️ circuit_discovery - Attribution patching sweep                        │
│    Script: src/pipelines/circuit_discovery.py                               │
│    Config: configs/gold/11_circuit_discovery.json                           │
│    Metrics: Attribution score (logit diff), Mode Score M                    │
│    Status: EXISTS but not re-run with standardized metadata                 │
│                                                                             │
│ 4. ⚠️ mlp_ablation_necessity (L0, L1, L2, L3)                               │
│    Script: src/pipelines/mlp_ablation_necessity.py                          │
│    Config: configs/mlp_ablation_necessity_l*.json                           │
│    Metrics: R_V delta, Mode Score M, p-values, effect sizes                 │
│    Status: CODE UPDATED for industry-grade metadata; DISK EVIDENCE MISSING   │
│                                                                             │
│ 5. ⚠️ mlp_sufficiency_test (L0)                                             │
│    Script: src/pipelines/mlp_sufficiency_test.py                            │
│    Config: configs/mlp_sufficiency_l0.json                                  │
│    Metrics: R_V restoration, Mode Score M                                   │
│    Status: CODE UPDATED; DISK EVIDENCE MISSING                               │
│                                                                             │
│ 6. ⚠️ mlp_combined_sufficiency_test (L0+L1)                                 │
│    Script: src/pipelines/mlp_combined_sufficiency_test.py                   │
│    Config: configs/combined_mlp_sufficiency_l0_l1.json                      │
│    Metrics: R_V restoration, Mode Score M, norm logs                        │
│    Status: CODE UPDATED; DISK EVIDENCE MISSING                               │
│                                                                             │
│ 7. ⚠️ position_specific_ablation (L0)                                       │
│    Script: src/pipelines/mlp_ablation_position_specific.py                  │
│    Config: configs/position_specific_l0_ablation.json                       │
│    Metrics: R_V delta by position, Mode Score M                             │
│    Status: EXISTS IN CODE; DISK STATUS UNKNOWN                               │
│                                                                             │
│ Dependencies: Phase 0                                                       │
│ Output (docs/claims): L0/L1/L3 necessary, L2 not necessary, L0+L1 anti-sufficient
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 1B: TRANSFER & STEERING ❌ MISSING FROM STAGE 2                       │
├─────────────────────────────────────────────────────────────────────────────┤
│ 8. ⚠️ mlp_steering_sweep                                                    │
│    Script: src/pipelines/mlp_steering_sweep.py                              │
│    Config: configs/mlp_steering_sweep_corrected.json                        │
│    Metrics: R_V delta, Mode Score M, coherence                              │
│    Finding: L3-L4 optimal for steering (not L0)                             │
│    Status: NOT RE-RUN in Stage 2                                            │
│                                                                             │
│ 9. ⚠️ random_direction_control                                              │
│    Script: src/pipelines/random_direction_control.py                        │
│    Config: configs/random_direction_control_l3_targeted.json                │
│    Metrics: R_V delta (random vs true), statistical comparison              │
│    Finding: L2 steering = artifact                                          │
│    Status: NOT RE-RUN in Stage 2                                            │
│                                                                             │
│ Dependencies: Phase 1A                                                      │
│ Output: True causal transfer layers, artifact validation                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 1C: LATE-LAYER ATTENTION ❌ MISSING FROM STAGE 2                      │
├─────────────────────────────────────────────────────────────────────────────┤
│ 10. ⚠️ p1_ablation                                                          │
│     Script: src/pipelines/p1_ablation.py                                    │
│     Config: configs/gold/17_p1_ablation.json                                │
│     Metrics: Recursion score, Mode Score M (NOT COMPUTED!)                  │
│     Finding: V-Proj primary, Residual amplifier, KV necessary               │
│     Status: EXISTS but uses legacy recursion_score, no mode_score_m         │
│                                                                             │
│ 11. ⚠️ surgical_sweep                                                       │
│     Script: src/pipelines/surgical_sweep.py                                 │
│     Config: configs/gold/15_surgical_sweep.json                             │
│     Metrics: recursion_score (regex), coherence (NOT mode_score_m!)         │
│     Finding: C2 config → 0.15 recursion, 20% success                        │
│     Status: EXISTS but uses legacy metrics, no mode_score_m                 │
│                                                                             │
│ Dependencies: Phase 1A, 1B                                                  │
│ Output: Component hierarchy, optimal steering config                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 1D: KV CACHE MECHANISM ❌ MISSING FROM STAGE 2                        │
├─────────────────────────────────────────────────────────────────────────────┤
│ 12. ⚠️ kv_mechanism                                                         │
│     Script: src/pipelines/kv_mechanism.py                                   │
│     Config: configs/gold/08_kv_mechanism.json                               │
│     Metrics: R_V transfer, geometry restoration (NO mode_score_m!)          │
│     Finding: KV replacement → 94% geometry transfer                         │
│     Status: EXISTS but uses R_V only, no mode_score_m                       │
│                                                                             │
│ 13. ⚠️ kv_sufficiency_matrix (OPTIONAL)                                     │
│     Script: src/pipelines/kv_sufficiency_matrix.py                          │
│     Config: configs/kv_sufficiency_matrix.json                              │
│     Metrics: Behavior transfer, controls                                    │
│     Status: May be redundant with kv_mechanism                              │
│                                                                             │
│ Dependencies: Phase 1C                                                      │
│ Output: KV cache geometry/content transfer mechanism                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Canonical Experiment Sets

### Mechanism Suite (Priority 1 - Causal Claims)

| ID | Experiment | Script | Config | Status | Controls |
|----|------------|--------|--------|--------|----------|
| M1 | L0 Necessity | `mlp_ablation_necessity.py` | `mlp_ablation_necessity_l0.json` | ✅ | Baseline, Ablation |
| M2 | L1 Necessity | `mlp_ablation_necessity.py` | `mlp_ablation_necessity_l1.json` | ✅ | Baseline, Ablation |
| M3 | L2 Necessity | `mlp_ablation_necessity.py` | `mlp_ablation_necessity_l2.json` | ✅ | Baseline, Ablation |
| M4 | L3 Necessity | `mlp_ablation_necessity.py` | `mlp_ablation_necessity_l3.json` | ✅ | Baseline, Ablation |
| M5 | L0 Sufficiency | `mlp_sufficiency_test.py` | `mlp_sufficiency_l0.json` | ✅ | Patch only |
| M6 | L0+L1 Anti-Sufficiency | `mlp_combined_sufficiency_test.py` | `combined_mlp_sufficiency_l0_l1.json` | ✅ | Patch only |
| M7 | L0 Position-Specific | `mlp_ablation_position_specific.py` | `position_specific_l0_ablation.json` | ⏳ | BOS/first4/last16/all |
| M8 | Random Direction Control | `random_direction_control.py` | `random_direction_control_l3.json` | ❌ | Random vs True |
| M9 | KV Interaction | `kv_mechanism.py` | `08_kv_mechanism.json` | ❌ | Swap vs Baseline |

### Pipeline Suite (Priority 2 - Discovery)

| ID | Experiment | Script | Config | Status | Purpose |
|----|------------|--------|--------|--------|---------|
| P1 | Circuit Discovery | `circuit_discovery.py` | `11_circuit_discovery.json` | ❌ | Attribution sweep |
| P2 | P1 Ablation | `p1_ablation.py` | `17_p1_ablation.json` | ❌ | Component hierarchy |
| P3 | Surgical Sweep | `surgical_sweep.py` | `15_surgical_sweep.json` | ❌ | Optimal config |
| P4 | Steering Layer Matrix | `steering_layer_matrix.py` | `09_layer_matrix.json` | ❌ | Layer importance |
| P5 | MLP Steering Sweep | `mlp_steering_sweep.py` | `mlp_steering_sweep_corrected.json` | ❌ | Transfer layers |

---

## 3. Compliance Matrix

### Industry-Grade Standard Checklist (v1.1: Disk-verified when possible)

| Pipeline | Prompt IDs | Bank Hash | Seed | Gen Params | Intervention Scope | Eval Window | mode_score_m | restore_norm | rv | Norm Logs | Run Index |
|----------|:----------:|:---------:|:----:|:----------:|:-----------------:|:-----------:|:------------:|:------------:|:--:|:---------:|:---------:|
| **mlp_ablation_necessity** | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **mlp_sufficiency_test** | ❌ | ⚠️ | ⚠️ | ❌ | ❌ | ❌ | ❌ | ❌ | ⚠️ | ❌ | ❌ |
| **mlp_combined_sufficiency** | ❌ | ⚠️ | ⚠️ | ❌ | ❌ | ❌ | ❌ | ❌ | ⚠️ | ❌ | ❌ |
| **circuit_discovery** | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ⚠️ | ❌ | ❌ | ❌ | ❌ |
| **p1_ablation** | ❌ | ❌ | ⚠️ | ⚠️ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **surgical_sweep** | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **kv_mechanism** | ❌ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ |
| **layer_sweep** | ❌ | ❌ | ⚠️ | ⚠️ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **steering_analysis** | ❌ | ❌ | ❌ | ⚠️ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

**Legend:** ✅ Disk-verified compliant | ⚠️ Code/doc suggests partial | ❌ Missing/Unverified on disk

### Critical Gaps (v1.1 reconciled)

1. **Docs contradict disk for run ledger:** multiple docs claim `results/RUN_INDEX.jsonl` exists, but it does not in this workspace.
2. **No disk evidence of industry-grade metadata:** no `metadata.json` found anywhere under `results/` at audit time.
3. **No disk evidence of industry-grade MLP suite outputs:** the only MLP run dir found (`results/runs/20260105_185135_mlp_ablation_necessity/`) contains `config.json` and prompt bank version files only (no CSV/summary).
4. **mode_score_m missing across legacy pipelines:** key legacy mechanisms (`p1_ablation`, `surgical_sweep`, `kv_mechanism`, `layer_sweep`, `l27_head_analysis`) do not appear to compute `mode_score_m` (string-level evidence).
5. **restore_norm (ModeScore restoration normalization) not implemented repo-wide.**

---

## 4. Metrics Alignment Report

### Current Metric Implementations

| Metric | File | Definition | Used In |
|--------|------|------------|---------|
| **mode_score_m** | `src/metrics/mode_score.py` | `logsumexp(logits[R]) - logsumexp(logits[T])` | `mlp_ablation_necessity`, `circuit_discovery` |
| **rv** | `src/metrics/rv.py` | `PR_late / PR_early` | All pipelines |
| **behavior_strict** | `src/metrics/behavior_strict.py` | Tiered gatekeeper (text-based) | `behavior_strict` pipeline |
| **behavior_states** | `src/metrics/behavior_states.py` | Phenomenological labeling (LEGACY) | None (deprecated) |
| **recursion_score** | inline in `p1_ablation.py`, `surgical_sweep.py` | Regex pattern matching | `p1_ablation`, `surgical_sweep` |

### Metric Drift Analysis

```
PROBLEM: Three incompatible behavioral metrics exist:

1. mode_score_m (logit-space, PRIMARY)
   - Defined: src/metrics/mode_score.py
   - Used by: mlp_ablation_necessity, circuit_discovery
   - Pros: Logit-level, interpretable, consistent
   - Cons: Requires baseline logits for comparison

2. recursion_score (text-space, LEGACY)
   - Defined: inline in multiple files (NOT centralized)
   - Used by: p1_ablation, surgical_sweep
   - Pros: Simple, fast
   - Cons: Regex-based, not comparable to mode_score_m

3. behavior_strict (text-space, SECONDARY)
   - Defined: src/metrics/behavior_strict.py
   - Used by: behavior_strict pipeline
   - Pros: Nuanced (gates + scoring)
   - Cons: Not comparable to mode_score_m

RECOMMENDATION: 
- Use mode_score_m as PRIMARY for all causal claims
- Use rv as SECONDARY geometry signature
- Deprecate inline recursion_score
- Use behavior_strict only for qualitative validation
```

### Required Metric Standardization

| Experiment | Current Metric | Required Metric | Change Needed |
|------------|----------------|-----------------|---------------|
| p1_ablation | recursion_score (inline) | mode_score_m | Add ModeScoreMetric computation |
| surgical_sweep | recursion_score (inline) | mode_score_m | Add ModeScoreMetric computation |
| kv_mechanism | rv only | rv + mode_score_m | Add ModeScoreMetric computation |
| layer_sweep | keyword count (inline) | mode_score_m | Add ModeScoreMetric computation |
| circuit_discovery | mode_score_m | mode_score_m | ✅ Already compliant |

---

## 5. Reproducibility Gaps

### Infrastructure Status (v1.1: separate “exists in code” vs “exists on disk”)

| Component | Status | Location | Issue |
|-----------|--------|----------|-------|
| PromptLoader | ✅ Working | `prompts/loader.py` | None |
| bank.json | ✅ Present | `prompts/bank.json` | 754 prompts |
| run_metadata.py | ✅ Exists | `src/utils/run_metadata.py` | Not reflected in on-disk run artifacts (no metadata.json found) |
| run_index.py | ✅ Working | `src/utils/run_index.py` | Creates CSV only |
| RUN_INDEX.jsonl | ❌ MISSING | Should be at `results/RUN_INDEX.jsonl` | File doesn't exist despite documentation |
| run_index.csv | ✅ Present | `results/run_index.csv` | 36 runs indexed |

### Missing Metadata in Runs (disk-verified)

| Field | Required | Present In |
|-------|----------|------------|
| `prompt_ids.recursive` | ✅ | ❌ (no evidence on disk) |
| `prompt_ids.baseline` | ✅ | ❌ (no evidence on disk) |
| `prompt_bank_version` | ✅ | Most pipelines |
| `git_commit` | ✅ | ❌ (no evidence on disk) |
| `model_id` | ✅ | Most pipelines |
| `seed` | ✅ | Most pipelines |
| `eval_window` | ✅ | ❌ (no evidence on disk) |
| `intervention_scope` | ✅ | ❌ (no evidence on disk) |
| `generation_params` | ❌ | None (max_new_tokens, temp not standardized) |
| `model_revision` | ❌ | None |

### PR Plan: Minimum Reproducibility Fixes (ordering updated for v1.1)

1. **Prove the ledger pipeline works end-to-end (single run):** run one MLP pipeline through the canonical runner and confirm it emits `metadata.json`, `summary.json`, and creates/appends `results/RUN_INDEX.jsonl`.
2. **Add generation params to metadata** - Log `max_new_tokens`, `temperature`, `top_p`
3. **Standardize intervention_scope** - Add to all patching pipelines
4. **Add model_revision** - Extract from transformers model config

---

## 6. Bloated/Duplicated Files List

### Redundant Meta Files (Recommend Archive)

| File | Issue | Action |
|------|-------|--------|
| `DEC*_*.md` (20+ files) | Daily session logs, outdated | Archive to `archive/session_logs/` |
| `BREAKTHROUGH_*.md` | Superseded by STAGE reports | Archive |
| `ITERATION_V*.md` | Superseded by final reports | Archive |
| `*_ANALYSIS.md` (duplicates) | Multiple versions exist | Keep latest, archive others |
| `AGENT_*.md` | Agent-specific, not canonical | Archive |
| `*_STATUS.md` | Transient status, outdated | Delete or archive |

### Inconsistent Findings Files

| File | Issue | Resolution |
|------|-------|------------|
| `COMPREHENSIVE_RESEARCH_SUMMARY.md` | Outdated findings | Update or delete |
| `FINAL_REPORT_DEC16.md` | Superseded by DEC19 | Archive |
| `MECHANISM_MAP.md` | May conflict with SPINE | Reconcile |
| `META_PATTERNS_AND_RAW_LOGIC.md` | Attractor theory (keep) | Validate against current findings |

### Recommended Archive Structure

```
archive/
├── session_logs/          # All DEC*_*.md files
├── deprecated_findings/   # BREAKTHROUGH_*, ITERATION_*
├── legacy_metrics/        # Old behavior_states references
└── superseded_reports/    # Old FINAL_REPORT files
```

---

## 7. Strongest Test in Repository (v1.1: evidence-aware)

### Current Champion (Disk Evidence): `behavior_strict` runs under `results/runs/`

**Why (disk-verified):**
- These runs are present in `results/run_index.csv` and have stable artifacts (`config.json`, `summary.json`, `report.md` in many cases).

**Limitation (major):**
- This is **not** industry-grade compliant for the current contract, because `behavior_strict` is a text-space metric and does not provide `mode_score_m` or `restore_norm(M)`.

### Best Candidate Champion (Code Evidence): `mlp_ablation_necessity` (L0/L1/L2/L3)

**Why (code-verified):**
- The MLP pipelines are the only ones that explicitly call `append_to_run_index()` (and are designed to log prompt IDs + bank hash).

**Blocking issue (disk reality):**
- The only observed MLP run directory contains no `summary.json`, no CSV, no `metadata.json`, and no `RUN_INDEX.jsonl` exists; therefore the industry-grade claim is **not yet demonstrated by artifacts**.

---

## 8. Top 10 Action Items (Ordered)

### Priority 1: Critical Infrastructure

| # | Action | Files to Modify | Est. Time |
|---|--------|-----------------|-----------|
| 1 | **Fix RUN_INDEX.jsonl creation** | `src/utils/run_metadata.py` | 15 min |
| 2 | **Add mode_score_m to p1_ablation** | `src/pipelines/p1_ablation.py` | 30 min |
| 3 | **Add mode_score_m to surgical_sweep** | `src/pipelines/surgical_sweep.py` | 30 min |
| 4 | **Add mode_score_m to kv_mechanism** | `src/pipelines/kv_mechanism.py` | 30 min |
| 5 | **Standardize generation params in metadata** | `src/utils/run_metadata.py` | 20 min |

### Priority 2: Complete SPINE

| # | Action | Experiment | Est. Time |
|---|--------|------------|-----------|
| 6 | **Re-run circuit_discovery with metadata** | `circuit_discovery.py` | 2 hr |
| 7 | **Re-run mlp_steering_sweep with metadata** | `mlp_steering_sweep.py` | 2 hr |
| 8 | **Re-run random_direction_control** | `random_direction_control.py` | 1 hr |
| 9 | **Re-run p1_ablation with mode_score_m** | `p1_ablation.py` | 2 hr |
| 10 | **Re-run kv_mechanism with mode_score_m** | `kv_mechanism.py` | 1 hr |

### Post-Fix Validation

After completing actions 1-5, re-run all canonical experiments and verify:
- [ ] RUN_INDEX.jsonl populated with all runs
- [ ] All runs have mode_score_m in summary.json
- [ ] All runs have prompt IDs in metadata.json
- [ ] All runs have standardized generation params

---

## 9. Code Changes Required

### Fix 1: RUN_INDEX.jsonl Creation

```python
# src/utils/run_metadata.py - Line 96
# CURRENT (creates but doesn't persist properly)
index_path = Path(__file__).parent.parent.parent / "results" / "RUN_INDEX.jsonl"

# VERIFY: Ensure directory exists and file is created
index_path.parent.mkdir(parents=True, exist_ok=True)
if not index_path.exists():
    index_path.touch()  # Create empty file if doesn't exist
```

### Fix 2: Add mode_score_m to p1_ablation

```python
# src/pipelines/p1_ablation.py - Add import
from src.metrics.mode_score import ModeScoreMetric

# In run_p1_ablation_from_config, add after model load:
mode_metric = ModeScoreMetric(tokenizer, device)

# In generate_with_config, compute mode_score:
with torch.no_grad():
    outputs = model(**inputs)
    mode_score = mode_metric.compute_score(outputs.logits)
```

### Fix 3: Standardize generation params

```python
# src/utils/run_metadata.py - Add to get_run_metadata()
def get_run_metadata(..., generation_params: Optional[Dict] = None):
    metadata = {
        # ... existing fields ...
        "generation_params": generation_params or {
            "max_new_tokens": cfg.get("params", {}).get("max_new_tokens", 200),
            "temperature": cfg.get("params", {}).get("temperature", 1.0),
            "top_p": cfg.get("params", {}).get("top_p", 1.0),
            "do_sample": cfg.get("params", {}).get("do_sample", False),
        },
    }
```

---

## 10. Summary (v1.1)

### Current State (disk-first)
- **Spine Completion:** Unknown (code is broad, but stage claims are not backed by artifacts in `results/`).
- **Metric Standardization:** Low in practice (most on-disk runs are legacy/text-space metrics; mode-score usage is not broadly evidenced).
- **Reproducibility:** Partial (a CSV index exists; the JSONL ledger does not; metadata artifacts are missing).
- **Documentation:** High volume with contradictions; treat “stage complete” docs as untrusted unless backed by artifacts.

### Key Insight

The repository has strong *code foundations* (PromptLoader, `ModeScoreMetric`, `run_metadata.py`) but inconsistent *execution evidence* on disk. The priority is to create **one fully evidenced, industry-grade run** (metadata + summary + ledger) and then port the contract across the spine.

---

**Audit Completed:** January 5, 2026  
**Next Review:** After completion of Top 10 Action Items

