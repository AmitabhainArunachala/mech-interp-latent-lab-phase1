# Experiments Directory Analysis

**Date:** 2026-02-05  
**Location:** `~/mech-interp-latent-lab-phase1/experiments/`

## Executive Summary

The `experiments/` directory contains **5 items** across 5 subdirectories:
- **3 experiment templates** (numbered 001-003) - stubs with READMEs only, never implemented
- **2 standalone test scripts** (phase0_metrics, phase5_steering) - functional but isolated

**Key Finding:** The `experiments/` directory is largely **disconnected** from the formal pipeline system in `src/pipelines/`. The active experimental framework lives in `src/pipelines/` with 44 registered experiments.

---

## Detailed Inventory

### 1. Numbered Experiment Templates (001-003)

| ID | Name | Status | Purpose |
|----|------|--------|---------|
| 001 | L4 vs Neutral | **STUB** | Test where "answer crystallizes" in latent space using ε (epsilon) + entropy metrics |
| 002 | Ablation @ Mid-Layer | **STUB** | Test if zeroing mid-layer attention disrupts crystallization |
| 003 | Length-Matched Control | **STUB** | Control test to distinguish token length effects from recursion effects |

**Assessment:** These are template READMEs with placeholders like `[FILL IN]`, `[FILL AFTER RUNNING]`. They were never implemented as runnable code. They appear to be early experimental design documents from Dec 2024.

---

### 2. Phase 0 Metrics: Jabberwocky Matrix Test

**File:** `phase0_metrics/jabberwocky_matrix_test.py`  
**Status:** **FUNCTIONAL / STANDALONE**  
**Date:** DEC10

**Purpose:**  
Test whether R_V (Participation Ratio) collapse is specific to recursive/self-referential prompts, or appears for "weird"/OOD prompts in general.

**Methodology:**
- Compares 4 prompt types: easy/normal, hard/normal, weird/non-recursive, recursive
- Measures PR at EARLY_LAYER=5 and TARGET_LAYER=27 (V-proj)
- Computes R_V = PR_L27 / PR_L5

**Dependencies:**
- `transformers`
- `torch`
- `numpy`

**Assessment:** 
- ✅ Runnable standalone script
- ⚠️ Hardcoded paths to `/workspace/` (Linux assumption)
- ⚠️ Not integrated with pipeline registry
- ⚠️ No config file exists for this experiment

---

### 3. Phase 5 Steering: L8 Local Reversibility Test

**File:** `phase5_steering/l8_local_reversibility_test.py`  
**Status:** **FUNCTIONAL / STANDALONE**  
**Date:** DEC10

**Purpose:**  
Test whether geometry at Layer 8 is locally symmetric around v8, or already asymmetric. Tests reversibility of residual stream manipulations.

**Methodology:**
- Captures residual stream at Layer 8 for baseline and recursive prompts
- Computes v8 = mean_rec - mean_base (the recursive "direction")
- Tests symmetry by adding/subtracting α * v8 for α ∈ [0, 0.5, 1.0, 1.5, 2.0]
- Measures participation ratios and distances

**Dependencies:**
- `transformers`
- `torch`
- `numpy`
- `pandas` (optional, for CSV export)

**Assessment:**
- ✅ Runnable standalone script
- ⚠️ Saves results to hardcoded `/workspace/` path
- ⚠️ Not integrated with pipeline registry
- ⚠️ Uses canonical prompts but not the canonical pipeline framework

---

## Relationship to Pipelines

### The Great Disconnect

The `experiments/` directory exists **outside** the formal experiment framework defined in `src/pipelines/`:

| Aspect | experiments/ | src/pipelines/ |
|--------|--------------|----------------|
| **Registry** | ❌ None | ✅ 44 registered experiments in `registry.py` |
| **Config-driven** | ❌ Hardcoded | ✅ JSON configs in `configs/` |
| **Run via** | Direct Python execution | `src/pipelines/run.py` or `registry.py` |
| **Output format** | CSV/stdout | Standardized `ExperimentResult` dataclass |
| **Baseline metrics** | ❌ No | ✅ Enforces Nanda-standard metrics (rv, logit_diff) |

### Pipeline Categories (for context)

The formal pipeline system has 3 tiers:

1. **Canonical (8 pipelines)** - Core paper findings
   - `rv_l27_causal_validation` - Main causal validation
   - `head_ablation_validation` - Head-level ablations
   - `mlp_ablation_necessity_prompt_pass` - MLP necessity tests
   - `multi_token_bridge` - Cross-architecture bridge
   - etc.

2. **Discovery (12 pipelines)** - Methodology tools
   - `behavioral_grounding` - Behavioral correlation
   - `path_patching_mechanism` - Causal path analysis
   - `kv_mechanism` - KV cache investigations
   - etc.

3. **Archive (24+ pipelines)** - Historical/superseded
   - `phase1_existence` - Early existence proofs
   - `steering` - Early steering experiments
   - `kitchen_sink` - Comprehensive sweeps
   - etc.

### Mapping Experiments to Pipeline Concepts

| experiments/ File | Related Pipeline Concept | Status |
|-------------------|--------------------------|--------|
| `001-l4-vs-neutral` | Similar to `rv_l27_causal_validation` | Never migrated |
| `002-ablation-layer-mid` | Similar to `head_ablation_validation` | Never migrated |
| `003-length-matched-control` | Similar to `confound_validation` | Never migrated |
| `jabberwocky_matrix_test.py` | Unique - no direct equivalent | Standalone |
| `l8_local_reversibility_test.py` | Similar to `path_patching_mechanism` | Standalone |

---

## Status Summary

### Working
- `jabberwocky_matrix_test.py` - Functional but isolated
- `l8_local_reversibility_test.py` - Functional but isolated

### Broken/Deprecated
- None (scripts are functional)

### Stubs (Never Implemented)
- `001-l4-vs-neutral/README.md` - Template only
- `002-ablation-layer-mid/README.md` - Template only  
- `003-length-matched-control/README.md` - Template only

### Missing Integration
- No experiments in `experiments/` are registered in `src/pipelines/registry.py`
- No config files exist for these experiments in `configs/`
- They cannot be run via the standard `run.py` framework

---

## Recommendations

1. **Archive or Implement** the 001-003 stubs
   - Either implement them as formal pipelines
   - Or move them to archive if concepts are superseded

2. **Integrate standalone scripts** or document isolation
   - `jabberwocky_matrix_test.py` and `l8_local_reversibility_test.py` could be:
     - Migrated to discovery pipelines with configs
     - Or documented as "reference implementations" for specific techniques

3. **Clarify directory purpose**
   - Currently `experiments/` appears to be a "sketchbook" for early ideas
   - Consider renaming to `experiments_archive/` or `experiment_drafts/`
   - Or establish clear conventions for when experiments graduate to pipelines

---

## Files Inventory

```
experiments/
├── 001-l4-vs-neutral/
│   └── README.md              # Template only
├── 002-ablation-layer-mid/
│   └── README.md              # Template only
├── 003-length-matched-control/
│   └── README.md              # Template only
├── phase0_metrics/
│   └── jabberwocky_matrix_test.py   # Functional standalone
└── phase5_steering/
    └── l8_local_reversibility_test.py # Functional standalone
```

**Total:** 5 files across 5 directories  
**Functional code:** 2 Python scripts  
**Documentation stubs:** 3 READMEs  
**Registered in pipeline system:** 0
