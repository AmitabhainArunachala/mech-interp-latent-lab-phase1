# Pipeline Deep Analysis Report

## Executive Summary

This document provides a comprehensive analysis of all pipelines in `~/mech-interp-latent-lab-phase1/src/pipelines/`.

### Quick Stats
- **Canonical Pipelines**: 9 active (core paper findings)
- **Discovery Pipelines**: 14 active (methodology tools)
- **Archive Pipelines**: ~25 (deprecated/superseded)
- **Deprecated Experiments**: 1 blocked (`mlp_ablation_necessity`)

---

## 1. Core Infrastructure

### 1.1 run.py
**Purpose**: Canonical config-driven experiment runner

**Key Functions**:
- Loads config JSON and validates experiment name
- Creates run directories with timestamps
- Enforces summary schema for canonical experiments
- Appends results to global `RUN_INDEX.jsonl`
- Supports strict mode for metric validation

**Config Requirements**:
```json
{
  "experiment": "rv_l27_causal_validation",
  "model": {"name": "mistralai/Mistral-7B-v0.1", "device": "cuda"},
  "params": {...},
  "seed": 42
}
```

**Output Format**:
- `summary.json` - Aggregated metrics and statistics
- `report.md` - Human-readable run report
- `config.json` - Snapshotted config for reproducibility
- Experiment-specific artifacts (CSV, JSONL)

### 1.2 registry.py
**Purpose**: Experiment registry + config validation

**Key Features**:
- Maps `config["experiment"]` to runnable functions
- Validates required baseline metrics (Nanda-standard)
- Enforces deprecation (blocks `mlp_ablation_necessity`)
- Returns `ExperimentResult` with summary + baseline_metrics

**Experiment Categories**:
```python
CANONICAL_EXPERIMENTS = {
    "rv_l27_causal_validation",
    "confound_validation", 
    "random_direction_control",
    "mlp_ablation_necessity_prompt_pass",
    "mlp_sufficiency_test",
    "combined_mlp_sufficiency_test",
    "head_ablation_validation",
}

GEOMETRY_ONLY_CANONICAL = {
    "rv_l27_causal_validation",
    "mlp_sufficiency_test",
    "combined_mlp_sufficiency_test", 
    "head_ablation_validation",
    "random_direction_control",
}
```

---

## 2. Canonical Pipelines (9 Active)

### 2.1 rv_l27_causal_validation.py ⭐ GOLD STANDARD
**Status**: VALIDATED - Core paper finding

**What it runs**: 
- Tests if patching v_proj at L27 from recursive→baseline transfers R_V contraction
- Includes controls: random norm-matched, shuffled-tokens, wrong-layer patch

**Config Requirements**:
```python
{
  "early_layer": 5,
  "target_layer": 27,
  "wrong_layer": 21,
  "window": 16,
  "max_pairs": 45,
  "pairing": {
    "recursive_groups": ["L5_refined", "L4_full", "L3_deeper"],
    "baseline_groups": ["long_control", "baseline_creative", "baseline_math"]
  }
}
```

**Output Format**:
- `rv_l27_causal_validation_pairs.csv` - Per-pair rows with all conditions
- Summary with: `rv_recursive_mean`, `rv_baseline_mean`, `rv_delta_mean`, `transfer_percent_estimate`

**Validation Status**: ✅ PRODUCTION-READY

---

### 2.2 confound_validation.py ⭐ GOLD STANDARD
**Status**: VALIDATED - Control validation

**What it runs**:
- Compares champions vs length-matched vs pseudo-recursive controls
- Computes R_V for each group and runs statistical tests

**Config Requirements**:
```python
{
  "early_layer": 5,
  "late_layer": 27,
  "window": 16,
  "n_champions": 30,
  "n_length_matched": 30,
  "n_pseudo_recursive": 30
}
```

**Output Format**:
- `confound_results.csv` - Per-prompt rows
- Summary with: mean/std/ci_95 for each group, t-test results, correlation analysis

**Validation Status**: ✅ PRODUCTION-READY

---

### 2.3 random_direction_control.py ⭐ GOLD STANDARD
**Status**: VALIDATED - Direction specificity control

**What it runs**:
- Tests if steering effect is specific to computed direction vs random/orthogonal vectors
- Generates random norm-matched vectors and orthogonal vector to true steering

**Config Requirements**:
```python
{
  "layer": 2,
  "alpha": 2.0,  # or list [1.0, 2.0, 3.0]
  "n_random": 5,
  "n_pairs": 10,
  "include_orthogonal": True,
  "window_size": 16,
  "max_new_tokens": 200
}
```

**Output Format**:
- `random_direction_control.csv` - Per-condition results
- `comparison_table.csv` - Aggregated by alpha
- Analysis with verdict: "REAL" vs "ARTIFACT"

**Validation Status**: ✅ PRODUCTION-READY

---

### 2.4 mlp_ablation_necessity_prompt_pass.py ⭐ GOLD STANDARD
**Status**: VALIDATED - Replaces deprecated mlp_ablation_necessity

**What it runs**:
- Tests if MLP at specific layer is NECESSARY for R_V contraction
- **PROMPT-PASS-ONLY MODE**: Measures R_V on SAME prompt text (no generation)
- Isolates geometric changes from generation artifacts

**Config Requirements**:
```python
{
  "model": "mistralai/Mistral-7B-v0.1",
  "layer": 0,  # Layer to ablate
  "n_pairs": 80,
  "window_size": 16,
  "early_layer": 5,
  "late_layer": 27
}
```

**Output Format**:
- `mlp_ablation_necessity_prompt_pass.csv` - Per-pair results
- Summary with: `rv_baseline`, `rv_ablated`, `pr_early_*`, `pr_late_*`, verdict

**Validation Status**: ✅ PRODUCTION-READY (replaces deprecated `mlp_ablation_necessity`)

---

### 2.5 mlp_sufficiency_test.py ⭐ GOLD STANDARD
**Status**: VALIDATED - Sufficiency testing

**What it runs**:
- Tests if L0 MLP is SUFFICIENT to induce contraction
- Patches L0 MLP from recursive into baseline prompt
- Measures R_V restoration and mode score changes

**Config Requirements**:
```python
{
  "model": "mistralai/Mistral-7B-v0.1",
  "layer": 0,
  "n_pairs": 30,
  "window_size": 16,
  "max_new_tokens": 200
}
```

**Output Format**:
- `mlp_sufficiency_test.csv` - Per-pair results
- Summary with: `rv_restoration_pct`, `mode_score_m`, verdict

**Validation Status**: ✅ PRODUCTION-READY

---

### 2.6 mlp_combined_sufficiency_test.py ⭐ GOLD STANDARD
**Status**: VALIDATED - Multi-layer sufficiency

**What it runs**:
- Tests if multiple MLP layers together are SUFFICIENT
- Default: L0+L1, but configurable to any set
- Includes norm logging to detect artifacts

**Config Requirements**:
```python
{
  "model": "mistralai/Mistral-7B-v0.1",
  "layers": [0, 1],  # or [0, 1, 18, 19, 20]
  "n_pairs": 30,
  "window_size": 16,
  "max_new_tokens": 200
}
```

**Output Format**:
- `combined_mlp_sufficiency_test.csv` - Per-pair results
- Summary with: `rv_restoration_pct`, `mode_restore_norm`, norm logs

**Validation Status**: ✅ PRODUCTION-READY

---

### 2.7 head_ablation_validation.py ⭐ GOLD STANDARD
**Status**: VALIDATED - KV-head specificity

**What it runs**:
- Ablates specific KV-heads at target layer vs control layer
- Tests if target KV-head drives R_V contraction
- Includes GQA aliasing awareness

**Config Requirements**:
```python
{
  "early_layer": 5,
  "target_layer": 27,
  "control_layer": 21,
  "window": 16,
  "target_kv_head": 2,
  "control_kv_head": 0,
  "n_recursive": 50,
  "n_baseline": 50
}
```

**Output Format**:
- `head_ablation_results.csv` - Per-prompt results
- `VERDICT.md` - Pass/fail report
- Summary with: pass_checks, all_passed, comparisons

**Validation Status**: ✅ PRODUCTION-READY

---

### 2.8 multi_token_bridge.py ⭐ GOLD STANDARD
**Status**: VALIDATED - R_V to behavior correlation

**What it runs**:
- Links R_V (prompt-time) to behavioral markers (generation-time)
- Generates text at multiple temperatures
- Tests 3 hypotheses: H1 (R_V vs word count), H2 (L4 vs L3 R_V), H3 (L4 markers)

**Config Requirements**:
```python
{
  "model": {"name": "google/gemma-2-9b"},
  "n_prompts": 20,
  "early_layer": 5,
  "late_layer": 38,
  "window": 16,
  "max_new_tokens": 200,
  "temperatures": [0.0, 0.7],
  "recursive_groups": ["champions", "L4_full", "L3_deeper"],
  "baseline_groups": ["baseline_factual", "baseline_math", "baseline_creative"]
}
```

**Output Format**:
- `rv_behavioral_correlation.csv` - Per-prompt results
- `VERDICT.md` - Hypothesis test results
- Summary with: per-temperature analysis, correlations

**Validation Status**: ✅ PRODUCTION-READY

---

## 3. Discovery Pipelines (14 Active)

### 3.1 behavioral_grounding.py
**Status**: EXPLORATORY - Generation analysis

**What it runs**:
- Generates under 3 conditions: baseline, baseline+patch, recursive
- Computes behavioral metrics: self_ref_rate, unique_word_ratio, repeat_4gram_frac

**Output**: `behavioral_grounding.jsonl`, `behavioral_grounding_summary.csv`

---

### 3.2 behavioral_grounding_batch.py
**Status**: EXPLORATORY - Large-scale generation

**What it runs**:
- Batch version of behavioral_grounding (n=100+)
- Sweeps multiple patch layers
- Quantifies collapse/degeneracy per layer

**Output**: `behavioral_grounding_batch.jsonl`, `behavioral_grounding_batch_summary.csv`

---

### 3.3 path_patching_mechanism.py
**Status**: EXPLORATORY - Component tracing

**What it runs**:
- Intervenes at specific components (v, o, resid) at intervention layer
- Measures R_V at measurement layer
- Tests which pathway causes R_V contraction

**Components**: v_proj, o_proj, residual stream
**Controls**: random, shuffled, opposite, wrong-layer

**Output**: `path_patching_mechanism.csv` with aggregates by component

---

### 3.4 temporal_stability.py ⭐ GOLD STANDARD
**Status**: VALIDATED - Multi-token dynamics

**What it runs**:
- Tracks R_V across autoregressive generation steps
- Tests if R_V contraction persists over multiple tokens
- Measures H31 entropy at each step

**Output**: `temporal_stability.csv` with step-by-step R_V trajectory

---

### 3.5 hysteresis.py ⭐ GOLD STANDARD
**Status**: VALIDATED - Attractor dynamics

**What it runs**:
- Tests "One-Way Door" dynamics
- Measures asymmetry: Efficiency(Base→Rec) - Efficiency(Rec→Base)
- Positive asymmetry = recursive state is attractor

**Output**: `hysteresis_results.csv` with asymmetry scores

---

### 3.6 kv_mechanism.py ⭐ GOLD STANDARD
**Status**: VALIDATED - KV cache transfer

**What it runs**:
- Tests if R_V contraction is stored in KV cache
- Swaps KV cache from recursive→baseline
- Measures geometry transfer (not just behavior)

**Output**: `kv_mechanism.csv` with transfer efficiency

---

### 3.7 layer_sweep.py
**Status**: EXPLORATORY - Layer-wise steering

**What it runs**:
- Tests steering at each layer L8-L27 individually
- Computes layer-specific steering vectors
- Finds causal source layers

**Output**: `layer_sweep_results.csv` with best layer identification

---

### 3.8 logit_lens_analysis.py
**Status**: EXPLORATORY - Crystallization points

**What it runs**:
- Logit lens + logit difference trajectory analysis
- Finds crystallization layers
- Identifies recursive token emergence

**Output**: `logit_lens_analysis.csv` with per-layer predictions/entropy

---

### 3.9 vproj_patching_analysis.py
**Status**: EXPLORATORY - Domain shift analysis

**What it runs**:
- Patches V_proj from recursive→baseline during generation
- Analyzes semantic domain shifts
- Key finding: L27 V_proj → philosophical outputs

**Output**: `vproj_patching_analysis.csv` with domain classifications

---

### 3.10 mlp_vproj_combined_sufficiency_test.py
**Status**: EXPLORATORY - Complete circuit test

**What it runs**:
- Tests MLP (gate+amplifier) + V_proj (readout) together
- Patches L0+L1+L18+L19+L20 MLP + L27 V_proj
- Tests complete circuit sufficiency

**Output**: `mlp_vproj_combined_sufficiency_test.csv`

---

### 3.11 c2_rv_measurement.py
**Status**: EXPLORATORY - C2 config bridge

**What it runs**:
- Runs C2 behavioral transfer config with R_V measurement
- Tests 3 conditions: baseline, KV-only, C2 full
- Bridges geometry→behavior gap

**Output**: `c2_rv_measurement.csv` with R_V trajectories

---

### 3.12 gemma_full_circuit_analysis.py
**Status**: VALIDATED - Cross-architecture

**What it runs**:
- Full validation protocol for Gemma 2 9B
- R_V layer sweep (all 42 layers)
- Logit lens trajectory
- Extended metrics (spectral, cosine)

**Output**: 
- `layer_sweep.csv`
- `logit_lens.csv`  
- `extended_metrics.csv`

---

### 3.13 gemma_head_decomposition.py
**Status**: VALIDATED - Head-wise analysis

**What it runs**:
- Tests each of 8 KV-heads at L3
- Identifies which heads drive R_V effect
- Proper controls: L3 (source) vs L5 (non-source)

**Output**: `head_ablation_raw.csv`, `head_summaries.csv`, `VERDICT.md`

---

### 3.14 eigenstate_direction_finder.py
**Status**: EXPLORATORY

**Purpose**: Finds eigenstate directions in activation space

---

## 4. Deprecated/Blocked Pipelines

### 4.1 mlp_ablation_necessity.py ❌ DEPRECATED
**Status**: BLOCKED in registry

**Why deprecated**:
- Measures R_V on generated text (contract violation)
- Cannot distinguish geometric shift from measurement artifact
- Replaced by `mlp_ablation_necessity_prompt_pass`

**Registry Enforcement**:
```python
if exp == "mlp_ablation_necessity":
    raise ConfigError(
        "Experiment 'mlp_ablation_necessity' is deprecated..."
        "Use 'mlp_ablation_necessity_prompt_pass' instead."
    )
```

---

## 5. Archive Pipelines (Historical)

The `archive/` directory contains ~25 superseded experiments:

### Key Historical Pipelines:
- `phase0_minimal_pairs.py` - Phase 0 existence proofs
- `phase0_metric_targets.py` - Metric target validation
- `phase1_existence.py` - Phase 1 core findings
- `steering.py` / `steering_analysis.py` - Early steering work
- `steering_layer_matrix.py` - Layer-wise steering sweep
- `minimal_recursive_intervention.py` - Minimal intervention tests
- `extended_context_steering.py` - Extended context tests
- `surgical_sweep.py` - Surgical intervention sweeps
- `kv_sufficiency_matrix.py` - KV sufficiency grid
- `circuit_discovery.py` - Circuit discovery tools
- And more...

**Status**: Retained for reference but not in active registry

---

## 6. Pipeline Usage Summary

### For Paper Claims (Validated):
1. `rv_l27_causal_validation` - Core causal claim
2. `confound_validation` - Control validation
3. `random_direction_control` - Direction specificity
4. `mlp_ablation_necessity_prompt_pass` - Necessity tests
5. `mlp_sufficiency_test` / `combined_mlp_sufficiency_test` - Sufficiency tests
6. `head_ablation_validation` - Head specificity
7. `multi_token_bridge` - Behavior link
8. `temporal_stability` - Multi-token dynamics
9. `hysteresis` - Attractor dynamics
10. `kv_mechanism` - KV cache storage

### For Exploration:
- `behavioral_grounding` / `behavioral_grounding_batch`
- `path_patching_mechanism`
- `layer_sweep`
- `logit_lens_analysis`
- `vproj_patching_analysis`
- `c2_rv_measurement`
- `eigenstate_direction_finder`

### For Cross-Architecture:
- `gemma_full_circuit_analysis`
- `gemma_head_decomposition`

---

## 7. Common Config Patterns

### Minimal Config:
```json
{
  "experiment": "rv_l27_causal_validation",
  "model": {"name": "mistralai/Mistral-7B-v0.1"},
  "params": {"max_pairs": 20},
  "seed": 42
}
```

### With Phase Scoping:
```json
{
  "experiment": "mlp_sufficiency_test",
  "model": {"name": "mistralai/Mistral-7B-v0.1"},
  "params": {"layer": 0, "n_pairs": 30},
  "results": {"phase": "phase2_generalization"},
  "seed": 42
}
```

### Gemma Config:
```json
{
  "experiment": "gemma_full_circuit_analysis",
  "model": {"name": "google/gemma-2-9b"},
  "params": {
    "n_prompts": 30,
    "early_layer": 5,
    "late_layer": 38,
    "num_layers": 42
  },
  "seed": 42
}
```

---

## 8. Output Schema Summary

### All Experiments Return:
```python
ExperimentResult(
    summary={
        "experiment": str,
        "n_pairs": int,
        "rv_recursive_mean": float | None,
        "rv_baseline_mean": float | None,
        "rv_delta_mean": float | None,
        "rv_cohens_d": float | None,
        "rv_p_value": float | None,
        "logit_diff_delta_mean": float | None,  # Optional
        "logit_diff_cohens_d": float | None,
        "logit_diff_p_value": float | None,
        "prompt_bank_version": str,
        ...
    },
    baseline_metrics={...}  # Optional but recommended
)
```

### Geometry-Only Experiments:
- Set `logit_diff_*` fields to `None`
- Excluded from strict mode validation

---

## 9. Recommendations

### For New Experiments:
1. **Use existing canonical patterns** - Copy from validated pipelines
2. **Always use prompt-pass mode** - Never measure R_V on generated text
3. **Include proper controls** - Random, shuffled, wrong-layer where applicable
4. **Return full ExperimentResult** - Include baseline_metrics
5. **Log prompt_bank_version** - Critical for reproducibility

### For Running Experiments:
1. **Start with canonical experiments** - They have proven configs
2. **Use strict mode for validation** - `python -m src.pipelines.run --config ... --strict`
3. **Check RUN_INDEX.jsonl** - Single source of truth for all runs
4. **Archive old results** - Keep results/ organized by phase

### For Deprecation:
1. **Update registry** - Add to deprecation check
2. **Document replacement** - Point to new experiment
3. **Move to archive** - Keep file but remove from registry
4. **Update configs** - Migrate existing configs to new experiment

---

## 10. Files Summary

| Category | Count | Files |
|----------|-------|-------|
| Core Infrastructure | 2 | run.py, registry.py |
| Canonical (Validated) | 9 | rv_l27_causal_validation.py, confound_validation.py, random_direction_control.py, mlp_ablation_necessity_prompt_pass.py, mlp_sufficiency_test.py, combined_mlp_sufficiency_test.py, head_ablation_validation.py, multi_token_bridge.py, (mlp_ablation_necessity.py deprecated) |
| Discovery (Exploratory) | 14 | behavioral_grounding.py, behavioral_grounding_batch.py, path_patching_mechanism.py, temporal_stability.py, hysteresis.py, kv_mechanism.py, layer_sweep.py, logit_lens_analysis.py, vproj_patching_analysis.py, mlp_vproj_combined_sufficiency_test.py, c2_rv_measurement.py, gemma_full_circuit_analysis.py, gemma_head_decomposition.py, eigenstate_direction_finder.py |
| Archive | ~25 | Various historical experiments |

**Total Active Pipelines**: 25 (9 canonical + 14 discovery + 2 core)
