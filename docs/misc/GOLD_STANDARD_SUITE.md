# Gold Standard Suite: 5 Canonical Pipelines

**Status:** PROPOSED  
**Goal:** Lab-grade reproducibility, automated validation, agent-friendly onboarding

---

## Quick Start (Agent Onboarding)

```bash
# 1. Run the full gold standard suite (takes ~30 min on GPU)
python -m src.pipelines.run_gold_standard --model mistralai/Mistral-7B-v0.1

# 2. Or run individual pipelines
python -m src.pipelines.run --config configs/gold/01_existence.json
python -m src.pipelines.run --config configs/gold/02_causality.json
python -m src.pipelines.run --config configs/gold/03_layer_map.json
python -m src.pipelines.run --config configs/gold/04_head_validation.json
python -m src.pipelines.run --config configs/gold/05_behavior_strict.json

# 3. View results
cat results/gold_standard/latest/VERDICT.md
```

---

## The 5 Pipelines

### Pipeline 1: EXISTENCE (Does R_V contraction exist?)

**Purpose:** Confirm R_V < 1.0 for recursive prompts with all confound controls

**Config:** `configs/gold/01_existence.json`
```json
{
  "experiment": "confound_validation",
  "params": {
    "model": "mistralai/Mistral-7B-v0.1",
    "n_champions": 30,
    "n_length_matched": 30,
    "n_pseudo_recursive": 30,
    "layer": 27,
    "window": 16,
    "seeds": [42, 123, 456]
  }
}
```

**Pass criteria:**
- [ ] Champions R_V < 0.6
- [ ] Champions < length_matched (p < 0.001)
- [ ] Champions < pseudo_recursive (p < 0.001)
- [ ] length_matched ≈ pseudo_recursive (p > 0.05)

**Outputs:** `confound_results.csv`, `summary.json`, `extended_stats.json`

---

### Pipeline 2: CAUSALITY (Is L27 causal for geometry?)

**Purpose:** Prove patching at L27 transfers geometric contraction

**Config:** `configs/gold/02_causality.json`
```json
{
  "experiment": "rv_l27_causal_validation",
  "params": {
    "model": "mistralai/Mistral-7B-v0.1",
    "n_pairs": 60,
    "target_layer": 27,
    "control_layer": 21,
    "window": 16,
    "seeds": [42, 123, 456],
    "controls": ["random", "shuffled", "wrong_layer", "opposite"]
  }
}
```

**Pass criteria:**
- [ ] Patched R_V < baseline R_V (p < 0.001)
- [ ] Transfer efficiency > 50%
- [ ] Random control: opposite direction (p < 0.001)
- [ ] Wrong-layer (L21) control: null effect (p > 0.05)

**Outputs:** `causal_validation.csv`, `summary.json`, `transfer_stats.json`

---

### Pipeline 3: LAYER MAP (Where does contraction happen?)

**Purpose:** Map R_V trajectory across all layers

**Config:** `configs/gold/03_layer_map.json`
```json
{
  "experiment": "path_patching_mechanism",
  "params": {
    "model": "mistralai/Mistral-7B-v0.1",
    "n_pairs": 40,
    "layers": [0, 5, 10, 15, 18, 21, 24, 25, 26, 27, 28, 30],
    "window": 16,
    "patch_types": ["none", "recursive", "shuffled", "random"]
  }
}
```

**Pass criteria:**
- [ ] Early layers (0-15): shuffled ≠ recursive
- [ ] Late layers (24-27): shuffled ≈ recursive (within 0.01)
- [ ] Random patches: expansion (R_V > 0.9) at all layers
- [ ] Peak effect at L27 ± 2

**Outputs:** `layer_trajectory.csv`, `summary.json`, `LAYER_MAP.md`

---

### Pipeline 4: HEAD VALIDATION (Which heads drive contraction?)

**Purpose:** Validate KV-head group effects with proper controls

**Config:** `configs/gold/04_head_validation.json`
```json
{
  "experiment": "head_ablation_validation",
  "params": {
    "model": "mistralai/Mistral-7B-v0.1",
    "n_recursive": 50,
    "n_baseline": 50,
    "target_layer": 27,
    "control_layer": 21,
    "target_kv_head": 2,
    "control_kv_head": 0,
    "window": 16
  }
}
```

**Pass criteria:**
- [ ] Target ablation significantly increases R_V (p < 0.001)
- [ ] Target > control head effect (p < 0.01)
- [ ] L27 > L21 effect (p < 0.001)
- [ ] Effect exists for BOTH prompt types (not recursive-specific)

**Outputs:** `head_validation.csv`, `summary.json`

---

### Pipeline 5: BEHAVIOR (Does geometry → behavior? STRICT)

**Purpose:** Test if geometric intervention causes genuine behavioral change (with degeneracy gates)

**Config:** `configs/gold/05_behavior_strict.json`
```json
{
  "experiment": "behavior_validation_strict",
  "params": {
    "model": "mistralai/Mistral-7B-v0.1",
    "n_pairs": 50,
    "generation": {
      "do_sample": false,
      "temperature": 0.0,
      "max_new_tokens": 100
    },
    "degeneracy_gates": {
      "max_repeat_4gram_frac": 0.3,
      "min_unique_word_ratio": 0.4
    },
    "behavior_classifier": "strict",
    "seeds": [42, 123, 456]
  }
}
```

**Pass criteria:**
- [ ] Baseline A_control expression rate < 10% (NOT 28%)
- [ ] Transfer expression rate > A_control + 20% (after degeneracy filter)
- [ ] Random KV does NOT increase expression (or increases degeneracy)
- [ ] L27 patching > L21 patching (if behavior is layer-specific)

**Outputs:** `behavior_strict.csv`, `summary.json`, `degeneracy_report.json`

---

## Artifacts Always Logged

Every pipeline run MUST log:

```
results/gold_standard/runs/{timestamp}_{pipeline}/
├── config.json           # Exact config used
├── summary.json          # Pass/fail + key stats
├── {pipeline}_results.csv  # Raw per-prompt data
├── prompt_bank_version.txt # SHA256 of prompts/bank.json
└── VERDICT.md            # Human-readable pass/fail
```

---

## Agent Onboarding Checklist

### First 5 Minutes
1. Read this file (`GOLD_STANDARD_SUITE.md`)
2. Check `results/gold_standard/latest/VERDICT.md` for current status
3. Run `python -m src.pipelines.run_gold_standard --dry-run` to see what would execute

### First 30 Minutes
1. Run Pipeline 1 (Existence) - confirms basic effect
2. Check `results/gold_standard/runs/*/summary.json` for pass/fail

### First Hour
1. Run full suite
2. Review `VERDICT.md`
3. If failures, check specific pipeline CSVs

### Key Files to Know
- `prompts/bank.json` - All prompts (single source of truth)
- `prompts/loader.py` - How to access prompts
- `src/metrics/rv.py` - Canonical R_V implementation
- `src/pipelines/run.py` - Config-driven runner
- `docs/MEASUREMENT_CONTRACT.md` - Parameter definitions

---

## What This Solves

| Problem (from agent reviews) | Solution |
|------------------------------|----------|
| Multiple R_V implementations | All pipelines use `src/metrics/rv.py` only |
| Behavior metric too permissive | Pipeline 5 has degeneracy gates + strict classifier |
| Missing controls | Every pipeline has mandatory control conditions |
| No multi-seed replication | `seeds: [42, 123, 456]` in all configs |
| Prompt bank version not logged | Always logged in `prompt_bank_version.txt` |
| Results scattered | All in `results/gold_standard/` |
| Agent confusion | Single `VERDICT.md` shows current status |

---

## Implementation Status (Updated Dec 16, 2025)

| Pipeline | Config Exists | Pipeline Code | Registered | Status |
|----------|---------------|---------------|------------|--------|
| 01_existence | ✅ | ✅ `confound_validation.py` | ✅ | **READY** |
| 02_causality | ✅ | ✅ `rv_l27_causal_validation.py` | ✅ | **READY** |
| 03_layer_map | ✅ | ✅ `path_patching_mechanism.py` | ✅ | **READY** |
| 04_head_validation | ✅ | ✅ `head_ablation_validation.py` | ✅ | **READY** |
| 05_behavior_strict | ✅ | ❌ Needs implementation | ❌ | **BROKEN** |

### Recent Fixes (Dec 16, 2025)

1. ✅ **Pipeline 4 now registered** - `head_ablation_validation` added to `registry.py`
2. ✅ **Prompt bank version logging** - All pipelines now log `prompt_bank_version.txt`
3. ✅ **Config-driven prompt selection** - `confound_validation.py` respects `n_champions`, `n_length_matched`, `n_pseudo_recursive`
4. ✅ **NaN filtering** - All pipelines filter NaN before statistical tests
5. ✅ **QUICK_START.md updated** - Setup instructions, troubleshooting added

### Remaining Issues

1. ❌ **Pipeline 5 (Behavior)** - Not implemented. Needs:
   - Degeneracy gates (4-gram repeat filter, unique word ratio)
   - Semantic similarity metric (not just keywords)
   - Human evaluation subset
   
2. ⚠️ **"L27 is peak" claim** - Based on N=1 tomography. Needs N=40+ layer sweep.

3. ⚠️ **GQA aliasing** - Head claims should use "KV-head group" not individual heads.

---

## Verified by Agent Reviews

All pipelines have been reviewed by:
- Claude Composer (Engineering: 6.5/10, Scientific: 7/10, Reproducibility: 6/10)
- GPT Codex (Engineering: 5/10, Scientific: 4/10, Reproducibility: 5/10)
- Gemini 3 Pro (Engineering: 8/10, Scientific: varies, Reproducibility: 9→6/10)

**Consensus:** Core geometry claims are solid (9/10), behavior claims are fragile (2/10).

---

*Updated 2025-12-16 based on agent review synthesis and fixes*

