# Quick Start for Agents

**Read time: 5 minutes**

---

## Setup (REQUIRED)

### Requirements
- **Python:** 3.9+ (tested on 3.10, 3.11)
- **GPU:** NVIDIA with 16GB+ VRAM (for Mistral-7B)
- **Disk:** ~20GB for model weights
- **CUDA:** 11.8+ (or 12.x)

### Installation

```bash
# 1. Clone and enter repo
cd mech-interp-latent-lab-phase1

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install torch transformers numpy pandas scipy

# 4. Verify installation
python -c "from src.metrics.rv import compute_rv; print('✅ R_V module loaded')"
python -c "from prompts.loader import PromptLoader; print(f'✅ Prompts: {len(PromptLoader().prompts)} loaded')"
```

---

## What This Repo Does

Investigates whether transformer language models exhibit **geometric contraction** in their internal representations when processing **recursive self-observation** prompts (e.g., "Observe this very process of observation").

**Key metric:** R_V = PR(late) / PR(early), where PR is participation ratio of V-projection singular values.

- R_V < 1.0 = contraction (recursive prompts)
- R_V ≈ 1.0 = no change (baseline prompts)

---

## Current Status (Dec 16, 2025)

| Claim | Status | Confidence |
|-------|--------|------------|
| R_V contraction exists at L27 | ✅ VERIFIED | 95% |
| L27 is causal for geometry | ✅ VERIFIED | 90% |
| KV-head group 2 drives contraction | ✅ VERIFIED | 85% |
| Behavior transfer works | ⚠️ FRAGILE | 40% |

**See:** `agent_reviews/responses/` for detailed audits

---

## 5-Minute Orientation

### 1. Key Files

```
prompts/bank.json          # 754 prompts (single source of truth)
prompts/loader.py          # How to load prompts
src/metrics/rv.py          # Canonical R_V implementation (USE THIS ONLY)
src/pipelines/run.py       # Config-driven experiment runner
configs/gold/              # Gold standard pipeline configs
```

### 2. Run Something

```bash
# Test that everything works
python -c "from src.metrics.rv import compute_rv; print('R_V module loaded')"

# Run a config-driven experiment
python -m src.pipelines.run --config configs/gold/01_existence.json

# Or run head validation (Pipeline 4)
python -m src.pipelines.run --config configs/gold/04_head_validation.json
```

### 3. Check Results

```
results/
├── gold_standard/         # Gold standard suite results
├── confound_validation/   # Confound control results
├── h18_h26_gold_standard/ # Head ablation validation (Dec 15)
└── ...
```

Each run creates:
- `summary.json` - Key statistics and pass/fail
- `*_results.csv` - Per-prompt raw data
- `prompt_bank_version.txt` - Reproducibility hash

---

## The 5 Gold Standard Pipelines

| # | Name | Purpose | Config | Status |
|---|------|---------|--------|--------|
| 1 | Existence | Does R_V contraction exist? | `configs/gold/01_existence.json` | ✅ Ready |
| 2 | Causality | Is L27 causal? | `configs/gold/02_causality.json` | ✅ Ready |
| 3 | Layer Map | Where does it happen? | `configs/gold/03_layer_map.json` | ✅ Ready |
| 4 | Head Validation | Which heads drive it? | `configs/gold/04_head_validation.json` | ✅ Ready |
| 5 | Behavior Strict | Does geometry → behavior? | `configs/gold/05_behavior_strict.json` | ⚠️ Broken |

**Details:** `GOLD_STANDARD_SUITE.md`

---

## Critical Things to Know

### DO ✅

- Use `src/metrics/rv.py` for ALL R_V calculations
- Use `prompts/loader.py` to access prompts
- Log `PromptLoader().version` in every experiment
- Include controls: random, shuffled, wrong-layer
- Run from repo root directory

### DON'T ❌

- Don't hardcode prompt lists in experiments
- Don't use `models/*.py` (some have inverse PR bug)
- Don't trust "behavior" metrics without degeneracy checks
- Don't claim L27 is special for behavior (n=300 shows L21=L27)
- Don't use `reproduce_results.py` (doesn't exist)

---

## Interpreting Results

### Pipeline 1 (Existence) - Check `summary.json`:
```json
{
  "mean_rv": {
    "champions": 0.45,      // ✅ Should be < 0.6
    "length_matched": 0.77, // Should be higher
    "pseudo_recursive": 0.72
  },
  "ttest": {
    "champions_vs_length_matched": {
      "p": 1e-15            // ✅ Should be < 0.001
    }
  }
}
```

### Pipeline 2 (Causality) - Check `summary.json`:
```json
{
  "transfer_percent_estimate": 95.7,  // ✅ Should be > 50%
  "tests": {
    "main_vs_wronglayer_paired_ttest": {
      "p": 0.49             // ✅ Should be > 0.05 (null for wrong layer)
    }
  }
}
```

### Pipeline 4 (Head Validation) - Check `VERDICT.md`:
```
✅ target_effect_significant: p=1.2e-08
✅ target_gt_control_head: 0.0872 > 0.0456
✅ target_layer_gt_control_layer: L27 > L21
```

---

## Known Issues (from Agent Reviews)

1. **Behavior metric is broken** — 28% false positive rate on baselines
2. **6 files in `models/*.py`** use wrong PR formula
3. **L27 is NOT special for behavior** — only for geometry
4. **Random KV increases "expression"** — metric is too permissive
5. **Pipeline 5 (Behavior)** — Not implemented, needs redesign

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'src'"
```bash
# Run from repo root
cd /path/to/mech-interp-latent-lab-phase1
# Or install in dev mode
pip install -e .
```

### "CUDA out of memory"
```bash
# Use CPU (slower but works)
# Edit config: "device": "cpu"
# Or reduce window size: "window": 8
```

### "FileNotFoundError: prompts/bank.json"
```bash
# Verify you're in repo root
ls prompts/bank.json
# Should exist with 754 prompts
```

### "NaN values in results"
- Check prompt length (must be >= window size, typically 16 tokens)
- Check model loaded correctly
- Try a different prompt

### "ConfigError: Unknown experiment"
```bash
# List available experiments
python -c "from src.pipelines.registry import get_registry; print(list(get_registry().keys()))"
```

---

## Where to Start

**If you want to understand the science:**
1. `README.md` → Overview
2. `PHASE1_FINAL_REPORT.md` → Cross-model results
3. `MISTRAL_L27_CAUSAL_VALIDATION_COMPLETE.md` → Causal evidence

**If you want to run experiments:**
1. `GOLD_STANDARD_SUITE.md` → Pipeline guide
2. `configs/gold/*.json` → Ready-to-run configs
3. `src/pipelines/run.py --help` → Runner usage

**If you want to audit:**
1. `agent_reviews/responses/*.md` → 6+ independent audits
2. `PROMPT_BANK_SEALED.md` → Prompt validation
3. `results/*/summary.json` → Artifact-backed stats

---

## Get Help

- **Prompt questions:** See `prompts/README.md`
- **Metric questions:** See `docs/MEASUREMENT_CONTRACT.md`
- **Pipeline questions:** See `src/pipelines/registry.py`
- **Historical context:** See `agent_reviews/responses/`

---

*Last updated: 2025-12-16*
