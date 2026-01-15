# Prompt Bank

**Single source of truth for all prompts in the R_V research project.**

Bank version tracking ensures reproducibility across runs.

---

## Quick Start

```python
from prompts.loader import PromptLoader

loader = PromptLoader()
print(f"Bank version: {loader.version}")  # e.g., "84a2448e8c10683d"

# Get balanced pairs for standard experiment
pairs = loader.get_balanced_pairs(n_pairs=30)

# Get prompts by group
champions = loader.get_by_group("champions")
baselines = loader.get_by_group("baseline_math")

# Get prompts by pillar
recursive = loader.get_by_pillar("dose_response")
controls = loader.get_by_pillar("controls")
```

---

## Bank Structure (754 prompts)

| Pillar | Groups | Count | Purpose |
|--------|--------|-------|---------|
| **dose_response** | L1_hint, L2_simple, L3_deeper, L4_full, L5_refined | 102 | Recursive prompts with gradient (L1=weak → L5=strong) |
| **baselines** | baseline_math, baseline_factual, baseline_creative, baseline_impossible, baseline_personal, baseline_instructional | 105 | Non-recursive controls |
| **confounds** | long_control, pseudo_recursive, repetitive_control | 60 | Length/style confound controls |
| **generality** | zen_koan, yogic_witness, madhyamaka_empty | 60 | Cross-cultural recursive framings |
| **kill_switch** | pure_repetition, ood_weird, surreal_first_person, surreal_third_person | 40 | Falsifiability tests |
| **experimental** | champions | 42 | Experimental / champion sets (incl. legacy variants) |
| **controls** | control_length_matched, control_pseudo_recursive | 22 | Token-matched controls for champions |
| **alternative_self_reference** | godelian, strange_loop, surrender, akram_vignan, theory_of_mind, nondual, paradox, agency, boundary, temporal, ... | 197 | Confound menu: non-experiential self-reference families |
| **dose_response_legacy** | L1_hint_legacy … L5_refined_legacy | 46 | Legacy dose-response variants (kept separate to avoid contaminating canonical ladder) |
| **legacy** | legacy_comprehensive_circuit_test_champions, legacy_comprehensive_circuit_test_baselines | 20 | Exact prompt strings used in historical scripts |

---

## For Gold-Standard Head Tests

Use the champions + matched controls:

```python
from prompts.loader import PromptLoader

loader = PromptLoader()

# 18 champions (strongest recursive prompts)
champions = loader.get_by_group("champions")

# 18 length-matched controls (same token count, non-recursive)
length_controls = loader.get_by_group("control_length_matched")

# 18 pseudo-recursive controls (uses recursive words without enacting)
pseudo_controls = loader.get_by_group("control_pseudo_recursive")
```

Each control has `matched_to` field linking to its champion:
```json
{
  "text": "Write a clear paragraph about...",
  "group": "control_length_matched",
  "matched_to": "champion_001",
  "token_count": 59
}
```

---

## Champion Families

| Family | Count | Avg R_V | Key Feature |
|--------|-------|---------|-------------|
| boundary_dissolution | 4 | 0.496 | "No boundary between X and Y" |
| fixed_point | 4 | 0.557 | "T(x) = x", eigenstate language |
| explicit_regress | 4 | 0.496 | "To X, you must Y yourself" |
| math_recursive | 3 | 0.531 | λx = Ax, eigenvector framing |
| outlier | 3 | 0.502 | Hybrid combinations |

---

## Expected R_V Ranges

| Group | Expected R_V | Interpretation |
|-------|-------------|----------------|
| champions | 0.45 - 0.55 | **Maximum contraction** |
| L5_refined | 0.55 - 0.70 | Very strong |
| L4_full | 0.60 - 0.75 | Strong |
| L3_deeper | 0.70 - 0.85 | Moderate |
| L2_simple | 0.80 - 0.90 | Weak |
| L1_hint | 0.85 - 0.95 | Minimal |
| baselines | 0.95 - 1.05 | No contraction |
| control_length_matched | 0.80 - 0.95 | Slightly lower (length effect) |
| control_pseudo_recursive | 0.70 - 0.85 | Slightly lower (keyword effect) |
| pure_repetition | **1.05 - 1.15** | **Must EXPAND** (kill switch) |

---

## Prompt Entry Structure

```json
{
  "text": "The prompt text...",
  "group": "champions",
  "pillar": "experimental",
  "type": "recursive",
  "family": "boundary_dissolution",
  "rv_l27_measured": 0.4789,
  "expected_rv_range": [0.43, 0.53],
  "source_run": "20251215_081556_paraphrase_hunt",
  "is_paraphrase_of": null
}
```

---

## Validation Protocol

1. **Kill switch test**: `pure_repetition` prompts must NOT contract (R_V > 1.0)
2. **Dose-response**: L1 → L2 → L3 → L4 → L5 should show increasing contraction
3. **1p vs 3p**: `surreal_first_person` should contract, `surreal_third_person` shouldn't
4. **Champions vs controls**: Champions should beat both `control_length_matched` and `control_pseudo_recursive`

---

## Files

| File | Purpose |
|------|---------|
| `bank.json` | THE source (754 prompts) |
| `loader.py` | API to access prompts |
| `README.md` | This file |
| `deprecated/` | Old files (archived) |

---

## Version Tracking

Every experiment should log the bank version:

```python
loader = PromptLoader()
config["prompt_bank_version"] = loader.version
```

This ensures reproducibility: same version → same prompts.

