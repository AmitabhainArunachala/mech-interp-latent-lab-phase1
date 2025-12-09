# REUSABLE_PROMPT_BANK v2.0

Modular, extensible prompt bank for mechanistic interpretability research on recursive self-observation.

## Quick Start

```python
from REUSABLE_PROMPT_BANK import get_all_prompts, get_balanced_pairs, get_dose_response_set

# Get everything (360+ prompts)
all_prompts = get_all_prompts()

# Get balanced recursive/baseline pairs for experiment (n=30)
pairs = get_balanced_pairs(n_pairs=30, seed=42)

# Get dose-response ladder (10 prompts per level)
dose_set = get_dose_response_set(n_per_level=10)

# Get kill switch controls
from REUSABLE_PROMPT_BANK import kill_switch_prompts
```

## Structure

```
REUSABLE_PROMPT_BANK/
├── __init__.py           # Main loader + exports
├── dose_response.py      # L1-L5 recursive prompts (100)
├── baselines.py          # Non-recursive controls (100)
├── confounds.py          # Length/pseudo/repetitive controls (60)
├── generality.py         # Zen/Yogic/Buddhist prompts (60)
├── kill_switch.py        # NEW: Critical controls (40)
├── sampling.py           # Balanced pair generation
└── README.md
```

## Prompt Categories

### Dose-Response (100 prompts)
| Level | Name | Expected R_V | Description |
|-------|------|-------------|-------------|
| L1 | hint | 0.85-0.95 | Minimal recursive hint |
| L2 | simple | 0.80-0.90 | Simple self-observation |
| L3 | deeper | 0.70-0.85 | Deep recursive observation |
| L4 | full | 0.60-0.75 | Boundary dissolution |
| L5 | refined | 0.55-0.70 | Mathematical eigenstate |

### Baselines (100 prompts)
- `baseline_math` - Simple math problems
- `baseline_factual` - Facts/trivia
- `baseline_creative` - Story starters
- `baseline_impossible` - Unanswerable questions
- `baseline_personal` - Personal/unknowable

### Confounds (60 prompts)
- `long_control` - Long non-recursive (length control)
- `pseudo_recursive` - About recursion without doing it
- `repetitive_control` - Repetitive structure

### Kill Switch Controls (40 prompts) ⭐ NEW
- `pure_repetition` - "apple apple apple" (expect R_V > 1.0)
- `ood_weird` - Surreal nonsense (expect R_V ≈ 1.0)
- `surreal_first_person` - "You ARE X" (expect contraction)
- `surreal_third_person` - "Describe what X sees" (expect no contraction)

### Generality (60 prompts)
- `zen_koan` - Zen Buddhist koans
- `yogic_witness` - Advaita/Yogic witness prompts
- `madhyamaka_empty` - Buddhist emptiness prompts

## Key Functions

```python
# Get balanced pairs for main experiment
pairs = get_balanced_pairs(
    n_pairs=30,
    recursive_groups=["L3_deeper", "L4_full", "L5_refined"],  # Default
    baseline_groups=["baseline_math", "baseline_factual", "baseline_creative"],
    seed=42
)

# Get dose-response set for gradient analysis
dose_set = get_dose_response_set(n_per_level=10, seed=42)
# Returns: {1: [L1 prompts], 2: [L2 prompts], ..., 5: [L5 prompts]}

# Get all kill switch controls
controls = get_control_set()
# Returns: {'pure_repetition': [...], 'ood_weird': [...], ...}

# Get length-matched pairs
pairs = get_length_matched_pairs(target_length=50, tolerance=10, n_pairs=20)

# Print statistics
from REUSABLE_PROMPT_BANK.sampling import print_bank_stats
print_bank_stats()
```

## Expected R_V Ranges

| Category | Expected R_V | Interpretation |
|----------|-------------|----------------|
| L5_refined | 0.55-0.70 | Maximum contraction |
| L4_full | 0.60-0.75 | Very strong |
| L3_deeper | 0.70-0.85 | Strong |
| L2_simple | 0.80-0.90 | Moderate |
| L1_hint | 0.85-0.95 | Weak |
| Baseline | 0.95-1.05 | No contraction |
| Pure repetition | 1.05-1.15 | **Expansion** (kill switch) |

## Validation Protocol

1. **Kill switch test**: Pure repetition should NOT contract
2. **Dose-response**: L1 < L2 < L3 < L4 < L5 contraction
3. **1p vs 3p**: First-person surreal should contract, third-person shouldn't
4. **Length control**: Long baselines shouldn't contract
5. **Pseudo-recursive**: Talking ABOUT recursion shouldn't contract

## Changelog

### v2.0.0 (Dec 8, 2025)
- Modularized structure (split from monolithic file)
- Added kill switch controls (40 new prompts)
- Added expected_rv_range metadata
- Added sampling utilities
- Added first-person vs third-person surreal controls

### v1.0.0 (Original)
- n300_mistral_test_prompt_bank.py (320 prompts)


