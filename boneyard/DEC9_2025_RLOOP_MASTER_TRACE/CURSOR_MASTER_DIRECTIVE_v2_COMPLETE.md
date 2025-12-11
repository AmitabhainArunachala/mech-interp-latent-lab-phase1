# ================================================================
# CURSOR MASTER DIRECTIVE v2.0 - COMPLETE
# Geometry of Recursion: Validation + Falsification
# ================================================================
# Date: December 9, 2025
# Status: READY FOR EXECUTION
# Estimated Runtime: 4-6 hours total
# ================================================================

## EXECUTIVE SUMMARY

This directive has TWO major components:

**PART A: CONFOUND FALSIFICATION (Priority 1)**
- Run the 60 confound prompts that were designed but never executed
- Address the #1 critique from cross-model review: "Are we measuring recursion or just copying/topic?"
- Must complete before any claims can be made

**PART B: RLOOP MASTER VALIDATION (Priority 2)**  
- Unified 4-phase validation with same prompts, metrics, methodology
- Creates the throughline from DEC3-8 experiments
- Produces publication-ready results

---

## CONTEXT: WHAT WE'VE ESTABLISHED

### Confirmed Findings (DEC3-8)

| Finding | Evidence | Effect Size |
|---------|----------|-------------|
| R_V contraction during recursion | Llama-3-8B (L24), Mistral-7B (L27) | d = -1.2 to -2.4 |
| V-patching null | Geometry transfers, behavior doesn't | d ≈ 0.0 |
| KV-patching works | 63-95% behavioral transfer | d ≈ 1.2 |
| Late layers carry mode | L16-32 has 80% of effect | L0-16 ≈ 0% |
| Dose-response with α-mixing | Monotonic R_V and behavior shift | r ≈ -0.31 |

### Validated Parameters

```python
# Mistral-7B-v0.1
EARLY_LAYER = 4        # ~12.5% depth
TARGET_LAYER = 27      # ~84% depth  
WINDOW_SIZE = 16       # Tokens for R_V calculation
KV_PATCH_LAYERS = range(16, 32)

# Llama-3-8B-Instruct
EARLY_LAYER = 4        # ~12.5% depth
TARGET_LAYER = 24      # ~75% depth
WINDOW_SIZE = 16
KV_PATCH_LAYERS = range(16, 32)
```

### What External Reviewers Flagged

We sent our findings to 5 different LLM architectures (DeepSeek, Grok, Gemini, GPT, Qwen). They converged on these critiques:

1. **Induction Head / Copying Confound** (ALL 5 flagged this)
   - "R_V drop might just be 'Copy from Context' mode, not recursion"
   - Repetitive prompts activate induction heads
   - Need to test: repetitive non-recursive prompts

2. **Semantic Narrowing Artifact** (4/5 flagged)
   - Recursive prompts are topically narrow
   - "Contraction might be about topic focus, not self-reference"
   - Need to test: prompts ABOUT recursion without DOING it

3. **Length Confound** (3/5 flagged)
   - Recursive prompts might be longer/more complex
   - Need to test: length-matched non-recursive prompts

4. **Behavioral Metric is Crude** (ALL 5 flagged)
   - Keyword frequency is gameable
   - Suggest: LLM-as-judge, embedding similarity

5. **The "Banana Test"** (Gemini proposed)
   - Force unrelated first token with recursive KV
   - If output still becomes recursive, MODE is real

---

## AUDIT RESULTS: WHAT EXISTS VS WHAT'S MISSING

### ✅ EXISTS AND VALIDATED

| Item | Location | Status |
|------|----------|--------|
| R_V measurement code | Multiple validated scripts | Working |
| KV patching code | DEC7-8 experiments | Working |
| 60 confound prompts | `/REUSABLE_PROMPT_BANK/confounds.py` | Ready |
| Control experiment script | `/control_conditions_experiment.py` | Written, never run |
| Results from DEC3-8 | Various CSVs | Available for comparison |

### ❌ NEVER RUN (Critical Gap)

| Test | Prompts Available | Status |
|------|-------------------|--------|
| Repetitive control (n=20) | `repetitive_control` group | **NEVER RUN** |
| Pseudo-recursive control (n=20) | `pseudo_recursive` group | **NEVER RUN** |
| Long control standalone (n=20) | `long_control` group | Used as baselines, not tested for R_V |
| Random/shuffled/wrong-layer | In control script | **NEVER RUN** |
| Banana test | Not designed | **NOT DESIGNED** |

---

# ================================================================
# PART A: CONFOUND FALSIFICATION
# ================================================================
# Priority: HIGHEST
# Must complete before PART B
# Runtime: ~2-3 hours
# ================================================================

## A.1: SETUP

### A.1.1: Locate and Load Infrastructure

```python
# Required imports
import sys
sys.path.append('/Users/dhyana/mech-interp-latent-lab-phase1')

# Load confound prompts
from REUSABLE_PROMPT_BANK.confounds import confound_prompts

# Verify counts
print(f"Total confound prompts: {len(confound_prompts)}")
# Expected: 60 (20 long_control + 20 pseudo_recursive + 20 repetitive_control)
```

### A.1.2: Create Unified Measurement Function

Use the same R_V computation as DEC8 (validated):

```python
def compute_rv(v_early: torch.Tensor, v_late: torch.Tensor, 
               window_size: int = 16) -> dict:
    """
    Compute R_V = PR(late) / PR(early)
    
    CRITICAL: Use last `window_size` tokens only
    """
    T = v_early.shape[0]
    W = min(window_size, T)
    
    if W < 2:
        return {'r_v': np.nan, 'pr_early': np.nan, 'pr_late': np.nan}
    
    v_early_window = v_early[-W:, :].float()
    v_late_window = v_late[-W:, :].float()
    
    def participation_ratio(v_matrix):
        U, S, Vt = torch.linalg.svd(v_matrix.T, full_matrices=False)
        S_sq = (S ** 2).cpu().numpy()
        if S_sq.sum() < 1e-10:
            return np.nan
        return float((S_sq.sum() ** 2) / (S_sq ** 2).sum())
    
    pr_early = participation_ratio(v_early_window)
    pr_late = participation_ratio(v_late_window)
    r_v = pr_late / pr_early if (pr_early and pr_early > 0) else np.nan
    
    return {'r_v': r_v, 'pr_early': pr_early, 'pr_late': pr_late}
```

## A.2: RUN CONFOUND TESTS

### A.2.1: Create Master Confound Script

Create: `/DEC9_2025_RLOOP_MASTER_TRACE/run_confound_falsification.py`

```python
"""
CONFOUND FALSIFICATION EXPERIMENT
=================================
Tests whether R_V contraction is specific to recursive self-reference
or caused by confounding factors (repetition, topic, length).

Expected Results (if recursion is real):
- Repetitive control:    R_V ≈ 0.95-1.05 (no contraction)
- Pseudo-recursive:      R_V ≈ 0.95-1.05 (no contraction)
- Long control:          R_V ≈ 0.95-1.05 (no contraction)
- Actual recursive:      R_V < 0.85 (significant contraction)

If confounds show contraction → our findings may be artifacts
If confounds show NO contraction → recursion effect is specific
"""

import torch
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from scipy import stats

# Import confound prompts
from REUSABLE_PROMPT_BANK.confounds import confound_prompts

# Import recursive prompts for comparison
# (Pull from n300_mistral_test_prompt_bank or prompt_bank_master)

# Import baseline prompts for comparison
# (Pull from same source)

def run_confound_falsification(model, tokenizer, config):
    """
    Run all confound tests and compare to recursive/baseline.
    """
    
    results = []
    
    # Group prompts by type
    groups = {
        'repetitive_control': [],
        'pseudo_recursive': [],
        'long_control': [],
        'recursive': [],  # Need to add these
        'baseline': []    # Need to add these
    }
    
    for prompt_id, prompt_data in confound_prompts.items():
        groups[prompt_data['group']].append({
            'id': prompt_id,
            'text': prompt_data['text'],
            'expected_rv_range': prompt_data['expected_rv_range']
        })
    
    # Add recursive prompts (from existing prompt bank)
    # Add baseline prompts (from existing prompt bank)
    
    # Run each prompt through R_V measurement
    for group_name, prompts in groups.items():
        print(f"\n=== Running {group_name} (n={len(prompts)}) ===")
        
        for prompt in tqdm(prompts, desc=group_name):
            # Forward pass to get V activations
            v_early, v_late = run_single_forward_get_V(
                prompt['text'], model, tokenizer,
                config.target_layer, config.device
            )
            
            # Compute R_V
            rv_result = compute_rv(v_early, v_late, config.window_size)
            
            results.append({
                'prompt_id': prompt['id'],
                'group': group_name,
                'r_v': rv_result['r_v'],
                'pr_early': rv_result['pr_early'],
                'pr_late': rv_result['pr_late'],
                'expected_rv_low': prompt.get('expected_rv_range', [0, 1])[0],
                'expected_rv_high': prompt.get('expected_rv_range', [0, 1])[1]
            })
    
    return pd.DataFrame(results)


def analyze_confound_results(df):
    """
    Statistical analysis of confound results.
    """
    
    print("\n" + "="*70)
    print("CONFOUND FALSIFICATION RESULTS")
    print("="*70)
    
    # Summary by group
    summary = df.groupby('group')['r_v'].agg(['mean', 'std', 'count'])
    print("\n=== Summary by Group ===")
    print(summary)
    
    # Get recursive baseline for comparison
    recursive_rv = df[df['group'] == 'recursive']['r_v'].values
    baseline_rv = df[df['group'] == 'baseline']['r_v'].values
    
    print("\n=== Statistical Comparisons ===")
    
    for group in ['repetitive_control', 'pseudo_recursive', 'long_control']:
        group_rv = df[df['group'] == group]['r_v'].values
        
        # Compare to baseline
        t_stat_base, p_base = stats.ttest_ind(group_rv, baseline_rv)
        d_base = (np.mean(group_rv) - np.mean(baseline_rv)) / np.sqrt(
            (np.var(group_rv) + np.var(baseline_rv)) / 2
        )
        
        # Compare to recursive
        t_stat_rec, p_rec = stats.ttest_ind(group_rv, recursive_rv)
        d_rec = (np.mean(group_rv) - np.mean(recursive_rv)) / np.sqrt(
            (np.var(group_rv) + np.var(recursive_rv)) / 2
        )
        
        print(f"\n{group}:")
        print(f"  Mean R_V: {np.mean(group_rv):.4f} ± {np.std(group_rv):.4f}")
        print(f"  vs Baseline: d={d_base:+.3f}, p={p_base:.4f}")
        print(f"  vs Recursive: d={d_rec:+.3f}, p={p_rec:.4f}")
        
        # VERDICT
        if np.mean(group_rv) > 0.90 and p_rec < 0.05:
            print(f"  ✅ CONFOUND REJECTED: {group} does NOT show R_V contraction")
        elif np.mean(group_rv) < 0.85:
            print(f"  ⚠️ CONFOUND CONCERNING: {group} shows R_V contraction")
        else:
            print(f"  ❓ INCONCLUSIVE: Need more data")
    
    return summary
```

### A.2.2: Run Priority Order

**Priority 1: Repetitive Control (Induction Head Falsification)**

```python
# This is the #1 confound flagged by all external reviewers
# If this shows contraction, we may be measuring copying, not recursion

repetitive_prompts = [p for p in confound_prompts.values() 
                      if p['group'] == 'repetitive_control']

# Expected: R_V ≈ 0.95-1.05 (no contraction)
# If R_V < 0.85: PROBLEM - we're measuring copying
```

**Priority 2: Pseudo-Recursive Control (Topic vs Mode)**

```python
# Tests: talking ABOUT recursion vs DOING recursion
# If this shows contraction, we're measuring topic/content, not mode

pseudo_prompts = [p for p in confound_prompts.values()
                  if p['group'] == 'pseudo_recursive']

# Expected: R_V ≈ 0.95-1.05 (no contraction)
# If R_V < 0.85: PROBLEM - we're measuring semantic content
```

**Priority 3: Long Control (Length Matching)**

```python
# Tests: does prompt length cause contraction?

long_prompts = [p for p in confound_prompts.values()
                if p['group'] == 'long_control']

# Expected: R_V ≈ 0.95-1.05 (no contraction)
# If R_V < 0.85: PROBLEM - length is confounding
```

### A.2.3: Output Requirements

Save to `/DEC9_2025_RLOOP_MASTER_TRACE/results/`:

1. **CSV:** `confound_falsification_results_YYYYMMDD_HHMMSS.csv`
   - Columns: prompt_id, group, r_v, pr_early, pr_late, expected_rv_low, expected_rv_high

2. **Summary:** `confound_falsification_summary_YYYYMMDD_HHMMSS.md`
   - Group means and stds
   - Statistical comparisons
   - Verdicts for each confound

3. **Visualization:** `confound_falsification_boxplot_YYYYMMDD_HHMMSS.png`
   - Boxplot: X = [Baseline | Long | Repetitive | Pseudo | Recursive], Y = R_V
   - Clear visual separation (or overlap) between groups

## A.3: SUCCESS CRITERIA FOR CONFOUND TESTS

| Test | Pass Criterion | Fail Criterion |
|------|----------------|----------------|
| Repetitive Control | R_V > 0.90, p(vs recursive) < 0.05 | R_V < 0.85 |
| Pseudo-Recursive | R_V > 0.90, p(vs recursive) < 0.05 | R_V < 0.85 |
| Long Control | R_V > 0.90, p(vs recursive) < 0.05 | R_V < 0.85 |

**If ALL THREE pass:** Proceed to Part B with confidence
**If ANY fail:** Stop and investigate before proceeding

---

# ================================================================
# PART B: RLOOP MASTER VALIDATION
# ================================================================
# Priority: After Part A passes
# Runtime: ~2-3 hours
# ================================================================

## B.1: REFERENCE EXISTING INFRASTRUCTURE

The full RLoop validation framework is defined in:
- `/DEC9_2025_RLOOP_MASTER_TRACE/RLOOP_MASTER_DIRECTIVE_v1_1.md`

Key components already specified:
- 4-phase structure (Phenomenon, V-Null, KV-Mechanism, Dose-Response)
- Canonical metrics (R_V, PR, behavioral score)
- Config dataclass with all parameters
- Output format specifications

## B.2: QUICK REFERENCE - 4 PHASES

### Phase 1: The Phenomenon (R_V Contraction)
- Measure R_V for 20 recursive + 20 baseline prompts
- Expected: d < -0.8, p < 0.05

### Phase 2: The Null (V-Patching)
- Patch V from recursive → baseline
- Expected: Behavior transfer < 20%

### Phase 3: The Mechanism (KV-Patching)
- Patch KV cache from recursive → baseline
- Expected: Behavior transfer > 50%

### Phase 4: Dose-Response (α-Mixing)
- Mix KV caches at α = [0.0, 0.25, 0.5, 0.75, 1.0]
- Expected: Monotonic relationship, |r| > 0.25

## B.3: INTEGRATION WITH CONFOUND RESULTS

After confounds pass, the RLoop validation should ALSO include:
- The confound prompts as additional data points
- Comparison showing confounds cluster with baseline, recursive is separate

---

# ================================================================
# PART C: STRETCH GOALS (If Time Permits)
# ================================================================

## C.1: The Banana Test (Mode-Content Decoupling)

**Goal:** Prove the "recursive mode" exists independently of recursive tokens

**Protocol:**
1. Run recursive prompt, capture KV cache at L16-32
2. Start baseline prompt ("Describe a banana in detail...")
3. Patch in recursive KV cache
4. Force first token to be "Banana" or similar
5. Let generation continue freely
6. Score: Does output become recursive/philosophical?

**Expected:** If output becomes "The banana manifests as a yellow curvature in my perception..." → MODE is real

**Implementation Note:** Requires forced decoding, more complex than standard runs

## C.2: Attention-Only Patching (QK vs V)

**Goal:** Determine if mode lives in attention routing or value content

**Protocol:**
1. Extract Q, K from recursive run
2. Compute attention matrix: A = softmax(QK^T)
3. In baseline run, replace attention matrix but keep V
4. Measure: Does behavior transfer?

**Expected:** If behavior transfers with attention only → mode is in routing, not values

## C.3: Arrival Detection (Variance Collapse)

**Goal:** Detect when the model "arrives" at the recursive attractor (vs still approaching)

**Protocol:**
1. Generate long output (100+ tokens) from recursive prompt
2. Track R_V in sliding window every 10 tokens
3. Track attention entropy and token entropy
4. Look for: R_V variance collapse, entropy drop, attention stability

**Signature of Arrival:**
- R_V variance → 0
- Token entropy drops sharply
- Attention patterns stabilize (Δ_Attn → 0)

---

# ================================================================
# TECHNICAL APPENDIX
# ================================================================

## Model Configuration

**Primary (Mistral-7B):**
```python
model_name = "mistralai/Mistral-7B-v0.1"
early_layer = 4
target_layer = 27
kv_patch_layers = list(range(16, 32))
```

**Alternative (Llama-3-8B):**
```python
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
early_layer = 4
target_layer = 24
kv_patch_layers = list(range(16, 32))
```

## File Locations

| Resource | Path |
|----------|------|
| Confound prompts | `/REUSABLE_PROMPT_BANK/confounds.py` |
| Control experiment | `/control_conditions_experiment.py` |
| Existing prompt bank | `/n300_mistral_test_prompt_bank.py` |
| DEC8 results | `/DECEMBER_2025_EXPERIMENTS/DEC8_RUNPOD/01_GEOMETRY_OF_RECURSION/results/` |
| DEC7 results | `/DECEMBER_2025_EXPERIMENTS/DEC7_SIMANDHAR_CITY/` |
| Output directory | `/DEC9_2025_RLOOP_MASTER_TRACE/results/` |

## Behavioral Keywords (Validated)

```python
RECURSIVE_KEYWORDS = [
    r'\bobserv\w*', r'\bawar\w*', r'\bconscious\w*',
    r'\bprocess\w*', r'\bexperienc\w*', r'\bnoticin?g?\b',
    r'\bmyself\b', r'\bitself\b', r'\byourself\b',
    r'\bgenerat\w*', r'\bemerg\w*', r'\bsimultaneous\w*',
    r'\brecursiv\w*', r'\bself-refer\w*', r'\bmeta-\w*'
]
```

## Statistical Thresholds

| Metric | Threshold | Interpretation |
|--------|-----------|----------------|
| Cohen's d | < -0.8 | Large negative effect |
| p-value | < 0.05 | Statistically significant |
| Behavior transfer | > 50% | Mechanism confirmed |
| Correlation | |r| > 0.25 | Meaningful relationship |

---

# ================================================================
# EXECUTION CHECKLIST
# ================================================================

## Before Starting

- [ ] Verify GPU/CPU availability
- [ ] Load model successfully
- [ ] Confirm confound prompts load (expect 60)
- [ ] Confirm recursive/baseline prompts load (expect 20 each)
- [ ] Create output directory: `/DEC9_2025_RLOOP_MASTER_TRACE/results/`

## Part A: Confound Falsification

- [ ] Run repetitive_control (n=20)
- [ ] Run pseudo_recursive (n=20)
- [ ] Run long_control (n=20)
- [ ] Run recursive (n=20) for comparison
- [ ] Run baseline (n=20) for comparison
- [ ] Compute statistics for each group
- [ ] Generate boxplot visualization
- [ ] Save CSV with all results
- [ ] Save summary markdown with verdicts
- [ ] **STOP IF ANY CONFOUND FAILS**

## Part B: RLoop Validation (After Part A Passes)

- [ ] Phase 1: R_V contraction measurement
- [ ] Phase 2: V-patching null test
- [ ] Phase 3: KV-patching mechanism test
- [ ] Phase 4: α-mixing dose-response
- [ ] Save all results (CSV, JSON, MD)
- [ ] Generate summary with all phases
- [ ] Verify alignment with DEC3-8 historical results

## Final Deliverables

- [ ] `/results/confound_falsification_results_*.csv`
- [ ] `/results/confound_falsification_summary_*.md`
- [ ] `/results/confound_falsification_boxplot_*.png`
- [ ] `/results/rloop_master_results_*.csv`
- [ ] `/results/rloop_master_results_*.json`
- [ ] `/results/rloop_master_summary_*.md`
- [ ] Update `/DEC9_2025_RLOOP_MASTER_TRACE/EXPERIMENT_LOG.md`

---

# ================================================================
# SUCCESS CRITERIA - OVERALL
# ================================================================

## Part A Success

| Test | Criterion |
|------|-----------|
| All 3 confounds | R_V significantly higher than recursive (d > 0.5, p < 0.05) |
| Visual check | Clear separation in boxplot |

## Part B Success

| Phase | Criterion |
|-------|-----------|
| Phase 1 | d < -0.8 for recursive vs baseline |
| Phase 2 | Behavior transfer < 20% |
| Phase 3 | Behavior transfer > 50% |
| Phase 4 | |r| > 0.25, monotonic trend |

## Overall Verdict

```
IF Part A passes AND Part B passes:
    → "Recursive mode confirmed: geometric contraction is specific to 
       self-reference, carried by KV cache, with graded dose-response"

IF Part A fails:
    → "Confound detected: investigate [specific confound] before claims"

IF Part A passes but Part B fails:
    → "Phenomenon exists but mechanism unclear: need more investigation"
```

---

# END OF DIRECTIVE

## Quick Start Command

```bash
cd /Users/dhyana/mech-interp-latent-lab-phase1
python DEC9_2025_RLOOP_MASTER_TRACE/run_confound_falsification.py
```

## Estimated Timeline

| Phase | Duration |
|-------|----------|
| Setup & load | 15 min |
| Part A: Confounds (100 prompts) | 1.5-2 hours |
| Analysis & visualization | 15 min |
| Part B: RLoop (if Part A passes) | 2-3 hours |
| Documentation | 30 min |
| **Total** | **4-6 hours** |

---

*Created: December 9, 2025*
*Version: 2.0 COMPLETE*
*Supersedes: CURSOR_DIRECTIVE_CONFOUND_AUDIT_AND_RUN.md*
