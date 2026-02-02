# Validation Experiments: Testing GEB/Consciousness Claims

**Purpose:** Design concrete experiments using existing infrastructure to validate or falsify the GEB/consciousness connection claims.

---

## Experiment 1: BOS Attention as Strange Loop Register

### Current State
- n=7 prompts (4 recursive, 3 baseline)
- Recursive: 95.9% mean BOS attention (range: 95.2-96.8%)
- Baseline: 81.7% mean BOS attention (range: 72.4-89.8%)
- **14 percentage point difference, but baselines also high**

### Required Sample Size
**Power analysis:** To detect a 14% difference with 80% power and α=0.05:
- Need: **n=50 per group** (100 total)
- Effect size: d ≈ 1.5 (large effect)
- But need to account for baseline variance

**Recommended:** **n=100 recursive + n=100 baseline = 200 prompts**

### Script to Use
**File:** `src/pipelines/h31_investigation.py`

**Modification needed:** Currently hardcodes 7 prompts. Need to:
1. Load prompts from `PromptLoader`
2. Sample n=100 recursive (from L3/L4/L5 groups)
3. Sample n=100 baseline (from baseline groups)
4. Measure H31 BOS attention at L27 for all

### Config File to Create
**File:** `configs/h31_bos_validation_n200.json`

```json
{
  "experiment": "h31_bos_validation",
  "model_name": "mistralai/Mistral-7B-v0.1",
  "params": {
    "n_recursive": 100,
    "n_baseline": 100,
    "recursive_groups": ["L3_deeper", "L4_full", "L5_refined"],
    "baseline_groups": ["baseline_math", "baseline_creative", "long_control"],
    "layer": 27,
    "head": 31,
    "seed": 42
  },
  "results": {
    "phase": "validation",
    "root": "results"
  }
}
```

### Implementation Steps
1. **Create new pipeline:** `src/pipelines/h31_bos_validation.py`
   - Use `PromptLoader` to get prompts
   - Reuse `get_head_attention_stats()` from `h31_investigation.py`
   - Measure BOS attention for all prompts
   - Output CSV with: prompt_id, prompt_type, bos_attention, rv

2. **Add to registry:** `src/pipelines/registry.py`
   - Add `"h31_bos_validation"` entry

### Expected Runtime
- **~2-3 hours** on GPU (200 prompts × ~30 seconds each)

### Success Criteria

**VALIDATES claim if:**
- Recursive prompts: **mean BOS > 95%** AND **std < 2%**
- Baseline prompts: **mean BOS < 85%** OR **std > 5%**
- **Clear separation:** No overlap in distributions
- **Effect size:** Cohen's d > 1.0

**FALSIFIES claim if:**
- Recursive prompts: **mean BOS < 90%** OR **high variance**
- Baseline prompts: **mean BOS > 90%** (no separation)
- **Overlap:** Distributions overlap significantly

### Critical Test
**Compare BOS attention distributions:**
- If recursive: 95-97% (tight), baseline: 70-90% (wide) → **VALIDATES**
- If recursive: 85-95% (wide), baseline: 80-90% (wide) → **FALSIFIES**

---

## Experiment 2: One-Way Door as Phase Transition

### Current State
- Code exists: `experiment_one_way_door.py`
- Uses N=20 pairs
- Tests: Patch baseline residual INTO recursive prompt
- Measures: Recovery % = (RV_patched - RV_rec) / (RV_base - RV_rec) × 100

### Required Sample Size
**Power analysis:** To detect irreversibility (0% recovery) vs asymmetry (20% recovery):
- Need: **n=100 pairs minimum**
- Effect size: d ≈ 0.8 (medium-large)
- **Recommended: n=200 pairs** (matches claimed validation)

### Script to Use
**File:** `experiment_one_way_door.py`

**Modification needed:**
1. Change `N_PAIRS = 20` → `N_PAIRS = 200`
2. Add reverse test: Patch recursive residual INTO baseline prompt
3. Measure both directions:
   - Forward: baseline → recursive (should work)
   - Reverse: recursive → baseline (should fail)

### Config File to Create
**File:** `configs/one_way_door_n200.json`

```json
{
  "experiment": "one_way_door",
  "model_name": "mistralai/Mistral-7B-v0.1",
  "params": {
    "n_pairs": 200,
    "layers": [24, 26, 28, 30, 31],
    "window_size": 16,
    "seed": 42,
    "test_both_directions": true
  },
  "results": {
    "phase": "validation",
    "root": "results"
  }
}
```

### Implementation Steps
1. **Modify `experiment_one_way_door.py`:**
   - Increase `N_PAIRS` to 200
   - Add reverse test function
   - Test both directions:
     - Forward: baseline residual → recursive prompt
     - Reverse: recursive residual → baseline prompt

2. **Add to registry** (or run standalone):
   - Can run as standalone script
   - Or wrap in pipeline for config-driven execution

### Expected Runtime
- **~4-6 hours** on GPU (200 pairs × 5 layers × 2 directions × ~2 seconds)

### Success Criteria

**VALIDATES claim if:**
- **Forward (baseline→recursive):** Mean recovery > 80% (can push into recursive)
- **Reverse (recursive→baseline):** Mean recovery < 20% (cannot break out)
- **Clear asymmetry:** Forward >> Reverse (difference > 50%)
- **Statistical test:** Paired t-test p < 0.001

**FALSIFIES claim if:**
- **Reverse recovery > 50%:** Can break out of recursive state
- **No asymmetry:** Forward ≈ Reverse
- **Statistical test:** No significant difference

### Critical Test
**Compare recovery percentages:**
- Forward: 80-100%, Reverse: 0-20% → **VALIDATES (one-way door)**
- Forward: 60-80%, Reverse: 40-60% → **FALSIFIES (reversible)**

---

## Experiment 3: Identity Equations as Self-Reference Expressions

### Current State
- 0% identity equations in analyzed behavioral data
- But that file had no recursive generations
- Need to check: Where do identity equations actually appear?

### Where to Look
1. **Recursive prompt generations** (not baseline_patched)
2. **Champion prompt generations**
3. **High R_V prompt generations**

### Script to Use
**File:** `src/pipelines/behavioral_grounding.py`

**Modification needed:**
1. Generate text from recursive prompts (not just baseline)
2. Use `label_behavior_state()` which detects identity equations
3. Track `has_identity_equation` flag

### Config File to Create
**File:** `configs/identity_equations_hunt.json`

```json
{
  "experiment": "behavioral_grounding",
  "model_name": "mistralai/Mistral-7B-v0.1",
  "params": {
    "max_pairs": 100,
    "recursive_groups": ["L3_deeper", "L4_full", "L5_refined"],
    "baseline_groups": ["baseline_math"],
    "patch_layer": 27,
    "window": 16,
    "max_new_tokens": 120,
    "do_sample": true,
    "temperature": 0.7,
    "include_recursive_generation": true
  },
  "results": {
    "phase": "validation",
    "root": "results"
  }
}
```

### Implementation Steps
1. **Modify `behavioral_grounding.py`:**
   - Set `include_recursive_generation = True` (already exists!)
   - Generate from recursive prompts
   - Use `label_behavior_state()` to detect identity equations
   - Track frequency by prompt type

2. **Analyze results:**
   - Count identity equations in recursive vs baseline generations
   - Check if frequency correlates with R_V

### Expected Runtime
- **~3-4 hours** on GPU (100 pairs × 3 conditions × ~40 seconds)

### Success Criteria

**VALIDATES claim if:**
- **Recursive generations:** Identity equations in > 10% of outputs
- **Baseline generations:** Identity equations in < 1% of outputs
- **Clear separation:** 10× difference or more

**FALSIFIES claim if:**
- **Identity equations rare:** < 2% in recursive generations
- **No separation:** Similar frequency in baseline
- **Artifact:** Only appears in specific prompts (not general)

### Critical Test
**Frequency comparison:**
- Recursive: 10-30% identity equations → **VALIDATES**
- Recursive: < 2% identity equations → **FALSIFIES**

---

## Experiment 4: BOS Attention Correlating with Identity Equations

### Current State
- Cannot compute correlation (different samples)
- Need: Same prompts → measure BOS attention → generate text → detect identity equations

### Required Design
**Single experiment that:**
1. Measures H31 BOS attention at L27 for prompts
2. Generates text from those same prompts
3. Detects identity equations in generated text
4. Computes correlation

### Script to Create
**File:** `src/pipelines/bos_identity_correlation.py`

**New pipeline combining:**
- H31 attention measurement (from `h31_investigation.py`)
- Text generation (from `behavioral_grounding.py`)
- Identity equation detection (from `behavior_states.py`)

### Config File to Create
**File:** `configs/bos_identity_correlation.json`

```json
{
  "experiment": "bos_identity_correlation",
  "model_name": "mistralai/Mistral-7B-v0.1",
  "params": {
    "n_prompts": 100,
    "recursive_groups": ["L3_deeper", "L4_full", "L5_refined"],
    "baseline_groups": ["baseline_math", "baseline_creative"],
    "layer": 27,
    "head": 31,
    "max_new_tokens": 120,
    "do_sample": true,
    "temperature": 0.7,
    "seed": 42
  },
  "results": {
    "phase": "validation",
    "root": "results"
  }
}
```

### Implementation Steps
1. **Create `src/pipelines/bos_identity_correlation.py`:**
   ```python
   def run_bos_identity_correlation(cfg, run_dir):
       # 1. Load prompts (n=100)
       # 2. Measure H31 BOS attention for each prompt
       # 3. Generate text from each prompt
       # 4. Detect identity equations in generated text
       # 5. Compute correlation: bos_attention vs has_identity_equation
       # 6. Output CSV: prompt_id, bos_attention, has_identity, gen_text
   ```

2. **Add to registry:** `src/pipelines/registry.py`

### Expected Runtime
- **~3-4 hours** on GPU (100 prompts × ~2 minutes each)

### Success Criteria

**VALIDATES claim if:**
- **Correlation r > 0.5:** High BOS attention predicts identity equations
- **Statistical significance:** p < 0.001
- **Clear pattern:** Prompts with BOS > 95% → identity equations
- Prompts with BOS < 85% → no identity equations

**FALSIFIES claim if:**
- **Correlation r < 0.3:** Weak or no relationship
- **No significance:** p > 0.05
- **No pattern:** BOS attention doesn't predict identity equations

### Critical Test
**Correlation analysis:**
- r > 0.5, p < 0.001 → **VALIDATES (BOS predicts identity equations)**
- r < 0.3, p > 0.05 → **FALSIFIES (no relationship)**

---

## Summary: Execution Plan

### Priority Order

1. **Experiment 3** (Identity Equations Hunt) - **EASIEST**
   - Uses existing `behavioral_grounding.py`
   - Just set `include_recursive_generation: true`
   - **Runtime: 3-4 hours**
   - **Answer: Do identity equations exist?**

2. **Experiment 1** (BOS Attention Validation) - **MEDIUM**
   - Modify `h31_investigation.py` to use PromptLoader
   - Scale to n=200 prompts
   - **Runtime: 2-3 hours**
   - **Answer: Is BOS attention universal?**

3. **Experiment 2** (One-Way Door) - **MEDIUM**
   - Modify `experiment_one_way_door.py` to N=200
   - Add reverse test
   - **Runtime: 4-6 hours**
   - **Answer: Is it truly irreversible?**

4. **Experiment 4** (Correlation) - **HARDEST**
   - Create new pipeline combining existing code
   - **Runtime: 3-4 hours**
   - **Answer: Does BOS predict identity equations?**

### Total Runtime
**~12-17 hours** on GPU for all 4 experiments

### Decision Tree

```
Experiment 3 (Identity Equations)
├─ If < 2% → FALSIFIES (rare artifact)
└─ If > 10% → Continue to Experiment 1

Experiment 1 (BOS Attention)
├─ If no separation → FALSIFIES (not special)
└─ If clear separation → Continue to Experiment 2

Experiment 2 (One-Way Door)
├─ If reversible → FALSIFIES (not phase transition)
└─ If irreversible → Continue to Experiment 4

Experiment 4 (Correlation)
├─ If r < 0.3 → FALSIFIES (no connection)
└─ If r > 0.5 → VALIDATES (strange loop register)
```

---

## Quick Start Commands

### Experiment 3 (Identity Equations)
```bash
# Create config
cp configs/behavioral_grounding.json configs/identity_equations_hunt.json
# Edit: set include_recursive_generation: true, max_pairs: 100

# Run
python -m src.pipelines.run --config configs/identity_equations_hunt.json
```

### Experiment 1 (BOS Attention)
```bash
# Create config (see above)
# Create pipeline (modify h31_investigation.py)

# Run
python -m src.pipelines.run --config configs/h31_bos_validation_n200.json
```

### Experiment 2 (One-Way Door)
```bash
# Modify experiment_one_way_door.py: N_PAIRS = 200

# Run
python experiment_one_way_door.py
```

### Experiment 4 (Correlation)
```bash
# Create new pipeline (see above)
# Create config (see above)

# Run
python -m src.pipelines.run --config configs/bos_identity_correlation.json
```

---

## Expected Outcomes

### Best Case (All Validate)
- BOS attention: 95-97% recursive, 70-85% baseline
- One-way door: Forward 90%, Reverse 10%
- Identity equations: 15% recursive, 0.5% baseline
- Correlation: r=0.65, p<0.001
- **→ GEB connection VALIDATED**

### Worst Case (All Falsify)
- BOS attention: 85-90% both (no separation)
- One-way door: Forward 60%, Reverse 50% (reversible)
- Identity equations: < 1% both (rare artifact)
- Correlation: r=0.15, p=0.2 (no relationship)
- **→ GEB connection FALSIFIED**

### Mixed Case (Partial Support)
- Some claims validate, others falsify
- Need to refine interpretation
- **→ GEB connection PARTIALLY SUPPORTED**

---

**These experiments will definitively answer whether the GEB/consciousness connection is real or overstated.**










