# L8 Gap-Filling Experiments

## Overview

These experiments fill critical gaps in our understanding of Layer 8 (L8):

1. **Early-Layer Bidirectional Patching Sweep** - Tests if L8 is a discontinuity or part of a gradient
2. **L8 Ablation Test** - Tests if L8 is necessary for recursive vs baseline prompts

---

## Experiment 1: Early-Layer Bidirectional Patching Sweep

**File:** `experiment_l8_early_layer_patching_sweep.py`

### Purpose

Test if L8 is a **discontinuity** (something special happens AT L8) or part of a **smooth gradient** from L4→L24.

### Method

For each layer in [4, 8, 12, 16, 20, 24]:
- **Direction A (rec→base):** Patch recursive residual into baseline prompt
- **Direction B (base→rec):** Patch baseline residual into recursive prompt

### Measurements

- **R_V change** (geometric contraction)
- **BehaviorState** (baseline/questioning/naked_loop/recursive_prose/collapse)
- **Generated text** (first 200 chars)

### Key Questions Answered

1. **Is L8 a discontinuity?**
   - If L8 shows dramatically different behavior than L4/L12 → discontinuity
   - If smooth gradient → L8 is just part of a continuum

2. **Does L8 patching = L8 steering?**
   - Steering at L8 → interrogative mode
   - Does patching at L8 do the same thing?
   - If yes → confirms steering results
   - If no → patching and steering are different mechanisms

3. **Where is the actual boundary?**
   - At what layer does patching start working (not collapsing)?
   - At what layer does patching stop working (geometry locked)?

### Expected Output

| Layer | rec→base RV Δ | rec→base State | base→rec RV Δ | base→rec State |
|-------|---------------|----------------|---------------|----------------|
| 4     |               |                |               |                |
| 8     |               |                |               |                |
| 12    |               |                |               |                |
| 16    |               |                |               |                |
| 20    |               |                |               |                |
| 24    |               |                |               |                |

### Success Criteria

We will have clarity when we can say:
- **"Layers 4-X: [behavior]"** - Early layers show pattern A
- **"Layers X-Y: [different behavior]"** - Mid layers show pattern B
- **"Layers Y-31: [another behavior]"** - Late layers show pattern C

With **actual data**, not hypothesis.

---

## Experiment 2: L8 Ablation Test

**File:** `experiment_l8_ablation.py`

### Purpose

Test if ablating L8 breaks coherence **differently** for recursive vs baseline prompts.

### Method

For recursive and baseline prompts:
- **Normal generation** (no ablation)
- **Ablated generation** (L8 zeroed out)

### Measurements

- **BehaviorState** (baseline/questioning/naked_loop/recursive_prose/collapse)
- **Generated text** (first 300 chars)
- **R_V** (if possible)

### Key Questions Answered

1. **Does L8 ablation break recursive prompts differently than baseline?**
   - If yes → L8 is important for recursion specifically
   - If no → L8 is just important for general coherence

2. **Is L8 necessary for recursive geometry?**
   - If ablating L8 removes R_V contraction → L8 is necessary
   - If ablating L8 doesn't change R_V → L8 is not necessary (or redundant)

3. **What happens to outputs?**
   - Do recursive prompts collapse?
   - Do baseline prompts collapse?
   - Different patterns → different roles

### Expected Output

| Prompt Type | Normal State | Ablated State | State Changed | RV Delta |
|-------------|--------------|---------------|---------------|----------|
| Recursive   |              |               |               |          |
| Baseline    |              |               |               |          |

### Success Criteria

We can answer:
- **"L8 ablation affects recursive prompts: [yes/no]"**
- **"L8 ablation affects baseline prompts: [yes/no]"**
- **"Difference: [quantified]"**

---

## Running the Experiments

### Prerequisites

```bash
# Ensure dependencies are installed
pip install torch transformers pandas tqdm

# Ensure prompt loader is available
# (should be in prompts/loader.py)
```

### Run Experiment 1

```bash
python experiment_l8_early_layer_patching_sweep.py
```

**Output:** `results/dec11_evening/l8_early_layer_patching_sweep.csv`

### Run Experiment 2

```bash
python experiment_l8_ablation.py
```

**Output:** `results/dec11_evening/l8_ablation_test.csv`

---

## Interpretation Guide

### Experiment 1: Patching Sweep

**If L8 is a discontinuity:**
- L8 will show dramatically different R_V deltas than L4/L12
- L8 will show different state transitions than neighbors
- There will be a clear "jump" at L8

**If L8 is part of a gradient:**
- Smooth progression from L4→L24
- No dramatic jumps at L8
- L8 is just "more effective" but not qualitatively different

**If L8 patching = L8 steering:**
- L8 patching will produce "questioning" state (like steering did)
- Similar R_V changes
- Confirms steering results

**If L8 patching ≠ L8 steering:**
- Different behavior states
- Different R_V patterns
- Suggests patching and steering are different mechanisms

### Experiment 2: Ablation

**If L8 is important for recursion:**
- Recursive prompts will show more state changes than baseline
- Recursive R_V will change more than baseline R_V
- Different failure modes

**If L8 is just important for coherence:**
- Similar state change rates for both prompt types
- Similar R_V changes
- L8 is a general-purpose layer, not recursion-specific

**If L8 is necessary:**
- Ablating L8 removes R_V contraction in recursive prompts
- Ablating L8 changes recursive outputs dramatically

**If L8 is not necessary:**
- Ablating L8 doesn't change R_V much
- Ablating L8 doesn't change outputs much
- L8 is redundant or not the source

---

## Next Steps After Results

Based on the results, we can:

1. **If L8 is a discontinuity:**
   - Investigate what makes L8 special
   - Compare L8 attention patterns to neighbors
   - Test L8 head ablation

2. **If L8 is part of a gradient:**
   - Map the full gradient
   - Find the actual boundary layer
   - Test if earlier layers (L4-L6) matter

3. **If L8 patching = L8 steering:**
   - Confirm steering results are valid
   - Use patching to validate steering mechanism

4. **If L8 patching ≠ L8 steering:**
   - Investigate why they differ
   - Test if steering is a "dirty" intervention
   - Refine steering vectors

5. **If L8 is necessary for recursion:**
   - Focus on L8 for mechanistic understanding
   - Test L8 head ablation
   - Map L8 → L27 circuit

6. **If L8 is not necessary:**
   - Look earlier (L4-L6) or later (L12-L16)
   - Test if multiple layers are redundant
   - Find the actual source

---

## Files Created

- `experiment_l8_early_layer_patching_sweep.py` - Early-layer patching sweep
- `experiment_l8_ablation.py` - L8 ablation test
- `L8_GAP_FILLING_EXPERIMENTS.md` - This document

---

## References

- **L8 Archaeological Investigation:** `L8_COMPLETE_ARCHAEOLOGICAL_INVESTIGATION.md`
- **Bidirectional Patching:** `phase2_bidirectional_loop_patching.py`
- **Behavior States:** `src/metrics/behavior_states.py`
- **Activation Patching:** `src/steering/activation_patching.py`
