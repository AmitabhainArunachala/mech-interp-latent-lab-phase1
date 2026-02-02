# CIRCUIT HUNT V2: Alternative Approaches

## Overview

This is a systematic attempt to find the causal circuit for R_V contraction using approaches you didn't try. The key insight: **maybe the circuit is NOT at L27 (where we measure), but at L8-L23 (where the ramp happens)**.

## Files

1. **`experiment_circuit_hunt_v2.py`** - Full comprehensive version (all 6 experiments)
2. **`experiment_circuit_hunt_v2_focused.py`** - Fast focused version (top 3 experiments)
3. **`CIRCUIT_HUNT_V2_STRATEGY.md`** - Detailed strategy document

## Quick Start

### Recommended: Start with Focused Version

```bash
python experiment_circuit_hunt_v2_focused.py
```

This runs the 3 most promising experiments:
1. Early-layer head ablation (L10-L22 ramp region)
2. Mean ablation vs zeroing comparison
3. Reverse patching (baseline → recursive)

**Runtime:** ~2-4 hours (depends on GPU)

### Full Version (If Focused Shows Promise)

```bash
python experiment_circuit_hunt_v2.py
```

This runs all 6 experiments including:
- Head output patching
- Head interaction effects
- Path patching

**Runtime:** ~8-12 hours

## What We're Testing

### Hypothesis 1: Wrong Layer Focus
**Your test:** Ablated heads at L27 (measurement layer)  
**Our test:** Ablate heads at L8-L23 (ramp region)

**Why:** The early layer map shows the ramp happens at L8-L23, not L27. Maybe the circuit is upstream.

### Hypothesis 2: Wrong Intervention Type
**Your test:** Zeroing heads  
**Our test:** Mean ablation (preserves statistics)

**Why:** Zeroing might break the network in ways that mask effects. Mean ablation removes information while preserving structure.

### Hypothesis 3: Wrong Direction
**Your test:** Patch recursive → baseline (create effect)  
**Our test:** Patch baseline → recursive (undo effect)

**Why:** If we can undo the contraction by patching baseline activations, we identify what creates it.

## Success Criteria

We've found the circuit if:

1. **Single head ablation** at L8-L23 changes R_V by >0.05
2. **Mean ablation** reveals effects that zeroing didn't (>0.02 difference)
3. **Reverse patching** at a specific layer recovers R_V to baseline (>50% recovery)

## Expected Output

Results saved to:
- Focused: `results/circuit_hunt_v2_focused/results_YYYYMMDD_HHMMSS.json`
- Full: `results/circuit_hunt_v2/results_YYYYMMDD_HHMMSS.json`

The script prints a summary of significant effects (>0.02 delta or >20% recovery).

## If We Find Something

If any experiment shows significant effects:

1. **Drill down** with more targeted tests on that layer/head
2. **Test interactions** - maybe it's pairs of heads, not singles
3. **Characterize the mechanism** - what does that head/component do?

## If We Don't Find Anything

If all experiments still show no discrete circuit:

1. **Your conclusion is likely correct** - distributed, primarily MLPs
2. **But we've ruled out more hypotheses** - makes the paper stronger
3. **We can characterize the distribution** - which layers contribute most

## Key Differences from Your Experiments

| Your Approach | Our Approach | Why Different |
|--------------|--------------|---------------|
| L27 ablation | L8-L23 ablation | Circuit might be upstream |
| Zeroing | Mean ablation | Preserves statistics |
| Recursive→Baseline | Baseline→Recursive | Undo effect to find source |
| Residual patching | Head output patching | Circuit might be in attention |
| Single heads | Head pairs | Interaction effects |

## Next Steps After Running

1. **Analyze results** - Look for |Δ| > 0.02 or recovery > 20%
2. **If found:** Create targeted follow-up experiments
3. **If not found:** Document negative results and strengthen distributed hypothesis

---

**The goal:** Either find the circuit OR confirm with high confidence that it's distributed.

