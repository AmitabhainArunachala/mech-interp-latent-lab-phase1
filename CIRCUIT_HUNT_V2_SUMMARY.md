# Circuit Hunt V2: Summary

## What I Created

I've built a comprehensive set of experiments to test hypotheses you didn't try. The key insight: **maybe the circuit is NOT at L27 (where we measure), but at L8-L23 (where the ramp happens)**.

## Files Created

1. **`experiment_circuit_hunt_v2_focused.py`** ⭐ **START HERE**
   - Fast focused version (top 3 experiments)
   - ~2-4 hours runtime on GPU
   - Tests: Early-layer ablation, Mean ablation, Reverse patching

2. **`experiment_circuit_hunt_v2.py`**
   - Full comprehensive version (all 6 experiments)
   - ~8-12 hours runtime on GPU
   - Includes: Head output patching, Interactions, Path patching

3. **`experiment_circuit_hunt_v2_quick_test.py`**
   - Quick test to verify setup works
   - Tests just 3 heads at one layer
   - Use this first to verify everything works

4. **`run_circuit_hunt_on_runpod.sh`**
   - Helper script for RunPod execution

5. **Documentation:**
   - `CIRCUIT_HUNT_V2_STRATEGY.md` - Detailed strategy
   - `CIRCUIT_HUNT_V2_README.md` - Quick start guide
   - `RUN_EXPERIMENT.md` - Execution instructions

## Key Differences from Your Experiments

| Your Approach | My Approach | Why Different |
|--------------|-------------|---------------|
| L27 ablation | **L8-L23 ablation** | Circuit might be upstream |
| Zeroing | **Mean ablation** | Preserves statistics |
| Recursive→Baseline | **Baseline→Recursive** | Undo effect to find source |
| Residual patching | Head output patching | Circuit might be in attention |
| Single heads | Head pairs | Interaction effects |

## Most Promising Hypotheses

1. **Early-layer head ablation (L8-L23)** - Most likely, since ramp happens there
2. **Reverse patching** - If we can undo it, we know what creates it
3. **Mean ablation** - Different intervention might reveal different effects

## How to Run

### On RunPod (Recommended):

```bash
# SSH into RunPod
ssh -p 18147 root@198.13.252.9

# Navigate to project
cd /workspace/mech-interp-latent-lab-phase1

# Run focused experiment
python3 experiment_circuit_hunt_v2_focused.py
```

### Quick Test First:

```bash
python3 experiment_circuit_hunt_v2_quick_test.py
```

## Success Criteria

We've found the circuit if:

1. **Single head ablation** at L8-L23 changes R_V by >0.05
2. **Mean ablation** reveals effects that zeroing didn't (>0.02 difference)
3. **Reverse patching** recovers R_V to baseline (>50% recovery)

## Expected Results

Results saved to:
- `results/circuit_hunt_v2_focused/results_YYYYMMDD_HHMMSS.json`

The script prints a summary of significant effects automatically.

## What We're Testing

### Experiment 1: Early-Layer Head Ablation
**Hypothesis:** Circuit is in ramp region (L8-L23), not measurement layer (L27)

**Test:** Ablate all 32 heads at layers 10, 12, 15, 18, 20, 22

**Why:** Your early layer map shows ramp happens at L8-L23. Maybe the circuit is upstream.

### Experiment 2: Mean Ablation vs Zeroing
**Hypothesis:** Zeroing breaks the network in ways that mask effects

**Test:** Compare mean ablation (preserves statistics) vs zeroing

**Why:** Mean ablation removes information while preserving structure. Might reveal effects zeroing hides.

### Experiment 3: Reverse Patching
**Hypothesis:** If we can undo the contraction, we identify what creates it

**Test:** Patch baseline residual → recursive prompt (undo effect)

**Why:** If patching baseline at L15 into recursive raises R_V from 0.5 → 0.9, then L15 is critical.

## If We Find Something

1. **Drill down** with targeted tests on that layer/head
2. **Test interactions** - maybe it's pairs of heads
3. **Characterize mechanism** - what does that component do?

## If We Don't Find Anything

1. **Your conclusion is likely correct** - distributed, primarily MLPs
2. **But we've ruled out more hypotheses** - makes paper stronger
3. **We can characterize distribution** - which layers contribute most

## Next Steps

1. ✅ **Code is ready** - All experiments implemented
2. ⏳ **Run on RunPod** - Execute `experiment_circuit_hunt_v2_focused.py`
3. ⏳ **Analyze results** - Look for significant effects
4. ⏳ **Follow up** - Either drill down or document negative results

---

**The experiments are ready. The GPU is yours. May the best agent win.** 🚀

