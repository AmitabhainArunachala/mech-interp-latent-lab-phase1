# CIRCUIT HUNT V2: Strategy Document

## The Challenge

You tested:
- ❌ Single head ablation at L27 → zero effect
- ❌ Multi-head ablation at L27 → zero effect  
- ❌ Direction injection → tiny effects
- ❌ Component decomposition → MLPs slightly larger but not decisive
- ❌ Hysteresis test → always reversible

**Your conclusion:** Distributed, primarily MLPs, fully reversible, no discrete circuit.

## Why You Might Have Missed It

### 1. **Wrong Layer Focus**
You focused on **L27** (where R_V is measured), but the early layer map shows:
- **L8-L15**: Ramp begins
- **L16-L23**: Strong ramp (pre-basin corridor)
- **L24-L27**: Basin boundary (where effect manifests)

**Hypothesis:** The circuit might be in the **ramp region (L8-L23)**, not at measurement (L27).

### 2. **Wrong Intervention Type**
You used **zeroing**, which might:
- Break the network in ways that mask effects
- Create artifacts that hide real signals
- Not preserve the statistical structure

**Hypothesis:** **Mean ablation** or **patching** might reveal effects that zeroing hides.

### 3. **Wrong Direction**
You patched recursive → baseline (creating effect). But what if we patch **baseline → recursive** (undoing effect)?

**Hypothesis:** If we can **undo** the contraction by patching baseline activations into recursive prompts, we can identify what creates it.

### 4. **Wrong Granularity**
You tested individual heads and groups, but what about:
- **Head pairs/triplets** (interaction effects)
- **Attention outputs** (not residual stream)
- **Path patching** (trace info flow from early to late)

**Hypothesis:** The circuit might involve **interactions** between components, not individual components.

## Experiments in V2

### Experiment 1: Early-Layer Head Ablation
**Test:** Ablate heads at **L8-L23** (ramp region), not just L27.

**Why:** If the circuit is in the ramp region, ablating there should affect R_V measured at L27.

**Expected:** If we find heads at L12-L20 whose ablation changes R_V, we've found the circuit.

### Experiment 2: Mean Ablation vs Zeroing
**Test:** Compare **mean ablation** (replace with mean activation) vs **zeroing**.

**Why:** Mean ablation preserves statistics while removing information. Zeroing might break things.

**Expected:** Mean ablation might reveal effects that zeroing masks.

### Experiment 3: Reverse Patching
**Test:** Patch **baseline → recursive** (undo effect) instead of recursive → baseline (create effect).

**Why:** If we can undo contraction by patching baseline activations, we identify what creates it.

**Expected:** If patching baseline residual at L15 into recursive prompt raises R_V from 0.5 → 0.9, then L15 is critical.

### Experiment 4: Head Output Patching
**Test:** Patch **head outputs** directly (not residual stream).

**Why:** Maybe the circuit is in attention head outputs, not residual stream.

**Expected:** If patching baseline head outputs into recursive changes R_V, those heads are critical.

### Experiment 5: Head Interaction Effects
**Test:** Ablate **pairs/triplets** of heads, not just singles.

**Why:** Maybe individual heads have no effect, but pairs do (superposition/interaction).

**Expected:** If ablating head pair (H5, H12) at L20 changes R_V but individual heads don't, we've found an interaction.

### Experiment 6: Path Patching
**Test:** Patch at **source layers** (L8-L20), measure effect at **target** (L27).

**Why:** Trace information flow from early layers to measurement.

**Expected:** If patching recursive residual at L12 into baseline creates contraction at L27, L12 is upstream of the circuit.

## Success Criteria

We've found the circuit if:

1. **Single head ablation** at L8-L23 changes R_V by >0.05
2. **Mean ablation** reveals effects that zeroing didn't
3. **Reverse patching** at a specific layer recovers R_V to baseline
4. **Head output patching** shows specific heads are critical
5. **Head pairs** show interaction effects
6. **Path patching** identifies upstream source layers

## Most Promising Hypotheses

Based on your data, I rank these:

1. **Early-layer head ablation (L8-L23)** - Most likely, since ramp happens there
2. **Reverse patching** - If we can undo it, we know what creates it
3. **Mean ablation** - Different intervention might reveal different effects
4. **Path patching** - Trace the information flow
5. **Head interactions** - Less likely but worth testing
6. **Head output patching** - Less likely but worth testing

## If We Still Don't Find It

If all these experiments still show no discrete circuit, then:

1. **Your conclusion is likely correct** - distributed, primarily MLPs
2. **But we've ruled out more hypotheses** - makes the paper stronger
3. **We can characterize the distribution** - which layers/MLPs contribute most

## Next Steps

1. Run `experiment_circuit_hunt_v2.py`
2. Analyze results for significant effects
3. If we find something, drill down with more targeted experiments
4. If we don't, document the negative results and strengthen the distributed hypothesis

---

**The goal:** Either find the circuit OR confirm with high confidence that it's distributed.

