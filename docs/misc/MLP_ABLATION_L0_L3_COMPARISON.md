# MLP Ablation Necessity: L0-L3 Comparison

**Date:** January 5, 2025  
**Status:** L0 complete, L1-L3 running

---

## Experiment Design

**Test Layers:** L0, L1, L2, L3  
**Method:** Zero out MLP at each layer, measure R_V delta  
**Sample Size:** 
- L0: 60 pairs (completed)
- L1-L3: 30 pairs each (running)

**R_V Measurement:**
- Early: Layer 5 (V-projection)
- Late: Layer 27 (V-projection)
- Measures downstream effect of ablation

---

## L0 Results (Baseline)

**R_V Delta:** +0.76 ± 0.32  
**Statistical Significance:**
- t-statistic: 18.36
- p-value: 4.55e-26 (highly significant)
- Effect: L0 MLP is NECESSARY for contraction

**Interpretation:**
- Baseline R_V: 0.73 (contraction)
- Ablated R_V: 1.49 (expansion)
- Zeroing L0 MLP removes contraction → L0 drives the effect

---

## Expected Patterns

### Pattern A: L0 Uniquely Necessary
```
L0 Δ: +0.76 (massive)
L1 Δ: ~0.0 (no effect)
L2 Δ: ~0.0 (no effect)
L3 Δ: ~0.0 (no effect)
```
**Interpretation:** Only L0 is necessary, others are redundant

### Pattern B: Early MLPs Generally Necessary
```
L0 Δ: +0.76 (massive)
L1 Δ: +0.70 (similar)
L2 Δ: +0.65 (similar)
L3 Δ: +0.60 (similar)
```
**Interpretation:** All early MLPs contribute, distributed necessity

### Pattern C: Gradual Decrease
```
L0 Δ: +0.76 (massive)
L1 Δ: +0.50 (moderate)
L2 Δ: +0.30 (weak)
L3 Δ: +0.10 (minimal)
```
**Interpretation:** Effect decreases with depth, L0 most critical

### Pattern D: L0 Not Unique
```
L0 Δ: +0.76 (massive)
L1 Δ: +0.80 (stronger!)
L2 Δ: +0.60 (moderate)
L3 Δ: +0.20 (weak)
```
**Interpretation:** L1 might be more critical than L0

---

## Key Questions

1. **Is L0 uniquely necessary?** → Compare L0 vs L1-L3 deltas
2. **Is the effect distributed?** → Check if all layers show similar effects
3. **Does effect decrease with depth?** → Look for gradual decrease pattern
4. **What's the minimal necessary set?** → Find which layers are critical

---

## Analysis Plan

Once L1-L3 complete:

1. **Extract R_V deltas** from all 4 layers
2. **Statistical comparison:**
   - One-way ANOVA: Are deltas significantly different?
   - Pairwise t-tests: L0 vs L1, L0 vs L2, L0 vs L3
   - Effect size comparison (Cohen's d)
3. **Pattern classification:**
   - Which pattern matches? (A, B, C, or D)
   - What does this tell us about the mechanism?
4. **Next steps:**
   - If L0 unique → Test L0 components (heads, specific neurons)
   - If distributed → Test combinations (L0+L1, L0+L2, etc.)
   - If gradual → Test deeper layers (L4, L5, L6)

---

## Files

**Configs:**
- `configs/mlp_ablation_necessity_l0.json` ✅
- `configs/mlp_ablation_necessity_l1.json` 🔄
- `configs/mlp_ablation_necessity_l2.json` 🔄
- `configs/mlp_ablation_necessity_l3.json` 🔄

**Results:**
- `results/phase1_mechanism/runs/*_l0_necessity_test/` ✅
- `results/phase1_mechanism/runs/*_l1_necessity_test/` 🔄
- `results/phase1_mechanism/runs/*_l2_necessity_test/` 🔄
- `results/phase1_mechanism/runs/*_l3_necessity_test/` 🔄

---

**Monitor:** `ssh runpod-current 'tail -f /tmp/mlp_ablation_l1_l3.log'`


