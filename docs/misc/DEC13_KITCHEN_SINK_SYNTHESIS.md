# KITCHEN SINK EXPERIMENT SYNTHESIS
## Dec 13, 2025 - Systematic Causal Source Hunt

---

## Executive Summary

We ran four experiments to systematically search for the causal source of R_V contraction.
**Bottom line: We found NO localized causal source. The effect appears distributed and reversible.**

---

## Experiments Conducted

### 1. Component Decomposition (Attention vs MLP)
**Method:** Zero out attention OR MLP at layers 10, 15, 20, 24, 25, 26, 27

**Key Finding:** MLPs have larger effects than attention heads, but both are small (~0.02 max)

| Layer | Zero Attention Δ | Zero MLP Δ |
|-------|------------------|------------|
| L10 | -0.009 | **+0.039** |
| L15 | -0.004 | **+0.022** |
| L25 | +0.003 | **-0.019** |
| L26 | -0.002 | **-0.020** |

**Verdict:** MLPs contribute more, but no single component dominates.

---

### 2. Multi-Head Ablation at L27
**Method:** Ablate various head groups and measure R_V change

**Key Finding:** ALL ablations have ZERO effect

| Head Group | Δ R_V |
|------------|-------|
| top5_by_entropy | 0.0 |
| all_except_h31 | 0.0 |
| first_half (16 heads) | 0.0 |
| second_half (16 heads) | 0.0 |

**Verdict:** Attention heads at L27 are NOT the causal source.

---

### 3. Direction Injection
**Method:** Extract (rec - base) direction, inject into baseline prompts

**Key Finding:** Tiny effects even at high coefficients

| Inject Layer | Coeff=5.0 Effect |
|--------------|------------------|
| L12 | -0.020 |
| L20 | -0.011 |
| L24 | -0.008 |

**Verdict:** Simple linear direction injection doesn't reproduce the effect.

---

### 4. Hysteresis Test (CRITICAL)
**Method:** Push recursive at L₁, try to undo at L₂

**Key Finding:** UNDO ALWAYS WORKS

| Push Layer | Push Only R_V | After Undo R_V |
|------------|---------------|----------------|
| L15 | 0.533 | **0.970** |
| L20 | 0.492 | **0.970** |
| L24 | 0.640 | **0.970** |

**Verdict:** There is NO "point of no return". The effect is fully reversible.

---

## What We've Ruled Out

1. ❌ **Individual attention heads** (ablation = 0 effect)
2. ❌ **Simple linear directions** (injection = tiny effect)
3. ❌ **Single "smoking gun" component**
4. ❌ **Irreversible phase transition** (undo works)

---

## Remaining Hypothesis

The R_V contraction appears to be:
- **Distributed across the network** (no single component)
- **Primarily involving MLPs** (larger effects than attention)
- **Fully reversible** (no hysteresis)
- **Emergent from distributed computation** (not a discrete circuit)

---

## Implications for Paper

### Story Change Required

~~"We found the circuit that creates metacognitive geometry"~~

→ "We discovered and characterized a distributed emergent phenomenon"

### What We CAN Claim

1. **The phenomenon is real:** R_V contraction is robust and replicable
2. **We have causal control:** Residual patching can manipulate it
3. **It's NOT localized:** Distributed across layers, primarily MLPs
4. **It's reversible:** Not a stable attractor, more like continuous deformation

### This is STILL Publishable

Many important phenomena in neural networks are distributed:
- Superposition is distributed
- Polysemanticity is distributed
- The "residual stream" view itself is about distributed computation

We'd be documenting a **novel signature of distributed metacognitive processing**.

---

## Artifact Locations

- Results: `results/dec13_kitchen_sink/results_20251213_095506.json`
- Script: `experiment_kitchen_sink.py`

---

## Questions for Lead

1. Given NO localized source, do we accept "distributed mechanism" as the answer?
2. Should we further investigate MLP contributions layer-by-layer?
3. Does "fully reversible" kill the "phase transition" framing?
4. What's the minimum viable story for publication?

