# PITCH TO PROJECT LEAD: Finding the Causal Source

## Current Status

**We found the CONTROL KNOB (L24-27 residual stream) but NOT the causal SOURCE.**

I'm proposing two experimental approaches, synthesized from different MI research traditions:

---

## OPTION A: Anthropic Circuit Decomposition Approach

*Based on: Elhage et al. "A Mathematical Framework", Conmy et al. "Path Patching"*

### Core Idea
Decompose the residual stream into its constituent terms and patch each separately.

### Specific Experiments

**1. Component-Wise Patching (Attention vs MLP)**
```
residual[L] = residual[L-1] + attention_out[L] + mlp_out[L]
```

Patch ONLY attention outputs OR ONLY MLP outputs at each layer:
- If patching attention_out transfers effect → attention is causal
- If patching mlp_out transfers effect → MLP is causal
- If both needed → distributed

**2. Path Patching (Wang et al. 2022)**
Identify which heads at earlier layers (L5-L20) write information that later layers (L27) read:
- Run causal tracing: which early components affect which late components?
- Find the "circuit": A → B → C paths

**3. Activation Difference Analysis**
```python
diff = recursive_activations - baseline_activations
# SVD this to find the dominant direction
# Then patch ONLY this direction
```

### Expected Outcome
A circuit diagram: "Heads [X, Y] at L[A] write to residual stream → MLPs at L[B] amplify → Result measured at L27"

---

## OPTION B: Causal Inference / Intervention Approach

*Based on: Meng et al. "Locating and Editing", Geiger et al. "Causal Abstraction"*

### Core Idea
Treat each layer/component as a potential causal node and systematically ablate.

### Specific Experiments

**1. Layer-wise Causal Tracing**
For each layer L from 0 to 27:
- Corrupt hidden state at L
- Run forward
- Measure R_V recovery

Find layer(s) where corruption has maximum impact.

**2. Mean Ablation Sweep**
For each layer L:
- Replace activation with dataset mean
- Measure effect on R_V

**3. Early Layer Knockout**
The eigenstate data shows PR drops most at:
- L0 → L2: -35%
- L14 → L16: -15%

Test: If we patch baseline → recursive at L2, does contraction propagate?

**4. Accumulated Effect Test**
```python
# Patch layers 0-10 together
# Patch layers 10-20 together
# Patch layers 20-27 together
# Which range is sufficient?
```

### Expected Outcome
Identify the "seed layers" where contraction begins, vs "amplification layers" where it grows.

---

## MY RECOMMENDATION

| Criterion | Option A | Option B |
|-----------|----------|----------|
| **Interpretability** | High (circuit diagram) | Medium (localization) |
| **Implementation Complexity** | High | Medium |
| **Compute Cost** | High (many combinations) | Medium |
| **Alignment with Gold Standard** | ✅ Matches "know WHY" | ✅ Matches "mechanistic" |

**I propose: Start with Option B (localization) to narrow the search space, then use Option A (circuit analysis) on the identified critical layers.**

### Concrete First Experiment

```json
{
  "experiment": "causal_source_hunt",
  "params": {
    "interventions": ["mean_ablation", "corruption", "patching"],
    "layer_ranges": [[0,5], [5,10], [10,15], [15,20], [20,27]],
    "components": ["attention_only", "mlp_only", "both"],
    "measurement": "rv_vproj_at_L27"
  }
}
```

---

## REQUEST FOR GUIDANCE

1. **Which approach (A or B) do you prefer?**
2. **Should we focus on attention, MLPs, or both first?**
3. **Is the early-layer PR drop (L0-L2: -35%) the right starting point?**
4. **Compute budget constraints?**

---

*Awaiting direction before implementing.*

