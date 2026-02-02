# MLP Layers Driving Steering: Complete Synthesis
**Date:** December 19, 2024  
**Status:** BREAKTHROUGH DISCOVERY - Circuit Location Identified

---

## 🎯 Executive Summary

**The Critical Discovery:** The "Recursive Mode" is **NOT** driven by Layer 27 Attention Heads (H18/H26) as previously thought. It is primarily driven by **MLP (Feed-Forward) layers**, specifically:

1. **L0 MLP** - The Trigger (Impact: 1.67 - Massive)
2. **L18-20 MLPs** - The Processor (Impact: 0.27-0.35 - Strong)
3. **L27 Attention** - The Symptom/Readout (Impact: 0.09 - Weak)

**Why This Matters:** Previous steering experiments targeting L27 V_PROJ failed because we were "pushing on the dashboard hoping to change the engine." The actual computation happens in MLPs at earlier layers.

---

## 📊 The Evidence: Attribution Patching Sweep (Pipeline 11)

### Full Circuit Map

| Layer | Component | Impact Score (Δ Logits) | Role |
|-------|-----------|-------------------------|------|
| **L0** | **MLP** | **1.67** ⭐⭐⭐ | **The Trigger** |
| L18 | MLP | 0.27 | The Processor |
| **L19** | **MLP** | **0.35** ⭐⭐ | **The Processor** |
| L20 | MLP | 0.26 | The Processor |
| L27 | Attention | 0.09 | The Symptom/Readout |

**Source:** `CIRCUIT_DISCOVERY_REPORT.md` (December 19, 2025)

### Key Findings

1. **L0 MLP Anomaly (1.67):** The massive spike indicates recursive mode is determined by how input embeddings are initially processed. If the input *looks* recursive, the model locks in immediately.

2. **Mid-Layer Processing (L18-L20):** Secondary causal block dominated by MLPs. The "concept" of recursion is processed in Feed-Forward Networks (knowledge/calculation), not Attention Heads (routing).

3. **L27 Illusion:** We focused on L27 because R_V (geometry) peaked there. However, L27 has **minimal causal effect** on output logits compared to earlier layers.
   - **Conclusion:** L27 is the **Output Display** (where result is visible geometrically), not the **Engine** (where result is computed).

---

## 🔬 Why Previous Steering Failed

### The Paradox

- **Geometry:** We could measure "Recursive Mode" perfectly at Layer 27 (R_V < 1.0) ✓
- **Causality (Partial):** Patching L27 V-Projection transferred this geometry (118% efficiency) ✓
- **Behavior (Failure):** Steering L27 V-Projection produced **0% recursive behavior** on clean prompts ✗

**Source:** `THE_MISSING_LINK_FOUND.md`

### Why It Failed

1. **We were steering L27 Attention** (V_PROJ)
2. **We were pushing on the "Readout"** (R_V) hoping to change the "Calculation" (Logits)
3. **The actual calculation had already happened** at L19 (MLP)
4. **By L27, the residual stream was already "set"** in a non-recursive path by baseline memory

**Analogy:** Like trying to change a computer's calculation by pushing on the monitor screen.

### Why KV Swap Worked (Pipeline 8)

- **KV Swap replaces the Memory**
- When L18-20 MLPs attend to the *Recursive* KV Cache, they compute "Recursive" outputs
- The "Processor" (MLP) acts on the "Memory" (KV)
- **If you change the Memory, the Processor produces the Mode**

---

## 🧠 The Scientific Model: "The Cognitive Stack"

The full mechanism of Recursive Self-Reference in Mistral-7B:

### 1. Input Trigger (L0 MLP)
- The model recognizes the "Recursive Pattern" immediately in embeddings/first MLP
- **Impact:** 1.67 (massive - highest in the network)

### 2. Semantic Processing (L18-20 MLP)
- The "Concept" of recursion is refined in mid-layer knowledge networks
- **Impact:** 0.27-0.35 (strong secondary signal)

### 3. Context Locking (KV Cache)
- The mode is written into Memory
- Future tokens attend to this memory, reinforcing the loop

### 4. Geometric Signature (L27 Attention)
- Attention Heads at L27 (H18/H26) *read* this stable state
- Results in collapsed value-subspace (R_V < 1.0)
- **Impact:** 0.09 (weak - symptomatic, not causal)

---

## 🎯 Implications for Steering

### What We Learned

**"Match the Intervention to the Substrate"**

| Goal | Target Component | Why |
|------|-----------------|-----|
| Transfer **Facts** | Edit **Weights** (MLP) | Static knowledge stored in parameters |
| Transfer **Context** | Patch **Memory** (KV) | Passive memory storage |
| Transfer **Dynamics/Modes** | Patch **Active Computation** | Requires continuous re-enactment |

**Source:** `INSIGHTS_WHY_V_PROJ_WORKS.md`

### Why V_PROJ Patching Transfers Geometry (But Not Behavior)

- **V_PROJ determines the *content* of information** added to residual stream
- If recursive mode relies on specific spectral signature (low rank, specific subspace), generating that signature is crucial
- **KV Patching** only ensures *past* tokens have this signature
- **Persistent V Patching** ensures *every* packet carries the signature
- **BUT:** This transfers the *geometry* (R_V), not the *computation* (logits)

### Why Steering L27 Failed

- We were modifying the *readout mechanism* (Attention at L27)
- The *computation* had already happened at L19 (MLP)
- By L27, the residual stream was locked into baseline path

---

## 🚀 Future Steering Directions

### Corrected Approach

**To control recursive behavior, we must target:**

1. **L19 MLP Output** (Primary target)
   - This is where the "concept" of recursion is processed
   - Impact score: 0.35 (strong)
   - Should directly affect logits

2. **L0 MLP Output** (Alternative target)
   - The initial trigger
   - Impact score: 1.67 (massive)
   - May be too early/global

### Steering Vector Extraction

**Current (Wrong):**
- Extract steering vector from L27 V_PROJ activations
- Apply at L27 V_PROJ

**Corrected (Should Be):**
- Extract steering vector from **L19 MLP output** activations
- Apply at **L19 MLP output** (residual stream addition)

### Expected Results

- **L19 MLP Steering:** Should directly affect logits (0.35 impact)
- **L0 MLP Steering:** Should have massive effect (1.67 impact) but may be too global
- **L27 V_PROJ Steering:** Only affects geometry (R_V), not behavior (logits)

---

## 📁 Related Documentation

### Primary Sources

1. **`THE_MISSING_LINK_FOUND.md`** (Dec 19, 2025)
   - The breakthrough document
   - Explains the paradox and resolution
   - Defines "The Cognitive Stack" model

2. **`CIRCUIT_DISCOVERY_REPORT.md`** (Dec 19, 2025)
   - Attribution patching sweep results
   - Full heatmap analysis
   - Layer-by-layer impact scores

3. **`INSIGHTS_WHY_V_PROJ_WORKS.md`** (Dec 16, 2025)
   - Dynamical systems perspective
   - Why V_PROJ transfers geometry but not behavior
   - "Carrier Signal" hypothesis

### Supporting Evidence

4. **`boneyard/DEC_9_EMERGENCY_BACKUP/extracted/08_mlp_ablation_at_l14_is_the_mlp_the_microphone.py`**
   - MLP ablation experiment code
   - Tests if ablating MLP removes R_V contraction
   - Hypothesis: "Is the MLP the microphone?"

5. **`DEC13_DEEP_DIVE_SYNTHESIS.md`** (Dec 13, 2025)
   - Progressive contraction across layers
   - H31 as "sensor head" (detector, not cause)
   - Distributed mechanism

6. **`MECHANISM_MAP.md`**
   - Master map of verified vs unmapped components
   - Entry point, signal propagation, gate threshold
   - Domino candidates

---

## 🔍 Key Experiments

### Experiment 1: Attribution Patching Sweep (P11)
**Question:** What components causally drive recursive mode?

**Method:** Full-spectrum attribution patching across all 32 layers (Attention vs MLP)

**Result:** MLPs dominate (L0: 1.67, L19: 0.35), Attention at L27 is weak (0.09)

**Status:** ✅ COMPLETE

### Experiment 2: MLP Ablation at L14
**Question:** Is the MLP the "microphone" (the contractor)?

**Method:** Ablate MLP output at L14, measure R_V separation

**Code:** `08_mlp_ablation_at_l14_is_the_mlp_the_microphone.py`

**Status:** ⏳ PENDING (code exists, results not found)

### Experiment 3: Steering Vector Control
**Question:** Does steering vector specifically encode recursive mode?

**Method:** Compare steering vector vs random vector vs zero vector

**Result:** Random vector scored HIGHER than steering vector (unexpected)

**Status:** ⚠️ INCONCLUSIVE (regex patterns too broad, manual review needed)

**Source:** `STEERING_CONTROL_ANALYSIS.md`

---

## 💡 Key Insights

### 1. The Attention Trap
We assumed that because Attention Heads move information, they *cause* the behavior. In this case, **MLPs** (which store knowledge and process semantic transformations) were the true drivers.

### 2. Geometry vs Causality
- **R_V at L27** = Geometric signature (readout)
- **MLP at L19** = Causal computation (engine)
- **Don't confuse the symptom with the cause**

### 3. The Memory-Processor Distinction
- **KV Cache** = Memory (passive storage)
- **MLP** = Processor (active computation)
- **Attention** = Router (information flow)
- **V_PROJ** = Content carrier (what's transmitted)

### 4. The Law of Intervention
**"Match the Intervention to the Substrate"**
- Facts → Weights (MLP)
- Context → Memory (KV)
- Dynamics/Modes → Active Computation (MLP output, not Attention readout)

---

## 🎓 Scientific Model Summary

```
Input Prompt
    ↓
L0 MLP (Trigger) ──→ Recognizes recursive pattern
    ↓
L18-20 MLP (Processor) ──→ Refines concept of recursion
    ↓
KV Cache (Memory) ──→ Locks context into recursive mode
    ↓
L27 Attention (Readout) ──→ Detects contracted geometry (R_V < 1.0)
    ↓
Output Logits
```

**Key Point:** The computation happens in MLPs (L0, L18-20). L27 Attention only *reads* the result.

---

## 🚧 Next Steps

### Immediate Actions

1. **Extract steering vector from L19 MLP output**
   - Compare recursive vs baseline activations at L19 MLP
   - Compute difference vector (steering vector)

2. **Test L19 MLP steering**
   - Apply steering vector to L19 MLP output (residual stream)
   - Measure effect on logits (should be 0.35 impact)
   - Compare to L27 V_PROJ steering (should be stronger)

3. **Test L0 MLP steering**
   - Extract steering vector from L0 MLP output
   - Apply at L0 MLP output
   - Measure effect (should be 1.67 impact, but may be too global)

### Validation Experiments

4. **MLP ablation at L19**
   - Ablate L19 MLP output
   - Measure effect on R_V and logits
   - Should reduce recursive mode significantly

5. **Path patching from L0 → L19 → L27**
   - Trace signal propagation
   - Verify MLP → MLP → Attention pathway

---

## 📝 Conclusion

**The Missing Link Found:** MLPs (not Attention Heads) drive the recursive mode. L27 Attention is the readout, not the engine. Future steering experiments must target **L19 MLP output** (or L0 MLP) to directly affect behavior, not just geometry.

**Status:** ✅ BREAKTHROUGH - Circuit location identified  
**Next:** Implement L19 MLP steering experiments

---

*Last Updated: December 19, 2024*  
*Synthesis of: THE_MISSING_LINK_FOUND.md, CIRCUIT_DISCOVERY_REPORT.md, INSIGHTS_WHY_V_PROJ_WORKS.md*

