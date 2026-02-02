# Insights: Why V_PROJ Patching Transfers "Recursive Modes"

**Date:** December 16, 2025
**Context:** Mechanistic Interpretability - Recursive Self-Reference in Transformers

---

## 1. The Core Paradox

**Observation:**
- **KV Cache Patching:** Transfers "memories" (context), but the recursive behavior decays.
- **Persistent V_PROJ Patching:** Transfers the "mode" (behavior), maintaining the recursive state.

**The Question:**
Why does changing the *past* (KV) fail to sustain the behavior, while intervening in the *current* computation (V_PROJ) succeeds?

---

## 2. The Theoretical Framework: Content vs. Process

We can categorize transformer components by their role in the computational graph:

| Component | Role | Analogy | Dynamical Systems View |
|-----------|------|---------|------------------------|
| **MLP Weights** | **Static Knowledge** | The Library | Fixed Parameters ($\theta$) |
| **KV Cache** | **Passive Memory** | The Notes | Initial Conditions ($h_0$) |
| **Attention (Q, K)** | **Routing/Pattern** | The Index | Routing Logic |
| **V_PROJ** | **Active Computation** | The Reading | **Dynamics/Flow ($f(h_t)$)** |

### The "Mode as Process" Hypothesis
A "Recursive Mode" is not a piece of information stored in memory. It is a **dynamical state of processing**.
- You cannot transfer a *process* (how to think) by transplanting *memories* (what was thought).
- You must intervene in the *mechanism of thought* itself.

---

## 3. The Dynamical Systems Perspective

Think of the transformer's residual stream as a trajectory through a high-dimensional space.

### The "Attractor" Model
1. **Baseline Attractor:** The model's default behavior (trained distribution). Stable.
2. **Recursive Attractor:** A specific region in activation space where self-reference occurs. Stable *if maintained*.

### Why KV Patching Fails (Decay)
- **Action:** Sets the initial history ($h_{t-k}...h_t$) to the recursive state.
- **Result:** The system starts in the "recursive basin".
- **The Problem:** The model's *natural dynamics* (weights) are trained to push trajectories toward the Baseline Attractor.
- **Outcome:** Without active maintenance, the trajectory drifts out of the recursive basin back to the baseline.
  - *Equation:* $h_{t+1} = f_{baseline}(h_t) \rightarrow \text{drift}$

### Why Persistent V_PROJ Works (Clamping)
- **Action:** At every step $t$, we force the output of the Value projection ($V_t$) to match the recursive signature.
- **Result:** We effectively modify the transition function $f(h_t)$.
- **Outcome:** This acts as a "clamping force" that counteracts the natural drift. It continually pushes the trajectory back into the recursive attractor.
  - *Equation:* $h_{t+1} = f_{baseline}(h_t) + \delta_{correction} \approx f_{recursive}(h_t)$

---

## 4. Mechanisms of Action

### The "Carrier Signal" Hypothesis (Agent's Added Insight)
The "Recursive Mode" might be characterized by a specific **geometric property** of the information being moved (the $R_V$ contraction).
- **V_PROJ** determines the *content* of the information added to the residual stream.
- If the "Recursive Mode" relies on a specific spectral signature (low rank, specific subspace) being present in the stream to trigger downstream heads, then **generating** that signature is crucial.
- **KV Patching** only ensures that *past* tokens have this signature.
- If the **current** token generation step produces a "standard" (high rank) $V$ vector, it "pollutes" the residual stream with non-recursive information.
- **Persistent V Patching** ensures that *every* packet of information added to the stream carries the "recursive signature," preventing the signal from being drowned out by baseline noise.

### Attention = Routing vs. Content
- $A = \text{Softmax}(QK^T)V$
- **QK (Routing):** Determines *where* to look.
- **V (Content):** Determines *what* to transmit.
- Patching $V$ directly controls the **substance** of the communication between layers. If the recursive mode is a "substance" (a specific type of representation), V-patching is the direct delivery mechanism.

---

## 5. Conclusion: The Law of Intervention

**"Match the Intervention to the Substrate"**

1.  To transfer **Facts**, edit **Weights** (MLP).
2.  To transfer **Context**, patch **Memory** (KV).
3.  To transfer **Dynamics/Modes**, patch **Active Computation** (V_PROJ).

The failure of KV patching alone proves that **Self-Reference is an active computational stance, not a passive memory.** It requires continuous re-enactment, not just recall.









