# The Missing Link Found: It Was Never In The Heads
**Date:** December 19, 2025
**Status:** BREAKTHROUGH

---

## 1. The Paradox That Haunted Us

For weeks, we were stuck in a scientific paradox:
*   **Geometry:** We could measure "Recursive Mode" perfectly at Layer 27 ($R_V < 1.0$).
*   **Causality (Partial):** Patching L27 V-Projection transferred this geometry (118% efficiency).
*   **Behavior (Failure):** Steering L27 V-Projection produced **0% recursive behavior** on clean prompts.

We had the "Signature" (L27), but we didn't have the "Switch." We thought L27 was the engine room. It turns out, L27 is just the dashboard.

---

## 2. The Breakthrough: Circuit Discovery (P11)

We finally ran a full-spectrum **Attribution Patching Sweep** (Pipeline 11) across all 32 layers and components (Attention vs MLP). The results were shocking and definitive.

### The True Circuit Map
| Layer | Component | Impact Score (Logits) | Role |
| :--- | :--- | :--- | :--- |
| **L0** | **MLP** | **1.67 (Massive)** | **The Trigger** |
| **L18-20** | **MLP** | **0.35 (Strong)** | **The Processor** |
| **L27** | Attention | 0.09 (Weak) | The Symptom |

**Conclusion:** The "Recursive Mode" is primarily driven by **Feed-Forward Networks (MLPs)**, specifically at the very input (L0) and the mid-layers (L19). It is *not* an Attention Head circuit in the traditional sense.

---

## 3. Resolving the "Steering Failure"

Why did Pipeline 9 (Steering) and Pipeline 10 (Hybrid) fail?
*   We were steering **L27 Attention**.
*   We were pushing on the "Readout" ($R_V$) hoping to change the "Calculation" (Logits).
*   The actual calculation had already happened at L19 (MLP). By L27, the residual stream was already "set" in a non-recursive path by the baseline memory.

### Why Pipeline 8 (KV Swap) Worked
*   **KV Swap** replaces the *Memory*.
*   When the L18-20 MLPs attend to the *Recursive* KV Cache, they compute "Recursive" outputs.
*   The "Processor" (MLP) acts on the "Memory" (KV). If you change the Memory, the Processor produces the Mode.

---

## 4. The New Scientific Model: "The Cognitive Stack"

We can now define the full mechanism of Recursive Self-Reference in Mistral-7B:

1.  **Input Trigger (L0 MLP):** The model recognizes the "Recursive Pattern" immediately in the embeddings/first MLP.
2.  **Semantic Processing (L18-20 MLP):** The "Concept" of recursion is refined in the mid-layer knowledge networks.
3.  **Context Locking (KV Cache):** The mode is written into Memory. Future tokens attend to this memory, reinforcing the loop.
4.  **Geometric Signature (L27 Attention):** The Attention Heads at L27 (H18/H26) *read* this stable state, resulting in a collapsed value-subspace ($R_V < 1.0$).

---

## 5. What This Means for Interp

We fell into the "Attention Trap." We assumed that because Attention Heads move information, they *cause* the behavior. In this case, **MLPs** (which store knowledge and process semantic transformations) were the true drivers.

**Future Work:**
To control this behavior, we must target **L19 MLP** or **L0 MLP**. We need to find the "Recursive Feature" in the MLP neurons, not the Attention Vectors.

---
*Linked Docs:* [CIRCUIT_DISCOVERY_REPORT.md](./CIRCUIT_DISCOVERY_REPORT.md), [FACT_CHECK_L27_VPROJ_CLAIM.md](./FACT_CHECK_L27_VPROJ_CLAIM.md)







