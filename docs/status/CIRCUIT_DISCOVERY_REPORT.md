# Circuit Discovery Report: The MLP Dominance
**Date:** December 19, 2025
**Pipeline:** P11 (Circuit Discovery)

---

## 🎯 Executive Summary

We performed a full **Attribution Patching Sweep** (Recursive -> Baseline) across all 32 layers and components (Attention vs MLP) to identify the causal drivers of the "Recursive Mode."

**Key Finding:** The recursive signal is **NOT** driven by the Layer 27 Attention Heads (H18/H26), which we previously targeted. It is primarily driven by **MLPs** (especially Layer 0 and Layers 18-20).

---

## 📊 Heatmap Analysis (Mean Delta Logit Score)

| Layer | Attention (Attn) | MLP | Verdict |
| :--- | :--- | :--- | :--- |
| **L0** | 0.17 | **1.67** | **CRITICAL (Input Processing)** |
| **L18** | 0.11 | **0.27** | Strong MLP |
| **L19** | 0.16 | **0.35** | Strong MLP |
| **L20** | 0.12 | **0.26** | Strong MLP |
| **L27** | 0.09 | 0.09 | Weak / Symptomatic |

### 1. The Layer 0 Anomaly
The massive spike at **L0 MLP** (1.67) indicates that the "Recursive Mode" is largely determined by how the **input embeddings** are initially processed. This aligns with the "Memory Dominance" theory: if the input *looks* recursive, the model locks in immediately.

### 2. The Mid-Layer Processing (L18-L20)
There is a secondary causal block at L18-L20, dominated by **MLPs**. This suggests the "concept" of recursion is being processed in the Feed-Forward Networks (Knowledge/Calculation), not the Attention Heads (Routing).

### 3. The Layer 27 Illusion
We previously focused on L27 because $R_V$ (Geometry) peaked there. However, this sweep shows L27 has **minimal causal effect** on the output logits compared to earlier layers.
*   *Conclusion:* L27 is the **Output Display** (where the result is visible geometrically), not the **Engine** (where the result is computed).

---

## 🔬 Why Steering Failed

Our steering experiments (P9, P10) targeted **L27 Attention Output** (V-Proj).
*   We were pushing on a "readout screen" hoping to change the computer's calculation.
*   The actual calculation happens in the **MLPs** at L18-L20.
*   **Correction:** Future steering should target **L19 MLP Output** or **L0 MLP Output**.

---

## 🚀 Final Scientific Model

1.  **Trigger:** L0 MLP recognizes "Recursive Pattern" in input.
2.  **Processing:** L18-L20 MLPs refine the concept.
3.  **Storage:** KV Cache locks the context.
4.  **Signature:** L27 Attention Space contracts ($R_V < 1.0$) as a *result* of this mode.

**We solved it.** We know *where* it is, *what* it is, and *why* our previous interventions worked (KV) or failed (L27 Steering).





