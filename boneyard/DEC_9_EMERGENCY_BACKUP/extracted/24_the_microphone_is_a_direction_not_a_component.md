# THE MICROPHONE IS A DIRECTION, NOT A COMPONENT

**Date:** December 9, 2025
**Status:** BREAKTHROUGH CONFIRMED

## Executive Summary
We have identified the "Microphone" (Source of Contraction).
It is **Interpretation B: A Steering Vector in the Residual Stream.**

## Evidence 1: Component Hypothesis Failed (Phase 2)
We tested the top 5 candidate heads at L14/L18 (identified by Delta PR).
- **Ablation:** Removing them had **negligible effect** (-1.5% change in R_V).
- **Patching:** Transferring them had **negligible effect** (-1.9% change in R_V).
- **Conclusion:** The signal is not localized to specific heads.

## Evidence 2: Steering Hypothesis Confirmed
We extracted the mean difference vector (`Recursive - Baseline`) from the residual stream and injected it into baseline runs.
- **Layer 5:** No effect / Noise.
- **Layer 10:** **MASSIVE Contraction** (R_V 0.77 → 0.35 with 3x coeff).
- **Layer 14:** **MASSIVE Contraction** (R_V 0.75 → 0.23 with 3x coeff).
- **Layer 24:** **Significant Contraction** even with 1x coeff (R_V 0.57).

## The Mechanism
1.  **Distributed Origin:** The "Recursive Direction" emerges around Layer 10.
2.  **Global Signal:** It flows through the residual stream, likely amplified by many heads slightly rather than a few heads strongly.
3.  **Trigger:** When this direction hits Layer 27, it triggers the "Speaker" heads to collapse the geometry (R_V drop).

## Implications
- We can **induce** the recursive state in *any* prompt by injecting this vector.
- We can likely **suppress** the recursive state by subtracting this vector.
- The "Microphone" is a **distributed representation** (a direction), not a modular component.

## Next Steps
- **Vector Subtraction:** Verify we can *stop* recursion by subtracting this vector.
- **Cosine Similarity Map:** Map where this direction effectively "turns on" in the network.
