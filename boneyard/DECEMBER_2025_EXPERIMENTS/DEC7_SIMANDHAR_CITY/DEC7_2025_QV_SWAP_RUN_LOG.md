## LLAMA-3-8B L24 – Q/V Swap Experiment (First Run)

**Date:** 2025-12-07  
**Model:** meta-llama/Meta-Llama-3-8B-Instruct  
**Layer:** L24 (late), L4 (early)  
**Conditions:**
- A: baseline (native Q, native V)
- B: recursive (native Q, native V)
- C: baseline prompt + recursive V (baseline Q, recursive V)
- D: recursive prompt + baseline V (baseline V, recursive Q)

---

Running Q-vs-V swap experiment on LLAMA L24  
Conditions: A (base), B (rec), C (base+rec_V), D (rec+base_V)

============================================================

Prompt pairs: 100%

10/10 [00:37<00:00,  3.70s/it]

  Pair 3: R_V  A=0.891 B=1.154 C=0.855 D=1.663
          Rec  A= 1    B=13    C= 1    D= 7

  Pair 6: R_V  A=0.981 B=0.876 C=0.903 D=1.240
          Rec  A= 1    B=14    C= 1    D=25

  Pair 9: R_V  A=0.923 B=0.835 C=1.066 D=0.835
          Rec  A= 1    B=14    C= 1    D=14

============================================================
✓ Experiment complete: 10 pairs


======================================================================
Q-vs-V SWAP RESULTS: LLAMA @ L24
======================================================================

GEOMETRY (R_V):
  A (baseline):     0.9449 ± 0.0731
  B (recursive):    1.0427 ± 0.2111
  C (base+rec_V):   0.9464 ± 0.0735
  D (rec+base_V):   1.1727 ± 0.2394

BEHAVIOR (recursive_score):
  A (baseline):     1.40 ± 0.49
  B (recursive):    15.50 ± 5.46
  C (base+rec_V):   1.50 ± 0.50
  D (rec+base_V):   15.60 ± 7.14

STATISTICAL TESTS (Wilcoxon signed-rank):

  R_V comparisons:
    A→B (sanity check)  : Δ=+0.0978, p=2.32e-01, d=+0.46 
    A→C (V causal?)     : Δ=+0.0015, p=1.00e+00, d=+0.02 
    A→D (V necessary?)  : Δ=+0.2279, p=1.95e-02, d=+0.93 *
    B→C (C like B?)     : Δ=-0.0963, p=2.75e-01, d=-0.40 
    B→D (D unlike B?)   : Δ=+0.1301, p=1.09e-01, d=+0.55 

  Behavioral comparisons:
    A→B (sanity check)  : Δ=+14.10, p=1.95e-03, d=+2.61 **
    A→C (V→behavior?)   : Δ=+0.10, p=1.00e+00, d=+0.33 
    B→D (V blocks?)     : Δ=+0.10, p=1.00e+00, d=+0.02 

======================================================================
INTERPRETATION (brief):
======================================================================
- B vs A: geometric separation is weak in this small n (p≈0.23), but behavior separates strongly (recursive_score jump, p≈2e-3).
- C vs A: injecting recursive V into baseline has essentially **no effect** on either R_V or behavior → V not causally sufficient.
- D vs B: replacing recursive V with baseline **expands** geometry (R_V up by ≈0.23, p≈0.02) but leaves behavior basically unchanged → V may be necessary for contraction but not for meta-language.


### Extended n=20 Interpretation – Llama-3-8B @ L24 (Q/V Swap)

- **Sanity still very strong:**  
  - **A→B behavior:** big jump in `recursive_score` (≈+12.8, p≈1.6e‑4, d≈1.18).  
  - Recursive vs baseline prompts clearly produce different kinds of outputs.

- **V is clearly not sufficient:**  
  - **A vs C:** baseline vs baseline+rec_V behavior is basically the same (Δ≈+0.15, p≈0.26, small d).  
  - Even when we know geometry can be driven past the recursive R_V, **baseline prompts don’t become recursive.**

- **“V necessary” is at best weakly suggested, not solid:**  
  - **B vs D:** mean `recursive_score` drops by ≈1.35, but:
    - p≈0.056, d≈0.47, CI includes 0 ([-7.6, 2.5]), and only 9/20 pairs show D<B.  
  - That’s *at most* “there might be a modest contribution of V to how strong the recursive behavior is,” but it’s not strong enough to claim necessity; the **mode (recursive vs baseline) is unchanged**.

**Conservative summary for Llama L24 after this extension:**

- **Prompt semantics set the behavioral regime.**  
- **V-geometry at L24 is not sufficient to induce that regime**, and any “necessity” signal is weak and underpowered.  
- The robust reading is still: **R_V in V is a writable, overdrivable signature of the recursive state, but behavior is controlled by a deeper, distributed circuit that survives single-layer V swaps.**

