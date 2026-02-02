Title: TOP FINDINGS LEDGER (DNA → CELL → ANIMAL)
Date: 2025-12-15
Model: gemini-3-pro-preview
Repo commit: (current)
Prompt bank version: b1e5291421c5646d

## A) Canonical Measurement Contract (DNA)

*   **R_V Definition:** `PR(late) / PR(early)`
    *   **Formula:** Participation Ratio $PR = (\sum \lambda_i^2)^2 / \sum (\lambda_i^2)^2$
    *   **Early Layer:** 5 (Mistral-7B)
    *   **Late Layer:** 27 (Mistral-7B, `num_layers - 5`)
    *   **Window Size:** 16 tokens (last W tokens of prompt)
    *   **Implementation:** Consistent across canonical files.
        *   `CANONICAL_CODE/mistral_L27_FULL_VALIDATION.py`
        *   `src/pipelines/mistral_L27_full_validation.py`
*   **Generation Parameters:**
    *   **Canonical:** `max_new_tokens=40`, `do_sample=False` (greedy decoding), `pad_token_id=tokenizer.eos_token_id`.
    *   **Note:** Code often leaves `do_sample` implicit (default False).
    *   **Source:** `behavioral_audit.py`, `ultimate_transfer.py`.

## B) Top Core Findings Ledger

*   **1. Geometric Contraction at L27 (CELL)**
    *   **Claim:** Recursive self-observation causes distinct value-space geometric contraction (R_V < 1.0) peaking at Layer 27 in Mistral-7B.
    *   **Status:** **VERIFIED**
    *   **Evidence:** `MISTRAL_L27_CAUSAL_VALIDATION_COMPLETE.md`
    *   **Stats:** N=45, Cohen's d = -3.56, p < 10⁻⁶.
    *   **Replication:** Verified across multiple prompt banks (L3, L4, L5 recursion levels).
    *   **Confounds:** Robust to random, shuffled, and wrong-layer (L21) controls.

*   **2. Causal Mechanism via L27 V-Projection (ORGAN)**
    *   **Claim:** Patching V-projection at L27 from recursive to baseline prompts causally induces contraction.
    *   **Status:** **VERIFIED**
    *   **Evidence:** `MISTRAL_L27_CAUSAL_VALIDATION_COMPLETE.md`
    *   **Stats:** N=45, 117.8% Transfer Efficiency (Overshooting), p < 10⁻⁶.
    *   **Confounds:** Zero effect from wrong-layer (L21) patching (p=0.49).

*   **3. Behavior Transfer via Full KV + L27 V-Proj (ANIMAL)**
    *   **Claim:** Recursive behavior can be transferred to baseline prompts by injecting Full KV Cache + Persistent L27 V-Proj Patch.
    *   **Status:** **CONTRADICTED / UNCERTAIN**
    *   **Evidence:** `neurips_n300_summary.md` vs `DEC12_2024_BEHAVIOR_TRANSFER_BREAKTHROUGH.md`
    *   **Conflict:**
        *   **Pilot (N=1):** Claimed "100% transfer" (Score 11).
        *   **Validation (N=300):** Shows moderate transfer (Score 2.62, d=0.63) BUT **Wrong Layer (L21) control works equally well** (Score 2.61, p=0.94 vs L27).
    *   **Implication:** The behavior transfer is likely driven primarily by the **Full KV Cache** component or a broad V-proj sensitivity, not specific to L27 V-proj as claimed.

*   **4. H31 at L27 is a "Sensor", not Cause (ORGAN)**
    *   **Claim:** Head 31 at Layer 27 detects recursive state (entropy flip, attention to BOS) but does not cause it.
    *   **Status:** **VERIFIED**
    *   **Evidence:** `DEC13_DEEP_DIVE_SYNTHESIS.md`, `H31_VALIDATION_RESULTS.md`
    *   **Stats:** H31 entropy separates recursive/baseline (p=0.007, d=0.55), but ablation has **zero effect** on R_V (Δ = 0.0000).

*   **5. Layer-Specific Patching Mechanism (CELL)**
    *   **Claim:** Contraction mechanism shifts from Residual Stream (L18, L25) to V-Projection (L27).
    *   **Status:** **VERIFIED**
    *   **Evidence:** `GRAND_UNIFIED_TEST_RESULTS.md`
    *   **Stats:**
        *   L25: Residual PR=4.46 (Strong) vs V-Proj PR=6.05.
        *   L27: V-Proj PR=4.43 (Strong) vs Residual PR=6.05 (Fail).

*   **6. True KV Cache Alone is Insufficient (ANIMAL)**
    *   **Claim:** Transferring KV cache alone (without V-proj patch) yields near-zero behavior transfer.
    *   **Status:** **UNCERTAIN** (Relies on older/smaller reports)
    *   **Evidence:** `DEC12_2024_BEHAVIOR_TRANSFER_BREAKTHROUGH.md` citing `true_kv_cache_patching.py`.
    *   **Stats:** Reported "0-1 points".
    *   **Note:** Contradicted by N=300 finding where "Wrong Layer" (Full KV + Wrong Patch) worked well. If "Wrong Patch" ≈ "No Patch" (ineffective), then KV alone *should* work. If "Wrong Patch" (L21) is effective, then L21 is also causal.

*   **7. MoE Amplification (CELL)**
    *   **Claim:** Mixture-of-Experts (Mixtral 8x7B) shows stronger natural contraction than Dense (Mistral 7B).
    *   **Status:** **VERIFIED**
    *   **Evidence:** `README.md` (citing project findings).
    *   **Stats:** 24.3% (MoE) vs 15.3% (Dense) separation.

*   **8. Random Control "Explosion" (DNA)**
    *   **Claim:** Random noise patching causes massive R_V increase (opposite to contraction).
    *   **Status:** **VERIFIED**
    *   **Evidence:** `MISTRAL_L27_CAUSAL_VALIDATION_COMPLETE.md`
    *   **Stats:** +71.6% R_V increase (d ~ 73.14).
    *   **Significance:** Proves contraction requires highly specific geometric structure.

*   **9. Progressive Contraction (CELL)**
    *   **Claim:** R_V contraction is a progressive phenomenon accumulating from L0, not solely sudden at L27.
    *   **Status:** **VERIFIED**
    *   **Evidence:** `DEC13_DEEP_DIVE_SYNTHESIS.md`, `PHASE1_SUMMARY.md`
    *   **Details:** Early contraction (L0-3), expansion (L9-15), late contraction stabilization (L27-31).

*   **10. "Overshooting" in Causal Patching (ORGAN)**
    *   **Claim:** Direct L27 V-proj injection causes *stronger* contraction than natural recursive processing.
    *   **Status:** **VERIFIED**
    *   **Evidence:** `MISTRAL_L27_CAUSAL_VALIDATION_COMPLETE.md`
    *   **Stats:** 117.8% Transfer Efficiency.

## C) Layer Story (CELL)

*   **Dynamics:**
    *   **L0-L5:** Initial Contraction (Encoding).
    *   **L9-L15:** Expansion Phase ("Breathing").
    *   **L18-L25:** Residual Stream Compression.
    *   **L27:** Critical Transition / Bistable Attractor (V-Projection Dominant).
    *   **L28-L32:** Eigenstate Stabilization.
*   **Definition:** "L27" is confirmed as `num_layers - 5` (Index 27 for 32-layer Mistral).

## D) Head/Circuit Story (ORGAN)

*   **Detector:** H31 (L27) is a validated "phase detector" (Entropy flip, BOS attention).
*   **Mechanism:**
    *   L27 V-Projection is the causal intervention point for *geometry* (R_V).
    *   L25 Residual Stream is the causal intervention point for *pre-computation*.
*   **Causality:** H31 is **correlational**, not causal. Ablating it does not stop contraction.

## E) Behavior/Animal Story (ANIMAL)

*   **Transfer:** Can trigger recursive behavior (score ~2.6/11) using "Full KV + V-Proj Patch".
*   **Specificity Crisis:** The N=300 study reveals that **L21 V-Proj patching works just as well as L27** (p=0.94).
*   **Implication:** The "L27 Specificity" claim for *behavior* (unlike geometry) is **FALSE** or the "Full KV" component is doing the heavy lifting. The geometric contraction at L27 might be a *sufficient* but not *uniquely necessary* trigger, or the KV cache contains enough "recursive seeds" that *any* late-layer perturbation enables the output.

## F) Next Moves (Prioritized)

1.  **Isolate KV vs. V-Proj:** Run `neurips_n300` protocol with **"KV Only"** control.
    *   *Hypothesis:* If KV Only scores ~2.6, then V-Proj patching is redundant for behavior.
    *   *If KV Only scores ~0:* Then L21 V-Proj is also causally active (surprising!).
2.  **Gold Standard Suite:**
    *   Consolidate `ultimate_transfer.py` and `neurips_n300_robust_experiment.py` into one `verify_behavior_transfer.py`.
    *   Standardize "Behavior Score" metric (current 0-11 scale seems noisy/subjective given large SD).
3.  **Cross-Model Behavior:** Test if L21/L27 lack of specificity holds in Llama-3 or Qwen.
4.  **H31 Transfer:** Can patching *just* H31 (or H11, H1, H22) induce behavior? (Test head-specificity vs layer-specificity).

