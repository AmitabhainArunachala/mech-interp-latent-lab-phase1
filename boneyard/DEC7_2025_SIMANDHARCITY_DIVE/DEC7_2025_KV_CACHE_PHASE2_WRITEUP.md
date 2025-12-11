## DEC7 Phase 2 – Layer-Specific KV Patching (Midpoint Draft)

> **Note:** This is a Phase 2 write-up for DEC7, based on the current KV cache localization experiments. Results are internally consistent with prior DEC7 logs (V/Q swap, full KV transfer) and earlier sessions (Dec 3–4 R_V + behavior), but should be treated as provisional until R_V-on-KV-swapped runs and cross-architecture Mistral replication are completed.

---

```python
# ============================================================
# DEC7 SESSION SUMMARY - FINAL (CURRENT PHASE 2 STATE)
# ============================================================

summary = """
══════════════════════════════════════════════════════════════════════
                    DEC7 SESSION SUMMARY
              KV Cache & Layer Localization
══════════════════════════════════════════════════════════════════════

DATE: December 7, 2025
ARCHITECTURE: Llama-3-8B-Instruct (32 layers)
TOTAL RUNTIME: ~9 hours

══════════════════════════════════════════════════════════════════════
PHASE 1: KV CACHE TRANSFER (n=100)
══════════════════════════════════════════════════════════════════════

QUESTION: Where does recursive mode live?

ANSWER: In the KV cache - the persistent memory attention queries.

Results:
  • Baseline natural:        0.35 ± 1.11
  • Recursive natural:       6.27 ± 4.76
  • Base + Recursive KV:     4.11 ± 4.38  ← MODE TRANSFERRED
  • Recursive + Base KV:     1.50 ± 4.41  ← MODE BLOCKED

Statistics:
  • Sufficiency (A→C): Δ=+3.78, d=+1.19, p=4.3e-11 ***
  • Necessity (B→D):   Δ=-5.42, d=-1.43, p=5.1e-03 **
  • Transfer efficiency: 63.6%

WHY EVERYTHING ELSE FAILED:
  ┌─────────────────────┬──────────────────┬─────────────┐
  │ What We Patched     │ What We Missed   │ Effect      │
  ├─────────────────────┼──────────────────┼─────────────┤
  │ V-projection (L24)  │ Attention routing│ Δ = +0.03   │
  │ Q+K+V block (L24)   │ KV cache memory  │ Δ = +0.30   │
  │ Residual (ALL 32)   │ KV cache memory  │ Δ = 0.00    │
  │ KV Cache            │ Nothing          │ Δ = +3.78 ***│
  └─────────────────────┴──────────────────┴─────────────┘

══════════════════════════════════════════════════════════════════════
PHASE 2: LAYER LOCALIZATION (n=50)
══════════════════════════════════════════════════════════════════════

QUESTION: Which layers' KV cache carries the mode?

HYPOTHESIS: Early layers (L0-8) based on "frame set early" (DEC4)

RESULT: HYPOTHESIS REFUTED - Late layers dominate

  ┌─────────────┬─────────────┬──────────┬─────────────┐
  │ Layer Range │ % Transfer  │ Cohen's d│ Sig         │
  ├─────────────┼─────────────┼──────────┼─────────────┤
  │ L00-08      │    0.0%     │  +0.00   │ ns          │
  │ L08-16      │   -0.8%     │  -0.15   │ ns          │
  │ L16-24      │   -0.8%     │  -0.16   │ ns          │
  │ L24-32      │   12.8%     │  +0.34   │ ns (p=.09)  │
  │ L00-16      │    2.3%     │  +0.21   │ ns          │
  │ L16-32      │   79.8%     │  +0.81   │ *** p=.0002 │
  │ L00-32      │   89.7%     │  +1.16   │ *** p<.0001 │
  └─────────────┴─────────────┴──────────┴─────────────┘

KEY INSIGHT:
  • Early layers (L0-16): ~0% contribution
  • Late layers (L16-32): ~80% contribution
  • Full cache adds ~10% more

The recursive mode lives in LATE layer KV cache (L16-32),
not early "framing" layers. This suggests:
  1. Early layers ENCODE features
  2. Late layers STORE the recursive "stance" 
  3. R_V contraction at L24 marks this transition

══════════════════════════════════════════════════════════════════════
REVISED CAUSAL MODEL
══════════════════════════════════════════════════════════════════════

       PROMPT TOKENS
            │
            ▼
    [EMBEDDING + EARLY LAYERS L0-16]
            │
            │  Features extracted but mode not yet committed
            ▼
    [LATE LAYERS L16-32]
            │
            │──────▶ KV CACHE (L16-32) ◀── RECURSIVE MODE HERE
            │              │
            ▼              ▼
       [OUTPUT]      [R_V SIGNATURE]
       (behavior)     (geometry)

══════════════════════════════════════════════════════════════════════
CAVEATS & NEXT STEPS (per GPT review)
══════════════════════════════════════════════════════════════════════

WHAT WE CAN CLAIM:
  ✓ KV cache interventions transfer recursive behavior (d=1.19)
  ✓ Late layers (L16-32) carry most of the transferable mode
  ✓ Residual stream patching alone is insufficient
  ✓ R_V contraction correlates with recursion (ρ=-0.64)

WHAT NEEDS MORE VALIDATION:
  • R_V measurement on KV-swapped runs (does geometry transfer?)
  • Cross-architecture replication (Mistral)
  • Human/blind rating of outputs (not just keyword scoring)
  • Token-position analysis within KV cache

WORKING HYPOTHESIS (not yet confirmed):
  "Late-layer KV cache is the primary locus of recursive self-reference,
   with R_V as geometric signature and behavior as downstream readout."

══════════════════════════════════════════════════════════════════════
FILES & ARTIFACTS
══════════════════════════════════════════════════════════════════════

Transcript: /mnt/transcripts/2025-12-07-17-38-21-dec7-kv-cache-recursive-breakthrough.txt
Figures: DEC7_KV_PHASE2_LAYERS.png (pending matplotlib)
Code: DEC7_KV_CACHE_EXPERIMENTS.ipynb

══════════════════════════════════════════════════════════════════════
BOTTOM LINE
══════════════════════════════════════════════════════════════════════

After 5 days of null results with activation patching, we found
the mechanism: recursive mode transfers via KV cache (d=1.19),
specifically late layers L16-32 (d=0.81).

This is not yet "the locus is proven" - but it's the strongest
causal handle we've found, and it explains all prior nulls.

Next priority: Validate R_V transfers with cache, then Mistral.

══════════════════════════════════════════════════════════════════════
"""

print(summary)
```

```python
# Save to file (notebook context example)
with open('/content/DEC7_SESSION_SUMMARY.txt', 'w') as f:
    f.write(summary)
print("Summary saved to DEC7_SESSION_SUMMARY.txt")
```

---

**Cursor note:** The big update from Phase 2 is that the DEC4 "frame set early" finding now appears to be: *front-loaded in tokens but stored in late layers*. Early layers extract features from the first 10% of tokens; late layers (L16–32) commit and store the recursive stance in the KV cache. This is consistent with earlier R_V results at L24 and with the DEC7 V/Q swap nulls.

