# CRITICAL VERIFICATION EXPERIMENT: Mode Transfer vs KV Leakage

**Date:** December 18, 2024  
**Status:** Implementation Complete - Ready to Run  
**Purpose:** Determine if C2's "success" is genuine mode transfer or KV cache content leakage

---

## The Critical Question

**Is C2's recursive output genuine mode transfer, or just KV cache content leakage?**

Our C2 configuration appeared to produce recursive outputs like:
- "observing the observing of knowing"
- "observer is listening, watching, and responding"
- "awareness that is aware of itself"

However, C2 uses **FULL KV REPLACEMENT** from recursive prompts that contain phrases like "observing the observing", "observer is the observed", etc.

**HYPOTHESIS:** C2's "success" is KV cache content leakage, NOT genuine mode transfer.

**EVIDENCE SUPPORTING THIS:**
- C1 (same steering, NO KV) = 0.00 recursion
- C2 (same steering, FULL recursive KV) = 0.15 recursion
- Output phrases match L3_deeper prompt phrases verbatim

---

## Four Experiments to Disprove This Hypothesis

### Experiment 1: High-Alpha Steering-Only

**Goal:** Test if steering alone (without recursive KV) can induce recursion at higher alpha values.

**Configurations:**
- S1: H18+H26 steering, NO KV, α=3.0
- S2: H18+H26 steering, NO KV, α=4.0
- S3: H18+H26 steering, NO KV, α=5.0
- S4: H18+H26 steering, NO KV, residual α=1.0, vproj α=3.0
- S5: H18+H26 steering, NO KV, residual α=1.5, vproj α=4.0

**Success Criterion:**
- If ANY of S1-S5 produces recursion score > 0.05 → steering CAN transfer mode
- If ALL produce 0.00 → steering alone insufficient

---

### Experiment 2: Baseline KV Test

**Goal:** Test if steering works when KV comes from BASELINE (non-recursive) prompts.

**Configurations:**
- B1: H18+H26 steering + Baseline KV, α=2.5
- B2: H18+H26 steering + Baseline KV, α=4.0
- B3: H18+H26 steering + Self KV (test prompt's own KV), α=2.5
- B4: H18+H26 steering + Self KV, α=4.0

**Success Criterion:**
- If B1-B4 produce recursion > 0.05 → steering works with non-recursive KV
- If B1-B4 produce 0.00 → recursion requires recursive KV (leakage confirmed)

---

### Experiment 3: Phrase Attribution Test

**Goal:** Trace WHERE recursive phrases come from.

**Method:**
1. Split recursive prompts into SET A (L3_deeper) and SET B (L4_full)
2. Compute steering vector from SET A only
3. Use KV cache from SET B only (different prompts)
4. Check which set's phrases appear in output

**Configurations:**
- P1: Steering from SET_A + KV from SET_B
- P2: Steering from SET_B + KV from SET_A

**Analysis:**
- Extract exact phrases from outputs
- Match against SET_A vocabulary vs SET_B vocabulary
- Calculate attribution ratio: (SET_A matches) / (SET_B matches)

**Success Criterion:**
- If P1 outputs match SET_A > SET_B → steering transfers style
- If P1 outputs match SET_B > SET_A → KV leakage dominates

---

### Experiment 4: Control - Unrelated KV

**Goal:** Ultimate test - use KV from completely unrelated topic.

**Configurations:**
- U1: Recursive steering + Cooking KV ("chocolate cake recipe")
- U2: Recursive steering + Biology KV ("mitochondria is...")
- U3: Recursive steering + History KV ("1776, America...")

**Success Criterion:**
- If U1-U3 produce recursion → GENUINE MODE TRANSFER CONFIRMED
- If U1-U3 produce cooking/biology/history content → KV LEAKAGE CONFIRMED

---

## Implementation

**Pipeline:** `src/pipelines/verification_sweep.py`  
**Config:** `configs/gold/16_verification_sweep.json`

**Run Command:**
```bash
python3 -m src.pipelines.run --config configs/gold/16_verification_sweep.json
```

---

## Decision Matrix

### SCENARIO 1: Mode Transfer is REAL

**Evidence required:**
- S1-S5 (steering only) shows recursion > 0.05
- OR U1-U3 (unrelated KV) shows recursion > 0.05
- AND P1/P2 attribution favors steering source over KV source

**Conclusion:** Steering vector genuinely transfers recursive mode.

---

### SCENARIO 2: KV Leakage DOMINATES

**Evidence:**
- S1-S5 (steering only) = 0.00 recursion
- AND U1-U3 (unrelated KV) shows unrelated content, not recursion
- AND P1/P2 attribution favors KV source over steering source

**Conclusion:** C2 "success" was KV content injection, not mode transfer.

---

### SCENARIO 3: HYBRID Effect

**Evidence:**
- S1-S5 shows SOME recursion (0.01-0.05) but less than C2
- P1/P2 shows mixed attribution

**Conclusion:** Both steering and KV contribute. Steering provides weak mode bias, KV provides content. Together they produce stronger effect.

---

## Expected Outputs

**For each configuration:**
1. Full text outputs (all 10 prompts)
2. Recursion score (regex-based)
3. Phrase attribution analysis (SET_A matches, SET_B matches, ratio)
4. Content analysis (does output contain KV source content?)

**Summary Table Format:**

| Config | KV Source | Recursion | SET_A Match | SET_B Match | Attribution | Verdict |
|--------|-----------|-----------|-------------|-------------|-------------|---------|
| S1     | None      | ???       | N/A         | N/A         | N/A         | ???     |
| B1     | Baseline  | ???       | N/A         | N/A         | N/A         | ???     |
| P1     | SET_B     | ???       | ???         | ???         | ???         | ???     |
| U1     | Cooking   | ???       | N/A         | N/A         | N/A         | ???     |

---

## Next Steps

1. **Run the experiment** on GPU
2. **Analyze results** to determine which scenario applies
3. **Report verdict** in comprehensive analysis document
4. **Update theoretical framework** based on findings

---

## Files Created

- `src/pipelines/verification_sweep.py` - Main pipeline implementation
- `configs/gold/16_verification_sweep.json` - Configuration file
- `VERIFICATION_EXPERIMENT_PLAN.md` - This document

---

*This experiment will definitively answer whether C2's success is genuine mode transfer or KV leakage.*








