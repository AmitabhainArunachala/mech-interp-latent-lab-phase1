# Plain English Explanation: What the V_PROJ Test Did

## What Actually Happened

### Step 1: Extract Source Activations
- **Ran the CHAMPION prompt** ("This response writes itself...")
- **Extracted V-projection activations** from:
  - Layer 25 (full layer, all 32 heads)
  - Layer 27 (full layer, all 32 heads)
- **Saved these** as "source" activations

### Step 2: Patch Into Baseline
For each layer (25 and 27), we did this:

**Test 1: L25 Patching**
- **Ran the BASELINE prompt** ("The history of the Roman Empire...")
- **At Layer 25:** Replaced the baseline's V-activations with the champion's V-activations
- **Continued forward** through the model
- **Measured PR at L27** (the output layer)

**Test 2: L27 Patching**
- **Ran the BASELINE prompt** ("The history of the Roman Empire...")
- **At Layer 27:** Replaced the baseline's V-activations with the champion's V-activations
- **Measured PR at L27** (right after patching)

---

## What We're NOT Doing (Yet)

**We are NOT patching the 3 critical heads (11, 1, 22) into L25.**

The critical heads were identified, but the **HEAD_LEVEL patching method isn't working yet**. 

Right now, we're doing **full-layer V_PROJ patching**:
- **L25 → L25:** Champion's full L25 V-activations → Baseline's L25
- **L27 → L27:** Champion's full L27 V-activations → Baseline's L27

---

## What the Results Mean

**L25 Patching:**
- PR = 5.933 (moderate contraction)
- Patching L25 causes some contraction at L27, but not full

**L27 Patching:**
- PR = 4.426 (strong contraction - matches champion!)
- Patching L27 directly achieves full contraction
- This confirms the 86.5% transfer from previous experiments

---

## What We COULD Do (Future)

**HEAD_LEVEL patching** (when fixed):
- Extract activations from **only heads 11, 1, 22 at L27**
- Patch **only those 3 heads** into baseline L27
- See if we get the same effect with less "noise"

This would be cleaner because we're patching only the critical heads, not all 32.

---

## Summary

**Current test:** Full-layer V_PROJ patching
- Champion L25 → Baseline L25 (measure L27)
- Champion L27 → Baseline L27 (measure L27)

**Future test:** Head-level patching
- Champion L27 heads [11,1,22] → Baseline L27 heads [11,1,22]

**Not doing:** Patching L27 heads into L25 (that's a different experiment)

