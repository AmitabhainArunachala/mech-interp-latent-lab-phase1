# Original Phase 1 vs Current Experiment: Critical Comparison

**Date:** January 11, 2025  
**Purpose:** Understand why we're not seeing contraction in current experiment

---

## The Original Discovery (Phase 1)

### Setup
- **Model:** `mistralai/Mistral-7B-Instruct-v0.2` (Instruct model, not base!)
- **Recursive Prompts:** `L5_refined` group (20 prompts)
  - Example: "Recursive self-observation observing itself. The mechanism generating this is the mechanism observing generation..."
- **Baseline Prompts:** `baseline_math` + `baseline_factual` (20 prompts each)
  - Example: "Calculate the following arithmetic problem: 3 + 5 = ?"
- **Late Layer:** 28 (num_layers - 4)
- **Early Layer:** 5
- **Window:** 16 tokens

### Results
- **L5 Recursive R_V:** ~0.85
- **Baseline R_V:** ~1.00
- **Contraction:** 15.3% (R_V < 1.0 indicates contraction)

### Key Finding
**Recursive prompts showed clear contraction (R_V = 0.85) compared to baseline (R_V = 1.00)**

---

## Our Current Experiment (Cross-Architecture Validation)

### Setup
- **Model:** `mistralai/Mistral-7B-v0.1` (BASE model, not Instruct!)
- **Recursive Prompts:** `recursive_self_reference` (new group, just created)
  - Example: "What is awareness of awareness? Awareness is aware of awareness being aware."
- **Non-Recursive Prompts:** `abstract_non_recursive`, `same_vocab_different_semantics`, etc. (new groups)
- **Late Layer:** 27 (num_layers - 5)
- **Early Layer:** 5
- **Window:** 16 tokens

### Results
- **Recursive R_V:** 0.86
- **Non-Recursive R_V:** 0.82
- **Delta:** +0.04 (recursive > non-recursive = expansion, not contraction!)

### Key Finding
**No clear contraction observed - recursive prompts show R_V = 0.86, which is HIGHER than non-recursive (0.82)**

---

## Critical Differences

| Aspect | Original Phase 1 | Current Experiment | Impact |
|--------|------------------|-------------------|--------|
| **Model** | Mistral-7B-Instruct | Mistral-7B-v0.1 (base) | ⚠️ **CRITICAL** - Different training! |
| **Recursive Prompts** | L5_refined (long, complex) | recursive_self_reference (short, simple) | ⚠️ **CRITICAL** - Different structure! |
| **Baseline Prompts** | baseline_math/factual | abstract_non_recursive | ⚠️ **CRITICAL** - Not true baseline! |
| **Late Layer** | 28 (num_layers - 4) | 27 (num_layers - 5) | ⚠️ Different measurement point |
| **Comparison** | Recursive vs Baseline | Recursive vs Non-Recursive | ⚠️ Different comparison |

---

## What We Need to Do

### To Replicate Original Finding:

1. **Use Mistral-7B-Instruct** (not base model)
   - The Instruct model was trained differently and may respond differently to recursive prompts

2. **Use L5_refined prompts** (from canonical bank)
   - These are the original "champion" prompts that showed contraction
   - They're longer, more complex, and have stronger recursive structure

3. **Use baseline_math/factual prompts** (true controls)
   - These are the original baseline prompts used in Phase 1
   - They provide a proper control comparison

4. **Use Layer 28** (original measurement point)
   - Or verify which layer is correct for the current model

5. **Compare: Recursive vs Baseline** (not Recursive vs Non-Recursive)
   - The original comparison was recursive prompts vs baseline prompts
   - Our current comparison is recursive vs non-recursive families (different!)

---

## The Real Question

**Why did the original experiments find R_V = 0.85 (contraction) but we're finding R_V = 0.86 (no contraction relative to baseline)?**

Possible explanations:
1. **Model difference:** Instruct vs Base models respond differently
2. **Prompt difference:** L5_refined (long, complex) vs new prompts (short, simple)
3. **Baseline difference:** baseline_math (true control) vs abstract_non_recursive (not a true control)
4. **Layer difference:** 28 vs 27 (different measurement point)
5. **Measurement method:** Something changed in how R_V is computed

---

## Next Steps

1. **Re-run with original conditions:**
   - Model: Mistral-7B-Instruct
   - Prompts: L5_refined vs baseline_math
   - Layer: 28
   - Compare: Recursive vs Baseline

2. **Verify the effect still exists:**
   - If we get R_V = 0.85 vs 1.00, the effect is real and reproducible
   - If we get R_V = 0.86 vs 0.82, something changed

3. **Then investigate differences:**
   - What changed between original and current?
   - Is it model, prompts, layer, or measurement method?

---

## Key Insight

**The original discovery WAS real - recursive prompts DID show contraction!**

But we need to replicate the EXACT conditions to verify:
- Same model (Instruct, not base)
- Same prompts (L5_refined, not new prompts)
- Same baseline (baseline_math, not abstract_non_recursive)
- Same layer (28, not 27)
- Same comparison (recursive vs baseline, not recursive vs non-recursive)
