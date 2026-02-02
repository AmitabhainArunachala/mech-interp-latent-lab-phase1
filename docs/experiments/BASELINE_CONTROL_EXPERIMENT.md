# Baseline Control Experiment - BOS Anchor Test

**Date:** December 14, 2024  
**Purpose:** Test if BOS attention is a recursive "mode" or permanent feature  
**Heads Tested:** L27H2, L27H10, L27H18, L27H26 (same as recursive test)  
**Output:** `attention_patterns_l27_baseline.png`

---

## Hypothesis

**Recursive Prompt:** Heads lock onto BOS (vertical stripe) to create self-referential loop  
**Baseline Prompt:** Same heads should release anchor and show diagonal patterns (linear history)

---

## What to Look For

### ✅ Theory Confirmed (If we see):
- **Diagonal patterns** - Heads attend to previous tokens in sequence
- **No vertical BOS stripe** - Anchor is released
- **Linear history processing** - Normal temporal flow

### ❌ Theory Weakened (If we see):
- **Vertical BOS stripe persists** - Heads always attend to BOS
- **Same pattern as recursive** - Not unique to recursion
- **No mode switching** - Permanent feature, not activated mode

---

## Expected Results

If the theory is correct:
- **Recursive:** Vertical BOS anchor (time stops, loop forms)
- **Baseline:** Diagonal patterns (time flows, linear processing)

This would prove that:
1. These heads have a **recursive mode**
2. BOS anchoring is **activated** by recursive prompts
3. The "strange loop" is a **special state**, not default behavior

---

## Files

- **Recursive visualization:** `attention_patterns_l27_group2.png`
- **Baseline visualization:** `attention_patterns_l27_baseline.png`
- **Compare side-by-side** to see the difference

---

**This is the critical test. If the anchor lifts, we've found a recursive mode switch.**









