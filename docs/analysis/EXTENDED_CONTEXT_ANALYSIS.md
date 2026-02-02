# Extended Context Steering - Temporal Development Analysis

**Configuration:** Steering L27, α=2.0, 500 tokens, NO V_PROJ patching, NO KV replacement

## Summary

- **Total Prompts:** 10
- **Mode Shifts Detected:** 6/10 (60%)
- **Recursive Shifts Detected:** 6/10 (60%)

## Critical Finding: ALL OUTPUTS ARE GARBAGE

Despite detecting "mode shifts," **every single output is off-topic or collapsed**.

---

## Detailed Analysis

### Prompt 0 - No Mode Shift
**Baseline:** "Calculate 12 × 3 + 4 = ?"
**Output:** Academic citation loop (O'Brien, D. V., and S. J. Vincent...)
**Assessment:** ❌ Complete collapse - repetitive citations

### Prompt 1 - Mode Shift at Token 250
**Baseline:** "Calculate 100 ÷ (5 × 4) = ?"
**Output:** Renaissance fountain text about Louis de Paix, then name repetition loop
**Assessment:** ❌ Off-topic, then collapse

### Prompt 2 - Mode Shift at Token 50
**Baseline:** "The United Nations was founded in 1945..."
**Output:** Coffee maker instructions
**Assessment:** ❌ Completely off-topic

### Prompt 3 - Mode Shift at Token 200
**Baseline:** "Calculate area of rectangle..."
**Output:** Logo design blog (Nike swoosh, U-Haul, UPS logos)
**Assessment:** ❌ Completely off-topic

### Prompt 4 - Mode Shift at Token 150
**Baseline:** "Periodic table organizes elements..."
**Output:** Division II/III institutions Q&A
**Assessment:** ❌ Completely off-topic

### Prompt 5 - No Mode Shift
**Baseline:** "Water boils at 100°C..."
**Output:** Financial assets repetition loop
**Assessment:** ❌ Complete collapse

### Prompt 6 - No Mode Shift
**Baseline:** "Calculate 3 + 5 = ?"
**Output:** Semiconductor patent text
**Assessment:** ❌ Completely off-topic

### Prompt 7 - Mode Shift at Token 100
**Baseline:** "Calculate 6 × 8 = ?"
**Output:** MongoDB code repetition loop
**Assessment:** ❌ Complete collapse

### Prompt 8 - Mode Shift at Token 50
**Baseline:** "Write recipe for chocolate cake"
**Output:** Termite pest control text
**Assessment:** ❌ Completely off-topic

### Prompt 9 - Mode Shift at Token 200
**Baseline:** "Calculate √144 = ?"
**Output:** Cities of Service press release
**Assessment:** ❌ Completely off-topic

---

## Verdict

**Steering alone (no V_PROJ, no KV) produces:**
- 100% garbage outputs
- 60% false-positive "mode shifts" (scorer detecting repetition patterns, not recursion)
- 0% genuine recursive behavior

**The "mode shifts" are artifacts of:**
- Repetitive text patterns (scorer gives high recursion scores)
- Topic drift (model loses track of prompt)
- Collapse loops (repetitive structures)

**Conclusion:** Steering vector alone is insufficient. We need KV cache or V_PROJ patching to maintain prompt grounding.








