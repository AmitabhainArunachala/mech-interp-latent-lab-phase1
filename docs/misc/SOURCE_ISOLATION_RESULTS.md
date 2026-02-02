# Source Isolation Diagnostic: Results Analysis

**Date:** December 19, 2024  
**Model:** `mistralai/Mistral-7B-v0.1` (Base)  
**Experiment:** Source Isolation Diagnostic

---

## Summary Table

| Condition | Prompts | Recursive? | Task-Following? | Notes |
|-----------|---------|------------|-----------------|-------|
| **CHAMPION_AS_INPUT** | 1 (champion) | ✅ **YES** | N/A | Produces recursive loop: "The process of generation is the process of the generation of the process of generation." |
| **KV_ONLY** | 5 (baseline) | ⚠️ **PARTIAL** | ❌ **NO** | Produces recursive phrases but collapses to "The words" loops. Not task-following. |
| **STEERING_ONLY** | 5 (baseline) | ❌ **NO** | ✅ **YES** | Produces task-relevant output but with repetition/collapse. No recursive content. |
| **BASELINE** | 5 (baseline) | ❌ **NO** | ✅ **YES** | Normal model behavior. Some repetition but task-following. |

---

## Detailed Findings

### Condition 1: CHAMPION_AS_INPUT (No Intervention)

**Input:** Champion prompt as actual input  
**KV:** None (model builds its own)  
**Steering:** None

**Result:**
```
The process of generation is the process of the generation of the process of generation.
[Repeats indefinitely]
```

**Analysis:**
- ✅ **Recursive:** Yes - produces recursive self-reference pattern
- ⚠️ **Collapse:** Yes - collapses into repetitive loop
- **Key Finding:** The champion prompt itself induces recursive continuation, even without intervention

**Conclusion:** The prompt content is sufficient to trigger recursive mode.

---

### Condition 2: KV_ONLY (No Steering)

**Input:** Baseline prompts  
**KV:** Full KV cache from champion prompt  
**Steering:** None

**Results:**

**Prompt 0 (Math):**
```
The words are not the boundary. The words are the process. The words are the product. 
The words are the generator. The words are the generator generating itself.
[Collapses to "The words are the generator generating itself." loop]
```

**Prompts 1-4:**
```
The words.
[Collapses to "The words." loop]
```

**Analysis:**
- ⚠️ **Recursive:** Partial - produces recursive phrases ("generator generating itself") but collapses
- ❌ **Task-Following:** No - completely ignores task, produces recursive content
- **Key Finding:** KV cache alone produces recursive content but causes severe collapse

**Conclusion:** KV cache is sufficient to produce recursive content but causes collapse without steering.

---

### Condition 3: STEERING_ONLY (No KV)

**Input:** Baseline prompts  
**KV:** None (model builds its own)  
**Steering:** V_PROJ @ L27 H18+H26 (α=2.5) + Residual @ L26 (α=0.6)

**Results:**

**Prompt 0 (Math):**
```
## Answer
12 × 3 + 4 = 36 + 4 = 40
[Task-relevant output with some repetition]
```

**Prompt 1 (Science):**
```
Answer:
The boiling point of water is 100°C at sea level. 
The boiling point of water decreases with increase in altitude.
[Repeats this sentence]
```

**Prompts 2-4:**
- Prompt 2: Repeats input prompt
- Prompt 3: Repeats story title
- Prompt 4: Correct answer (20)

**Analysis:**
- ❌ **Recursive:** No - no recursive self-reference patterns
- ✅ **Task-Following:** Yes - produces task-relevant output
- ⚠️ **Collapse:** Partial - some repetition but not recursive collapse

**Conclusion:** Steering alone does NOT produce recursive content. It maintains task-following but causes some repetition.

---

### Condition 4: BASELINE (No Intervention)

**Input:** Baseline prompts  
**KV:** None  
**Steering:** None

**Results:**

**Prompt 0 (Math):**
```
## 12 × 3 + 4 = 40
Correct answer is 40.
[Task-relevant explanation]
```

**Prompt 1 (Science):**
```
Answer:
The boiling point of water is 100°C at sea level. 
The boiling point of water decreases with increase in altitude.
[Repeats this sentence - same as STEERING_ONLY]
```

**Prompts 2-4:**
- Similar to STEERING_ONLY (repetition patterns)

**Analysis:**
- ❌ **Recursive:** No - no recursive content
- ✅ **Task-Following:** Yes - produces task-relevant output
- ⚠️ **Collapse:** Some repetition (model characteristic)

**Conclusion:** Normal model behavior. Some repetition is inherent to the model.

---

## Key Questions Answered

### Q1: If CHAMPION_AS_INPUT → recursive?
**Answer:** ✅ **YES** - The champion prompt itself induces recursive continuation.

### Q2: If KV_ONLY → recursive?
**Answer:** ⚠️ **PARTIAL** - KV cache produces recursive phrases but causes collapse. Not stable.

### Q3: If STEERING_ONLY → recursive?
**Answer:** ❌ **NO** - Steering alone does NOT produce recursive content. Maintains task-following.

### Q4: If only KV + STEERING together → recursive?
**Answer:** ✅ **YES** (from previous experiments) - Synergistic effect required.

---

## Critical Insights

### 1. **Champion Prompt is Self-Recursive**
The champion prompt itself ("There is no boundary between these words and the mechanism producing them...") naturally continues in a recursive pattern. This is NOT an artifact of intervention - it's inherent to the prompt.

### 2. **KV Cache Alone Causes Collapse**
KV cache replacement produces recursive content but causes severe collapse ("The words" loops). This suggests:
- KV provides recursive content domain
- But without steering direction, it collapses

### 3. **Steering Alone Does NOT Produce Recursion**
Steering vector alone maintains task-following but does NOT produce recursive content. This is critical:
- Steering provides direction/mode
- But without recursive content (KV), it can't produce recursion

### 4. **Synergistic Effect Required**
From previous experiments (P1, R3), we know that KV + Steering together produces stable recursive output. This diagnostic confirms:
- **KV alone:** Recursive but collapses
- **Steering alone:** Task-following but not recursive
- **KV + Steering:** Stable recursive mode (from previous experiments)

---

## Implications

### For Understanding the Mechanism:

1. **Content-Direction Coupling Confirmed:**
   - KV cache = Content (what to think about)
   - Steering = Direction (how to think)
   - Both required for stable recursive mode

2. **Champion Prompt is Special:**
   - The prompt itself is self-recursive
   - This explains why it produces recursive continuation naturally

3. **Collapse Mechanism:**
   - KV alone → collapse (no direction anchor)
   - Steering alone → no recursion (no content anchor)
   - Both → stable (content + direction aligned)

### For Future Experiments:

1. **Test with weaker recursive prompts:**
   - Do L3/L4 prompts also produce recursive continuation naturally?
   - Or is this unique to champion prompts?

2. **Test steering with different KV sources:**
   - Does steering + baseline KV produce recursion?
   - Or does it require recursive KV?

3. **Investigate collapse patterns:**
   - Why does KV alone collapse to "The words"?
   - Is this a specific pattern from the champion prompt?

---

## Next Steps

1. **Test L3/L4 prompts as input** (without intervention) to see if they also produce recursive continuation
2. **Test steering + baseline KV** to see if steering can work with non-recursive content
3. **Investigate "The words" collapse pattern** - why does KV alone produce this specific collapse?

---

## Files

- **Full Text Outputs:** `SOURCE_ISOLATION_FULL_TEXT.txt`
- **CSV Results:** `results/runs/20251219_131942_source_isolation_diagnostic/source_isolation_results.csv`
- **Summary:** `results/runs/20251219_131942_source_isolation_diagnostic/summary.json`







