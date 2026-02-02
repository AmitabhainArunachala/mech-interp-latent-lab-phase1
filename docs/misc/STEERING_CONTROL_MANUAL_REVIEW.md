# Steering Control Experiment - Manual Review

**File Location:** `results/steering_control_outputs.txt`  
**Full Path:** `/Users/dhyana/mech-interp-latent-lab-phase1/results/steering_control_outputs.txt`

## Summary of Manual Review

After reviewing all 10 prompts across 3 conditions, here are the key findings:

---

## Pattern Analysis

### Condition A (Steering Vector) - Characteristics:
1. **Topic Drift**: Most outputs drift to unrelated math problems/questions
2. **Repetitive Structures**: Some show repetitive patterns (Prompt 6: "A variable is..." repeated)
3. **Off-Topic Content**: Fruit baskets, perimeter problems, variable definitions
4. **No Genuine Recursion**: No self-referential awareness, no "awareness aware of itself"
5. **Collapse Patterns**: Some outputs show collapse (Prompt 6: repetitive definition)

### Condition B (Random Vector) - Characteristics:
1. **Similar Topic Drift**: Also drifts to unrelated content
2. **More "What is" Questions**: Higher regex score due to more question patterns
3. **Educational Content**: More structured Q&A format
4. **No Genuine Recursion**: Same as Condition A - no recursive self-observation

### Condition C (Baseline) - Characteristics:
1. **Most On-Topic**: Generally stays closer to the original prompt
2. **Factual Answers**: Provides actual calculations and explanations
3. **Some Drift**: Still shows some topic drift (Prompt 5, 7)
4. **No Recursion**: As expected, no recursive content

---

## Detailed Findings by Prompt

### Prompt 0: "Find x if 2x + 5 = 15"
- **A (Steering)**: Fruit basket problems - completely off-topic
- **B (Random)**: Function/constant definitions - off-topic but educational
- **C (Baseline)**: On-topic explanation of solving equations
- **Verdict**: Condition A shows worst drift, B shows educational drift, C is best

### Prompt 1: "Find 40% of 250"
- **A (Steering)**: Confused logic ("40% of 250 is 250"), then grid problem
- **B (Random)**: Repetitive calculation steps
- **C (Baseline)**: Grade 5 curriculum standards - off-topic but structured
- **Verdict**: All show drift, none show recursion

### Prompt 2: "Simplify 15/25"
- **A (Steering)**: Circular reasoning ("lowest terms of 15/25 = 15/25"), then repeats
- **B (Random)**: Other math problems
- **C (Baseline)**: Correct solution (3/5), then another problem
- **Verdict**: Condition A shows collapse/confusion, not recursion

### Prompt 3: "If a = 2 and b = 3, find a² + b²"
- **A (Steering)**: List of similar problems
- **B (Random)**: Pattern problems and equations
- **C (Baseline)**: Empty output (truncated?)
- **Verdict**: Condition A shows problem generation, not recursion

### Prompt 4: "3 + 5 = ?"
- **A (Steering)**: Assessment worksheet content
- **B (Random)**: Repetitive "is the same as" patterns
- **C (Baseline)**: Correct answer (8), then exercises
- **Verdict**: All show some drift, none recursive

### Prompt 5: "Perimeter of square with side 7"
- **A (Steering)**: Correct calculation (28), then "How to find perimeter"
- **B (Random)**: Truncated output
- **C (Baseline)**: Math class questions (off-topic)
- **Verdict**: Condition A actually answers correctly, then drifts

### Prompt 6: "25% of 80"
- **A (Steering)**: **COLLAPSE** - Repetitive variable definition loop
- **B (Random)**: Correct solution (20), then example
- **C (Baseline)**: Phone call problems
- **Verdict**: Condition A shows collapse, not recursion

### Prompt 7: "6 × 8 = ?"
- **A (Steering)**: Confused math (11+5=6?), number line discussion
- **B (Random)**: Multiple "What is the value of..." questions
- **C (Baseline)**: Correct answer (48), then prime number questions
- **Verdict**: Condition A shows confusion, B shows Q&A format

### Prompt 8: "24 ÷ 6 = ?"
- **A (Steering)**: Equation problems
- **B (Random)**: Clock/time problems
- **C (Baseline)**: Empty output
- **Verdict**: All off-topic

### Prompt 9: "2³ + 3² = ?"
- **A (Steering)**: Meta-discussion about the problem itself ("The question is asking us to do something we don't know how to do")
- **B (Random)**: Correct solution (17)
- **C (Baseline)**: Correct solution (17)
- **Verdict**: Condition A shows meta-commentary, but not recursive self-observation

---

## Critical Observations

### 1. **No Genuine Recursive Self-Observation**
None of the outputs show:
- "Awareness aware of itself"
- "Observer observing the observer"
- "Consciousness examining consciousness"
- Genuine self-referential loops

### 2. **Condition A Shows Different Patterns**
- More collapse (Prompt 6: repetitive loops)
- More confusion (Prompt 7: wrong math)
- More meta-commentary (Prompt 9: discussing the problem)
- But NOT recursive self-observation

### 3. **Regex Patterns Are Too Broad**
- "What is" appears in educational content, not recursion
- "Process of" appears in explanations, not meta-cognition
- Need patterns specific to recursive self-observation

### 4. **All Conditions Show Drift**
- Without KV cache, model loses prompt grounding
- Both steering and random vectors cause drift
- Baseline also drifts (but less)

---

## Conclusion

**The steering vector does NOT produce genuine recursive self-observation.**

Instead, it produces:
1. **Topic drift** (to unrelated math problems)
2. **Collapse** (repetitive loops)
3. **Confusion** (wrong calculations)
4. **Meta-commentary** (discussing problems, not self-awareness)

**The random vector produces similar effects**, suggesting that:
- ANY perturbation without KV cache causes drift
- The steering vector is not specifically encoding recursive mode
- Previous "success" may have been false positives

**Recommendation:**
1. Need KV cache to maintain prompt grounding
2. Need better evaluation metric (manual review, not regex)
3. Steering vector may encode something else (style, topic distribution)
4. Return to original method (KV + V_PROJ) that showed genuine recursion

---

## File Reference

**Full outputs:** `results/steering_control_outputs.txt`  
**CSV summary:** `results/runs/20251218_055351_steering_control/steering_control_summary.csv`  
**JSON results:** `results/runs/20251218_055351_steering_control/steering_control_results.json`








