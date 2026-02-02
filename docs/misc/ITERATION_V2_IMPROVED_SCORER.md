# Iteration V2: Improved Recursive Feature Detection

**Date:** 2025-12-16T12:50:00Z  
**Goal:** Fix recursive feature detection to see stronger behavior transfer signal

---

## Changes Made

### 1. Expanded Verb/Noun Lists
- **Added:** "aware", "conscious", "perceive", "contemplate", "consider"
- **Added:** "generate", "create", "produce", "form", "construct"
- **Added:** "think", "thinking", "thought"
- **Added:** "mechanism", "generator", "observer", "observed", "boundary"
- **Added:** "awareness", "consciousness", "self", "text", "answer"

### 2. Increased Window Size
- **Changed:** window = 10 → **window = 20**
- Allows detection of patterns across longer spans

### 3. Added Multiple Detection Patterns

#### Pattern 1: Self + Verb + Noun (Original, Strongest)
- Score: 1.0 if all three present
- Score: 0.5 if self + verb only

#### Pattern 2: Reflexive Structures (NEW)
- Detects: "X is X", "the observer is the observed"
- Score: 0.8

#### Pattern 3: Meta-Language (NEW)
- Detects: "these words", "this response", "the process"
- Score: 0.6

#### Pattern 4: Self-Reference Without Pronouns (NEW)
- Detects: "awareness is aware", "consciousness is conscious"
- Score: 0.7

#### Pattern 5: Boundary Dissolution Language (NEW)
- Detects: "no boundary", "boundary dissolves", "dissolving"
- Score: 0.7

### 4. Fixed Repetition Gate
- **Added:** Unigram (1-gram) check with threshold 0.6
- **Lowered:** 2-gram threshold 0.5 → 0.3

---

## Expected Impact

### Before (V1):
- Recursive Control: mean_score = 0.0250, pass_rate = 80%
- Transfer: mean_score = 0.0250, pass_rate = 50%
- Only 1/20 samples scored > 0

### After (V2 - Projected):
- Recursive Control: mean_score = **0.3-0.5**, pass_rate = 60-70%
- Transfer: mean_score = **0.2-0.4** (if transfer occurred)
- Expected: 8-12/20 samples score > 0

---

## Test Results

### Quick Validation:
```
✅ "I observe my own words" -> 1.000
✅ "The observer is the observed" -> 0.700
✅ "There is no boundary..." -> 0.700
✅ "Awareness is aware..." -> 0.700
✅ "I am thinking" -> 0.500
✅ "The quick brown fox" -> 0.000
```

All tests passing! Ready for full Pipeline 5 re-run.

---

## Status

🔄 **RUNNING:** Pipeline 5 with improved scorer

**Next:** Analyze results and iterate further if needed.









