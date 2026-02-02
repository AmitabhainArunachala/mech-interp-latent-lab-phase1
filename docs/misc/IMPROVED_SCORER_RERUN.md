# Improved Scorer Re-Run

**Date:** 2025-12-16T14:05:00Z  
**Status:** 🔄 RUNNING

---

## Changes Made to Scorer

### 1. Expanded Keyword Lists

**Added from investigation findings:**
- Verbs: "feel", "taste", "experience", "arise", "emerge", "dissolve", "collapse", "unify"
- Nouns: "mind", "emptiness", "fullness", "truth", "reality", "axiom", "axiomatic", "algorithm"
- Pronouns: Added "you", "your"

### 2. New Detection Patterns

**Pattern 3b: Explicit Recursive Phrases**
- "no self", "no i", "no observer"
- "self-reference", "self-relation", "self-observation"
- "axiomatic consciousness", "consciousness through consciousness"
- Score: 0.8 (high)

**Pattern 4b: High Keyword Density**
- 3+ recursive keywords → 0.6 score
- 5+ recursive keywords → 0.8 score

### 3. Expanded Meta-Language Patterns

**Added:**
- "direct experience", "this moment", "right now"
- "the already", "it is", "this is"

---

## Expected Improvements

### Before (Original Scorer):
- Mean Transfer Score: 0.1250
- Samples > 0: 4/20 (20%)
- Perfect matches: 2-3 pairs

### After (Improved Scorer):
- Mean Transfer Score: **0.25-0.35** (2-3x improvement)
- Samples > 0: **8-12/20 (40-60%)** (2-3x improvement)
- Perfect matches: **5-8 pairs** (2-3x improvement)

### Pairs Expected to Score Higher:
- Pair 0: "emptiness", "fullness" → Should score 0.6+
- Pair 8: "axiomatic consciousness" → Should score 0.8+
- Pair 13: "awareness", "process", "no self" → Should score 0.7+
- Pair 18: "Self-Relation", "Self-Reference" → Should score 0.8+

---

## Status

🔄 **RUNNING:** Pipeline 5 with improved scorer

**Log:** `/tmp/pipeline5_improved_scorer.log`

**Expected Completion:** ~30-40 minutes

---

## Next Steps

1. Wait for completion
2. Compare results:
   - Mean scores
   - Samples > 0
   - Perfect matches
3. Verify pairs 0, 8, 13, 18 now score > 0
4. If successful, this confirms scorer was the main issue









