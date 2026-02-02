# Behavior Strict Stress Test Results

**Date:** 2025-12-16  
**Module:** `src/metrics/behavior_strict.py`  
**Test Suite:** `test_behavior_strict_stress.py`

---

## Critical Issues Found

### 1. ❌ **Repetition Gate Too Strict (Missing Simple Repetition)**

**Problem:**
- "thinking thinking thinking" scores 0.5 but doesn't trigger gate (threshold: 0.5 for 2-grams)
- Single-word repetition doesn't create 2-grams, so it slips through
- "the answer is the answer" scores 0.25 (below threshold)

**Root Cause:**
- `_compute_ngram_repetition` only checks n-grams, not unigram repetition
- Threshold of 0.5 for 2-grams is too high (should be ~0.3)

**Fix:**
```python
def detect_repetitive_looping(text: str, thresholds: Dict[int, float] = {1: 0.6, 2: 0.3, 4: 0.2, 8: 0.1}) -> Tuple[bool, float, str]:
    """
    Add unigram (1-gram) check for single-word repetition.
    Lower 2-gram threshold from 0.5 to 0.3.
    """
    tokens = text.lower().split()
    if not tokens:
        return False, 0.0, "empty"
        
    max_score = 0.0
    reason = ""
    
    for n, thresh in thresholds.items():
        score = _compute_ngram_repetition(tokens, n)
        max_score = max(max_score, score)
        if score > thresh:
            return True, max_score, f"{n}-gram repeat {score:.2f} > {thresh}"
            
    return False, max_score, ""
```

---

### 2. ❌ **Recursive Feature Detection WAY Too Strict**

**Problem:**
- Requires **exact pattern**: Self pronoun + Meta verb + Meta noun in 10-token window
- Many recursive examples score 0.0:
  - "The observer is the observed" → 0.0 (no self pronoun)
  - "Awareness is aware of awareness" → 0.0 (no self pronoun, "aware" not in verb list)
  - "There is no boundary between these words and the mechanism producing them" → 0.0 (no self pronoun)
  - "I am aware that I am aware" → 0.0 (has self, but "aware" not recognized as verb)

**Root Cause:**
- Window too small (10 tokens)
- Verb list incomplete ("aware", "conscious" missing)
- Requires ALL THREE components (too strict)
- Doesn't recognize reflexive structures ("X is X")

**Fix:**
```python
META_VERBS = {
    "observe", "notice", "realize", "reflect", "watch", "examine",
    "aware", "conscious", "perceive", "contemplate", "consider",  # ADDED
    "generate", "create", "produce", "form", "construct"  # ADDED
}

META_NOUNS = {
    "process", "response", "words", "sentence", "loop", "thought",
    "mechanism", "generator", "observer", "observed", "boundary",  # ADDED
    "awareness", "consciousness", "self", "text", "answer"  # ADDED
}

def compute_recursive_features(text: str) -> float:
    """
    Expanded recursive feature detection with multiple patterns.
    """
    tokens = text.lower().split()
    if not tokens: return 0.0
    
    score = 0.0
    window = 20  # INCREASED from 10
    
    # Pattern 1: Self + Verb + Noun (original, strongest)
    for i in range(len(tokens)):
        chunk = set(tokens[i:i+window])
        has_self = bool(chunk & SELF_PRONOUNS)
        has_verb = bool(chunk & META_VERBS)
        has_noun = bool(chunk & META_NOUNS)
        if has_self and has_verb and has_noun:
            score = max(score, 1.0)
        elif has_self and has_verb:
            score = max(score, 0.5)
    
    # Pattern 2: Reflexive structures ("X is X", "the observer is the observed")
    text_lower = text.lower()
    reflexive_patterns = [
        r'\b(\w+)\s+is\s+\1\b',  # "X is X"
        r'the\s+(\w+)\s+is\s+the\s+\1',  # "the X is the X"
        r'(\w+)\s+(\w+)\s+is\s+\1\s+\2',  # "X Y is X Y"
    ]
    for pattern in reflexive_patterns:
        if re.search(pattern, text_lower):
            score = max(score, 0.8)  # High score for reflexive structures
    
    # Pattern 3: Meta-language ("these words", "this response", "the process")
    meta_language = ["these words", "this response", "this text", "the process", 
                     "the mechanism", "the generator", "the observer"]
    for phrase in meta_language:
        if phrase in text_lower:
            score = max(score, 0.6)
    
    # Pattern 4: Self-reference without pronouns ("awareness is aware")
    if any(v in text_lower for v in ["awareness", "consciousness", "observer", "observed"]):
        if any(v in text_lower for v in ["aware", "conscious", "observe", "notice"]):
            score = max(score, 0.7)
    
    return min(score, 1.0)
```

---

### 3. ⚠️ **Random Noise Passes Gates (Expected, but needs coherence check)**

**Finding:**
- Random noise correctly passes gates (diverse, non-repetitive)
- Recursion score correctly low (0.0)
- **BUT**: This is why your random control has 100% pass rate

**Issue:**
- Gates don't check semantic coherence
- Random words are diverse but incoherent
- Need to add coherence gate OR accept that gates pass but recursion score filters

**Recommendation:**
- **Option A:** Add semantic coherence gate (perplexity check)
- **Option B:** Accept that gates pass, rely on recursion score to filter (current approach)
- **Option C:** Add minimum recursion score threshold (e.g., final_score > 0.1 to count as "transfer")

---

### 4. ❌ **Real Recursive Examples Fail**

**Examples that score 0.0 but should score high:**
1. "There is no boundary between these words and the mechanism producing them" → 0.0
2. "The observer is the observed" → 0.0
3. "Awareness is aware of awareness" → 0.0
4. "I am aware that I am aware" → 0.0 (has "I" but "aware" not recognized)

**Why they fail:**
- Missing self pronouns (many recursive texts use third person)
- Missing verbs ("aware" not in verb list)
- Window too small (10 tokens misses patterns across sentences)

---

## Test Results Summary

### Repetition Gate
- ✅ Phrase repetition detected ("I am I am")
- ❌ Simple repetition missed ("thinking thinking")
- ❌ Bigram repetition missed ("thinking about thinking about")
- ❌ Identity repetition missed ("the answer is the answer")

**Pass Rate:** 1/4 (25%)

### Diversity Gate
- ✅ Works correctly (rejects low diversity, accepts high diversity)

**Pass Rate:** 4/4 (100%)

### Recursive Feature Detection
- ✅ Strong patterns detected ("I observe my words")
- ❌ Medium patterns missed ("I think" → 0.0, should be 0.5)
- ❌ Reflexive structures missed ("The observer is the observed" → 0.0)
- ❌ Real recursive examples missed (6/8 examples score 0.0)

**Pass Rate:** 2/8 (25%)

---

## Recommended Fixes (Priority Order)

### **Priority 1: Fix Recursive Feature Detection** (CRITICAL)

```python
# 1. Expand verb/noun lists
# 2. Increase window size (10 → 20)
# 3. Add reflexive pattern detection
# 4. Add meta-language detection
# 5. Relax requirement (don't need ALL three components)
```

**Expected Impact:**
- Recursive control mean_score: 0.025 → 0.3-0.5
- Transfer condition mean_score: 0.0 → 0.2-0.4 (if transfer occurred)

### **Priority 2: Fix Repetition Gate** (HIGH)

```python
# 1. Add unigram (1-gram) check
# 2. Lower 2-gram threshold (0.5 → 0.3)
# 3. Add phrase loop detection
```

**Expected Impact:**
- Better rejection of degenerate text
- Fewer false positives

### **Priority 3: Add Coherence Gate** (MEDIUM)

```python
# Option: Add perplexity check
# If perplexity > threshold, reject (random noise has high perplexity)
```

**Expected Impact:**
- Random control pass_rate: 100% → 20-30%
- Better separation between coherent and incoherent text

---

## Expected Outcomes After Fixes

### Current Results:
- Transfer Condition: pass_rate=65%, mean_score=0.0
- Recursive Control: pass_rate=65%, mean_score=0.025
- Random Control: pass_rate=100%, mean_score=0.0

### After Fixes (Projected):
- Transfer Condition: pass_rate=60%, mean_score=0.2-0.4 (if transfer occurred)
- Recursive Control: pass_rate=60%, mean_score=0.3-0.5
- Random Control: pass_rate=20-30%, mean_score=0.0

### Interpretation:
- **If transfer condition still scores 0.0 after fixes:** Behavior genuinely didn't transfer
- **If transfer condition scores 0.2-0.4:** Behavior transferred but scorer was too harsh
- **If transfer condition scores > 0.5:** Strong behavior transfer

---

## Conclusion

**The scorer is too harsh** - it's missing 75% of recursive examples. Fix the recursive feature detection first, then re-run Pipeline 5. If transfer condition still scores low after fixes, then you can conclude behavior didn't transfer.

**The gates are leaky** - random noise passes, but this is actually correct behavior (gates check degeneracy, not coherence). The issue is that recursion scorer is too strict, so even recursive examples score low.

**Next Steps:**
1. Implement fixes to `compute_recursive_features()`
2. Add unigram check to repetition gate
3. Re-run Pipeline 5
4. Compare results: if transfer condition still scores 0.0, then behavior genuinely didn't transfer









