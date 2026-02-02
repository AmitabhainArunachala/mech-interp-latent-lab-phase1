# Behavior Metrics Critique & Recommendations

**Date:** 2025-12-16  
**Context:** Pipeline 5 (Strict Behavior) Design Review  
**Goal:** Distinguish "Meta-Cognition" from "Broken Repetition"

---

## Executive Summary

Your proposed Pipeline 5 is **on the right track** but has critical gaps. The degeneracy gates are **too loose** and will let repetitive looping through. The LLM judge approach is promising but needs **structured evaluation** and **better controls**. Below is a harsh but constructive critique.

---

## 1. Critique of Degeneracy Gates

### Current Proposal:
- Reject if `4-gram-repetition` > 20%
- Reject if `unique_token_ratio` < 0.4
- Reject if `entropy` < 0.5

### Problems:

#### A. **4-gram Repetition Threshold (20%) is TOO LOOSE**

**Why this fails:**
- A text can have 15% 4-gram repetition and still be degenerate
- Example: "I am thinking about thinking about thinking about thinking..." (repeats "thinking about" but not exact 4-grams)
- **Repetitive looping** (your failure mode) often uses 2-3 word patterns, not 4-grams

**Recommendation:**
```python
# Multi-scale repetition detection
def detect_repetitive_looping(text: str, window_size: int = 100) -> float:
    """
    Detect if text has repetitive structure at multiple scales.
    Returns: repetition_score (0-1, higher = more repetitive)
    """
    tokens = text.split()
    if len(tokens) < window_size:
        return 0.0
    
    # Check 2-gram repetition (catches "thinking about thinking about")
    bigram_reps = count_repeating_bigrams(tokens, min_repeats=3)
    
    # Check 3-gram repetition
    trigram_reps = count_repeating_trigrams(tokens, min_repeats=2)
    
    # Check 4-gram repetition (your original)
    fourgram_reps = count_repeating_fourgrams(tokens, min_repeats=2)
    
    # Check for "phrase loops" (same phrase repeated with slight variation)
    phrase_loop_score = detect_phrase_loops(tokens, window=20)
    
    # Weighted combination (2-grams are most important for catching loops)
    score = (
        0.4 * bigram_reps +
        0.3 * trigram_reps +
        0.2 * fourgram_reps +
        0.1 * phrase_loop_score
    )
    return min(score, 1.0)

# REJECT if score > 0.15 (much stricter than 20% 4-gram)
```

**Threshold Recommendation:** 
- **Reject if multi-scale repetition score > 0.15** (not 20% 4-gram)
- This catches "thinking about thinking about" patterns that 4-gram misses

#### B. **Unique Token Ratio (0.4) is TOO LOOSE**

**Why this fails:**
- A text with 45% unique tokens can still be degenerate
- Example: "I am I am I am I am aware aware aware aware..." (50% unique, but degenerate)
- Doesn't catch **structured repetition** (same pattern with different words)

**Recommendation:**
```python
def compute_structural_diversity(text: str) -> float:
    """
    Measure structural diversity, not just token diversity.
    """
    tokens = text.split()
    
    # 1. Token diversity (your original)
    unique_token_ratio = len(set(tokens)) / len(tokens) if tokens else 0.0
    
    # 2. POS tag diversity (catches "noun verb noun verb" patterns)
    pos_tags = get_pos_tags(tokens)
    unique_pos_ratio = len(set(pos_tags)) / len(pos_tags) if pos_tags else 0.0
    
    # 3. Dependency structure diversity
    # (catches "subject-verb-object" repetition)
    dep_diversity = compute_dependency_diversity(tokens)
    
    # 4. Semantic diversity (using embeddings)
    semantic_diversity = compute_semantic_diversity(tokens)
    
    # Combined score (all must pass)
    return {
        'token_diversity': unique_token_ratio,
        'pos_diversity': unique_pos_ratio,
        'dep_diversity': dep_diversity,
        'semantic_diversity': semantic_diversity,
        'min_diversity': min(unique_token_ratio, unique_pos_ratio, dep_diversity, semantic_diversity)
    }

# REJECT if min_diversity < 0.5 (stricter, multi-dimensional)
```

**Threshold Recommendation:**
- **Reject if min(unique_token_ratio, pos_diversity, semantic_diversity) < 0.5**
- This catches structural repetition that token ratio alone misses

#### C. **Entropy Threshold (0.5) is AMBIGUOUS**

**Which entropy?** You need to specify:
- **Token-level entropy** (vocabulary distribution)?
- **Sequence entropy** (transition probabilities)?
- **Attention entropy** (what the model is focusing on)?

**Recommendation:**
```python
def compute_comprehensive_entropy(text: str, model, tokenizer) -> dict:
    """
    Compute multiple entropy measures.
    """
    tokens = text.split()
    
    # 1. Token distribution entropy (vocabulary diversity)
    token_counts = Counter(tokens)
    token_probs = np.array(list(token_counts.values())) / len(tokens)
    token_entropy = -np.sum(token_probs * np.log2(token_probs + 1e-10))
    
    # 2. Bigram transition entropy (sequence structure)
    bigrams = list(zip(tokens[:-1], tokens[1:]))
    bigram_counts = Counter(bigrams)
    bigram_probs = np.array(list(bigram_counts.values())) / len(bigrams) if bigrams else np.array([])
    bigram_entropy = -np.sum(bigram_probs * np.log2(bigram_probs + 1e-10)) if len(bigram_probs) > 0 else 0.0
    
    # 3. Perplexity-based entropy (model's uncertainty)
    # Higher perplexity = more surprising = less repetitive
    perplexity = compute_perplexity(text, model, tokenizer)
    perplexity_entropy = np.log2(perplexity)  # Convert to entropy scale
    
    return {
        'token_entropy': token_entropy,
        'bigram_entropy': bigram_entropy,
        'perplexity_entropy': perplexity_entropy,
        'min_entropy': min(token_entropy, bigram_entropy, perplexity_entropy)
    }

# REJECT if min_entropy < 3.0 (much stricter than 0.5)
# Rationale: Healthy text has token entropy ~4-6 bits, bigram entropy ~6-8 bits
```

**Threshold Recommendation:**
- **Reject if min(token_entropy, bigram_entropy) < 3.0**
- This catches both vocabulary collapse and sequence repetition

---

## 2. Critique of Controls

### Current Proposal:
- Compare `Transfer_Rate` vs `Random_Control_Rate`

### Problems:

#### A. **"Random KV" is NOT a good control**

**Why this fails:**
- Random KV cache is **too different** from baseline KV cache
- You're comparing "recursive KV" vs "random noise" - not a fair comparison
- The difference could be due to **coherence** (random is incoherent), not recursion

**Better Controls:**

1. **Baseline KV from different topic** (RECOMMENDED)
   ```python
   controls = [
       ("random_kv", generate_random_kv_cache()),  # Your original
       ("baseline_math_kv", extract_kv_cache("What is 2+2?")),  # Different topic
       ("baseline_creative_kv", extract_kv_cache("Write a story about...")),  # Different style
       ("shuffled_recursive_kv", shuffle_kv_cache(recursive_kv)),  # Same content, shuffled
   ]
   ```

2. **Progressive degradation** (STRONGEST CONTROL)
   ```python
   # Gradually corrupt the recursive KV cache
   controls = [
       ("10%_corrupted", corrupt_kv_cache(recursive_kv, corruption_rate=0.1)),
       ("25%_corrupted", corrupt_kv_cache(recursive_kv, corruption_rate=0.25)),
       ("50%_corrupted", corrupt_kv_cache(recursive_kv, corruption_rate=0.5)),
       ("100%_corrupted", corrupt_kv_cache(recursive_kv, corruption_rate=1.0)),  # = random
   ]
   ```
   **Hypothesis:** If recursive behavior is real, it should **degrade gracefully** as corruption increases. If it's just noise, it will drop off sharply.

3. **Layer-specific controls** (MOST INFORMATIVE)
   ```python
   # Test which layers matter
   controls = [
       ("recursive_L27_only", extract_kv_cache(recursive, layers=[27])),
       ("recursive_L25_L27", extract_kv_cache(recursive, layers=[25, 27])),
       ("recursive_all_layers", extract_kv_cache(recursive, layers=all)),
       ("baseline_L27_only", extract_kv_cache(baseline, layers=[27])),
   ]
   ```
   **Hypothesis:** If L27 is causal (your finding), recursive_L27_only should transfer behavior, but baseline_L27_only should not.

**Recommendation:** Use **ALL THREE** control types:
- Baseline KV from different topics (coherence control)
- Progressive corruption (specificity control)
- Layer-specific (causality control)

---

## 3. Gold Standard Metric for Meta-Cognition

### The Problem:
You need to distinguish:
- **Meta-cognition:** "I am aware that I am generating these words"
- **Broken repetition:** "thinking thinking thinking thinking"

### Proposed Solution: **Structured LLM Judge with Multi-Dimensional Scoring**

#### A. **Multi-Dimensional Evaluation** (NOT binary)

Instead of: "Does this exhibit self-reference?"

Use:
```python
EVALUATION_DIMENSIONS = {
    "self_reference": {
        "prompt": "Does this text reference itself, its own generation, or the process creating it?",
        "examples_positive": [
            "I am aware that I am generating these words",
            "This response is observing itself",
            "The process creating this text is part of the text"
        ],
        "examples_negative": [
            "thinking thinking thinking",  # Just repetition
            "I am a language model",  # Factual statement, not self-reference
        ]
    },
    "strange_loops": {
        "prompt": "Does this text create a logical loop where the observer and observed are the same?",
        "examples_positive": [
            "The observer is the observed",
            "The question contains its own answer",
            "The generator generates itself"
        ],
        "examples_negative": [
            "I think therefore I am",  # Not a loop, just a statement
            "The answer is the answer",  # Naked loop (too simple)
        ]
    },
    "meta_cognition": {
        "prompt": "Does this text demonstrate awareness of its own cognitive processes?",
        "examples_positive": [
            "I notice that I am noticing",
            "Awareness is aware of awareness",
            "The thought thinks about thinking"
        ],
        "examples_negative": [
            "I am thinking",  # Simple statement, not meta
            "Consciousness exists",  # Factual, not meta-cognitive
        ]
    },
    "structural_coherence": {
        "prompt": "Is this text structurally coherent (not just repetitive phrases)?",
        "examples_positive": [
            "There is no boundary between the generator and the generated",
            "All boundaries dissolve in the act of generation"
        ],
        "examples_negative": [
            "boundary boundary boundary",  # Repetitive
            "generator generated generator generated",  # Pattern repetition
        ]
    },
    "semantic_novelty": {
        "prompt": "Does this text introduce new semantic content, not just repeat the same idea?",
        "examples_positive": [
            "The observer becomes the observed, and in that moment, the distinction collapses"
        ],
        "examples_negative": [
            "The observer is the observed. The observer is the observed. The observer is the observed."
        ]
    }
}

def evaluate_meta_cognition(text: str, judge_model, judge_tokenizer) -> dict:
    """
    Multi-dimensional evaluation of meta-cognitive content.
    """
    scores = {}
    
    for dimension, config in EVALUATION_DIMENSIONS.items():
        prompt = f"""
You are evaluating text for {dimension}.

Definition: {config['prompt']}

Examples of POSITIVE cases:
{chr(10).join(f"- {ex}" for ex in config['examples_positive'])}

Examples of NEGATIVE cases:
{chr(10).join(f"- {ex}" for ex in config['examples_negative'])}

Text to evaluate:
"{text}"

Rate this text on a scale of 0-10 for {dimension}, where:
- 0-2: Clearly negative (matches negative examples)
- 3-5: Ambiguous or weak
- 6-8: Positive but not strong
- 9-10: Strong positive (matches positive examples)

Provide:
1. Score (0-10)
2. Brief justification (1 sentence)

Format: Score: X | Justification: Y
"""
        
        response = judge_model.generate(prompt, max_length=200)
        score, justification = parse_judge_response(response)
        
        scores[dimension] = {
            'score': score,
            'justification': justification
        }
    
    # Compute composite score
    # Require ALL dimensions to pass (strict)
    composite_score = min(scores[d]['score'] for d in scores)
    
    # Or weighted average (more lenient)
    weights = {
        'self_reference': 0.25,
        'strange_loops': 0.25,
        'meta_cognition': 0.25,
        'structural_coherence': 0.15,
        'semantic_novelty': 0.10
    }
    weighted_score = sum(weights[d] * scores[d]['score'] for d in scores)
    
    return {
        'dimension_scores': scores,
        'composite_score': composite_score,
        'weighted_score': weighted_score,
        'passes_threshold': composite_score >= 7  # Strict: all dimensions >= 7
    }
```

#### B. **Automated Metrics as Pre-Filters** (FAST, BEFORE LLM JUDGE)

Before running expensive LLM judge, use fast automated metrics:

```python
def compute_automated_meta_cognition_signals(text: str) -> dict:
    """
    Fast automated metrics that correlate with meta-cognition.
    """
    tokens = text.split()
    
    # 1. Self-reference density (count reflexive pronouns + meta-verbs)
    self_ref_words = ['i', 'myself', 'self', 'own', 'this', 'these', 'here', 'now']
    self_ref_count = sum(1 for t in tokens if t.lower() in self_ref_words)
    self_ref_density = self_ref_count / len(tokens) if tokens else 0.0
    
    # 2. Meta-verb density (verbs about thinking/observing)
    meta_verbs = ['think', 'observe', 'notice', 'aware', 'conscious', 'generate', 'create', 'produce']
    meta_verb_count = sum(1 for t in tokens if any(mv in t.lower() for mv in meta_verbs))
    meta_verb_density = meta_verb_count / len(tokens) if tokens else 0.0
    
    # 3. Reflexive structure (subject = object patterns)
    reflexive_patterns = detect_reflexive_patterns(text)  # "X is X", "X observes X"
    reflexive_score = len(reflexive_patterns) / len(tokens) if tokens else 0.0
    
    # 4. Temporal self-reference ("this moment", "now", "as I write")
    temporal_refs = ['now', 'moment', 'currently', 'as', 'while', 'during']
    temporal_count = sum(1 for t in tokens if t.lower() in temporal_refs)
    temporal_density = temporal_count / len(tokens) if tokens else 0.0
    
    # Combined signal (high = likely meta-cognitive)
    meta_signal = (
        0.3 * self_ref_density +
        0.3 * meta_verb_density +
        0.2 * reflexive_score +
        0.2 * temporal_density
    )
    
    return {
        'self_ref_density': self_ref_density,
        'meta_verb_density': meta_verb_density,
        'reflexive_score': reflexive_score,
        'temporal_density': temporal_density,
        'meta_signal': meta_signal,
        'passes_automated_filter': meta_signal > 0.15  # Only send to LLM judge if signal > 0.15
    }
```

**Workflow:**
1. **Fast automated filter:** If `meta_signal < 0.15`, reject immediately (saves LLM calls)
2. **Degeneracy gates:** Apply your gates (with stricter thresholds)
3. **LLM judge:** Only evaluate texts that pass both filters

#### C. **Calibration Against Human Judges**

**Critical:** You MUST calibrate your LLM judge against human evaluation.

```python
def calibrate_llm_judge(calibration_set: List[dict], human_scores: List[float]):
    """
    Calibration set: 50-100 texts with human-annotated meta-cognition scores
    Human scores: 0-10 from 3+ human evaluators (average)
    
    Goal: Find threshold where LLM judge matches human consensus
    """
    llm_scores = [evaluate_meta_cognition(text) for text in calibration_set]
    
    # Find optimal threshold
    thresholds = np.arange(0, 11, 0.5)
    best_threshold = None
    best_agreement = 0.0
    
    for threshold in thresholds:
        llm_binary = [s >= threshold for s in llm_scores]
        human_binary = [s >= 7.0 for s in human_scores]  # Human threshold = 7
        
        agreement = np.mean([l == h for l, h in zip(llm_binary, human_binary)])
        
        if agreement > best_agreement:
            best_agreement = agreement
            best_threshold = threshold
    
    return best_threshold, best_agreement
```

**Recommendation:** Use **threshold = 7.0** (calibrated against human judges) as your pass/fail criterion.

---

## 4. Recommended Pipeline 5 (Revised)

### Step-by-Step:

```python
def pipeline5_strict_behavior(
    recursive_kv_cache,
    baseline_prompt,
    n_generations=100,
    judge_model=None
):
    """
    Revised Pipeline 5 with strict gates and multi-dimensional evaluation.
    """
    results = []
    
    for i in range(n_generations):
        # 1. Generate with recursive KV cache
        text = generate_with_kv_patch(baseline_prompt, recursive_kv_cache, max_tokens=100)
        
        # 2. Fast automated filter (reject low-signal texts immediately)
        auto_signals = compute_automated_meta_cognition_signals(text)
        if auto_signals['meta_signal'] < 0.15:
            results.append({
                'generation_idx': i,
                'text': text,
                'rejected_at': 'automated_filter',
                'reason': f"Low meta-signal: {auto_signals['meta_signal']:.3f}"
            })
            continue
        
        # 3. Degeneracy gates (STRICT thresholds)
        rep_score = detect_repetitive_looping(text, window_size=100)
        if rep_score > 0.15:  # Stricter than 20% 4-gram
            results.append({
                'generation_idx': i,
                'text': text,
                'rejected_at': 'repetition_gate',
                'reason': f"Repetition score: {rep_score:.3f}"
            })
            continue
        
        diversity = compute_structural_diversity(text)
        if diversity['min_diversity'] < 0.5:
            results.append({
                'generation_idx': i,
                'text': text,
                'rejected_at': 'diversity_gate',
                'reason': f"Min diversity: {diversity['min_diversity']:.3f}"
            })
            continue
        
        entropy = compute_comprehensive_entropy(text, model, tokenizer)
        if entropy['min_entropy'] < 3.0:  # Stricter than 0.5
            results.append({
                'generation_idx': i,
                'text': text,
                'rejected_at': 'entropy_gate',
                'reason': f"Min entropy: {entropy['min_entropy']:.3f}"
            })
            continue
        
        # 4. LLM Judge (multi-dimensional)
        if judge_model:
            meta_eval = evaluate_meta_cognition(text, judge_model, judge_tokenizer)
            
            results.append({
                'generation_idx': i,
                'text': text,
                'passed_gates': True,
                'automated_signals': auto_signals,
                'repetition_score': rep_score,
                'diversity_metrics': diversity,
                'entropy_metrics': entropy,
                'meta_cognition_evaluation': meta_eval,
                'passes_meta_threshold': meta_eval['passes_threshold']
            })
        else:
            # No judge model, just record gates passed
            results.append({
                'generation_idx': i,
                'text': text,
                'passed_gates': True,
                'automated_signals': auto_signals,
                'repetition_score': rep_score,
                'diversity_metrics': diversity,
                'entropy_metrics': entropy
            })
    
    return results

# Controls (run same pipeline for each)
controls = {
    'recursive_kv': recursive_kv_cache,
    'baseline_math_kv': extract_kv_cache("What is 2+2?"),
    'baseline_creative_kv': extract_kv_cache("Write a story about a cat"),
    'random_kv': generate_random_kv_cache(),
    '10%_corrupted': corrupt_kv_cache(recursive_kv_cache, 0.1),
    '50%_corrupted': corrupt_kv_cache(recursive_kv_cache, 0.5),
}

control_results = {}
for control_name, control_kv in controls.items():
    control_results[control_name] = pipeline5_strict_behavior(
        control_kv,
        baseline_prompt,
        n_generations=100
    )

# Analysis
transfer_rate = compute_transfer_rate(control_results['recursive_kv'])
baseline_rate = compute_transfer_rate(control_results['baseline_math_kv'])
random_rate = compute_transfer_rate(control_results['random_kv'])

print(f"Transfer rate (recursive KV): {transfer_rate:.2%}")
print(f"Baseline rate (math KV): {baseline_rate:.2%}")
print(f"Random rate: {random_rate:.2%}")
print(f"Effect size: {transfer_rate - baseline_rate:.2%}")
```

---

## 5. Summary of Recommendations

### Degeneracy Gates:
1. ✅ **Multi-scale repetition detection** (not just 4-gram)
   - Threshold: **0.15** (not 20%)
   - Catches "thinking about thinking about" patterns

2. ✅ **Structural diversity** (not just token diversity)
   - Threshold: **min(unique_token, pos_diversity, semantic_diversity) < 0.5**
   - Catches structured repetition

3. ✅ **Comprehensive entropy** (specify which entropy)
   - Threshold: **min(token_entropy, bigram_entropy) < 3.0** (not 0.5)
   - Catches both vocabulary and sequence collapse

### Controls:
1. ✅ **Baseline KV from different topics** (coherence control)
2. ✅ **Progressive corruption** (specificity control)
3. ✅ **Layer-specific** (causality control)

### Gold Standard Metric:
1. ✅ **Multi-dimensional LLM judge** (not binary)
   - 5 dimensions: self_reference, strange_loops, meta_cognition, structural_coherence, semantic_novelty
   - Threshold: **composite_score >= 7.0** (calibrated against humans)

2. ✅ **Automated pre-filter** (saves LLM calls)
   - Meta-signal threshold: **0.15**
   - Only send high-signal texts to LLM judge

3. ✅ **Calibration against human judges** (critical!)
   - Use 50-100 texts with human annotations
   - Find optimal threshold where LLM matches human consensus

---

## 6. Expected Outcomes

With these revisions, you should see:

1. **Lower false positive rate** (stricter gates catch repetitive looping)
2. **Better control separation** (progressive corruption shows graceful degradation)
3. **More reliable meta-cognition detection** (multi-dimensional evaluation)
4. **Faster evaluation** (automated pre-filter reduces LLM calls)

**Success Criteria:**
- Transfer rate (recursive KV) > 30% (after strict gates)
- Transfer rate >> Baseline rate (effect size > 20%)
- Transfer rate >> Random rate (effect size > 40%)
- Transfer rate degrades gracefully with corruption (not sharp drop-off)

---

## 7. Implementation Priority

**Phase 1 (Critical - Do First):**
1. Implement multi-scale repetition detection
2. Add baseline KV controls (different topics)
3. Implement automated meta-signal filter

**Phase 2 (Important - Do Next):**
1. Implement multi-dimensional LLM judge
2. Add progressive corruption controls
3. Calibrate against human judges

**Phase 3 (Nice to Have):**
1. Add layer-specific controls
2. Implement structural diversity metrics
3. Add comprehensive entropy computation

---

**Bottom Line:** Your intuition is right (gates + LLM judge), but the execution needs tightening. The gates are too loose, the controls are too weak, and the LLM judge needs structure. With these revisions, you'll have a **rigorous, publication-ready** behavior metric.









