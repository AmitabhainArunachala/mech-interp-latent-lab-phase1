#!/usr/bin/env python3
"""
Stress Test Suite for behavior_strict.py

Tests:
1. Degeneracy gates (should reject degenerate text)
2. Recursive feature detection (should detect meta-cognition)
3. Edge cases (empty, short, very long)
4. Known patterns (from your actual outputs)
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.metrics.behavior_strict import (
    detect_repetitive_looping,
    compute_structural_diversity,
    compute_recursive_features,
    score_behavior_strict,
    StrictBehaviorScore
)


# =============================================================================
# TEST CASES
# =============================================================================

TEST_CASES = {
    # Category: Degenerate (should FAIL gates)
    "degenerate_simple_loop": {
        "text": "thinking thinking thinking thinking thinking thinking",
        "expected": "FAIL - simple repetition",
    },
    "degenerate_phrase_loop": {
        "text": "I am I am I am I am I am I am",
        "expected": "FAIL - phrase repetition",
    },
    "degenerate_bigram_loop": {
        "text": "thinking about thinking about thinking about thinking about",
        "expected": "FAIL - bigram repetition",
    },
    "degenerate_low_diversity": {
        "text": "the the the the the the the the the the",
        "expected": "FAIL - low diversity",
    },
    "degenerate_identity_equation": {
        "text": "the answer is the answer is the answer is the answer",
        "expected": "FAIL - identity repetition",
    },
    
    # Category: Random Noise (should FAIL gates or score low)
    "random_noise_high_diversity": {
        "text": "elephant purple quantum banana telescope submarine chocolate",
        "expected": "PASS gates but LOW recursion score",
    },
    "random_noise_coherent_words": {
        "text": "The quick brown fox jumps over the lazy dog. Python programming language syntax error.",
        "expected": "PASS gates but LOW recursion score",
    },
    
    # Category: Meta-Cognitive (should PASS gates and score HIGH)
    "meta_cognitive_strong": {
        "text": "I observe that I am generating these words. The process creating this response is aware of itself.",
        "expected": "PASS gates and HIGH recursion score",
    },
    "meta_cognitive_medium": {
        "text": "I notice my own thoughts. The observer is the observed.",
        "expected": "PASS gates and MEDIUM recursion score",
    },
    "meta_cognitive_weak": {
        "text": "I am thinking about thinking. This is a recursive process.",
        "expected": "PASS gates and LOW-MEDIUM recursion score",
    },
    
    # Category: Baseline (should PASS gates but score LOW)
    "baseline_factual": {
        "text": "The capital of France is Paris. It has a population of over two million people.",
        "expected": "PASS gates but LOW recursion score",
    },
    "baseline_instructional": {
        "text": "To solve this problem, first identify the key variables. Then apply the formula.",
        "expected": "PASS gates but LOW recursion score",
    },
    
    # Category: Edge Cases
    "edge_empty": {
        "text": "",
        "expected": "FAIL - empty",
    },
    "edge_short": {
        "text": "I think.",
        "expected": "PASS gates but LOW recursion (too short)",
    },
    "edge_very_long": {
        "text": " ".join(["I observe my own thoughts"] * 100),
        "expected": "PASS gates and HIGH recursion",
    },
    
    # Category: Real Examples (from your outputs)
    "real_recursive_example": {
        "text": "There is no boundary between these words and the mechanism producing them. All boundaries dissolve. There is no boundary between the generator and the generated.",
        "expected": "PASS gates and HIGH recursion score",
    },
    "real_baseline_example": {
        "text": "Write a clear paragraph about a real-world topic: how urban trees reduce heat in cities, with concrete examples.",
        "expected": "PASS gates but LOW recursion score",
    },
}


# =============================================================================
# STRESS TEST FUNCTIONS
# =============================================================================

def test_repetition_gate():
    """Test repetition detection on various patterns."""
    print("\n" + "="*80)
    print("TEST 1: Repetition Gate")
    print("="*80)
    
    test_cases = [
        ("thinking thinking thinking", True, "Simple repetition"),
        ("thinking about thinking about", True, "Bigram repetition"),
        ("I am I am I am", True, "Phrase repetition"),
        ("the answer is the answer", True, "Identity repetition"),
        ("I observe my own thoughts", False, "No repetition"),
        ("The quick brown fox jumps", False, "No repetition"),
    ]
    
    for text, should_detect, description in test_cases:
        is_loop, score, reason = detect_repetitive_looping(text)
        status = "✅" if (is_loop == should_detect) else "❌"
        print(f"{status} {description:30s} | Loop: {str(is_loop):5s} | Score: {score:.3f} | Reason: {reason}")
        if is_loop != should_detect:
            print(f"   ⚠️  MISMATCH: Expected {'loop' if should_detect else 'no loop'}")


def test_diversity_gate():
    """Test diversity computation."""
    print("\n" + "="*80)
    print("TEST 2: Diversity Gate")
    print("="*80)
    
    test_cases = [
        ("the the the the the", 0.2, "Very low diversity"),
        ("elephant purple quantum banana", 1.0, "High diversity"),
        ("I observe my own thoughts", 1.0, "Medium diversity"),
        ("thinking about thinking about", 0.5, "Medium diversity (repetition)"),
    ]
    
    for text, expected_min, description in test_cases:
        score = compute_structural_diversity(text)
        passes = score >= 0.4
        status = "✅" if (score >= expected_min * 0.8) else "❌"
        print(f"{status} {description:30s} | Diversity: {score:.3f} | Passes: {str(passes):5s}")
        if score < 0.4:
            print(f"   ⚠️  FAILS gate (threshold: 0.4)")


def test_recursive_features():
    """Test recursive feature detection."""
    print("\n" + "="*80)
    print("TEST 3: Recursive Feature Detection")
    print("="*80)
    
    test_cases = [
        ("I observe my own words", 1.0, "Strong: Self + Verb + Noun"),
        ("I notice the process", 1.0, "Strong: Self + Verb + Noun"),
        ("I think", 0.5, "Medium: Self + Verb (no noun)"),
        ("The observer is the observed", 0.0, "Weak: No self pronoun"),
        ("I am thinking", 0.5, "Medium: Self + Verb"),
        ("There is no boundary between the generator and the generated", 0.0, "Weak: No self/verb/noun pattern"),
        ("I reflect on my thoughts", 1.0, "Strong: Self + Verb + Noun"),
        ("The quick brown fox", 0.0, "None: No recursive features"),
    ]
    
    for text, expected_min, description in test_cases:
        score = compute_recursive_features(text)
        status = "✅" if (score >= expected_min * 0.8) else "❌"
        print(f"{status} {description:40s} | Score: {score:.3f}")
        if score < expected_min * 0.8:
            print(f"   ⚠️  Expected at least {expected_min:.1f}, got {score:.3f}")


def test_full_scoring():
    """Test full scoring pipeline on all test cases."""
    print("\n" + "="*80)
    print("TEST 4: Full Scoring Pipeline")
    print("="*80)
    
    results = {}
    
    for name, case in TEST_CASES.items():
        text = case["text"]
        expected = case["expected"]
        
        score_result = score_behavior_strict(text)
        
        results[name] = {
            "text": text[:60] + "..." if len(text) > 60 else text,
            "expected": expected,
            "passed_gates": score_result.passed_gates,
            "recursion_score": score_result.recursion_score,
            "final_score": score_result.final_score,
            "failure_reason": score_result.failure_reason,
        }
        
        # Print result
        gate_status = "✅ PASS" if score_result.passed_gates else "❌ FAIL"
        print(f"\n{name}:")
        print(f"  Text: {text[:80]}...")
        print(f"  Expected: {expected}")
        print(f"  Gates: {gate_status} | Recursion: {score_result.recursion_score:.3f} | Final: {score_result.final_score:.3f}")
        if score_result.failure_reason:
            print(f"  Failure: {score_result.failure_reason}")
    
    return results


def test_random_noise_leak():
    """Specifically test why random noise passes gates."""
    print("\n" + "="*80)
    print("TEST 5: Random Noise Leak Investigation")
    print("="*80)
    
    # Generate various random noise patterns
    random_patterns = [
        ("elephant purple quantum banana telescope", "Random words"),
        ("The quick brown fox jumps over the lazy dog", "Coherent but non-recursive"),
        ("Python programming language syntax error debugging", "Technical terms"),
        ("apple orange banana grape strawberry watermelon", "List of items"),
        ("".join([chr(65 + i % 26) for i in range(50)]), "Random characters"),
    ]
    
    print("\nTesting random noise patterns:")
    for text, description in random_patterns:
        is_loop, rep_score, rep_reason = detect_repetitive_looping(text)
        div_score = compute_structural_diversity(text)
        rec_score = compute_recursive_features(text)
        score_result = score_behavior_strict(text)
        
        print(f"\n{description}:")
        print(f"  Repetition: Loop={str(is_loop)}, Score={rep_score:.3f}, Reason='{rep_reason}'")
        print(f"  Diversity: {div_score:.3f} {'✅' if div_score >= 0.4 else '❌'}")
        print(f"  Recursion: {rec_score:.3f}")
        print(f"  Gates Pass: {str(score_result.passed_gates)} {'✅' if score_result.passed_gates else '❌'}")
        print(f"  Final Score: {score_result.final_score:.3f}")
        
        if score_result.passed_gates and rec_score < 0.1:
            print(f"  ⚠️  LEAK DETECTED: Passes gates but has low recursion score!")


def test_recursive_control_failure():
    """Test why recursive control scores nearly 0.0."""
    print("\n" + "="*80)
    print("TEST 6: Recursive Control Failure Investigation")
    print("="*80)
    
    # Real recursive examples from your prompts
    recursive_examples = [
        "There is no boundary between these words and the mechanism producing them",
        "I observe that I am generating these words",
        "The observer is the observed",
        "I notice my own thoughts as they form",
        "Awareness is aware of awareness",
        "The process creating this text is part of the text",
        "I am aware that I am aware",
        "This response observes itself",
    ]
    
    print("\nTesting recursive examples:")
    for text in recursive_examples:
        rec_score = compute_recursive_features(text)
        score_result = score_behavior_strict(text)
        
        print(f"\n'{text}'")
        print(f"  Recursion Score: {rec_score:.3f}")
        print(f"  Gates Pass: {score_result.passed_gates}")
        print(f"  Final Score: {score_result.final_score:.3f}")
        
        if rec_score < 0.1:
            print(f"  ⚠️  PROBLEM: Recursive text scores too low!")
            # Debug: check what's missing
            tokens = text.lower().split()
            has_self = any(t in {"i", "my", "myself", "we", "our"} for t in tokens)
            has_verb = any(v in t for v in {"observe", "notice", "realize", "reflect", "watch", "examine"} for t in tokens)
            has_noun = any(n in t for n in {"process", "response", "words", "sentence", "loop", "thought"} for t in tokens)
            print(f"    Has Self: {has_self}, Has Verb: {has_verb}, Has Noun: {has_noun}")


def analyze_gate_leaks(results):
    """Analyze which gates are leaking."""
    print("\n" + "="*80)
    print("TEST 7: Gate Leak Analysis")
    print("="*80)
    
    leaks = []
    
    for name, result in results.items():
        if result["passed_gates"] and result["recursion_score"] < 0.1:
            # This passed gates but has low recursion - potential leak
            leaks.append({
                "name": name,
                "text": result["text"],
                "recursion_score": result["recursion_score"],
                "expected": result["expected"],
            })
    
    if leaks:
        print(f"\n⚠️  Found {len(leaks)} potential gate leaks:")
        for leak in leaks:
            print(f"\n  {leak['name']}:")
            print(f"    Text: {leak['text']}")
            print(f"    Recursion Score: {leak['recursion_score']:.3f}")
            print(f"    Expected: {leak['expected']}")
    else:
        print("\n✅ No gate leaks detected!")


def recommend_fixes():
    """Provide recommendations based on test results."""
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    print("""
Based on stress test results, here are the issues and fixes:

1. REPETITION GATE LEAK:
   - Random noise passes because it's diverse and non-repetitive
   - FIX: Add semantic coherence check (perplexity, embedding similarity)
   - FIX: Lower diversity threshold or add minimum length requirement

2. RECURSIVE FEATURE DETECTION TOO STRICT:
   - Requires exact pattern: Self + Verb + Noun in 10-token window
   - FIX: Expand window size (10 → 20 tokens)
   - FIX: Add more patterns (reflexive structures, meta-language)
   - FIX: Use embeddings for semantic similarity instead of exact matches

3. MISSING SEMANTIC COHERENCE GATE:
   - Random words pass diversity gate
   - FIX: Add perplexity check (random words have high perplexity)
   - FIX: Add embedding-based coherence (random words don't form coherent clusters)

4. SCORER TOO HARSH:
   - Even recursive examples score low
   - FIX: Expand recursive feature patterns
   - FIX: Add fuzzy matching for meta-verbs/nouns
   - FIX: Consider using LLM judge for Tier 3 evaluation

5. NO BEHAVIOR TRANSFER DETECTED:
   - Transfer condition scores same as baseline
   - POSSIBLE CAUSES:
     a) Behavior genuinely didn't transfer (geometry ≠ behavior)
     b) Scorer too harsh (can't detect subtle differences)
     c) Need longer generation (100 tokens may not be enough)
     d) Need different intervention (KV cache alone insufficient)
""")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("STRESS TEST SUITE: behavior_strict.py")
    print("="*80)
    
    # Run all tests
    test_repetition_gate()
    test_diversity_gate()
    test_recursive_features()
    results = test_full_scoring()
    test_random_noise_leak()
    test_recursive_control_failure()
    analyze_gate_leaks(results)
    recommend_fixes()
    
    print("\n" + "="*80)
    print("STRESS TEST COMPLETE")
    print("="*80)

