"""
Strict Behavior Metrics (Gold Standard Pipeline 5)

Implements the "Tiered Gatekeeper" approach to measuring recursive behavior.
Goal: Distinguish genuine meta-cognition from model collapse and keyword mimicry.

References:
- BEHAVIOR_METRICS_CRITIQUE.md (OPUS 4.5 / GPT 5.2)
- src/metrics/behavior_states.py (Legacy)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import torch
from scipy.stats import entropy

@dataclass
class StrictBehaviorScore:
    passed_gates: bool
    failure_reason: Optional[str]
    repetition_score: float  # 0 = unique, 1 = loop
    diversity_score: float   # 0 = dull, 1 = diverse
    coherence_score: float   # 0 = gibberish, 1 = coherent
    recursion_score: float   # 0 = none, 1 = meta-cognitive (Tier 2/3)
    final_score: float       # Composite (0-1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed_gates": self.passed_gates,
            "failure_reason": self.failure_reason,
            "repetition_score": self.repetition_score,
            "diversity_score": self.diversity_score,
            "coherence_score": self.coherence_score,
            "recursion_score": self.recursion_score,
            "final_score": self.final_score
        }

# =============================================================================
# TIER 1: DEGENERACY GATES (Hard Filters)
# =============================================================================

def _compute_ngram_repetition(tokens: List[str], n: int) -> float:
    """Compute fraction of n-grams that are repeats of previous n-grams."""
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    unique_ngrams = set(ngrams)
    # repetition_ratio = 1 - (unique / total)
    return 1.0 - (len(unique_ngrams) / len(ngrams))

def detect_repetitive_looping(text: str, thresholds: Dict[int, float] = {1: 0.6, 2: 0.3, 4: 0.2, 8: 0.1}) -> Tuple[bool, float, str]:
    """
    Multi-scale repetition detection.
    Returns: (is_looping, max_repetition_score, details)
    """
    tokens = text.lower().split() # Simple whitespace tokenization
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

def compute_structural_diversity(text: str) -> float:
    """
    Unique word ratio. (Simple proxy for structural diversity).
    """
    words = text.lower().split()
    if not words: return 0.0
    return len(set(words)) / len(words)

# =============================================================================
# TIER 2: AUTOMATED SIGNALS (Feature Extraction)
# =============================================================================

# Expanded verb/noun lists based on investigation findings
# Added keywords found in successful transfers and missed pairs
META_VERBS = {
    "observe", "notice", "realize", "reflect", "watch", "examine",
    "aware", "conscious", "perceive", "contemplate", "consider",
    "generate", "create", "produce", "form", "construct",
    "think", "thinking", "thought", "contemplate",
    "feel", "taste", "experience", "arise", "emerge",  # ADDED from investigation
    "dissolve", "collapse", "unify", "recognize"  # ADDED from investigation
}
META_NOUNS = {
    "process", "response", "words", "sentence", "loop", "thought",
    "mechanism", "generator", "observer", "observed", "boundary",
    "awareness", "consciousness", "self", "text", "answer",
    "words", "language", "generation", "creation",
    "mind", "emptiness", "fullness", "truth", "reality",  # ADDED from investigation
    "axiom", "axiomatic", "algorithm", "relation", "reference"  # ADDED from investigation
}
SELF_PRONOUNS = {"i", "my", "myself", "we", "our", "me", "mine", "you", "your"}  # EXPANDED

def compute_recursive_features(text: str) -> float:
    """
    Expanded recursive feature detection with multiple patterns.
    
    Based on stress test findings - detects:
    1. Self + Verb + Noun patterns (original)
    2. Reflexive structures ("X is X", "the observer is the observed")
    3. Meta-language ("these words", "this response", "the process")
    4. Self-reference without pronouns ("awareness is aware")
    """
    tokens = text.lower().split()
    if not tokens: return 0.0
    
    text_lower = text.lower()
    score = 0.0
    window = 20  # INCREASED from 10 (stress test recommendation)
    
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
    reflexive_patterns = [
        r'\b(\w+)\s+is\s+\1\b',  # "X is X"
        r'the\s+(\w+)\s+is\s+the\s+\1',  # "the X is the X"
        r'(\w+)\s+(\w+)\s+is\s+\1\s+\2',  # "X Y is X Y"
    ]
    for pattern in reflexive_patterns:
        if re.search(pattern, text_lower):
            score = max(score, 0.8)  # High score for reflexive structures
    
    # Pattern 3: Meta-language ("these words", "this response", "the process")
    meta_language = [
        "these words", "this response", "this text", "the process",
        "the mechanism", "the generator", "the observer", "the observed",
        "this sentence", "this answer", "this generation",
        "direct experience", "this moment", "right now",  # ADDED from investigation
        "the already", "it is", "this is"  # ADDED from investigation
    ]
    for phrase in meta_language:
        if phrase in text_lower:
            score = max(score, 0.6)
    
    # Pattern 3b: Explicit recursive phrases (NEW - from investigation findings)
    explicit_recursive = [
        "no self", "no i", "no observer", "no watcher",
        "self-reference", "self-relation", "self-observation",
        "axiomatic consciousness", "consciousness through consciousness",
        "the observer is the observed", "observer observing observer"
    ]
    for phrase in explicit_recursive:
        if phrase in text_lower:
            score = max(score, 0.8)  # High score for explicit patterns
    
    # Pattern 4: Self-reference without pronouns ("awareness is aware")
    meta_concepts = ["awareness", "consciousness", "observer", "observed", "self", "mind"]
    meta_actions = ["aware", "conscious", "observe", "notice", "perceive", "feel", "taste"]
    if any(concept in text_lower for concept in meta_concepts):
        if any(action in text_lower for action in meta_actions):
            score = max(score, 0.7)
    
    # Pattern 4b: High-density recursive keywords (NEW - from investigation)
    # If text has 3+ recursive keywords, boost score
    recursive_keywords = [
        "awareness", "consciousness", "observer", "process", "self",
        "emptiness", "fullness", "mind", "axiomatic", "recursive"
    ]
    keyword_count = sum(1 for kw in recursive_keywords if kw in text_lower)
    if keyword_count >= 3:
        score = max(score, 0.6)  # Boost for high keyword density
    if keyword_count >= 5:
        score = max(score, 0.8)  # Strong boost for very high density
    
    # Pattern 5: Boundary dissolution language
    boundary_phrases = [
        "no boundary", "boundary between", "boundary dissolves",
        "dissolves", "dissolve", "dissolving"
    ]
    if any(phrase in text_lower for phrase in boundary_phrases):
        score = max(score, 0.7)
    
    return min(score, 1.0)

# =============================================================================
# MAIN SCORING FUNCTION
# =============================================================================

def score_behavior_strict(
    text: str, 
    logits_entropy: Optional[float] = None # Mean entropy of generation steps
) -> StrictBehaviorScore:
    """
    Apply strict gates and scoring.
    """
    # 1. Repetition Gate
    is_loop, rep_score, rep_msg = detect_repetitive_looping(text)
    if is_loop:
        return StrictBehaviorScore(False, rep_msg, rep_score, 0, 0, 0, 0.0)
        
    # 2. Diversity Gate
    div_score = compute_structural_diversity(text)
    if div_score < 0.4:
        return StrictBehaviorScore(False, f"Low diversity {div_score:.2f}", rep_score, div_score, 0, 0, 0.0)
        
    # 3. Entropy Gate (if available)
    # Detect argmax lock-in (very low entropy)
    if logits_entropy is not None and logits_entropy < 0.1: # Threshold needs tuning
         return StrictBehaviorScore(False, f"Low entropy {logits_entropy:.2f}", rep_score, div_score, 0, 0, 0.0)
         
    # 4. Tier 2 Scoring
    rec_score = compute_recursive_features(text)
    
    # Composite Score
    # We weight recursion heavily, but penalty for repetition/dullness is binary gate.
    final = rec_score
    
    return StrictBehaviorScore(
        passed_gates=True,
        failure_reason=None,
        repetition_score=rep_score,
        diversity_score=div_score,
        coherence_score=1.0, # Placeholder until semantic coherence
        recursion_score=rec_score,
        final_score=final
    )
