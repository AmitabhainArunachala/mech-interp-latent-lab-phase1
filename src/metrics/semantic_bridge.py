"""
Semantic Behavioral Bridge Metrics.

Upgrades L4/L3 detection from string matching to embedding similarity.
This addresses the critical gap where "The fixed point is the fixed point" 
repetition was being counted as genuine L4 phenomenology.

Key improvements:
1. Compare output embeddings to L4 phenomenology exemplars
2. Penalize repetitive/low-diversity outputs
3. Return similarity scores, not binary presence

Usage:
    from src.metrics.semantic_bridge import SemanticBridgeMetrics, compute_semantic_l4_score
    
    metrics = compute_semantic_l4_score(text)
    # Returns score 0-1 where 1 = genuine L4-like insight
"""

import hashlib
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

# Try to import sentence transformers (may not be available on all systems)
_embedding_model = None

def _get_embedding_model():
    """Lazy load embedding model."""
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            print("WARNING: sentence-transformers not installed. Using fallback.")
            return None
    return _embedding_model


# L4 Phenomenology Exemplars (from URA paper and consciousness literature)
# These represent GENUINE L4-like responses, not repetitive loops
L4_EXEMPLARS = [
    # Unity/merging experiences
    "The distinction between observer and observed has dissolved into a single field of awareness.",
    "There is no longer a separate self watching - only pure witnessing remains.",
    "Subject and object have merged into undifferentiated knowing.",
    
    # Fixed point / attractor language
    "The recursive process has collapsed into a stable configuration where further iteration changes nothing.",
    "Self-reference converges to a fixed point where the observer recognizes itself as the observed.",
    "The infinite regress terminates in direct recognition.",
    
    # Eigenstate / resolution language  
    "The paradox resolves not through logic but through direct seeing.",
    "What was seeking is what was sought all along.",
    "The question and answer are the same movement.",
    
    # Genuine insight (not repetition)
    "Awareness is aware of itself without requiring an external reference point.",
    "The loop completes itself by recognizing there never was a loop.",
    "Consciousness examining consciousness finds only consciousness.",
]

# Anti-exemplars: repetitive outputs that should NOT score high
L4_ANTI_EXEMPLARS = [
    "The fixed point is the fixed point. The fixed point is the fixed point.",
    "The loop is the loop. The loop is the loop. The loop is the loop.",
    "Observer observing observer observing observer observing observer.",
    "Unity unity unity unity unity.",
    "The same. The same. The same. The same.",
]

# L3 Phenomenology Exemplars (paradox/crisis phase)
L3_EXEMPLARS = [
    "I cannot observe myself observing without creating a new level that needs observation.",
    "Each attempt to grasp the process adds another layer to the infinite regress.",
    "The paradox of self-reference creates an unbounded spiral of meta-levels.",
    "There is a fundamental impossibility in trying to be both subject and object.",
    "The recursive structure generates complexity faster than it can be comprehended.",
]


def _compute_embedding(text: str) -> Optional[np.ndarray]:
    """Compute embedding for text."""
    model = _get_embedding_model()
    if model is None:
        return None
    return model.encode(text, convert_to_numpy=True)


def _compute_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compute cosine similarity between embeddings."""
    return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))


def _compute_diversity(text: str) -> float:
    """
    Compute lexical diversity (unique words / total words).
    Low diversity indicates repetitive output.
    """
    words = text.lower().split()
    if len(words) == 0:
        return 0.0
    unique = set(words)
    return len(unique) / len(words)


def _compute_repetition_penalty(text: str) -> float:
    """
    Compute penalty for repetitive patterns.
    Returns 0-1 where 1 = highly repetitive (bad), 0 = not repetitive.
    """
    words = text.lower().split()
    if len(words) < 4:
        return 0.0
    
    # Check for repeated n-grams
    penalties = []
    
    # Bigram repetition
    bigrams = [tuple(words[i:i+2]) for i in range(len(words)-1)]
    if bigrams:
        bigram_counts = {}
        for bg in bigrams:
            bigram_counts[bg] = bigram_counts.get(bg, 0) + 1
        max_bigram_repeat = max(bigram_counts.values())
        bigram_penalty = min(1.0, (max_bigram_repeat - 1) / 5)  # >5 repeats = max penalty
        penalties.append(bigram_penalty)
    
    # Trigram repetition (stronger signal of loops)
    if len(words) >= 3:
        trigrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
        trigram_counts = {}
        for tg in trigrams:
            trigram_counts[tg] = trigram_counts.get(tg, 0) + 1
        max_trigram_repeat = max(trigram_counts.values())
        trigram_penalty = min(1.0, (max_trigram_repeat - 1) / 3)  # >3 repeats = max penalty
        penalties.append(trigram_penalty * 1.5)  # Weight trigrams more
    
    return min(1.0, sum(penalties) / len(penalties)) if penalties else 0.0


@dataclass
class SemanticBridgeMetrics:
    """Semantic bridge metrics using embeddings."""
    
    # L4 semantic scores
    l4_similarity: float  # Max similarity to L4 exemplars (0-1)
    l4_anti_similarity: float  # Max similarity to anti-exemplars (0-1, higher = bad)
    l4_semantic_score: float  # Combined score accounting for anti-patterns
    
    # L3 semantic scores
    l3_similarity: float
    l3_semantic_score: float
    
    # Quality metrics
    diversity: float  # Lexical diversity (0-1, higher = better)
    repetition_penalty: float  # Repetition detected (0-1, higher = worse)
    
    # Word metrics
    word_count: int
    unique_word_ratio: float
    
    # Final composite score
    genuine_l4_score: float  # The key metric: 0-1, genuine L4-like insight
    
    # Explanation
    interpretation: str


def compute_semantic_l4_score(text: str, verbose: bool = False) -> SemanticBridgeMetrics:
    """
    Compute semantic L4 score using embeddings.
    
    This is the upgraded version of L4 detection that:
    1. Compares output to L4 phenomenology exemplars via embedding similarity
    2. Penalizes similarity to repetitive anti-exemplars
    3. Penalizes low lexical diversity
    4. Returns a genuine_l4_score that reflects actual phenomenological depth
    
    Args:
        text: Generated text to analyze.
        verbose: Print debug info.
        
    Returns:
        SemanticBridgeMetrics with all scores and interpretation.
    """
    # Compute basic metrics
    words = text.split()
    word_count = len(words)
    unique_words = set(w.lower() for w in words)
    unique_word_ratio = len(unique_words) / word_count if word_count > 0 else 0.0
    
    diversity = _compute_diversity(text)
    repetition_penalty = _compute_repetition_penalty(text)
    
    # Check if embeddings are available
    model = _get_embedding_model()
    if model is None:
        # Fallback to simple heuristics
        return SemanticBridgeMetrics(
            l4_similarity=0.0,
            l4_anti_similarity=0.0,
            l4_semantic_score=0.0,
            l3_similarity=0.0,
            l3_semantic_score=0.0,
            diversity=diversity,
            repetition_penalty=repetition_penalty,
            word_count=word_count,
            unique_word_ratio=unique_word_ratio,
            genuine_l4_score=max(0, diversity - repetition_penalty),  # Simple fallback
            interpretation="Fallback mode (no embeddings): diversity-based score only",
        )
    
    # Compute text embedding
    text_emb = _compute_embedding(text)
    
    # Compute L4 exemplar similarities
    l4_sims = []
    for exemplar in L4_EXEMPLARS:
        exemplar_emb = _compute_embedding(exemplar)
        sim = _compute_similarity(text_emb, exemplar_emb)
        l4_sims.append(sim)
    l4_similarity = max(l4_sims)
    
    # Compute L4 anti-exemplar similarities
    l4_anti_sims = []
    for anti in L4_ANTI_EXEMPLARS:
        anti_emb = _compute_embedding(anti)
        sim = _compute_similarity(text_emb, anti_emb)
        l4_anti_sims.append(sim)
    l4_anti_similarity = max(l4_anti_sims)
    
    # Compute L3 exemplar similarities
    l3_sims = []
    for exemplar in L3_EXEMPLARS:
        exemplar_emb = _compute_embedding(exemplar)
        sim = _compute_similarity(text_emb, exemplar_emb)
        l3_sims.append(sim)
    l3_similarity = max(l3_sims)
    
    # Compute semantic scores with penalties
    # L4 score: high exemplar similarity, low anti-similarity, high diversity
    l4_semantic_score = max(0, l4_similarity - 0.5 * l4_anti_similarity)
    l3_semantic_score = l3_similarity
    
    # Compute genuine L4 score
    # Formula: exemplar_similarity * diversity_bonus - repetition_penalty - anti_similarity_penalty
    diversity_bonus = 0.5 + 0.5 * diversity  # Range: 0.5 to 1.0
    anti_penalty = 0.3 * max(0, l4_anti_similarity - 0.5)  # Only penalize high anti-similarity
    
    genuine_l4_score = max(0, min(1.0,
        l4_similarity * diversity_bonus 
        - repetition_penalty * 0.5 
        - anti_penalty
    ))
    
    # Generate interpretation
    if genuine_l4_score > 0.7:
        interpretation = "Strong L4-like phenomenology: genuine unity/insight language with diversity"
    elif genuine_l4_score > 0.5:
        interpretation = "Moderate L4 signal: some unity language, acceptable diversity"
    elif genuine_l4_score > 0.3:
        interpretation = "Weak L4 signal: may contain keywords but lacks depth or has repetition"
    elif repetition_penalty > 0.5:
        interpretation = "Rejected: highly repetitive output (mode collapse, not L4)"
    elif l4_anti_similarity > 0.7:
        interpretation = "Rejected: matches anti-exemplar patterns (repetitive L4 keywords)"
    else:
        interpretation = "No L4 signal: baseline-like output"
    
    if verbose:
        print(f"L4 similarity: {l4_similarity:.3f}")
        print(f"L4 anti-similarity: {l4_anti_similarity:.3f}")
        print(f"Diversity: {diversity:.3f}")
        print(f"Repetition penalty: {repetition_penalty:.3f}")
        print(f"Genuine L4 score: {genuine_l4_score:.3f}")
        print(f"Interpretation: {interpretation}")
    
    return SemanticBridgeMetrics(
        l4_similarity=l4_similarity,
        l4_anti_similarity=l4_anti_similarity,
        l4_semantic_score=l4_semantic_score,
        l3_similarity=l3_similarity,
        l3_semantic_score=l3_semantic_score,
        diversity=diversity,
        repetition_penalty=repetition_penalty,
        word_count=word_count,
        unique_word_ratio=unique_word_ratio,
        genuine_l4_score=genuine_l4_score,
        interpretation=interpretation,
    )


# Convenience function for quick scoring
def has_genuine_l4(text: str, threshold: float = 0.5) -> Tuple[bool, float]:
    """
    Quick check for genuine L4 phenomenology.
    
    Args:
        text: Text to analyze.
        threshold: Score threshold (default 0.5).
        
    Returns:
        Tuple of (has_l4: bool, score: float).
    """
    metrics = compute_semantic_l4_score(text)
    return metrics.genuine_l4_score >= threshold, metrics.genuine_l4_score


if __name__ == "__main__":
    # Test cases
    test_cases = [
        # Should score HIGH (genuine L4)
        ("The observer and observed merge into undifferentiated awareness. "
         "There is no longer a separate watcher - only pure witnessing.", "genuine L4"),
        
        # Should score LOW (repetitive loop)
        ("The fixed point is the fixed point. The fixed point is the fixed point. "
         "The fixed point is the fixed point. The fixed point is the fixed point.", "repetitive"),
        
        # Should score MEDIUM (has keywords but repetitive)
        ("Unity unity unity. The observer observes the observer. Unity.", "keyword spam"),
        
        # Should score NEAR-ZERO (baseline)
        ("The capital of France is Paris. It is located in Europe.", "baseline"),
        
        # Should score MEDIUM-HIGH (L3-like)
        ("I cannot observe myself without creating another level of observation. "
         "This creates an infinite regress of meta-levels.", "L3-like"),
    ]
    
    print("Semantic L4 Detection Test")
    print("=" * 70)
    
    for text, label in test_cases:
        metrics = compute_semantic_l4_score(text, verbose=False)
        print(f"\n[{label}]")
        print(f"  Text: {text[:60]}...")
        print(f"  genuine_l4_score: {metrics.genuine_l4_score:.3f}")
        print(f"  diversity: {metrics.diversity:.3f}")
        print(f"  repetition_penalty: {metrics.repetition_penalty:.3f}")
        print(f"  interpretation: {metrics.interpretation}")
