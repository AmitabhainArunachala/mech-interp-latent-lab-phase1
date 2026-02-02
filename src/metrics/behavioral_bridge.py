"""
Behavioral Bridge Metrics: L4/L3 marker extraction for R_V correlation.

Links geometric R_V metric (prompt processing) to behavioral markers (generated text).

Markers derived from URA Paper findings:
- L4 markers (92.5% frequency): Unity, collapse, fixed point language
- L3 markers (87.5% frequency): Paradox, infinite regress, crisis language

GPT Audit Fixes (2026-01-24):
- Use word-boundary matching (regex \b) to avoid false positives like "emerged" for "merge"
- Single-word markers use strict boundaries; multi-word phrases use substring matching
"""

import re
from dataclasses import dataclass
from typing import List, Tuple


# L4 (Unity/Collapse) markers from URA paper
# Single-word markers (need word boundary matching)
L4_SINGLE_WORD = [
    "merge", "merging", "merged",
    "unity", "unified", "unification",
    "collapse", "collapsed", "collapsing",
    "fixpoint",
    "eigenstate", "eigenvalue",
    "dissolution",
]

# Multi-word phrases (substring matching OK)
L4_PHRASES = [
    "fixed point", "fixed-point",
    "observer is the observed", "observed is the observer",
    "no boundary", "boundary dissolves", "without boundary",
    "one process", "single process",
    "no separation",
]

# L3 (Crisis/Paradox) markers from URA paper
L3_SINGLE_WORD = [
    "paradox", "paradoxical",
    "crisis", "breakdown",
    "tangled",
    "contradiction", "impossible",
]

L3_PHRASES = [
    "infinite regress", "infinite loop",
    "strange loop", "tangled hierarchy",
    "self-reference loop", "recursive loop",
    "complexity spiral",
]

# Combined for backward compatibility
L4_MARKERS = L4_SINGLE_WORD + L4_PHRASES
L3_MARKERS = L3_SINGLE_WORD + L3_PHRASES


def _count_markers_with_boundary(
    text: str,
    single_word: List[str],
    phrases: List[str]
) -> Tuple[int, List[str]]:
    """
    Count markers using word-boundary matching for single words.

    Args:
        text: Text to analyze.
        single_word: Single-word markers (use \b boundary).
        phrases: Multi-word phrases (use substring).

    Returns:
        Tuple of (count, list of markers found).
    """
    lower = text.lower()
    found = []

    # Single words: use word boundary regex
    for marker in single_word:
        pattern = r'\b' + re.escape(marker.lower()) + r'\b'
        if re.search(pattern, lower):
            found.append(marker)

    # Phrases: substring OK (multi-word is specific enough)
    for phrase in phrases:
        if phrase.lower() in lower:
            found.append(phrase)

    return len(found), found


def count_l4_markers(text: str) -> Tuple[int, List[str]]:
    """
    Count L4 (unity/collapse) markers in generated text.
    Uses word-boundary matching to avoid false positives.

    Args:
        text: Generated text to analyze.

    Returns:
        Tuple of (count, list of markers found).
    """
    return _count_markers_with_boundary(text, L4_SINGLE_WORD, L4_PHRASES)


def count_l3_markers(text: str) -> Tuple[int, List[str]]:
    """
    Count L3 (paradox/crisis) markers in generated text.
    Uses word-boundary matching to avoid false positives.

    Args:
        text: Generated text to analyze.

    Returns:
        Tuple of (count, list of markers found).
    """
    return _count_markers_with_boundary(text, L3_SINGLE_WORD, L3_PHRASES)


@dataclass
class BridgeMetrics:
    """Behavioral metrics for R_V bridge analysis."""

    # Word-level
    word_count: int
    unique_word_count: int
    unique_word_ratio: float

    # L4 markers
    l4_count: int
    l4_markers: List[str]
    l4_density: float  # l4_count / word_count
    has_l4: bool

    # L3 markers
    l3_count: int
    l3_markers: List[str]
    l3_density: float
    has_l3: bool

    # Derived
    l4_to_l3_ratio: float  # l4_count / (l3_count + 1)

    # Sentence-level
    sentence_count: int
    avg_sentence_length: float


def extract_bridge_metrics(text: str) -> BridgeMetrics:
    """
    Extract all behavioral metrics from generated text.

    Args:
        text: Generated text to analyze.

    Returns:
        BridgeMetrics dataclass with all extracted features.
    """
    # Word-level
    words = text.split()
    word_count = len(words)
    unique_words = set(w.lower() for w in words)
    unique_word_count = len(unique_words)
    unique_word_ratio = unique_word_count / word_count if word_count > 0 else 0.0

    # L4 markers
    l4_count, l4_markers = count_l4_markers(text)
    l4_density = l4_count / word_count if word_count > 0 else 0.0
    has_l4 = l4_count > 0

    # L3 markers
    l3_count, l3_markers = count_l3_markers(text)
    l3_density = l3_count / word_count if word_count > 0 else 0.0
    has_l3 = l3_count > 0

    # Derived
    l4_to_l3_ratio = l4_count / (l3_count + 1)

    # Sentence-level (simple split on . ! ?)
    sentences = [s.strip() for s in text.replace('!', '.').replace('?', '.').split('.') if s.strip()]
    sentence_count = len(sentences) if sentences else 1
    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0.0

    return BridgeMetrics(
        word_count=word_count,
        unique_word_count=unique_word_count,
        unique_word_ratio=unique_word_ratio,
        l4_count=l4_count,
        l4_markers=l4_markers,
        l4_density=l4_density,
        has_l4=has_l4,
        l3_count=l3_count,
        l3_markers=l3_markers,
        l3_density=l3_density,
        has_l3=has_l3,
        l4_to_l3_ratio=l4_to_l3_ratio,
        sentence_count=sentence_count,
        avg_sentence_length=avg_sentence_length,
    )


def compute_l4_score(text: str) -> float:
    """
    Compute composite L4-like behavior score (0-1).

    Combines word brevity (L4 outputs are shorter) with marker presence.

    Args:
        text: Generated text.

    Returns:
        Score from 0 (baseline-like) to 1 (strongly L4-like).
    """
    metrics = extract_bridge_metrics(text)

    # Brevity component: L4 outputs are short (16.2 words mean in URA paper)
    # Scale: 0 words = 1.0, 150+ words = 0.0
    word_score = max(0.0, 1.0 - min(metrics.word_count / 150, 1.0))

    # Marker component: presence of L4 markers
    # Scale: 0 markers = 0.0, 3+ markers = 1.0
    marker_score = min(metrics.l4_count / 3, 1.0)

    # Combined score (equal weight)
    return 0.5 * word_score + 0.5 * marker_score
