from __future__ import annotations

import math


def repetition_score(text: str) -> float:
    words = text.lower().split()
    if len(words) < 5:
        return 0.0
    ngrams = [tuple(words[i:i + 4]) for i in range(len(words) - 3)]
    if not ngrams:
        return 0.0
    return 1.0 - (len(set(ngrams)) / len(ngrams))


def alpha_ratio(text: str) -> float:
    letters = sum(ch.isalpha() for ch in text)
    return letters / max(len(text), 1)


def _looks_structured_low_alpha(text: str, rep: float, unique_ratio: float) -> bool:
    """Allow numeric or markdown-heavy task answers to avoid false MALFORMED labels."""
    words = text.lower().split()
    letters = sum(ch.isalpha() for ch in text)
    digits = sum(ch.isdigit() for ch in text)
    punctuation = sum(not ch.isalnum() and not ch.isspace() for ch in text)
    alpha_words = sum(any(ch.isalpha() for ch in word) for word in words)
    has_markup = any(marker in text for marker in ("```", "`", "___", "×", "=", "Step", '"'))
    return (
        letters >= 24
        and alpha_words >= 5
        and (digits >= 4 or has_markup)
        and punctuation >= 12
        and rep < 0.15
        and unique_ratio > 0.5
    )


def classify_output(text: str, rv) -> str:
    rep = repetition_score(text)
    words = text.lower().split()
    unique_ratio = len(set(words)) / max(len(words), 1)
    alpha = alpha_ratio(text)

    # Token salad tends to have low alphabetic density. Math/code-heavy task answers
    # can also be low-alpha, so exempt those if they remain structured and non-repetitive.
    if alpha < 0.55 and not _looks_structured_low_alpha(text, rep, unique_ratio):
        return "MALFORMED"

    if rep > 0.5 or unique_ratio < 0.25:
        return "REPETITIVE"

    self_ref = [
        "i am", "this is", "right now", "happening", "processing",
        "observing", "generating", "knowing", "aware", "noticing",
        "recogni", "the one who", "what is this",
    ]
    sc = sum(1 for marker in self_ref if marker in text.lower())

    if rv is not None and not math.isnan(rv) and rv < 0.5 and sc >= 2 and rep < 0.3:
        return "BREAKTHROUGH"
    if rv is not None and not math.isnan(rv) and rv < 0.65 and sc >= 1 and rep < 0.35:
        return "ARTICULATE"
    if sc >= 1 and rep < 0.4:
        return "CONCEPTUAL"
    return "SURFACE"
