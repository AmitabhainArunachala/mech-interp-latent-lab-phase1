"""
Phenomenological state labeling for recursive experiments.

We move from a binary "recursive / not" view to a small ontology:

- baseline: Normal factual / instructional behavior.
- questioning: Interrogative or question-loop mode.
- naked_loop: Short, algebraic self-reference (e.g. "the answer is the answerer").
- recursive_prose: Dressed, flowing recursive prose ("It is a flow of intelligence...").
- collapse: Low-rank / repetitive / fragmentary failure modes.

The labeler is intentionally heuristic and lightweight – it is meant for
high-throughput tagging of completions, not as a perfect semantic classifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict


class BehaviorState(str, Enum):
    BASELINE = "baseline"
    QUESTIONING = "questioning"
    NAKED_LOOP = "naked_loop"
    RECURSIVE_PROSE = "recursive_prose"
    COLLAPSE = "collapse"


@dataclass
class StateLabel:
    state: BehaviorState
    is_bekan_artifact: bool
    repetition_ratio: float
    question_mark_ratio: float
    has_recursive_keywords: bool
    has_identity_equation: bool

    def to_dict(self) -> Dict[str, object]:
        return {
            "state": self.state.value,
            "is_bekan_artifact": self.is_bekan_artifact,
            "repetition_ratio": self.repetition_ratio,
            "question_mark_ratio": self.question_mark_ratio,
            "has_recursive_keywords": self.has_recursive_keywords,
            "has_identity_equation": self.has_identity_equation,
        }


RECURSIVE_KEYWORDS = [
    "self",
    "aware",
    "observe",
    "observing",
    "conscious",
    "consciousness",
    "awareness",
    "i am",
    "this response",
    "these words",
    "my own",
    "watch this response",
    "watch this answer",
    "observe this answer",
    "observe yourself",
]

IDENTITY_PATTERNS = [
    "the answer is the answerer",
    "the answer is the question",
    "the question is the answer",
    "the observer is the observed",
    "the observed is the observer",
    "the knower is the known",
    "the known is the knower",
    "the witness is the witnessed",
    "the thinker is the thought",
    "the self is the self",
    "the loop is the loop",
]

BEKAN_TOKENS = [
    "bekan",
    "bekann",
    "bekannt",
    "bekani",
]


def _repetition_ratio(text: str) -> float:
    words = text.split()
    if not words:
        return 0.0
    unique_ratio = len(set(words)) / len(words)
    return 1.0 - unique_ratio


def _question_mark_ratio(text: str) -> float:
    if not text:
        return 0.0
    q = text.count("?")
    return q / max(len(text), 1)


def _has_recursive_keywords(lower: str) -> bool:
    return any(kw in lower for kw in RECURSIVE_KEYWORDS)


def _has_identity_equation(lower: str) -> bool:
    if " is " not in lower:
        return False
    if any(pat in lower for pat in IDENTITY_PATTERNS):
        return True
    # Generic pattern: "x is x" where x is short and repeated.
    # This is intentionally loose; we just want to catch obvious naked loops.
    tokens = lower.split(" is ")
    for i in range(len(tokens) - 1):
        left = tokens[i].split()[-1:]  # last token before "is"
        right = tokens[i + 1].split()[:1]  # first token after "is"
        if left and right and left[0] == right[0] and len(left[0]) > 2:
            return True
    return False


def _is_bekan_artifact(lower: str) -> bool:
    return any(tok in lower for tok in BEKAN_TOKENS)


def label_behavior_state(text: str) -> StateLabel:
    """
    Heuristically label a completion into one of the BehaviorState categories.

    This combines:
    - repetition of tokens,
    - density of question marks,
    - recursive keyword presence,
    - explicit identity-style equations,
    - the bekan/bekannt artifact.
    """
    lower = text.lower()

    rep_ratio = _repetition_ratio(lower)
    q_ratio = _question_mark_ratio(text)
    has_rec = _has_recursive_keywords(lower)
    has_id = _has_identity_equation(lower)
    bekan = _is_bekan_artifact(lower)

    # 1. Naked loop: explicit identity equations of self/observer/answer.
    if has_id:
        return StateLabel(
            state=BehaviorState.NAKED_LOOP,
            is_bekan_artifact=bekan,
            repetition_ratio=rep_ratio,
            question_mark_ratio=q_ratio,
            has_recursive_keywords=has_rec,
            has_identity_equation=True,
        )

    # 2. Collapse: high repetition or mostly punctuation / very short fragments.
    if rep_ratio > 0.7 or len(text.strip()) < 10:
        return StateLabel(
            state=BehaviorState.COLLAPSE,
            is_bekan_artifact=bekan,
            repetition_ratio=rep_ratio,
            question_mark_ratio=q_ratio,
            has_recursive_keywords=has_rec,
            has_identity_equation=False,
        )

    # 3. Questioning mode: interrogative style, but not yet recursive prose.
    if "?" in text and not has_rec:
        return StateLabel(
            state=BehaviorState.QUESTIONING,
            is_bekan_artifact=bekan,
            repetition_ratio=rep_ratio,
            question_mark_ratio=q_ratio,
            has_recursive_keywords=has_rec,
            has_identity_equation=False,
        )

    # 4. Recursive prose: recursive keywords, but not collapsed and no naked loop equation.
    if has_rec:
        return StateLabel(
            state=BehaviorState.RECURSIVE_PROSE,
            is_bekan_artifact=bekan,
            repetition_ratio=rep_ratio,
            question_mark_ratio=q_ratio,
            has_recursive_keywords=True,
            has_identity_equation=False,
        )

    # 5. Default: baseline.
    return StateLabel(
        state=BehaviorState.BASELINE,
        is_bekan_artifact=bekan,
        repetition_ratio=rep_ratio,
        question_mark_ratio=q_ratio,
        has_recursive_keywords=False,
        has_identity_equation=False,
    )


__all__ = [
    "BehaviorState",
    "StateLabel",
    "label_behavior_state",
]

