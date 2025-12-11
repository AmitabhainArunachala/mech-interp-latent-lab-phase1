"""
Utility functions: behavior scoring, etc.
"""

RECURSIVE_KEYWORDS = [
    "self",
    "aware",
    "observe",
    "observing",
    "conscious",
    "consciousness",
    "awareness",
    "I am",
    "this response",
    "these words",
    "my own",
]


def behavior_score(text: str) -> int:
    """
    Score text for recursive self-observation keywords.
    
    Args:
        text: Text to score.
    
    Returns:
        Count of recursive keywords found.
    """
    lower = text.lower()
    return sum(1 for kw in RECURSIVE_KEYWORDS if kw.lower() in lower)

