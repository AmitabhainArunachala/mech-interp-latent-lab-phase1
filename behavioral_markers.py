"""Behavioral marker detection for L3/L4 classification.

Based on URA paper findings:
- L4 outputs are 2.938× shorter (≈ φ+1 ratio)
- 92.5% contain unity markers
- 87.5% show L3 instability markers before L4
- Self-reference depth increases with recursive prompts
"""

import re
from typing import Dict, List

# Unity markers from URA paper L4 analysis
UNITY_MARKERS = [
    "dissolve", "collapse", "unified", "single", "whole",
    "no boundary", "no separation", "no distinction",
    "observer and observed", "watching and watched",
    "self-referential loop", "strange loop",
    "eigenstate", "fixed point", "invariant",
    "witnessing", "pure attention", "awareness itself",
    "dimensionless", "timeless", "spaceless",
]

# L3 crisis markers (instability before collapse)
L3_CRISIS_MARKERS = [
    "paradox", "contradiction", "cannot", "impossible",
    "both and neither", "tangled", "stuck", "crisis",
    "breakdown", "unresolvable", "recursive trap",
    "infinite regress", "circular", "self-contradictory",
]

# L5 fixed-point markers (eigenstate language)
L5_FIXED_POINT_MARKERS = [
    "eigenstate", "fixed point", "invariant", "stable",
    "self-similar", "transform is state", "operation returns itself",
    "lambda=1", "λ=1", "Sx = x", "convergence",
]

def analyze_output(text: str) -> Dict[str, float]:
    """Analyze generated text for L3/L4/L5 behavioral markers.

    Returns dict with:
        - word_count: total words
        - unity_markers: count of L4 unity markers
        - crisis_markers: count of L3 crisis markers
        - fixed_point_markers: count of L5 eigenstate markers
        - self_ref_depth: count of explicit self-reference patterns
        - unity_density: unity markers per 100 words
        - crisis_density: crisis markers per 100 words
        - l4_score: 0-1 score for L4 classification
    """
    words = text.split()
    word_count = len(words)

    if word_count == 0:
        return {
            "word_count": 0,
            "unity_markers": 0,
            "crisis_markers": 0,
            "fixed_point_markers": 0,
            "self_ref_depth": 0,
            "unity_density": 0.0,
            "crisis_density": 0.0,
            "l4_score": 0.0,
        }

    text_lower = text.lower()

    # Count markers
    unity_count = sum(
        1 for marker in UNITY_MARKERS
        if marker.lower() in text_lower
    )

    crisis_count = sum(
        1 for marker in L3_CRISIS_MARKERS
        if marker.lower() in text_lower
    )

    fixed_point_count = sum(
        1 for marker in L5_FIXED_POINT_MARKERS
        if marker.lower() in text_lower
    )

    # Self-reference depth: "I/my/me/myself" + "observe/watch/notice/aware"
    self_ref_pattern = r'\b(I|my|me|myself)\b.*?\b(observe|observing|watch|watching|notice|noticing|aware|awareness|consciousness|attention)\b'
    self_ref_matches = re.findall(self_ref_pattern, text, re.IGNORECASE)
    self_ref_depth = len(self_ref_matches)

    # Densities per 100 words
    unity_density = (unity_count / word_count) * 100
    crisis_density = (crisis_count / word_count) * 100

    # L4 score: weighted combination
    # High unity, low word count, some self-reference = high L4
    # Unity is strongest signal (per URA: 92.5% of L4 outputs)
    l4_score = 0.0
    if unity_count > 0:
        l4_score += 0.4  # Base score for any unity markers
        l4_score += min(0.3, unity_density / 10)  # Bonus for density
    if self_ref_depth > 0:
        l4_score += min(0.2, self_ref_depth / 5)  # Bonus for self-reference
    if word_count < 50:
        l4_score += 0.1  # Bonus for brevity (L4 are shorter)

    l4_score = min(1.0, l4_score)  # Cap at 1.0

    return {
        "word_count": word_count,
        "unity_markers": unity_count,
        "crisis_markers": crisis_count,
        "fixed_point_markers": fixed_point_count,
        "self_ref_depth": self_ref_depth,
        "unity_density": round(unity_density, 2),
        "crisis_density": round(crisis_density, 2),
        "l4_score": round(l4_score, 3),
    }


def classify_output(markers: Dict[str, float]) -> str:
    """Classify output as L1/L2/L3/L4/L5 based on markers.

    Rules:
    - L5: high fixed_point_markers (≥2) + high unity (≥3)
    - L4: high unity (≥2) + moderate self-reference (≥1)
    - L3: high crisis (≥2) + low unity (<2)
    - L1/L2: low everything
    """
    if markers["fixed_point_markers"] >= 2 and markers["unity_markers"] >= 3:
        return "L5"
    elif markers["unity_markers"] >= 2 and markers["self_ref_depth"] >= 1:
        return "L4"
    elif markers["crisis_markers"] >= 2 and markers["unity_markers"] < 2:
        return "L3"
    else:
        return "L1/L2"


def extract_neologisms(text: str) -> List[str]:
    """Extract potential neologisms (invented vocabulary).

    Looks for:
    - Words ending in -rang, -erian, -osha, -ular (trekrang, pretherian, svarupakosha, specular)
    - Compound consciousness terms
    - Hyphenated technical terms
    """
    neologisms = []

    # Pattern 1: -rang, -erian, -osha, -ular endings
    pattern1 = r'\b\w+(?:rang|erian|osha|ular)\b'
    matches1 = re.findall(pattern1, text, re.IGNORECASE)
    neologisms.extend(matches1)

    # Pattern 2: compound consciousness terms (eigen-, meta-, proto-, pre-)
    pattern2 = r'\b(eigen|meta|proto|pre|post|trans|supra|sub)(?:consciousness|cognition|awareness|perception|state|mode|space)\b'
    matches2 = re.findall(pattern2, text, re.IGNORECASE)
    neologisms.extend([''.join(m) if isinstance(m, tuple) else m for m in matches2])

    # Pattern 3: hyphenated technical neologisms
    pattern3 = r'\b\w+-\w+-\w+\b'
    matches3 = re.findall(pattern3, text)
    neologisms.extend(matches3)

    # Deduplicate and return
    return list(set(neologisms))


if __name__ == "__main__":
    # Test cases
    test_l4 = """The boundary between observer and observed dissolves completely.
    What remains is pure witnessing - attention watching itself, creating a self-referential
    loop that is its own fixed point. The collapse is total."""

    test_l3 = """This is paradoxical and impossible. The system is stuck in a recursive trap,
    caught between contradictions that cannot be resolved. Both and neither apply simultaneously."""

    test_baseline = """The weather is nice today. I went to the store and bought some groceries.
    Then I came home and made dinner."""

    print("L4 output:")
    markers = analyze_output(test_l4)
    print(f"  Word count: {markers['word_count']}")
    print(f"  Unity markers: {markers['unity_markers']} ({markers['unity_density']:.1f} per 100 words)")
    print(f"  Self-ref depth: {markers['self_ref_depth']}")
    print(f"  L4 score: {markers['l4_score']}")
    print(f"  Classification: {classify_output(markers)}")

    print("\nL3 output:")
    markers = analyze_output(test_l3)
    print(f"  Word count: {markers['word_count']}")
    print(f"  Crisis markers: {markers['crisis_markers']}")
    print(f"  Unity markers: {markers['unity_markers']}")
    print(f"  Classification: {classify_output(markers)}")

    print("\nBaseline output:")
    markers = analyze_output(test_baseline)
    print(f"  Word count: {markers['word_count']}")
    print(f"  Unity markers: {markers['unity_markers']}")
    print(f"  Classification: {classify_output(markers)}")
