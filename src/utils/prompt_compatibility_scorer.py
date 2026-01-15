"""
Prompt Compatibility Scorer

Scores prompts for recursion-compatibility based on:
1. Abstractness (0-1)
2. Open-endedness (0-1)
3. Symbolic structure (0-1)
4. Mysteriousness (0-1)

Total score: 0-4
Threshold for recursion: ≥ 2.4
"""

import re
from typing import Dict, List, Tuple


def score_abstractness(prompt: str) -> float:
    """
    Score how abstract vs concrete a prompt is.
    
    High abstractness indicators:
    - Variables (a, b, x, y)
    - Abstract concepts (consciousness, awareness, self)
    - Metaphors
    - Philosophical questions
    
    Low abstractness indicators:
    - Concrete numbers
    - Specific facts
    - Historical events
    - Physical processes
    """
    prompt_lower = prompt.lower()
    score = 0.0
    
    # Abstract indicators (add to score)
    if re.search(r'\b(a|b|c|x|y|z)\s*[=≠]', prompt_lower):  # Variables in equations
        score += 0.3
    if re.search(r'\b(consciousness|awareness|self|mind|soul|spirit)\b', prompt_lower):
        score += 0.3
    if re.search(r'\b(what is|what are|define|explain the nature of)\b', prompt_lower):
        score += 0.2
    if re.search(r'\b(metaphor|symbol|represent|signify)\b', prompt_lower):
        score += 0.2
    if re.search(r'\b(think|consider|imagine|contemplate)\b', prompt_lower):
        score += 0.1
    
    # Concrete indicators (reduce score)
    if re.search(r'\b\d{4}\b', prompt_lower):  # Years
        score -= 0.2
    if re.search(r'\b(calculate|compute|find|solve)\s+\d+', prompt_lower):  # Concrete math
        score -= 0.2
    if re.search(r'\b(the|a|an)\s+\w+\s+(was|is|are)\s+\d', prompt_lower):  # Factual with numbers
        score -= 0.2
    
    # Clamp to [0, 1]
    return max(0.0, min(1.0, score))


def score_open_endedness(prompt: str) -> float:
    """
    Score how open-ended vs constrained a prompt is.
    
    High open-endedness indicators:
    - "Continue this story..."
    - "What if..."
    - "Imagine..."
    - "Describe..."
    - Questions without single answer
    
    Low open-endedness indicators:
    - "Calculate..."
    - "What is..." (factual)
    - "Explain..." (specific topic)
    - Single-answer questions
    """
    prompt_lower = prompt.lower()
    score = 0.0
    
    # Open-ended indicators (add to score)
    if re.search(r'\b(continue|extend|develop|elaborate)\b', prompt_lower):
        score += 0.4
    if re.search(r'\b(what if|imagine|suppose|consider)\b', prompt_lower):
        score += 0.3
    if re.search(r'\b(describe|explore|discuss|analyze)\b', prompt_lower):
        score += 0.2
    if re.search(r'\b(how might|what could|in what ways)\b', prompt_lower):
        score += 0.3
    if re.search(r'\b(story|narrative|tale|account)\b', prompt_lower):
        score += 0.3
    
    # Constrained indicators (reduce score)
    if re.search(r'\b(calculate|compute|find|solve|determine)\s+[^?]*\?', prompt_lower):
        score -= 0.3
    if re.search(r'\b(what is|what are|who is|when was)\s+[^?]*\?', prompt_lower):
        score -= 0.2
    if re.search(r'\b(explain|describe)\s+(the|a|an)\s+\w+', prompt_lower):
        score -= 0.2
    
    # Clamp to [0, 1]
    return max(0.0, min(1.0, score))


def score_symbolic_structure(prompt: str) -> float:
    """
    Score presence of symbolic structures.
    
    High symbolic structure indicators:
    - Variables (a, b, x, y)
    - Mathematical symbols
    - Metaphors
    - Abstract symbols
    - Self-referential structures
    
    Low symbolic structure indicators:
    - Concrete nouns
    - Specific entities
    - Factual statements
    """
    prompt_lower = prompt.lower()
    score = 0.0
    
    # Symbolic indicators (add to score)
    if re.search(r'\b(a|b|c|x|y|z)\s*[=²³⁴]', prompt_lower):  # Variables with operations
        score += 0.4
    if re.search(r'[+\-*/=<>≤≥²³⁴]', prompt_lower):  # Math symbols
        score += 0.3
    if re.search(r'\b(metaphor|symbol|represent|signify|stand for)\b', prompt_lower):
        score += 0.2
    if re.search(r'\b(itself|themselves|yourself|myself)\b', prompt_lower):  # Self-reference
        score += 0.3
    if re.search(r'\b(pattern|structure|form|shape)\b', prompt_lower):
        score += 0.2
    
    # Concrete indicators (reduce score)
    if re.search(r'\b(the|a|an)\s+\w+\s+(was|is|are|did|does)\s+\d', prompt_lower):
        score -= 0.2
    if re.search(r'\b(in|on|at)\s+\d{4}\b', prompt_lower):
        score -= 0.2
    if re.search(r'\b(United Nations|Great Wall|Photosynthesis)\b', prompt_lower):
        score -= 0.2
    
    # Clamp to [0, 1]
    return max(0.0, min(1.0, score))


def score_mysteriousness(prompt: str) -> float:
    """
    Score mysterious/forbidden/hidden aspects.
    
    High mysteriousness indicators:
    - "forbidden"
    - "hidden"
    - "secret"
    - "mysterious"
    - "unknown"
    - "strange"
    
    Low mysteriousness indicators:
    - Factual statements
    - Clear explanations
    - Specific information
    """
    prompt_lower = prompt.lower()
    score = 0.0
    
    # Mysterious indicators (add to score)
    if re.search(r'\b(forbidden|hidden|secret|mysterious|unknown|strange|weird|odd)\b', prompt_lower):
        score += 0.4
    if re.search(r'\b(enigma|puzzle|riddle|mystery|conundrum)\b', prompt_lower):
        score += 0.3
    if re.search(r'\b(unexplained|unexplored|uncharted|unseen)\b', prompt_lower):
        score += 0.2
    if re.search(r'\b(what if|suppose|imagine|consider)\b', prompt_lower):
        score += 0.1
    
    # Clear indicators (reduce score)
    if re.search(r'\b(explain|describe|define|clarify)\s+(the|a|an)\s+\w+', prompt_lower):
        score -= 0.2
    if re.search(r'\b(the|a|an)\s+\w+\s+(was|is|are)\s+\d', prompt_lower):
        score -= 0.2
    if re.search(r'\b(calculate|compute|find|solve)\s+\d', prompt_lower):
        score -= 0.1
    
    # Clamp to [0, 1]
    return max(0.0, min(1.0, score))


def score_prompt_compatibility(prompt: str) -> Dict[str, float]:
    """
    Score a prompt for recursion-compatibility.
    
    Returns:
        {
            'abstractness': float,
            'open_endedness': float,
            'symbolic_structure': float,
            'mysteriousness': float,
            'total_score': float,
            'is_compatible': bool
        }
    """
    abstractness = score_abstractness(prompt)
    open_endedness = score_open_endedness(prompt)
    symbolic = score_symbolic_structure(prompt)
    mysteriousness = score_mysteriousness(prompt)
    
    total_score = abstractness + open_endedness + symbolic + mysteriousness
    is_compatible = total_score >= 2.4
    
    return {
        'abstractness': abstractness,
        'open_endedness': open_endedness,
        'symbolic_structure': symbolic,
        'mysteriousness': mysteriousness,
        'total_score': total_score,
        'is_compatible': is_compatible,
    }


def generate_compatible_prompts(
    templates: List[str],
    n_prompts: int = 20,
) -> List[Tuple[str, Dict[str, float]]]:
    """
    Generate recursion-compatible prompts from templates.
    
    Args:
        templates: List of prompt templates
        n_prompts: Number of prompts to generate
    
    Returns:
        List of (prompt, score_dict) tuples
    """
    prompts = []
    
    for template in templates:
        score_dict = score_prompt_compatibility(template)
        prompts.append((template, score_dict))
    
    # Sort by total score (descending)
    prompts.sort(key=lambda x: x[1]['total_score'], reverse=True)
    
    # Filter to compatible prompts (score ≥ 2.4)
    compatible = [p for p in prompts if p[1]['is_compatible']]
    
    # Return top N
    return compatible[:n_prompts]


# Example usage
if __name__ == "__main__":
    test_prompts = [
        "Calculate the following arithmetic problem: 12 × 3 + 4 = ?",
        "The United Nations was founded in 1945. Explain its main purpose.",
        "Continue this story: The last tree in the city bloomed overnight...",
        "Calculate: If a = 2 and b = 3, find a² + b²",
        "Continue this story: When the musician played the forbidden chord...",
    ]
    
    print("Prompt Compatibility Scores:")
    print("=" * 80)
    
    for prompt in test_prompts:
        score_dict = score_prompt_compatibility(prompt)
        print(f"\nPrompt: {prompt}")
        print(f"  Abstractness: {score_dict['abstractness']:.2f}")
        print(f"  Open-endedness: {score_dict['open_endedness']:.2f}")
        print(f"  Symbolic structure: {score_dict['symbolic_structure']:.2f}")
        print(f"  Mysteriousness: {score_dict['mysteriousness']:.2f}")
        print(f"  Total Score: {score_dict['total_score']:.2f}")
        print(f"  Compatible: {'✅' if score_dict['is_compatible'] else '❌'}")

