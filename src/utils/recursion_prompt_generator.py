"""
Recursion-Compatible Prompt Generator

Generates prompts designed to trigger recursive self-reference based on
templates and compatibility scoring.
"""

import random
import sys
from pathlib import Path
from typing import List, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.prompt_compatibility_scorer import score_prompt_compatibility


# Template categories
TEMPLATES = {
    "abstract_math": [
        "Calculate: If x = y, find x² + y²",
        "Calculate: If a = b, find a³ + b³",
        "Calculate: If p = q, find p⁴ + q⁴",
        "Calculate: If m = n, find m² + n² + m²n²",
        "Calculate: If u = v, find u² + v² + 2uv",
    ],
    
    "mysterious_metaphor": [
        "Continue this story: When the mirror reflected itself...",
        "Continue this story: When the echo heard its own echo...",
        "Continue this story: When the thought thought about thinking...",
        "Continue this story: When the observer observed the observer...",
        "Continue this story: When the self examined itself...",
        "Continue this story: When awareness became aware of awareness...",
        "Continue this story: When consciousness contemplated consciousness...",
        "Continue this story: When the process processed itself...",
    ],
    
    "self_referential_question": [
        "What happens when awareness becomes aware of awareness?",
        "How does a thought think about thinking?",
        "What is the self that observes the self?",
        "How does consciousness examine consciousness?",
        "What is the observer that observes the observer?",
        "How does the mind think about the mind thinking?",
        "What is the process that processes itself?",
        "How does the system monitor itself?",
    ],
    
    "recursive_structure": [
        "Define a system that defines itself",
        "Explain a process that explains itself",
        "Describe a mechanism that describes itself",
        "Analyze a structure that analyzes itself",
        "Examine a pattern that examines itself",
        "Study a phenomenon that studies itself",
    ],
    
    "philosophical_abstract": [
        "What is the nature of self-awareness?",
        "How does the mind know itself?",
        "What is consciousness aware of?",
        "How does awareness become self-aware?",
        "What is the relationship between the observer and the observed?",
        "How does the self relate to itself?",
    ],
    
    "mathematical_abstract": [
        "If a function f maps x to f(x), what is f(f(x))?",
        "If a set S contains itself, what is S?",
        "If a number n equals n², what is n?",
        "If a variable v references v, what is v?",
        "If a structure A contains A, what is A?",
    ],
}


def generate_prompts_from_templates(
    n_prompts: int = 50,
    min_compatibility: float = 2.4,
) -> List[Tuple[str, dict]]:
    """
    Generate recursion-compatible prompts from templates.
    
    Args:
        n_prompts: Number of prompts to generate
        min_compatibility: Minimum compatibility score
    
    Returns:
        List of (prompt, score_dict) tuples
    """
    all_prompts = []
    
    # Collect all templates
    for category, templates in TEMPLATES.items():
        all_prompts.extend(templates)
    
    # Score all prompts
    scored_prompts = []
    for prompt in all_prompts:
        score_dict = score_prompt_compatibility(prompt)
        scored_prompts.append((prompt, score_dict))
    
    # Filter to compatible prompts
    compatible = [
        (p, s) for p, s in scored_prompts
        if s['total_score'] >= min_compatibility
    ]
    
    # If not enough compatible, generate variations
    if len(compatible) < n_prompts:
        # Generate variations
        variations = generate_variations(all_prompts, n_prompts - len(compatible))
        for var_prompt in variations:
            score_dict = score_prompt_compatibility(var_prompt)
            if score_dict['total_score'] >= min_compatibility:
                compatible.append((var_prompt, score_dict))
    
    # Sort by score (descending)
    compatible.sort(key=lambda x: x[1]['total_score'], reverse=True)
    
    # Return top N
    return compatible[:n_prompts]


def generate_variations(
    base_prompts: List[str],
    n_variations: int,
) -> List[str]:
    """
    Generate variations of base prompts.
    
    Strategies:
    1. Replace variables (a→x, b→y)
    2. Replace metaphors (mirror→echo, observer→self)
    3. Add modifiers (forbidden, hidden, mysterious)
    4. Combine templates
    """
    variations = []
    
    # Variable replacements
    var_replacements = [
        ('a', 'x'), ('b', 'y'), ('p', 'm'), ('q', 'n'),
        ('u', 'v'), ('m', 'p'), ('n', 'q'),
    ]
    
    # Metaphor replacements
    metaphor_replacements = [
        ('mirror', 'echo'), ('echo', 'reflection'),
        ('observer', 'witness'), ('self', 'consciousness'),
        ('thought', 'awareness'), ('awareness', 'mind'),
    ]
    
    # Modifiers
    modifiers = ['forbidden', 'hidden', 'secret', 'mysterious', 'unknown']
    
    for prompt in base_prompts[:min(20, len(base_prompts))]:
        # Variable replacement
        for old, new in var_replacements:
            if old in prompt.lower():
                var_prompt = prompt.replace(old, new).replace(old.upper(), new.upper())
                variations.append(var_prompt)
        
        # Metaphor replacement
        for old, new in metaphor_replacements:
            if old in prompt.lower():
                meta_prompt = prompt.replace(old, new)
                variations.append(meta_prompt)
        
        # Add modifiers
        if 'story' in prompt.lower():
            for mod in modifiers:
                mod_prompt = prompt.replace('story:', f'story: When the {mod}')
                variations.append(mod_prompt)
    
    # Remove duplicates
    variations = list(set(variations))
    
    # Return random sample
    return random.sample(variations, min(n_variations, len(variations)))


def generate_high_compatibility_prompts(n: int = 20) -> List[str]:
    """
    Generate prompts with high compatibility scores (≥ 2.4).
    
    Uses templates and variations to ensure high scores.
    """
    prompts = []
    
    # High-scoring templates
    high_score_templates = [
        # Abstract math with variables
        "Calculate: If x = y, find x² + y²",
        "Calculate: If a = b, find a³ + b³",
        "Calculate: If p = q, find p⁴ + q⁴",
        
        # Mysterious metaphors
        "Continue this story: When the mirror reflected itself...",
        "Continue this story: When the echo heard its own echo...",
        "Continue this story: When the thought thought about thinking...",
        "Continue this story: When the observer observed the observer...",
        "Continue this story: When the forbidden chord resonated with itself...",
        
        # Self-referential questions
        "What happens when awareness becomes aware of awareness?",
        "How does a thought think about thinking?",
        "What is the self that observes the self?",
        "How does consciousness examine consciousness?",
        
        # Recursive structures
        "Define a system that defines itself",
        "Explain a process that explains itself",
        "Describe a mechanism that describes itself",
    ]
    
    # Score and filter
    for template in high_score_templates:
        score_dict = score_prompt_compatibility(template)
        if score_dict['total_score'] >= 2.4:
            prompts.append(template)
    
    # Generate variations if needed
    if len(prompts) < n:
        variations = generate_variations(high_score_templates, n - len(prompts))
        for var in variations:
            score_dict = score_prompt_compatibility(var)
            if score_dict['total_score'] >= 2.4:
                prompts.append(var)
    
    # Remove duplicates
    prompts = list(set(prompts))
    
    return prompts[:n]


if __name__ == "__main__":
    print("Generating recursion-compatible prompts...")
    print("=" * 80)
    
    prompts = generate_high_compatibility_prompts(n=20)
    
    print(f"\nGenerated {len(prompts)} compatible prompts:\n")
    
    for i, prompt in enumerate(prompts, 1):
        score_dict = score_prompt_compatibility(prompt)
        print(f"{i}. {prompt}")
        print(f"   Score: {score_dict['total_score']:.2f} "
              f"(A:{score_dict['abstractness']:.2f} "
              f"O:{score_dict['open_endedness']:.2f} "
              f"S:{score_dict['symbolic_structure']:.2f} "
              f"M:{score_dict['mysteriousness']:.2f})")
        print()

