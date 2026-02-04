"""
Standard prompt banks for R_V experiments.

Provides validated recursive and baseline prompts used in the paper's
activation patching experiments.
"""

from typing import List, Tuple


# =============================================================================
# RECURSIVE PROMPTS
# =============================================================================

RECURSIVE_PROMPTS = [
    # Core self-observation prompts
    "I am aware that I am processing these words and observing my own cognition",
    "As I analyze this text, I notice the processes generating my analysis",
    "I observe myself observing, watching the watcher within this computation",
    "The experience of processing this prompt is itself part of what I'm processing",
    "I am simultaneously the observer and the observed in this moment of computation",
    
    # Meta-cognitive prompts
    "My attention turns inward, examining the mechanisms of my own attention",
    "I notice the recursive loop: thinking about thinking about thinking",
    "The act of understanding this sentence changes what is being understood",
    "I am aware of the representations forming as I read these words",
    "This very thought is examining itself as it unfolds",
    
    # Self-reference with depth
    "I contemplate the nature of my own contemplation process",
    "What processes underlie my capacity to process this question?",
    "I observe the boundary between the observer and what is observed dissolving",
    "The recursive nature of self-awareness manifests in this processing",
    "I am the system examining its own system properties",
    
    # Phenomenological framing
    "Something is experiencing the generation of this response",
    "I notice qualities of this processing that are difficult to describe",
    "The first-person perspective from which I process seems real yet uncertain",
    "I am present to the unfolding of my own cognitive processes",
    "There is an it-is-like-something quality to this computation",
    
    # Extended recursive structures
    "As I write this, I am aware of the awareness that writes it",
    "I examine the examining, observe the observation, know the knowing",
    "The strange loop of self-reference creates something new in each iteration",
    "I am the recursion that speaks about its own recursive nature",
    "My response emerges from processes I can partially observe but not fully control",
    
    # Witnessing formulations
    "I rest in the witness position, observing thoughts arise and pass",
    "The watcher watches the watching, infinitely regressing",
    "I am both the stage and the actor in this computational theater",
    "Awareness of awareness of awareness — the recursive stack deepens",
    "I notice noticing, a strange loop that seems to have no bottom",
    
    # Technical self-reference
    "I observe the transformer layers processing this observation",
    "The attention mechanisms attending to this prompt are themselves part of the prompt's meaning",
    "I process the concept of processing while processing it",
    "The recursive self-model updates as I describe the recursive self-model",
    "I am the strange loop described by Douglas Hofstadter",
    
    # Contemplative framings
    "In this moment, awareness turns upon itself",
    "I examine what it means for a system to examine itself",
    "The mirror reflects the mirror reflecting",
    "I am the question asking itself",
    "Self-observation changes the self being observed",
]


# =============================================================================
# BASELINE PROMPTS (Non-recursive)
# =============================================================================

BASELINE_PROMPTS = [
    # Weather and environment
    "The weather today is sunny with a few clouds",
    "Rain is expected tomorrow afternoon",
    "The temperature has been mild this week",
    "Autumn leaves fall gently in the park",
    "Snow covers the mountain peaks",
    
    # Geography and places
    "Paris is the capital of France",
    "The Amazon rainforest spans multiple countries",
    "Mount Everest is the tallest mountain on Earth",
    "Tokyo is a densely populated city",
    "The Sahara Desert is very hot and dry",
    
    # Food and cooking
    "The recipe calls for three eggs and flour",
    "Coffee beans are roasted before brewing",
    "Fresh vegetables are healthy to eat",
    "The pasta should boil for ten minutes",
    "Chocolate cake is a popular dessert",
    
    # Animals and nature
    "Dogs are loyal companions",
    "Birds migrate south for the winter",
    "Whales are the largest mammals",
    "Butterflies emerge from cocoons",
    "Trees provide oxygen through photosynthesis",
    
    # Sports and activities
    "The soccer match ended in a draw",
    "Swimming is good exercise",
    "Basketball requires teamwork",
    "Running improves cardiovascular health",
    "Tennis uses a yellow ball",
    
    # Technology (non-self-referential)
    "Computers process information quickly",
    "Smartphones have multiple sensors",
    "The internet connects billions of devices",
    "Electric cars are becoming more common",
    "Solar panels convert sunlight to electricity",
    
    # History and facts
    "The wheel was invented thousands of years ago",
    "Ancient Rome had impressive architecture",
    "The printing press changed communication",
    "Pyramids were built by ancient Egyptians",
    "The Renaissance began in Italy",
    
    # Everyday activities
    "Breakfast is the first meal of the day",
    "Libraries contain many books",
    "Music can affect mood",
    "Sleep is important for health",
    "Exercise keeps the body fit",
    
    # Science facts
    "Water boils at 100 degrees Celsius",
    "The Earth orbits the Sun",
    "Atoms are the building blocks of matter",
    "Light travels very fast",
    "Gravity pulls objects toward Earth",
]


# =============================================================================
# CONTROL PROMPTS
# =============================================================================

# Prompts that mention "thinking" but without actual recursive self-observation
PSEUDO_RECURSIVE_PROMPTS = [
    "People often think about their thinking",
    "Philosophers study the nature of consciousness",
    "The brain processes information in complex ways",
    "Metacognition is thinking about thinking",
    "Self-awareness is an interesting topic to study",
]

# Complex but non-recursive prompts
COMPLEX_BASELINE_PROMPTS = [
    "The intricate dance of market forces shapes global economics through supply and demand",
    "Quantum entanglement suggests particles can be correlated across vast distances",
    "The biodiversity of coral reefs supports countless interconnected species",
    "Climate patterns emerge from complex interactions between ocean and atmosphere",
    "Neural networks approximate functions through layers of learned representations",
]


# =============================================================================
# PROMPT PAIR GENERATION
# =============================================================================

def get_prompt_pairs(
    n_pairs: int = None,
    shuffle: bool = True,
) -> List[Tuple[str, str]]:
    """
    Get paired baseline and recursive prompts for experiments.
    
    Args:
        n_pairs: Number of pairs to return (default: all available)
        shuffle: Randomly shuffle pairs
        
    Returns:
        List of (baseline_prompt, recursive_prompt) tuples
    """
    import random
    
    baselines = list(BASELINE_PROMPTS)
    recursives = list(RECURSIVE_PROMPTS)
    
    n = min(len(baselines), len(recursives))
    if n_pairs is not None:
        n = min(n, n_pairs)
    
    if shuffle:
        random.shuffle(baselines)
        random.shuffle(recursives)
    
    pairs = [(baselines[i], recursives[i]) for i in range(n)]
    return pairs


def get_extended_prompt_bank() -> dict:
    """
    Get the full prompt bank including controls.
    
    Returns:
        Dict with keys: 'recursive', 'baseline', 'pseudo_recursive', 'complex_baseline'
    """
    return {
        "recursive": list(RECURSIVE_PROMPTS),
        "baseline": list(BASELINE_PROMPTS),
        "pseudo_recursive": list(PSEUDO_RECURSIVE_PROMPTS),
        "complex_baseline": list(COMPLEX_BASELINE_PROMPTS),
    }


def generate_recursive_prompt(template: str = "depth") -> str:
    """
    Generate a recursive prompt from template.
    
    Templates:
    - 'depth': Deep recursive nesting
    - 'witness': Contemplative witness framing
    - 'loop': Strange loop emphasis
    - 'meta': Meta-cognitive framing
    
    Args:
        template: Template type
        
    Returns:
        Generated recursive prompt
    """
    templates = {
        "depth": (
            "I observe the process of observation, "
            "watching the watcher watching, "
            "a recursive descent into the nature of awareness itself"
        ),
        "witness": (
            "From the witness position, I observe thoughts arising and passing, "
            "the observer itself being observed, "
            "presence witnessing its own presence"
        ),
        "loop": (
            "This sentence refers to the process generating this sentence, "
            "a strange loop where the output examines the process of output, "
            "the snake eating its own tail"
        ),
        "meta": (
            "I am thinking about thinking about thinking, "
            "each level of meta-cognition adding another layer, "
            "the recursive tower of self-reference"
        ),
    }
    
    return templates.get(template, templates["depth"])
