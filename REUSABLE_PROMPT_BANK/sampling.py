# === SAMPLING UTILITIES ===
# Functions for balanced prompt selection in experiments

import random
from typing import Dict, List, Tuple, Optional
import random

def get_all_prompts() -> Dict:
    """Load all prompts from all modules into a single dictionary."""
    from .dose_response import dose_response_prompts
    from .baselines import baseline_prompts
    from .confounds import confound_prompts
    from .generality import generality_prompts
    from .kill_switch import kill_switch_prompts
    
    all_prompts = {}
    all_prompts.update(dose_response_prompts)
    all_prompts.update(baseline_prompts)
    all_prompts.update(confound_prompts)
    all_prompts.update(generality_prompts)
    all_prompts.update(kill_switch_prompts)
    
    return all_prompts


def get_prompts_by_pillar(pillar: str) -> Dict:
    """Get all prompts belonging to a specific pillar."""
    all_prompts = get_all_prompts()
    return {k: v for k, v in all_prompts.items() if v.get("pillar") == pillar}


def get_prompts_by_group(group: str) -> Dict:
    """Get all prompts belonging to a specific group."""
    all_prompts = get_all_prompts()
    return {k: v for k, v in all_prompts.items() if v.get("group") == group}


def get_recursive_prompts() -> Dict:
    """Get all recursive prompts (dose_response pillar)."""
    return get_prompts_by_pillar("dose_response")


def get_baseline_prompts() -> Dict:
    """Get all baseline prompts."""
    return get_prompts_by_pillar("baselines")


def get_balanced_pairs(
    n_pairs: int = 30,
    recursive_groups: Optional[List[str]] = None,
    baseline_groups: Optional[List[str]] = None,
    seed: int = 42
) -> List[Tuple[Dict, Dict]]:
    """
    Generate balanced recursive/baseline prompt pairs for experiments.
    
    Args:
        n_pairs: Number of pairs to generate
        recursive_groups: List of recursive groups to sample from (default: L3-L5)
        baseline_groups: List of baseline groups to sample from
        seed: Random seed for reproducibility
    
    Returns:
        List of (recursive_prompt, baseline_prompt) tuples
    """
    random.seed(seed)
    
    # Default to strongest recursive prompts
    if recursive_groups is None:
        recursive_groups = ["L3_deeper", "L4_full", "L5_refined"]
    
    # Default to all baseline groups except personal (unknowable)
    if baseline_groups is None:
        baseline_groups = ["baseline_math", "baseline_factual", "baseline_creative"]
    
    all_prompts = get_all_prompts()
    
    # Filter prompts
    recursive = [(k, v) for k, v in all_prompts.items() if v.get("group") in recursive_groups]
    baseline = [(k, v) for k, v in all_prompts.items() if v.get("group") in baseline_groups]
    
    # Sample
    n_rec = min(n_pairs, len(recursive))
    n_base = min(n_pairs, len(baseline))
    
    sampled_rec = random.sample(recursive, n_rec)
    sampled_base = random.sample(baseline, n_base)
    
    # Pair up
    pairs = []
    for i in range(min(n_rec, n_base)):
        rec_key, rec_val = sampled_rec[i]
        base_key, base_val = sampled_base[i]
        pairs.append((
            {"key": rec_key, **rec_val},
            {"key": base_key, **base_val}
        ))
    
    return pairs


def get_dose_response_set(n_per_level: int = 10, seed: int = 42) -> Dict[int, List[Dict]]:
    """
    Get prompts across all 5 recursion levels for dose-response analysis.
    
    Args:
        n_per_level: Number of prompts per level
        seed: Random seed
    
    Returns:
        Dict mapping level (1-5) to list of prompts
    """
    random.seed(seed)
    
    from .dose_response import dose_response_prompts
    
    result = {1: [], 2: [], 3: [], 4: [], 5: []}
    
    for level in range(1, 6):
        level_prompts = [(k, v) for k, v in dose_response_prompts.items() 
                        if v.get("level") == level]
        n_sample = min(n_per_level, len(level_prompts))
        sampled = random.sample(level_prompts, n_sample)
        result[level] = [{"key": k, **v} for k, v in sampled]
    
    return result


def get_control_set() -> Dict[str, List[Dict]]:
    """
    Get all control conditions for validity checking.
    
    Returns:
        Dict with keys: 'pure_repetition', 'ood_weird', 'surreal_1p', 'surreal_3p'
    """
    from .kill_switch import kill_switch_prompts
    
    controls = {
        "pure_repetition": [],
        "ood_weird": [],
        "surreal_first_person": [],
        "surreal_third_person": [],
    }
    
    for k, v in kill_switch_prompts.items():
        group = v.get("group")
        if group in controls:
            controls[group].append({"key": k, **v})
    
    return controls


def get_length_matched_pairs(
    target_length: int = 50,
    tolerance: int = 10,
    n_pairs: int = 20,
    seed: int = 42
) -> List[Tuple[Dict, Dict]]:
    """
    Get recursive/baseline pairs matched on approximate token length.
    
    Args:
        target_length: Target token count (approximate)
        tolerance: Allowed deviation from target
        n_pairs: Number of pairs
        seed: Random seed
    
    Returns:
        List of (recursive, baseline) prompt pairs
    """
    random.seed(seed)
    all_prompts = get_all_prompts()
    
    # Estimate token count (rough: words * 1.3)
    def est_tokens(text):
        return int(len(text.split()) * 1.3)
    
    recursive = [(k, v) for k, v in all_prompts.items() 
                 if v.get("pillar") == "dose_response"
                 and target_length - tolerance <= est_tokens(v["text"]) <= target_length + tolerance]
    
    baseline = [(k, v) for k, v in all_prompts.items()
                if v.get("pillar") == "baselines"
                and target_length - tolerance <= est_tokens(v["text"]) <= target_length + tolerance]
    
    # If not enough matches, relax tolerance
    if len(recursive) < n_pairs or len(baseline) < n_pairs:
        print(f"Warning: Only found {len(recursive)} recursive and {len(baseline)} baseline prompts in length range")
    
    n_sample = min(n_pairs, len(recursive), len(baseline))
    sampled_rec = random.sample(recursive, n_sample) if n_sample > 0 else []
    sampled_base = random.sample(baseline, n_sample) if n_sample > 0 else []
    
    pairs = []
    for i in range(len(sampled_rec)):
        rec_key, rec_val = sampled_rec[i]
        base_key, base_val = sampled_base[i]
        pairs.append((
            {"key": rec_key, "est_tokens": est_tokens(rec_val["text"]), **rec_val},
            {"key": base_key, "est_tokens": est_tokens(base_val["text"]), **base_val}
        ))
    
    return pairs


def get_prompts_by_type(prompt_type: str, limit: int = 20, seed: int = 42) -> List[str]:
    """
    Get prompts filtered by structural type (completion/instructional/creative/recursive).
    
    Args:
        prompt_type: One of "completion", "instructional", "creative", "recursive"
        limit: Maximum number of prompts to return
        seed: Random seed for reproducibility
    
    Returns:
        List of prompt text strings
    """
    random.seed(seed)
    
    all_prompts = get_all_prompts()
    filtered = [
        v["text"] for k, v in all_prompts.items()
        if v.get("type") == prompt_type
    ]
    
    random.shuffle(filtered)
    return filtered[:limit]


def get_validated_pairs(n_pairs: int = 20, seed: int = 42) -> List[Tuple[str, str]]:
    """
    Get the DEC8-validated recursive/baseline pairs that are known to work.
    
    Uses the DEC8 working prompts:
    - Recursive: L3_deeper_DEC8_01 through L3_deeper_DEC8_05
    - Baseline: baseline_instructional_01 through baseline_instructional_05
    
    Args:
        n_pairs: Number of pairs to return (max 5, as there are 5 validated pairs)
        seed: Random seed (not used since we return fixed pairs)
    
    Returns:
        List of (recursive_prompt, baseline_prompt) tuples
    """
    from .dose_response import dose_response_prompts
    from .baselines import baseline_prompts
    
    # Get DEC8 validated prompts
    recursive_keys = [f"L3_deeper_DEC8_{i:02d}" for i in range(1, 6)]
    baseline_keys = [f"baseline_instructional_{i:02d}" for i in range(1, 6)]
    
    pairs = []
    for i in range(min(n_pairs, 5)):
        rec_key = recursive_keys[i]
        base_key = baseline_keys[i]
        
        if rec_key in dose_response_prompts and base_key in baseline_prompts:
            pairs.append((
                dose_response_prompts[rec_key]["text"],
                baseline_prompts[base_key]["text"]
            ))
    
    return pairs


# Quick stats function
def print_bank_stats():
    """Print statistics about the prompt bank."""
    all_prompts = get_all_prompts()
    
    print(f"\n{'='*60}")
    print(f"REUSABLE PROMPT BANK STATISTICS")
    print(f"{'='*60}")
    print(f"Total prompts: {len(all_prompts)}")
    
    # By pillar
    pillars = {}
    for k, v in all_prompts.items():
        p = v.get("pillar", "unknown")
        pillars[p] = pillars.get(p, 0) + 1
    
    print(f"\nBy pillar:")
    for p, c in sorted(pillars.items()):
        print(f"  {p}: {c}")
    
    # By group
    groups = {}
    for k, v in all_prompts.items():
        g = v.get("group", "unknown")
        groups[g] = groups.get(g, 0) + 1
    
    print(f"\nBy group:")
    for g, c in sorted(groups.items()):
        print(f"  {g}: {c}")
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    print_bank_stats()


