# === REUSABLE PROMPT BANK v2.0 ===
# Modular, extensible prompt bank for mechanistic interpretability research
# Maintains backward compatibility with n300_mistral_test_prompt_bank.py

"""
Usage:
    from REUSABLE_PROMPT_BANK import get_all_prompts, get_balanced_pairs, get_dose_response_set
    
    # Get everything
    all_prompts = get_all_prompts()
    
    # Get balanced recursive/baseline pairs for experiment
    pairs = get_balanced_pairs(n_pairs=30, seed=42)
    
    # Get dose-response ladder
    dose_set = get_dose_response_set(n_per_level=10)
    
    # Get specific controls
    from REUSABLE_PROMPT_BANK import kill_switch, confounds
"""

from .dose_response import dose_response_prompts
from .baselines import baseline_prompts
from .confounds import confound_prompts
from .generality import generality_prompts
from .kill_switch import kill_switch_prompts
from .sampling import (
    get_all_prompts,
    get_balanced_pairs,
    get_dose_response_set,
    get_control_set,
    get_prompts_by_group,
    get_prompts_by_pillar,
)

__version__ = "2.0.0"
__all__ = [
    "get_all_prompts",
    "get_balanced_pairs", 
    "get_dose_response_set",
    "get_control_set",
    "get_prompts_by_group",
    "get_prompts_by_pillar",
    "dose_response_prompts",
    "baseline_prompts",
    "confound_prompts",
    "generality_prompts",
    "kill_switch_prompts",
]

print(f"REUSABLE_PROMPT_BANK v{__version__} loaded")


