"""
Model Physics Registry.

Centralizes the "physical constants" of each model architecture for our experiments.
This prevents magic numbers (e.g. 'layer 27', 'head 2') from being hardcoded in pipelines.

Usage:
    from src.core.model_physics import get_model_physics
    physics = get_model_physics("mistralai/Mistral-7B-v0.1")
    target_layer = physics.late_layer
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class ModelPhysics:
    name: str
    # Layers
    early_layer: int  # Where entry ramp ends (~15% depth)
    late_layer: int   # Where contraction peaks (~85% depth)
    
    # Attention Heads (for ablation/intervention)
    # These are specific to the "Recursive Circuit" if known
    suppressor_heads: List[Tuple[int, int]]  # [(layer, head), ...]
    amplifier_heads: List[Tuple[int, int]]
    
    # Architecture details (often fetched from config, but overrides allowed here)
    num_kv_heads: Optional[int] = None
    
    # Window size recommendation (native to the effect in this model)
    default_window: int = 16


# The Registry
_REGISTRY = {
    "mistralai/Mistral-7B-v0.1": ModelPhysics(
        name="mistralai/Mistral-7B-v0.1",
        early_layer=5,
        late_layer=27,
        suppressor_heads=[(27, 18), (27, 26)], # Note: GQA aliasing applies
        amplifier_heads=[(27, 2), (27, 10)],
        default_window=16
    ),
    "mistralai/Mistral-7B-Instruct-v0.1": ModelPhysics(
        name="mistralai/Mistral-7B-Instruct-v0.1",
        early_layer=5,
        late_layer=27,
        suppressor_heads=[], # To be discovered
        amplifier_heads=[],
        default_window=16
    ),
    # Pythia scaling (approximate depth mapping)
    "EleutherAI/pythia-6.9b": ModelPhysics(
        name="EleutherAI/pythia-6.9b",
        early_layer=5, # 32 layers total, similar to Mistral
        late_layer=27,
        suppressor_heads=[],
        amplifier_heads=[],
        default_window=16
    ),
    "meta-llama/Meta-Llama-3-8B": ModelPhysics(
        name="meta-llama/Meta-Llama-3-8B",
        early_layer=5,
        late_layer=28, # 32 layers, maybe deeper peak?
        suppressor_heads=[],
        amplifier_heads=[],
        default_window=16
    )
}

def get_model_physics(model_name: str) -> ModelPhysics:
    """
    Get the physics constants for a model.
    Falls back to Mistral-7B defaults if unknown, with a warning.
    """
    if model_name in _REGISTRY:
        return _REGISTRY[model_name]
    
    # Fallback / Heuristic
    # If we don't know the model, we assume a standard 32-layer transformer
    # and map 15% / 85% depth.
    print(f"⚠️  Warning: Model {model_name} not in physics registry. Using heuristics.")
    
    return ModelPhysics(
        name=model_name,
        early_layer=5,
        late_layer=27, # Safe default for 7B class
        suppressor_heads=[],
        amplifier_heads=[],
        default_window=16
    )









