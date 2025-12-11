"""
Steering module: Activation patching, KV caching, and intervention logic.

The Intervention Invariant:
- Use Python context managers for all model modifications.
- Never leave a hook attached after a function returns.
- KV Cache patching must respect the DynamicCache structure.
"""

from .activation_patching import apply_steering_vector
from .kv_cache import (
    capture_past_key_values,
    extract_kv_list,
    generate_with_kv,
    mix_kv_to_dynamic_cache,
)

__all__ = [
    "apply_steering_vector",
    "capture_past_key_values",
    "extract_kv_list",
    "generate_with_kv",
    "mix_kv_to_dynamic_cache",
]

