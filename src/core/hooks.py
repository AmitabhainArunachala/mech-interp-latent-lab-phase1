"""
Standardized hook context managers for model interventions.

The Intervention Invariant:
- Use Python context managers (`with hook(...):`) for all model modifications.
- Never leave a hook attached after a function returns.
"""

from contextlib import contextmanager
from typing import Optional

import torch
from transformers import PreTrainedModel


@contextmanager
def capture_v_projection(
    model: PreTrainedModel,
    layer_idx: int,
    storage_list: Optional[list] = None,
):
    """
    Context manager to capture V-projection outputs at a specific layer.
    
    Args:
        model: The transformer model.
        layer_idx: Layer index (0-indexed).
        storage_list: Optional list to append captured tensor. If None, returns via context.
    
    Yields:
        Dictionary with "v" key containing the captured tensor.
    
    Example:
        >>> with capture_v_projection(model, layer_idx=27) as storage:
        ...     model(**inputs)
        >>> v_tensor = storage["v"]
    """
    storage = {"v": None}
    
    def hook_fn(module, inp, out):
        storage["v"] = out.detach()
        return out
    
    layer = model.model.layers[layer_idx].self_attn
    handle = layer.v_proj.register_forward_hook(hook_fn)
    
    try:
        yield storage
        if storage_list is not None:
            storage_list.append(storage["v"])
    finally:
        handle.remove()


@contextmanager
def capture_hidden_states(
    model: PreTrainedModel,
    layer_idx: int,
    storage_list: Optional[list] = None,
):
    """
    Context manager to capture hidden states at a specific layer.
    
    Args:
        model: The transformer model.
        layer_idx: Layer index (0-indexed).
        storage_list: Optional list to append captured tensor.
    
    Yields:
        Dictionary with "hidden" key containing the captured tensor.
    """
    storage = {"hidden": None}
    
    def hook_fn(module, inp, out):
        # For most transformers, hidden states are in outputs[0]
        if isinstance(out, tuple):
            storage["hidden"] = out[0].detach()
        else:
            storage["hidden"] = out.detach()
        return out
    
    # Hook into the layer output (post-attention + MLP)
    layer = model.model.layers[layer_idx]
    handle = layer.register_forward_hook(hook_fn)
    
    try:
        yield storage
        if storage_list is not None:
            storage_list.append(storage["hidden"])
    finally:
        handle.remove()

