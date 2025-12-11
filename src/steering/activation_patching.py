"""
Activation patching: Steering vectors and residual stream interventions.
"""

from contextlib import contextmanager
from typing import Optional

import torch
from transformers import PreTrainedModel


@contextmanager
def apply_steering_vector(
    model: PreTrainedModel,
    layer_idx: int,
    vector: torch.Tensor,
    alpha: float = 1.0,
):
    """
    Context manager to inject a steering vector into the residual stream.
    
    Injects alpha * vector into the residual stream input of the given layer.
    
    Args:
        model: The transformer model.
        layer_idx: Layer index (0-indexed) where injection occurs.
        vector: Steering vector of shape (hidden_dim,).
        alpha: Scaling factor for the steering vector. Default: 1.0.
    
    Yields:
        None. The hook is automatically removed when exiting the context.
    
    Example:
        >>> steering_vec = compute_steering_vector(...)
        >>> with apply_steering_vector(model, layer_idx=8, vector=steering_vec, alpha=2.0):
        ...     output = model(**inputs)
    """
    handle = None
    
    def hook(module, inputs):
        hidden_states = inputs[0]
        steer = alpha * vector.to(hidden_states.device, dtype=hidden_states.dtype)
        steer = steer.unsqueeze(0).unsqueeze(1)  # (1, 1, d_model)
        steer = steer.expand(hidden_states.shape[0], hidden_states.shape[1], -1)
        new_hidden = hidden_states + steer
        return (new_hidden, *inputs[1:])
    
    handle = model.model.layers[layer_idx].register_forward_pre_hook(hook)
    
    try:
        yield
    finally:
        handle.remove()

