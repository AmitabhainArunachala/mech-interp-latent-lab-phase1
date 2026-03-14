"""
Activation patching: steering vectors for residual and MLP interventions.
"""

from contextlib import contextmanager
from typing import Optional

import torch
from transformers import PreTrainedModel


def _apply_vector_to_positions(
    hidden_states: torch.Tensor,
    vector: torch.Tensor,
    *,
    alpha: float,
    token_window: Optional[int],
) -> torch.Tensor:
    steer = alpha * vector.to(hidden_states.device, dtype=hidden_states.dtype)
    steer = steer.unsqueeze(0).unsqueeze(1)
    if token_window is None or token_window <= 0:
        steer = steer.expand(hidden_states.shape[0], hidden_states.shape[1], -1)
        return hidden_states + steer

    new_hidden = hidden_states.clone()
    window = min(int(token_window), hidden_states.shape[1])
    steer = steer.expand(hidden_states.shape[0], window, -1)
    new_hidden[:, -window:, :] = new_hidden[:, -window:, :] + steer
    return new_hidden


@contextmanager
def apply_steering_vector(
    model: PreTrainedModel,
    layer_idx: int,
    vector: torch.Tensor,
    alpha: float = 1.0,
    token_window: Optional[int] = None,
):
    """
    Context manager to inject a steering vector into the residual stream.
    
    Injects alpha * vector into the residual stream input of the given layer.
    
    Args:
        model: The transformer model.
        layer_idx: Layer index (0-indexed) where injection occurs.
        vector: Steering vector of shape (hidden_dim,).
        alpha: Scaling factor for the steering vector. Default: 1.0.
        token_window: If set, steer only the final N token positions.
    
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
        new_hidden = _apply_vector_to_positions(
            hidden_states,
            vector,
            alpha=alpha,
            token_window=token_window,
        )
        return (new_hidden, *inputs[1:])
    
    handle = model.model.layers[layer_idx].register_forward_pre_hook(hook)
    
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def apply_mlp_steering_vector(
    model: PreTrainedModel,
    layer_idx: int,
    vector: torch.Tensor,
    alpha: float = 1.0,
    token_window: Optional[int] = None,
):
    """
    Inject a steering vector into a layer's MLP output.

    Args:
        model: The transformer model.
        layer_idx: Layer index (0-indexed) where injection occurs.
        vector: Steering vector of shape (hidden_dim,).
        alpha: Scaling factor for the steering vector.
        token_window: If set, steer only the final N token positions.
    """
    handle = None

    def hook(_module, _inputs, output):
        out_tensor = output[0] if isinstance(output, tuple) else output
        steered = _apply_vector_to_positions(
            out_tensor,
            vector,
            alpha=alpha,
            token_window=token_window,
        )
        if isinstance(output, tuple):
            return (steered,) + output[1:]
        return steered

    handle = model.model.layers[layer_idx].mlp.register_forward_hook(hook)

    try:
        yield
    finally:
        handle.remove()
