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
def capture_attention_patterns(
    model: PreTrainedModel,
    layer_idx: int,
):
    """
    Context manager to capture attention weights at a specific layer.
    
    NOTE: Requires model forward pass with output_attentions=True.
    
    Args:
        model: The transformer model.
        layer_idx: Layer index (0-indexed).
    
    Yields:
        Dictionary with "attn_weights" key containing tensor of shape
        (batch, num_heads, seq_len, seq_len).
    
    Example:
        >>> with capture_attention_patterns(model, layer_idx=27) as storage:
        ...     model(**inputs, output_attentions=True)
        >>> attn = storage["attn_weights"]  # (1, 32, seq, seq)
    """
    storage = {"attn_weights": None}
    
    def hook_fn(module, inp, out):
        # Mistral self_attn returns (attn_output, attn_weights, past_kv) when output_attentions=True
        if isinstance(out, tuple) and len(out) > 1 and out[1] is not None:
            storage["attn_weights"] = out[1].detach()
        return out
    
    layer = model.model.layers[layer_idx].self_attn
    handle = layer.register_forward_hook(hook_fn)
    
    try:
        yield storage
    finally:
        handle.remove()


@contextmanager
def capture_head_output(
    model: PreTrainedModel,
    layer_idx: int,
    head_idx: int,
):
    """
    Context manager to capture a specific attention head's contribution.
    
    Extracts the per-head output by reshaping the attention output tensor.
    
    Args:
        model: The transformer model.
        layer_idx: Layer index (0-indexed).
        head_idx: Head index (0-indexed).
    
    Yields:
        Dictionary with "head_output" key containing tensor of shape
        (batch, seq_len, head_dim).
    """
    storage = {"head_output": None}
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads
    
    def hook_fn(module, inp, out):
        # out[0] is attn_output: (batch, seq, hidden_size)
        attn_output = out[0].detach()
        batch, seq, hidden = attn_output.shape
        
        # Reshape to (batch, seq, num_heads, head_dim)
        reshaped = attn_output.view(batch, seq, num_heads, head_dim)
        storage["head_output"] = reshaped[:, :, head_idx, :].clone()
        return out
    
    layer = model.model.layers[layer_idx].self_attn
    handle = layer.register_forward_hook(hook_fn)
    
    try:
        yield storage
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

