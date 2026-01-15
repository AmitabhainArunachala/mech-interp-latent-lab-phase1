"""
Logit capture utilities for capturing model logits during generation.

Provides context managers to hook into model forward pass and capture
logits at each generation step for Mode Score M computation.
"""

from contextlib import contextmanager
from typing import List, Optional
import torch
from transformers import PreTrainedModel


@contextmanager
def capture_logits(
    model: PreTrainedModel,
    storage_list: Optional[List[torch.Tensor]] = None,
):
    """
    Context manager to capture logits during model forward pass.
    
    Captures logits from the language model head (lm_head) output.
    Works with both single forward passes and generation loops.
    
    Args:
        model: The transformer model
        storage_list: Optional list to append captured logits.
                     If None, returns via context manager yield.
    
    Yields:
        List of captured logit tensors, each of shape (batch, seq_len, vocab_size)
        or (seq_len, vocab_size) if batch=1
    
    Example:
        >>> logits_list = []
        >>> with capture_logits(model, logits_list):
        ...     outputs = model(**inputs)
        >>> # logits_list now contains captured logits
    
    Note:
        For generation loops, logits are captured at each step.
        For single forward passes, logits are captured once.
    """
    captured_logits = []
    
    def hook_fn(module, inp, out):
        """
        Hook function to capture logits from lm_head output.
        
        For most models, lm_head outputs logits directly.
        For some models, outputs may be a tuple (logits, ...).
        """
        if isinstance(out, tuple):
            # Some models return (logits, ...)
            logits = out[0]
        else:
            logits = out
        
        # Detach to avoid gradient tracking
        captured_logits.append(logits.detach())
        return out
    
    # Hook into the language model head
    # Most models have model.lm_head, but some have different names
    if hasattr(model, 'lm_head'):
        handle = model.lm_head.register_forward_hook(hook_fn)
    elif hasattr(model, 'embed_out'):  # Some models use embed_out
        handle = model.embed_out.register_forward_hook(hook_fn)
    elif hasattr(model, 'head'):  # Fallback
        handle = model.head.register_forward_hook(hook_fn)
    else:
        raise ValueError(
            f"Could not find language model head. "
            f"Model has attributes: {[attr for attr in dir(model) if not attr.startswith('_')]}"
        )
    
    try:
        yield captured_logits
        if storage_list is not None:
            storage_list.extend(captured_logits)
    finally:
        handle.remove()


@contextmanager
def capture_logits_during_generation(
    model: PreTrainedModel,
    storage_list: Optional[List[torch.Tensor]] = None,
    max_steps: Optional[int] = None,
):
    """
    Context manager specifically for capturing logits during token-by-token generation.
    
    This is useful when generating with past_key_values (KV cache) where
    we generate one token at a time and want to capture logits at each step.
    
    Args:
        model: The transformer model
        storage_list: Optional list to append captured logits
        max_steps: Maximum number of generation steps to capture (None = all steps)
    
    Yields:
        List of captured logit tensors, one per generation step
    
    Example:
        >>> logits_list = []
        >>> with capture_logits_during_generation(model, logits_list, max_steps=10):
        ...     for _ in range(10):
        ...         outputs = model(input_ids=current_ids, past_key_values=past_kv)
        ...         # logits captured automatically
    """
    step_count = [0]  # Use list to allow modification in closure
    
    captured_logits = []
    
    def hook_fn(module, inp, out):
        if max_steps is not None and step_count[0] >= max_steps:
            return out
        
        if isinstance(out, tuple):
            logits = out[0]
        else:
            logits = out
        
        # For generation, we typically only care about the last token's logits
        # Shape: (batch, vocab_size) or (batch, seq_len, vocab_size)
        if logits.dim() == 3:
            # Take last token
            logits = logits[:, -1:, :]  # (batch, 1, vocab_size)
        
        captured_logits.append(logits.detach())
        step_count[0] += 1
        return out
    
    if hasattr(model, 'lm_head'):
        handle = model.lm_head.register_forward_hook(hook_fn)
    elif hasattr(model, 'embed_out'):
        handle = model.embed_out.register_forward_hook(hook_fn)
    elif hasattr(model, 'head'):
        handle = model.head.register_forward_hook(hook_fn)
    else:
        raise ValueError("Could not find language model head")
    
    try:
        yield captured_logits
        if storage_list is not None:
            storage_list.extend(captured_logits)
    finally:
        handle.remove()


def extract_logits_from_outputs(outputs) -> torch.Tensor:
    """
    Extract logits from model outputs, handling different output formats.
    
    Args:
        outputs: Model outputs (can be tuple, dict, or tensor)
    
    Returns:
        Logits tensor of shape (batch, seq_len, vocab_size) or (seq_len, vocab_size)
    """
    if isinstance(outputs, torch.Tensor):
        return outputs
    elif isinstance(outputs, tuple):
        # First element is usually logits
        return outputs[0]
    elif isinstance(outputs, dict):
        if 'logits' in outputs:
            return outputs['logits']
        elif 'logit' in outputs:
            return outputs['logit']
        else:
            raise ValueError(f"Could not find logits in outputs dict. Keys: {outputs.keys()}")
    else:
        raise ValueError(f"Unknown output type: {type(outputs)}")


__all__ = [
    "capture_logits",
    "capture_logits_during_generation",
    "extract_logits_from_outputs",
]







