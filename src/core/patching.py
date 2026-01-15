"""
Persistent activation patching utilities.

Key class: PersistentVPatcher - maintains V_PROJ patching during generation.
Based on Dec 12 breakthrough: KV cache + persistent V_PROJ = 100% behavior transfer.
"""

from __future__ import annotations

from typing import Optional

import torch


class PersistentVPatcher:
    """
    Patches V_PROJ output at a specific layer during generation.
    
    Maintains geometric signature throughout generation by replacing
    computed V_PROJ activations with pre-extracted recursive V_PROJ activations.
    
    Usage:
        # Extract V_PROJ activation from recursive prompt
        v_activation = extract_v_activation(model, tokenizer, recursive_prompt, layer=27)
        
        # Create patcher
        patcher = PersistentVPatcher(model, v_activation)
        patcher.register(layer_idx=27)
        
        # Generate (patcher is active)
        generated = model.generate(...)
        
        # Clean up
        patcher.remove()
    """
    
    def __init__(self, model, v_activation: torch.Tensor):
        """
        Initialize patcher with V_PROJ activation to inject.
        
        Args:
            model: The transformer model
            v_activation: V_PROJ output from recursive prompt
                         Shape: (seq_len, hidden_dim) or (batch, seq_len, hidden_dim)
        """
        self.model = model
        # Ensure v_activation is 2D: (seq_len, hidden_dim)
        if v_activation.dim() == 3:
            v_activation = v_activation[0]  # Remove batch dimension
        self.v_activation = v_activation.detach()  # Shape: (seq_len, hidden_dim)
        self.handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self.layer_idx: Optional[int] = None
    
    def register(self, layer_idx: int):
        """
        Register forward hook at specified layer to patch V_PROJ output.
        
        Args:
            layer_idx: Layer index (0-indexed, e.g., 27 for L27)
        """
        if self.handle is not None:
            raise RuntimeError("Patcher already registered. Call remove() first.")
        
        self.layer_idx = layer_idx
        layer = self.model.model.layers[layer_idx].self_attn
        
        def hook_fn(module, inp, out):
            """
            Hook function that replaces V_PROJ output with patched activation.
            
            Args:
                module: The v_proj module
                inp: Input to v_proj (hidden states)
                out: Output from v_proj (batch, seq_len, hidden_dim)
            
            Returns:
                Patched output with same shape as out
            """
            # out shape: (batch, seq_len, hidden_dim)
            batch, seq_len, hidden_dim = out.shape
            
            # Dec 12 breakthrough: patch last 16 tokens (window_size)
            # This matches the R_V metric window and maintains geometric signature
            window_size = 16
            v_len = min(seq_len, self.v_activation.shape[0], window_size)
            
            # Use last window_size tokens from v_activation
            # v_activation[-window_size:] shape: (window_size, hidden_dim) or less
            v_slice = self.v_activation[-v_len:, :]  # (v_len, hidden_dim)
            
            # Expand to (batch, v_len, hidden_dim)
            patched_v = v_slice.unsqueeze(0)  # (1, v_len, hidden_dim)
            
            # If batch > 1, repeat
            if batch > 1:
                patched_v = patched_v.repeat(batch, 1, 1)
            
            # Replace the last v_len tokens (maintains geometric signature)
            out_patched = out.clone()
            out_patched[:, -v_len:, :] = patched_v[:, :v_len, :].to(
                out_patched.device, dtype=out_patched.dtype
            )
            
            return out_patched
        
        self.handle = layer.v_proj.register_forward_hook(hook_fn)
    
    def remove(self):
        """Remove the forward hook."""
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
            self.layer_idx = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - automatically removes hook."""
        self.remove()


def extract_v_activation(
    model,
    tokenizer,
    prompt: str,
    layer_idx: int,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Extract V_PROJ activation at specified layer for a prompt.
    
    Args:
        model: The transformer model
        tokenizer: The tokenizer
        prompt: Input prompt text
        layer_idx: Layer index (0-indexed)
        device: Device to run on
    
    Returns:
        V_PROJ activation tensor, shape: (seq_len, hidden_dim)
    """
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    
    v_activation = None
    
    def capture_hook(module, inp, out):
        nonlocal v_activation
        # out shape: (batch, seq_len, hidden_dim)
        v_activation = out[0].detach()  # Remove batch dimension: (seq_len, hidden_dim)
        return out
    
    layer = model.model.layers[layer_idx].self_attn
    handle = layer.v_proj.register_forward_hook(capture_hook)
    
    try:
        with torch.no_grad():
            _ = model(**inputs, use_cache=False)
    finally:
        handle.remove()
    
    if v_activation is None:
        raise RuntimeError(f"Failed to capture V_PROJ activation at layer {layer_idx}")
    
    return v_activation


class PersistentResidualPatcher:
    """
    Patches residual stream input at a specific layer during generation.
    
    Based on Dec 12 breakthrough: L18 RESIDUAL + L27 V_PROJ = 100% behavior transfer.
    """
    
    def __init__(self, model, residual_activation: torch.Tensor):
        """
        Initialize patcher with residual activation to inject.
        
        Args:
            model: The transformer model
            residual_activation: Residual stream input from recursive prompt
                               Shape: (seq_len, hidden_dim) or (batch, seq_len, hidden_dim)
        """
        self.model = model
        # Ensure residual_activation is 2D: (seq_len, hidden_dim)
        if residual_activation.dim() == 3:
            residual_activation = residual_activation[0]  # Remove batch dimension
        self.residual_activation = residual_activation.detach()  # Shape: (seq_len, hidden_dim)
        self.handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self.layer_idx: Optional[int] = None
    
    def register(self, layer_idx: int):
        """
        Register forward pre-hook at specified layer to patch residual input.
        
        Args:
            layer_idx: Layer index (0-indexed, e.g., 18 for L18)
        """
        if self.handle is not None:
            raise RuntimeError("Patcher already registered. Call remove() first.")
        
        self.layer_idx = layer_idx
        layer = self.model.model.layers[layer_idx]
        
        def hook_fn(module, inp):
            """
            Pre-hook function that patches residual stream input.
            
            Args:
                module: The transformer layer
                inp: Input tuple (hidden_states, ...)
            
            Returns:
                Modified input tuple with patched hidden_states
            """
            # inp[0] is hidden_states: (batch, seq_len, hidden_dim)
            hidden_states = inp[0]
            batch, seq_len, hidden_dim = hidden_states.shape
            
            # Dec 12 breakthrough: patch last 16 tokens (window_size)
            window_size = 16
            r_len = min(seq_len, self.residual_activation.shape[0], window_size)
            
            # Use last window_size tokens from residual_activation
            r_slice = self.residual_activation[-r_len:, :]  # (r_len, hidden_dim)
            
            # Expand to (batch, r_len, hidden_dim)
            patched_r = r_slice.unsqueeze(0)  # (1, r_len, hidden_dim)
            
            # If batch > 1, repeat
            if batch > 1:
                patched_r = patched_r.repeat(batch, 1, 1)
            
            # Replace the last r_len tokens
            patched_hidden = hidden_states.clone()
            patched_hidden[:, -r_len:, :] = patched_r[:, :r_len, :].to(
                patched_hidden.device, dtype=patched_hidden.dtype
            )
            
            # Return modified input tuple
            return (patched_hidden,) + inp[1:]
        
        self.handle = layer.register_forward_pre_hook(hook_fn)
    
    def remove(self):
        """Remove the forward pre-hook."""
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
            self.layer_idx = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - automatically removes hook."""
        self.remove()


def extract_residual_activation(
    model,
    tokenizer,
    prompt: str,
    layer_idx: int,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Extract residual stream input at specified layer for a prompt.
    
    Args:
        model: The transformer model
        tokenizer: The tokenizer
        prompt: Input prompt text
        layer_idx: Layer index (0-indexed)
        device: Device to run on
    
    Returns:
        Residual stream input tensor, shape: (seq_len, hidden_dim)
    """
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    
    residual_activation = None
    
    def capture_hook(module, inp):
        nonlocal residual_activation
        # inp[0] is hidden_states: (batch, seq_len, hidden_dim)
        residual_activation = inp[0][0].detach()  # Remove batch dimension: (seq_len, hidden_dim)
        return inp
    
    layer = model.model.layers[layer_idx]
    handle = layer.register_forward_pre_hook(capture_hook)
    
    try:
        with torch.no_grad():
            _ = model(**inputs, use_cache=False)
    finally:
        handle.remove()
    
    if residual_activation is None:
        raise RuntimeError(f"Failed to capture residual activation at layer {layer_idx}")
    
    return residual_activation


__all__ = [
    "PersistentVPatcher",
    "PersistentResidualPatcher",
    "extract_v_activation",
    "extract_residual_activation",
]

