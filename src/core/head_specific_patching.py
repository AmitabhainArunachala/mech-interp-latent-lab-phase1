"""Head-specific patching utilities for surgical interventions."""
from __future__ import annotations

from typing import Optional
import torch


class HeadSpecificVPatcher:
    """
    Patches V_PROJ output at specific attention heads only.
    
    For Mistral-7B:
    - 32 attention heads
    - 128 dims per head
    - Total hidden_dim = 4096
    - Head H_i spans dims [i*128 : (i+1)*128]
    
    Usage:
        # Extract V_PROJ activation from recursive prompt
        v_activation = extract_v_activation(model, tokenizer, recursive_prompt, layer=27)
        
        # Create patcher for H18 and H26 only
        patcher = HeadSpecificVPatcher(model, v_activation, target_heads=[18, 26])
        patcher.register(layer_idx=27)
        
        # Generate (only H18+H26 are patched)
        generated = model.generate(...)
        
        # Clean up
        patcher.remove()
    """
    
    def __init__(
        self,
        model,
        v_activation: torch.Tensor,
        target_heads: list[int],
        window_size: int = 16,
    ):
        """
        Initialize patcher with V_PROJ activation and target heads.
        
        Args:
            model: The transformer model
            v_activation: V_PROJ output from recursive prompt
                         Shape: (seq_len, hidden_dim) or (batch, seq_len, hidden_dim)
            target_heads: List of head indices to patch (0-31 for Mistral-7B)
            window_size: Number of tokens to patch (last N tokens)
        """
        self.model = model
        # Ensure v_activation is 2D: (seq_len, hidden_dim)
        if v_activation.dim() == 3:
            v_activation = v_activation[0]  # Remove batch dimension
        self.v_activation = v_activation.detach()  # Shape: (seq_len, hidden_dim)
        self.target_heads = target_heads
        self.window_size = window_size
        self.handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self.layer_idx: Optional[int] = None
        
        # Mistral-7B: 32 heads, 128 dims per head, 4096 total
        self.num_heads = 32
        self.head_dim = 128
        self.hidden_dim = 4096
        
        # Compute dim ranges for each target head
        self.target_dims = []
        for head_idx in target_heads:
            start_dim = head_idx * self.head_dim
            end_dim = (head_idx + 1) * self.head_dim
            self.target_dims.append((start_dim, end_dim))
    
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
            Hook function that patches ONLY target heads' V_PROJ output.
            
            Args:
                module: The v_proj module
                inp: Input to v_proj (hidden states)
                out: Output from v_proj (batch, seq_len, hidden_dim)
            
            Returns:
                Patched output with same shape as out
            """
            # out shape: (batch, seq_len, hidden_dim)
            batch, seq_len, hidden_dim = out.shape
            
            # Use last window_size tokens from v_activation
            v_len = min(seq_len, self.v_activation.shape[0], self.window_size)
            v_slice = self.v_activation[-v_len:, :]  # (v_len, hidden_dim)
            
            # Clone output to avoid in-place modification
            out_patched = out.clone()
            
            # Patch ONLY the target head dimensions
            for start_dim, end_dim in self.target_dims:
                # Extract target head slice from recursive V_PROJ
                v_head_slice = v_slice[:, start_dim:end_dim]  # (v_len, head_dim)
                
                # Expand to batch dimension
                v_head_batch = v_head_slice.unsqueeze(0)  # (1, v_len, head_dim)
                if batch > 1:
                    v_head_batch = v_head_batch.repeat(batch, 1, 1)
                
                # Replace ONLY this head's dimensions in the output
                out_patched[:, -v_len:, start_dim:end_dim] = v_head_batch[:, :v_len, :].to(
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


class HeadSpecificSteeringPatcher:
    """
    Applies steering vector ONLY to specific attention heads' V_PROJ output.
    
    Usage:
        # Compute full steering vector
        steering_vector = compute_steering_vector(...)  # (4096,)
        
        # Create patcher for H18 and H26 only
        patcher = HeadSpecificSteeringPatcher(
            model, steering_vector, target_heads=[18, 26], alpha=2.0
        )
        patcher.register(layer_idx=27)
        
        # Generate (steering applied only to H18+H26)
        generated = model.generate(...)
        
        # Clean up
        patcher.remove()
    """
    
    def __init__(
        self,
        model,
        steering_vector: torch.Tensor,
        target_heads: list[int],
        alpha: float = 1.0,
    ):
        """
        Initialize patcher with steering vector and target heads.
        
        Args:
            model: The transformer model
            steering_vector: Full steering vector (hidden_dim,)
            target_heads: List of head indices to steer (0-31 for Mistral-7B)
            alpha: Scaling factor for steering strength
        """
        self.model = model
        self.steering_vector = steering_vector.detach().to(model.device)
        self.target_heads = target_heads
        self.alpha = alpha
        self.handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self.layer_idx: Optional[int] = None
        
        # Mistral-7B: 32 heads, 128 dims per head, 4096 total
        self.num_heads = 32
        self.head_dim = 128
        self.hidden_dim = 4096
        
        # Compute dim ranges for each target head
        self.target_dims = []
        for head_idx in target_heads:
            start_dim = head_idx * self.head_dim
            end_dim = (head_idx + 1) * self.head_dim
            self.target_dims.append((start_dim, end_dim))
    
    def register(self, layer_idx: int):
        """
        Register forward hook at specified layer to apply steering.
        
        Args:
            layer_idx: Layer index (0-indexed, e.g., 27 for L27)
        """
        if self.handle is not None:
            raise RuntimeError("Patcher already registered. Call remove() first.")
        
        self.layer_idx = layer_idx
        layer = self.model.model.layers[layer_idx].self_attn
        
        def hook_fn(module, inp, out):
            """
            Hook function that adds steering vector ONLY to target heads.
            
            Args:
                module: The v_proj module
                inp: Input to v_proj (hidden states)
                out: Output from v_proj (batch, seq_len, hidden_dim)
            
            Returns:
                Steered output with same shape as out
            """
            # Clone output to avoid in-place modification
            out_steered = out.clone()
            
            # Apply steering ONLY to target head dimensions
            for start_dim, end_dim in self.target_dims:
                # Extract steering vector for this head
                steering_head = self.steering_vector[start_dim:end_dim]  # (head_dim,)
                
                # Add steering to this head's output
                # out[:, :, start_dim:end_dim] shape: (batch, seq_len, head_dim)
                out_steered[:, :, start_dim:end_dim] += (
                    self.alpha * steering_head.unsqueeze(0).unsqueeze(0)
                )
            
            return out_steered
        
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








