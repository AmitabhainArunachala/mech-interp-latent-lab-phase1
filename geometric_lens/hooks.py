"""
Model-agnostic hook context managers for capturing internal representations.

All hooks follow the Intervention Invariant:
- Use Python context managers for all model modifications.
- Never leave a hook attached after exiting the context.
"""

from contextlib import contextmanager
from typing import Optional

import torch

from .models import (
    get_v_proj_module,
    get_k_proj_module,
    get_q_proj_module,
    get_self_attn_module,
    get_layers,
    extract_v_from_output,
    extract_k_from_output,
    extract_q_from_output,
)


@contextmanager
def capture_v_projection(model, layer_idx: int):
    """
    Capture V-projection outputs at a specific layer.

    Works across all supported architectures (separate v_proj, fused QKV).

    Yields:
        dict with "v" key → tensor of shape (batch, seq, v_dim)
    """
    storage = {"v": None}
    module, kind = get_v_proj_module(model, layer_idx)

    def hook_fn(mod, inp, out):
        try:
            storage["v"] = extract_v_from_output(out, kind).detach()
        except Exception:
            storage["v"] = None
        return out

    handle = module.register_forward_hook(hook_fn)
    try:
        yield storage
    finally:
        handle.remove()


@contextmanager
def capture_k_projection(model, layer_idx: int):
    """
    Capture K-projection outputs at a specific layer.

    Yields:
        dict with "k" key → tensor of shape (batch, seq, k_dim)
    """
    storage = {"k": None}
    module, kind = get_k_proj_module(model, layer_idx)

    def hook_fn(mod, inp, out):
        try:
            storage["k"] = extract_k_from_output(out, kind).detach()
        except Exception:
            storage["k"] = None
        return out

    handle = module.register_forward_hook(hook_fn)
    try:
        yield storage
    finally:
        handle.remove()


@contextmanager
def capture_q_projection(model, layer_idx: int):
    """
    Capture Q-projection outputs at a specific layer.

    Yields:
        dict with "q" key → tensor of shape (batch, seq, q_dim)
    """
    storage = {"q": None}
    module, kind = get_q_proj_module(model, layer_idx)

    def hook_fn(mod, inp, out):
        try:
            storage["q"] = extract_q_from_output(out, kind).detach()
        except Exception:
            storage["q"] = None
        return out

    handle = module.register_forward_hook(hook_fn)
    try:
        yield storage
    finally:
        handle.remove()


@contextmanager
def capture_attention_patterns(model, layer_idx: int):
    """
    Capture attention weights at a specific layer.

    NOTE: Requires model forward pass with output_attentions=True
    and attn_implementation="eager".

    Yields:
        dict with "attn_weights" key → (batch, num_heads, seq, seq)
    """
    storage = {"attn_weights": None}
    attn_module = get_self_attn_module(model, layer_idx)

    def hook_fn(mod, inp, out):
        if isinstance(out, tuple) and len(out) > 1 and out[1] is not None:
            storage["attn_weights"] = out[1].detach()
        return out

    handle = attn_module.register_forward_hook(hook_fn)
    try:
        yield storage
    finally:
        handle.remove()


@contextmanager
def capture_hidden_states(model, layer_idx: int):
    """
    Capture hidden states (residual stream) at a specific layer.

    Yields:
        dict with "hidden" key → (batch, seq, hidden_size)
    """
    storage = {"hidden": None}
    layers = get_layers(model)
    layer = layers[layer_idx]

    def hook_fn(mod, inp, out):
        if isinstance(out, tuple):
            storage["hidden"] = out[0].detach()
        else:
            storage["hidden"] = out.detach()
        return out

    handle = layer.register_forward_hook(hook_fn)
    try:
        yield storage
    finally:
        handle.remove()


@contextmanager
def capture_multi_layer(model, layer_indices: list, component: str = "v"):
    """
    Capture a component across multiple layers in a single forward pass.

    Args:
        model: The transformer model.
        layer_indices: List of layer indices to capture.
        component: One of "v", "k", "q", "hidden".

    Yields:
        dict mapping layer_idx → tensor
    """
    storage = {}
    handles = []

    for layer_idx in layer_indices:
        storage[layer_idx] = None

        if component == "v":
            module, kind = get_v_proj_module(model, layer_idx)
            extractor = lambda out, k=kind: extract_v_from_output(out, k)
        elif component == "k":
            module, kind = get_k_proj_module(model, layer_idx)
            extractor = lambda out, k=kind: extract_k_from_output(out, k)
        elif component == "q":
            module, kind = get_q_proj_module(model, layer_idx)
            extractor = lambda out, k=kind: extract_q_from_output(out, k)
        elif component == "hidden":
            layers = get_layers(model)
            module = layers[layer_idx]
            extractor = lambda out: out[0] if isinstance(out, tuple) else out
        else:
            raise ValueError(f"Unknown component: {component}")

        def make_hook(idx, ext):
            def hook_fn(mod, inp, out):
                try:
                    storage[idx] = ext(out).detach()
                except Exception:
                    storage[idx] = None
                return out
            return hook_fn

        handle = module.register_forward_hook(make_hook(layer_idx, extractor))
        handles.append(handle)

    try:
        yield storage
    finally:
        for h in handles:
            h.remove()
