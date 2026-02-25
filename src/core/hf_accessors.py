"""
HuggingFace model accessors.

Goal: provide a minimal compatibility layer for common transformer model layouts
so pipelines don't hardcode `model.model.layers[...]` and silently break on
architectures like GPTNeoX (Pythia).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import torch

VProjKind = Literal["v_proj", "fused_qkv"]


def get_layers(model: Any) -> Sequence[Any]:
    """
    Return the model's transformer block list across common HF architectures.

    Supported (best-effort):
    - Llama/Mistral/Qwen/Gemma/Mixtral style: model.model.layers
    - OPT style: model.model.decoder.layers
    - GPTNeoX (Pythia): model.gpt_neox.layers
    - GPT-2 style: model.transformer.h
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        return model.model.decoder.layers
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise AttributeError(
        "Unsupported model architecture: could not locate transformer layers "
        "(tried model.model.layers, model.model.decoder.layers, model.gpt_neox.layers, model.transformer.h)."
    )


@dataclass(frozen=True)
class VProjHookPoint:
    """
    Defines where to hook to observe/patch the value projection.

    - kind="v_proj": module output is already V (batch, seq, hidden)
    - kind="fused_qkv": module output is QKV concatenation (batch, seq, 3*hidden)
    """

    kind: VProjKind
    module: torch.nn.Module


def get_vproj_hookpoint(model: Any, layer_idx: int) -> VProjHookPoint:
    """
    Get the module to hook for a layer's V projection.
    """
    layers = get_layers(model)
    layer = layers[layer_idx]

    # Llama/Mistral style
    if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "v_proj"):
        return VProjHookPoint(kind="v_proj", module=layer.self_attn.v_proj)

    # GPTNeoX style (Pythia): fused QKV
    if hasattr(layer, "attention") and hasattr(layer.attention, "query_key_value"):
        return VProjHookPoint(kind="fused_qkv", module=layer.attention.query_key_value)

    # GPT-2 style: fused QKV
    if hasattr(layer, "attn") and hasattr(layer.attn, "c_attn"):
        return VProjHookPoint(kind="fused_qkv", module=layer.attn.c_attn)

    raise AttributeError(
        "Unsupported layer attention structure: could not find v_proj or fused qkv module "
        f"at layer {layer_idx} (layer type: {type(layer).__name__})."
    )


def extract_v_from_hook_output(hookpoint: VProjHookPoint, out: torch.Tensor) -> torch.Tensor:
    """
    Extract V tensor from the hooked module output.

    Returns a tensor shaped (batch, seq, hidden).
    """
    if hookpoint.kind == "v_proj":
        return out

    # fused qkv
    if out.dim() != 3:
        raise ValueError(f"Expected fused QKV output to be 3D (batch, seq, 3*hidden); got shape {tuple(out.shape)}")
    d = int(out.shape[-1])
    if d % 3 != 0:
        raise ValueError(f"Fused QKV last dim must be divisible by 3; got {d}")
    h = d // 3
    return out[:, :, 2 * h : 3 * h]

