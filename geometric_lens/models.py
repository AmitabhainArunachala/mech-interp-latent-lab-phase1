"""
Model registry and architecture auto-detection for GeometricLens.

Supports: Llama/Mistral/Qwen/Gemma, OPT, GPTNeoX (Pythia), GPT-2, Phi-3.
Automatically determines layer access paths, projection types, and default
early/late layer indices for R_V measurement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Literal, Optional, Sequence, Tuple

import torch
import torch.nn as nn


ProjKind = Literal["separate", "fused_qkv"]


@dataclass(frozen=True)
class ModelSpec:
    """Physical constants and architecture metadata for a model."""

    name: str
    num_layers: int
    num_heads: int
    num_kv_heads: int
    hidden_size: int
    head_dim: int
    proj_kind: ProjKind  # separate v_proj vs fused QKV

    # Default measurement layers (~15% and ~84% depth)
    early_layer: int
    late_layer: int

    # Known circuit heads (if identified)
    suppressor_heads: List[Tuple[int, int]] = field(default_factory=list)
    amplifier_heads: List[Tuple[int, int]] = field(default_factory=list)

    @property
    def d_model(self) -> int:
        return self.hidden_size

    @property
    def is_gqa(self) -> bool:
        return self.num_kv_heads < self.num_heads

    @property
    def kv_group_size(self) -> int:
        return self.num_heads // self.num_kv_heads if self.num_kv_heads > 0 else 1


# ── Layer accessors ───────────────────────────────────────────────────────────

def get_layers(model: Any) -> Sequence[nn.Module]:
    """Return the transformer block list for any supported HF architecture."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        return model.model.decoder.layers
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise AttributeError(
        f"Unsupported architecture: could not locate transformer layers "
        f"(model type: {type(model).__name__})."
    )


def get_final_norm(model: Any) -> nn.Module:
    """Return the final layer norm before the LM head."""
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm
    if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "final_layer_norm"):
        return model.model.decoder.final_layer_norm
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "final_layer_norm"):
        return model.gpt_neox.final_layer_norm
    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        return model.transformer.ln_f
    raise AttributeError(f"Could not find final layer norm for {type(model).__name__}.")


def get_v_proj_module(model: Any, layer_idx: int) -> Tuple[nn.Module, ProjKind]:
    """Get the V-projection module and its kind for a given layer."""
    layers = get_layers(model)
    layer = layers[layer_idx]

    # Llama/Mistral/Qwen/Gemma/Phi: separate v_proj
    if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "v_proj"):
        return layer.self_attn.v_proj, "separate"

    # OPT: separate v_proj under self_attn
    if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "v_proj"):
        return layer.self_attn.v_proj, "separate"

    # Phi-3: fused QKV via qkv_proj
    if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "qkv_proj"):
        return layer.self_attn.qkv_proj, "fused_qkv"

    # GPTNeoX (Pythia): fused QKV
    if hasattr(layer, "attention") and hasattr(layer.attention, "query_key_value"):
        return layer.attention.query_key_value, "fused_qkv"

    # GPT-2: fused QKV
    if hasattr(layer, "attn") and hasattr(layer.attn, "c_attn"):
        return layer.attn.c_attn, "fused_qkv"

    raise AttributeError(
        f"Could not find V-projection at layer {layer_idx} "
        f"(layer type: {type(layer).__name__})."
    )


def get_k_proj_module(model: Any, layer_idx: int) -> Tuple[nn.Module, ProjKind]:
    """Get the K-projection module and its kind for a given layer."""
    layers = get_layers(model)
    layer = layers[layer_idx]

    if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "k_proj"):
        return layer.self_attn.k_proj, "separate"
    if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "qkv_proj"):
        return layer.self_attn.qkv_proj, "fused_qkv"
    if hasattr(layer, "attention") and hasattr(layer.attention, "query_key_value"):
        return layer.attention.query_key_value, "fused_qkv"
    if hasattr(layer, "attn") and hasattr(layer.attn, "c_attn"):
        return layer.attn.c_attn, "fused_qkv"

    raise AttributeError(f"Could not find K-projection at layer {layer_idx}.")


def get_q_proj_module(model: Any, layer_idx: int) -> Tuple[nn.Module, ProjKind]:
    """Get the Q-projection module and its kind for a given layer."""
    layers = get_layers(model)
    layer = layers[layer_idx]

    if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "q_proj"):
        return layer.self_attn.q_proj, "separate"
    if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "qkv_proj"):
        return layer.self_attn.qkv_proj, "fused_qkv"
    if hasattr(layer, "attention") and hasattr(layer.attention, "query_key_value"):
        return layer.attention.query_key_value, "fused_qkv"
    if hasattr(layer, "attn") and hasattr(layer.attn, "c_attn"):
        return layer.attn.c_attn, "fused_qkv"

    raise AttributeError(f"Could not find Q-projection at layer {layer_idx}.")


def get_self_attn_module(model: Any, layer_idx: int) -> nn.Module:
    """Get the self-attention module for a given layer."""
    layers = get_layers(model)
    layer = layers[layer_idx]

    if hasattr(layer, "self_attn"):
        return layer.self_attn
    if hasattr(layer, "attention"):
        return layer.attention
    if hasattr(layer, "attn"):
        return layer.attn

    raise AttributeError(f"Could not find attention module at layer {layer_idx}.")


def extract_v_from_output(output: torch.Tensor, kind: ProjKind) -> torch.Tensor:
    """Extract V tensor from projection module output."""
    if kind == "separate":
        return output
    # fused_qkv: output is (batch, seq, 3*hidden) — V is the last third
    d = output.shape[-1]
    h = d // 3
    return output[..., 2 * h : 3 * h]


def extract_k_from_output(output: torch.Tensor, kind: ProjKind) -> torch.Tensor:
    """Extract K tensor from projection module output."""
    if kind == "separate":
        return output
    d = output.shape[-1]
    h = d // 3
    return output[..., h : 2 * h]


def extract_q_from_output(output: torch.Tensor, kind: ProjKind) -> torch.Tensor:
    """Extract Q tensor from projection module output."""
    if kind == "separate":
        return output
    d = output.shape[-1]
    h = d // 3
    return output[..., :h]


# ── Model Registry ────────────────────────────────────────────────────────────

# Known model specs (can be extended at runtime)
_KNOWN_SPECS = {
    "mistralai/Mistral-7B-v0.1": ModelSpec(
        name="mistralai/Mistral-7B-v0.1",
        num_layers=32, num_heads=32, num_kv_heads=8,
        hidden_size=4096, head_dim=128, proj_kind="separate",
        early_layer=5, late_layer=27,
        suppressor_heads=[(27, 18), (27, 26)],
        amplifier_heads=[(27, 2), (27, 10)],
    ),
    "facebook/opt-6.7b": ModelSpec(
        name="facebook/opt-6.7b",
        num_layers=32, num_heads=32, num_kv_heads=32,
        hidden_size=4096, head_dim=128, proj_kind="separate",
        early_layer=5, late_layer=27,
    ),
    "openai-community/gpt2-xl": ModelSpec(
        name="openai-community/gpt2-xl",
        num_layers=48, num_heads=25, num_kv_heads=25,
        hidden_size=1600, head_dim=64, proj_kind="fused_qkv",
        early_layer=7, late_layer=40,
    ),
    "Qwen/Qwen2.5-7B": ModelSpec(
        name="Qwen/Qwen2.5-7B",
        num_layers=32, num_heads=32, num_kv_heads=8,
        hidden_size=4096, head_dim=128, proj_kind="separate",
        early_layer=5, late_layer=27,
    ),
    "EleutherAI/pythia-1.4b": ModelSpec(
        name="EleutherAI/pythia-1.4b",
        num_layers=24, num_heads=16, num_kv_heads=16,
        hidden_size=2048, head_dim=128, proj_kind="fused_qkv",
        early_layer=4, late_layer=20,
    ),
    "EleutherAI/pythia-410m": ModelSpec(
        name="EleutherAI/pythia-410m",
        num_layers=24, num_heads=16, num_kv_heads=16,
        hidden_size=1024, head_dim=64, proj_kind="fused_qkv",
        early_layer=4, late_layer=20,
    ),
    "EleutherAI/pythia-1b": ModelSpec(
        name="EleutherAI/pythia-1b",
        num_layers=16, num_heads=8, num_kv_heads=8,
        hidden_size=2048, head_dim=256, proj_kind="fused_qkv",
        early_layer=2, late_layer=13,
    ),
    "EleutherAI/pythia-2.8b": ModelSpec(
        name="EleutherAI/pythia-2.8b",
        num_layers=32, num_heads=32, num_kv_heads=32,
        hidden_size=2560, head_dim=80, proj_kind="fused_qkv",
        early_layer=5, late_layer=27,
    ),
    "EleutherAI/pythia-6.9b": ModelSpec(
        name="EleutherAI/pythia-6.9b",
        num_layers=32, num_heads=32, num_kv_heads=32,
        hidden_size=4096, head_dim=128, proj_kind="fused_qkv",
        early_layer=5, late_layer=27,
    ),
}


class ModelRegistry:
    """Registry for model architecture specs. Auto-detects from HF config when possible."""

    def __init__(self):
        self._specs = dict(_KNOWN_SPECS)

    def register(self, name: str, spec: ModelSpec):
        """Register a custom model spec."""
        self._specs[name] = spec

    def get(self, model_name: str) -> Optional[ModelSpec]:
        """Get spec by name, or None if unknown."""
        return self._specs.get(model_name)

    def auto_detect(self, model: Any, model_name: str = "") -> ModelSpec:
        """
        Auto-detect model spec from a loaded HuggingFace model.

        Falls back to heuristic layer mapping if model is unknown.
        """
        # Check registry first
        if model_name and model_name in self._specs:
            return self._specs[model_name]

        # Auto-detect from config
        cfg = model.config
        num_layers = getattr(cfg, "num_hidden_layers", 32)
        num_heads = getattr(cfg, "num_attention_heads", 32)
        num_kv_heads = getattr(cfg, "num_key_value_heads", num_heads)
        hidden_size = getattr(cfg, "hidden_size", 4096)
        head_dim = hidden_size // num_heads

        # Detect proj kind
        try:
            _, kind = get_v_proj_module(model, 0)
        except AttributeError:
            kind = "separate"

        # Heuristic layer selection: ~15% and ~84% depth
        early = max(1, int(num_layers * 0.15))
        late = min(num_layers - 1, int(num_layers * 0.84))

        spec = ModelSpec(
            name=model_name or type(model).__name__,
            num_layers=num_layers,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            hidden_size=hidden_size,
            head_dim=head_dim,
            proj_kind=kind,
            early_layer=early,
            late_layer=late,
        )

        # Cache it
        if model_name:
            self._specs[model_name] = spec

        return spec

    def list_known(self) -> list[str]:
        """List all known model names."""
        return list(self._specs.keys())


# Module-level singleton
registry = ModelRegistry()
