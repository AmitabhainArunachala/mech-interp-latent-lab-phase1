"""
Head-specific patching utilities for surgical interventions.

Important: for GQA models (e.g. Mistral), `v_proj` lives in **KV-head space**:
`v_proj_out_dim = num_key_value_heads * head_dim`, NOT `hidden_size`.

If you pass `head_space="q"`, query-head indices will be mapped to the
corresponding KV-head indices using the same grouping as HF `repeat_kv`
(contiguous blocks).
"""

from __future__ import annotations

from typing import Iterable, Literal, Optional

import torch

from .hf_accessors import extract_v_from_hook_output, get_vproj_hookpoint

HeadSpace = Literal["kv", "q"]


def _get_hidden_size_and_num_heads(model) -> tuple[int, int]:
    cfg = getattr(model, "config", None)
    if cfg is None:
        raise ValueError("Model has no config; cannot infer attention head layout.")

    hidden_size = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None)
    num_heads = getattr(cfg, "num_attention_heads", None) or getattr(cfg, "n_head", None)
    if hidden_size is None or num_heads is None:
        raise ValueError(
            "Model config missing hidden size / num heads (expected config.hidden_size|n_embd and "
            "config.num_attention_heads|n_head)."
        )
    return int(hidden_size), int(num_heads)


def _get_head_dim(model) -> int:
    cfg = model.config
    hidden_size, num_heads = _get_hidden_size_and_num_heads(model)
    head_dim = getattr(cfg, "head_dim", None) or (hidden_size // num_heads)
    return int(head_dim)


def _resolve_v_heads(
    *,
    model,
    v_dim: int,
    heads: Iterable[int],
    head_space: HeadSpace,
) -> list[int]:
    """
    Resolve requested head indices into indices over the hooked V-space.

    - head_space="kv": `heads` must index V-heads directly (0..num_v_heads-1)
    - head_space="q": `heads` index query heads; we map them to V/KV heads via
      contiguous grouping: kv = q // group_size
    """
    head_dim = _get_head_dim(model)
    _hidden_size, num_q_heads = _get_hidden_size_and_num_heads(model)

    if v_dim % head_dim != 0:
        raise ValueError(f"V dim {v_dim} is not divisible by head_dim {head_dim}.")
    num_v_heads = v_dim // head_dim
    if num_q_heads % num_v_heads != 0:
        raise ValueError(
            f"num_attention_heads {num_q_heads} must be divisible by num_v_heads {num_v_heads}."
        )
    group_size = num_q_heads // num_v_heads

    heads_list = [int(h) for h in heads]
    if head_space == "kv":
        bad = [h for h in heads_list if h < 0 or h >= num_v_heads]
        if bad:
            raise ValueError(f"KV/V heads out of range (0..{num_v_heads-1}): {bad}")
        return sorted(set(heads_list))

    if head_space == "q":
        bad = [h for h in heads_list if h < 0 or h >= num_q_heads]
        if bad:
            raise ValueError(f"Q heads out of range (0..{num_q_heads-1}): {bad}")
        mapped = [h // group_size for h in heads_list]
        return sorted(set(mapped))

    raise ValueError(f"Unknown head_space: {head_space}")


class HeadSpecificVPatcher:
    """
    Patch V projection activations for specific heads only.

    Under GQA, this patches KV-heads, not individual query heads.
    """

    def __init__(
        self,
        model,
        v_activation: torch.Tensor,
        target_heads: list[int],
        window_size: int = 16,
        *,
        head_space: HeadSpace = "kv",
    ):
        self.model = model
        if v_activation.dim() == 3:
            v_activation = v_activation[0]
        self.v_activation = v_activation.detach()

        self.window_size = int(window_size)
        self.handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self.layer_idx: Optional[int] = None

        v_dim = int(self.v_activation.shape[-1])
        head_dim = _get_head_dim(model)
        v_heads = _resolve_v_heads(model=model, v_dim=v_dim, heads=target_heads, head_space=head_space)
        self.target_dims = [(h * head_dim, (h + 1) * head_dim) for h in v_heads]

    def register(self, layer_idx: int):
        if self.handle is not None:
            raise RuntimeError("Patcher already registered. Call remove() first.")

        self.layer_idx = int(layer_idx)
        hookpoint = get_vproj_hookpoint(self.model, layer_idx=self.layer_idx)

        def hook_fn(_module, _inp, out):
            v = extract_v_from_hook_output(hookpoint, out)
            batch, seq_len, _v_dim = v.shape

            v_len = min(seq_len, int(self.v_activation.shape[0]), self.window_size)
            if v_len <= 0:
                return out

            v_slice = self.v_activation[-v_len:, :]
            out_patched = out.clone()

            # Get a view into the output's V slice (handles fused QKV too).
            if hookpoint.kind == "v_proj":
                v_out = out_patched
            else:
                hidden = int(out_patched.shape[-1] // 3)
                v_out = out_patched[:, :, 2 * hidden : 3 * hidden]

            for start_dim, end_dim in self.target_dims:
                v_head_slice = v_slice[:, start_dim:end_dim]
                if v_head_slice.numel() == 0:
                    continue
                v_head_batch = v_head_slice.unsqueeze(0)
                if batch > 1:
                    v_head_batch = v_head_batch.repeat(batch, 1, 1)
                v_out[:, -v_len:, start_dim:end_dim] = v_head_batch[:, :v_len, :].to(
                    v_out.device, dtype=v_out.dtype
                )

            return out_patched

        self.handle = hookpoint.module.register_forward_hook(hook_fn)

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
            self.layer_idx = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove()


class HeadSpecificSteeringPatcher:
    """
    Apply a steering vector to specific V-heads only.

    For GQA models, this applies to KV-heads unless you pass `head_space="q"`
    (in which case query heads are mapped to the corresponding KV heads).
    """

    def __init__(
        self,
        model,
        steering_vector: torch.Tensor,
        target_heads: list[int],
        alpha: float = 1.0,
        *,
        head_space: HeadSpace = "kv",
    ):
        self.model = model
        self.steering_vector = steering_vector.detach().to(model.device)
        self.alpha = float(alpha)

        self.handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self.layer_idx: Optional[int] = None

        v_dim = int(self.steering_vector.shape[-1])
        head_dim = _get_head_dim(model)
        v_heads = _resolve_v_heads(model=model, v_dim=v_dim, heads=target_heads, head_space=head_space)
        self.target_dims = [(h * head_dim, (h + 1) * head_dim) for h in v_heads]

    def register(self, layer_idx: int):
        if self.handle is not None:
            raise RuntimeError("Patcher already registered. Call remove() first.")

        self.layer_idx = int(layer_idx)
        hookpoint = get_vproj_hookpoint(self.model, layer_idx=self.layer_idx)

        def hook_fn(_module, _inp, out):
            out_steered = out.clone()

            if hookpoint.kind == "v_proj":
                v_out = out_steered
            else:
                hidden = int(out_steered.shape[-1] // 3)
                v_out = out_steered[:, :, 2 * hidden : 3 * hidden]

            for start_dim, end_dim in self.target_dims:
                steering_head = self.steering_vector[start_dim:end_dim]
                if steering_head.numel() == 0:
                    continue
                v_out[:, :, start_dim:end_dim] += self.alpha * steering_head.unsqueeze(0).unsqueeze(0)

            return out_steered

        self.handle = hookpoint.module.register_forward_hook(hook_fn)

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
            self.layer_idx = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove()







