"""
KV Cache utilities: Extraction, mixing, and generation with cached keys/values.

KV Cache patching must respect the DynamicCache structure of the specific
HuggingFace version in use.
"""

from typing import List, Optional, Sequence, Tuple

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.cache_utils import DynamicCache


def capture_past_key_values(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    device: str = "cuda",
) -> DynamicCache:
    """
    Extract past_key_values (KV cache) from a prompt forward pass.
    
    Args:
        model: The transformer model.
        tokenizer: The tokenizer.
        prompt: Input prompt text.
        device: Target device.
    
    Returns:
        DynamicCache containing the KV cache.
    """
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**enc, use_cache=True, return_dict=True)
    return out.past_key_values


def extract_kv_list(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    device: str = "cuda",
    max_length: int = 512,
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
    """
    Extract KV cache as a list of (K, V) tensors in float32 plus input_ids.
    
    Args:
        model: The transformer model.
        tokenizer: The tokenizer.
        prompt: Input prompt text.
        device: Target device.
        max_length: Maximum sequence length.
    
    Returns:
        Tuple of (kv_list, input_ids) where kv_list is a list of (K, V) tuples
        for each layer, and input_ids is the tokenized input.
    """
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=max_length
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True, return_dict=True)
        past_kv = outputs.past_key_values
    
    kv_list: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for k, v in past_kv:
        kv_list.append((k.float(), v.float()))
    
    return kv_list, inputs["input_ids"]


def mix_kv_to_dynamic_cache(
    base_kv: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    rec_kv: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    layer_start: int,
    layer_end: int,
    alpha: float = 1.0,
) -> DynamicCache:
    """
    Mix KV caches with α-mixing, then convert to a DynamicCache.
    
    For layers in [layer_start, layer_end), apply α-mixing:
    - α = 0.0: Pure baseline KV
    - α = 1.0: Pure recursive KV
    - 0 < α < 1: Linear interpolation
    
    Args:
        base_kv: Baseline KV cache as list of (K, V) tuples.
        rec_kv: Recursive KV cache as list of (K, V) tuples.
        layer_start: Start layer index (inclusive).
        layer_end: End layer index (exclusive).
        alpha: Mixing coefficient. Default: 1.0.
    
    Returns:
        DynamicCache containing the mixed KV cache.
    
    Note:
        Handles different sequence lengths by truncating to minimum length.
    """
    mixed_kv = DynamicCache()
    num_layers = len(base_kv)
    patch_layers = set(range(layer_start, layer_end))
    
    for layer_idx in range(num_layers):
        k_base, v_base = base_kv[layer_idx]
        
        if layer_idx in patch_layers and alpha > 0:
            k_rec, v_rec = rec_kv[layer_idx]
            
            if alpha == 1.0:
                # Pure recursive for this layer
                k_out = k_rec.half()
                v_out = v_rec.half()
            else:
                # Mix in float32, then convert to half
                min_seq = min(k_base.shape[2], k_rec.shape[2])
                k_base_t = k_base[:, :, :min_seq, :]
                v_base_t = v_base[:, :, :min_seq, :]
                k_rec_t = k_rec[:, :, :min_seq, :]
                v_rec_t = v_rec[:, :, :min_seq, :]
                
                k_out = ((1 - alpha) * k_base_t + alpha * k_rec_t).half()
                v_out = ((1 - alpha) * v_base_t + alpha * v_rec_t).half()
        else:
            k_out = k_base.half()
            v_out = v_base.half()
        
        mixed_kv.update(k_out, v_out, layer_idx)
    
    return mixed_kv


def generate_with_kv(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    past_key_values: Optional[DynamicCache] = None,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    device: str = "cuda",
) -> str:
    """
    Generate continuation with optional KV cache.
    
    Args:
        model: The transformer model.
        tokenizer: The tokenizer.
        prompt: Input prompt text.
        past_key_values: Optional KV cache to use. If None, generates from scratch.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature. 0.0 = greedy.
        device: Target device.
    
    Returns:
        Generated text (decoded).
    """
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0.0,
        "temperature": temperature if temperature > 0 else None,
        "use_cache": True,
        "pad_token_id": tokenizer.eos_token_id,
    }
    
    if past_key_values is not None:
        past_len = past_key_values[0][0].shape[2]
        input_ids = enc["input_ids"]
        attn_mask = torch.ones(
            (1, past_len + input_ids.shape[1]), device=device, dtype=torch.long
        )
        position_ids = torch.arange(
            past_len, past_len + input_ids.shape[1], device=device
        ).unsqueeze(0)
        gen = model.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            **generate_kwargs,
        )
    else:
        gen = model.generate(**enc, **generate_kwargs)
    
    return tokenizer.decode(gen[0], skip_special_tokens=True)

