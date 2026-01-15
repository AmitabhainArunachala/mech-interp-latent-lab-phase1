"""
Model loading utilities.

Standard: Mistral-7B Base (v0.1) is the reference reality.
All other models are comparative studies.
"""

import os
import random
from typing import Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    # NumPy is used in many pipelines for prompt sampling and should be seeded too.
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(
    model_name: str = "mistralai/Mistral-7B-v0.1",
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.float16,
    attn_implementation: str = "sdpa",
    token: Optional[str] = None,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a model and tokenizer with standard configuration.
    
    Args:
        model_name: HuggingFace model identifier. Default: Mistral-7B Base.
        device: Target device ("cuda" or "cpu").
        torch_dtype: Data type for model weights. Default: float16.
        attn_implementation: Attention implementation. Use "eager" if you need
            to capture attention weights with output_attentions=True. Default: "sdpa".
        token: HuggingFace token for gated models. If None, uses HF_TOKEN env var.
    
    Returns:
        Tuple of (model, tokenizer). Model is in eval mode.
    
    Note:
        Instruct models are treated as a separate phenotype (confounding factor).
        Default to Base models for clean experiments.
        
        For attention pattern capture, use attn_implementation="eager" since SDPA
        doesn't support output_attentions=True.
    """
    # Get token from parameter or environment variable
    hf_token = token or os.environ.get("HF_TOKEN")
    
    # Prefer slow tokenizers for consistency across model families, but fall back to fast
    # when sentencepiece-based slow tokenizers error (some repos ship tokenizer assets in
    # ways that break slow loading).
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, token=hf_token)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
        attn_implementation=attn_implementation,
        token=hf_token,
    )
    model.eval()
    return model, tokenizer

