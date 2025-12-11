"""
Model loading utilities.

Standard: Mistral-7B Base (v0.1) is the reference reality.
All other models are comparative studies.
"""

import random
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(
    model_name: str = "mistralai/Mistral-7B-v0.1",
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.float16,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a model and tokenizer with standard configuration.
    
    Args:
        model_name: HuggingFace model identifier. Default: Mistral-7B Base.
        device: Target device ("cuda" or "cpu").
        torch_dtype: Data type for model weights. Default: float16.
    
    Returns:
        Tuple of (model, tokenizer). Model is in eval mode.
    
    Note:
        Instruct models are treated as a separate phenotype (confounding factor).
        Default to Base models for clean experiments.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer

