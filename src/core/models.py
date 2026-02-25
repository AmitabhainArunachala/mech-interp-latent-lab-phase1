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


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value.
        deterministic: If True, enables CUDA deterministic mode for full
            reproducibility. May slightly impact performance. Default: True.

    Note:
        For GPU experiments, deterministic=True ensures bit-exact reproducibility
        across runs. This is critical for publication-grade experiments.
    """
    random.seed(seed)
    # NumPy is used in many pipelines for prompt sampling and should be seeded too.
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # GPU determinism flags for full reproducibility
    if deterministic and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Set CUBLAS workspace config for deterministic matmul
        import os
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        # Enable deterministic algorithms but allow unsupported CUDA ops to warn.
        # Sampling kernels (e.g., cumsum in top-p) may not have deterministic implementations.
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            # Some operations don't have deterministic implementations
            pass


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
    
    # Respect explicit device requests. "auto" is only used when requested.
    if device == "auto":
        device_map: object = "auto"
    elif device == "cuda":
        device_map = {"": "cuda:0"}
    elif device.startswith("cuda:"):
        device_map = {"": device}
    elif device in {"mps", "cpu"}:
        device_map = {"": device}
    else:
        device_map = "auto"

    def _load(attn_impl: str):
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            attn_implementation=attn_impl,
            token=hf_token,
            low_cpu_mem_usage=True,
        )

    try:
        model = _load(attn_implementation)
    except ValueError as e:
        # Newer Torch/Transformers combos can reject SDPA for Mistral.
        # Fall back to eager attention so experiments still run.
        if "scaled_dot_product_attention" in str(e) and attn_implementation != "eager":
            model = _load("eager")
        else:
            raise
    model.eval()
    return model, tokenizer
