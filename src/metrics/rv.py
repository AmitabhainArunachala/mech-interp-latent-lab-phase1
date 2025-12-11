"""
R_V metric: Geometric contraction in value-space.

R_V is defined as PR_late / PR_early on the V-projection during the prompt forward pass.

Where:
- PR (Participation Ratio) = (Σλᵢ²)² / Σ(λᵢ²)²
- λᵢ are singular values from SVD of the V-projection window
- Early layer: 5 (after initial processing)
- Late layer: num_layers - 5 (typically 27 for 32-layer models)
- Window: Last W=16 tokens of the prompt
"""

from typing import Optional

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from ..core.hooks import capture_v_projection


def participation_ratio(
    v_tensor: Optional[torch.Tensor],
    window_size: int = 16,
) -> float:
    """
    Compute Participation Ratio (PR) from V-projection tensor.
    
    PR measures effective dimensionality:
    PR = (Σλᵢ²)² / Σ(λᵢ²)²
    
    Where λᵢ are singular values from SVD of the last W tokens.
    
    Args:
        v_tensor: V-projection tensor of shape (batch, seq_len, hidden_dim) or (seq_len, hidden_dim).
        window_size: Number of tokens to use from the end. Default: 16.
    
    Returns:
        Participation ratio (float). Returns NaN if computation fails.
    
    Note:
        Handles numerical stability: checks for degenerate cases, zero norms, etc.
    """
    if v_tensor is None:
        return float("nan")
    
    # Handle batch dimension
    if v_tensor.dim() == 3:
        v_tensor = v_tensor[0]  # Take first batch
    
    T, D = v_tensor.shape
    W = min(window_size, T)
    
    if W == 0:
        return float("nan")
    
    # Extract last W tokens
    v_window = v_tensor[-W:, :].float()
    
    try:
        # SVD with numerical stability guards
        U, S, Vt = torch.linalg.svd(v_window.T, full_matrices=False)
        S_np = S.cpu().numpy()
        S_sq = S_np ** 2
        
        # Check for degeneracy
        total_variance = S_sq.sum()
        if total_variance < 1e-10:
            return float("nan")
        
        # Compute PR
        p = S_sq / total_variance  # Normalized eigenvalues
        pr = (S_sq.sum() ** 2) / (S_sq ** 2).sum()
        
        return float(pr)
    except Exception:
        return float("nan")


def compute_rv(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    text: str,
    early: int = 5,
    late: int = 27,
    window: int = 16,
    device: str = "cuda",
) -> float:
    """
    Compute R_V metric: PR_late / PR_early.
    
    R_V measures geometric contraction in value-space during recursive self-observation.
    R_V < 1.0 indicates contraction (dimensionality reduction).
    
    Args:
        model: The transformer model (must be in eval mode).
        tokenizer: The tokenizer.
        text: Input prompt text.
        early: Early layer index. Default: 5.
        late: Late layer index. Default: 27 (for 32-layer models).
        window: Window size for PR calculation. Default: 16.
        device: Target device.
    
    Returns:
        R_V value (float). Returns NaN if computation fails.
    
    Note:
        Always measures on prompt tokens (last W tokens), not generated tokens.
    """
    # Tokenize
    enc = tokenizer(text, return_tensors="pt", truncation=True).to(device)
    
    # Capture V-projections
    v_early = None
    v_late = None
    
    with capture_v_projection(model, early) as storage_early:
        with torch.no_grad():
            model(**enc)
        v_early = storage_early.get("v")
    
    with capture_v_projection(model, late) as storage_late:
        with torch.no_grad():
            model(**enc)
        v_late = storage_late.get("v")
    
    # Compute PRs
    pr_early = participation_ratio(v_early, window)
    pr_late = participation_ratio(v_late, window)
    
    # Check for invalid values
    if pr_early == 0 or np.isnan(pr_early) or np.isnan(pr_late):
        return float("nan")
    
    return float(pr_late / pr_early)

