"""
Core R_V metrics for measuring geometric signatures in transformer value spaces.

The key insight: recursive self-reference induces characteristic geometric contraction
in the participation ratio of value activations at late integration layers.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import numpy as np


@dataclass
class RVResult:
    """Result of R_V computation."""
    
    rv: float  # Participation ratio (R_V)
    effective_rank: float  # Effective rank (entropy-based)
    singular_values: np.ndarray  # Raw singular values
    v_parallel_norm: Optional[float] = None  # In-subspace component norm
    v_perp_norm: Optional[float] = None  # Orthogonal component norm
    
    @property
    def dual_ratio(self) -> Optional[float]:
        """Ratio of parallel to perpendicular components."""
        if self.v_parallel_norm is not None and self.v_perp_norm is not None:
            if self.v_perp_norm > 1e-10:
                return self.v_parallel_norm / self.v_perp_norm
        return None


def compute_participation_ratio(singular_values: np.ndarray) -> float:
    """
    Compute participation ratio (PR) from singular values.
    
    PR = (Σ σ_i²)² / Σ σ_i⁴ = 1 / Σ p_i² where p_i = σ_i² / Σ σ_j²
    
    This measures effective dimensionality:
    - PR ≈ 1: Information concentrated in one dimension
    - PR ≈ D: Information uniformly distributed across D dimensions
    
    Args:
        singular_values: Array of singular values
        
    Returns:
        Participation ratio (float)
    """
    s_sq = singular_values ** 2
    total_variance = s_sq.sum()
    
    if total_variance < 1e-10:
        return np.nan
    
    # Normalized eigenvalues (probabilities)
    p = s_sq / total_variance
    # Correct PR formula: 1 / sum of squared probabilities
    pr = 1.0 / (p ** 2).sum()
    return float(pr)


def compute_effective_rank(singular_values: np.ndarray) -> float:
    """
    Compute effective rank using entropy of normalized singular values.
    
    eff_rank = exp(-Σ p_i log p_i) where p_i = σ_i² / Σ σ_j²
    
    Equivalent formulation: 1 / Σ p_i²
    
    Args:
        singular_values: Array of singular values
        
    Returns:
        Effective rank (float)
    """
    s_sq = singular_values ** 2
    
    if s_sq.sum() < 1e-10:
        return np.nan
    
    p = s_sq / s_sq.sum()
    eff_rank = 1.0 / (p ** 2).sum()
    return float(eff_rank)


def compute_rv(
    v_tensor: torch.Tensor,
    window_size: int = 16,
    return_components: bool = False,
) -> Union[float, RVResult]:
    """
    Compute R_V (participation ratio) for a value tensor.
    
    The R_V metric measures the effective dimensionality of the value space
    over a sliding window of tokens. Recursive self-reference induces
    characteristic contraction (lower R_V) at late integration layers.
    
    Args:
        v_tensor: Value tensor of shape (batch, seq_len, dim) or (seq_len, dim)
        window_size: Number of final tokens to analyze (default: 16)
        return_components: If True, return full RVResult with decomposition
        
    Returns:
        If return_components=False: participation ratio (float)
        If return_components=True: RVResult with full decomposition
        
    Example:
        >>> v = model.layers[27].self_attn.v_proj(hidden_states)
        >>> rv = compute_rv(v, window_size=16)
        >>> print(f"R_V = {rv:.4f}")
    """
    if v_tensor is None:
        return np.nan if not return_components else RVResult(np.nan, np.nan, np.array([]))
    
    # Handle batch dimension
    if v_tensor.dim() == 3:
        v_tensor = v_tensor[0]  # Take first batch element
    
    T, D = v_tensor.shape
    W = min(window_size, T)
    
    if W < 2:
        return np.nan if not return_components else RVResult(np.nan, np.nan, np.array([]))
    
    # Extract window
    v_window = v_tensor[-W:, :].float()
    
    try:
        # SVD of transposed window (D x W matrix)
        U, S, Vt = torch.linalg.svd(v_window.T, full_matrices=False)
        S_np = S.cpu().numpy()
        
        pr = compute_participation_ratio(S_np)
        eff_rank = compute_effective_rank(S_np)
        
        if not return_components:
            return pr
        
        return RVResult(
            rv=pr,
            effective_rank=eff_rank,
            singular_values=S_np,
        )
        
    except Exception:
        return np.nan if not return_components else RVResult(np.nan, np.nan, np.array([]))


def compute_dual_space_decomposition(
    v_tensor: torch.Tensor,
    subspace_basis: torch.Tensor,
    window_size: int = 16,
) -> RVResult:
    """
    Compute R_V with dual-space decomposition into V_parallel and V_perpendicular.
    
    The dual-space model decomposes value activations into:
    - V_parallel: Component within the recursive subspace (in-subspace)
    - V_perpendicular: Component orthogonal to recursive subspace
    
    Key finding: These components contract coordinately (r ≈ 0.904),
    revealing unified geometric mechanism.
    
    Args:
        v_tensor: Value tensor of shape (seq_len, dim)
        subspace_basis: Orthonormal basis for recursive subspace (k, dim)
        window_size: Number of final tokens to analyze
        
    Returns:
        RVResult with dual-space norms populated
        
    Example:
        >>> # Get basis from recursive prompt activations
        >>> basis = extract_recursive_subspace(v_recursive, n_components=10)
        >>> # Decompose baseline activations
        >>> result = compute_dual_space_decomposition(v_baseline, basis)
        >>> print(f"V_par/V_perp ratio: {result.dual_ratio:.4f}")
    """
    if v_tensor.dim() == 3:
        v_tensor = v_tensor[0]
    
    T, D = v_tensor.shape
    W = min(window_size, T)
    v_window = v_tensor[-W:, :].float()
    
    # Project onto subspace
    # V_parallel = V @ B^T @ B  (projection onto subspace spanned by B)
    # V_perpendicular = V - V_parallel
    
    B = subspace_basis.float().to(v_window.device)
    v_parallel = v_window @ B.T @ B
    v_perp = v_window - v_parallel
    
    # Compute norms
    v_par_norm = float(v_parallel.norm().cpu())
    v_perp_norm = float(v_perp.norm().cpu())
    
    # Compute R_V on full tensor
    try:
        U, S, Vt = torch.linalg.svd(v_window.T, full_matrices=False)
        S_np = S.cpu().numpy()
        pr = compute_participation_ratio(S_np)
        eff_rank = compute_effective_rank(S_np)
    except Exception:
        pr, eff_rank, S_np = np.nan, np.nan, np.array([])
    
    return RVResult(
        rv=pr,
        effective_rank=eff_rank,
        singular_values=S_np,
        v_parallel_norm=v_par_norm,
        v_perp_norm=v_perp_norm,
    )


def extract_recursive_subspace(
    v_recursive: torch.Tensor,
    n_components: int = 10,
    window_size: int = 16,
) -> torch.Tensor:
    """
    Extract principal subspace from recursive prompt activations.
    
    Uses SVD to find the top-k principal directions of the recursive
    activation geometry.
    
    Args:
        v_recursive: Value tensor from recursive prompt
        n_components: Number of principal components to keep
        window_size: Number of final tokens to analyze
        
    Returns:
        Orthonormal basis tensor of shape (n_components, dim)
    """
    if v_recursive.dim() == 3:
        v_recursive = v_recursive[0]
    
    T, D = v_recursive.shape
    W = min(window_size, T)
    v_window = v_recursive[-W:, :].float()
    
    # SVD: V = U @ S @ Vt
    # Right singular vectors Vt are principal directions in D-space
    U, S, Vt = torch.linalg.svd(v_window.T, full_matrices=False)
    
    # Take top-k components (rows of U are directions in D-space)
    k = min(n_components, U.shape[1])
    basis = U[:, :k].T  # Shape: (k, D)
    
    return basis


def compute_rv_layerwise(
    model,
    tokenizer,
    text: str,
    window_size: int = 16,
    device: str = "cuda",
) -> dict:
    """
    Compute R_V at every layer for a given prompt.
    
    This reveals the layer profile: recursive prompts show characteristic
    contraction peaking at L25-L27 (78-84% depth in Mistral-7B).
    
    Args:
        model: HuggingFace transformer model
        tokenizer: Corresponding tokenizer
        text: Input text
        window_size: Number of final tokens
        device: Computation device
        
    Returns:
        Dict mapping layer index to R_V value
        
    Example:
        >>> rv_profile = compute_rv_layerwise(model, tokenizer, recursive_prompt)
        >>> critical_layer = min(rv_profile, key=rv_profile.get)
        >>> print(f"Maximum contraction at layer {critical_layer}")
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    n_layers = len(model.model.layers)
    rv_by_layer = {}
    
    with torch.no_grad():
        for layer_idx in range(n_layers):
            captured = []
            
            def hook_fn(m, inp, out, storage=captured):
                storage.append(out.detach())
                return out
            
            # Register hook on v_proj
            layer = model.model.layers[layer_idx].self_attn
            handle = layer.v_proj.register_forward_hook(hook_fn)
            
            _ = model(**inputs)
            
            handle.remove()
            
            if captured:
                v = captured[0][0]  # Remove batch dim
                rv = compute_rv(v, window_size=window_size)
                rv_by_layer[layer_idx] = rv
    
    return rv_by_layer
