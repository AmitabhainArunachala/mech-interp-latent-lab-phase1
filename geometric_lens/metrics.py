"""
Geometric metrics for transformer representations.

Core metrics:
- Participation Ratio (PR): effective dimensionality of V-projection column space
- R_V: PR_late / PR_early — geometric contraction ratio
- Spectral Stats: top-1 ratio, spectral gap, effective rank, condition number
- Cosine Similarity: directional alignment between early/late representations
- Attention Entropy: focus/diffuseness of attention patterns

All metrics operate on raw tensors and are model-agnostic.
The GeometricProbe class (in probe.py) handles model interaction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class SpectralStats:
    """Spectral shape statistics from SVD singular values."""

    top1_ratio: float        # σ₁ / Σσᵢ — dominance of first component
    spectral_gap: float      # σ₁ - σ₂ — separation of top direction
    effective_rank: float    # exp(entropy of normalized σ²)
    condition_number: float  # σ_max / σ_min

    def to_dict(self) -> Dict[str, float]:
        return {
            "top1_ratio": self.top1_ratio,
            "spectral_gap": self.spectral_gap,
            "effective_rank": self.effective_rank,
            "condition_number": self.condition_number,
        }


# ── Core: Participation Ratio ────────────────────────────────────────────────

def participation_ratio(
    tensor: Optional[torch.Tensor],
    window_size: int = 16,
) -> float:
    """
    Compute Participation Ratio (PR) — effective dimensionality.

    PR = (Σλᵢ²)² / Σ(λᵢ²)²

    where λᵢ are singular values from SVD of the last W tokens.

    Args:
        tensor: Shape (batch, seq, dim) or (seq, dim).
        window_size: Number of tokens from end. Default: 16.

    Returns:
        PR value (float). NaN if computation fails.

    Measurement contract:
        - SVD in float64 for numerical stability
        - Returns NaN (not 0) for degenerate inputs
        - Requires T >= window_size (no silent truncation)
    """
    if tensor is None:
        return float("nan")

    if tensor.dim() == 3:
        tensor = tensor[0]

    T, D = tensor.shape
    if T < window_size:
        return float("nan")

    # Always compute SVD on CPU to avoid cusolver convergence failures
    # that trigger CUDA device-side asserts and corrupt the CUDA context
    W = min(window_size, T)
    v_cpu = tensor[-W:, :].cpu().double()

    # Guard against NaN/Inf in activations (e.g. fp16 overflow at deep layers)
    if torch.isnan(v_cpu).any() or torch.isinf(v_cpu).any():
        return float("nan")

    try:
        U, S, Vt = torch.linalg.svd(v_cpu.T, full_matrices=False)
        S_np = S.numpy()
    except Exception:
        return float("nan")

    S_sq = S_np ** 2
    total_variance = S_sq.sum()
    if total_variance < 1e-10:
        return float("nan")

    pr = (S_sq.sum() ** 2) / (S_sq ** 2).sum()
    return float(pr)


# ── R_V Metric ───────────────────────────────────────────────────────────────

def compute_rv(
    v_early: Optional[torch.Tensor],
    v_late: Optional[torch.Tensor],
    window: int = 16,
) -> float:
    """
    Compute R_V = PR_late / PR_early from pre-captured tensors.

    Args:
        v_early: V-projection at early layer.
        v_late: V-projection at late layer.
        window: Window size for PR.

    Returns:
        R_V value. R_V < 1 = contraction, R_V > 1 = expansion.
    """
    pr_early = participation_ratio(v_early, window)
    pr_late = participation_ratio(v_late, window)

    if pr_early == 0 or np.isnan(pr_early) or np.isnan(pr_late):
        return float("nan")

    return float(pr_late / pr_early)


def compute_rv_with_components(
    v_early: Optional[torch.Tensor],
    v_late: Optional[torch.Tensor],
    window: int = 16,
) -> Tuple[float, float, float]:
    """
    Compute R_V with separate PR components.

    Returns:
        (rv, pr_early, pr_late)
    """
    pr_early = participation_ratio(v_early, window)
    pr_late = participation_ratio(v_late, window)

    if pr_early == 0 or np.isnan(pr_early) or np.isnan(pr_late):
        return (float("nan"), float("nan"), float("nan"))

    rv = float(pr_late / pr_early)
    return (rv, float(pr_early), float(pr_late))


# ── Spectral Stats ───────────────────────────────────────────────────────────

def compute_spectral_stats(
    tensor: Optional[torch.Tensor],
    window_size: int = 16,
) -> SpectralStats:
    """
    Compute spectral shape statistics from V-projection (or any tensor).

    Beyond PR (single number), reveals the shape of the spectrum:
    - top1_ratio: variance dominance of first direction
    - spectral_gap: separation between top two directions
    - effective_rank: exp(entropy of normalized σ²)
    - condition_number: numerical stability indicator

    Args:
        tensor: Shape (batch, seq, dim) or (seq, dim).
        window_size: Tokens from end.

    Returns:
        SpectralStats dataclass. NaN-filled on failure.
    """
    nan_result = SpectralStats(
        top1_ratio=float("nan"), spectral_gap=float("nan"),
        effective_rank=float("nan"), condition_number=float("nan"),
    )

    if tensor is None:
        return nan_result

    if tensor.dim() == 3:
        tensor = tensor[0]

    T, D = tensor.shape
    W = min(window_size, T)
    if W == 0:
        return nan_result

    v_cpu = tensor[-W:, :].cpu().float()

    try:
        U, S, Vt = torch.linalg.svd(v_cpu.T, full_matrices=False)
        S_np = S.numpy()

        if len(S_np) == 0 or S_np.sum() < 1e-10:
            return nan_result

        top1_ratio = float(S_np[0] / S_np.sum())
        spectral_gap = float(S_np[0] - S_np[1]) if len(S_np) >= 2 else float(S_np[0])

        S_sq = S_np ** 2
        p = S_sq / S_sq.sum()
        p = p[p > 1e-10]
        entropy = -np.sum(p * np.log(p))
        effective_rank = float(np.exp(entropy))

        condition_number = float(S_np[0] / S_np[-1]) if S_np[-1] > 1e-10 else float("inf")

        return SpectralStats(
            top1_ratio=top1_ratio,
            spectral_gap=spectral_gap,
            effective_rank=effective_rank,
            condition_number=condition_number,
        )
    except Exception:
        return nan_result


# ── Cosine Similarity ────────────────────────────────────────────────────────

def compute_cosine_similarity(
    v_early: Optional[torch.Tensor],
    v_late: Optional[torch.Tensor],
    window_size: int = 16,
) -> float:
    """
    Cosine similarity between early and late layer representations.

    Complements R_V (dimensionality) with directional information.
    High cosine = representations point in same direction.

    Args:
        v_early: V-projection at early layer (batch, seq, dim) or (seq, dim)
        v_late: V-projection at late layer
        window_size: Tokens from end

    Returns:
        Cosine similarity in [-1, 1]. NaN on failure.
    """
    if v_early is None or v_late is None:
        return float("nan")

    if v_early.dim() == 3:
        v_early = v_early[0]
    if v_late.dim() == 3:
        v_late = v_late[0]

    W = min(window_size, v_early.shape[0], v_late.shape[0])
    if W == 0:
        return float("nan")

    early_vec = v_early[-W:, :].float().mean(dim=0)
    late_vec = v_late[-W:, :].float().mean(dim=0)

    norm_early = torch.norm(early_vec)
    norm_late = torch.norm(late_vec)

    if norm_early < 1e-10 or norm_late < 1e-10:
        return float("nan")

    cos = torch.dot(early_vec, late_vec) / (norm_early * norm_late)
    return float(cos.cpu().item())


# ── Attention Entropy ────────────────────────────────────────────────────────

def compute_attention_entropy(
    attn_weights: Optional[torch.Tensor],
    head: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Compute attention entropy and max weight from attention pattern tensor.

    Args:
        attn_weights: Shape (batch, num_heads, seq, seq).
        head: Specific head index. None = average across heads.

    Returns:
        (entropy, max_weight). NaN on failure.
    """
    if attn_weights is None:
        return (float("nan"), float("nan"))

    try:
        if head is not None:
            attn = attn_weights[0, head, -1, :]  # Last query position
        else:
            attn = attn_weights[0, :, -1, :].mean(dim=0)

        attn = attn.float()
        attn = attn / (attn.sum() + 1e-10)

        log_attn = torch.log(attn + 1e-10)
        entropy = -(attn * log_attn).sum().item()
        max_weight = attn.max().item()

        return (float(entropy), float(max_weight))
    except Exception:
        return (float("nan"), float("nan"))


# ── Eigenvalue Dominance ─────────────────────────────────────────────────────

def compute_eigenvalue_dominance(
    tensor: Optional[torch.Tensor],
    window_size: int = 16,
    top_k: int = 5,
) -> Dict[str, float]:
    """
    Compute eigenvalue dominance statistics.

    λ₁/Σλ measures how much variance is captured by the top direction.
    High dominance during self-referential processing = representation
    collapsing onto a single eigenstate.

    Args:
        tensor: Shape (batch, seq, dim) or (seq, dim).
        window_size: Tokens from end.
        top_k: Number of top eigenvalues to report.

    Returns:
        Dict with dominance ratio, top-k eigenvalue ratios, and Herfindahl index.
    """
    if tensor is None:
        return {"dominance": float("nan")}

    if tensor.dim() == 3:
        tensor = tensor[0]

    T, D = tensor.shape
    W = min(window_size, T)
    if W == 0:
        return {"dominance": float("nan")}

    v_cpu = tensor[-W:, :].cpu().double()

    try:
        U, S, Vt = torch.linalg.svd(v_cpu.T, full_matrices=False)
        S_np = S.numpy()
        S_sq = S_np ** 2
        total = S_sq.sum()

        if total < 1e-10:
            return {"dominance": float("nan")}

        p = S_sq / total
        dominance = float(p[0])

        # Top-k ratios
        top_k_ratios = {f"lambda_{i+1}_ratio": float(p[i]) for i in range(min(top_k, len(p)))}

        # Herfindahl index (sum of squared proportions)
        herfindahl = float(np.sum(p ** 2))

        return {
            "dominance": dominance,
            "herfindahl": herfindahl,
            **top_k_ratios,
        }
    except Exception:
        return {"dominance": float("nan")}
