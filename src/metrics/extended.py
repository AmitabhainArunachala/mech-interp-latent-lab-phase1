"""
Extended metrics for publication-grade R_V research.

Complements R_V (dimensionality) with:
1. Cosine Similarity - directional alignment between early/late representations
2. Spectral Shape Stats - top-1 ratio, spectral gap, effective rank
3. Attention Entropy - focus/diffuseness at readout layer

These are cheap to compute (we already have the tensors) and add interpretive value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


@dataclass
class SpectralStats:
    """Spectral shape statistics from SVD singular values."""

    top1_ratio: float        # σ₁ / Σσᵢ — dominance of first component
    spectral_gap: float      # σ₁ - σ₂ — separation of top direction
    effective_rank: float    # exp(entropy of normalized σ²) — another dimensionality measure
    condition_number: float  # σ_max / σ_min — numerical stability indicator

    def to_dict(self) -> Dict[str, float]:
        return {
            "top1_ratio": self.top1_ratio,
            "spectral_gap": self.spectral_gap,
            "effective_rank": self.effective_rank,
            "condition_number": self.condition_number,
        }


@dataclass
class ExtendedMetrics:
    """Extended metrics complementing R_V."""

    # Directional alignment
    cosine_early_late: float       # cos(early_repr, late_repr)

    # Spectral shape at early layer
    spectral_early: SpectralStats

    # Spectral shape at late layer
    spectral_late: SpectralStats

    # Attention entropy at readout layer
    attention_entropy: float       # H(attention weights)
    attention_max_weight: float    # Max attention weight (focus indicator)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cosine_early_late": self.cosine_early_late,
            "spectral_early_top1_ratio": self.spectral_early.top1_ratio,
            "spectral_early_spectral_gap": self.spectral_early.spectral_gap,
            "spectral_early_effective_rank": self.spectral_early.effective_rank,
            "spectral_late_top1_ratio": self.spectral_late.top1_ratio,
            "spectral_late_spectral_gap": self.spectral_late.spectral_gap,
            "spectral_late_effective_rank": self.spectral_late.effective_rank,
            "attention_entropy": self.attention_entropy,
            "attention_max_weight": self.attention_max_weight,
        }


def compute_cosine_similarity(
    v_early: torch.Tensor,
    v_late: torch.Tensor,
    window_size: int = 16,
) -> float:
    """
    Compute cosine similarity between early and late layer representations.

    Complements R_V (dimensionality) with directional information.
    High cosine = representations point in same direction.

    Args:
        v_early: V-projection at early layer (batch, seq, hidden) or (seq, hidden)
        v_late: V-projection at late layer
        window_size: Number of tokens to use from end

    Returns:
        Cosine similarity in [-1, 1]. Returns NaN on failure.
    """
    if v_early is None or v_late is None:
        return float("nan")

    # Handle batch dimension
    if v_early.dim() == 3:
        v_early = v_early[0]
    if v_late.dim() == 3:
        v_late = v_late[0]

    T_early, D = v_early.shape
    T_late, _ = v_late.shape

    W = min(window_size, T_early, T_late)
    if W == 0:
        return float("nan")

    # Extract last W tokens and flatten to single vector (mean pooling)
    early_vec = v_early[-W:, :].float().mean(dim=0)
    late_vec = v_late[-W:, :].float().mean(dim=0)

    # Compute cosine similarity
    norm_early = torch.norm(early_vec)
    norm_late = torch.norm(late_vec)

    if norm_early < 1e-10 or norm_late < 1e-10:
        return float("nan")

    cos_sim = torch.dot(early_vec, late_vec) / (norm_early * norm_late)
    return float(cos_sim.cpu().item())


def compute_spectral_stats(
    v_tensor: torch.Tensor,
    window_size: int = 16,
) -> SpectralStats:
    """
    Compute spectral shape statistics from V-projection.

    Beyond PR (single number), these reveal the *shape* of the spectrum:
    - top1_ratio: Is variance dominated by one direction?
    - spectral_gap: How separated is the top direction?
    - effective_rank: Alternative to PR, exp(entropy)
    - condition_number: Numerical stability

    Args:
        v_tensor: V-projection tensor (batch, seq, hidden) or (seq, hidden)
        window_size: Number of tokens to use from end

    Returns:
        SpectralStats dataclass. Returns NaN-filled on failure.
    """
    nan_result = SpectralStats(
        top1_ratio=float("nan"),
        spectral_gap=float("nan"),
        effective_rank=float("nan"),
        condition_number=float("nan"),
    )

    if v_tensor is None:
        return nan_result

    # Handle batch dimension
    if v_tensor.dim() == 3:
        v_tensor = v_tensor[0]

    T, D = v_tensor.shape
    W = min(window_size, T)

    if W == 0:
        return nan_result

    v_window = v_tensor[-W:, :].float()

    try:
        # SVD
        U, S, Vt = torch.linalg.svd(v_window.T, full_matrices=False)
        S_np = S.cpu().numpy()

        if len(S_np) == 0 or S_np.sum() < 1e-10:
            return nan_result

        # Top-1 ratio: σ₁ / Σσᵢ
        top1_ratio = float(S_np[0] / S_np.sum())

        # Spectral gap: σ₁ - σ₂
        if len(S_np) >= 2:
            spectral_gap = float(S_np[0] - S_np[1])
        else:
            spectral_gap = float(S_np[0])

        # Effective rank: exp(entropy of normalized σ²)
        S_sq = S_np ** 2
        p = S_sq / S_sq.sum()
        p = p[p > 1e-10]  # Filter near-zero for log stability
        entropy = -np.sum(p * np.log(p))
        effective_rank = float(np.exp(entropy))

        # Condition number: σ_max / σ_min
        if S_np[-1] > 1e-10:
            condition_number = float(S_np[0] / S_np[-1])
        else:
            condition_number = float("inf")

        return SpectralStats(
            top1_ratio=top1_ratio,
            spectral_gap=spectral_gap,
            effective_rank=effective_rank,
            condition_number=condition_number,
        )

    except Exception:
        return nan_result


def compute_attention_entropy(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    text: str,
    layer: int,
    head: Optional[int] = None,
    device: str = "cuda",
) -> Tuple[float, float]:
    """
    Compute attention entropy at specified layer.

    Measures how "focused" vs "diffuse" attention is.
    Low entropy = focused on few positions (sharp attention).
    High entropy = spread across many positions (diffuse).

    Args:
        model: Transformer model
        tokenizer: Tokenizer
        text: Input text
        layer: Layer index to measure
        head: Specific head (None = average across heads)
        device: Compute device

    Returns:
        Tuple of (entropy, max_weight). Returns (NaN, NaN) on failure.
    """
    try:
        enc = tokenizer(text, return_tensors="pt", truncation=True).to(device)

        with torch.no_grad():
            outputs = model(**enc, output_attentions=True)

        # Get attention weights at specified layer
        # Shape: (batch, num_heads, seq_len, seq_len)
        attn = outputs.attentions[layer]

        if head is not None:
            # Specific head
            attn_weights = attn[0, head, -1, :]  # Last token's attention
        else:
            # Average across heads
            attn_weights = attn[0, :, -1, :].mean(dim=0)

        # Compute entropy
        attn_np = attn_weights.cpu().numpy()
        attn_np = attn_np[attn_np > 1e-10]  # Filter zeros

        if len(attn_np) == 0:
            return float("nan"), float("nan")

        # Normalize (should already sum to 1, but ensure)
        attn_np = attn_np / attn_np.sum()

        entropy = float(-np.sum(attn_np * np.log(attn_np + 1e-10)))
        max_weight = float(attn_weights.max().cpu().item())

        return entropy, max_weight

    except Exception:
        return float("nan"), float("nan")


def compute_extended_metrics(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    text: str,
    early_layer: int = 5,
    late_layer: int = 27,
    window_size: int = 16,
    device: str = "cuda",
) -> ExtendedMetrics:
    """
    Compute all extended metrics for a prompt.

    This is the main entry point. Call after you've already computed R_V.

    Args:
        model: Transformer model
        tokenizer: Tokenizer
        text: Input prompt
        early_layer: Early layer index (default: 5)
        late_layer: Late layer index (default: 27)
        window_size: Token window (default: 16)
        device: Compute device

    Returns:
        ExtendedMetrics dataclass with all metrics
    """
    from ..core.hooks import capture_v_projection

    # Capture V-projections at both layers
    enc = tokenizer(text, return_tensors="pt", truncation=True).to(device)

    v_early = None
    v_late = None

    with capture_v_projection(model, early_layer) as storage_early:
        with torch.no_grad():
            model(**enc)
        v_early = storage_early.get("v")

    with capture_v_projection(model, late_layer) as storage_late:
        with torch.no_grad():
            model(**enc)
        v_late = storage_late.get("v")

    # Compute cosine similarity
    cosine = compute_cosine_similarity(v_early, v_late, window_size)

    # Compute spectral stats at both layers
    spectral_early = compute_spectral_stats(v_early, window_size)
    spectral_late = compute_spectral_stats(v_late, window_size)

    # Compute attention entropy at late layer
    attn_entropy, attn_max = compute_attention_entropy(
        model, tokenizer, text, late_layer, head=None, device=device
    )

    return ExtendedMetrics(
        cosine_early_late=cosine,
        spectral_early=spectral_early,
        spectral_late=spectral_late,
        attention_entropy=attn_entropy,
        attention_max_weight=attn_max,
    )


def compute_extended_batch(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    early_layer: int = 5,
    late_layer: int = 27,
    window_size: int = 16,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Compute extended metrics over a batch and return statistics.

    Args:
        model: Transformer model
        tokenizer: Tokenizer
        prompts: List of prompts
        early_layer: Early layer index
        late_layer: Late layer index
        window_size: Token window
        device: Compute device

    Returns:
        Dict with mean/std/ci for each metric
    """
    from scipy import stats

    results = []
    for prompt in prompts:
        try:
            ext = compute_extended_metrics(
                model, tokenizer, prompt,
                early_layer, late_layer, window_size, device
            )
            results.append(ext.to_dict())
        except Exception as e:
            print(f"  Warning: Failed on prompt: {e}")
            continue

    if len(results) == 0:
        return {"error": "No valid results"}

    # Aggregate statistics
    def compute_ci_95(arr: np.ndarray) -> Tuple[float, float]:
        if len(arr) < 2:
            return (float("nan"), float("nan"))
        sem = stats.sem(arr)
        ci = stats.t.interval(0.95, len(arr)-1, loc=np.mean(arr), scale=sem)
        return (float(ci[0]), float(ci[1]))

    summary = {"n": len(results)}

    # For each metric, compute mean/std/ci
    keys = results[0].keys()
    for key in keys:
        values = np.array([r[key] for r in results if not np.isnan(r[key])])
        if len(values) > 0:
            summary[f"{key}_mean"] = float(np.mean(values))
            summary[f"{key}_std"] = float(np.std(values))
            summary[f"{key}_ci_95"] = compute_ci_95(values)

    return summary
