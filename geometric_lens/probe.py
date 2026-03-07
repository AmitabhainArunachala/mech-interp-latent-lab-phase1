"""
GeometricProbe: High-level API for measuring geometric signatures in transformers.

Usage:
    from geometric_lens import GeometricProbe

    # With a model name (auto-loads)
    probe = GeometricProbe("mistralai/Mistral-7B-v0.1", device="cuda")
    result = probe.measure("What is the nature of self-reference?")

    # With a pre-loaded model
    probe = GeometricProbe(model=model, tokenizer=tokenizer)
    result = probe.measure(text, metrics=["rv", "spectral", "cosine"])

    # Batch measurement
    results = probe.measure_batch(texts)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
import numpy as np

from .hooks import (
    capture_v_projection,
    capture_k_projection,
    capture_q_projection,
    capture_attention_patterns,
    capture_hidden_states,
    capture_multi_layer,
)
from .metrics import (
    participation_ratio,
    compute_rv,
    compute_rv_with_components,
    compute_spectral_stats,
    compute_cosine_similarity,
    compute_attention_entropy,
    compute_eigenvalue_dominance,
    SpectralStats,
)
from .models import ModelRegistry, ModelSpec, get_layers

MetricName = Literal[
    "rv", "pr", "spectral", "cosine", "attn_entropy",
    "eigenvalue", "qk_pr", "all",
]


@dataclass
class GeometricResult:
    """Result of a geometric measurement on a single text."""

    text: str

    # Core R_V
    rv: float = float("nan")
    pr_early: float = float("nan")
    pr_late: float = float("nan")

    # Spectral stats (late layer)
    spectral_late: Optional[SpectralStats] = None
    spectral_early: Optional[SpectralStats] = None

    # Directional
    cosine: float = float("nan")

    # Attention
    attn_entropy: float = float("nan")
    attn_max: float = float("nan")

    # Eigenvalue dominance
    eigenvalue_dominance: Optional[Dict[str, float]] = None

    # QK participation ratios
    k_pr_early: float = float("nan")
    k_pr_late: float = float("nan")
    q_pr_early: float = float("nan")
    q_pr_late: float = float("nan")

    # Model info
    model_name: str = ""
    early_layer: int = 0
    late_layer: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to flat dict for JSON serialization."""
        d = {
            "text": self.text[:200],
            "rv": self.rv,
            "pr_early": self.pr_early,
            "pr_late": self.pr_late,
            "cosine": self.cosine,
            "attn_entropy": self.attn_entropy,
            "attn_max": self.attn_max,
            "k_pr_early": self.k_pr_early,
            "k_pr_late": self.k_pr_late,
            "q_pr_early": self.q_pr_early,
            "q_pr_late": self.q_pr_late,
            "model_name": self.model_name,
            "early_layer": self.early_layer,
            "late_layer": self.late_layer,
        }
        if self.spectral_late:
            for k, v in self.spectral_late.to_dict().items():
                d[f"spectral_late_{k}"] = v
        if self.spectral_early:
            for k, v in self.spectral_early.to_dict().items():
                d[f"spectral_early_{k}"] = v
        if self.eigenvalue_dominance:
            for k, v in self.eigenvalue_dominance.items():
                d[f"eigen_{k}"] = v
        return d


class GeometricProbe:
    """
    High-level probe for measuring geometric signatures of computation.

    Handles model loading, tokenization, hook management, and metric computation.
    """

    def __init__(
        self,
        model_name: str = "",
        model: Any = None,
        tokenizer: Any = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "eager",
        window: int = 16,
        max_length: int = 512,
        early_layer: Optional[int] = None,
        late_layer: Optional[int] = None,
    ):
        """
        Initialize probe.

        Args:
            model_name: HuggingFace model name. If provided, model is auto-loaded.
            model: Pre-loaded model (alternative to model_name).
            tokenizer: Pre-loaded tokenizer (required if model is provided).
            device: Target device.
            dtype: Model dtype.
            attn_implementation: Attention implementation ("eager" needed for attn patterns).
            window: Default window size for PR/spectral computations.
            max_length: Max token length for inputs.
            early_layer: Override early layer (default: from registry).
            late_layer: Override late layer (default: from registry).
        """
        self.device = device
        self.window = window
        self.max_length = max_length
        self._registry = ModelRegistry()

        if model is not None:
            self.model = model
            self.tokenizer = tokenizer
            self.model_name = model_name or getattr(model.config, "_name_or_path", "unknown")
        elif model_name:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.model_name = model_name
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map=device,
                attn_implementation=attn_implementation,
            )
        else:
            raise ValueError("Must provide either model_name or model+tokenizer.")

        self.model.eval()

        # Get model spec
        self.spec = self._registry.auto_detect(self.model, self.model_name)

        # Allow overrides
        self.early_layer = early_layer if early_layer is not None else self.spec.early_layer
        self.late_layer = late_layer if late_layer is not None else self.spec.late_layer

    def measure(
        self,
        text: str,
        metrics: Union[List[MetricName], MetricName] = "all",
    ) -> GeometricResult:
        """
        Measure geometric properties of a text.

        Args:
            text: Input text to analyze.
            metrics: Which metrics to compute. "all" = everything.

        Returns:
            GeometricResult with requested metrics.
        """
        if isinstance(metrics, str):
            metrics = [metrics]
        if "all" in metrics:
            metrics = ["rv", "pr", "spectral", "cosine", "attn_entropy", "eigenvalue", "qk_pr"]

        result = GeometricResult(
            text=text,
            model_name=self.model_name,
            early_layer=self.early_layer,
            late_layer=self.late_layer,
        )

        enc = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=self.max_length
        ).to(self.device)

        # ── V-projection metrics ──
        needs_v = any(m in metrics for m in ["rv", "pr", "spectral", "cosine", "eigenvalue"])
        v_early_tensor = None
        v_late_tensor = None

        if needs_v:
            with capture_v_projection(self.model, self.early_layer) as se:
                with torch.no_grad():
                    self.model(**enc)
                v_early_tensor = se.get("v")

            with capture_v_projection(self.model, self.late_layer) as sl:
                with torch.no_grad():
                    self.model(**enc)
                v_late_tensor = sl.get("v")

        if "rv" in metrics or "pr" in metrics:
            rv, pr_e, pr_l = compute_rv_with_components(
                v_early_tensor, v_late_tensor, self.window
            )
            result.rv = rv
            result.pr_early = pr_e
            result.pr_late = pr_l

        if "spectral" in metrics:
            result.spectral_late = compute_spectral_stats(v_late_tensor, self.window)
            result.spectral_early = compute_spectral_stats(v_early_tensor, self.window)

        if "cosine" in metrics:
            result.cosine = compute_cosine_similarity(
                v_early_tensor, v_late_tensor, self.window
            )

        if "eigenvalue" in metrics:
            result.eigenvalue_dominance = compute_eigenvalue_dominance(
                v_late_tensor, self.window
            )

        # ── Attention entropy ──
        if "attn_entropy" in metrics:
            with capture_attention_patterns(self.model, self.late_layer) as sa:
                with torch.no_grad():
                    self.model(**enc, output_attentions=True)
                attn = sa.get("attn_weights")
            entropy, max_w = compute_attention_entropy(attn)
            result.attn_entropy = entropy
            result.attn_max = max_w

        # ── QK participation ratios ──
        if "qk_pr" in metrics:
            with capture_k_projection(self.model, self.early_layer) as sk:
                with torch.no_grad():
                    self.model(**enc)
                k_early = sk.get("k")
            with capture_k_projection(self.model, self.late_layer) as sk:
                with torch.no_grad():
                    self.model(**enc)
                k_late = sk.get("k")
            with capture_q_projection(self.model, self.early_layer) as sq:
                with torch.no_grad():
                    self.model(**enc)
                q_early = sq.get("q")
            with capture_q_projection(self.model, self.late_layer) as sq:
                with torch.no_grad():
                    self.model(**enc)
                q_late = sq.get("q")

            result.k_pr_early = participation_ratio(k_early, self.window)
            result.k_pr_late = participation_ratio(k_late, self.window)
            result.q_pr_early = participation_ratio(q_early, self.window)
            result.q_pr_late = participation_ratio(q_late, self.window)

        return result

    def measure_batch(
        self,
        texts: List[str],
        metrics: Union[List[MetricName], MetricName] = "all",
        progress: bool = True,
    ) -> List[GeometricResult]:
        """
        Measure geometric properties of multiple texts.

        Args:
            texts: List of input texts.
            metrics: Which metrics to compute.
            progress: Print progress.

        Returns:
            List of GeometricResult.
        """
        results = []
        for i, text in enumerate(texts):
            if progress and (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(texts)}]")
            try:
                r = self.measure(text, metrics=metrics)
                results.append(r)
            except Exception as e:
                print(f"  [error on prompt {i}: {e}]")
                results.append(GeometricResult(text=text, model_name=self.model_name))
        return results

    def layer_sweep(
        self,
        text: str,
        component: str = "v",
        window: Optional[int] = None,
    ) -> Dict[int, float]:
        """
        Compute PR at every layer for a given text.

        Args:
            text: Input text.
            component: "v", "k", "q", or "hidden".
            window: Window size (default: self.window).

        Returns:
            Dict mapping layer_idx → PR value.
        """
        w = window or self.window
        enc = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=self.max_length
        ).to(self.device)

        all_layers = list(range(self.spec.num_layers))

        with capture_multi_layer(self.model, all_layers, component=component) as storage:
            with torch.no_grad():
                self.model(**enc)

        return {
            layer_idx: participation_ratio(tensor, w)
            for layer_idx, tensor in storage.items()
        }
