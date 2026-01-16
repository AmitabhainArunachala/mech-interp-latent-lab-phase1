"""
Nanda-Standard Baseline Metrics Suite.

This module provides a unified interface for computing ALL standard metrics
that EVERY experiment should report. Following Nanda (2023): "Logit difference
is a fantastic metric" because it's linear in the residual stream.

Usage:
    from src.metrics.baseline_suite import BaselineMetricsSuite
    
    suite = BaselineMetricsSuite(model, tokenizer, device)
    
    # Single prompt analysis
    metrics = suite.compute_all(prompt, return_trajectories=False)
    
    # Comparison analysis (recursive vs baseline)
    comparison = suite.compute_comparison(recursive_prompt, baseline_prompt)
    
    # Validate results meet requirements
    suite.validate_results(comparison)

Required metrics for publication-grade claims:
1. R_V (geometric contraction) - our novel metric
2. logit_diff (Nanda-standard linear metric for attribution)
3. logit_lens (crystallization point)
4. activation_norms (diagnostic for intervention effects)
5. mode_score_m (behavioral mode classifier)
"""

from __future__ import annotations

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .rv import compute_rv, participation_ratio
from .mode_score import ModeScoreMetric
from .logit_lens import compute_logit_lens_trajectory, LogitLensResult
from .logit_diff import LogitDiffMetric, LogitDiffResult
from .extended import (
    compute_cosine_similarity,
    compute_spectral_stats,
    compute_attention_entropy,
    SpectralStats,
)


@dataclass
class BaselineMetrics:
    """Container for all baseline metrics for a single prompt."""

    # Geometric (our novel metric)
    rv: float
    pr_early: float  # Participation ratio at early layer
    pr_late: float   # Participation ratio at late layer

    # Nanda-standard (linear in residual stream)
    logit_diff: float
    logit_diff_details: Optional[LogitDiffResult] = None

    # Logit lens
    crystallization_layer: Optional[int] = None
    min_entropy_layer: Optional[int] = None
    final_prediction: Optional[str] = None
    final_prob: Optional[float] = None

    # Mode score (behavioral classifier)
    mode_score_m: Optional[float] = None

    # Activation norms (diagnostic)
    residual_norm_early: Optional[float] = None
    residual_norm_late: Optional[float] = None

    # === EXTENDED METRICS (publication-grade) ===

    # Directional alignment (complements R_V dimensionality)
    cosine_early_late: Optional[float] = None

    # Spectral shape at early layer
    spectral_early_top1_ratio: Optional[float] = None
    spectral_early_spectral_gap: Optional[float] = None
    spectral_early_effective_rank: Optional[float] = None

    # Spectral shape at late layer
    spectral_late_top1_ratio: Optional[float] = None
    spectral_late_spectral_gap: Optional[float] = None
    spectral_late_effective_rank: Optional[float] = None

    # Attention focus at readout layer
    attention_entropy: Optional[float] = None
    attention_max_weight: Optional[float] = None

    # Trajectories (optional, for detailed analysis)
    logit_diff_trajectory: Optional[List[LogitDiffResult]] = field(default=None, repr=False)
    logit_lens_trajectory: Optional[List[LogitLensResult]] = field(default=None, repr=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to flat dictionary for CSV/JSON output."""
        d = {
            # Core metrics
            "rv": self.rv,
            "pr_early": self.pr_early,
            "pr_late": self.pr_late,
            "logit_diff": self.logit_diff,
            "crystallization_layer": self.crystallization_layer,
            "min_entropy_layer": self.min_entropy_layer,
            "final_prediction": self.final_prediction,
            "final_prob": self.final_prob,
            "mode_score_m": self.mode_score_m,
            "residual_norm_early": self.residual_norm_early,
            "residual_norm_late": self.residual_norm_late,
            # Extended metrics (publication-grade)
            "cosine_early_late": self.cosine_early_late,
            "spectral_early_top1_ratio": self.spectral_early_top1_ratio,
            "spectral_early_spectral_gap": self.spectral_early_spectral_gap,
            "spectral_early_effective_rank": self.spectral_early_effective_rank,
            "spectral_late_top1_ratio": self.spectral_late_top1_ratio,
            "spectral_late_spectral_gap": self.spectral_late_spectral_gap,
            "spectral_late_effective_rank": self.spectral_late_effective_rank,
            "attention_entropy": self.attention_entropy,
            "attention_max_weight": self.attention_max_weight,
        }

        # Add logit diff details if available
        if self.logit_diff_details:
            d["logit_diff_top_recursive_token"] = self.logit_diff_details.top_recursive_token
            d["logit_diff_top_recursive_logit"] = self.logit_diff_details.top_recursive_logit
            d["logit_diff_top_baseline_token"] = self.logit_diff_details.top_baseline_token
            d["logit_diff_top_baseline_logit"] = self.logit_diff_details.top_baseline_logit

        return d


@dataclass
class ComparisonMetrics:
    """Comparison metrics between recursive and baseline prompts."""
    
    recursive: BaselineMetrics
    baseline: BaselineMetrics
    
    # Deltas
    rv_delta: float
    logit_diff_delta: float
    mode_score_delta: Optional[float]
    
    # Effect sizes (Cohen's d approximation where applicable)
    # Note: True effect size requires multiple samples
    rv_effect: float  # (rec - base) / pooled_std_estimate
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to flat dictionary."""
        rec_dict = {f"rec_{k}": v for k, v in self.recursive.to_dict().items()}
        base_dict = {f"base_{k}": v for k, v in self.baseline.to_dict().items()}
        
        return {
            **rec_dict,
            **base_dict,
            "rv_delta": self.rv_delta,
            "logit_diff_delta": self.logit_diff_delta,
            "mode_score_delta": self.mode_score_delta,
            "rv_effect": self.rv_effect,
        }


class BaselineMetricsSuite:
    """
    Nanda-standard baseline metrics suite.
    
    Run this with EVERY experiment for consistent measurement.
    """
    
    REQUIRED_METRICS = ["rv", "logit_diff", "activation_norms"]
    RECOMMENDED_METRICS = ["logit_lens", "mode_score_m"]
    
    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
        early_layer: int = 5,
        late_layer: Optional[int] = None,
        window_size: int = 16,
    ):
        """
        Initialize baseline metrics suite.
        
        Args:
            model: HuggingFace model
            tokenizer: Tokenizer
            device: Compute device
            early_layer: Early layer for R_V computation (default: 5)
            late_layer: Late layer for R_V (default: num_layers - 5)
            window_size: Token window for PR computation (default: 16)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.early_layer = early_layer
        self.late_layer = late_layer or (model.config.num_hidden_layers - 5)
        self.window_size = window_size
        
        # Initialize component metrics
        self._logit_diff_metric = None
        self._mode_score_metric = None
        
        print(f"[BaselineMetricsSuite] Initialized with:")
        print(f"  Early layer: {self.early_layer}")
        print(f"  Late layer: {self.late_layer}")
        print(f"  Window size: {self.window_size}")
    
    @property
    def logit_diff_metric(self) -> LogitDiffMetric:
        """Lazy initialization of LogitDiffMetric."""
        if self._logit_diff_metric is None:
            self._logit_diff_metric = LogitDiffMetric(self.tokenizer, device=self.device)
        return self._logit_diff_metric
    
    @property
    def mode_score_metric(self) -> ModeScoreMetric:
        """Lazy initialization of ModeScoreMetric."""
        if self._mode_score_metric is None:
            self._mode_score_metric = ModeScoreMetric(self.tokenizer, device=self.device)
        return self._mode_score_metric
    
    def compute_all(
        self,
        prompt: str,
        baseline_logits: Optional[torch.Tensor] = None,
        return_trajectories: bool = False,
    ) -> BaselineMetrics:
        """
        Compute all baseline metrics for a single prompt.
        
        Args:
            prompt: Input text
            baseline_logits: Baseline logits for mode_score_m (optional)
            return_trajectories: Include per-layer trajectories
        
        Returns:
            BaselineMetrics dataclass with all metrics
        """
        # === R_V (Geometric) ===
        rv = compute_rv(
            self.model, self.tokenizer, prompt,
            early=self.early_layer, late=self.late_layer,
            window=self.window_size, device=self.device
        )
        
        # Get hidden states for other metrics
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        hidden_states = outputs.hidden_states
        logits = outputs.logits
        
        # === Participation Ratios (for detailed analysis) ===
        pr_early = self._compute_pr_at_layer(hidden_states, self.early_layer)
        pr_late = self._compute_pr_at_layer(hidden_states, self.late_layer)
        
        # === Logit Difference (Nanda-standard) ===
        logit_diff_result = self.logit_diff_metric.compute(logits, position=-1)
        
        logit_diff_trajectory = None
        if return_trajectories:
            logit_diff_trajectory = self.logit_diff_metric.compute_trajectory(
                hidden_states, self.model, position=-1
            )
        
        # === Logit Lens ===
        logit_lens_results, logit_lens_meta = compute_logit_lens_trajectory(
            self.model, self.tokenizer, prompt,
            target_position=-1, top_k=5, device=self.device
        )
        
        logit_lens_trajectory = logit_lens_results if return_trajectories else None
        
        # === Mode Score M ===
        mode_score = None
        if baseline_logits is not None:
            try:
                mode_score = self.mode_score_metric.compute_score(
                    logits, baseline_logits=baseline_logits
                )
            except Exception:
                mode_score = None
        
        # === Activation Norms ===
        residual_norm_early = hidden_states[self.early_layer][0, -self.window_size:, :].norm().item()
        residual_norm_late = hidden_states[self.late_layer][0, -self.window_size:, :].norm().item()

        # === EXTENDED METRICS (publication-grade) ===

        # Get V-projections for extended metrics
        from ..core.hooks import capture_v_projection

        v_early = None
        v_late = None

        with capture_v_projection(self.model, self.early_layer) as storage_early:
            with torch.no_grad():
                self.model(**inputs)
            v_early = storage_early.get("v")

        with capture_v_projection(self.model, self.late_layer) as storage_late:
            with torch.no_grad():
                self.model(**inputs)
            v_late = storage_late.get("v")

        # Cosine similarity (directional alignment)
        cosine = compute_cosine_similarity(v_early, v_late, self.window_size)

        # Spectral stats at both layers
        spectral_early = compute_spectral_stats(v_early, self.window_size)
        spectral_late = compute_spectral_stats(v_late, self.window_size)

        # Attention entropy at late layer
        attn_entropy, attn_max = compute_attention_entropy(
            self.model, self.tokenizer, prompt, self.late_layer,
            head=None, device=self.device
        )

        return BaselineMetrics(
            rv=rv,
            pr_early=pr_early,
            pr_late=pr_late,
            logit_diff=logit_diff_result.logit_diff,
            logit_diff_details=logit_diff_result,
            crystallization_layer=logit_lens_meta["crystallization_layer"],
            min_entropy_layer=logit_lens_meta["min_entropy_layer"],
            final_prediction=logit_lens_meta["final_prediction"],
            final_prob=logit_lens_meta["final_prob"],
            mode_score_m=mode_score,
            residual_norm_early=residual_norm_early,
            residual_norm_late=residual_norm_late,
            # Extended metrics
            cosine_early_late=cosine,
            spectral_early_top1_ratio=spectral_early.top1_ratio,
            spectral_early_spectral_gap=spectral_early.spectral_gap,
            spectral_early_effective_rank=spectral_early.effective_rank,
            spectral_late_top1_ratio=spectral_late.top1_ratio,
            spectral_late_spectral_gap=spectral_late.spectral_gap,
            spectral_late_effective_rank=spectral_late.effective_rank,
            attention_entropy=attn_entropy,
            attention_max_weight=attn_max,
            logit_diff_trajectory=logit_diff_trajectory,
            logit_lens_trajectory=logit_lens_trajectory,
        )
    
    def compute_comparison(
        self,
        recursive_prompt: str,
        baseline_prompt: str,
        return_trajectories: bool = False,
    ) -> ComparisonMetrics:
        """
        Compare recursive vs baseline prompt with all metrics.
        
        Args:
            recursive_prompt: Recursive/experimental prompt
            baseline_prompt: Baseline/control prompt
            return_trajectories: Include per-layer trajectories
        
        Returns:
            ComparisonMetrics with both prompt metrics and deltas
        """
        # First compute baseline to get logits for mode score
        base_inputs = self.tokenizer(baseline_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            base_outputs = self.model(**base_inputs)
        base_logits = base_outputs.logits
        
        # Compute full metrics
        rec_metrics = self.compute_all(
            recursive_prompt, baseline_logits=base_logits,
            return_trajectories=return_trajectories
        )
        base_metrics = self.compute_all(
            baseline_prompt, baseline_logits=base_logits,
            return_trajectories=return_trajectories
        )
        
        # Compute deltas
        rv_delta = rec_metrics.rv - base_metrics.rv
        logit_diff_delta = rec_metrics.logit_diff - base_metrics.logit_diff
        
        mode_score_delta = None
        if rec_metrics.mode_score_m is not None and base_metrics.mode_score_m is not None:
            mode_score_delta = rec_metrics.mode_score_m - base_metrics.mode_score_m
        
        # Rough effect size (pooled std estimate using range/4)
        rv_range = abs(rec_metrics.rv - base_metrics.rv)
        rv_effect = rv_delta / max(rv_range / 4, 0.001)  # Avoid division by zero
        
        return ComparisonMetrics(
            recursive=rec_metrics,
            baseline=base_metrics,
            rv_delta=rv_delta,
            logit_diff_delta=logit_diff_delta,
            mode_score_delta=mode_score_delta,
            rv_effect=rv_effect,
        )
    
    def compute_batch_statistics(
        self,
        recursive_prompts: List[str],
        baseline_prompts: List[str],
    ) -> Dict[str, Any]:
        """
        Compute statistics over multiple prompt pairs.
        
        Returns summary statistics with proper effect sizes and p-values.
        """
        comparisons = []
        
        for rec, base in zip(recursive_prompts, baseline_prompts):
            try:
                comp = self.compute_comparison(rec, base, return_trajectories=False)
                comparisons.append(comp)
            except Exception as e:
                print(f"  Warning: Failed to compute comparison: {e}")
                continue
        
        if len(comparisons) == 0:
            return {"error": "No valid comparisons"}
        
        # Extract arrays
        rv_recs = np.array([c.recursive.rv for c in comparisons])
        rv_bases = np.array([c.baseline.rv for c in comparisons])
        rv_deltas = np.array([c.rv_delta for c in comparisons])
        
        logit_diff_recs = np.array([c.recursive.logit_diff for c in comparisons])
        logit_diff_bases = np.array([c.baseline.logit_diff for c in comparisons])
        logit_diff_deltas = np.array([c.logit_diff_delta for c in comparisons])
        
        # Compute statistics
        from scipy import stats

        n = len(comparisons)

        # Helper for 95% CI
        def compute_ci_95(arr: np.ndarray) -> Tuple[float, float]:
            """Compute 95% confidence interval for mean."""
            if len(arr) < 2:
                return (float("nan"), float("nan"))
            sem = stats.sem(arr)
            ci = stats.t.interval(0.95, len(arr)-1, loc=np.mean(arr), scale=sem)
            return (float(ci[0]), float(ci[1]))

        # Helper for proper Cohen's d (pooled std)
        def compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
            """Cohen's d with pooled standard deviation."""
            n1, n2 = len(group1), len(group2)
            var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
            # Pooled std
            pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
            if pooled_std < 1e-10:
                return 0.0
            return float((np.mean(group1) - np.mean(group2)) / pooled_std)

        # R_V
        rv_t_stat, rv_p_value = stats.ttest_rel(rv_recs, rv_bases)
        rv_cohens_d = compute_cohens_d(rv_recs, rv_bases)
        rv_delta_ci = compute_ci_95(rv_deltas)

        # Logit diff
        ld_t_stat, ld_p_value = stats.ttest_rel(logit_diff_recs, logit_diff_bases)
        ld_cohens_d = compute_cohens_d(logit_diff_recs, logit_diff_bases)
        ld_delta_ci = compute_ci_95(logit_diff_deltas)

        return {
            "n_pairs": n,

            # R_V statistics
            "rv_recursive_mean": float(np.mean(rv_recs)),
            "rv_recursive_std": float(np.std(rv_recs)),
            "rv_baseline_mean": float(np.mean(rv_bases)),
            "rv_baseline_std": float(np.std(rv_bases)),
            "rv_delta_mean": float(np.mean(rv_deltas)),
            "rv_delta_std": float(np.std(rv_deltas)),
            "rv_delta_ci_95": rv_delta_ci,
            "rv_t_statistic": float(rv_t_stat),
            "rv_p_value": float(rv_p_value),
            "rv_cohens_d": float(rv_cohens_d),

            # Logit difference statistics
            "logit_diff_recursive_mean": float(np.mean(logit_diff_recs)),
            "logit_diff_recursive_std": float(np.std(logit_diff_recs)),
            "logit_diff_baseline_mean": float(np.mean(logit_diff_bases)),
            "logit_diff_baseline_std": float(np.std(logit_diff_bases)),
            "logit_diff_delta_mean": float(np.mean(logit_diff_deltas)),
            "logit_diff_delta_std": float(np.std(logit_diff_deltas)),
            "logit_diff_delta_ci_95": ld_delta_ci,
            "logit_diff_t_statistic": float(ld_t_stat),
            "logit_diff_p_value": float(ld_p_value),
            "logit_diff_cohens_d": float(ld_cohens_d),
        }
    
    def _compute_pr_at_layer(
        self,
        hidden_states: Tuple[torch.Tensor, ...],
        layer_idx: int,
    ) -> float:
        """Compute participation ratio at a specific layer."""
        h = hidden_states[layer_idx]
        # Use last W tokens
        h_window = h[0, -self.window_size:, :]
        return participation_ratio(h_window, window_size=self.window_size)
    
    def validate_results(self, metrics: BaselineMetrics) -> Dict[str, bool]:
        """
        Validate that results contain required metrics.
        
        Returns dict of {metric_name: is_valid}.
        """
        validations = {}
        
        # Required metrics
        validations["rv"] = not np.isnan(metrics.rv)
        validations["logit_diff"] = not np.isnan(metrics.logit_diff)
        validations["activation_norms"] = (
            metrics.residual_norm_early is not None and
            metrics.residual_norm_late is not None
        )
        
        # Recommended metrics
        validations["logit_lens"] = metrics.crystallization_layer is not None
        validations["mode_score_m"] = metrics.mode_score_m is not None
        
        return validations
    
    @staticmethod
    def validate_summary(summary: Dict[str, Any]) -> List[str]:
        """
        Validate that a summary dict contains required baseline metrics.
        
        Returns list of missing required metrics.
        """
        required = ["rv", "logit_diff"]
        recommended = ["mode_score_m", "residual_norm_early", "residual_norm_late"]
        
        missing = []
        for metric in required:
            if metric not in summary or summary[metric] is None:
                missing.append(f"REQUIRED: {metric}")
        
        for metric in recommended:
            if metric not in summary or summary[metric] is None:
                missing.append(f"RECOMMENDED: {metric}")
        
        return missing


# Convenience function for quick integration
def compute_baseline_metrics(
    model,
    tokenizer,
    prompt: str,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    One-liner for computing baseline metrics.
    
    Usage:
        metrics = compute_baseline_metrics(model, tokenizer, "My prompt")
    """
    suite = BaselineMetricsSuite(model, tokenizer, device)
    result = suite.compute_all(prompt)
    return result.to_dict()
