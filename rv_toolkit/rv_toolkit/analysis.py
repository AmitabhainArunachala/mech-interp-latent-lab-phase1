"""
Statistical analysis utilities for R_V experiments.

Implements effect size computation, significance testing, and 
cross-architecture validation analysis.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy import stats


@dataclass
class AnalysisResult:
    """Result of statistical analysis."""
    
    mean_baseline: float
    mean_patched: float
    mean_recursive: float
    mean_delta: float
    std_delta: float
    cohens_d: float
    p_value: float
    n_samples: int
    transfer_efficiency: float
    confidence_interval: Tuple[float, float]
    
    def __repr__(self):
        return (
            f"AnalysisResult(\n"
            f"  n={self.n_samples}\n"
            f"  Δ_RV = {self.mean_delta:.4f} ± {self.std_delta:.4f}\n"
            f"  Cohen's d = {self.cohens_d:.2f}\n"
            f"  p = {self.p_value:.2e}\n"
            f"  Transfer efficiency = {self.transfer_efficiency:.1%}\n"
            f")"
        )


def compute_effect_size(
    baseline_values: np.ndarray,
    treatment_values: np.ndarray,
) -> float:
    """
    Compute Cohen's d effect size.
    
    d = (M1 - M2) / pooled_std
    
    Args:
        baseline_values: Baseline measurements
        treatment_values: Treatment measurements
        
    Returns:
        Cohen's d (negative indicates treatment reduces values)
    """
    n1, n2 = len(baseline_values), len(treatment_values)
    var1, var2 = np.var(baseline_values, ddof=1), np.var(treatment_values, ddof=1)
    
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std < 1e-10:
        return np.nan
    
    return (np.mean(treatment_values) - np.mean(baseline_values)) / pooled_std


def compute_transfer_efficiency(
    baseline_rv: float,
    patched_rv: float,
    target_rv: float,
) -> float:
    """
    Compute transfer efficiency: how much of the target geometry was transferred.
    
    efficiency = (patched - baseline) / (target - baseline)
    
    100% = patching fully reproduces target geometry
    >100% = patching overshoots (common due to context interaction)
    <100% = partial transfer
    
    Args:
        baseline_rv: R_V of unpatched baseline
        patched_rv: R_V after patching
        target_rv: R_V of recursive prompt (target geometry)
        
    Returns:
        Transfer efficiency as fraction
    """
    gap = target_rv - baseline_rv
    
    if abs(gap) < 1e-10:
        return np.nan
    
    return (patched_rv - baseline_rv) / gap


def run_statistical_tests(
    baseline_rvs: List[float],
    patched_rvs: List[float],
    recursive_rvs: List[float] = None,
    alpha: float = 0.05,
) -> AnalysisResult:
    """
    Run comprehensive statistical analysis on patching results.
    
    Tests:
    1. Paired t-test for patching effect
    2. Effect size (Cohen's d)
    3. Transfer efficiency
    4. Confidence intervals
    
    Args:
        baseline_rvs: R_V values before patching
        patched_rvs: R_V values after patching
        recursive_rvs: R_V values for recursive prompts (optional)
        alpha: Significance level for confidence intervals
        
    Returns:
        AnalysisResult with all statistics
    """
    # Clean data
    baseline = np.array([x for x in baseline_rvs if not np.isnan(x)])
    patched = np.array([x for x in patched_rvs if not np.isnan(x)])
    
    n = min(len(baseline), len(patched))
    baseline = baseline[:n]
    patched = patched[:n]
    
    if n < 3:
        return AnalysisResult(
            mean_baseline=np.nan,
            mean_patched=np.nan,
            mean_recursive=np.nan,
            mean_delta=np.nan,
            std_delta=np.nan,
            cohens_d=np.nan,
            p_value=np.nan,
            n_samples=n,
            transfer_efficiency=np.nan,
            confidence_interval=(np.nan, np.nan),
        )
    
    # Basic statistics
    deltas = patched - baseline
    mean_delta = np.mean(deltas)
    std_delta = np.std(deltas, ddof=1)
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(patched, baseline)
    
    # Effect size
    cohens_d = compute_effect_size(baseline, patched)
    
    # Confidence interval for delta
    se = std_delta / np.sqrt(n)
    t_crit = stats.t.ppf(1 - alpha/2, n - 1)
    ci_low = mean_delta - t_crit * se
    ci_high = mean_delta + t_crit * se
    
    # Transfer efficiency
    mean_recursive = np.nan
    transfer_eff = np.nan
    
    if recursive_rvs is not None:
        recursive = np.array([x for x in recursive_rvs if not np.isnan(x)])[:n]
        if len(recursive) > 0:
            mean_recursive = np.mean(recursive)
            transfer_eff = compute_transfer_efficiency(
                np.mean(baseline),
                np.mean(patched),
                mean_recursive,
            )
    
    return AnalysisResult(
        mean_baseline=np.mean(baseline),
        mean_patched=np.mean(patched),
        mean_recursive=mean_recursive,
        mean_delta=mean_delta,
        std_delta=std_delta,
        cohens_d=cohens_d,
        p_value=p_value,
        n_samples=n,
        transfer_efficiency=transfer_eff,
        confidence_interval=(ci_low, ci_high),
    )


def compare_architectures(
    results_by_model: Dict[str, List[Tuple[float, float]]],
) -> Dict[str, AnalysisResult]:
    """
    Compare R_V patching effects across architectures.
    
    Key finding from paper: Effect is universal across architectures,
    with MoE models showing amplified effects (d = -4.21 for Mixtral).
    
    Args:
        results_by_model: Dict mapping model name to list of (baseline_rv, patched_rv) tuples
        
    Returns:
        Dict mapping model name to AnalysisResult
    """
    analysis = {}
    
    for model_name, pairs in results_by_model.items():
        baselines = [p[0] for p in pairs]
        patched = [p[1] for p in pairs]
        
        analysis[model_name] = run_statistical_tests(baselines, patched)
    
    return analysis


def detect_homeostasis(
    layer_deltas: Dict[int, float],
    intervention_layer: int = 27,
) -> Dict:
    """
    Detect geometric homeostasis pattern in cross-layer effects.
    
    Key finding: While intervention at L27 contracts geometry,
    downstream layers (L28-L31) show compensatory expansion,
    demonstrating system-level geometric regulation.
    
    Args:
        layer_deltas: Dict mapping layer to R_V change
        intervention_layer: Layer where patching occurred
        
    Returns:
        Dict with homeostasis metrics
    """
    upstream_layers = [l for l in layer_deltas if l < intervention_layer]
    downstream_layers = [l for l in layer_deltas if l > intervention_layer]
    
    upstream_deltas = [layer_deltas[l] for l in upstream_layers]
    downstream_deltas = [layer_deltas[l] for l in downstream_layers]
    
    intervention_delta = layer_deltas.get(intervention_layer, 0)
    
    # Check for compensation
    downstream_mean = np.mean(downstream_deltas) if downstream_deltas else 0
    
    # Compensation detected if downstream expands while intervention contracts
    compensation_detected = (
        intervention_delta < 0 and 
        downstream_mean > 0 and
        len(downstream_deltas) > 0
    )
    
    # Correlation between intervention and downstream (expect negative = compensation)
    if len(downstream_deltas) >= 3:
        # Use layer distance as predictor
        distances = [l - intervention_layer for l in downstream_layers]
        corr, _ = stats.pearsonr(distances, downstream_deltas)
    else:
        corr = np.nan
    
    return {
        "intervention_layer": intervention_layer,
        "intervention_delta": intervention_delta,
        "downstream_mean_delta": downstream_mean,
        "compensation_detected": compensation_detected,
        "distance_correlation": corr,
        "upstream_layers": upstream_layers,
        "downstream_layers": downstream_layers,
    }


def bootstrap_confidence_interval(
    values: np.ndarray,
    statistic: callable = np.mean,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval for a statistic.
    
    Args:
        values: Data values
        statistic: Function to compute (default: mean)
        n_bootstrap: Number of bootstrap samples
        alpha: Significance level
        
    Returns:
        (lower, upper) confidence interval bounds
    """
    n = len(values)
    bootstrap_stats = []
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=n, replace=True)
        bootstrap_stats.append(statistic(sample))
    
    bootstrap_stats = np.array(bootstrap_stats)
    
    lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    
    return (lower, upper)
