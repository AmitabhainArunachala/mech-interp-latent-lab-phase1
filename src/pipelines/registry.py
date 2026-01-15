"""
Experiment registry + config validation.

This is the "blessed" mapping from config["experiment"] -> runnable function.

Nanda-Standard Baseline Metrics (2023):
- Every experiment SHOULD include baseline metrics for proper attribution
- Required: rv, logit_diff
- Recommended: logit_lens, mode_score_m, activation_norms
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


class ConfigError(ValueError):
    pass


# Required baseline metrics for publication-grade claims
REQUIRED_BASELINE_METRICS = ["rv", "logit_diff"]
RECOMMENDED_BASELINE_METRICS = ["logit_lens", "mode_score_m", "activation_norms"]


@dataclass
class ExperimentResult:
    """
    Contract for experiment results.
    
    Every experiment returns:
    - summary: Aggregated metrics and statistics
    - baseline_metrics: (Optional but RECOMMENDED) Nanda-standard baseline metrics
    
    Experiments without baseline_metrics will trigger a warning.
    """

    summary: Dict[str, Any]
    baseline_metrics: Optional[Dict[str, Any]] = field(default=None)
    
    def __post_init__(self):
        """Validate and warn about missing baseline metrics."""
        self._validate_baseline_metrics()
    
    def _validate_baseline_metrics(self) -> None:
        """Check for required baseline metrics and warn if missing."""
        if self.baseline_metrics is None:
            # Check if baseline metrics are in summary (legacy support)
            has_rv = "rv" in self.summary or "rv_baseline_mean" in self.summary
            has_logit_diff = "logit_diff" in self.summary or "logit_diff_delta" in self.summary
            
            if not has_rv or not has_logit_diff:
                warnings.warn(
                    f"Experiment '{self.summary.get('experiment', 'unknown')}' missing baseline metrics. "
                    f"For Nanda-standard causal claims, use BaselineMetricsSuite.",
                    UserWarning,
                    stacklevel=3
                )
        else:
            # Validate baseline_metrics contents
            missing = []
            for metric in REQUIRED_BASELINE_METRICS:
                if metric not in self.baseline_metrics:
                    missing.append(metric)
            
            if missing:
                warnings.warn(
                    f"Experiment missing required baseline metrics: {missing}",
                    UserWarning,
                    stacklevel=3
                )
    
    def get_baseline_metric(self, name: str) -> Optional[Any]:
        """Get a baseline metric by name, checking both baseline_metrics and summary."""
        if self.baseline_metrics and name in self.baseline_metrics:
            return self.baseline_metrics[name]
        return self.summary.get(name)
    
    def has_nanda_standard_metrics(self) -> bool:
        """Check if this result has Nanda-standard metrics for attribution."""
        has_rv = self.get_baseline_metric("rv") is not None
        has_logit_diff = self.get_baseline_metric("logit_diff") is not None
        return has_rv and has_logit_diff


ExperimentFn = Callable[[Dict[str, Any], Path], ExperimentResult]


def _require(d: Dict[str, Any], key: str) -> Any:
    if key not in d:
        raise ConfigError(f"Missing required key: {key}")
    return d[key]


def _validate_top_level(cfg: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    exp = _require(cfg, "experiment")
    if not isinstance(exp, str) or not exp.strip():
        raise ConfigError("config['experiment'] must be a non-empty string")
    params = cfg.get("params", {})
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise ConfigError("config['params'] must be an object/dict")
    return exp, params


def get_registry() -> Dict[str, ExperimentFn]:
    # Local imports to keep import-time side effects minimal.
    # Organized by tier: canonical, discovery, archive

    # === CANONICAL (7) - Core paper findings ===
    from .canonical.rv_l27_causal_validation import run_rv_l27_causal_validation_from_config
    from .canonical.confound_validation import run_confound_validation_from_config
    from .canonical.random_direction_control import run_random_direction_control_from_config
    from .canonical.mlp_ablation_necessity import run_mlp_ablation_necessity_from_config
    from .canonical.mlp_sufficiency_test import run_mlp_sufficiency_test_from_config
    from .canonical.mlp_combined_sufficiency_test import run_combined_mlp_sufficiency_test_from_config
    from .canonical.head_ablation_validation import run_head_ablation_validation_from_config

    # === DISCOVERY (12) - Methodology tools ===
    from .discovery.behavioral_grounding import run_behavioral_grounding_from_config
    from .discovery.behavioral_grounding_batch import run_behavioral_grounding_batch_from_config
    from .discovery.path_patching_mechanism import run_path_patching_mechanism_from_config
    from .discovery.temporal_stability import run_temporal_stability_from_config
    from .discovery.hysteresis import run_hysteresis_from_config
    from .discovery.kv_mechanism import run_kv_mechanism_from_config
    from .discovery.layer_sweep import run_layer_sweep_from_config
    from .discovery.logit_lens_analysis import run_logit_lens_analysis_from_config
    from .discovery.vproj_patching_analysis import run_vproj_patching_analysis_from_config
    from .discovery.mlp_vproj_combined_sufficiency_test import run_mlp_vproj_combined_sufficiency_test_from_config
    from .discovery.c2_rv_measurement import run_c2_rv_measurement_from_config

    # === ARCHIVE - Historical/superseded ===
    from .archive.phase1_existence import run_phase1_existence_from_config
    from .cross_architecture_validation import run_cross_architecture_validation_from_config
    from .archive.phase0_minimal_pairs import run_phase0_minimal_pairs_from_config
    from .archive.phase0_metric_targets import run_phase0_metric_targets_from_config
    from .archive.l27_head_analysis import run_l27_head_analysis_from_config
    from .archive.kv_sufficiency_matrix import run_kv_sufficiency_matrix_from_config
    from .archive.behavior_strict import run_behavior_strict_from_config
    from .archive.steering import run_steering_from_config
    from .archive.steering_analysis import run_steering_analysis_from_config
    from .archive.steering_layer_matrix import run_layer_matrix_from_config
    from .archive.minimal_recursive_intervention import run_minimal_recursive_intervention_from_config
    from .archive.extended_context_steering import run_extended_context_steering_from_config
    from .archive.steering_control import run_steering_control_from_config
    from .archive.triple_system_intervention import run_triple_system_intervention_from_config
    from .archive.surgical_sweep import run_surgical_sweep_from_config
    from .archive.verification_sweep import run_verification_experiments
    from .archive.p1_ablation import run_p1_ablation_from_config
    from .archive.sprint_head_specific_steering.pipeline import run_sprint_head_specific_steering
    from .archive.retrocompute_mode_score import run_retrocompute_mode_score_from_config
    from .archive.ioi_causal_test import run_ioi_causal_test_from_config
    from .archive.importance_sweep import run_importance_sweep_from_config
    from .archive.geometry_behavior import run_geometry_behavior_from_config
    from .archive.source_isolation_diagnostic import run_source_isolation_diagnostic_from_config
    from .archive.kitchen_sink import run_kitchen_sink_from_config
    from .archive.circuit_discovery import run_circuit_discovery_from_config
    from .archive.mlp_steering_sweep import run_mlp_steering_sweep_from_config
    from .archive.mlp_ablation_position_specific import run_position_specific_ablation_from_config

    return {
        "phase0_minimal_pairs": run_phase0_minimal_pairs_from_config,
        "phase0_metric_targets": run_phase0_metric_targets_from_config,
        "phase1_existence": run_phase1_existence_from_config,
        "rv_l27_causal_validation": run_rv_l27_causal_validation_from_config,
        "l27_head_analysis": run_l27_head_analysis_from_config,
        "path_patching_mechanism": run_path_patching_mechanism_from_config,
        "behavioral_grounding": run_behavioral_grounding_from_config,
        "behavioral_grounding_batch": run_behavioral_grounding_batch_from_config,
        "confound_validation": run_confound_validation_from_config,
        "kv_sufficiency_matrix": run_kv_sufficiency_matrix_from_config,
        "head_ablation_validation": run_head_ablation_validation_from_config,
        "temporal_stability": run_temporal_stability_from_config,
        "hysteresis": run_hysteresis_from_config,
        "kv_mechanism": run_kv_mechanism_from_config,
        "behavior_strict": run_behavior_strict_from_config,
        "steering": run_steering_from_config,
        "steering_analysis": run_steering_analysis_from_config,
        "steering_layer_matrix": run_layer_matrix_from_config,
        "minimal_recursive_intervention": run_minimal_recursive_intervention_from_config,
        "extended_context_steering": run_extended_context_steering_from_config,
        "steering_control": run_steering_control_from_config,
        "triple_system_intervention": run_triple_system_intervention_from_config,
        "surgical_sweep": run_surgical_sweep_from_config,
        "verification_sweep": run_verification_experiments,
        "p1_ablation": run_p1_ablation_from_config,
        "sprint_head_specific_steering": run_sprint_head_specific_steering,
        "retrocompute_mode_score": run_retrocompute_mode_score_from_config,
        "ioi_causal_test": run_ioi_causal_test_from_config,
        "importance_sweep": run_importance_sweep_from_config,
        "geometry_behavior": run_geometry_behavior_from_config,
        "source_isolation_diagnostic": run_source_isolation_diagnostic_from_config,
        "layer_sweep": run_layer_sweep_from_config,
        "kitchen_sink": run_kitchen_sink_from_config,
        "circuit_discovery": run_circuit_discovery_from_config,
        "mlp_steering_sweep": run_mlp_steering_sweep_from_config,
        "random_direction_control": run_random_direction_control_from_config,
        "mlp_ablation_necessity": run_mlp_ablation_necessity_from_config,
        "position_specific_ablation": run_position_specific_ablation_from_config,
        "mlp_sufficiency_test": run_mlp_sufficiency_test_from_config,
        "combined_mlp_sufficiency_test": run_combined_mlp_sufficiency_test_from_config,
        "logit_lens_analysis": run_logit_lens_analysis_from_config,
        "vproj_patching_analysis": run_vproj_patching_analysis_from_config,
        "mlp_vproj_combined_sufficiency_test": run_mlp_vproj_combined_sufficiency_test_from_config,
        "cross_architecture_validation": run_cross_architecture_validation_from_config,
        "c2_rv_measurement": run_c2_rv_measurement_from_config,
    }


def run_from_config(cfg: Dict[str, Any], run_dir: Path) -> ExperimentResult:
    exp, _params = _validate_top_level(cfg)
    reg = get_registry()
    if exp not in reg:
        raise ConfigError(
            f"Unknown experiment '{exp}'. Known: {sorted(reg.keys())}"
        )
    return reg[exp](cfg, run_dir)


