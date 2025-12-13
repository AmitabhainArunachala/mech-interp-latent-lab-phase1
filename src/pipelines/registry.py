"""
Experiment registry + config validation.

This is the "blessed" mapping from config["experiment"] -> runnable function.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Tuple


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ExperimentResult:
    """
    Minimal contract: each experiment returns a dict summary plus optional artifact paths.
    """

    summary: Dict[str, Any]


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
    from .phase1_existence import run_phase1_existence_from_config
    from .rv_l27_causal_validation import run_rv_l27_causal_validation_from_config
    from .phase0_minimal_pairs import run_phase0_minimal_pairs_from_config
    from .phase0_metric_targets import run_phase0_metric_targets_from_config
    from .l27_head_analysis import run_l27_head_analysis_from_config
    from .path_patching_mechanism import run_path_patching_mechanism_from_config
    from .behavioral_grounding import run_behavioral_grounding_from_config

    return {
        "phase0_minimal_pairs": run_phase0_minimal_pairs_from_config,
        "phase0_metric_targets": run_phase0_metric_targets_from_config,
        "phase1_existence": run_phase1_existence_from_config,
        "rv_l27_causal_validation": run_rv_l27_causal_validation_from_config,
        "l27_head_analysis": run_l27_head_analysis_from_config,
        "path_patching_mechanism": run_path_patching_mechanism_from_config,
        "behavioral_grounding": run_behavioral_grounding_from_config,
    }


def run_from_config(cfg: Dict[str, Any], run_dir: Path) -> ExperimentResult:
    exp, _params = _validate_top_level(cfg)
    reg = get_registry()
    if exp not in reg:
        raise ConfigError(
            f"Unknown experiment '{exp}'. Known: {sorted(reg.keys())}"
        )
    return reg[exp](cfg, run_dir)


