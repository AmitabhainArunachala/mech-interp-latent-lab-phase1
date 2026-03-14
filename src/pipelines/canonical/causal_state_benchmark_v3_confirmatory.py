"""
Confirmatory v3 alias for the causal state benchmark.

This reuses the v2 benchmark runner but records a distinct experiment name so
the promoted confirmatory configuration is first-class in the registry and
validation surface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from src.pipelines.registry import ExperimentResult

from .causal_state_benchmark_v2 import run_causal_state_benchmark_v2_from_config


def run_causal_state_benchmark_v3_confirmatory_from_config(
    cfg: Dict[str, Any], run_dir: Path
) -> ExperimentResult:
    base_result = run_causal_state_benchmark_v2_from_config(cfg, run_dir)
    summary = dict(base_result.summary)
    summary["experiment"] = "causal_state_benchmark_v3_confirmatory"
    summary["confirmatory_parent_experiment"] = "causal_state_benchmark_v2"
    return ExperimentResult(summary=summary, baseline_metrics=base_result.baseline_metrics)
