from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipelines.canonical import causal_state_benchmark_v3_confirmatory as module
from src.pipelines.registry import ExperimentResult


def test_v3_confirmatory_wrapper_relabels_experiment(tmp_path: Path, monkeypatch) -> None:
    def fake_runner(_cfg, _run_dir):
        return ExperimentResult(
            summary={"experiment": "causal_state_benchmark_v2", "rv_recursive_mean": 0.61},
            baseline_metrics={"rv": 0.61, "logit_diff": None},
        )

    monkeypatch.setattr(module, "run_causal_state_benchmark_v2_from_config", fake_runner)
    result = module.run_causal_state_benchmark_v3_confirmatory_from_config({}, tmp_path)

    assert result.summary["experiment"] == "causal_state_benchmark_v3_confirmatory"
    assert result.summary["confirmatory_parent_experiment"] == "causal_state_benchmark_v2"
    assert result.baseline_metrics == {"rv": 0.61, "logit_diff": None}
