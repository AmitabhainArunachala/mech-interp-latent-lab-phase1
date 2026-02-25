import math
import sys
from pathlib import Path

import pandas as pd

# Ensure repo root is importable when tests run from arbitrary CWD.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipelines.discovery.c2_stats import compute_paired_rv_stats


def test_compute_paired_rv_stats_drops_nan_pairs():
    df = pd.DataFrame(
        [
            {"config": "baseline", "prompt_idx": 0, "rv_mean": 0.70},
            {"config": "baseline", "prompt_idx": 1, "rv_mean": float("nan")},
            {"config": "baseline", "prompt_idx": 2, "rv_mean": 0.66},
            {"config": "kv_only", "prompt_idx": 0, "rv_mean": 0.58},
            {"config": "kv_only", "prompt_idx": 1, "rv_mean": 0.57},
            {"config": "kv_only", "prompt_idx": 2, "rv_mean": 0.60},
        ]
    )

    stats = compute_paired_rv_stats(df, "baseline", "kv_only")

    assert stats["n_pairs_total"] == 3
    assert stats["n_pairs_valid"] == 2
    assert math.isclose(stats["rv_delta_mean"], 0.09, rel_tol=1e-9)
    assert stats["p_value"] == stats["p_value"]  # not NaN


def test_compute_paired_rv_stats_handles_no_overlap():
    df = pd.DataFrame(
        [
            {"config": "baseline", "prompt_idx": 0, "rv_mean": 0.70},
            {"config": "kv_only", "prompt_idx": 1, "rv_mean": 0.58},
        ]
    )

    stats = compute_paired_rv_stats(df, "baseline", "kv_only")

    assert stats == {"n_pairs_total": 0, "n_pairs_valid": 0}
