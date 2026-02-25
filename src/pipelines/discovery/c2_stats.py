"""Lightweight stats helpers for C2 analysis (no model/runtime dependencies)."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy import stats


def compute_paired_rv_stats(df: pd.DataFrame, config_a: str, config_b: str) -> Dict[str, Any]:
    """
    Compute paired RV stats by prompt index for two configs.

    Drops rows where either side has NaN RV, so pair count reflects valid matched data.
    """
    left = (
        df[df["config"] == config_a][["prompt_idx", "rv_mean"]]
        .rename(columns={"rv_mean": "rv_a"})
    )
    right = (
        df[df["config"] == config_b][["prompt_idx", "rv_mean"]]
        .rename(columns={"rv_mean": "rv_b"})
    )
    merged = left.merge(right, on="prompt_idx", how="inner")

    if merged.empty:
        return {"n_pairs_total": 0, "n_pairs_valid": 0}

    valid = merged["rv_a"].notna() & merged["rv_b"].notna()
    paired = merged.loc[valid]
    n_total = int(len(merged))
    n_valid = int(len(paired))

    if n_valid < 2:
        return {
            "n_pairs_total": n_total,
            "n_pairs_valid": n_valid,
            "rv_a_mean": float(paired["rv_a"].mean()) if n_valid > 0 else float("nan"),
            "rv_b_mean": float(paired["rv_b"].mean()) if n_valid > 0 else float("nan"),
            "rv_delta_mean": float((paired["rv_a"] - paired["rv_b"]).mean()) if n_valid > 0 else float("nan"),
            "t_statistic": float("nan"),
            "p_value": float("nan"),
            "cohens_dz": float("nan"),
            "cohens_d_av": float("nan"),
        }

    rv_a = paired["rv_a"].to_numpy()
    rv_b = paired["rv_b"].to_numpy()
    deltas = rv_a - rv_b

    t_stat, p_value = stats.ttest_rel(rv_a, rv_b)

    delta_std = float(np.std(deltas, ddof=1))
    d_z = float(np.mean(deltas) / delta_std) if delta_std > 0 else 0.0

    avg_std = float(np.sqrt((np.var(rv_a, ddof=1) + np.var(rv_b, ddof=1)) / 2))
    d_av = float(np.mean(deltas) / avg_std) if avg_std > 0 else 0.0

    return {
        "n_pairs_total": n_total,
        "n_pairs_valid": n_valid,
        "rv_a_mean": float(np.mean(rv_a)),
        "rv_b_mean": float(np.mean(rv_b)),
        "rv_delta_mean": float(np.mean(deltas)),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "cohens_dz": d_z,
        "cohens_d_av": d_av,
    }

