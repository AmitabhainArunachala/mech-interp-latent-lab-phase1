"""
Confound validation: score champions + controls with R_V (prompt-pass) and compute stats.

Outputs:
- confound_results.csv (per-prompt rows)
- summary.json (written by canonical runner)
- prompt_bank_version.txt (for reproducibility)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from prompts.loader import PromptLoader
from src.core.models import load_model
from src.metrics.rv import compute_rv
from src.pipelines.registry import ExperimentResult


def _tok_len(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d with pooled standard deviation."""
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    n1, n2 = len(a), len(b)
    var1, var2 = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std < 1e-10:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_std)


def _ci_95(arr: np.ndarray) -> Dict[str, float]:
    """95% confidence interval for mean."""
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        return {"ci_lower": float("nan"), "ci_upper": float("nan")}
    try:
        from scipy import stats
        sem = stats.sem(arr)
        ci = stats.t.interval(0.95, len(arr) - 1, loc=np.mean(arr), scale=sem)
        return {"ci_lower": float(ci[0]), "ci_upper": float(ci[1])}
    except Exception:
        return {"ci_lower": float("nan"), "ci_upper": float("nan")}


def _ttest(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    # Welch t-test implemented with scipy if available; otherwise numpy fallback (approx).
    # Now includes Cohen's d for effect size
    try:
        from scipy.stats import ttest_ind

        res = ttest_ind(a, b, equal_var=False, nan_policy="omit")
        return {"t": float(res.statistic), "p": float(res.pvalue), "cohens_d": _cohens_d(a, b)}
    except Exception:
        a = a[~np.isnan(a)]
        b = b[~np.isnan(b)]
        if len(a) < 2 or len(b) < 2:
            return {"t": float("nan"), "p": float("nan"), "cohens_d": float("nan")}
        # Very rough fallback: compute t statistic, omit p-value (nan) without scipy.
        va = np.var(a, ddof=1)
        vb = np.var(b, ddof=1)
        t = (np.mean(a) - np.mean(b)) / np.sqrt(va / len(a) + vb / len(b))
        return {"t": float(t), "p": float("nan"), "cohens_d": _cohens_d(a, b)}


def _pearson(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    try:
        from scipy.stats import pearsonr

        r, p = pearsonr(x, y)
        return {"r": float(r), "p": float(p)}
    except Exception:
        return {"r": float("nan"), "p": float("nan")}


def run_confound_validation_from_config(cfg: Dict[str, Any], run_dir: Path) -> ExperimentResult:
    model_cfg = cfg.get("model") or {}
    p = cfg.get("params") or {}

    # Model config (support both old and new config formats)
    model_name = str(model_cfg.get("name") or p.get("model_name") or p.get("model") or "mistralai/Mistral-7B-v0.1")
    device = str(model_cfg.get("device") or p.get("device") or "cuda")
    torch_dtype = str(p.get("torch_dtype") or "float16")
    attn_implementation = str(p.get("attn_implementation") or "sdpa")

    early = int(p.get("early_layer") or 5)
    late = int(p.get("late_layer") or p.get("layer") or 27)
    window = int(p.get("window") or 16)

    # Sample sizes from config
    n_champions = int(p.get("n_champions") or 30)
    n_length_matched = int(p.get("n_length_matched") or 30)
    n_pseudo_recursive = int(p.get("n_pseudo_recursive") or 30)
    seed = int(cfg.get("seed") or p.get("seed") or 42)

    # Load prompts via PromptLoader (canonical source)
    loader = PromptLoader()
    bank_version = loader.version
    bank = loader.prompts  # Full dict with metadata
    
    # Log prompt bank version for reproducibility
    (run_dir / "prompt_bank_version.txt").write_text(bank_version)
    (run_dir / "prompt_bank_version.json").write_text(
        json.dumps({"version": bank_version}, indent=2) + "\n"
    )

    # Get prompts from canonical bank (get_by_group returns List[str], so we filter manually for metadata)
    def get_prompts_with_metadata(group: str) -> List[Dict]:
        return [{"id": k, **v} for k, v in bank.items() if v.get("group") == group]
    
    champions_list = get_prompts_with_metadata("champions")
    length_matched_list = get_prompts_with_metadata("control_length_matched")
    pseudo_recursive_list = get_prompts_with_metadata("control_pseudo_recursive")

    # Sample according to config params
    rng = np.random.default_rng(seed)
    
    if len(champions_list) > n_champions:
        indices = rng.choice(len(champions_list), n_champions, replace=False)
        champions_list = [champions_list[i] for i in indices]
    if len(length_matched_list) > n_length_matched:
        indices = rng.choice(len(length_matched_list), n_length_matched, replace=False)
        length_matched_list = [length_matched_list[i] for i in indices]
    if len(pseudo_recursive_list) > n_pseudo_recursive:
        indices = rng.choice(len(pseudo_recursive_list), n_pseudo_recursive, replace=False)
        pseudo_recursive_list = [pseudo_recursive_list[i] for i in indices]

    # Load model + tokenizer
    import torch

    dtype = torch.float16 if torch_dtype == "float16" else torch.bfloat16
    model, tokenizer = load_model(model_name, device=device, torch_dtype=dtype, attn_implementation=attn_implementation)

    rows: List[Dict[str, object]] = []

    # Champions
    for ch in champions_list:
        text = str(ch["text"])
        tok_len = int(_tok_len(tokenizer, text))
        # Skip if too short for window
        if tok_len < window:
            continue
        rv = compute_rv(model, tokenizer, text, early=early, late=late, window=window, device=device)
        if np.isnan(rv):
            continue
        rows.append(
            {
                "prompt_id": str(ch.get("id") or ch.get("prompt_id", "")),
                "prompt_type": "champion",
                "family": str(ch.get("family") or ch.get("group") or ""),
                "text": text,
                "token_count": tok_len,
                "rv_l27": float(rv),
            }
        )

    # Length-matched controls
    for c in length_matched_list:
        text = str(c["text"])
        tok_len = int(_tok_len(tokenizer, text))
        if tok_len < window:
            continue
        rv = compute_rv(model, tokenizer, text, early=early, late=late, window=window, device=device)
        if np.isnan(rv):
            continue
        rows.append(
            {
                "prompt_id": str(c.get("id") or c.get("control_id", "")),
                "prompt_type": "length_matched",
                "family": "",
                "text": text,
                "token_count": tok_len,
                "rv_l27": float(rv),
                "matched_to": str(c.get("matched_to") or ""),
            }
        )

    # Pseudo-recursive controls
    for c in pseudo_recursive_list:
        text = str(c["text"])
        tok_len = int(_tok_len(tokenizer, text))
        if tok_len < window:
            continue
        rv = compute_rv(model, tokenizer, text, early=early, late=late, window=window, device=device)
        if np.isnan(rv):
            continue
        rows.append(
            {
                "prompt_id": str(c.get("id") or c.get("control_id", "")),
                "prompt_type": "pseudo_recursive",
                "family": "",
                "text": text,
                "token_count": tok_len,
                "rv_l27": float(rv),
                "matched_to": str(c.get("matched_to") or ""),
            }
        )

    df = pd.DataFrame(rows)
    out_csv = run_dir / "confound_results.csv"
    df.to_csv(out_csv, index=False)

    # Group stats (filter NaN before stats)
    def arr(mask):
        vals = df.loc[mask, "rv_l27"].astype(float).to_numpy()
        return vals[~np.isnan(vals)]

    rv_ch = arr(df["prompt_type"] == "champion")
    rv_len = arr(df["prompt_type"] == "length_matched")
    rv_pseudo = arr(df["prompt_type"] == "pseudo_recursive")

    # Token count correlation (all rows, filter NaN)
    valid_mask = ~np.isnan(df["rv_l27"].astype(float).to_numpy())
    x = df.loc[valid_mask, "token_count"].astype(float).to_numpy()
    y = df.loc[valid_mask, "rv_l27"].astype(float).to_numpy()
    corr_all = _pearson(x, y)

    summary = {
        "experiment": "confound_validation",
        "prompt_bank_version": bank_version,
        "n_total": int(len(df)),
        "n_champions": int((df["prompt_type"] == "champion").sum()),
        "n_length_matched": int((df["prompt_type"] == "length_matched").sum()),
        "n_pseudo_recursive": int((df["prompt_type"] == "pseudo_recursive").sum()),
        "mean_rv": {
            "champions": float(np.nanmean(rv_ch)) if len(rv_ch) > 0 else float("nan"),
            "length_matched": float(np.nanmean(rv_len)) if len(rv_len) > 0 else float("nan"),
            "pseudo_recursive": float(np.nanmean(rv_pseudo)) if len(rv_pseudo) > 0 else float("nan"),
        },
        "std_rv": {
            "champions": float(np.nanstd(rv_ch)) if len(rv_ch) > 0 else float("nan"),
            "length_matched": float(np.nanstd(rv_len)) if len(rv_len) > 0 else float("nan"),
            "pseudo_recursive": float(np.nanstd(rv_pseudo)) if len(rv_pseudo) > 0 else float("nan"),
        },
        "ci_95_rv": {
            "champions": _ci_95(rv_ch),
            "length_matched": _ci_95(rv_len),
            "pseudo_recursive": _ci_95(rv_pseudo),
        },
        "ttest": {
            "champions_vs_length_matched": _ttest(rv_ch, rv_len),
            "champions_vs_pseudo_recursive": _ttest(rv_ch, rv_pseudo),
            "length_matched_vs_pseudo_recursive": _ttest(rv_len, rv_pseudo),
        },
        "corr_token_count_vs_rv": corr_all,
        "artifacts": {
            "confound_results_csv": str(out_csv),
            "prompt_bank_version_txt": str(run_dir / "prompt_bank_version.txt"),
        },
        "params": {
            "model_name": model_name,
            "early_layer": early,
            "late_layer": late,
            "window": window,
            "n_champions": n_champions,
            "n_length_matched": n_length_matched,
            "n_pseudo_recursive": n_pseudo_recursive,
            "seed": seed,
        },
    }

    return ExperimentResult(summary=summary)


__all__ = ["run_confound_validation_from_config"]


