"""
Phase 0 Pipeline B: Metric target validation (what does R_V measure?).

We measure PR / R_V over multiple targets:
- v_proj.weight (weight-space geometry proxy)
- v_proj outputs (current canonical)
- hidden outputs (residual-stream proxy)

Then we compute simple correlations and group summaries across prompt families.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

from prompts.loader import PromptLoader
from src.core.models import load_model, set_seed
from src.metrics.rv import participation_ratio
from src.pipelines.registry import ExperimentResult


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _pr_2d(mat: torch.Tensor | None) -> float:
    """
    PR for a 2D matrix via singular values: (sum s^2)^2 / sum s^4
    """
    if mat is None:
        return float("nan")
    try:
        m = mat.float()
        _, s, _ = torch.linalg.svd(m, full_matrices=False)
        s2 = (s ** 2).cpu().numpy()
        denom = float((s2 ** 2).sum())
        if denom < 1e-12:
            return float("nan")
        return float((s2.sum() ** 2) / denom)
    except Exception:
        return float("nan")


def _capture_multi(model, tokenizer, text: str, layer_idxs: Sequence[int], device: str, max_length: int) -> Tuple[Dict[int, torch.Tensor | None], Dict[int, torch.Tensor | None]]:
    """
    Capture:
    - v_proj outputs per layer (seq, d)
    - layer outputs (hidden) per layer (seq, d)
    """
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    v_store: Dict[int, torch.Tensor | None] = {i: None for i in layer_idxs}
    h_store: Dict[int, torch.Tensor | None] = {i: None for i in layer_idxs}
    handles = []

    def make_v_hook(layer_idx: int):
        def hook_fn(_module, _inp, out):
            v_store[layer_idx] = out.detach()[0]
            return out

        return hook_fn

    def make_h_hook(layer_idx: int):
        def hook_fn(_module, _inp, out):
            # Some HF layers may return tuples; handle both.
            if isinstance(out, tuple):
                h_store[layer_idx] = out[0].detach()[0]
            else:
                h_store[layer_idx] = out.detach()[0]
            return out

        return hook_fn

    for idx in layer_idxs:
        handles.append(model.model.layers[idx].self_attn.v_proj.register_forward_hook(make_v_hook(idx)))
        handles.append(model.model.layers[idx].register_forward_hook(make_h_hook(idx)))

    try:
        with torch.no_grad():
            model(**enc)
    finally:
        for h in handles:
            h.remove()

    return v_store, h_store


def _pearson(xs: List[float], ys: List[float]) -> float:
    x = np.asarray([a for a, b in zip(xs, ys) if not (np.isnan(a) or np.isnan(b))], dtype=np.float64)
    y = np.asarray([b for a, b in zip(xs, ys) if not (np.isnan(a) or np.isnan(b))], dtype=np.float64)
    if x.size < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sqrt((x * x).sum() * (y * y).sum()))
    if denom < 1e-12:
        return float("nan")
    return float((x * y).sum() / denom)


def run_phase0_metric_targets_from_config(cfg: Dict[str, Any], run_dir: Path) -> ExperimentResult:
    model_cfg = cfg.get("model") or {}
    params = cfg.get("params") or {}

    seed = int(cfg.get("seed") or 0)
    device = str(model_cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_name = str(model_cfg.get("name") or "mistralai/Mistral-7B-v0.1")

    early_layer = int(params.get("early_layer") or 5)
    late_layer = int(params.get("late_layer") or 27)
    window = int(params.get("window") or 16)
    max_length = int(params.get("max_length") or 512)

    sampling = params.get("sampling") or {}
    groups = sampling.get("groups") or {
        "dose_response": {"pillar": "dose_response", "limit": 10},
        "baselines": {"pillar": "baselines", "limit": 10},
        "confounds": {"pillar": "confounds", "limit": 10},
    }

    set_seed(seed)
    model, tokenizer = load_model(model_name, device=device)
    loader = PromptLoader()
    bank_version = loader.version
    (run_dir / "prompt_bank_version.txt").write_text(bank_version)
    (run_dir / "prompt_bank_version.json").write_text(
        json.dumps({"version": bank_version}, indent=2) + "\n"
    )

    # Weight PRs (constant across prompts for a given model/layer)
    w_early = model.model.layers[early_layer].self_attn.v_proj.weight.detach()
    w_late = model.model.layers[late_layer].self_attn.v_proj.weight.detach()
    pr_w_early = _pr_2d(w_early)
    pr_w_late = _pr_2d(w_late)
    rv_w = float(pr_w_late / pr_w_early) if pr_w_early and not np.isnan(pr_w_early) and not np.isnan(pr_w_late) else float("nan")

    rows: List[Dict[str, Any]] = []
    for group_name, spec in groups.items():
        pillar = spec.get("pillar")
        g = spec.get("group")
        limit = int(spec.get("limit") or 10)
        seed_g = int(spec.get("seed") or seed)

        if pillar:
            prompts = loader.get_by_pillar(pillar=pillar, limit=limit, seed=seed_g)
        elif g:
            prompts = loader.get_by_group(group=g, limit=limit, seed=seed_g)
        else:
            raise ValueError(f"Group spec must include pillar or group: {group_name}")

        for idx, text in enumerate(prompts):
            v_store, h_store = _capture_multi(model, tokenizer, text, [early_layer, late_layer], device=device, max_length=max_length)
            v_e = v_store[early_layer]
            v_l = v_store[late_layer]
            h_e = h_store[early_layer]
            h_l = h_store[late_layer]

            pr_v_e = participation_ratio(v_e, window_size=window)
            pr_v_l = participation_ratio(v_l, window_size=window)
            rv_v = float(pr_v_l / pr_v_e) if pr_v_e and not (np.isnan(pr_v_e) or np.isnan(pr_v_l)) else float("nan")

            pr_h_e = participation_ratio(h_e, window_size=window)
            pr_h_l = participation_ratio(h_l, window_size=window)
            rv_h = float(pr_h_l / pr_h_e) if pr_h_e and not (np.isnan(pr_h_e) or np.isnan(pr_h_l)) else float("nan")

            rows.append(
                {
                    "group": group_name,
                    "idx": idx,
                    "text_sha": _sha(text),
                    "text_len_chars": len(text),
                    "early_layer": early_layer,
                    "late_layer": late_layer,
                    "window": window,
                    "pr_w_early": pr_w_early,
                    "pr_w_late": pr_w_late,
                    "rv_weight": rv_w,
                    "pr_v_early": pr_v_e,
                    "pr_v_late": pr_v_l,
                    "rv_vproj": rv_v,
                    "pr_h_early": pr_h_e,
                    "pr_h_late": pr_h_l,
                    "rv_hidden": rv_h,
                }
            )

    out_csv = run_dir / "phase0_metric_targets.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Correlations across prompts: do the targets agree?
    rv_v_list = [r["rv_vproj"] for r in rows]
    rv_h_list = [r["rv_hidden"] for r in rows]
    corr_rv_v_vs_h = _pearson(rv_v_list, rv_h_list)

    pr_v_late_list = [r["pr_v_late"] for r in rows]
    pr_h_late_list = [r["pr_h_late"] for r in rows]
    corr_pr_vl_vs_hl = _pearson(pr_v_late_list, pr_h_late_list)

    # Group summaries
    by_group: Dict[str, Any] = {}
    for g in sorted({r["group"] for r in rows}):
        g_rows = [r for r in rows if r["group"] == g]
        def _ms(field: str) -> Dict[str, float]:
            arr = np.asarray([rr[field] for rr in g_rows if not np.isnan(rr[field])], dtype=np.float64)
            return {"n": float(arr.size), "mean": float(arr.mean()) if arr.size else float("nan"), "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0}
        by_group[g] = {
            "rv_vproj": _ms("rv_vproj"),
            "rv_hidden": _ms("rv_hidden"),
            "pr_v_late": _ms("pr_v_late"),
            "pr_h_late": _ms("pr_h_late"),
        }

    summary = {
        "experiment": "phase0_metric_targets",
        "model_name": model_name,
        "device": device,
        "seed": seed,
        "prompt_bank_version": bank_version,
        "params": {
            "early_layer": early_layer,
            "late_layer": late_layer,
            "window": window,
            "max_length": max_length,
            "groups": groups,
        },
        "n_rows": len(rows),
        "weight_pr": {
            "pr_w_early": pr_w_early,
            "pr_w_late": pr_w_late,
            "rv_weight": rv_w,
        },
        "correlations": {
            "pearson_rv_vproj_vs_rv_hidden": corr_rv_v_vs_h,
            "pearson_pr_v_late_vs_pr_h_late": corr_pr_vl_vs_hl,
        },
        "by_group": by_group,
        "artifacts": {"csv": str(out_csv)},
    }

    return ExperimentResult(summary=summary)


