"""
Hysteresis / "one-way door" test for the residual-stream basin.

We apply TWO sequential residual patches in a single forward pass:
- push at push_layer (typically earlier, e.g. 20 or 24)
- undo at undo_layer (typically later, e.g. 24 or 27)

We then measure R_V at measurement_layer (default 27), using PR of v_proj
at early_layer vs measurement_layer, matching the rest of the repo.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from prompts.loader import PromptLoader
from src.core.models import load_model, set_seed
from src.metrics.rv import participation_ratio
from src.pipelines.registry import ExperimentResult


@dataclass
class PairSpec:
    src_id: str
    tgt_id: str
    src_group: str
    tgt_group: str


def _token_len(tokenizer, text: str) -> int:
    return int(len(tokenizer.encode(text)))


def _rv_from_pr(pr_early: float, pr_late: float) -> float:
    if pr_early == 0 or math.isnan(pr_early) or math.isnan(pr_late):
        return float("nan")
    return float(pr_late / pr_early)


def _make_patch(src: torch.Tensor, *, window: int, kind: str) -> torch.Tensor:
    """
    src: (T,D). Return (W,D) patch.
    kind: "src" | "opposite" | "random" | "none"
    """
    T = int(src.shape[0])
    W = min(int(window), T)
    if W <= 0:
        return src[:0]
    src_w = src[-W:, :]
    if kind == "src":
        return src_w
    if kind == "opposite":
        return -src_w
    if kind == "random":
        noise = torch.randn_like(src_w)
        denom = float(noise.norm().item()) + 1e-12
        noise = noise * (src_w.norm() / denom)
        return noise
    if kind == "none":
        return src[:0]
    raise ValueError(f"Unknown kind: {kind}")


def _capture_resid(model, tokenizer, text: str, *, layer_idx: int, device: str, max_length: int) -> Optional[torch.Tensor]:
    """
    Capture residual stream input to block layer_idx. Returns (T,D) with batch removed.
    """
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    out: Dict[str, Optional[torch.Tensor]] = {"x": None}

    def hook_fn(_module, inputs):
        out["x"] = inputs[0].detach()[0]
        return None

    h = None
    try:
        h = model.model.layers[layer_idx].register_forward_pre_hook(hook_fn)
        with torch.no_grad():
            model(**enc)
    finally:
        if h is not None:
            h.remove()
    return out["x"]


def _run_with_two_resid_patches_and_measure_rv(
    model,
    tokenizer,
    *,
    baseline_text: str,
    push_layer: int,
    push_patch: torch.Tensor,
    push_kind: str,
    undo_layer: int,
    undo_patch: torch.Tensor,
    undo_kind: str,
    early_layer: int,
    measurement_layer: int,
    window: int,
    device: str,
    max_length: int,
) -> Tuple[float, float, float]:
    """
    Apply push patch at push_layer and undo patch at undo_layer (both resid pre-hooks).
    Measure PR at early_layer and measurement_layer via v_proj hooks.
    """
    enc = tokenizer(baseline_text, return_tensors="pt", truncation=True, max_length=max_length).to(device)

    v_early: Optional[torch.Tensor] = None
    v_meas: Optional[torch.Tensor] = None

    def hook_capture_v_early(_module, _inp, out):
        nonlocal v_early
        v_early = out.detach()[0]
        return out

    def hook_capture_v_meas(_module, _inp, out):
        nonlocal v_meas
        v_meas = out.detach()[0]
        return out

    def make_resid_patch_hook(patch: torch.Tensor, kind: str):
        def hook_fn(_module, inputs):
            if kind == "none":
                return None
            hidden = inputs[0]
            hidden2 = hidden.clone()
            B, T, _D = hidden2.shape
            W = min(int(window), T, int(patch.shape[0]))
            if W > 0:
                hidden2[:, -W:, :] = patch[-W:, :].unsqueeze(0).expand(B, -1, -1)
            return (hidden2, *inputs[1:])

        return hook_fn

    handles = []
    try:
        if early_layer == measurement_layer:
            raise ValueError("early_layer and measurement_layer must differ")
        if push_layer == undo_layer:
            raise ValueError("push_layer and undo_layer must differ for hysteresis test")
        if push_layer > measurement_layer or undo_layer > measurement_layer:
            # We keep this strict for interpretability: patching after measurement can no-op.
            raise ValueError("push_layer/undo_layer must be <= measurement_layer")

        handles.append(model.model.layers[early_layer].self_attn.v_proj.register_forward_hook(hook_capture_v_early))
        handles.append(model.model.layers[measurement_layer].self_attn.v_proj.register_forward_hook(hook_capture_v_meas))

        # Apply in forward order (push earlier, undo later)
        handles.append(model.model.layers[push_layer].register_forward_pre_hook(make_resid_patch_hook(push_patch, push_kind)))
        handles.append(model.model.layers[undo_layer].register_forward_pre_hook(make_resid_patch_hook(undo_patch, undo_kind)))

        with torch.no_grad():
            model(**enc)
    finally:
        for h in handles:
            h.remove()

    pr_early = participation_ratio(v_early, window_size=window)
    pr_meas = participation_ratio(v_meas, window_size=window)
    rv = _rv_from_pr(pr_early, pr_meas)
    return pr_early, pr_meas, rv


def run_hysteresis_patching_from_config(cfg: Dict[str, Any], run_dir: Path) -> ExperimentResult:
    """
    Config schema (minimal):
    - model: {name, device}
    - params:
        - early_layer, measurement_layer
        - windows: [16,32]
        - push_layers: [20,24]
        - undo_layers: [24,27]
        - max_pairs, n_repeats, max_length
        - source_groups: prompt-bank groups used to source the push patch (default recursive groups)
        - target_groups: prompt-bank groups used as baseline text to patch into (default baseline groups)
        - undo_kind: "none" | "baseline" | "opposite" | "random"
    """
    model_cfg = cfg.get("model") or {}
    params = cfg.get("params") or {}

    seed = int(cfg.get("seed") or 0)
    device = str(model_cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_name = str(model_cfg.get("name") or "mistralai/Mistral-7B-v0.1")

    set_seed(seed)

    early_layer = int(params.get("early_layer") or 5)
    measurement_layer = int(params.get("measurement_layer") or 27)
    windows: List[int] = [int(x) for x in (params.get("windows") or [16, 32])]
    windows = [w for w in windows if w > 0]

    push_layers: List[int] = [int(x) for x in (params.get("push_layers") or [20, 24])]
    undo_layers: List[int] = [int(x) for x in (params.get("undo_layers") or [24, 27])]
    push_layers = sorted(set(push_layers))
    undo_layers = sorted(set(undo_layers))

    max_pairs = int(params.get("max_pairs") or 30)
    n_repeats = int(params.get("n_repeats") or 1)
    max_length = int(params.get("max_length") or 512)

    source_groups = params.get("source_groups") or ["L5_refined", "L4_full", "L3_deeper"]
    target_groups = params.get("target_groups") or ["long_control", "baseline_creative", "baseline_math"]

    undo_kinds: List[str] = [str(x) for x in (params.get("undo_kinds") or ["none", "baseline", "opposite", "random"])]
    # always include baseline reset and none
    if "none" not in undo_kinds:
        undo_kinds.append("none")
    if "baseline" not in undo_kinds:
        undo_kinds.append("baseline")

    model, tokenizer = load_model(model_name, device=device)
    loader = PromptLoader()
    bank = loader.prompts

    # Build candidate source/target pairs by group.
    candidates: List[PairSpec] = []
    for sg in source_groups:
        src_ids = [k for k, v in bank.items() if v.get("group") == sg]
        for tg in target_groups:
            tgt_ids = [k for k, v in bank.items() if v.get("group") == tg]
            for i in range(min(len(src_ids), len(tgt_ids))):
                candidates.append(PairSpec(src_id=src_ids[i], tgt_id=tgt_ids[i], src_group=sg, tgt_group=tg))

    rng = np.random.default_rng(seed + 20251213)
    rng.shuffle(candidates)

    pairs: List[PairSpec] = []
    for spec in candidates:
        tgt_text = bank[spec.tgt_id]["text"]
        # Ensure target text is long enough for the largest window
        if _token_len(tokenizer, tgt_text) >= max(windows):
            pairs.append(spec)
        if len(pairs) >= max_pairs:
            break

    rows: List[Dict[str, Any]] = []

    for spec in pairs:
        src_text = bank[spec.src_id]["text"]
        tgt_text = bank[spec.tgt_id]["text"]

        # Pre-capture baseline residuals for all undo layers (for baseline reset), and source residuals for all push layers.
        src_resid: Dict[int, Optional[torch.Tensor]] = {}
        base_resid: Dict[int, Optional[torch.Tensor]] = {}

        for L in set(push_layers + undo_layers):
            src_resid[L] = _capture_resid(model, tokenizer, src_text, layer_idx=L, device=device, max_length=max_length)
            base_resid[L] = _capture_resid(model, tokenizer, tgt_text, layer_idx=L, device=device, max_length=max_length)

        for rep in range(n_repeats):
            for w in windows:
                for push_L in push_layers:
                    if push_L >= measurement_layer:
                        continue
                    src_t = src_resid.get(push_L)
                    if src_t is None:
                        continue
                    push_patch = _make_patch(src_t, window=w, kind="src")

                    for undo_L in undo_layers:
                        if undo_L <= push_L or undo_L > measurement_layer:
                            continue

                        for undo_kind in undo_kinds:
                            if undo_kind == "baseline":
                                bt = base_resid.get(undo_L)
                                if bt is None:
                                    continue
                                undo_patch = _make_patch(bt, window=w, kind="src")
                            elif undo_kind in ("opposite", "random", "none"):
                                # define opposite/random relative to the *push* patch, to test rescue vs reinforcement
                                undo_patch = _make_patch(push_patch, window=w, kind=undo_kind)
                            else:
                                raise ValueError(f"Unknown undo_kind: {undo_kind}")

                            pr_e, pr_m, rv = _run_with_two_resid_patches_and_measure_rv(
                                model,
                                tokenizer,
                                baseline_text=tgt_text,
                                push_layer=push_L,
                                push_patch=push_patch,
                                push_kind="src",
                                undo_layer=undo_L,
                                undo_patch=undo_patch,
                                undo_kind=undo_kind,
                                early_layer=early_layer,
                                measurement_layer=measurement_layer,
                                window=w,
                                device=device,
                                max_length=max_length,
                            )

                            rows.append(
                                {
                                    "src_id": spec.src_id,
                                    "tgt_id": spec.tgt_id,
                                    "src_group": spec.src_group,
                                    "tgt_group": spec.tgt_group,
                                    "rep": rep,
                                    "window": int(w),
                                    "push_layer": int(push_L),
                                    "undo_layer": int(undo_L),
                                    "undo_kind": str(undo_kind),
                                    "pr_early": float(pr_e),
                                    "pr_meas": float(pr_m),
                                    "rv": float(rv),
                                }
                            )

    csv_path = run_dir / "hysteresis_patching.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["rv"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Summary: mean/std by window x push_layer x undo_layer x undo_kind
    def _mean_std(xs: List[float]) -> Dict[str, float]:
        arr = np.asarray([x for x in xs if not np.isnan(x)], dtype=np.float64)
        if arr.size == 0:
            return {"mean": float("nan"), "std": float("nan"), "n": 0.0}
        return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1) if arr.size > 1 else 0.0), "n": float(arr.size)}

    by: Dict[str, Any] = {}
    for w0 in sorted(set(int(r["window"]) for r in rows)):
        by[str(w0)] = {}
        sub_w = [r for r in rows if int(r["window"]) == w0]
        for push_L in sorted(set(int(r["push_layer"]) for r in sub_w)):
            by[str(w0)][str(push_L)] = {}
            sub_p = [r for r in sub_w if int(r["push_layer"]) == push_L]
            for undo_L in sorted(set(int(r["undo_layer"]) for r in sub_p)):
                by[str(w0)][str(push_L)][str(undo_L)] = {}
                sub_u = [r for r in sub_p if int(r["undo_layer"]) == undo_L]
                for kind in sorted(set(str(r["undo_kind"]) for r in sub_u)):
                    xs = [float(r["rv"]) for r in sub_u if str(r["undo_kind"]) == kind]
                    by[str(w0)][str(push_L)][str(undo_L)][kind] = _mean_std(xs)

    summary: Dict[str, Any] = {
        "experiment": "hysteresis_patching",
        "device": device,
        "model_name": model_name,
        "n_rows": len(rows),
        "params": {
            "early_layer": early_layer,
            "measurement_layer": measurement_layer,
            "windows": windows,
            "push_layers": push_layers,
            "undo_layers": undo_layers,
            "undo_kinds": undo_kinds,
            "max_pairs": max_pairs,
            "n_repeats": n_repeats,
            "max_length": max_length,
            "source_groups": source_groups,
            "target_groups": target_groups,
        },
        "by_window_by_push_by_undo": by,
        "artifacts": {"csv": str(csv_path)},
    }

    return ExperimentResult(summary=summary)


