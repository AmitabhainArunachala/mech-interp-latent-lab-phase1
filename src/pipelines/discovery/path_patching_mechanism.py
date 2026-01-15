"""
Path patching mechanism probe: which pathway can *cause* R_V contraction?

We intervene at an "intervention layer" by swapping one component's activations
from a recursive prompt into a baseline prompt, then we measure R_V in V-space
at a (possibly different) measurement layer.

Components supported:
- "v": self_attn.v_proj output
- "o": self_attn.o_proj output
- "resid": residual stream input to the transformer block (layer pre-hook)

Controls (required):
- random: norm-matched noise patch
- shuffled: token-shuffle within the patch window
- opposite: sign flip of the recursive patch
- wrong-layer: apply the same procedure, but at a different layer index
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch

from prompts.loader import PromptLoader
from src.core.models import load_model, set_seed
from src.metrics.rv import participation_ratio
from src.pipelines.registry import ExperimentResult

Component = Literal["v", "o", "resid"]
PatchType = Literal["none", "recursive", "random", "shuffled", "opposite"]


@dataclass
class PairSpec:
    rec_id: str
    base_id: str
    rec_group: str
    base_group: str


def _token_len(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text))


def _rv_from_pr(pr_early: float, pr_late: float) -> float:
    if pr_early == 0 or math.isnan(pr_early) or math.isnan(pr_late):
        return float("nan")
    return float(pr_late / pr_early)


def _mean_std(xs: List[float]) -> Dict[str, float]:
    arr = np.asarray([x for x in xs if not np.isnan(x)], dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0.0}
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1) if arr.size > 1 else 0.0), "n": float(arr.size)}


def _capture_component(
    model,
    tokenizer,
    text: str,
    *,
    layer_idx: int,
    component: Component,
    device: str,
    max_length: int,
) -> Optional[torch.Tensor]:
    """
    Return a (seq, d_model) tensor (batch removed) for the requested component.
    """
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    out_storage: Dict[str, Optional[torch.Tensor]] = {"x": None}
    handles = []

    def hook_capture_out(_module, _inp, out):
        out_storage["x"] = out.detach()[0]
        return out

    def hook_capture_resid(_module, inputs):
        # inputs[0]: (batch, seq, d_model)
        out_storage["x"] = inputs[0].detach()[0]
        return None

    try:
        layer = model.model.layers[layer_idx]
        if component == "v":
            handles.append(layer.self_attn.v_proj.register_forward_hook(hook_capture_out))
        elif component == "o":
            handles.append(layer.self_attn.o_proj.register_forward_hook(hook_capture_out))
        elif component == "resid":
            handles.append(layer.register_forward_pre_hook(hook_capture_resid))
        else:
            raise ValueError(f"Unknown component: {component}")

        with torch.no_grad():
            model(**enc)
    finally:
        for h in handles:
            h.remove()

    return out_storage["x"]


def _capture_resid_multi(
    model,
    tokenizer,
    text: str,
    *,
    layer_idxs: List[int],
    device: str,
    max_length: int,
) -> Dict[int, Optional[torch.Tensor]]:
    """
    Capture residual stream inputs for multiple layers in a single forward pass.
    Returns {layer_idx: (seq, d_model) or None}.
    """
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    storage: Dict[int, Optional[torch.Tensor]] = {i: None for i in layer_idxs}
    handles = []

    def make_hook(layer_idx: int):
        def hook_fn(_module, inputs):
            storage[layer_idx] = inputs[0].detach()[0]
            return None

        return hook_fn

    try:
        for idx in layer_idxs:
            handles.append(model.model.layers[idx].register_forward_pre_hook(make_hook(idx)))
        with torch.no_grad():
            model(**enc)
    finally:
        for h in handles:
            h.remove()

    return storage


def _make_patch(
    src: torch.Tensor,
    *,
    patch_type: PatchType,
    window: int,
) -> torch.Tensor:
    """
    Produce a (W, d) patch tensor from the source (seq, d), on the same device/dtype.
    """
    T = int(src.shape[0])
    W = min(window, T)
    if W <= 0:
        return src[:0]
    src_w = src[-W:, :]

    if patch_type == "recursive":
        return src_w
    if patch_type == "opposite":
        return -src_w
    if patch_type == "shuffled":
        perm = torch.randperm(W, device=src.device)
        return src_w[perm, :]
    if patch_type == "random":
        noise = torch.randn_like(src_w)
        denom = float(noise.norm().item()) + 1e-12
        noise = noise * (src_w.norm() / denom)
        return noise
    if patch_type == "none":
        return src[:0]
    raise ValueError(f"Unknown patch_type: {patch_type}")


def _run_with_component_patch_and_measure_rv(
    model,
    tokenizer,
    *,
    baseline_text: str,
    component: Component,
    patch_layer: int,
    patch: torch.Tensor,
    patch_type: PatchType,
    early_layer: int,
    measurement_layer: int,
    window: int,
    device: str,
    max_length: int,
) -> Tuple[float, float, float]:
    """
    Apply a patch at `patch_layer` to `component` while running baseline_text.
    Measure PR at early_layer and measurement_layer using v_proj outputs,
    returning (pr_early, pr_meas, rv).
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

    def hook_patch_out(_module, _inp, out):
        # out: (B, T, D)
        if patch_type == "none":
            return out
        out2 = out.clone()
        B, T, _D = out2.shape
        W = min(window, T, int(patch.shape[0]))
        if W > 0:
            out2[:, -W:, :] = patch[-W:, :].unsqueeze(0).expand(B, -1, -1)
        return out2

    def hook_patch_resid(_module, inputs):
        # inputs[0]: (B, T, D)
        if patch_type == "none":
            return None
        hidden = inputs[0]
        hidden2 = hidden.clone()
        B, T, _D = hidden2.shape
        W = min(window, T, int(patch.shape[0]))
        if W > 0:
            hidden2[:, -W:, :] = patch[-W:, :].unsqueeze(0).expand(B, -1, -1)
        return (hidden2, *inputs[1:])

    handles = []
    try:
        if early_layer == measurement_layer:
            raise ValueError("early_layer and measurement_layer must differ")

        handles.append(model.model.layers[early_layer].self_attn.v_proj.register_forward_hook(hook_capture_v_early))
        handles.append(model.model.layers[measurement_layer].self_attn.v_proj.register_forward_hook(hook_capture_v_meas))

        layer = model.model.layers[patch_layer]
        if component == "v":
            handles.append(layer.self_attn.v_proj.register_forward_hook(hook_patch_out))
        elif component == "o":
            handles.append(layer.self_attn.o_proj.register_forward_hook(hook_patch_out))
        elif component == "resid":
            handles.append(layer.register_forward_pre_hook(hook_patch_resid))
        else:
            raise ValueError(f"Unknown component: {component}")

        with torch.no_grad():
            model(**enc)
    finally:
        for h in handles:
            h.remove()

    pr_early = participation_ratio(v_early, window_size=window)
    pr_meas = participation_ratio(v_meas, window_size=window)
    rv = _rv_from_pr(pr_early, pr_meas)
    return pr_early, pr_meas, rv


def run_path_patching_mechanism_from_config(cfg: Dict[str, Any], run_dir: Path) -> ExperimentResult:
    model_cfg = cfg.get("model") or {}
    params = cfg.get("params") or {}

    seed = int(cfg.get("seed") or 0)
    device = str(model_cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_name = str(model_cfg.get("name") or "mistralai/Mistral-7B-v0.1")

    set_seed(seed)

    early_layer = int(params.get("early_layer") or 5)
    measurement_layer = int(params.get("measurement_layer") or 27)
    target_layer = int(params.get("target_layer") or 27)
    wrong_layer = int(params.get("wrong_layer") or 21)

    window = int(params.get("window") or 16)
    windows: List[int] = list(params.get("windows") or [window])
    windows = [int(w) for w in windows]
    windows = [w for w in windows if w > 0]
    if not windows:
        windows = [window]

    max_pairs = int(params.get("max_pairs") or 24)
    max_length = int(params.get("max_length") or 512)
    n_repeats = int(params.get("n_repeats") or 1)

    components: List[Component] = list(params.get("components") or ["v", "o", "resid"])
    patch_layers: List[int] = list(params.get("patch_layers") or [target_layer])
    patch_layers = [int(x) for x in patch_layers]
    patch_layers = sorted(set(patch_layers))

    pairing = params.get("pairing") or {}
    recursive_groups = pairing.get("recursive_groups") or ["L5_refined", "L4_full", "L3_deeper"]
    baseline_groups = pairing.get("baseline_groups") or ["long_control", "baseline_creative", "baseline_math"]

    patch_types: List[PatchType] = list(params.get("patch_types") or ["none", "recursive", "random", "shuffled", "opposite"])
    patch_types = [str(x) for x in patch_types]  # type: ignore[assignment]
    include_wrong_layer_control = bool(params.get("include_wrong_layer_control", True))

    model, tokenizer = load_model(model_name, device=device)
    loader = PromptLoader()
    bank_version = loader.version
    (run_dir / "prompt_bank_version.txt").write_text(bank_version)
    (run_dir / "prompt_bank_version.json").write_text(
        json.dumps({"version": bank_version}, indent=2) + "\n"
    )
    bank = loader.prompts

    # Build candidate pairs by group.
    candidates: List[PairSpec] = []
    for rec_group in recursive_groups:
        rec_ids = [k for k, v in bank.items() if v.get("group") == rec_group]
        for base_group in baseline_groups:
            base_ids = [k for k, v in bank.items() if v.get("group") == base_group]
            for i in range(min(len(rec_ids), len(base_ids))):
                candidates.append(
                    PairSpec(
                        rec_id=rec_ids[i],
                        base_id=base_ids[i],
                        rec_group=rec_group,
                        base_group=base_group,
                    )
                )

    rng = np.random.default_rng(seed + 20251213)
    rng.shuffle(candidates)

    pairs: List[PairSpec] = []
    for spec in candidates:
        base_text = bank[spec.base_id]["text"]
        if _token_len(tokenizer, base_text) >= window:
            pairs.append(spec)
        if len(pairs) >= max_pairs:
            break

    rows: List[Dict[str, Any]] = []
    # Cache for pt="none" so we don't re-run identical baselines for every patch_layer.
    # Key: (rec_id, base_id, rep, component, window) -> (pr_early, pr_meas, rv)
    none_cache: Dict[Tuple[str, str, int, str, int], Tuple[float, float, float]] = {}

    for spec in pairs:
        rec_text = bank[spec.rec_id]["text"]
        base_text = bank[spec.base_id]["text"]

        # Capture sources for all patch_layers (and optional wrong_layer) for each component.
        sources_by_layer: Dict[int, Dict[str, Optional[torch.Tensor]]] = {}
        capture_layers = sorted(set(patch_layers + ([wrong_layer] if include_wrong_layer_control else [])))
        for layer_idx in capture_layers:
            sources_by_layer[layer_idx] = {comp: None for comp in components}

        # Optimization: resid can be captured for many layers in one pass.
        if "resid" in components and capture_layers:
            resid_map = _capture_resid_multi(
                model,
                tokenizer,
                rec_text,
                layer_idxs=capture_layers,
                device=device,
                max_length=max_length,
            )
            for layer_idx, t in resid_map.items():
                sources_by_layer[layer_idx]["resid"] = t

        # For v/o, fall back to per-layer capture (still OK for small layer sets).
        for layer_idx in capture_layers:
            for comp in components:
                if comp == "resid":
                    continue
                sources_by_layer[layer_idx][comp] = _capture_component(
                    model,
                    tokenizer,
                    rec_text,
                    layer_idx=layer_idx,
                    component=comp,
                    device=device,
                    max_length=max_length,
                )

        for rep in range(n_repeats):
            for comp in components:
                for w in windows:
                    # Sweep patch layer(s)
                    for pl in patch_layers:
                        src = sources_by_layer.get(pl, {}).get(comp)
                        if src is None:
                            continue
                        for pt in patch_types:
                            if pt == "none":
                                k = (spec.rec_id, spec.base_id, rep, str(comp), int(w))
                                if k in none_cache:
                                    pr_e, pr_m, rv = none_cache[k]
                                    rows.append(
                                        {
                                            "rec_id": spec.rec_id,
                                            "base_id": spec.base_id,
                                            "rec_group": spec.rec_group,
                                            "base_group": spec.base_group,
                                            "rep": rep,
                                            "component": comp,
                                            "patch_type": pt,
                                            "patch_layer": pl,
                                            "measurement_layer": measurement_layer,
                                            "early_layer": early_layer,
                                            "window": w,
                                            "pr_early": pr_e,
                                            "pr_meas": pr_m,
                                            "rv": rv,
                                        }
                                    )
                                    continue

                            patch = _make_patch(src, patch_type=pt, window=w)
                            pr_e, pr_m, rv = _run_with_component_patch_and_measure_rv(
                                model,
                                tokenizer,
                                baseline_text=base_text,
                                component=comp,
                                patch_layer=pl,
                                patch=patch,
                                patch_type=pt,
                                early_layer=early_layer,
                                measurement_layer=measurement_layer,
                                window=w,
                                device=device,
                                max_length=max_length,
                            )
                            rows.append(
                                {
                                    "rec_id": spec.rec_id,
                                    "base_id": spec.base_id,
                                    "rec_group": spec.rec_group,
                                    "base_group": spec.base_group,
                                    "rep": rep,
                                    "component": comp,
                                    "patch_type": pt,
                                    "patch_layer": pl,
                                    "measurement_layer": measurement_layer,
                                    "early_layer": early_layer,
                                    "window": w,
                                    "pr_early": pr_e,
                                    "pr_meas": pr_m,
                                    "rv": rv,
                                }
                            )
                            if pt == "none":
                                none_cache[(spec.rec_id, spec.base_id, rep, str(comp), int(w))] = (pr_e, pr_m, rv)

                    if include_wrong_layer_control:
                        # Wrong-layer control always runs (recursive patch at wrong_layer).
                        src_wrong = sources_by_layer.get(wrong_layer, {}).get(comp)
                        if src_wrong is not None:
                            patch_wrong = _make_patch(src_wrong, patch_type="recursive", window=w)
                            pr_e, pr_m, rv = _run_with_component_patch_and_measure_rv(
                                model,
                                tokenizer,
                                baseline_text=base_text,
                                component=comp,
                                patch_layer=wrong_layer,
                                patch=patch_wrong,
                                patch_type="recursive",
                                early_layer=early_layer,
                                measurement_layer=measurement_layer,
                                window=w,
                                device=device,
                                max_length=max_length,
                            )
                            rows.append(
                                {
                                    "rec_id": spec.rec_id,
                                    "base_id": spec.base_id,
                                    "rec_group": spec.rec_group,
                                    "base_group": spec.base_group,
                                    "rep": rep,
                                    "component": comp,
                                    "patch_type": "wrong_layer",
                                    "patch_layer": wrong_layer,
                                    "measurement_layer": measurement_layer,
                                    "early_layer": early_layer,
                                    "window": w,
                                    "pr_early": pr_e,
                                    "pr_meas": pr_m,
                                    "rv": rv,
                                }
                            )

    csv_path = run_dir / "path_patching_mechanism.csv"
    # Defensive: ensure parent exists (should already, but avoids rare cwd/path edge cases).
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["rv"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Summary aggregates: by component x patch_type (global) and per-window.
    by_comp: Dict[str, Dict[str, Dict[str, float]]] = {}
    by_comp_by_window: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    by_comp_by_window_by_layer: Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, float]]]]] = {}
    for comp in components:
        by_comp[comp] = {}
        by_comp_by_window[comp] = {}
        by_comp_by_window_by_layer[comp] = {}
        comp_rows = [r for r in rows if r.get("component") == comp]
        for pt in sorted({r.get("patch_type") for r in comp_rows}):
            xs = [float(r["rv"]) for r in comp_rows if r.get("patch_type") == pt]
            by_comp[comp][str(pt)] = _mean_std(xs)
        for w in sorted(set(int(r["window"]) for r in comp_rows)):
            by_comp_by_window[comp][str(w)] = {}
            by_comp_by_window_by_layer[comp][str(w)] = {}
            sub = [r for r in comp_rows if int(r["window"]) == w]
            for pt in sorted({r.get("patch_type") for r in sub}):
                xs = [float(r["rv"]) for r in sub if r.get("patch_type") == pt]
                by_comp_by_window[comp][str(w)][str(pt)] = _mean_std(xs)

            # Layer-resolved aggregates (critical for sweeps)
            for pl in sorted(set(int(r["patch_layer"]) for r in sub)):
                by_comp_by_window_by_layer[comp][str(w)][str(pl)] = {}
                sub_pl = [r for r in sub if int(r["patch_layer"]) == pl]
                for pt in sorted({r.get("patch_type") for r in sub_pl}):
                    xs = [float(r["rv"]) for r in sub_pl if r.get("patch_type") == pt]
                    by_comp_by_window_by_layer[comp][str(w)][str(pl)][str(pt)] = _mean_std(xs)

    summary: Dict[str, Any] = {
        "experiment": "path_patching_mechanism",
        "device": device,
        "model_name": model_name,
        "prompt_bank_version": bank_version,
        "n_rows": len(rows),
        "params": {
            "early_layer": early_layer,
            "measurement_layer": measurement_layer,
            "target_layer": target_layer,
            "wrong_layer": wrong_layer,
            "window": window,
            "windows": windows,
            "max_pairs": max_pairs,
            "max_length": max_length,
            "n_repeats": n_repeats,
            "components": components,
            "patch_layers": patch_layers,
            "patch_types": patch_types,
            "include_wrong_layer_control": include_wrong_layer_control,
            "pairing": {"recursive_groups": recursive_groups, "baseline_groups": baseline_groups},
        },
        "by_component": by_comp,
        "by_component_by_window": by_comp_by_window,
        "by_component_by_window_by_layer": by_comp_by_window_by_layer,
        "artifacts": {"csv": str(csv_path)},
    }

    return ExperimentResult(summary=summary)


