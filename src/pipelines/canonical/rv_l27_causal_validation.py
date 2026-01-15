"""
R_V causal validation experiment (value-projection patching) with required controls.

This experiment is meant to be *canonical*:
- Patch v_proj at a target layer using V from a recursive prompt.
- Controls: random norm-matched, shuffled-tokens, wrong-layer patch.
- Always logs per-pair rows + a summary.json.

Important nuance:
- We measure R_V at the *measurement layer* (usually the patch layer).
  This matches the repo's "measure at the intervention point" philosophy.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
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
    rec_id: str
    base_id: str
    rec_group: str
    base_group: str


def _token_len(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text))


def _capture_vproj_multi(
    model,
    tokenizer,
    text: str,
    layer_idxs: List[int],
    device: str,
    max_length: int = 512,
) -> Dict[int, Optional[torch.Tensor]]:
    """
    Capture v_proj outputs for multiple layers in a single forward pass.
    Returns tensors shaped (seq_len, hidden_dim) (batch removed), or None.
    """
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    storage: Dict[int, Optional[torch.Tensor]] = {i: None for i in layer_idxs}
    handles = []

    def make_hook(layer_idx: int):
        def hook_fn(_module, _inp, out):
            # out: (batch, seq, d)
            storage[layer_idx] = out.detach()[0]
            return out

        return hook_fn

    for idx in layer_idxs:
        layer = model.model.layers[idx].self_attn
        handles.append(layer.v_proj.register_forward_hook(make_hook(idx)))

    try:
        with torch.no_grad():
            model(**enc)
    finally:
        for h in handles:
            h.remove()

    return storage


def _patched_forward_capture_rv(
    model,
    tokenizer,
    baseline_text: str,
    patch_source: torch.Tensor,
    *,
    device: str,
    early_layer: int,
    patch_layer: int,
    window: int,
    patch_type: str,  # "recursive" | "random" | "shuffled" | "none"
    capture_extra_layer: Optional[int] = None,
    max_length: int = 512,
) -> Tuple[float, float, Optional[float]]:
    """
    Run a baseline prompt while patching v_proj at patch_layer (or not).
    Captures PR at early_layer and patch_layer (post-patch), returns (pr_early, pr_meas, pr_extra?).
    """
    enc = tokenizer(baseline_text, return_tensors="pt", truncation=True, max_length=max_length).to(device)

    v_early: Optional[torch.Tensor] = None
    v_meas: Optional[torch.Tensor] = None
    v_extra: Optional[torch.Tensor] = None

    def hook_capture_early(_module, _inp, out):
        nonlocal v_early
        v_early = out.detach()[0]
        return out

    def hook_patch_and_capture(_module, _inp, out):
        nonlocal v_meas
        # out: (batch, seq, d)
        out2 = out
        if patch_type != "none":
            out2 = out.clone()
            B, T, D = out2.shape
            src = patch_source.to(out2.device, dtype=out2.dtype)
            W = min(window, T, src.shape[0])
            if W > 0:
                if patch_type == "recursive":
                    patch = src[-W:, :]
                elif patch_type == "random":
                    src_w = src[-W:, :]
                    noise = torch.randn_like(src_w)
                    denom = float(noise.norm().item()) + 1e-12
                    noise = noise * (src_w.norm() / denom)
                    patch = noise
                elif patch_type == "shuffled":
                    perm = torch.randperm(W, device=src.device)
                    patch = src[-W:, :][perm, :]
                else:
                    raise ValueError(f"Unknown patch_type: {patch_type}")

                out2[:, -W:, :] = patch.unsqueeze(0).expand(B, -1, -1)

        v_meas = out2.detach()[0]
        return out2

    def hook_capture_extra(_module, _inp, out):
        nonlocal v_extra
        v_extra = out.detach()[0]
        return out

    handles = []
    try:
        handles.append(model.model.layers[early_layer].self_attn.v_proj.register_forward_hook(hook_capture_early))
        if patch_layer == early_layer:
            raise ValueError("early_layer and patch_layer must be different")
        handles.append(model.model.layers[patch_layer].self_attn.v_proj.register_forward_hook(hook_patch_and_capture))

        if capture_extra_layer is not None:
            # If extra equals patch_layer, it would capture the pre-patch output. Disallow.
            if capture_extra_layer in (early_layer, patch_layer):
                raise ValueError("capture_extra_layer must differ from early_layer and patch_layer")
            handles.append(model.model.layers[capture_extra_layer].self_attn.v_proj.register_forward_hook(hook_capture_extra))

        with torch.no_grad():
            model(**enc)
    finally:
        for h in handles:
            h.remove()

    pr_early = participation_ratio(v_early, window_size=window)
    pr_meas = participation_ratio(v_meas, window_size=window)
    pr_extra_out = participation_ratio(v_extra, window_size=window) if v_extra is not None else None
    return pr_early, pr_meas, pr_extra_out


def _rv_from_pr(pr_early: float, pr_late: float) -> float:
    if pr_early == 0 or math.isnan(pr_early) or math.isnan(pr_late):
        return float("nan")
    return float(pr_late / pr_early)


def _try_import_scipy():
    try:
        from scipy import stats  # type: ignore

        return stats
    except Exception:
        return None


def _mean_std(xs: List[float]) -> Dict[str, Any]:
    """Compute mean, std, n, and 95% CI for a list of values."""
    arr = np.asarray([x for x in xs if not np.isnan(x)], dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0.0, "ci_95": (float("nan"), float("nan"))}

    mean = float(arr.mean())
    std = float(arr.std(ddof=1) if arr.size > 1 else 0.0)
    n = float(arr.size)

    # 95% CI
    stats_mod = _try_import_scipy()
    if stats_mod is not None and arr.size >= 2:
        sem = stats_mod.sem(arr)
        ci = stats_mod.t.interval(0.95, arr.size - 1, loc=mean, scale=sem)
        ci_95 = (float(ci[0]), float(ci[1]))
    else:
        ci_95 = (float("nan"), float("nan"))

    return {"mean": mean, "std": std, "n": n, "ci_95": ci_95}


def run_rv_l27_causal_validation_from_config(cfg: Dict[str, Any], run_dir: Path) -> ExperimentResult:
    model_cfg = cfg.get("model") or {}
    params = cfg.get("params") or {}

    seed = int(cfg.get("seed") or 0)
    device = str(model_cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_name = str(model_cfg.get("name") or "mistralai/Mistral-7B-v0.1")

    early_layer = int(params.get("early_layer") or 5)
    target_layer = int(params.get("target_layer") or 27)
    wrong_layer = int(params.get("wrong_layer") or 21)
    window = int(params.get("window") or 16)
    max_pairs = int(params.get("max_pairs") or 45)
    max_length = int(params.get("max_length") or 512)

    pairing = params.get("pairing") or {}
    recursive_groups = pairing.get("recursive_groups") or ["L5_refined", "L4_full", "L3_deeper"]
    baseline_groups = pairing.get("baseline_groups") or ["long_control", "baseline_creative", "baseline_math"]

    measure_target_after_wrong_patch = bool(params.get("measure_target_after_wrong_patch") or False)

    # Load model + prompts
    set_seed(seed)
    model, tokenizer = load_model(model_name, device=device)
    loader = PromptLoader()
    bank = loader.prompts
    bank_version = loader.version
    
    # Log prompt bank version for reproducibility
    (run_dir / "prompt_bank_version.txt").write_text(bank_version)
    (run_dir / "prompt_bank_version.json").write_text(
        json.dumps({"version": bank_version}, indent=2) + "\n"
    )

    # Build candidate pairs from IDs so we can log provenance.
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

    rng = np.random.default_rng(seed + 12345)
    rng.shuffle(candidates)

    # Filter for baseline length >= window (so PR is defined) then cap.
    pairs: List[PairSpec] = []
    for spec in candidates:
        base_text = bank[spec.base_id]["text"]
        if _token_len(tokenizer, base_text) >= window:
            pairs.append(spec)
        if len(pairs) >= max_pairs:
            break

    rows: List[Dict[str, Any]] = []

    for pair_idx, spec in enumerate(pairs):
        rec_text = bank[spec.rec_id]["text"]
        base_text = bank[spec.base_id]["text"]

        # Capture recursive V at early + target (patch source)
        rec_vs = _capture_vproj_multi(
            model,
            tokenizer,
            rec_text,
            layer_idxs=[early_layer, target_layer],
            device=device,
            max_length=max_length,
        )
        v_rec_early = rec_vs[early_layer]
        v_rec_target = rec_vs[target_layer]

        # Baseline (no patch) at early + target
        base_vs = _capture_vproj_multi(
            model,
            tokenizer,
            base_text,
            layer_idxs=[early_layer, target_layer],
            device=device,
            max_length=max_length,
        )
        v_base_early = base_vs[early_layer]
        v_base_target = base_vs[target_layer]

        pr5_rec = participation_ratio(v_rec_early, window_size=window)
        prT_rec = participation_ratio(v_rec_target, window_size=window)
        pr5_base = participation_ratio(v_base_early, window_size=window)
        prT_base = participation_ratio(v_base_target, window_size=window)

        rv_rec = _rv_from_pr(pr5_rec, prT_rec)
        rv_base = _rv_from_pr(pr5_base, prT_base)

        # Main + controls: patch at target layer and measure at target layer
        pr5_main, prT_main, _ = _patched_forward_capture_rv(
            model,
            tokenizer,
            base_text,
            patch_source=v_rec_target if v_rec_target is not None else torch.empty(0),
            device=device,
            early_layer=early_layer,
            patch_layer=target_layer,
            window=window,
            patch_type="recursive",
            capture_extra_layer=None,
            max_length=max_length,
        )
        pr5_rand, prT_rand, _ = _patched_forward_capture_rv(
            model,
            tokenizer,
            base_text,
            patch_source=v_rec_target if v_rec_target is not None else torch.empty(0),
            device=device,
            early_layer=early_layer,
            patch_layer=target_layer,
            window=window,
            patch_type="random",
            capture_extra_layer=None,
            max_length=max_length,
        )
        pr5_shuf, prT_shuf, _ = _patched_forward_capture_rv(
            model,
            tokenizer,
            base_text,
            patch_source=v_rec_target if v_rec_target is not None else torch.empty(0),
            device=device,
            early_layer=early_layer,
            patch_layer=target_layer,
            window=window,
            patch_type="shuffled",
            capture_extra_layer=None,
            max_length=max_length,
        )

        rv_patch_main = _rv_from_pr(pr5_main, prT_main)
        rv_patch_rand = _rv_from_pr(pr5_rand, prT_rand)
        rv_patch_shuf = _rv_from_pr(pr5_shuf, prT_shuf)

        # Wrong-layer: patch at wrong_layer, measure at wrong_layer (and optionally also capture target_layer)
        # First get recursive V at wrong layer (source)
        wrong_vs = _capture_vproj_multi(
            model,
            tokenizer,
            rec_text,
            layer_idxs=[wrong_layer],
            device=device,
            max_length=max_length,
        )
        v_rec_wrong = wrong_vs[wrong_layer]

        extra = target_layer if measure_target_after_wrong_patch else None
        pr5_wrong, prW_wrong, prT_after_wrong = _patched_forward_capture_rv(
            model,
            tokenizer,
            base_text,
            patch_source=v_rec_wrong if v_rec_wrong is not None else torch.empty(0),
            device=device,
            early_layer=early_layer,
            patch_layer=wrong_layer,
            window=window,
            patch_type="recursive",
            capture_extra_layer=extra,
            max_length=max_length,
        )
        rv_patch_wronglayer = _rv_from_pr(pr5_wrong, prW_wrong)
        rv_target_after_wrongpatch = (
            _rv_from_pr(pr5_wrong, prT_after_wrong) if prT_after_wrong is not None else float("nan")
        )

        row = {
            "pair_idx": pair_idx,
            **asdict(spec),
            "early_layer": early_layer,
            "target_layer": target_layer,
            "wrong_layer": wrong_layer,
            "window": window,
            # baselines
            "rv_recursive": rv_rec,
            "rv_baseline": rv_base,
            # patched (target)
            "rv_patch_main": rv_patch_main,
            "rv_patch_random": rv_patch_rand,
            "rv_patch_shuffled": rv_patch_shuf,
            # wrong-layer patched (measured at wrong layer)
            "rv_patch_wronglayer": rv_patch_wronglayer,
            # optional: measure target after wrong-layer patch
            "rv_target_after_wrongpatch": rv_target_after_wrongpatch,
            # deltas vs baseline
            "delta_main": rv_patch_main - rv_base,
            "delta_random": rv_patch_rand - rv_base,
            "delta_shuffled": rv_patch_shuf - rv_base,
            "delta_wronglayer": rv_patch_wronglayer - rv_base,
        }
        rows.append(row)

    # Write per-pair CSV
    out_csv = run_dir / "rv_l27_causal_validation_pairs.csv"
    if rows:
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    # Summary stats + tests
    deltas_main = [r["delta_main"] for r in rows]
    deltas_rand = [r["delta_random"] for r in rows]
    deltas_shuf = [r["delta_shuffled"] for r in rows]
    deltas_wrong = [r["delta_wronglayer"] for r in rows]

    stats = _try_import_scipy()
    tests: Dict[str, Any] = {}
    if stats is not None:
        arr = np.asarray([x for x in deltas_main if not np.isnan(x)], dtype=np.float64)
        if arr.size > 1:
            t_stat, p_val = stats.ttest_1samp(arr, 0.0, alternative="less")
            # Cohen's d for one-sample t-test: mean / std
            cohens_d = float(arr.mean() / arr.std(ddof=1)) if arr.std(ddof=1) > 0 else 0.0
            tests["main_effect_ttest_1samp_less_0"] = {
                "t": float(t_stat),
                "p": float(p_val),
                "n": int(arr.size),
                "cohens_d": cohens_d,
            }

        # paired comparisons (main vs controls)
        def _paired(a: List[float], b: List[float]) -> Dict[str, float]:
            aa = np.asarray(a, dtype=np.float64)
            bb = np.asarray(b, dtype=np.float64)
            m = (~np.isnan(aa)) & (~np.isnan(bb))
            if int(m.sum()) < 2:
                return {"t": float("nan"), "p": float("nan"), "n": float(m.sum()), "cohens_d": float("nan")}
            t, p = stats.ttest_rel(aa[m], bb[m])
            # Cohen's d for paired: mean(diff) / std(diff)
            diff = aa[m] - bb[m]
            d = float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 0 else 0.0
            return {"t": float(t), "p": float(p), "n": float(m.sum()), "cohens_d": d}

        tests["main_vs_random_paired_ttest"] = _paired(deltas_main, deltas_rand)
        tests["main_vs_shuffled_paired_ttest"] = _paired(deltas_main, deltas_shuf)
        tests["main_vs_wronglayer_paired_ttest"] = _paired(deltas_main, deltas_wrong)

    # Transfer efficiency estimate (vs natural gap)
    rv_rec_list = [r["rv_recursive"] for r in rows]
    rv_base_list = [r["rv_baseline"] for r in rows]
    
    # gap: How much R_V drops from baseline to recursive (Positive value, e.g. 1.0 - 0.5 = 0.5)
    gap = float(np.nanmean(rv_base_list) - np.nanmean(rv_rec_list))
    
    # restored: How much the patch pulls R_V down from baseline towards recursive
    # delta_main is (patch - baseline). If patch works, it's negative (e.g. 0.55 - 1.0 = -0.45).
    # So we want (baseline - patch) which is -delta_main.
    restored = float(np.nanmean(rv_base_list) - np.nanmean([r["rv_patch_main"] for r in rows]))
    
    transfer = float(restored / gap * 100.0) if rows and gap > 1e-9 else 0.0

    summary = {
        "experiment": "rv_l27_causal_validation",
        "model_name": model_name,
        "device": device,
        "seed": seed,
        "prompt_bank_version": bank_version,
        "params": {
            "early_layer": early_layer,
            "target_layer": target_layer,
            "wrong_layer": wrong_layer,
            "window": window,
            "max_pairs": max_pairs,
            "max_length": max_length,
            "pairing": {"recursive_groups": recursive_groups, "baseline_groups": baseline_groups},
            "measure_target_after_wrong_patch": measure_target_after_wrong_patch,
        },
        "n_pairs": len(rows),
        "rv_recursive": _mean_std(rv_rec_list),
        "rv_baseline": _mean_std(rv_base_list),
        "delta_main": _mean_std(deltas_main),
        "delta_random": _mean_std(deltas_rand),
        "delta_shuffled": _mean_std(deltas_shuf),
        "delta_wronglayer": _mean_std(deltas_wrong),
        "transfer_percent_estimate": transfer,
        "tests": tests,
        "artifacts": {"pairs_csv": str(out_csv)},
        "notes": {
            "measurement": "main/random/shuffled measured at target_layer; wronglayer measured at wrong_layer",
        },
    }

    return ExperimentResult(summary=summary)


