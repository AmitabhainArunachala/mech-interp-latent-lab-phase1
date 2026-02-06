"""
R_V KV-cache + L27 V_PROJ patching bridge experiment with behavioral scoring.

Goal:
- Patch KV cache (content anchor) using recursive prompt.
- Patch V_PROJ at L27 (geometry) using recursive V activations.
- Measure prompt-pass R_V for recursive/baseline/patch conditions.
- Generate baseline vs patched outputs and score behavior.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy import stats
from transformers.cache_utils import DynamicCache

from prompts.loader import PromptLoader
from src.core.models import load_model, set_seed
from src.core.patching import PersistentVPatcher, extract_v_activation
from src.metrics.behavior_strict import score_behavior_strict
from src.metrics.behavioral_bridge import extract_bridge_metrics
from src.metrics.logit_diff import LogitDiffMetric
from src.metrics.rv import compute_rv_with_components, participation_ratio
from src.pipelines.registry import ExperimentResult


def _token_len(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text))


def _safe_mean(values: List[float]) -> Optional[float]:
    clean = [v for v in values if not np.isnan(v)]
    if not clean:
        return None
    return float(np.mean(clean))


def _compute_cohens_d(values: List[float]) -> Optional[float]:
    clean = [v for v in values if not np.isnan(v)]
    if len(clean) < 2:
        return None
    std = float(np.std(clean, ddof=1))
    if std < 1e-10:
        return None
    return float(np.mean(clean) / std)


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
    max_length: int = 512,
) -> Tuple[float, float]:
    """
    Run a baseline prompt while patching v_proj at patch_layer.
    Returns (pr_early, pr_patch_layer) after patching.
    """
    enc = tokenizer(baseline_text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    v_early: Optional[torch.Tensor] = None
    v_patch: Optional[torch.Tensor] = None

    def hook_capture_early(_module, _inp, out):
        nonlocal v_early
        v_early = out.detach()[0]
        return out

    def hook_patch_and_capture(_module, _inp, out):
        nonlocal v_patch
        out2 = out.clone()
        B, T, _D = out2.shape
        src = patch_source.to(out2.device, dtype=out2.dtype)
        W = min(window, T, src.shape[0])
        if W > 0:
            out2[:, -W:, :] = src[-W:, :].unsqueeze(0).expand(B, -1, -1)
        v_patch = out2.detach()[0]
        return out2

    layer_early = model.model.layers[early_layer].self_attn
    layer_patch = model.model.layers[patch_layer].self_attn

    h_early = layer_early.v_proj.register_forward_hook(hook_capture_early)
    h_patch = layer_patch.v_proj.register_forward_hook(hook_patch_and_capture)
    try:
        with torch.no_grad():
            model(**enc)
    finally:
        h_patch.remove()
        h_early.remove()

    pr_early = participation_ratio(v_early, window_size=window)
    pr_patch = participation_ratio(v_patch, window_size=window)
    return pr_early, pr_patch


def _build_kv_cache(
    model,
    tokenizer,
    prompt: str,
    *,
    device: str,
    max_length: int,
) -> Tuple[Any, torch.Tensor]:
    """Build KV cache from a prompt. Returns (past_key_values, input_ids)."""
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    with torch.no_grad():
        out = model(**enc, use_cache=True)
    return out.past_key_values, enc["input_ids"]


def _get_kv_seq_length(past_key_values) -> int:
    """Get sequence length from KV cache (handles both tuple and DynamicCache)."""
    if hasattr(past_key_values, 'get_seq_length'):
        return past_key_values.get_seq_length()
    # Tuple format: ((k1, v1), (k2, v2), ...) where k has shape [B, H, T, D]
    return past_key_values[0][0].shape[2]


def _patch_kv_cache(
    base_kv,
    rec_kv,
    *,
    patch_window: int,
):
    """Patch last patch_window tokens of base KV with recursive KV."""
    # Handle both tuple and DynamicCache formats
    if hasattr(base_kv, '__len__'):
        num_layers = len(base_kv)
    else:
        num_layers = base_kv.get_num_layers() if hasattr(base_kv, 'get_num_layers') else 32
    
    patched_layers = []
    for layer_idx in range(num_layers):
        # Access layer KV (works for both tuple and DynamicCache)
        if hasattr(base_kv, '__getitem__'):
            k_base, v_base = base_kv[layer_idx]
            k_rec, v_rec = rec_kv[layer_idx]
        else:
            k_base, v_base = base_kv.key_cache[layer_idx], base_kv.value_cache[layer_idx]
            k_rec, v_rec = rec_kv.key_cache[layer_idx], rec_kv.value_cache[layer_idx]
        
        k_out = k_base.clone()
        v_out = v_base.clone()
        L = min(k_base.shape[2], k_rec.shape[2], patch_window)
        if L > 0:
            k_out[:, :, -L:, :] = k_rec[:, :, -L:, :].to(k_out.dtype)
            v_out[:, :, -L:, :] = v_rec[:, :, -L:, :].to(v_out.dtype)
        patched_layers.append((k_out, v_out))
    
    # Return as tuple (works universally)
    return tuple(patched_layers)


def _generate_with_kv(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    *,
    kv_cache,
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
    top_p: float,
    device: str,
) -> Tuple[str, bool, int]:
    """Generate text using KV cache."""
    generated = input_ids.clone()
    current_kv = kv_cache
    eos_reached = False
    
    for _ in range(max_new_tokens):
        # Get past length for position_ids
        past_len = _get_kv_seq_length(current_kv) if current_kv is not None else 0
        
        with torch.no_grad():
            out = model(
                generated[:, -1:],
                past_key_values=current_kv,
                use_cache=True,
            )
        logits = out.logits[:, -1, :]
        
        if do_sample and temperature > 0:
            probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
            if top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cum_probs = torch.cumsum(sorted_probs, dim=-1)
                cutoff = cum_probs > top_p
                cutoff[..., 0] = False
                sorted_probs[cutoff] = 0.0
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                next_token = torch.multinomial(sorted_probs, num_samples=1)
                next_tok = sorted_idx.gather(-1, next_token)
            else:
                next_tok = torch.multinomial(probs, num_samples=1)
        else:
            next_tok = logits.argmax(dim=-1, keepdim=True)
        
        generated = torch.cat([generated, next_tok], dim=1)
        current_kv = out.past_key_values
        
        if tokenizer.eos_token_id is not None and next_tok.item() == tokenizer.eos_token_id:
            eos_reached = True
            break
    
    gen_len = int(generated.shape[1] - input_ids.shape[1])
    truncated = not eos_reached and gen_len >= max_new_tokens
    text = tokenizer.decode(generated[0], skip_special_tokens=True)
    return text, truncated, gen_len


def run_rv_l27_kv_patching_bridge_from_config(
    cfg: Dict[str, Any],
    run_dir: Path,
) -> ExperimentResult:
    params = cfg.get("params", {})
    model_name = params.get("model", "mistralai/Mistral-7B-v0.1")
    device = params.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    seed = int(params.get("seed", 42))
    n_pairs = int(params.get("n_pairs", 20))
    early_layer = int(params.get("early_layer", 5))
    patch_layer = int(params.get("target_layer", 27))
    window = int(params.get("window", 16))
    patch_window = int(params.get("patch_window", window))
    max_length = int(params.get("max_length", 512))
    max_new_tokens = int(params.get("max_new_tokens", 800))
    temperature = float(params.get("temperature", 0.0))
    do_sample = bool(params.get("do_sample", temperature > 0.0))
    top_p = float(params.get("top_p", 0.95))
    recursive_groups = params.get("recursive_groups", ["L3_deeper", "L4_full", "L5_refined"])
    baseline_groups = params.get("baseline_groups", ["long_control", "baseline_creative", "baseline_math"])
    skip_short_baseline = bool(params.get("skip_short_baseline", True))

    set_seed(seed)
    model, tokenizer = load_model(model_name, device=device)
    model.eval()

    loader = PromptLoader()
    rng = random.Random(seed)

    rec_items = [
        (k, v["text"], v.get("group"))
        for k, v in loader.prompts.items()
        if v.get("group") in recursive_groups
    ]
    base_items = [
        (k, v["text"], v.get("group"))
        for k, v in loader.prompts.items()
        if v.get("group") in baseline_groups
    ]

    rng.shuffle(rec_items)
    rng.shuffle(base_items)
    pair_count = min(n_pairs, len(rec_items), len(base_items))
    pairs = list(zip(rec_items[:pair_count], base_items[:pair_count]))

    logit_metric = LogitDiffMetric(tokenizer, device=device)

    rows: List[Dict[str, Any]] = []
    rv_deltas: List[float] = []
    logit_deltas: List[float] = []
    behavior_strict_deltas: List[float] = []
    word_count_deltas: List[float] = []
    l4_count_deltas: List[float] = []
    truncated_baseline = 0
    truncated_patched = 0
    skipped_short = 0

    for (rec_id, rec_text, rec_group), (base_id, base_text, base_group) in pairs:
        base_len = _token_len(tokenizer, base_text)
        rec_len = _token_len(tokenizer, rec_text)

        if skip_short_baseline and base_len < window:
            skipped_short += 1
            continue

        # Prompt-pass R_V
        rv_rec, pr5_rec, pr27_rec = compute_rv_with_components(
            model, tokenizer, rec_text, early=early_layer, late=patch_layer, window=window, device=device
        )
        rv_base, pr5_base, pr27_base = compute_rv_with_components(
            model, tokenizer, base_text, early=early_layer, late=patch_layer, window=window, device=device
        )

        # V-proj patch source
        v_source = extract_v_activation(model, tokenizer, rec_text, layer_idx=patch_layer, device=device)

        # Prompt-pass R_V with v_proj patch applied (geometry transfer)
        pr5_patch, pr27_patch = _patched_forward_capture_rv(
            model,
            tokenizer,
            base_text,
            v_source,
            device=device,
            early_layer=early_layer,
            patch_layer=patch_layer,
            window=window,
            max_length=max_length,
        )
        rv_patch = pr27_patch / pr5_patch if (pr5_patch and pr5_patch > 0) else float("nan")
        delta = rv_patch - rv_base

        # Logit diff on prompt-pass
        with torch.no_grad():
            rec_logits = model(**tokenizer(rec_text, return_tensors="pt", truncation=True, max_length=max_length).to(device)).logits
            base_logits = model(**tokenizer(base_text, return_tensors="pt", truncation=True, max_length=max_length).to(device)).logits
        logit_rec = logit_metric.compute(rec_logits, position=-1).logit_diff
        logit_base = logit_metric.compute(base_logits, position=-1).logit_diff
        logit_delta = logit_rec - logit_base

        # Build KV caches (recursive and baseline)
        rec_kv, _rec_ids = _build_kv_cache(model, tokenizer, rec_text, device=device, max_length=max_length)
        base_kv, base_ids = _build_kv_cache(model, tokenizer, base_text, device=device, max_length=max_length)
        patched_kv = _patch_kv_cache(base_kv, rec_kv, patch_window=patch_window)

        # Generate baseline (no KV patch, no vproj patch)
        baseline_out, base_trunc, base_gen_len = _generate_with_kv(
            model,
            tokenizer,
            base_ids,
            kv_cache=base_kv,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            top_p=top_p,
            device=device,
        )
        if base_trunc:
            truncated_baseline += 1

        # Generate patched (KV + persistent V_PROJ)
        patcher = PersistentVPatcher(model, v_source)
        patcher.register(patch_layer)
        try:
            patched_out, patch_trunc, patch_gen_len = _generate_with_kv(
                model,
                tokenizer,
                base_ids,
                kv_cache=patched_kv,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                top_p=top_p,
                device=device,
            )
        finally:
            patcher.remove()
        if patch_trunc:
            truncated_patched += 1

        # Behavior metrics
        base_bridge = extract_bridge_metrics(baseline_out)
        patch_bridge = extract_bridge_metrics(patched_out)
        base_strict = score_behavior_strict(baseline_out)
        patch_strict = score_behavior_strict(patched_out)

        behavior_strict_delta = patch_strict.final_score - base_strict.final_score
        word_count_delta = patch_bridge.word_count - base_bridge.word_count
        l4_count_delta = patch_bridge.l4_count - base_bridge.l4_count

        rv_deltas.append(delta)
        logit_deltas.append(logit_delta)
        behavior_strict_deltas.append(behavior_strict_delta)
        word_count_deltas.append(word_count_delta)
        l4_count_deltas.append(l4_count_delta)

        rows.append(
            {
                "rec_id": rec_id,
                "rec_group": rec_group,
                "base_id": base_id,
                "base_group": base_group,
                "rec_len": rec_len,
                "base_len": base_len,
                "rv_rec": rv_rec,
                "rv_base": rv_base,
                "rv_patch": rv_patch,
                "rv_delta": delta,
                "pr5_rec": pr5_rec,
                "pr27_rec": pr27_rec,
                "pr5_base": pr5_base,
                "pr27_base": pr27_base,
                "pr5_patch": pr5_patch,
                "pr27_patch": pr27_patch,
                "logit_diff_rec": logit_rec,
                "logit_diff_base": logit_base,
                "logit_diff_delta": logit_delta,
                "baseline_output": baseline_out,
                "patched_output": patched_out,
                "baseline_truncated": base_trunc,
                "patched_truncated": patch_trunc,
                "baseline_gen_len": base_gen_len,
                "patched_gen_len": patch_gen_len,
                "baseline_strict_score": base_strict.final_score,
                "patched_strict_score": patch_strict.final_score,
                "baseline_l4_count": base_bridge.l4_count,
                "patched_l4_count": patch_bridge.l4_count,
                "baseline_word_count": base_bridge.word_count,
                "patched_word_count": patch_bridge.word_count,
                "baseline_unique_ratio": base_bridge.unique_word_ratio,
                "patched_unique_ratio": patch_bridge.unique_word_ratio,
            }
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    per_sample_path = run_dir / "per_sample.csv"
    if rows:
        with per_sample_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    n_valid = len(rows)
    rv_delta_mean = _safe_mean(rv_deltas)
    logit_delta_mean = _safe_mean(logit_deltas)
    behavior_strict_delta_mean = _safe_mean(behavior_strict_deltas)
    word_count_delta_mean = _safe_mean(word_count_deltas)
    l4_count_delta_mean = _safe_mean(l4_count_deltas)

    rv_p_value = None
    if len(rv_deltas) >= 3:
        _, rv_p_value = stats.ttest_1samp(rv_deltas, 0.0)
        rv_p_value = float(rv_p_value)

    logit_p_value = None
    if len(logit_deltas) >= 3:
        _, logit_p_value = stats.ttest_1samp(logit_deltas, 0.0)
        logit_p_value = float(logit_p_value)

    behavior_p_value = None
    if len(behavior_strict_deltas) >= 3:
        _, behavior_p_value = stats.ttest_1samp(behavior_strict_deltas, 0.0)
        behavior_p_value = float(behavior_p_value)

    summary = {
        "experiment": "rv_l27_kv_patching_bridge",
        "version": "v1",
        "model": model_name,
        "device": device,
        "n_pairs": n_valid,
        "seed": seed,
        "early_layer": early_layer,
        "target_layer": patch_layer,
        "window": window,
        "patch_window": patch_window,
        "max_length": max_length,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "do_sample": do_sample,
        "top_p": top_p,
        "recursive_groups": recursive_groups,
        "baseline_groups": baseline_groups,
        "n_skipped_short_baseline": skipped_short,
        "n_truncated_baseline": truncated_baseline,
        "n_truncated_patched": truncated_patched,
        "rv_recursive_mean": _safe_mean([r["rv_rec"] for r in rows]),
        "rv_baseline_mean": _safe_mean([r["rv_base"] for r in rows]),
        "rv_patched_mean": _safe_mean([r["rv_patch"] for r in rows]),
        "rv_delta_mean": rv_delta_mean,
        "rv_cohens_d": _compute_cohens_d(rv_deltas),
        "rv_p_value": rv_p_value,
        "logit_diff_delta_mean": logit_delta_mean,
        "logit_diff_cohens_d": _compute_cohens_d(logit_deltas),
        "logit_diff_p_value": logit_p_value,
        "behavior_strict_delta_mean": behavior_strict_delta_mean,
        "behavior_strict_p_value": behavior_p_value,
        "behavior_word_count_delta_mean": word_count_delta_mean,
        "behavior_l4_count_delta_mean": l4_count_delta_mean,
        "schema_version": "metrics_summary_v1",
        "prompt_bank_version": loader.version,
        "timestamp": run_dir.name.split("_")[0] if "_" in run_dir.name else "",
        "artifacts": {
            "per_sample_csv": str(per_sample_path),
            "config": str(run_dir / "config.json"),
            "report": str(run_dir / "report.md"),
            "summary": str(run_dir / "summary.json"),
        },
    }

    return ExperimentResult(summary=summary)
