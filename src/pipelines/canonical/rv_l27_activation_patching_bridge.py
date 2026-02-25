"""
R_V activation patching bridge experiment with behavioral scoring.

Goal:
- Patch V_PROJ at Layer 27 using recursive prompt activations.
- Measure prompt-pass R_V for recursive/baseline/patch conditions.
- Generate baseline vs patched outputs and score behavioral metrics.

This is designed to test causal influence of geometric contraction on behavior.
"""

from __future__ import annotations

import csv
import math
import random
import signal
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy import stats
from tqdm import tqdm

from prompts.loader import PromptLoader
from src.core.models import load_model, set_seed
from src.core.patching import PersistentVPatcher, extract_v_activation
from src.core.head_specific_patching import HeadSpecificVPatcher
from src.metrics.behavior_strict import score_behavior_strict
from src.metrics.behavioral_bridge import extract_bridge_metrics
from src.metrics.logit_diff import LogitDiffMetric
from src.metrics.rv import compute_rv_with_components, participation_ratio
from src.pipelines.registry import ExperimentResult


def _token_len(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text))


def _compute_cohens_d(values: List[float]) -> Optional[float]:
    clean = [v for v in values if not np.isnan(v)]
    if len(clean) < 2:
        return None
    std = float(np.std(clean, ddof=1))
    if std < 1e-10:
        return None
    return float(np.mean(clean) / std)


def _safe_mean(values: List[float]) -> Optional[float]:
    clean = [v for v in values if not np.isnan(v)]
    if not clean:
        return None
    return float(np.mean(clean))


def _get_head_dim_ranges(
    heads: List[int], head_dim: int
) -> List[Tuple[int, int]]:
    """Get (start, end) dim ranges for specified attention heads."""
    return [(h * head_dim, (h + 1) * head_dim) for h in heads]


def _infer_vproj_layout(model) -> Tuple[int, int]:
    """
    Infer (head_dim, num_v_heads) for the v_proj hookpoint.

    For GQA models (e.g. Mistral): num_v_heads == num_key_value_heads.
    For MHA models: num_v_heads == num_attention_heads.
    """
    cfg = model.config
    num_q_heads = getattr(cfg, "num_attention_heads", None) or getattr(cfg, "n_head", None)
    hidden_size = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None)
    if num_q_heads is None or hidden_size is None:
        raise ValueError(
            "Cannot infer head layout from model.config (need num_attention_heads|n_head and hidden_size|n_embd)."
        )
    num_q_heads = int(num_q_heads)
    hidden_size = int(hidden_size)
    head_dim = int(getattr(cfg, "head_dim", None) or (hidden_size // num_q_heads))

    num_kv_heads = int(getattr(cfg, "num_key_value_heads", num_q_heads))
    # v_proj output is KV-head space under GQA: (num_kv_heads * head_dim).
    num_v_heads = num_kv_heads
    return head_dim, num_v_heads


def _resolve_kv_heads(
    model,
    requested_heads: List[int],
    *,
    head_space: str,
    num_v_heads: int,
) -> List[int]:
    """
    Resolve configured head indices into KV/V-head indices (0..num_v_heads-1).

    - head_space="kv": indices are interpreted directly as KV/V heads.
    - head_space="q": indices are query heads; we map them to KV/V heads via
      contiguous grouping (same as HF repeat_kv): kv = q // group_size.
    """
    cfg = model.config
    num_q_heads = getattr(cfg, "num_attention_heads", None) or getattr(cfg, "n_head", None)
    if num_q_heads is None:
        raise ValueError("Cannot infer num_attention_heads from model.config (need num_attention_heads|n_head).")
    num_q_heads = int(num_q_heads)

    if head_space not in ("kv", "q"):
        raise ValueError(f"head_space must be 'kv' or 'q' (got {head_space!r})")

    if head_space == "kv":
        bad = [h for h in requested_heads if h < 0 or h >= num_v_heads]
        if bad:
            raise ValueError(f"KV/V heads out of range (0..{num_v_heads-1}): {bad}")
        return sorted(set(int(h) for h in requested_heads))

    # q -> kv mapping
    if num_q_heads % num_v_heads != 0:
        raise ValueError(
            f"num_attention_heads {num_q_heads} must be divisible by num_v_heads {num_v_heads}."
        )
    group_size = num_q_heads // num_v_heads
    bad = [h for h in requested_heads if h < 0 or h >= num_q_heads]
    if bad:
        raise ValueError(f"Q heads out of range (0..{num_q_heads-1}): {bad}")
    return sorted(set(int(h) // group_size for h in requested_heads))


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
    patch_mode: str = "full",
    patch_heads: Optional[List[int]] = None,
    head_space: str = "kv",
) -> Tuple[float, float]:
    """
    Run a baseline prompt while patching v_proj at patch_layer.
    Returns (pr_early, pr_patch_layer) after patching.

    Args:
        patch_mode: "full" (all dims), "head_specific" (only patch_heads dims),
                    "random_head" (random head dims, norm-matched control).
        patch_heads: List of head indices to patch when mode is "head_specific".
    """
    enc = tokenizer(baseline_text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    v_early: Optional[torch.Tensor] = None
    v_patch: Optional[torch.Tensor] = None

    # Precompute dim ranges for head-specific modes (KV/V-head space).
    head_dim, num_v_heads = _infer_vproj_layout(model)
    effective_kv_heads: Optional[List[int]] = None
    head_dim_ranges: Optional[List[Tuple[int, int]]] = None
    if patch_mode == "head_specific" and patch_heads is not None:
        effective_kv_heads = _resolve_kv_heads(model, patch_heads, head_space=head_space, num_v_heads=num_v_heads)
        head_dim_ranges = _get_head_dim_ranges(effective_kv_heads, head_dim=head_dim)
    elif patch_mode == "random_head":
        # Sample the same number of KV heads as the target condition.
        requested = patch_heads or [2]
        effective_kv_heads = _resolve_kv_heads(model, requested, head_space=head_space, num_v_heads=num_v_heads)
        target_set = set(effective_kv_heads)
        all_heads = [h for h in range(num_v_heads) if h not in target_set]
        rng = random.Random(42)
        k = max(1, len(effective_kv_heads))
        random_heads = rng.sample(all_heads, min(k, len(all_heads)))
        head_dim_ranges = _get_head_dim_ranges(random_heads, head_dim=head_dim)

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
            if patch_mode == "full":
                # Patch all dims (original behavior)
                out2[:, -W:, :] = src[-W:, :].unsqueeze(0).expand(B, -1, -1)
            elif head_dim_ranges is not None:
                # Patch only specific head dims
                for start_dim, end_dim in head_dim_ranges:
                    out2[:, -W:, start_dim:end_dim] = (
                        src[-W:, start_dim:end_dim]
                        .unsqueeze(0)
                        .expand(B, -1, -1)
                    )
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


def _generate_text(
    model,
    tokenizer,
    prompt: str,
    *,
    device: str,
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
    top_p: float,
    max_length: int,
    timeout_sec: Optional[int] = None,
) -> Tuple[str, bool, int]:
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    input_len = enc["input_ids"].shape[1]

    class _GenerationTimeout(Exception):
        pass

    def _run_generate() -> torch.Tensor:
        with torch.no_grad():
            return model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
            )

    if timeout_sec is not None and timeout_sec > 0 and hasattr(signal, "SIGALRM"):
        def _timeout_handler(_sig, _frame):
            raise _GenerationTimeout(f"generation timed out after {timeout_sec}s")

        prev_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(int(timeout_sec))
        try:
            output_ids = _run_generate()
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, prev_handler)
    else:
        output_ids = _run_generate()

    gen_len = int(output_ids.shape[1] - input_len)
    truncated = gen_len >= max_new_tokens
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return text, truncated, gen_len


def run_rv_l27_activation_patching_bridge_from_config(
    cfg: Dict[str, Any],
    run_dir: Path,
) -> ExperimentResult:
    """
    Run activation patching with behavioral scoring from config.
    """
    params = cfg.get("params", {})
    model_name = params.get("model", "mistralai/Mistral-7B-v0.1")
    device = params.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    seed = int(params.get("seed", 42))
    n_pairs = int(params.get("n_pairs", 20))
    early_layer = int(params.get("early_layer", 5))
    patch_layer = int(params.get("target_layer", 27))
    window = int(params.get("window", 16))
    max_length = int(params.get("max_length", 512))
    max_new_tokens = int(params.get("max_new_tokens", 800))
    generation_timeout_sec = params.get("generation_timeout_sec")
    if generation_timeout_sec is not None:
        generation_timeout_sec = int(generation_timeout_sec)
    temperature = float(params.get("temperature", 0.0))
    do_sample = bool(params.get("do_sample", temperature > 0.0))
    top_p = float(params.get("top_p", 0.95))
    recursive_groups = params.get("recursive_groups", ["L3_deeper", "L4_full", "L5_refined"])
    baseline_groups = params.get("baseline_groups", ["long_control", "baseline_creative", "baseline_math"])
    skip_short_baseline = bool(params.get("skip_short_baseline", True))
    patch_mode = params.get("patch_mode", "full")  # "full", "head_specific", "random_head"
    patch_heads = params.get("patch_heads", [2])  # KV/V heads for head_specific mode
    head_space = params.get("head_space", "kv")  # "kv" (preferred) or "q" (maps Q heads -> KV heads)
    donor_type = params.get("donor_type", "recursive")  # "recursive" or "baseline"

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
    pair_errors: List[Dict[str, Any]] = []
    truncated_baseline = 0
    truncated_patched = 0
    skipped_short = 0
    checkpoint_every = max(1, int(params.get("checkpoint_every", 1)))

    per_sample_path = run_dir / "per_sample.csv"
    error_path = run_dir / "pair_errors.csv"

    def _flush_rows() -> None:
        if not rows:
            return
        with per_sample_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def _flush_errors() -> None:
        if not pair_errors:
            return
        with error_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(pair_errors[0].keys()))
            writer.writeheader()
            writer.writerows(pair_errors)

    print(f"\n{'='*60}")
    print(f"ACTIVATION PATCHING BRIDGE (n={pair_count})")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    head_dim, num_v_heads = _infer_vproj_layout(model)
    effective_kv_heads = None
    if patch_mode != "full":
        try:
            effective_kv_heads = _resolve_kv_heads(model, patch_heads, head_space=head_space, num_v_heads=num_v_heads)
        except Exception:
            effective_kv_heads = None
    print(
        f"Patch mode: {patch_mode} (head_space={head_space}, requested={patch_heads}, "
        f"effective_kv={effective_kv_heads}), Donor: {donor_type}"
    )
    print(f"Patch layer: {patch_layer}, Early layer: {early_layer}, Window: {window}")
    print(f"Generation: max_new_tokens={max_new_tokens}, temp={temperature}")
    print(f"{'='*60}\n")

    for pair_index, ((rec_id, rec_text, rec_group), (base_id, base_text, base_group)) in enumerate(
        tqdm(pairs, desc="Bridge pairs", unit="pair", dynamic_ncols=True),
        start=1,
    ):
        base_len = _token_len(tokenizer, base_text)
        rec_len = _token_len(tokenizer, rec_text)

        if skip_short_baseline and base_len < window:
            skipped_short += 1
            continue

        try:
            # Prompt-pass R_V (recursive, baseline)
            rv_rec, pr5_rec, pr27_rec = compute_rv_with_components(
                model, tokenizer, rec_text, early=early_layer, late=patch_layer, window=window, device=device
            )
            rv_base, pr5_base, pr27_base = compute_rv_with_components(
                model, tokenizer, base_text, early=early_layer, late=patch_layer, window=window, device=device
            )

            # Patch source — donor type determines which prompt's V_PROJ we use
            if donor_type == "recursive":
                v_source = extract_v_activation(model, tokenizer, rec_text, layer_idx=patch_layer, device=device)
            elif donor_type == "baseline":
                # Use baseline prompt from NEXT pair as donor (different from target)
                current_idx = next(i for i, p in enumerate(pairs) if p[1][0] == base_id)
                donor_pair_idx = (current_idx + 1) % len(pairs)
                donor_text = pairs[donor_pair_idx][1][1]  # baseline text from next pair
                v_source = extract_v_activation(model, tokenizer, donor_text, layer_idx=patch_layer, device=device)
            else:
                raise ValueError(f"Unknown donor_type: {donor_type}")

            # Prompt-pass R_V with patch applied to baseline
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
                patch_mode=patch_mode,
                patch_heads=patch_heads,
                head_space=head_space,
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

            # Generate baseline (unpatched)
            baseline_out, base_trunc, base_gen_len = _generate_text(
                model,
                tokenizer,
                base_text,
                device=device,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                top_p=top_p,
                max_length=max_length,
                timeout_sec=generation_timeout_sec,
            )
            if base_trunc:
                truncated_baseline += 1

            # Generate baseline (patched) — uses mode-appropriate patcher
            if patch_mode == "full":
                patcher = PersistentVPatcher(model, v_source)
                patcher.register(patch_layer)
            elif patch_mode in ("head_specific", "random_head"):
                # Determine which KV/V heads to patch (sampled in KV space).
                _head_dim, num_v_heads = _infer_vproj_layout(model)
                effective = _resolve_kv_heads(model, patch_heads, head_space=head_space, num_v_heads=num_v_heads)
                if patch_mode == "head_specific":
                    target_heads_gen = effective
                else:  # random_head control
                    target_set = set(effective)
                    all_heads = [h for h in range(num_v_heads) if h not in target_set]
                    gen_rng = random.Random(42)
                    k = max(1, len(effective))
                    target_heads_gen = gen_rng.sample(all_heads, min(k, len(all_heads)))
                patcher = HeadSpecificVPatcher(
                    model,
                    v_source,
                    target_heads=target_heads_gen,
                    window_size=window,
                    head_space="kv",
                )
                patcher.register(patch_layer)
            else:
                raise ValueError(f"Unknown patch_mode: {patch_mode}")
            try:
                patched_out, patch_trunc, patch_gen_len = _generate_text(
                    model,
                    tokenizer,
                    base_text,
                    device=device,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    top_p=top_p,
                    max_length=max_length,
                    timeout_sec=generation_timeout_sec,
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
        except Exception as exc:
            pair_errors.append(
                {
                    "pair_index": pair_index,
                    "rec_id": rec_id,
                    "base_id": base_id,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(limit=5),
                }
            )
            _flush_errors()
            print(
                f"[warning] pair {pair_index}/{pair_count} failed: {type(exc).__name__}: {exc}",
                flush=True,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        if (pair_index % checkpoint_every) == 0:
            _flush_rows()
            _flush_errors()
            print(
                f"[progress] pair={pair_index}/{pair_count} kept={len(rows)} errors={len(pair_errors)}",
                flush=True,
            )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Final write for per-sample + errors.
    _flush_rows()
    _flush_errors()

    # Summary stats
    n_valid = len(rows)
    rv_delta_mean = _safe_mean(rv_deltas)
    logit_delta_mean = _safe_mean(logit_deltas)
    behavior_strict_delta_mean = _safe_mean(behavior_strict_deltas)
    word_count_delta_mean = _safe_mean(word_count_deltas)
    l4_count_delta_mean = _safe_mean(l4_count_deltas)

    rv_p_value = None
    if len(rv_deltas) >= 3:
        rv_t, rv_p_value = stats.ttest_1samp(rv_deltas, 0.0)
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
        "experiment": "rv_l27_activation_patching_bridge",
        "version": "v4_gqa_headspace",
        "patch_mode": patch_mode,
        "head_space": head_space if patch_mode != "full" else None,
        "patch_heads_requested": patch_heads if patch_mode != "full" else "all",
        "patch_kv_heads_effective": (
            _resolve_kv_heads(model, patch_heads, head_space=head_space, num_v_heads=_infer_vproj_layout(model)[1])
            if patch_mode != "full"
            else "all"
        ),
        "v_head_dim": head_dim,
        "v_num_heads": num_v_heads,
        "donor_type": donor_type,
        "model": model_name,
        "device": device,
        "n_pairs": n_valid,
        "seed": seed,
        "early_layer": early_layer,
        "target_layer": patch_layer,
        "window": window,
        "max_length": max_length,
        "max_new_tokens": max_new_tokens,
        "generation_timeout_sec": generation_timeout_sec,
        "temperature": temperature,
        "do_sample": do_sample,
        "top_p": top_p,
        "recursive_groups": recursive_groups,
        "baseline_groups": baseline_groups,
        "n_skipped_short_baseline": skipped_short,
        "n_pair_errors": len(pair_errors),
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
        "artifacts": {
            "per_sample_csv": str(per_sample_path),
            "pair_errors_csv": str(error_path) if pair_errors else None,
        },
    }

    return ExperimentResult(summary=summary)
