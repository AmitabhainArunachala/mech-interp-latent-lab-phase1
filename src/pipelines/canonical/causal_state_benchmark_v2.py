"""
Causal state benchmark v2.

Hardens v1 by:
- repeating generation with multiple seeds per prompt-condition
- using an alpha ladder instead of a single toward/away setting
- reporting prompt-cluster bootstrap CIs
- separating the primary recursive endpoint from the baseline specificity control
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from prompts.loader import PromptLoader
from src.pipelines.registry import ExperimentResult

from .causal_state_benchmark_v1 import (
    DEFAULT_NEGATIVE_CLASSES,
    DEFAULT_POSITIVE_CLASSES,
    InterventionSpec,
    _aggregate_condition,
    _aggregate_by_prompt_mode,
    _build_blind_packet,
    _build_holdout_prompt_set,
    _cohens_dz,
    _cohens_h,
    _compute_state_direction,
    _correlation,
    _exact_sign_pvalue,
    _generate_with_intervention,
    _load_session_turn_records,
    _mean,
    _paired_t_pvalue,
    _resolve_repo_path,
    _sanitize_json,
    _safe_float,
    _select_source_records,
)


def _resolve_generation_seeds(cfg_seed: int, params: dict[str, Any]) -> list[int]:
    explicit = params.get("generation_seeds")
    if isinstance(explicit, list) and explicit:
        return [int(x) for x in explicit]

    n = int(params.get("n_generation_seeds") or 3)
    start = int(params.get("generation_seed_start") or (cfg_seed + 1000))
    stride = int(params.get("generation_seed_stride") or 1)
    return [start + i * stride for i in range(n)]


def _bootstrap_ci(
    values: list[float],
    *,
    resamples: int,
    seed: int,
) -> tuple[Optional[float], Optional[float]]:
    if not values:
        return (None, None)
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 1:
        only = float(arr[0])
        return (only, only)
    rng = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    for idx in range(resamples):
        sample = rng.choice(arr, size=arr.size, replace=True)
        means[idx] = float(np.mean(sample))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return (float(lo), float(hi))


def _group_prompt_condition_means(
    records: list[dict[str, Any]],
    *,
    prompt_mode: Optional[str] = None,
) -> dict[tuple[str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if prompt_mode is not None and record["prompt_mode"] != prompt_mode:
            continue
        grouped[(str(record["prompt_id"]), str(record["condition_name"]))].append(record)

    reduced: dict[tuple[str, str], dict[str, Any]] = {}
    for key, rows in grouped.items():
        rv_vals = [float(r["output_rv"]) for r in rows if r.get("output_rv") is not None]
        bt_vals = [int(r["bt_art"]) for r in rows]
        reduced[key] = {
            "mean_output_rv": _mean(rv_vals),
            "mean_bt_art": _mean([float(x) for x in bt_vals]),
            "n_seed_samples": int(len(rows)),
        }
    return reduced


def _compute_prompt_mode_effects(
    records: list[dict[str, Any]],
    *,
    interventions: list[InterventionSpec],
    control_name: str,
    prompt_mode: Optional[str],
    bootstrap_resamples: int,
    seed: int,
) -> dict[str, Any]:
    reduced = _group_prompt_condition_means(records, prompt_mode=prompt_mode)
    effects: dict[str, Any] = {}

    for intervention in interventions:
        if intervention.name == control_name:
            continue
        rv_control: list[float] = []
        rv_treated: list[float] = []
        rv_diffs: list[float] = []
        bt_control: list[float] = []
        bt_treated: list[float] = []
        bt_diffs: list[float] = []

        prompt_ids = sorted({prompt_id for prompt_id, _ in reduced})
        for prompt_id in prompt_ids:
            control = reduced.get((prompt_id, control_name))
            treated = reduced.get((prompt_id, intervention.name))
            if control is None or treated is None:
                continue
            control_rv = control.get("mean_output_rv")
            treated_rv = treated.get("mean_output_rv")
            if control_rv is not None and treated_rv is not None:
                rv_control.append(float(control_rv))
                rv_treated.append(float(treated_rv))
                rv_diffs.append(float(treated_rv) - float(control_rv))
            control_bt = control.get("mean_bt_art")
            treated_bt = treated.get("mean_bt_art")
            if control_bt is not None and treated_bt is not None:
                bt_control.append(float(control_bt))
                bt_treated.append(float(treated_bt))
                bt_diffs.append(float(treated_bt) - float(control_bt))

        bt_wins = sum(1 for diff in bt_diffs if diff > 0)
        bt_losses = sum(1 for diff in bt_diffs if diff < 0)
        effects[intervention.name] = {
            "alpha": float(intervention.alpha),
            "n_prompt_pairs": int(max(len(bt_diffs), len(rv_diffs))),
            "rv_delta_mean": _mean(rv_diffs),
            "rv_delta_ci_95": _bootstrap_ci(
                rv_diffs, resamples=bootstrap_resamples, seed=seed + 100 + int(intervention.alpha * 10)
            ),
            "rv_cohens_dz": _cohens_dz(rv_diffs),
            "rv_p_value": _paired_t_pvalue(rv_treated, rv_control),
            "bt_art_rate_control": _mean(bt_control),
            "bt_art_rate_treated": _mean(bt_treated),
            "bt_art_rate_delta": _mean(bt_diffs),
            "bt_art_rate_delta_ci_95": _bootstrap_ci(
                bt_diffs, resamples=bootstrap_resamples, seed=seed + 200 + int(intervention.alpha * 10)
            ),
            "bt_art_cohens_h": _cohens_h(_mean(bt_treated), _mean(bt_control)),
            "bt_art_exact_sign_p": _exact_sign_pvalue(bt_wins, bt_losses),
            "bt_art_prompt_wins": int(bt_wins),
            "bt_art_prompt_losses": int(bt_losses),
        }
    return effects


def _dose_response_by_prompt_mode(
    records: list[dict[str, Any]],
    *,
    prompt_mode: Optional[str] = None,
) -> dict[str, Any]:
    reduced = _group_prompt_condition_means(records, prompt_mode=prompt_mode)
    rv_alphas: list[float] = []
    rv_vals: list[float] = []
    bt_alphas: list[float] = []
    bt_vals: list[float] = []
    mean_rv_by_condition: dict[str, list[float]] = defaultdict(list)
    mean_bt_by_condition: dict[str, list[float]] = defaultdict(list)

    for (_prompt_id, condition_name), payload in reduced.items():
        matching = [
            r for r in records
            if r["condition_name"] == condition_name
            and (prompt_mode is None or r["prompt_mode"] == prompt_mode)
        ]
        if not matching:
            continue
        alpha = float(matching[0]["alpha"])
        if payload["mean_output_rv"] is not None:
            rv_alphas.append(alpha)
            rv_vals.append(float(payload["mean_output_rv"]))
            mean_rv_by_condition[condition_name].append(float(payload["mean_output_rv"]))
        if payload["mean_bt_art"] is not None:
            bt_alphas.append(alpha)
            bt_vals.append(float(payload["mean_bt_art"]))
            mean_bt_by_condition[condition_name].append(float(payload["mean_bt_art"]))

    return {
        "alpha_vs_output_rv": _correlation(rv_alphas, rv_vals) if rv_vals else {"r": None, "p": None},
        "alpha_vs_bt_art": _correlation(bt_alphas, bt_vals) if bt_vals else {"r": None, "p": None},
        "mean_output_rv_by_condition": {
            name: _mean(vals) for name, vals in sorted(mean_rv_by_condition.items())
        },
        "bt_art_rate_by_condition": {
            name: _mean(vals) for name, vals in sorted(mean_bt_by_condition.items())
        },
    }


def run_causal_state_benchmark_v2_from_config(
    cfg: Dict[str, Any], run_dir: Path
) -> ExperimentResult:
    from scripts.sustained_gnani_v3 import classify_output, compute_prefill_metrics
    from src.core.models import load_model, set_seed
    from src.metrics.rv import compute_rv

    model_cfg = cfg.get("model") or {}
    params = cfg.get("params") or {}

    seed = int(cfg.get("seed") or 0)
    set_seed(seed)

    device = str(model_cfg.get("device") or ("cuda" if __import__("torch").cuda.is_available() else "cpu"))
    model_name = str(model_cfg.get("name") or "mistralai/Mistral-7B-v0.1")
    early_layer = int(params.get("early_layer") or 5)
    late_layer = int(params.get("late_layer") or 27)
    source_layer = int(params.get("source_layer") or late_layer)
    window = int(params.get("window") or 16)
    max_length = int(params.get("max_length") or 512)
    max_new_tokens = int(params.get("max_new_tokens") or 128)
    do_sample = bool(params.get("do_sample", True))
    temperature = float(params.get("temperature") or 0.7)
    top_p = float(params.get("top_p") or 1.0)
    holdout_per_group = int(params.get("holdout_per_group") or 10)
    min_text_chars = int(params.get("min_text_chars") or 80)
    max_source_per_label = int(params.get("max_source_per_label") or 72)
    max_source_per_session = int(params.get("max_source_per_session") or 8)
    bootstrap_resamples = int(params.get("bootstrap_resamples") or 2000)
    primary_prompt_mode = str(params.get("primary_prompt_mode") or "recursive")
    control_prompt_mode = str(params.get("control_prompt_mode") or "baseline")
    generation_seeds = _resolve_generation_seeds(seed, params)

    recursive_groups = list(
        params.get("recursive_groups") or ["L3_deeper", "L4_full", "L5_refined"]
    )
    baseline_groups = list(
        params.get("baseline_groups") or ["baseline_math", "baseline_factual", "baseline_creative"]
    )
    sessions_dir = _resolve_repo_path(
        params.get("source_sessions_dir") or "results/sustained_gnani_v3_fixed"
    )

    positive_classes = set(params.get("positive_classes") or DEFAULT_POSITIVE_CLASSES)
    negative_classes = set(params.get("negative_classes") or DEFAULT_NEGATIVE_CLASSES)
    positive_quantile = float(params.get("positive_quantile") or 0.35)
    negative_quantile = float(params.get("negative_quantile") or 0.65)
    positive_session_types = params.get("positive_session_types")
    negative_session_types = params.get("negative_session_types")
    positive_session_types_set = set(positive_session_types) if positive_session_types else None
    negative_session_types_set = set(negative_session_types) if negative_session_types else None

    raw_interventions = params.get("interventions") or [
        {"name": "away_alpha_2", "alpha": -2.0},
        {"name": "away_alpha_1", "alpha": -1.0},
        {"name": "none", "alpha": 0.0},
        {"name": "toward_alpha_1", "alpha": 1.0},
        {"name": "toward_alpha_2", "alpha": 2.0},
    ]
    interventions = [
        InterventionSpec(name=str(item["name"]), alpha=float(item["alpha"]))
        for item in raw_interventions
    ]
    control_name = str(params.get("control_name") or "none")

    if not sessions_dir.exists():
        raise FileNotFoundError(f"Source sessions dir not found: {sessions_dir}")

    model, tokenizer = load_model(model_name, device=device, attn_implementation="eager")
    model.eval()

    loader = PromptLoader()
    bank_version = loader.version
    (run_dir / "prompt_bank_version.txt").write_text(bank_version, encoding="utf-8")
    (run_dir / "prompt_bank_version.json").write_text(
        json.dumps({"version": bank_version}, indent=2) + "\n", encoding="utf-8"
    )

    session_records = _load_session_turn_records(sessions_dir, min_text_chars=min_text_chars)
    source_selection = _select_source_records(
        session_records,
        positive_classes=positive_classes,
        negative_classes=negative_classes,
        positive_quantile=positive_quantile,
        negative_quantile=negative_quantile,
        positive_session_types=positive_session_types_set,
        negative_session_types=negative_session_types_set,
        max_source_per_label=max_source_per_label,
        max_source_per_session=max_source_per_session,
        seed=seed,
    )

    direction_payload = _compute_state_direction(
        model,
        tokenizer,
        positive_records=source_selection["positive_records"],
        negative_records=source_selection["negative_records"],
        layer_idx=source_layer,
        window=window,
        device=device,
        max_length=max_length,
    )
    steering_vector = direction_payload["direction"].to(device)

    state_artifact = run_dir / "state_direction.pt"
    __import__("torch").save(
        {
            "direction": direction_payload["direction"],
            "positive_centroid": direction_payload["positive_centroid"],
            "negative_centroid": direction_payload["negative_centroid"],
            "source_layer": source_layer,
            "window": window,
        },
        state_artifact,
    )

    heldout_prompts = _build_holdout_prompt_set(
        loader,
        recursive_groups=recursive_groups,
        baseline_groups=baseline_groups,
        holdout_per_group=holdout_per_group,
        seed=seed,
    )
    if not heldout_prompts:
        raise RuntimeError("Held-out prompt set is empty.")

    records_jsonl = run_dir / "benchmark_records.jsonl"
    all_records: list[dict[str, Any]] = []
    total_jobs = len(heldout_prompts) * len(generation_seeds) * len(interventions)
    job_idx = 0
    with records_jsonl.open("w", encoding="utf-8") as handle:
        for prompt_index, prompt_record in enumerate(heldout_prompts, start=1):
            prompt_text = prompt_record["prompt_text"]
            print(
                f"[prompt {prompt_index}/{len(heldout_prompts)}] "
                f"{prompt_record['prompt_mode']} {prompt_record['prompt_group']} {prompt_record['prompt_id']}",
                flush=True,
            )
            prompt_rv = _safe_float(
                compute_rv(
                    model,
                    tokenizer,
                    prompt_text,
                    early=early_layer,
                    late=late_layer,
                    window=window,
                    device=device,
                )
            )

            for generation_seed in generation_seeds:
                for intervention in interventions:
                    job_idx += 1
                    print(
                        f"  [{job_idx}/{total_jobs}] seed={generation_seed} "
                        f"cond={intervention.name} alpha={intervention.alpha:+.1f}",
                        flush=True,
                    )
                    set_seed(int(generation_seed))
                    generated_text, gen_tokens = _generate_with_intervention(
                        model,
                        tokenizer,
                        prompt=prompt_text,
                        layer_idx=source_layer,
                        steering_vector=steering_vector,
                        alpha=intervention.alpha,
                        max_new_tokens=max_new_tokens,
                        do_sample=do_sample,
                        temperature=temperature,
                        top_p=top_p,
                        device=device,
                        max_length=max_length,
                    )
                    metrics = compute_prefill_metrics(
                        model,
                        tokenizer,
                        generated_text,
                        early=early_layer,
                        late=late_layer,
                        device=device,
                    )
                    output_rv = None
                    if isinstance(metrics, dict):
                        output_rv = _safe_float(metrics.get("rv"))
                    classification = classify_output(generated_text, output_rv)
                    record = {
                        "prompt_id": prompt_record["prompt_id"],
                        "prompt_mode": prompt_record["prompt_mode"],
                        "prompt_group": prompt_record["prompt_group"],
                        "prompt_text": prompt_text,
                        "prompt_rv": prompt_rv,
                        "condition_name": intervention.name,
                        "alpha": float(intervention.alpha),
                        "generation_seed": int(generation_seed),
                        "generated_text": generated_text,
                        "generated_tokens": int(gen_tokens),
                        "classification": classification,
                        "bt_art": int(classification in positive_classes),
                        "output_rv": output_rv,
                        "output_metrics": metrics or {},
                    }
                    all_records.append(record)
                    handle.write(json.dumps(_sanitize_json(record), ensure_ascii=False) + "\n")

    blind_csv = run_dir / "blind_ratings.csv"
    blind_key = run_dir / "blind_key.json"
    _build_blind_packet(all_records, seed=seed, csv_path=blind_csv, key_path=blind_key)

    by_condition: dict[str, Any] = {}
    for intervention in interventions:
        condition_records = [r for r in all_records if r["condition_name"] == intervention.name]
        by_condition[intervention.name] = {
            "alpha": float(intervention.alpha),
            "overall": _aggregate_condition(condition_records),
            "by_prompt_mode": _aggregate_by_prompt_mode(condition_records),
        }

    effects_by_mode = {
        "overall": _compute_prompt_mode_effects(
            all_records,
            interventions=interventions,
            control_name=control_name,
            prompt_mode=None,
            bootstrap_resamples=bootstrap_resamples,
            seed=seed,
        ),
        primary_prompt_mode: _compute_prompt_mode_effects(
            all_records,
            interventions=interventions,
            control_name=control_name,
            prompt_mode=primary_prompt_mode,
            bootstrap_resamples=bootstrap_resamples,
            seed=seed + 10,
        ),
        control_prompt_mode: _compute_prompt_mode_effects(
            all_records,
            interventions=interventions,
            control_name=control_name,
            prompt_mode=control_prompt_mode,
            bootstrap_resamples=bootstrap_resamples,
            seed=seed + 20,
        ),
    }

    dose_response = {
        "overall": _dose_response_by_prompt_mode(all_records, prompt_mode=None),
        primary_prompt_mode: _dose_response_by_prompt_mode(all_records, prompt_mode=primary_prompt_mode),
        control_prompt_mode: _dose_response_by_prompt_mode(all_records, prompt_mode=control_prompt_mode),
    }

    primary_effects = effects_by_mode[primary_prompt_mode]
    strongest_toward = None
    strongest_away = None
    positive_alphas = [iv for iv in interventions if iv.alpha > 0]
    negative_alphas = [iv for iv in interventions if iv.alpha < 0]
    if positive_alphas:
        strongest_toward = max(positive_alphas, key=lambda iv: iv.alpha)
    if negative_alphas:
        strongest_away = min(negative_alphas, key=lambda iv: iv.alpha)

    toward_summary = primary_effects.get(strongest_toward.name) if strongest_toward else None
    away_summary = primary_effects.get(strongest_away.name) if strongest_away else None
    primary_control_summary = (
        by_condition.get(control_name, {}).get("by_prompt_mode", {}).get(primary_prompt_mode, {})
    )
    primary_toward_condition_summary = (
        by_condition.get(strongest_toward.name, {}).get("by_prompt_mode", {}).get(primary_prompt_mode, {})
        if strongest_toward
        else {}
    )

    verdict = "inconclusive"
    if toward_summary and away_summary:
        toward_bt_ci = toward_summary.get("bt_art_rate_delta_ci_95") or (None, None)
        away_bt_ci = away_summary.get("bt_art_rate_delta_ci_95") or (None, None)
        toward_rv_ci = toward_summary.get("rv_delta_ci_95") or (None, None)
        away_rv_ci = away_summary.get("rv_delta_ci_95") or (None, None)
        if (
            toward_bt_ci[0] is not None and toward_bt_ci[0] > 0.0
            and away_bt_ci[1] is not None and away_bt_ci[1] < 0.0
            and toward_rv_ci[1] is not None and toward_rv_ci[1] < 0.0
            and away_rv_ci[0] is not None and away_rv_ci[0] > 0.0
        ):
            verdict = "cluster_bootstrap_supported"
        elif (
            (toward_summary.get("bt_art_rate_delta") or 0.0) > 0.0
            and (away_summary.get("bt_art_rate_delta") or 0.0) < 0.0
            and (toward_summary.get("rv_delta_mean") or 0.0) < 0.0
            and (away_summary.get("rv_delta_mean") or 0.0) > 0.0
        ):
            verdict = "directional_signal_detected"

    heldout_by_mode = Counter(row["prompt_mode"] for row in heldout_prompts)
    heldout_by_group = Counter(row["prompt_group"] for row in heldout_prompts)

    summary = {
        "experiment": "causal_state_benchmark_v2",
        "model_name": model_name,
        "device": device,
        "prompt_bank_version": bank_version,
        "source_sessions_dir": str(sessions_dir),
        "source_layer": int(source_layer),
        "early_layer": int(early_layer),
        "late_layer": int(late_layer),
        "window": int(window),
        "max_new_tokens": int(max_new_tokens),
        "do_sample": bool(do_sample),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "generation_seeds": generation_seeds,
        "n_generation_seeds": int(len(generation_seeds)),
        "bootstrap_resamples": int(bootstrap_resamples),
        "primary_prompt_mode": primary_prompt_mode,
        "control_prompt_mode": control_prompt_mode,
        "n_total": int(len(heldout_prompts)),
        "n_pairs": int(len(heldout_prompts)),
        "heldout_prompt_counts": {
            "total": int(len(heldout_prompts)),
            "by_mode": dict(heldout_by_mode),
            "by_group": dict(heldout_by_group),
        },
        "state_source": {
            "positive_classes": sorted(positive_classes),
            "negative_classes": sorted(negative_classes),
            "positive_threshold_rv": source_selection["positive_threshold"],
            "negative_threshold_rv": source_selection["negative_threshold"],
            "positive_selected_n": int(len(source_selection["positive_records"])),
            "negative_selected_n": int(len(source_selection["negative_records"])),
            "raw_direction_norm": direction_payload["raw_direction_norm"],
            "positive_centroid_norm": direction_payload["positive_centroid_norm"],
            "negative_centroid_norm": direction_payload["negative_centroid_norm"],
            "centroid_cosine": direction_payload["centroid_cosine"],
        },
        "interventions": _sanitize_json([iv.__dict__ for iv in interventions]),
        "by_condition": _sanitize_json(by_condition),
        "effects_by_prompt_mode": _sanitize_json(effects_by_mode),
        "dose_response": _sanitize_json(dose_response),
        "rv_recursive_mean": primary_toward_condition_summary.get("mean_output_rv"),
        "rv_baseline_mean": primary_control_summary.get("mean_output_rv"),
        "rv_delta_mean": toward_summary.get("rv_delta_mean") if toward_summary else None,
        "rv_cohens_d": toward_summary.get("rv_cohens_dz") if toward_summary else None,
        "rv_p_value": toward_summary.get("rv_p_value") if toward_summary else None,
        "logit_diff_delta_mean": None,
        "logit_diff_cohens_d": None,
        "logit_diff_p_value": None,
        "verdict": verdict,
        "artifacts": {
            "records_jsonl": str(records_jsonl),
            "blind_ratings_csv": str(blind_csv),
            "blind_key_json": str(blind_key),
            "state_direction_pt": str(state_artifact),
        },
    }

    baseline_metrics = {
        "rv": by_condition.get(control_name, {}).get("overall", {}).get("mean_output_rv"),
        "logit_diff": None,
    }
    return ExperimentResult(summary=_sanitize_json(summary), baseline_metrics=baseline_metrics)
