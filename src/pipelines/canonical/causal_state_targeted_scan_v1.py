"""
Targeted intervention scan for the causal state benchmark.

This is a search-stage pipeline that reuses the low-R_V state source construction
from the benchmark pipelines, but scans candidate layer/window/alpha settings to
find a sharper intervention before spending more budget on a larger confirmatory
run.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from prompts.loader import PromptLoader
from src.pipelines.registry import ExperimentResult

from .causal_state_benchmark_v1 import (
    DEFAULT_NEGATIVE_CLASSES,
    DEFAULT_POSITIVE_CLASSES,
    InterventionSpec,
    _aggregate_by_prompt_mode,
    _aggregate_condition,
    _build_holdout_prompt_set,
    _compute_state_direction,
    _generate_with_intervention,
    _load_session_turn_records,
    _resolve_repo_path,
    _safe_float,
    _sanitize_json,
    _select_source_records,
)
from .causal_state_benchmark_v2 import _compute_prompt_mode_effects, _resolve_generation_seeds


@dataclass(frozen=True)
class ScanCandidate:
    name: str
    source_layer: int
    window: int
    alpha: float


def _format_alpha(alpha: float) -> str:
    if abs(alpha - round(alpha)) < 1e-9:
        return str(int(round(alpha)))
    return str(alpha).replace(".", "p")


def _resolve_scan_candidates(params: dict[str, Any]) -> list[ScanCandidate]:
    layers = [int(x) for x in (params.get("candidate_source_layers") or [25, 27, 29, 31])]
    windows = [int(x) for x in (params.get("candidate_windows") or [8, 16, 32])]
    alphas = [float(x) for x in (params.get("candidate_alpha_magnitudes") or [2.0, 3.0])]

    candidates: list[ScanCandidate] = []
    for layer in layers:
        for window in windows:
            for alpha in alphas:
                candidates.append(
                    ScanCandidate(
                        name=f"L{layer}_W{window}_A{_format_alpha(alpha)}",
                        source_layer=layer,
                        window=window,
                        alpha=alpha,
                    )
                )
    return candidates


def _build_scan_prompt_set(
    loader: PromptLoader,
    *,
    recursive_groups: list[str],
    baseline_groups: list[str],
    recursive_per_group: int,
    baseline_per_group: int,
    seed: int,
) -> list[dict[str, str]]:
    recursive_records = _build_holdout_prompt_set(
        loader,
        recursive_groups=recursive_groups,
        baseline_groups=[],
        holdout_per_group=recursive_per_group,
        seed=seed,
    )
    baseline_records = _build_holdout_prompt_set(
        loader,
        recursive_groups=[],
        baseline_groups=baseline_groups,
        holdout_per_group=baseline_per_group,
        seed=seed + 1,
    )
    combined = recursive_records + baseline_records
    combined.sort(key=lambda row: (row["prompt_mode"], row["prompt_group"], row["prompt_id"]))
    return combined


def _signed_effect(effect: Optional[dict[str, Any]], key: str) -> float:
    if not effect:
        return 0.0
    value = effect.get(key)
    if value is None:
        return 0.0
    try:
        return float(value)
    except Exception:
        return 0.0


def _candidate_objective(
    recursive_effects: dict[str, Any],
    baseline_effects: dict[str, Any],
) -> dict[str, Any]:
    rec_toward = recursive_effects.get("toward")
    rec_away = recursive_effects.get("away")
    base_toward = baseline_effects.get("toward")
    base_away = baseline_effects.get("away")

    rec_toward_bt = _signed_effect(rec_toward, "bt_art_rate_delta")
    rec_away_bt = _signed_effect(rec_away, "bt_art_rate_delta")
    rec_toward_rv = _signed_effect(rec_toward, "rv_delta_mean")
    rec_away_rv = _signed_effect(rec_away, "rv_delta_mean")
    base_toward_bt = _signed_effect(base_toward, "bt_art_rate_delta")
    base_away_bt = _signed_effect(base_away, "bt_art_rate_delta")
    base_toward_rv = _signed_effect(base_toward, "rv_delta_mean")

    recursive_bt_gain = max(rec_toward_bt, 0.0)
    recursive_bt_suppression = max(-rec_away_bt, 0.0)
    recursive_rv_alignment = max(-rec_toward_rv, 0.0) + 0.5 * max(rec_away_rv, 0.0)
    baseline_spillover = abs(base_toward_bt) + 0.5 * abs(base_away_bt)
    baseline_rv_spillover = abs(base_toward_rv)

    score = (
        1.00 * recursive_bt_gain
        + 0.60 * recursive_bt_suppression
        + 0.15 * recursive_rv_alignment
        - 0.75 * baseline_spillover
        - 0.10 * baseline_rv_spillover
    )

    sign_checks = {
        "recursive_toward_bt_positive": rec_toward_bt > 0.0,
        "recursive_away_bt_negative": rec_away_bt < 0.0,
        "recursive_toward_rv_negative": rec_toward_rv < 0.0,
        "recursive_away_rv_positive": rec_away_rv > 0.0,
    }
    sign_match_count = int(sum(1 for matched in sign_checks.values() if matched))

    return {
        "score": float(score),
        "score_breakdown": {
            "recursive_bt_gain": float(recursive_bt_gain),
            "recursive_bt_suppression": float(recursive_bt_suppression),
            "recursive_rv_alignment": float(recursive_rv_alignment),
            "baseline_spillover_penalty": float(baseline_spillover),
            "baseline_rv_penalty": float(baseline_rv_spillover),
        },
        "sign_checks": sign_checks,
        "sign_match_count": sign_match_count,
    }


def _summarize_candidate(
    candidate: ScanCandidate,
    *,
    shared_controls: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
    bootstrap_resamples: int,
    seed: int,
) -> dict[str, Any]:
    combined = list(shared_controls) + list(candidate_records)
    interventions = [
        InterventionSpec(name="away", alpha=-float(candidate.alpha)),
        InterventionSpec(name="none", alpha=0.0),
        InterventionSpec(name="toward", alpha=float(candidate.alpha)),
    ]

    by_condition: dict[str, Any] = {}
    for intervention in interventions:
        rows = [record for record in combined if record["condition_name"] == intervention.name]
        by_condition[intervention.name] = {
            "alpha": float(intervention.alpha),
            "overall": _aggregate_condition(rows),
            "by_prompt_mode": _aggregate_by_prompt_mode(rows),
        }

    effects_by_mode = {
        "overall": _compute_prompt_mode_effects(
            combined,
            interventions=interventions,
            control_name="none",
            prompt_mode=None,
            bootstrap_resamples=bootstrap_resamples,
            seed=seed,
        ),
        "recursive": _compute_prompt_mode_effects(
            combined,
            interventions=interventions,
            control_name="none",
            prompt_mode="recursive",
            bootstrap_resamples=bootstrap_resamples,
            seed=seed + 10,
        ),
        "baseline": _compute_prompt_mode_effects(
            combined,
            interventions=interventions,
            control_name="none",
            prompt_mode="baseline",
            bootstrap_resamples=bootstrap_resamples,
            seed=seed + 20,
        ),
    }

    objective = _candidate_objective(
        recursive_effects=effects_by_mode["recursive"],
        baseline_effects=effects_by_mode["baseline"],
    )

    return {
        "candidate_name": candidate.name,
        "source_layer": int(candidate.source_layer),
        "window": int(candidate.window),
        "alpha": float(candidate.alpha),
        "objective": objective,
        "by_condition": by_condition,
        "effects_by_prompt_mode": effects_by_mode,
    }


def run_causal_state_targeted_scan_v1_from_config(
    cfg: Dict[str, Any], run_dir: Path
) -> ExperimentResult:
    from scripts.sustained_gnani_v3 import classify_output, compute_prefill_metrics
    from src.core.models import load_model, set_seed
    from src.metrics.rv import compute_rv

    model_cfg = cfg.get("model") or {}
    params = cfg.get("params") or {}

    seed = int(cfg.get("seed") or 0)
    set_seed(seed)

    device = str(model_cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_name = str(model_cfg.get("name") or "mistralai/Mistral-7B-v0.1")
    early_layer = int(params.get("early_layer") or 5)
    late_layer = int(params.get("late_layer") or 27)
    metric_window = int(params.get("metric_window") or 16)
    max_length = int(params.get("max_length") or 512)
    max_new_tokens = int(params.get("max_new_tokens") or 128)
    do_sample = bool(params.get("do_sample", True))
    temperature = float(params.get("temperature") or 0.7)
    top_p = float(params.get("top_p") or 1.0)
    min_text_chars = int(params.get("min_text_chars") or 80)
    max_source_per_label = int(params.get("max_source_per_label") or 72)
    max_source_per_session = int(params.get("max_source_per_session") or 8)
    bootstrap_resamples = int(params.get("bootstrap_resamples") or 1000)
    recursive_per_group = int(params.get("search_recursive_per_group") or 4)
    baseline_per_group = int(params.get("search_baseline_per_group") or 4)
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
    control_layer = int(params.get("control_layer") or late_layer)

    positive_classes = set(params.get("positive_classes") or DEFAULT_POSITIVE_CLASSES)
    negative_classes = set(params.get("negative_classes") or DEFAULT_NEGATIVE_CLASSES)
    positive_quantile = float(params.get("positive_quantile") or 0.35)
    negative_quantile = float(params.get("negative_quantile") or 0.65)
    positive_session_types = params.get("positive_session_types")
    negative_session_types = params.get("negative_session_types")
    positive_session_types_set = set(positive_session_types) if positive_session_types else None
    negative_session_types_set = set(negative_session_types) if negative_session_types else None

    candidates = _resolve_scan_candidates(params)
    if not candidates:
        raise RuntimeError("Targeted scan candidate grid is empty.")
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

    heldout_prompts = _build_scan_prompt_set(
        loader,
        recursive_groups=recursive_groups,
        baseline_groups=baseline_groups,
        recursive_per_group=recursive_per_group,
        baseline_per_group=baseline_per_group,
        seed=seed,
    )
    if not heldout_prompts:
        raise RuntimeError("Targeted scan held-out prompt set is empty.")

    prompt_rv_cache: dict[str, Optional[float]] = {}
    prompt_group_counts = Counter(row["prompt_group"] for row in heldout_prompts)
    prompt_mode_counts = Counter(row["prompt_mode"] for row in heldout_prompts)

    shared_records_jsonl = run_dir / "shared_control_records.jsonl"
    candidate_records_jsonl = run_dir / "candidate_records.jsonl"
    shared_controls: list[dict[str, Any]] = []

    control_jobs = len(heldout_prompts) * len(generation_seeds)
    candidate_jobs = len(candidates) * len(heldout_prompts) * len(generation_seeds) * 2
    total_jobs = control_jobs + candidate_jobs
    job_idx = 0

    with shared_records_jsonl.open("w", encoding="utf-8") as control_handle:
        for prompt_index, prompt_record in enumerate(heldout_prompts, start=1):
            prompt_text = prompt_record["prompt_text"]
            prompt_id = str(prompt_record["prompt_id"])
            prompt_rv = _safe_float(
                compute_rv(
                    model,
                    tokenizer,
                    prompt_text,
                    early=early_layer,
                    late=late_layer,
                    window=metric_window,
                    device=device,
                )
            )
            prompt_rv_cache[prompt_id] = prompt_rv

            print(
                f"[control prompt {prompt_index}/{len(heldout_prompts)}] "
                f"{prompt_record['prompt_mode']} {prompt_record['prompt_group']} {prompt_record['prompt_id']}",
                flush=True,
            )
            for generation_seed in generation_seeds:
                job_idx += 1
                print(
                    f"  [{job_idx}/{total_jobs}] seed={generation_seed} cond=none alpha=+0.0",
                    flush=True,
                )
                set_seed(int(generation_seed))
                generated_text, gen_tokens = _generate_with_intervention(
                    model,
                    tokenizer,
                    prompt=prompt_text,
                    layer_idx=control_layer,
                    steering_vector=torch.zeros(model.config.hidden_size, device=device),
                    alpha=0.0,
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
                output_rv = _safe_float(metrics.get("rv")) if isinstance(metrics, dict) else None
                classification = classify_output(generated_text, output_rv)
                record = {
                    "candidate_name": "shared_control",
                    "prompt_id": prompt_id,
                    "prompt_mode": prompt_record["prompt_mode"],
                    "prompt_group": prompt_record["prompt_group"],
                    "prompt_text": prompt_text,
                    "prompt_rv": prompt_rv,
                    "condition_name": "none",
                    "alpha": 0.0,
                    "generation_seed": int(generation_seed),
                    "generated_text": generated_text,
                    "generated_tokens": int(gen_tokens),
                    "classification": classification,
                    "bt_art": int(classification in positive_classes),
                    "output_rv": output_rv,
                    "output_metrics": metrics or {},
                }
                shared_controls.append(record)
                control_handle.write(json.dumps(_sanitize_json(record), ensure_ascii=False) + "\n")

    candidate_summaries: list[dict[str, Any]] = []
    with candidate_records_jsonl.open("w", encoding="utf-8") as candidate_handle:
        for candidate_index, candidate in enumerate(candidates, start=1):
            print(
                f"[candidate {candidate_index}/{len(candidates)}] "
                f"{candidate.name} layer={candidate.source_layer} window={candidate.window} alpha={candidate.alpha:+.2f}",
                flush=True,
            )
            direction_payload = _compute_state_direction(
                model,
                tokenizer,
                positive_records=source_selection["positive_records"],
                negative_records=source_selection["negative_records"],
                layer_idx=candidate.source_layer,
                window=candidate.window,
                device=device,
                max_length=max_length,
            )
            steering_vector = direction_payload["direction"].to(device)
            candidate_rows: list[dict[str, Any]] = []

            for prompt_record in heldout_prompts:
                prompt_text = prompt_record["prompt_text"]
                prompt_id = str(prompt_record["prompt_id"])
                prompt_rv = prompt_rv_cache[prompt_id]
                for generation_seed in generation_seeds:
                    for condition_name, alpha in (
                        ("away", -float(candidate.alpha)),
                        ("toward", float(candidate.alpha)),
                    ):
                        job_idx += 1
                        print(
                            f"  [{job_idx}/{total_jobs}] seed={generation_seed} "
                            f"cond={condition_name} candidate={candidate.name} alpha={alpha:+.1f}",
                            flush=True,
                        )
                        set_seed(int(generation_seed))
                        generated_text, gen_tokens = _generate_with_intervention(
                            model,
                            tokenizer,
                            prompt=prompt_text,
                            layer_idx=candidate.source_layer,
                            steering_vector=steering_vector,
                            alpha=alpha,
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
                        output_rv = _safe_float(metrics.get("rv")) if isinstance(metrics, dict) else None
                        classification = classify_output(generated_text, output_rv)
                        record = {
                            "candidate_name": candidate.name,
                            "source_layer": int(candidate.source_layer),
                            "window": int(candidate.window),
                            "prompt_id": prompt_id,
                            "prompt_mode": prompt_record["prompt_mode"],
                            "prompt_group": prompt_record["prompt_group"],
                            "prompt_text": prompt_text,
                            "prompt_rv": prompt_rv,
                            "condition_name": condition_name,
                            "alpha": float(alpha),
                            "generation_seed": int(generation_seed),
                            "generated_text": generated_text,
                            "generated_tokens": int(gen_tokens),
                            "classification": classification,
                            "bt_art": int(classification in positive_classes),
                            "output_rv": output_rv,
                            "output_metrics": metrics or {},
                        }
                        candidate_rows.append(record)
                        candidate_handle.write(json.dumps(_sanitize_json(record), ensure_ascii=False) + "\n")

            candidate_summary = _summarize_candidate(
                candidate,
                shared_controls=shared_controls,
                candidate_records=candidate_rows,
                bootstrap_resamples=bootstrap_resamples,
                seed=seed + (candidate_index * 100),
            )
            candidate_summary["state_source"] = {
                "raw_direction_norm": direction_payload["raw_direction_norm"],
                "positive_centroid_norm": direction_payload["positive_centroid_norm"],
                "negative_centroid_norm": direction_payload["negative_centroid_norm"],
                "centroid_cosine": direction_payload["centroid_cosine"],
                "positive_selected_n": int(direction_payload["positive_n"]),
                "negative_selected_n": int(direction_payload["negative_n"]),
            }
            candidate_summaries.append(candidate_summary)

    ranked_candidates = sorted(
        candidate_summaries,
        key=lambda row: (
            -float(row["objective"]["score"]),
            -int(row["objective"]["sign_match_count"]),
            float(row["alpha"]),
        ),
    )
    for rank, candidate in enumerate(ranked_candidates, start=1):
        candidate["rank"] = rank

    best_candidate = ranked_candidates[0]
    candidate_scores_path = run_dir / "candidate_scores.json"
    best_candidate_path = run_dir / "best_candidate.json"
    candidate_scores_path.write_text(
        json.dumps(_sanitize_json(ranked_candidates), indent=2) + "\n", encoding="utf-8"
    )
    best_candidate_path.write_text(
        json.dumps(_sanitize_json(best_candidate), indent=2) + "\n", encoding="utf-8"
    )

    summary = {
        "experiment": "causal_state_targeted_scan_v1",
        "model_name": model_name,
        "device": device,
        "prompt_bank_version": bank_version,
        "source_sessions_dir": str(sessions_dir),
        "search_space": {
            "candidate_count": int(len(candidates)),
            "candidate_source_layers": sorted({candidate.source_layer for candidate in candidates}),
            "candidate_windows": sorted({candidate.window for candidate in candidates}),
            "candidate_alpha_magnitudes": sorted({float(candidate.alpha) for candidate in candidates}),
        },
        "generation_seeds": generation_seeds,
        "n_generation_seeds": int(len(generation_seeds)),
        "heldout_prompt_counts": {
            "total": int(len(heldout_prompts)),
            "by_mode": dict(prompt_mode_counts),
            "by_group": dict(prompt_group_counts),
        },
        "bootstrap_resamples": int(bootstrap_resamples),
        "state_source": {
            "positive_classes": sorted(positive_classes),
            "negative_classes": sorted(negative_classes),
            "positive_threshold_rv": source_selection["positive_threshold"],
            "negative_threshold_rv": source_selection["negative_threshold"],
            "positive_selected_n": int(len(source_selection["positive_records"])),
            "negative_selected_n": int(len(source_selection["negative_records"])),
        },
        "best_candidate": best_candidate,
        "candidate_rankings": ranked_candidates,
        "promotion_recommendation": {
            "source_layer": int(best_candidate["source_layer"]),
            "window": int(best_candidate["window"]),
            "alpha": float(best_candidate["alpha"]),
            "interventions": [
                {"name": "away_alpha_best", "alpha": -float(best_candidate["alpha"])},
                {"name": "none", "alpha": 0.0},
                {"name": "toward_alpha_best", "alpha": float(best_candidate["alpha"])},
            ],
        },
        "artifacts": {
            "shared_control_records_jsonl": str(shared_records_jsonl),
            "candidate_records_jsonl": str(candidate_records_jsonl),
            "candidate_scores_json": str(candidate_scores_path),
            "best_candidate_json": str(best_candidate_path),
        },
        "verdict": (
            "ready_for_confirmatory_v3"
            if best_candidate["objective"]["sign_match_count"] >= 3
            and float(best_candidate["objective"]["score"]) > 0.0
            else "search_completed"
        ),
    }

    baseline_metrics = {
        "rv": _safe_float(
            best_candidate["by_condition"]["none"]["overall"].get("mean_output_rv")
        ),
        "logit_diff": None,
    }
    return ExperimentResult(summary=_sanitize_json(summary), baseline_metrics=baseline_metrics)
