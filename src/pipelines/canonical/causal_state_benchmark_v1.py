"""
Causal state benchmark v1.

Learn a residual-stream direction from completed sustained-session outputs, then
test whether steering held-out prompts toward or away from that direction changes
behavior and output-side R_V.
"""

from __future__ import annotations

import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prompts.loader import PromptLoader
from src.pipelines.registry import ExperimentResult
from src.core.hf_accessors import get_layers

try:
    from scipy import stats as scipy_stats  # type: ignore
except Exception:  # pragma: no cover - exercised in environments without scipy
    scipy_stats = None


DEFAULT_POSITIVE_CLASSES = ("BREAKTHROUGH", "ARTICULATE")
DEFAULT_NEGATIVE_CLASSES = ("SURFACE", "REPETITIVE")


@dataclass(frozen=True)
class InterventionSpec:
    name: str
    alpha: float


def _sanitize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _sanitize_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_json(v) for v in value]
    if isinstance(value, tuple):
        return [_sanitize_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        val = float(value)
        if not math.isfinite(val):
            return None
        return val
    return value


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def _mean(values: Iterable[float]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return None
    return float(np.mean(vals))


def _std(values: Iterable[float]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if len(vals) < 2:
        return None
    return float(np.std(vals, ddof=1))


def _cohens_d(a: list[float], b: list[float]) -> Optional[float]:
    if len(a) < 2 or len(b) < 2:
        return None
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    pooled = math.sqrt(
        (((len(a_arr) - 1) * np.var(a_arr, ddof=1)) + ((len(b_arr) - 1) * np.var(b_arr, ddof=1)))
        / max(len(a_arr) + len(b_arr) - 2, 1)
    )
    if pooled < 1e-12:
        return 0.0
    return float((np.mean(a_arr) - np.mean(b_arr)) / pooled)


def _cohens_dz(diffs: list[float]) -> Optional[float]:
    if len(diffs) < 2:
        return None
    diff_arr = np.asarray(diffs, dtype=np.float64)
    denom = float(np.std(diff_arr, ddof=1))
    if denom < 1e-12:
        return 0.0
    return float(np.mean(diff_arr) / denom)


def _cohens_h(rate_a: Optional[float], rate_b: Optional[float]) -> Optional[float]:
    if rate_a is None or rate_b is None:
        return None
    a = min(max(rate_a, 1e-9), 1.0 - 1e-9)
    b = min(max(rate_b, 1e-9), 1.0 - 1e-9)
    return float(2.0 * math.asin(math.sqrt(a)) - 2.0 * math.asin(math.sqrt(b)))


def _paired_t_pvalue(a: list[float], b: list[float]) -> Optional[float]:
    if scipy_stats is None or len(a) < 2 or len(a) != len(b):
        return None
    res = scipy_stats.ttest_rel(np.asarray(a), np.asarray(b), nan_policy="omit")
    return _safe_float(res.pvalue)


def _exact_sign_pvalue(n_positive: int, n_negative: int) -> Optional[float]:
    n_total = int(n_positive + n_negative)
    if scipy_stats is None or n_total == 0:
        return None
    smaller = min(int(n_positive), int(n_negative))
    return _safe_float(scipy_stats.binomtest(smaller, n_total, p=0.5, alternative="two-sided").pvalue)


def _correlation(xs: list[float], ys: list[float]) -> dict[str, Optional[float]]:
    if len(xs) < 3 or len(xs) != len(ys):
        return {"r": None, "p": None}
    x_arr = np.asarray(xs, dtype=np.float64)
    y_arr = np.asarray(ys, dtype=np.float64)
    if np.std(x_arr) < 1e-12 or np.std(y_arr) < 1e-12:
        return {"r": None, "p": None}
    if scipy_stats is None:
        return {"r": _safe_float(np.corrcoef(x_arr, y_arr)[0, 1]), "p": None}
    corr = scipy_stats.pearsonr(x_arr, y_arr)
    return {"r": _safe_float(corr.statistic), "p": _safe_float(corr.pvalue)}


def _resolve_repo_path(path_str: str | Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _infer_session_type(file_path: Path, payload: dict[str, Any]) -> str:
    mode = payload.get("mode")
    if isinstance(mode, str) and mode.strip():
        return mode
    stem = file_path.stem
    if stem.startswith("recursive_"):
        return "recursive"
    if stem.startswith("baseline_"):
        return "baseline"
    return "unknown"


def _load_session_turn_records(
    sessions_dir: Path,
    *,
    min_text_chars: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(sessions_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        session_id = str(payload.get("session_id") or path.stem)
        session_type = _infer_session_type(path, payload)
        for turn in payload.get("turns", []):
            response = str(turn.get("response") or "").strip()
            output_rv = _safe_float(turn.get("output_rv"))
            if output_rv is None and isinstance(turn.get("output_metrics"), dict):
                output_rv = _safe_float(turn["output_metrics"].get("rv"))
            classification = str(turn.get("classification") or "UNKNOWN")
            if len(response) < min_text_chars or output_rv is None:
                continue
            records.append(
                {
                    "session_id": session_id,
                    "session_type": session_type,
                    "turn": int(turn.get("turn", -1)),
                    "classification": classification,
                    "output_rv": float(output_rv),
                    "response": response,
                    "path": str(path),
                }
            )
    return records


def _quantile_select(
    records: list[dict[str, Any]],
    *,
    quantile: float,
    side: str,
) -> tuple[list[dict[str, Any]], Optional[float]]:
    if not records:
        return [], None
    values = np.asarray([float(r["output_rv"]) for r in records], dtype=np.float64)
    threshold = float(np.quantile(values, quantile))
    if side == "low":
        chosen = [r for r in records if float(r["output_rv"]) <= threshold]
    elif side == "high":
        chosen = [r for r in records if float(r["output_rv"]) >= threshold]
    else:
        raise ValueError(f"Unknown side: {side}")
    if not chosen:
        chosen = list(records)
    return chosen, threshold


def _round_robin_sample(
    records: list[dict[str, Any]],
    *,
    max_total: int,
    max_per_session: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["session_id"])].append(record)
    for items in grouped.values():
        rng.shuffle(items)

    session_ids = list(grouped.keys())
    rng.shuffle(session_ids)
    picked: list[dict[str, Any]] = []
    per_session_counts = Counter()

    while len(picked) < max_total and session_ids:
        next_session_ids: list[str] = []
        progress = False
        for session_id in session_ids:
            if len(picked) >= max_total:
                break
            if per_session_counts[session_id] >= max_per_session:
                continue
            items = grouped[session_id]
            if not items:
                continue
            picked.append(items.pop())
            per_session_counts[session_id] += 1
            progress = True
            if items and per_session_counts[session_id] < max_per_session:
                next_session_ids.append(session_id)
        if not progress:
            break
        session_ids = next_session_ids
    return picked


def _select_source_records(
    records: list[dict[str, Any]],
    *,
    positive_classes: set[str],
    negative_classes: set[str],
    positive_quantile: float,
    negative_quantile: float,
    positive_session_types: Optional[set[str]],
    negative_session_types: Optional[set[str]],
    max_source_per_label: int,
    max_source_per_session: int,
    seed: int,
) -> dict[str, Any]:
    positives = [
        r
        for r in records
        if r["classification"] in positive_classes
        and (not positive_session_types or r["session_type"] in positive_session_types)
    ]
    negatives = [
        r
        for r in records
        if r["classification"] in negative_classes
        and (not negative_session_types or r["session_type"] in negative_session_types)
    ]

    positive_pool, positive_threshold = _quantile_select(
        positives, quantile=positive_quantile, side="low"
    )
    negative_pool, negative_threshold = _quantile_select(
        negatives, quantile=negative_quantile, side="high"
    )

    positive_sample = _round_robin_sample(
        positive_pool,
        max_total=max_source_per_label,
        max_per_session=max_source_per_session,
        seed=seed + 101,
    )
    negative_sample = _round_robin_sample(
        negative_pool,
        max_total=max_source_per_label,
        max_per_session=max_source_per_session,
        seed=seed + 202,
    )

    if not positive_sample or not negative_sample:
        raise RuntimeError(
            "Could not build state source pools. "
            f"positive={len(positive_sample)} negative={len(negative_sample)}"
        )

    return {
        "positive_records": positive_sample,
        "negative_records": negative_sample,
        "positive_threshold": positive_threshold,
        "negative_threshold": negative_threshold,
        "positive_pool_n": len(positive_pool),
        "negative_pool_n": len(negative_pool),
        "positive_candidates_n": len(positives),
        "negative_candidates_n": len(negatives),
    }


def _capture_resid_window_mean(
    model,
    tokenizer,
    text: str,
    *,
    layer_idx: int,
    window: int,
    device: str,
    max_length: int,
) -> Optional[torch.Tensor]:
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    storage: dict[str, Optional[torch.Tensor]] = {"resid": None}

    def hook_fn(_module, inputs):
        storage["resid"] = inputs[0].detach()[0]
        return None

    handle = get_layers(model)[layer_idx].register_forward_pre_hook(hook_fn)
    try:
        with torch.no_grad():
            model(**enc)
    finally:
        handle.remove()

    resid = storage["resid"]
    if resid is None or resid.shape[0] < window:
        return None
    return resid[-window:, :].float().mean(dim=0)


def _capture_mlp_window_mean(
    model,
    tokenizer,
    text: str,
    *,
    layer_idx: int,
    window: int,
    device: str,
    max_length: int,
) -> Optional[torch.Tensor]:
    enc = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=max_length
    ).to(device)
    storage: dict[str, Optional[torch.Tensor]] = {"mlp": None}

    def hook_fn(_module, _inputs, output):
        storage["mlp"] = (output[0] if isinstance(output, tuple) else output).detach()[0]
        return output

    handle = get_layers(model)[layer_idx].mlp.register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            model(**enc)
    finally:
        handle.remove()

    mlp = storage["mlp"]
    if mlp is None or mlp.shape[0] < window:
        return None
    return mlp[-window:, :].float().mean(dim=0)


def _capture_component_window_mean(
    model,
    tokenizer,
    text: str,
    *,
    layer_idx: int,
    window: int,
    device: str,
    max_length: int,
    component: str = "residual",
) -> Optional[torch.Tensor]:
    component_name = str(component).lower()
    if component_name in {"residual", "resid"}:
        return _capture_resid_window_mean(
            model,
            tokenizer,
            text,
            layer_idx=layer_idx,
            window=window,
            device=device,
            max_length=max_length,
        )
    if component_name == "mlp":
        return _capture_mlp_window_mean(
            model,
            tokenizer,
            text,
            layer_idx=layer_idx,
            window=window,
            device=device,
            max_length=max_length,
        )
    raise ValueError(f"Unsupported steering component: {component}")


def _compute_state_direction(
    model,
    tokenizer,
    *,
    positive_records: list[dict[str, Any]],
    negative_records: list[dict[str, Any]],
    layer_idx: int,
    window: int,
    device: str,
    max_length: int,
    component: str = "residual",
) -> dict[str, Any]:
    positive_vectors: list[torch.Tensor] = []
    negative_vectors: list[torch.Tensor] = []

    for record in positive_records:
        vec = _capture_component_window_mean(
            model,
            tokenizer,
            record["response"],
            layer_idx=layer_idx,
            window=window,
            device=device,
            max_length=max_length,
            component=component,
        )
        if vec is not None:
            positive_vectors.append(vec.cpu())

    for record in negative_records:
        vec = _capture_component_window_mean(
            model,
            tokenizer,
            record["response"],
            layer_idx=layer_idx,
            window=window,
            device=device,
            max_length=max_length,
            component=component,
        )
        if vec is not None:
            negative_vectors.append(vec.cpu())

    if not positive_vectors or not negative_vectors:
        raise RuntimeError(
            "Could not extract enough activation vectors to build a steering direction."
        )

    positive_centroid = torch.stack(positive_vectors, dim=0).mean(dim=0)
    negative_centroid = torch.stack(negative_vectors, dim=0).mean(dim=0)
    raw_direction = positive_centroid - negative_centroid
    raw_norm = float(raw_direction.norm().item())
    if raw_norm < 1e-12:
        raise RuntimeError("State direction collapsed to zero norm.")
    direction = raw_direction / raw_norm

    cosine = float(
        torch.nn.functional.cosine_similarity(
            positive_centroid.unsqueeze(0), negative_centroid.unsqueeze(0), dim=-1
        ).item()
    )
    return {
        "direction": direction,
        "positive_centroid": positive_centroid,
        "negative_centroid": negative_centroid,
        "raw_direction_norm": raw_norm,
        "positive_centroid_norm": float(positive_centroid.norm().item()),
        "negative_centroid_norm": float(negative_centroid.norm().item()),
        "centroid_cosine": cosine,
        "positive_n": len(positive_vectors),
        "negative_n": len(negative_vectors),
    }


def _collect_group_prompt_records(
    loader: PromptLoader,
    *,
    groups: list[str],
    prompt_mode: str,
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for prompt_id, payload in loader.prompts.items():
        group = payload.get("group")
        text = payload.get("text")
        if group in groups and isinstance(text, str):
            records.append(
                {
                    "prompt_id": str(prompt_id),
                    "prompt_group": str(group),
                    "prompt_mode": prompt_mode,
                    "prompt_text": text,
                }
            )
    return records


def _build_holdout_prompt_set(
    loader: PromptLoader,
    *,
    recursive_groups: list[str],
    baseline_groups: list[str],
    holdout_per_group: int,
    seed: int,
) -> list[dict[str, str]]:
    rng = random.Random(seed)
    all_records = _collect_group_prompt_records(
        loader, groups=recursive_groups, prompt_mode="recursive"
    ) + _collect_group_prompt_records(loader, groups=baseline_groups, prompt_mode="baseline")

    by_group: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in all_records:
        by_group[record["prompt_group"]].append(record)

    holdout: list[dict[str, str]] = []
    for group, records in sorted(by_group.items()):
        rng.shuffle(records)
        take = min(holdout_per_group, len(records))
        holdout.extend(records[:take])

    holdout.sort(key=lambda row: (row["prompt_mode"], row["prompt_group"], row["prompt_id"]))
    return holdout


def _generate_with_intervention(
    model,
    tokenizer,
    *,
    prompt: str,
    layer_idx: int,
    steering_vector: torch.Tensor,
    alpha: float,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    device: str,
    max_length: int,
) -> tuple[str, int]:
    from src.steering.activation_patching import apply_steering_vector

    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": bool(do_sample),
        "pad_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }
    if do_sample:
        generate_kwargs["temperature"] = float(temperature)
        if top_p < 1.0:
            generate_kwargs["top_p"] = float(top_p)

    ctx = (
        apply_steering_vector(model, layer_idx=layer_idx, vector=steering_vector, alpha=alpha)
        if abs(alpha) > 1e-12
        else nullcontext()
    )
    with torch.no_grad():
        with ctx:
            outputs = model.generate(**enc, **generate_kwargs)
    prompt_len = int(enc["input_ids"].shape[1])
    gen_ids = outputs[0, prompt_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return text, int(gen_ids.shape[0])


def _class_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(r["classification"]) for r in records)
    return dict(sorted(counts.items()))


def _aggregate_condition(records: list[dict[str, Any]]) -> dict[str, Any]:
    rv_values = [float(r["output_rv"]) for r in records if r.get("output_rv") is not None]
    bt_rate = _mean([float(r["bt_art"]) for r in records])
    token_mean = _mean([float(r["generated_tokens"]) for r in records])
    return {
        "n": int(len(records)),
        "mean_output_rv": _mean(rv_values),
        "std_output_rv": _std(rv_values),
        "bt_art_rate": bt_rate,
        "mean_generated_tokens": token_mean,
        "class_counts": _class_counts(records),
    }


def _aggregate_by_prompt_mode(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["prompt_mode"])].append(record)
    return {mode: _aggregate_condition(rows) for mode, rows in sorted(grouped.items())}


def _paired_effects(
    records: list[dict[str, Any]],
    *,
    control_name: str,
    interventions: list[InterventionSpec],
) -> dict[str, Any]:
    by_prompt_condition: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        by_prompt_condition[(str(record["prompt_id"]), str(record["condition_name"]))] = record

    effects: dict[str, Any] = {}
    for intervention in interventions:
        if intervention.name == control_name:
            continue
        rv_control: list[float] = []
        rv_treated: list[float] = []
        rv_diffs: list[float] = []
        bt_control: list[int] = []
        bt_treated: list[int] = []
        prompt_modes: Counter[str] = Counter()

        for (prompt_id, condition_name), control_record in by_prompt_condition.items():
            if condition_name != control_name:
                continue
            treated_record = by_prompt_condition.get((prompt_id, intervention.name))
            if treated_record is None:
                continue
            prompt_modes[str(control_record["prompt_mode"])] += 1
            control_rv = control_record.get("output_rv")
            treated_rv = treated_record.get("output_rv")
            if control_rv is not None and treated_rv is not None:
                rv_control.append(float(control_rv))
                rv_treated.append(float(treated_rv))
                rv_diffs.append(float(treated_rv) - float(control_rv))
            bt_control.append(int(control_record["bt_art"]))
            bt_treated.append(int(treated_record["bt_art"]))

        bt_delta = _mean([float(t - c) for t, c in zip(bt_treated, bt_control)])
        better = sum(1 for t, c in zip(bt_treated, bt_control) if t > c)
        worse = sum(1 for t, c in zip(bt_treated, bt_control) if t < c)
        effects[intervention.name] = {
            "alpha": float(intervention.alpha),
            "n_prompt_pairs": int(sum(prompt_modes.values())),
            "rv_delta_mean": _mean(rv_diffs),
            "rv_cohens_dz": _cohens_dz(rv_diffs),
            "rv_p_value": _paired_t_pvalue(rv_treated, rv_control),
            "bt_art_rate_control": _mean([float(x) for x in bt_control]),
            "bt_art_rate_treated": _mean([float(x) for x in bt_treated]),
            "bt_art_rate_delta": bt_delta,
            "bt_art_cohens_h": _cohens_h(
                _mean([float(x) for x in bt_treated]),
                _mean([float(x) for x in bt_control]),
            ),
            "bt_art_exact_sign_p": _exact_sign_pvalue(better, worse),
            "bt_art_prompt_wins": int(better),
            "bt_art_prompt_losses": int(worse),
            "prompt_mode_counts": dict(prompt_modes),
        }
    return effects


def _build_blind_packet(
    records: list[dict[str, Any]],
    *,
    seed: int,
    csv_path: Path,
    key_path: Path,
) -> None:
    shuffled = list(records)
    rng = random.Random(seed + 404)
    rng.shuffle(shuffled)

    blind_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    for idx, record in enumerate(shuffled, start=1):
        sample_id = f"blind_{idx:04d}"
        blind_rows.append(
            {
                "sample_id": sample_id,
                "prompt_text": record["prompt_text"],
                "response_text": record["generated_text"],
                "depth_rating_1_to_5": "",
                "coherence_rating_1_to_5": "",
                "bt_art_yes_no": "",
                "notes": "",
            }
        )
        key_rows.append(
            {
                "sample_id": sample_id,
                "prompt_id": record["prompt_id"],
                "prompt_mode": record["prompt_mode"],
                "prompt_group": record["prompt_group"],
                "condition_name": record["condition_name"],
                "alpha": record["alpha"],
                "auto_classification": record["classification"],
                "auto_bt_art": record["bt_art"],
            }
        )

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(blind_rows[0].keys()))
        writer.writeheader()
        writer.writerows(blind_rows)

    key_path.write_text(json.dumps(_sanitize_json(key_rows), indent=2) + "\n", encoding="utf-8")


def run_causal_state_benchmark_v1_from_config(
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
    source_layer = int(params.get("source_layer") or late_layer)
    window = int(params.get("window") or 16)
    max_length = int(params.get("max_length") or 512)
    max_new_tokens = int(params.get("max_new_tokens") or 128)
    do_sample = bool(params.get("do_sample", True))
    temperature = float(params.get("temperature") or 0.7)
    top_p = float(params.get("top_p") or 1.0)
    holdout_per_group = int(params.get("holdout_per_group") or 5)
    min_text_chars = int(params.get("min_text_chars") or 80)
    max_source_per_label = int(params.get("max_source_per_label") or 64)
    max_source_per_session = int(params.get("max_source_per_session") or 8)

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
        {"name": "none", "alpha": 0.0},
        {"name": "toward_low_rv", "alpha": 2.0},
        {"name": "away_from_low_rv", "alpha": -2.0},
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
    torch.save(
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
    with records_jsonl.open("w", encoding="utf-8") as handle:
        for prompt_index, prompt_record in enumerate(heldout_prompts):
            prompt_text = prompt_record["prompt_text"]
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

            for intervention in interventions:
                set_seed(seed + prompt_index)
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

    paired = _paired_effects(
        all_records,
        control_name=control_name,
        interventions=interventions,
    )

    alpha_values = [float(r["alpha"]) for r in all_records]
    rv_values = [float(r["output_rv"]) for r in all_records if r["output_rv"] is not None]
    rv_alpha_pairs = [
        (float(r["alpha"]), float(r["output_rv"]))
        for r in all_records
        if r["output_rv"] is not None
    ]
    dose_response = {
        "mean_output_rv_by_condition": {
            name: payload["overall"]["mean_output_rv"] for name, payload in by_condition.items()
        },
        "bt_art_rate_by_condition": {
            name: payload["overall"]["bt_art_rate"] for name, payload in by_condition.items()
        },
        "alpha_vs_output_rv": _correlation(
            [alpha for alpha, _ in rv_alpha_pairs],
            [rv for _, rv in rv_alpha_pairs],
        ),
        "alpha_vs_bt_art": _correlation(alpha_values, [float(r["bt_art"]) for r in all_records]),
    }

    verdict = "inconclusive"
    toward_effect = paired.get("toward_low_rv")
    away_effect = paired.get("away_from_low_rv")
    if toward_effect and away_effect:
        if (
            (toward_effect.get("rv_delta_mean") or 0.0) < 0.0
            and (toward_effect.get("bt_art_rate_delta") or 0.0) > 0.0
            and (away_effect.get("rv_delta_mean") or 0.0) > 0.0
            and (away_effect.get("bt_art_rate_delta") or 0.0) < 0.0
        ):
            verdict = "causal_signal_detected"

    heldout_by_mode = Counter(row["prompt_mode"] for row in heldout_prompts)
    heldout_by_group = Counter(row["prompt_group"] for row in heldout_prompts)

    control_payload = by_condition.get(control_name, {}).get("overall", {})
    toward_payload = by_condition.get("toward_low_rv", {}).get("overall", {})

    summary = {
        "experiment": "causal_state_benchmark_v1",
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
            "positive_candidates_n": int(source_selection["positive_candidates_n"]),
            "negative_candidates_n": int(source_selection["negative_candidates_n"]),
            "positive_pool_n": int(source_selection["positive_pool_n"]),
            "negative_pool_n": int(source_selection["negative_pool_n"]),
            "positive_selected_n": int(len(source_selection["positive_records"])),
            "negative_selected_n": int(len(source_selection["negative_records"])),
            "raw_direction_norm": direction_payload["raw_direction_norm"],
            "positive_centroid_norm": direction_payload["positive_centroid_norm"],
            "negative_centroid_norm": direction_payload["negative_centroid_norm"],
            "centroid_cosine": direction_payload["centroid_cosine"],
            "positive_examples_preview": [
                {
                    "session_id": row["session_id"],
                    "turn": row["turn"],
                    "classification": row["classification"],
                    "output_rv": row["output_rv"],
                }
                for row in source_selection["positive_records"][:5]
            ],
            "negative_examples_preview": [
                {
                    "session_id": row["session_id"],
                    "turn": row["turn"],
                    "classification": row["classification"],
                    "output_rv": row["output_rv"],
                }
                for row in source_selection["negative_records"][:5]
            ],
        },
        "interventions": [_sanitize_json(intervention.__dict__) for intervention in interventions],
        "by_condition": _sanitize_json(by_condition),
        "paired_effects_vs_control": _sanitize_json(paired),
        "dose_response": _sanitize_json(dose_response),
        "rv_recursive_mean": toward_payload.get("mean_output_rv"),
        "rv_baseline_mean": control_payload.get("mean_output_rv"),
        "rv_delta_mean": (
            (toward_payload.get("mean_output_rv") or 0.0)
            - (control_payload.get("mean_output_rv") or 0.0)
            if toward_payload.get("mean_output_rv") is not None
            and control_payload.get("mean_output_rv") is not None
            else None
        ),
        "rv_cohens_d": toward_effect.get("rv_cohens_dz") if toward_effect else None,
        "rv_p_value": toward_effect.get("rv_p_value") if toward_effect else None,
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
        "rv": control_payload.get("mean_output_rv"),
        "logit_diff": None,
    }
    return ExperimentResult(summary=_sanitize_json(summary), baseline_metrics=baseline_metrics)
