"""
Causal state benchmark v4 — multi-site directional steering.

Extends v2 by learning state directions at MULTIPLE layers and applying
them simultaneously during generation. This tests whether multi-site
steering can achieve what single-site cannot: inducing recursive behavior
on baseline prompts.

Key innovation: uses contextlib.ExitStack to nest arbitrary numbers of
apply_steering_vector context managers, one per intervention layer.

The intervention grid specifies per-layer alphas, enabling:
  - Single-site controls (gate only, bridge only)
  - Multi-site induction (gate + bridge)
  - Suppression (negative alphas at both sites)
  - Asymmetric combinations (strong gate, weak bridge, etc.)
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from contextlib import ExitStack
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
    _load_session_turn_records,
    _mean,
    _paired_t_pvalue,
    _resolve_repo_path,
    _sanitize_json,
    _safe_float,
    _select_source_records,
)
from .causal_state_benchmark_v2 import (
    _bootstrap_ci,
    _compute_prompt_mode_effects,
    _dose_response_by_prompt_mode,
    _group_prompt_condition_means,
    _resolve_generation_seeds,
)


# ---------------------------------------------------------------------------
# Multi-site generation
# ---------------------------------------------------------------------------

def _generate_with_multisite_intervention(
    model,
    tokenizer,
    *,
    prompt: str,
    layer_specs: list[dict[str, Any]],
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    device: str,
    max_length: int,
) -> tuple[str, int]:
    """Generate text with steering vectors applied at multiple layers simultaneously.

    Args:
        layer_specs: List of dicts, each with:
            - layer_idx (int): layer to steer
            - steering_vector (torch.Tensor): direction to add
            - alpha (float): scaling factor (0.0 = no intervention)
    """
    import torch
    from src.steering.activation_patching import (
        apply_mlp_steering_vector,
        apply_steering_vector,
    )

    enc = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=max_length
    ).to(device)

    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": bool(do_sample),
    }
    if do_sample:
        generate_kwargs["temperature"] = float(temperature)
        generate_kwargs["top_p"] = float(top_p)

    with torch.no_grad(), ExitStack() as stack:
        for spec in layer_specs:
            if abs(spec["alpha"]) > 1e-9:
                component = str(spec.get("component") or "residual").lower()
                token_window = spec.get("token_window")
                if component in {"residual", "resid"}:
                    stack.enter_context(
                        apply_steering_vector(
                            model,
                            spec["layer_idx"],
                            spec["steering_vector"],
                            spec["alpha"],
                            token_window=token_window,
                        )
                    )
                elif component == "mlp":
                    stack.enter_context(
                        apply_mlp_steering_vector(
                            model,
                            spec["layer_idx"],
                            spec["steering_vector"],
                            spec["alpha"],
                            token_window=token_window,
                        )
                    )
                else:
                    raise ValueError(f"Unsupported multisite component: {component}")
        output = model.generate(**enc, **generate_kwargs)

    generated_ids = output[0][enc.input_ids.shape[1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True), len(generated_ids)


# ---------------------------------------------------------------------------
# Multi-site intervention spec
# ---------------------------------------------------------------------------

class MultisiteIntervention:
    """An intervention that specifies per-role alphas."""

    __slots__ = ("name", "alphas", "prompt_suffix_by_mode")

    def __init__(
        self,
        name: str,
        alphas: dict[str, float],
        prompt_suffix_by_mode: dict[str, str] | None = None,
    ):
        self.name = name
        self.alphas = dict(alphas)  # role -> alpha
        self.prompt_suffix_by_mode = dict(prompt_suffix_by_mode or {})

    def to_dict(self) -> dict[str, Any]:
        payload = {"name": self.name, "alphas": dict(self.alphas)}
        if self.prompt_suffix_by_mode:
            payload["prompt_suffix_by_mode"] = dict(self.prompt_suffix_by_mode)
        return payload


def _parse_multisite_interventions(
    raw: list[dict[str, Any]],
) -> list[MultisiteIntervention]:
    out: list[MultisiteIntervention] = []
    for item in raw:
        name = str(item["name"])
        alphas = {str(k): float(v) for k, v in item["alphas"].items()}
        prompt_suffix_by_mode = {
            str(k): str(v)
            for k, v in (item.get("prompt_suffix_by_mode") or {}).items()
            if str(v).strip()
        }
        out.append(
            MultisiteIntervention(
                name=name,
                alphas=alphas,
                prompt_suffix_by_mode=prompt_suffix_by_mode,
            )
        )
    return out


def _total_alpha(intervention: MultisiteIntervention) -> float:
    """Sum of all per-role alphas — used for dose-response analysis."""
    return sum(intervention.alphas.values())


# ---------------------------------------------------------------------------
# Synergy detection
# ---------------------------------------------------------------------------

def _compute_synergy(
    records: list[dict[str, Any]],
    *,
    gate_only_name: str,
    bridge_only_name: str,
    both_name: str,
    control_name: str,
    prompt_mode: Optional[str] = None,
) -> dict[str, Any]:
    """Test for synergy: does gate+bridge > gate_alone + bridge_alone?

    Synergy = both_effect - (gate_effect + bridge_effect)
    If positive, the sites interact synergistically.
    """
    reduced = _group_prompt_condition_means(records, prompt_mode=prompt_mode)
    prompt_ids = sorted({pid for pid, _ in reduced})

    synergy_bt: list[float] = []
    synergy_rv: list[float] = []

    for pid in prompt_ids:
        ctrl = reduced.get((pid, control_name))
        gate = reduced.get((pid, gate_only_name))
        bridge = reduced.get((pid, bridge_only_name))
        both = reduced.get((pid, both_name))
        if not all([ctrl, gate, bridge, both]):
            continue

        ctrl_bt = ctrl.get("mean_bt_art", 0.0) or 0.0
        gate_bt = (gate.get("mean_bt_art", 0.0) or 0.0) - ctrl_bt
        bridge_bt = (bridge.get("mean_bt_art", 0.0) or 0.0) - ctrl_bt
        both_bt = (both.get("mean_bt_art", 0.0) or 0.0) - ctrl_bt
        synergy_bt.append(both_bt - (gate_bt + bridge_bt))

        ctrl_rv = ctrl.get("mean_output_rv") or 0.0
        gate_rv = (gate.get("mean_output_rv") or 0.0) - ctrl_rv
        bridge_rv = (bridge.get("mean_output_rv") or 0.0) - ctrl_rv
        both_rv = (both.get("mean_output_rv") or 0.0) - ctrl_rv
        synergy_rv.append(both_rv - (gate_rv + bridge_rv))

    return {
        "n_prompts": len(synergy_bt),
        "bt_art_synergy_mean": _mean(synergy_bt),
        "bt_art_synergy_dz": _cohens_dz(synergy_bt),
        "bt_art_synergy_p": _paired_t_pvalue(
            [s + 0.0 for s in synergy_bt],
            [0.0] * len(synergy_bt),
        ),
        "rv_synergy_mean": _mean(synergy_rv),
        "rv_synergy_dz": _cohens_dz(synergy_rv),
        "rv_synergy_p": _paired_t_pvalue(
            [s + 0.0 for s in synergy_rv],
            [0.0] * len(synergy_rv),
        ),
    }


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def run_causal_state_benchmark_v4_multisite_from_config(
    cfg: Dict[str, Any], run_dir: Path
) -> ExperimentResult:
    import torch
    from scripts.sustained_gnani_v3 import classify_output, compute_prefill_metrics
    from src.core.models import load_model, set_seed
    from src.metrics.rv import compute_rv

    model_cfg = cfg.get("model") or {}
    params = cfg.get("params") or {}

    seed = int(cfg.get("seed") or 0)
    set_seed(seed)

    device = str(
        model_cfg.get("device")
        or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model_name = str(model_cfg.get("name") or "mistralai/Mistral-7B-v0.1")
    early_layer = int(params.get("early_layer") or 5)
    late_layer = int(params.get("late_layer") or 27)
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
    control_name = str(params.get("control_name") or "control")

    # ---- Source layer definitions ----
    raw_source_layers = params.get("source_layers") or {
        "gate": {"layer": 5, "window": 32},
        "bridge": {"layer": 25, "window": 32},
    }
    source_layer_defs: dict[str, dict[str, Any]] = {}
    for role, spec in raw_source_layers.items():
        source_layer_defs[str(role)] = {
            "layer": int(spec["layer"]),
            "window": int(spec.get("window") or 32),
            "component": str(spec.get("component") or "residual"),
            "token_window": (
                int(spec["token_window"])
                if spec.get("token_window") is not None
                else None
            ),
        }

    # ---- Multisite interventions ----
    raw_interventions = params.get("multisite_interventions") or _default_interventions(
        list(source_layer_defs.keys())
    )
    interventions = _parse_multisite_interventions(raw_interventions)

    # Also build InterventionSpec list for v2-compatible analysis
    # Use total_alpha as the "alpha" for dose-response
    v2_interventions = [
        InterventionSpec(name=iv.name, alpha=_total_alpha(iv))
        for iv in interventions
    ]

    # ---- Standard setup ----
    recursive_groups = list(
        params.get("recursive_groups") or ["L3_deeper", "L4_full", "L5_refined"]
    )
    baseline_groups = list(
        params.get("baseline_groups")
        or ["baseline_math", "baseline_factual", "baseline_creative"]
    )
    sessions_dir = _resolve_repo_path(
        params.get("source_sessions_dir") or "results/sustained_gnani_v3_fixed"
    )
    positive_classes = set(
        params.get("positive_classes") or DEFAULT_POSITIVE_CLASSES
    )
    negative_classes = set(
        params.get("negative_classes") or DEFAULT_NEGATIVE_CLASSES
    )
    positive_quantile = float(params.get("positive_quantile") or 0.35)
    negative_quantile = float(params.get("negative_quantile") or 0.65)
    positive_session_types = params.get("positive_session_types")
    negative_session_types = params.get("negative_session_types")
    positive_session_types_set = (
        set(positive_session_types) if positive_session_types else None
    )
    negative_session_types_set = (
        set(negative_session_types) if negative_session_types else None
    )

    if not sessions_dir.exists():
        raise FileNotFoundError(f"Source sessions dir not found: {sessions_dir}")

    model, tokenizer = load_model(
        model_name, device=device, attn_implementation="eager"
    )
    model.eval()

    loader = PromptLoader()
    bank_version = loader.version
    (run_dir / "prompt_bank_version.txt").write_text(bank_version, encoding="utf-8")
    (run_dir / "prompt_bank_version.json").write_text(
        json.dumps({"version": bank_version}, indent=2) + "\n", encoding="utf-8"
    )

    # ---- Load source records (shared across all layers) ----
    session_records = _load_session_turn_records(
        sessions_dir, min_text_chars=min_text_chars
    )
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

    # ---- Learn state direction at EACH layer ----
    steering_vectors: dict[str, torch.Tensor] = {}
    direction_payloads: dict[str, dict[str, Any]] = {}
    state_directions_artifact: dict[str, Any] = {}

    for role, layer_def in source_layer_defs.items():
        layer_idx = layer_def["layer"]
        window = layer_def["window"]
        print(
            f"[v4] Computing state direction for role={role} layer={layer_idx} window={window}",
            flush=True,
        )
        payload = _compute_state_direction(
            model,
            tokenizer,
            positive_records=source_selection["positive_records"],
            negative_records=source_selection["negative_records"],
            layer_idx=layer_idx,
            window=window,
            device=device,
            max_length=max_length,
            component=str(layer_def.get("component") or "residual"),
        )
        steering_vectors[role] = payload["direction"].to(device)
        direction_payloads[role] = payload
        state_directions_artifact[role] = {
            "direction": payload["direction"],
            "positive_centroid": payload["positive_centroid"],
            "negative_centroid": payload["negative_centroid"],
            "layer": layer_idx,
            "window": window,
            "component": str(layer_def.get("component") or "residual"),
            "token_window": layer_def.get("token_window"),
            "raw_direction_norm": payload["raw_direction_norm"],
            "positive_centroid_norm": payload["positive_centroid_norm"],
            "negative_centroid_norm": payload["negative_centroid_norm"],
            "centroid_cosine": payload["centroid_cosine"],
        }

    state_artifact_path = run_dir / "state_directions.pt"
    torch.save(state_directions_artifact, state_artifact_path)

    # ---- Build held-out prompt set ----
    heldout_prompts = _build_holdout_prompt_set(
        loader,
        recursive_groups=recursive_groups,
        baseline_groups=baseline_groups,
        holdout_per_group=holdout_per_group,
        seed=seed,
    )
    if not heldout_prompts:
        raise RuntimeError("Held-out prompt set is empty.")

    # ---- Main generation loop ----
    records_jsonl = run_dir / "benchmark_records.jsonl"
    all_records: list[dict[str, Any]] = []
    total_jobs = len(heldout_prompts) * len(generation_seeds) * len(interventions)
    job_idx = 0

    # Use the "first" layer def's window for prompt R_V measurement
    rv_window = list(source_layer_defs.values())[0]["window"]

    prompt_variant_cache: dict[tuple[str, str], dict[str, Any]] = {}

    with records_jsonl.open("w", encoding="utf-8") as handle:
        for prompt_index, prompt_record in enumerate(heldout_prompts, start=1):
            prompt_text = prompt_record["prompt_text"]
            print(
                f"[prompt {prompt_index}/{len(heldout_prompts)}] "
                f"{prompt_record['prompt_mode']} {prompt_record['prompt_group']} "
                f"{prompt_record['prompt_id']}",
                flush=True,
            )
            prompt_rv = _safe_float(
                compute_rv(
                    model,
                    tokenizer,
                    prompt_text,
                    early=early_layer,
                    late=late_layer,
                    window=rv_window,
                    device=device,
                )
            )

            for generation_seed in generation_seeds:
                for intervention in interventions:
                    job_idx += 1

                    cache_key = (
                        str(prompt_record["prompt_id"]),
                        str(intervention.name),
                    )
                    cached_variant = prompt_variant_cache.get(cache_key)
                    if cached_variant is None:
                        prompt_mode = str(prompt_record["prompt_mode"])
                        prompt_suffix = (
                            intervention.prompt_suffix_by_mode.get(prompt_mode) or ""
                        ).strip()
                        conditioned_prompt_text = prompt_text
                        if prompt_suffix:
                            conditioned_prompt_text = (
                                prompt_text.rstrip() + "\n\n" + prompt_suffix
                            )
                        conditioned_prompt_rv = (
                            prompt_rv
                            if conditioned_prompt_text == prompt_text
                            else _safe_float(
                                compute_rv(
                                    model,
                                    tokenizer,
                                    conditioned_prompt_text,
                                    early=early_layer,
                                    late=late_layer,
                                    window=rv_window,
                                    device=device,
                                )
                            )
                        )
                        cached_variant = {
                            "generation_prompt_text": conditioned_prompt_text,
                            "generation_prompt_rv": conditioned_prompt_rv,
                            "prompt_suffix": prompt_suffix,
                        }
                        prompt_variant_cache[cache_key] = cached_variant

                    # Build per-layer specs
                    layer_specs: list[dict[str, Any]] = []
                    alpha_summary: dict[str, float] = {}
                    for role, layer_def in source_layer_defs.items():
                        alpha = float(intervention.alphas.get(role, 0.0))
                        alpha_summary[role] = alpha
                        layer_specs.append(
                            {
                                "layer_idx": layer_def["layer"],
                                "steering_vector": steering_vectors[role],
                                "alpha": alpha,
                                "component": str(layer_def.get("component") or "residual"),
                                "token_window": layer_def.get("token_window"),
                            }
                        )

                    active_str = " ".join(
                        f"{r}={a:+.1f}" for r, a in alpha_summary.items()
                    )
                    print(
                        f"  [{job_idx}/{total_jobs}] seed={generation_seed} "
                        f"cond={intervention.name} {active_str}",
                        flush=True,
                    )

                    set_seed(int(generation_seed))
                    generated_text, gen_tokens = _generate_with_multisite_intervention(
                        model,
                        tokenizer,
                        prompt=str(cached_variant["generation_prompt_text"]),
                        layer_specs=layer_specs,
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

                    # Use total_alpha for v2-compatible analysis
                    record = {
                        "prompt_id": prompt_record["prompt_id"],
                        "prompt_mode": prompt_record["prompt_mode"],
                        "prompt_group": prompt_record["prompt_group"],
                        "prompt_text": prompt_text,
                        "prompt_rv": prompt_rv,
                        "generation_prompt_text": cached_variant["generation_prompt_text"],
                        "generation_prompt_rv": cached_variant["generation_prompt_rv"],
                        "prompt_suffix_applied": cached_variant["prompt_suffix"],
                        "condition_name": intervention.name,
                        "alpha": _total_alpha(intervention),
                        "alphas": dict(alpha_summary),
                        "generation_seed": int(generation_seed),
                        "generated_text": generated_text,
                        "generated_tokens": int(gen_tokens),
                        "classification": classification,
                        "bt_art": int(classification in positive_classes),
                        "output_rv": output_rv,
                        "output_metrics": metrics or {},
                    }
                    all_records.append(record)
                    handle.write(
                        json.dumps(_sanitize_json(record), ensure_ascii=False) + "\n"
                    )

    # ---- Blind ratings ----
    blind_csv = run_dir / "blind_ratings.csv"
    blind_key = run_dir / "blind_key.json"
    _build_blind_packet(
        all_records, seed=seed, csv_path=blind_csv, key_path=blind_key
    )

    # ---- Aggregate by condition ----
    by_condition: dict[str, Any] = {}
    for intervention in interventions:
        condition_records = [
            r for r in all_records if r["condition_name"] == intervention.name
        ]
        by_condition[intervention.name] = {
            "alphas": dict(intervention.alphas),
            "total_alpha": _total_alpha(intervention),
            "overall": _aggregate_condition(condition_records),
            "by_prompt_mode": _aggregate_by_prompt_mode(condition_records),
        }

    # ---- Effects analysis (reuse v2 machinery) ----
    effects_by_mode = {
        "overall": _compute_prompt_mode_effects(
            all_records,
            interventions=v2_interventions,
            control_name=control_name,
            prompt_mode=None,
            bootstrap_resamples=bootstrap_resamples,
            seed=seed,
        ),
        primary_prompt_mode: _compute_prompt_mode_effects(
            all_records,
            interventions=v2_interventions,
            control_name=control_name,
            prompt_mode=primary_prompt_mode,
            bootstrap_resamples=bootstrap_resamples,
            seed=seed + 10,
        ),
        control_prompt_mode: _compute_prompt_mode_effects(
            all_records,
            interventions=v2_interventions,
            control_name=control_name,
            prompt_mode=control_prompt_mode,
            bootstrap_resamples=bootstrap_resamples,
            seed=seed + 20,
        ),
    }

    # ---- Dose-response (total alpha) ----
    dose_response = {
        "overall": _dose_response_by_prompt_mode(all_records, prompt_mode=None),
        primary_prompt_mode: _dose_response_by_prompt_mode(
            all_records, prompt_mode=primary_prompt_mode
        ),
        control_prompt_mode: _dose_response_by_prompt_mode(
            all_records, prompt_mode=control_prompt_mode
        ),
    }

    # ---- Synergy analysis ----
    # Try to detect synergy if we have the standard intervention names
    synergy: dict[str, Any] = {}
    roles = sorted(source_layer_defs.keys())
    if len(roles) == 2:
        r0, r1 = roles[0], roles[1]
        # Look for matching intervention names
        iv_names = {iv.name for iv in interventions}
        for alpha_tag in ["2", "3"]:
            gate_name = f"{r0}_only_{alpha_tag}"
            bridge_name = f"{r1}_only_{alpha_tag}"
            both_name = f"both_{alpha_tag}"
            if all(n in iv_names for n in [gate_name, bridge_name, both_name]):
                for mode_label, mode_val in [
                    ("overall", None),
                    (primary_prompt_mode, primary_prompt_mode),
                    (control_prompt_mode, control_prompt_mode),
                ]:
                    key = f"alpha_{alpha_tag}_{mode_label}"
                    synergy[key] = _compute_synergy(
                        all_records,
                        gate_only_name=gate_name,
                        bridge_only_name=bridge_name,
                        both_name=both_name,
                        control_name=control_name,
                        prompt_mode=mode_val,
                    )

    # ---- Verdict ----
    verdict = _compute_multisite_verdict(
        by_condition=by_condition,
        effects_by_mode=effects_by_mode,
        synergy=synergy,
        control_name=control_name,
        primary_prompt_mode=primary_prompt_mode,
        control_prompt_mode=control_prompt_mode,
        roles=roles,
    )

    # ---- Summary ----
    heldout_by_mode = Counter(row["prompt_mode"] for row in heldout_prompts)
    heldout_by_group = Counter(row["prompt_group"] for row in heldout_prompts)

    summary: dict[str, Any] = {
        "experiment": "causal_state_benchmark_v4_multisite",
        "model_name": model_name,
        "device": device,
        "prompt_bank_version": bank_version,
        "source_sessions_dir": str(sessions_dir),
        "source_layers": {
            role: {
                "layer": d["layer"],
                "window": d["window"],
                "component": str(d.get("component") or "residual"),
                "token_window": d.get("token_window"),
                "direction_norm": direction_payloads[role]["raw_direction_norm"],
                "centroid_cosine": direction_payloads[role]["centroid_cosine"],
            }
            for role, d in source_layer_defs.items()
        },
        "early_layer": int(early_layer),
        "late_layer": int(late_layer),
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
        },
        "multisite_interventions": _sanitize_json(
            [iv.to_dict() for iv in interventions]
        ),
        "by_condition": _sanitize_json(by_condition),
        "effects_by_prompt_mode": _sanitize_json(effects_by_mode),
        "dose_response": _sanitize_json(dose_response),
        "synergy": _sanitize_json(synergy) if synergy else {},
        "verdict": verdict,
        "timestamp": __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S"),
        "artifacts": {
            "records_jsonl": str(records_jsonl),
            "blind_ratings_csv": str(blind_csv),
            "blind_key_json": str(blind_key),
            "state_directions_pt": str(state_artifact_path),
        },
    }

    baseline_metrics = {
        "rv": (
            by_condition.get(control_name, {})
            .get("overall", {})
            .get("mean_output_rv")
        ),
        "logit_diff": None,
    }

    return ExperimentResult(
        summary=_sanitize_json(summary), baseline_metrics=baseline_metrics
    )


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------

def _compute_multisite_verdict(
    *,
    by_condition: dict[str, Any],
    effects_by_mode: dict[str, Any],
    synergy: dict[str, Any],
    control_name: str,
    primary_prompt_mode: str,
    control_prompt_mode: str,
    roles: list[str],
) -> str:
    """Determine verdict from multi-site results.

    Hierarchy:
      1. multisite_sufficient — baseline BT+ART induced > 15% by multi-site
      2. multisite_synergy — gate+bridge > gate+bridge individually (p<0.05)
      3. multisite_additive — multi-site amplifies but no synergy
      4. single_site_only — only single-site effects detected
      5. inconclusive — nothing significant
    """
    # Check if any multi-site condition induced baseline BT+ART > 15%
    baseline_induced = False
    for cond_name, cond_data in by_condition.items():
        if cond_name == control_name:
            continue
        alphas = cond_data.get("alphas", {})
        n_active = sum(1 for v in alphas.values() if abs(v) > 1e-9)
        if n_active < 2:
            continue
        baseline_data = cond_data.get("by_prompt_mode", {}).get(
            control_prompt_mode, {}
        )
        bt_rate = baseline_data.get("bt_art_rate", 0.0)
        if bt_rate is not None and bt_rate > 0.15:
            baseline_induced = True
            break

    if baseline_induced:
        return "multisite_sufficient"

    # Check synergy
    synergy_detected = False
    for key, syn_data in synergy.items():
        if primary_prompt_mode in key:
            p = syn_data.get("bt_art_synergy_p")
            mean = syn_data.get("bt_art_synergy_mean", 0.0)
            if p is not None and p < 0.05 and mean is not None and mean > 0:
                synergy_detected = True
                break

    if synergy_detected:
        return "multisite_synergy"

    # Check if any multi-site condition shows significant primary BT+ART gain
    primary_effects = effects_by_mode.get(primary_prompt_mode, {})
    any_multisite_effect = False
    for cond_name, effect in primary_effects.items():
        bt_delta = effect.get("bt_art_rate_delta", 0.0) or 0.0
        if bt_delta > 0:
            any_multisite_effect = True
            break

    if any_multisite_effect:
        return "multisite_additive"

    # Check single-site effects
    any_single = False
    for cond_name, effect in primary_effects.items():
        bt_delta = effect.get("bt_art_rate_delta", 0.0) or 0.0
        if bt_delta > 0:
            any_single = True
            break

    if any_single:
        return "single_site_only"

    return "inconclusive"


# ---------------------------------------------------------------------------
# Default intervention grid
# ---------------------------------------------------------------------------

def _default_interventions(roles: list[str]) -> list[dict[str, Any]]:
    """Generate default intervention grid for 2 roles."""
    if len(roles) != 2:
        raise ValueError(
            f"Default interventions require exactly 2 roles, got {len(roles)}: {roles}"
        )
    r0, r1 = roles[0], roles[1]
    return [
        {"name": "control", "alphas": {r0: 0.0, r1: 0.0}},
        # Single-site controls
        {"name": f"{r0}_only_2", "alphas": {r0: 2.0, r1: 0.0}},
        {"name": f"{r0}_only_3", "alphas": {r0: 3.0, r1: 0.0}},
        {"name": f"{r1}_only_2", "alphas": {r0: 0.0, r1: 2.0}},
        {"name": f"{r1}_only_3", "alphas": {r0: 0.0, r1: 3.0}},
        # Multi-site induction
        {"name": "both_2", "alphas": {r0: 2.0, r1: 2.0}},
        {"name": "both_3", "alphas": {r0: 3.0, r1: 3.0}},
        # Asymmetric
        {"name": f"{r0}_3_{r1}_2", "alphas": {r0: 3.0, r1: 2.0}},
        {"name": f"{r0}_2_{r1}_3", "alphas": {r0: 2.0, r1: 3.0}},
        # Suppression
        {"name": "suppress_both", "alphas": {r0: -2.0, r1: -2.0}},
    ]
