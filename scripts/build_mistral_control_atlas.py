#!/usr/bin/env python3
"""
Build a replay-first control-atlas dataset for the static Mistral viewer.

The atlas is grounded in locked experiment artifacts, then projects them into
an animated replay format for the website. The replay trajectories are
synthetic summaries built from measured control points rather than raw hidden
states; the viewer labels them accordingly.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
WEBSITE_DATA = REPO_ROOT / "website" / "data" / "mistral-control-atlas.json"
WEBSITE_DATA_JS = REPO_ROOT / "website" / "data" / "mistral-control-atlas-data.js"

PATH_PATCHING = REPO_ROOT / "results" / "path_patching" / "path_patching_summary_20260312_125939.json"
ANCHOR_BUNDLE = REPO_ROOT / "results" / "phase1_mechanism" / "runs" / "20260314_133516_causal_state_benchmark_v4_multisite_mistral_anchor_bundle_v5_ordinary_baselines_confirmatory"
INDUCED_PERSISTENCE = REPO_ROOT / "results" / "induced_persistence_followup_v2_long" / "20260314_151808"
SUBSPACE_L27 = REPO_ROOT / "results" / "subspace_component_steering_l27_v1" / "20260314_144647" / "summary.json"
L27_VALIDATION = REPO_ROOT / "results" / "phase1_mechanism" / "runs" / "20260312_133759_head_ablation_validation_mistral_l27_kv2_modern_core_measurement__summary.json"
L25_TARGET = REPO_ROOT / "results" / "phase1_mechanism" / "runs" / "20260311_055109_causal_state_targeted_scan_v1_mistral_targeted_scan_v1" / "best_candidate.json"

CONDITIONS = [
    "control",
    "anchor_only",
    "bridge_only_3",
    "anchor_bridge_3",
    "anchor_early_mlp_0p125_bridge_3",
]
PROMPT_MODES = ["baseline", "recursive"]
CLASS_RANK = {
    "BREAKTHROUGH": 5,
    "ARTICULATE": 4,
    "CONCEPTUAL": 3,
    "SURFACE": 2,
    "REPETITIVE": 1,
    "MALFORMED": 0,
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def smoothstep(t: float) -> float:
    t = clamp(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * smoothstep(t)


def gaussian(layer: int, center: float, width: float) -> float:
    return math.exp(-((layer - center) ** 2) / (2.0 * width * width))


def build_layer_profile(path_patching: dict) -> dict:
    layers = []
    indexed = defaultdict(dict)
    for row in path_patching["results"]:
        indexed[int(row["layer"])][row["component"]] = row

    max_abs_d = max(abs(row["cohens_d"]) for row in path_patching["results"])
    for layer in range(32):
        residual = indexed[layer].get("residual", {})
        mlp = indexed[layer].get("mlp", {})
        v_proj = indexed[layer].get("v_proj", {})
        max_component = max(
            abs(float(residual.get("cohens_d", 0.0))),
            abs(float(mlp.get("cohens_d", 0.0))),
            abs(float(v_proj.get("cohens_d", 0.0))),
        )
        layers.append(
            {
                "layer": layer,
                "depth": round(layer / 31.0, 4),
                "residual_d": float(residual.get("cohens_d", 0.0)),
                "mlp_d": float(mlp.get("cohens_d", 0.0)),
                "v_proj_d": float(v_proj.get("cohens_d", 0.0)),
                "field_strength": round(max_component / max_abs_d, 4),
            }
        )

    return {
        "layers": layers,
        "top_hits": [
            {
                "label": "L5 residual",
                "layer": 5,
                "component": "residual",
                "cohens_d": indexed[5]["residual"]["cohens_d"],
            },
            {
                "label": "L4 mlp",
                "layer": 4,
                "component": "mlp",
                "cohens_d": indexed[4]["mlp"]["cohens_d"],
            },
            {
                "label": "L5 v_proj",
                "layer": 5,
                "component": "v_proj",
                "cohens_d": indexed[5]["v_proj"]["cohens_d"],
            },
            {
                "label": "L27 v_proj",
                "layer": 27,
                "component": "v_proj",
                "cohens_d": indexed[27]["v_proj"]["cohens_d"],
            },
        ],
    }


def load_anchor_records() -> list[dict]:
    records_path = ANCHOR_BUNDLE / "benchmark_records.jsonl"
    rows = []
    for line in records_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def select_exemplar(rows: list[dict], condition: str, prompt_mode: str) -> dict:
    bucket = [
        row
        for row in rows
        if row["condition_name"] == condition and row["prompt_mode"] == prompt_mode
    ]
    bucket.sort(
        key=lambda row: (
            int(row.get("bt_art", 0)),
            CLASS_RANK.get(row.get("classification"), 0),
            -float(row.get("output_rv", 999.0)),
        ),
        reverse=True,
    )
    exemplar = bucket[0]
    return {
        "prompt_id": exemplar["prompt_id"],
        "prompt_group": exemplar["prompt_group"],
        "classification": exemplar["classification"],
        "bt_art": int(exemplar.get("bt_art", 0)),
        "prompt_rv": float(exemplar.get("prompt_rv", float("nan"))),
        "output_rv": float(exemplar.get("output_rv", float("nan"))),
        "prompt_text": exemplar["prompt_text"],
        "generated_text": exemplar["generated_text"],
    }


def build_condition_trajectory(
    *,
    condition: str,
    prompt_mode: str,
    metrics: dict,
    exemplar: dict,
    layer_profile: dict,
    l25_target: dict,
    l27_validation: dict,
) -> list[dict]:
    layers = layer_profile["layers"]
    early_source = sum(max(0.0, layer["residual_d"]) for layer in layers[:6]) / 6.0
    early_source = clamp(early_source / 4.2, 0.0, 1.0)
    bridge_delta = l25_target["effects_by_prompt_mode"][prompt_mode]["toward"]["bt_art_rate_delta"]
    l27_effect = l27_validation["analysis"][prompt_mode]["target_at_target_layer"]["mean"]
    bt_art = metrics["bt_art_rate"]
    output_rv = metrics["mean_output_rv"]

    anchor_strength = 1.0 if "anchor" in condition else 0.0
    bridge_strength = 1.0 if "bridge" in condition else 0.0
    mlp_strength = 1.0 if "early_mlp" in condition else 0.0
    recursive_bias = 0.18 if prompt_mode == "recursive" else -0.04
    compression = clamp((0.70 - output_rv) / 0.18, -1.0, 1.0)

    control_layers = [0, 5, 25, 27, 31]
    x_nodes = [
        -0.62 + recursive_bias + anchor_strength * 0.18,
        -0.25 + recursive_bias + early_source * 0.22 + anchor_strength * 0.20 + mlp_strength * 0.18,
        -0.08 + recursive_bias + bridge_strength * 0.28 + bt_art * 1.15 + bridge_delta * 0.75,
        0.08 + recursive_bias + bridge_strength * 0.22 + bt_art * 0.92 + compression * 0.28,
        0.04 + recursive_bias + bt_art * 0.85 + compression * 0.34,
    ]
    z_nodes = [
        0.42 - anchor_strength * 0.08,
        0.28 - early_source * 0.22 - mlp_strength * 0.10,
        0.05 - bridge_strength * 0.20 - bt_art * 0.36,
        -0.28 - compression * 0.38 - l27_effect * 2.2,
        -0.22 - compression * 0.28 - l27_effect * 1.4,
    ]

    prompt_rv = exemplar["prompt_rv"]
    output_rv_ex = exemplar["output_rv"]
    rv_drop = clamp((prompt_rv - output_rv_ex) / 0.40, -1.0, 1.0)

    points = []
    for layer in range(32):
        idx = 0
        while idx < len(control_layers) - 2 and layer > control_layers[idx + 1]:
            idx += 1
        left_layer = control_layers[idx]
        right_layer = control_layers[idx + 1]
        local_t = 0.0 if right_layer == left_layer else (layer - left_layer) / (right_layer - left_layer)
        x = lerp(x_nodes[idx], x_nodes[idx + 1], local_t)
        z = lerp(z_nodes[idx], z_nodes[idx + 1], local_t)
        field = layers[layer]
        x += 0.04 * max(0.0, field["residual_d"]) / 4.2
        z -= 0.05 * abs(field["v_proj_d"]) / 2.6 * gaussian(layer, 27.0, 3.0)
        x += 0.02 * rv_drop * math.sin(layer * 0.55 + (0.1 if prompt_mode == "recursive" else 0.0))
        z += 0.03 * math.cos(layer * 0.36 + anchor_strength * 0.8 + bridge_strength * 0.3)
        points.append(
            {
                "layer": layer,
                "x": round(x, 4),
                "y": round(layer / 31.0, 4),
                "z": round(z, 4),
            }
        )
    return points


def build_anchor_bundle(anchor_summary: dict, layer_profile: dict, l25_target: dict, l27_validation: dict) -> dict:
    rows = load_anchor_records()
    by_condition = {}
    for condition in CONDITIONS:
        by_condition[condition] = {"modes": {}}
        for prompt_mode in PROMPT_MODES:
            metrics = anchor_summary["by_condition"][condition]["by_prompt_mode"][prompt_mode]
            exemplar = select_exemplar(rows, condition, prompt_mode)
            by_condition[condition]["modes"][prompt_mode] = {
                "metrics": {
                    "bt_art_rate": metrics["bt_art_rate"],
                    "mean_output_rv": metrics["mean_output_rv"],
                    "n": metrics["n"],
                    "mean_generated_tokens": metrics["mean_generated_tokens"],
                    "class_counts": metrics["class_counts"],
                },
                "effect_vs_control": anchor_summary["effects_by_prompt_mode"][prompt_mode].get(condition),
                "exemplar": exemplar,
                "trajectory": build_condition_trajectory(
                    condition=condition,
                    prompt_mode=prompt_mode,
                    metrics=metrics,
                    exemplar=exemplar,
                    layer_profile=layer_profile,
                    l25_target=l25_target,
                    l27_validation=l27_validation,
                ),
            }
    return by_condition


def aggregate_turn_series(sessions: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for session in sessions:
        for turn in session["turns"]:
            grouped[int(turn["turn"])].append(turn)

    series = []
    for turn_idx in sorted(grouped):
        bucket = grouped[turn_idx]
        bt_art = sum(1 for turn in bucket if turn["classification"] in ("BREAKTHROUGH", "ARTICULATE"))
        clean = sum(1 for turn in bucket if turn["clean"])
        repetitive = sum(1 for turn in bucket if turn["classification"] == "REPETITIVE")
        valid_rv = [turn["output_rv"] for turn in bucket if not math.isnan(float(turn["output_rv"]))]
        mean_rv = sum(valid_rv) / len(valid_rv) if valid_rv else float("nan")
        series.append(
            {
                "turn": turn_idx,
                "bt_art_rate": round(bt_art / len(bucket), 4),
                "clean_rate": round(clean / len(bucket), 4),
                "repetitive_rate": round(repetitive / len(bucket), 4),
                "mean_output_rv": round(mean_rv, 4) if not math.isnan(mean_rv) else None,
            }
        )
    return series


def build_persistence() -> dict:
    summary = load_json(INDUCED_PERSISTENCE / "summary.json")
    sessions = json.loads((INDUCED_PERSISTENCE / "sessions.json").read_text(encoding="utf-8"))
    by_source = defaultdict(list)
    for session in sessions:
        by_source[session["source_condition"]].append(session)

    result = {
        "meta": {
            "experiment": summary["experiment"],
            "max_turns": summary["max_turns"],
            "n_seed_sessions": summary["n_seed_sessions"],
            "source_conditions": summary["source_conditions"],
        },
        "by_source_condition": {},
    }
    for condition, agg in summary["by_source_condition"].items():
        condition_sessions = by_source[condition]
        exemplar_session = max(condition_sessions, key=lambda row: (row["bt_art_rate"], -row["mean_rv"]))
        result["by_source_condition"][condition] = {
            "aggregate": agg,
            "turn_series": aggregate_turn_series(condition_sessions),
            "exemplar_session": {
                "source_prompt_id": exemplar_session["source_prompt_id"],
                "source_group": exemplar_session["source_group"],
                "source_classification": exemplar_session["source_classification"],
                "bt_art_rate": exemplar_session["bt_art_rate"],
                "mean_rv": exemplar_session["mean_rv"],
                "turns": [
                    {
                        "turn": turn["turn"],
                        "classification": turn["classification"],
                        "output_rv": None if math.isnan(float(turn["output_rv"])) else round(float(turn["output_rv"]), 4),
                        "response": turn["response"][:320],
                    }
                    for turn in exemplar_session["turns"]
                ],
            },
        }
    return result


def build_subspace(layer_profile: dict, l27_validation: dict) -> dict:
    summary = load_json(SUBSPACE_L27)
    ranked = []
    for key, value in summary["effects_vs_control"].items():
        method, alpha = key.split("::")
        ranked.append(
            {
                "method": method,
                "alpha": float(alpha),
                "baseline_bt_art_delta": value["baseline_bt_art_delta"],
                "recursive_bt_art_delta": value["recursive_bt_art_delta"],
                "baseline_rv_delta": value["baseline_rv_delta"],
                "recursive_rv_delta": value["recursive_rv_delta"],
            }
        )
    ranked.sort(key=lambda row: (row["recursive_bt_art_delta"], -abs(row["baseline_bt_art_delta"])), reverse=True)

    methods = ["control", "mean_diff", "pca_pc1", "subspace3_parallel", "orthogonal_residual"]
    trajectories = {}
    for method in methods:
        alpha = 0.0 if method == "control" else 4.0
        recursive_control = summary["by_mode_method_alpha"]["recursive::control::0.0"]
        baseline_control = summary["by_mode_method_alpha"]["baseline::control::0.0"]
        recursive_row = summary["by_mode_method_alpha"][f"recursive::{method}::{alpha:.1f}" if method != "control" else "recursive::control::0.0"]
        baseline_row = summary["by_mode_method_alpha"][f"baseline::{method}::{alpha:.1f}" if method != "control" else "baseline::control::0.0"]
        base_gain = recursive_row["bt_art_rate"] - recursive_control["bt_art_rate"]
        compression = clamp((recursive_control["mean_output_rv"] - recursive_row["mean_output_rv"]) / 0.08, -1.0, 1.0)
        points = []
        for layer in range(32):
            early_field = max(0.0, layer_profile["layers"][layer]["residual_d"]) / 4.2
            local = gaussian(layer, 25.0, 2.5) * 0.35 + gaussian(layer, 27.0, 2.0) * 0.8
            method_bias = {
                "control": 0.0,
                "mean_diff": 0.08,
                "pca_pc1": 0.16,
                "subspace3_parallel": 0.32,
                "orthogonal_residual": -0.12,
            }[method]
            x = -0.5 + early_field * 0.12 + local * (base_gain * 2.2 + method_bias)
            z = 0.28 - local * (compression * 0.9 + method_bias * 0.5)
            if layer >= 27:
                z -= l27_validation["analysis"]["recursive"]["target_at_target_layer"]["mean"] * 2.4
            points.append({"layer": layer, "x": round(x, 4), "y": round(layer / 31.0, 4), "z": round(z, 4)})
        trajectories[method] = {
            "alpha": alpha,
            "metrics_by_mode": {
                "baseline": baseline_row,
                "recursive": recursive_row,
            },
            "effects_vs_control": {
                "baseline_bt_art_delta": baseline_row["bt_art_rate"] - baseline_control["bt_art_rate"],
                "recursive_bt_art_delta": recursive_row["bt_art_rate"] - recursive_control["bt_art_rate"],
                "baseline_rv_delta": baseline_row["mean_output_rv"] - baseline_control["mean_output_rv"],
                "recursive_rv_delta": recursive_row["mean_output_rv"] - recursive_control["mean_output_rv"],
            },
            "trajectory": points,
        }

    return {
        "meta": {
            "layer": summary["layer"],
            "vector_metadata": summary["vector_metadata"],
            "decomposition": summary["decomposition"],
            "winners": summary["winners"],
        },
        "ranked_effects": ranked[:8],
        "methods": trajectories,
    }


def build_summary_cards(anchor_bundle: dict, persistence: dict, subspace: dict, l25_target: dict, l27_validation: dict) -> list[dict]:
    return [
        {
            "title": "Early Source Region",
            "value": "L0-L5",
            "detail": "Path patching peaks at L5 residual d=4.15, with L4 mlp and L5 v_proj as the strongest early non-residual handles.",
        },
        {
            "title": "Late Controller",
            "value": "L25",
            "detail": f"Targeted steering winner {l25_target['candidate_name']} lifts recursive BT+ART from 0.42 to 0.67 on its frozen benchmark slice.",
        },
        {
            "title": "Readout Cluster",
            "value": "L27",
            "detail": f"Modern L27 validation remains extremely strong, recursive target-at-target mean {l27_validation['analysis']['recursive']['target_at_target_layer']['mean']:.3f}.",
        },
        {
            "title": "Anchor Effect",
            "value": "16.7%",
            "detail": "On ordinary baselines, anchor+bridge pushes BT+ART from 3.1% control to 16.7%, showing genuine induction beyond matched controls.",
        },
        {
            "title": "Persistence",
            "value": "24 turns",
            "detail": "Induced anchor conditions stay clean over 24 turns, but persistence remains moderate rather than fully self-sustaining.",
        },
        {
            "title": "Subspace Winner",
            "value": "parallel",
            "detail": f"At L27, subspace3_parallel @ 4.0 reaches recursive BT+ART {subspace['meta']['winners']['recursive']['bt_art_rate']:.2f} while baseline remains low.",
        },
    ]


def build_dataset() -> dict:
    path_patching = load_json(PATH_PATCHING)
    anchor_summary = load_json(ANCHOR_BUNDLE / "summary.json")
    l25_target = load_json(L25_TARGET)
    l27_validation = load_json(L27_VALIDATION)

    layer_profile = build_layer_profile(path_patching)
    anchor_bundle = build_anchor_bundle(anchor_summary, layer_profile, l25_target, l27_validation)
    persistence = build_persistence()
    subspace = build_subspace(layer_profile, l27_validation)

    return {
        "meta": {
            "title": "Mistral Live Control Atlas",
            "model": "mistralai/Mistral-7B-v0.1",
            "total_layers": 32,
            "trajectory_note": "Trajectory lines are replay syntheses from measured control points, not raw hidden-state captures.",
            "artifacts": {
                "path_patching": str(PATH_PATCHING.relative_to(REPO_ROOT)),
                "anchor_bundle": str((ANCHOR_BUNDLE / 'summary.json').relative_to(REPO_ROOT)),
                "induced_persistence": str((INDUCED_PERSISTENCE / 'summary.json').relative_to(REPO_ROOT)),
                "subspace_l27": str(SUBSPACE_L27.relative_to(REPO_ROOT)),
                "l27_validation": str(L27_VALIDATION.relative_to(REPO_ROOT)),
                "l25_target": str(L25_TARGET.relative_to(REPO_ROOT)),
            },
        },
        "architecture": {
            "zones": [
                {"label": "Early source", "start": 0, "end": 5, "color": "#f59e0b"},
                {"label": "Controller", "start": 25, "end": 25, "color": "#10b981"},
                {"label": "Readout cluster", "start": 27, "end": 27, "color": "#38bdf8"},
            ],
            "layer_profile": layer_profile,
        },
        "anchor_bundle": anchor_bundle,
        "persistence": persistence,
        "subspace": subspace,
        "summary_cards": build_summary_cards(anchor_bundle, persistence, subspace, l25_target, l27_validation),
    }


def main() -> int:
    dataset = build_dataset()
    WEBSITE_DATA.parent.mkdir(parents=True, exist_ok=True)
    WEBSITE_DATA.write_text(json.dumps(dataset, indent=2), encoding="utf-8")
    WEBSITE_DATA_JS.write_text(
        "window.MISTRAL_CONTROL_ATLAS_DATA = " + json.dumps(dataset, indent=2) + ";\n",
        encoding="utf-8",
    )
    print(WEBSITE_DATA)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
