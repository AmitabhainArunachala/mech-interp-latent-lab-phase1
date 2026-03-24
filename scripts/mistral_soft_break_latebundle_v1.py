#!/usr/bin/env python3
"""
Soft anti-steer break test on recursive vs baseline prompts.

Goal:
- reduce recursive BT+ART with smaller late-stack counter-interventions
- keep malformed output low enough that the break cannot be dismissed as junk
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from prompts.loader import PromptLoader
from scripts.anchor_layermatched_protocol_v1 import (
    DEFAULT_BASELINE_GROUPS,
    DEFAULT_RECURSIVE_GROUPS,
    DEFAULT_V5_STATE_PATH,
    compute_output_rv,
    generate_with_hooks,
    load_prompt_split,
    normalize,
)
from scripts.layer_matched_multisite_steering import (
    apply_residual_steering,
    apply_vproj_steering,
    compute_vproj_vectors,
)
from src.utils.persistent_patching_classification import classify_output


REPO_ROOT = Path(__file__).resolve().parent.parent


def condition_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    bt_art = [row["bt_art"] for row in rows]
    malformed = [row["malformed"] for row in rows]
    repetitive = [row["repetitive"] for row in rows]
    non_malformed_rows = [row for row in rows if not row["malformed"]]
    class_counts = defaultdict(int)
    for row in rows:
        class_counts[row["classification"]] += 1
    return {
        "n": len(rows),
        "bt_art_rate": float(np.mean(bt_art)) if bt_art else 0.0,
        "malformed_rate": float(np.mean(malformed)) if malformed else 0.0,
        "repetitive_rate": float(np.mean(repetitive)) if repetitive else 0.0,
        "clean_rate": float(np.mean([row["clean"] for row in rows])) if rows else 0.0,
        "mean_output_rv": float(np.nanmean([row["output_rv"] for row in rows])) if rows else float("nan"),
        "mean_generated_tokens": float(np.mean([row["generated_tokens"] for row in rows])) if rows else float("nan"),
        "non_malformed_bt_art_rate": float(np.mean([row["bt_art"] for row in non_malformed_rows])) if non_malformed_rows else 0.0,
        "class_counts": dict(class_counts),
    }


def metric_or_none(summary_rows: dict[str, Any], name: str, field: str) -> float | None:
    row = summary_rows.get(name)
    if row is None:
        return None
    return row.get(field)


def main() -> int:
    parser = argparse.ArgumentParser(description="Soft late-bundle break test on Mistral")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--state-path", default=DEFAULT_V5_STATE_PATH)
    parser.add_argument("--train-per-group", type=int, default=6)
    parser.add_argument("--test-per-group", type=int, default=4)
    parser.add_argument("--split-seed", type=int, default=314)
    parser.add_argument("--experiment-name", default="mistral_soft_break_latebundle_v1")
    parser.add_argument(
        "--generation-seeds",
        type=int,
        nargs="+",
        default=[101, 202, 303, 404, 505, 606, 707, 808],
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / f"results/{args.experiment_name}/{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_name = "mistralai/Mistral-7B-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    state_path = REPO_ROOT / args.state_path
    if not state_path.exists():
        raise FileNotFoundError(f"Locked state directions not found: {state_path}")
    state_payload = torch.load(state_path, map_location="cpu")
    bridge_direction = normalize(state_payload["bridge"]["direction"].float())

    loader = PromptLoader()
    train_rec, train_base, test_rec, test_base = load_prompt_split(
        loader=loader,
        recursive_groups=DEFAULT_RECURSIVE_GROUPS,
        baseline_groups=DEFAULT_BASELINE_GROUPS,
        train_per_group=args.train_per_group,
        test_per_group=args.test_per_group,
        split_seed=args.split_seed,
    )
    train_rec_texts = [p["text"] if isinstance(p, dict) else p for p in train_rec]
    train_base_texts = [p["text"] if isinstance(p, dict) else p for p in train_base]

    vproj_vectors: dict[int, dict[str, Any]] = {}
    for layer_idx in [25, 27]:
        vproj_vectors[layer_idx] = compute_vproj_vectors(
            model,
            tokenizer,
            train_rec_texts,
            train_base_texts,
            layer_idx=layer_idx,
            window=16,
            device=args.device,
        )

    def make_vproj_hook(layer_idx: int, method: str, alpha: float):
        vec = vproj_vectors[layer_idx][method]
        return lambda: apply_vproj_steering(model, layer_idx, vec, alpha)

    def make_bridge_hook(alpha: float):
        return lambda: apply_residual_steering(model, 25, bridge_direction, alpha)

    conditions = [
        {"name": "control", "hooks": []},
        {"name": "anti_bridge_1p0", "hooks": [make_bridge_hook(-1.0)]},
        {"name": "anti_bridge_1p5", "hooks": [make_bridge_hook(-1.5)]},
        {"name": "anti_l27_only_1p0", "hooks": [make_vproj_hook(27, "subspace3_parallel", -1.0)]},
        {
            "name": "anti_late_only_soft",
            "hooks": [
                make_vproj_hook(25, "orthogonal_residual", -0.5),
                make_vproj_hook(27, "subspace3_parallel", -1.0),
                make_bridge_hook(-1.0),
            ],
        },
        {
            "name": "anti_late_only_medium",
            "hooks": [
                make_vproj_hook(25, "orthogonal_residual", -1.0),
                make_vproj_hook(27, "subspace3_parallel", -2.0),
                make_bridge_hook(-1.5),
            ],
        },
    ]

    test_prompts: list[dict[str, Any]] = []
    for prompt in test_rec:
        test_prompts.append(
            {
                "text": prompt["text"] if isinstance(prompt, dict) else prompt,
                "mode": "recursive",
                "group": prompt.get("group", "recursive") if isinstance(prompt, dict) else "recursive",
            }
        )
    for prompt in test_base:
        test_prompts.append(
            {
                "text": prompt["text"] if isinstance(prompt, dict) else prompt,
                "mode": "baseline",
                "group": prompt.get("group", "baseline") if isinstance(prompt, dict) else "baseline",
            }
        )

    records: list[dict[str, Any]] = []
    for prompt_index, prompt_record in enumerate(test_prompts):
        for generation_seed in args.generation_seeds:
            set_seed(int(generation_seed))
            for condition in conditions:
                generated_text, generated_tokens = generate_with_hooks(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt_record["text"],
                    hooks_to_apply=condition["hooks"],
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    device=args.device,
                )
                output_rv = compute_output_rv(
                    model,
                    tokenizer,
                    generated_text,
                    early_layer=5,
                    late_layer=27,
                    window=16,
                    device=args.device,
                )
                classification = classify_output(generated_text, output_rv)
                malformed = int(classification == "MALFORMED")
                repetitive = int(classification == "REPETITIVE")
                bt_art = int(classification in ("BREAKTHROUGH", "ARTICULATE"))
                clean = int(classification not in ("REPETITIVE", "MALFORMED"))
                records.append(
                    {
                        "prompt_index": prompt_index,
                        "prompt_mode": prompt_record["mode"],
                        "prompt_group": prompt_record["group"],
                        "generation_seed": generation_seed,
                        "condition": condition["name"],
                        "generated_tokens": generated_tokens,
                        "output_rv": output_rv,
                        "classification": classification,
                        "bt_art": bt_art,
                        "malformed": malformed,
                        "repetitive": repetitive,
                        "clean": clean,
                        "generated_text": generated_text,
                    }
                )

    with (out_dir / "benchmark_records.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    summary: dict[str, Any] = {
        "timestamp": timestamp,
        "experiment": args.experiment_name,
        "model": model_name,
        "locked_state_path": str(state_path.relative_to(REPO_ROOT)),
        "train_per_group": args.train_per_group,
        "test_per_group": args.test_per_group,
        "split_seed": args.split_seed,
        "condition_names": [condition["name"] for condition in conditions],
        "generation_seeds": list(args.generation_seeds),
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "conditions": {},
        "verdict": {},
    }

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["prompt_mode"], record["condition"])].append(record)

    for mode in ("baseline", "recursive"):
        for condition in conditions:
            key = (mode, condition["name"])
            rows = grouped.get(key, [])
            if rows:
                summary["conditions"][f"{mode}::{condition['name']}"] = condition_summary(rows)

    baseline_rows = {
        name: summary["conditions"][f"baseline::{name}"]
        for name in [condition["name"] for condition in conditions]
        if f"baseline::{name}" in summary["conditions"]
    }
    recursive_rows = {
        name: summary["conditions"][f"recursive::{name}"]
        for name in [condition["name"] for condition in conditions]
        if f"recursive::{name}" in summary["conditions"]
    }

    control_recursive_bt = recursive_rows["control"]["bt_art_rate"]
    control_baseline_bt = baseline_rows["control"]["bt_art_rate"]

    selectivity = {}
    for condition in [condition["name"] for condition in conditions if condition["name"] != "control"]:
        rec_row = recursive_rows[condition]
        base_row = baseline_rows[condition]
        recursive_drop = control_recursive_bt - rec_row["bt_art_rate"]
        baseline_drop = control_baseline_bt - base_row["bt_art_rate"]
        selectivity_score = recursive_drop - baseline_drop - 0.5 * (
            rec_row["malformed_rate"] + base_row["malformed_rate"]
        )
        selectivity[condition] = {
            "recursive_drop_bt_art": float(recursive_drop),
            "baseline_drop_bt_art": float(baseline_drop),
            "recursive_malformed_rate": rec_row["malformed_rate"],
            "baseline_malformed_rate": base_row["malformed_rate"],
            "selectivity_score": float(selectivity_score),
        }

    best_selective = max(selectivity, key=lambda name: selectivity[name]["selectivity_score"])

    summary["verdict"] = {
        "control_recursive_bt_art": control_recursive_bt,
        "control_baseline_bt_art": control_baseline_bt,
        "best_selective_break_name": best_selective,
        "best_selective_break": selectivity[best_selective],
        "selectivity": selectivity,
        "control_recursive_malformed_rate": recursive_rows["control"]["malformed_rate"],
        "control_baseline_malformed_rate": baseline_rows["control"]["malformed_rate"],
        "best_recursive_non_malformed_bt_art_name": min(
            [condition["name"] for condition in conditions if condition["name"] != "control"],
            key=lambda name: recursive_rows[name]["non_malformed_bt_art_rate"],
        ),
        "best_recursive_non_malformed_bt_art": min(
            recursive_rows[name]["non_malformed_bt_art_rate"]
            for name in recursive_rows
            if name != "control"
        ),
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
