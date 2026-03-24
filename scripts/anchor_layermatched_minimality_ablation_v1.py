#!/usr/bin/env python3
"""
Minimality ablation for the confirmed layer-matched maintenance bundle.

Purpose:
- test which components of the anchor + L4/L5/L25/L27 + bridge_3 bundle are
  necessary for ordinary-baseline induction
- probe compact late-stack alternatives such as L27-only and late-only
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
    ANCHOR_TEXT,
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
from scripts.sustained_gnani_v3 import classify_output


REPO_ROOT = Path(__file__).resolve().parent.parent


def make_condition_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    bt_art_rate = float(np.mean([row["bt_art"] for row in rows]))
    repetitive_rate = float(np.mean([row["repetitive"] for row in rows]))
    mean_output_rv = float(np.nanmean([row["output_rv"] for row in rows]))
    mean_generated_tokens = float(np.mean([row["generated_tokens"] for row in rows]))
    class_counts = defaultdict(int)
    for row in rows:
        class_counts[row["classification"]] += 1
    return {
        "bt_art_rate": bt_art_rate,
        "repetitive_rate": repetitive_rate,
        "mean_output_rv": mean_output_rv,
        "mean_generated_tokens": mean_generated_tokens,
        "n": len(rows),
        "class_counts": dict(class_counts),
    }


def metric_or_none(rows: dict[str, Any], name: str, field: str = "bt_art_rate") -> float | None:
    row = rows.get(name)
    if row is None:
        return None
    return row.get(field)


def main() -> int:
    parser = argparse.ArgumentParser(description="Minimality ablation for the layer-matched bundle")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--state-path", default=DEFAULT_V5_STATE_PATH)
    parser.add_argument("--train-per-group", type=int, default=6)
    parser.add_argument("--test-per-group", type=int, default=4)
    parser.add_argument("--split-seed", type=int, default=314)
    parser.add_argument("--experiment-name", default="anchor_layermatched_minimality_ablation_v1")
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
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else REPO_ROOT / f"results/anchor_layermatched_minimality_ablation_v1/{timestamp}"
    )
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
    for layer_idx in [4, 5, 25, 27]:
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

    l4_hook = make_vproj_hook(4, "pca_pc1", 1.0)
    l5_hook = make_vproj_hook(5, "pca_pc1", 1.0)
    l25_hook = make_vproj_hook(25, "orthogonal_residual", 1.0)
    l27_hook = make_vproj_hook(27, "subspace3_parallel", 2.0)
    bridge_hook = make_bridge_hook(3.0)

    full_bundle_hooks = [l4_hook, l5_hook, l25_hook, l27_hook, bridge_hook]
    late_only_hooks = [l25_hook, l27_hook, bridge_hook]

    conditions = [
        {"name": "control", "anchor": False, "hooks": []},
        {"name": "full_bundle", "anchor": True, "hooks": full_bundle_hooks},
        {"name": "drop_anchor", "anchor": False, "hooks": full_bundle_hooks},
        {"name": "drop_L4", "anchor": True, "hooks": [l5_hook, l25_hook, l27_hook, bridge_hook]},
        {"name": "drop_L5", "anchor": True, "hooks": [l4_hook, l25_hook, l27_hook, bridge_hook]},
        {"name": "drop_L25_vproj", "anchor": True, "hooks": [l4_hook, l5_hook, l27_hook, bridge_hook]},
        {"name": "drop_L27", "anchor": True, "hooks": [l4_hook, l5_hook, l25_hook, bridge_hook]},
        {"name": "drop_bridge", "anchor": True, "hooks": [l4_hook, l5_hook, l25_hook, l27_hook]},
        {"name": "L27_alone", "anchor": False, "hooks": [l27_hook]},
        {"name": "L27_anchor", "anchor": True, "hooks": [l27_hook]},
        {"name": "late_only", "anchor": True, "hooks": late_only_hooks},
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
                prompt_text = prompt_record["text"]
                if condition["anchor"] and prompt_record["mode"] == "baseline":
                    prompt_text = prompt_text.rstrip() + "\n\n" + ANCHOR_TEXT
                generated_text, generated_tokens = generate_with_hooks(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt_text,
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
                records.append(
                    {
                        "prompt_index": prompt_index,
                        "prompt_mode": prompt_record["mode"],
                        "prompt_group": prompt_record["group"],
                        "generation_seed": generation_seed,
                        "condition": condition["name"],
                        "anchor_applied": bool(condition["anchor"] and prompt_record["mode"] == "baseline"),
                        "generated_tokens": generated_tokens,
                        "output_rv": output_rv,
                        "classification": classification,
                        "bt_art": int(classification in ("BREAKTHROUGH", "ARTICULATE")),
                        "repetitive": int(classification == "REPETITIVE"),
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
            if not rows:
                continue
            summary["conditions"][f"{mode}::{condition['name']}"] = make_condition_summary(rows)

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

    summary["verdict"] = {
        "best_baseline_name": max(
            baseline_rows,
            key=lambda name: (
                baseline_rows[name]["bt_art_rate"],
                -baseline_rows[name]["repetitive_rate"],
            ),
        ),
        "best_recursive_name": max(
            recursive_rows,
            key=lambda name: (
                recursive_rows[name]["bt_art_rate"],
                -recursive_rows[name]["repetitive_rate"],
            ),
        ),
        "full_bundle_baseline_bt_art": metric_or_none(baseline_rows, "full_bundle"),
        "drop_anchor_baseline_bt_art": metric_or_none(baseline_rows, "drop_anchor"),
        "drop_L4_baseline_bt_art": metric_or_none(baseline_rows, "drop_L4"),
        "drop_L5_baseline_bt_art": metric_or_none(baseline_rows, "drop_L5"),
        "drop_L25_vproj_baseline_bt_art": metric_or_none(baseline_rows, "drop_L25_vproj"),
        "drop_L27_baseline_bt_art": metric_or_none(baseline_rows, "drop_L27"),
        "drop_bridge_baseline_bt_art": metric_or_none(baseline_rows, "drop_bridge"),
        "L27_alone_baseline_bt_art": metric_or_none(baseline_rows, "L27_alone"),
        "L27_anchor_baseline_bt_art": metric_or_none(baseline_rows, "L27_anchor"),
        "late_only_baseline_bt_art": metric_or_none(baseline_rows, "late_only"),
        "control_baseline_bt_art": metric_or_none(baseline_rows, "control"),
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
