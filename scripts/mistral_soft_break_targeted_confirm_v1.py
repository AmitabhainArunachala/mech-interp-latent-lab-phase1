#!/usr/bin/env python3
"""
Higher-power confirm for the best soft selective break handle.

This takes a chosen late-stack anti condition, scale, and token window and
tests it head-to-head against control with a larger prompt/seed budget.
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
BASE_L25 = -0.5
BASE_L27 = -1.0
BASE_BRIDGE = -1.0


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


def parse_generation_seeds(raw: str | None) -> list[int]:
    if not raw:
        return [101, 202, 303, 404, 505, 606, 707, 808, 909, 1001, 1102, 1203]
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_token_window(raw: str) -> int | None:
    lowered = raw.strip().lower()
    if lowered in {"full", "none", "all"}:
        return None
    return int(lowered)


def main() -> int:
    parser = argparse.ArgumentParser(description="Higher-power confirm for the best soft break handle")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--state-path", default=DEFAULT_V5_STATE_PATH)
    parser.add_argument("--train-per-group", type=int, default=6)
    parser.add_argument("--test-per-group", type=int, default=6)
    parser.add_argument("--split-seed", type=int, default=314)
    parser.add_argument("--experiment-name", default="mistral_soft_break_targeted_confirm_v1")
    parser.add_argument("--condition-name", required=True)
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--token-window", default="full")
    parser.add_argument("--generation-seeds", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    args = parser.parse_args()

    token_window = parse_token_window(args.token_window)
    generation_seeds = parse_generation_seeds(args.generation_seeds)

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
        return lambda: apply_vproj_steering(model, layer_idx, vec, alpha, token_window=token_window)

    def make_bridge_hook(alpha: float):
        return lambda: apply_residual_steering(model, 25, bridge_direction, alpha, token_window=token_window)

    scale = args.scale
    hook_bank = {
        "anti_l25_only": [make_vproj_hook(25, "orthogonal_residual", BASE_L25 * scale)],
        "anti_l27_only": [make_vproj_hook(27, "subspace3_parallel", BASE_L27 * scale)],
        "anti_bridge_only": [make_bridge_hook(BASE_BRIDGE * scale)],
        "anti_l25_l27": [
            make_vproj_hook(25, "orthogonal_residual", BASE_L25 * scale),
            make_vproj_hook(27, "subspace3_parallel", BASE_L27 * scale),
        ],
        "anti_l25_bridge": [
            make_vproj_hook(25, "orthogonal_residual", BASE_L25 * scale),
            make_bridge_hook(BASE_BRIDGE * scale),
        ],
        "anti_l27_bridge": [
            make_vproj_hook(27, "subspace3_parallel", BASE_L27 * scale),
            make_bridge_hook(BASE_BRIDGE * scale),
        ],
        "anti_late_full": [
            make_vproj_hook(25, "orthogonal_residual", BASE_L25 * scale),
            make_vproj_hook(27, "subspace3_parallel", BASE_L27 * scale),
            make_bridge_hook(BASE_BRIDGE * scale),
        ],
    }
    if args.condition_name not in hook_bank:
        raise ValueError(f"Unknown condition name: {args.condition_name}")

    conditions = [
        {"name": "control", "hooks": []},
        {"name": args.condition_name, "hooks": hook_bank[args.condition_name]},
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
        for generation_seed in generation_seeds:
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
                        "token_window": token_window,
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
        "condition_name": args.condition_name,
        "scale": scale,
        "token_window": token_window,
        "generation_seeds": generation_seeds,
        "conditions": {},
        "verdict": {},
    }

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["prompt_mode"], record["condition"])].append(record)

    for mode in ("baseline", "recursive"):
        for condition in conditions:
            key = (mode, condition["name"])
            summary["conditions"][f"{mode}::{condition['name']}"] = condition_summary(grouped.get(key, []))

    control_recursive = summary["conditions"]["recursive::control"]["bt_art_rate"]
    control_baseline = summary["conditions"]["baseline::control"]["bt_art_rate"]
    target_recursive = summary["conditions"][f"recursive::{args.condition_name}"]["bt_art_rate"]
    target_baseline = summary["conditions"][f"baseline::{args.condition_name}"]["bt_art_rate"]
    target_recursive_malformed = summary["conditions"][f"recursive::{args.condition_name}"]["malformed_rate"]
    target_baseline_malformed = summary["conditions"][f"baseline::{args.condition_name}"]["malformed_rate"]
    summary["verdict"] = {
        "control_recursive_bt_art": control_recursive,
        "control_baseline_bt_art": control_baseline,
        "target_recursive_bt_art": target_recursive,
        "target_baseline_bt_art": target_baseline,
        "recursive_drop_bt_art": control_recursive - target_recursive,
        "baseline_drop_bt_art": control_baseline - target_baseline,
        "target_recursive_malformed_rate": target_recursive_malformed,
        "target_baseline_malformed_rate": target_baseline_malformed,
        "selectivity_score": (
            (control_recursive - target_recursive)
            - (control_baseline - target_baseline)
            - 0.5 * (target_recursive_malformed + target_baseline_malformed)
        ),
    }

    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary["verdict"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
