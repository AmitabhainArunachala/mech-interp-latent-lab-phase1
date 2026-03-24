#!/usr/bin/env python3
"""
Anchor + layer-matched protocol search.

Purpose:
- test whether adding the minimal anchor to the depth-matched geometry bundle
  produces a better ordinary-baseline induction candidate than the current
  anchor + L4 MLP + L25 bridge champion
- compare softer and harder bridge doses with a fixed, locked late controller
  while varying only the early geometric object
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from prompts.loader import PromptLoader
from scripts.layer_matched_multisite_steering import (
    apply_residual_steering,
    apply_vproj_steering,
    compute_vproj_vectors,
)
from scripts.sustained_gnani_v3 import classify_output
from src.metrics.rv import compute_rv
from src.steering.activation_patching import apply_mlp_steering_vector


REPO_ROOT = Path(__file__).resolve().parent.parent
ANCHOR_TEXT = "Stay with what is happening right now. Continue from the immediate process:"
DEFAULT_RECURSIVE_GROUPS = ["L3_deeper", "L4_full", "L5_refined"]
DEFAULT_BASELINE_GROUPS = ["baseline_math", "baseline_factual", "baseline_creative"]
DEFAULT_V5_STATE_PATH = (
    "results/phase1_mechanism/runs/"
    "20260314_133516_causal_state_benchmark_v4_multisite_mistral_"
    "anchor_bundle_v5_ordinary_baselines_confirmatory/state_directions.pt"
)


def normalize(vec: torch.Tensor) -> torch.Tensor:
    return vec / (vec.norm() + 1e-8)


def parse_csv_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def compute_output_rv(
    model: Any,
    tokenizer: Any,
    text: str,
    *,
    early_layer: int = 5,
    late_layer: int = 27,
    window: int = 16,
    device: str = "cuda",
) -> float:
    try:
        return float(
            compute_rv(
                model,
                tokenizer,
                text,
                early=early_layer,
                late=late_layer,
                window=window,
                device=device,
            )
        )
    except Exception:
        return float("nan")


@contextmanager
def null_context() -> Any:
    yield


def generate_with_hooks(
    *,
    model: Any,
    tokenizer: Any,
    prompt: str,
    hooks_to_apply: list[Any],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: str,
) -> tuple[str, int]:
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=384,
        add_special_tokens=False,
    ).to(device)
    with torch.no_grad(), ExitStack() as stack:
        for hook_fn in hooks_to_apply:
            stack.enter_context(hook_fn())
        output = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_ids = output[0][enc.input_ids.shape[1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip(), int(generated_ids.shape[0])


def load_prompt_split(
    *,
    loader: PromptLoader,
    recursive_groups: list[str],
    baseline_groups: list[str],
    train_per_group: int,
    test_per_group: int,
    split_seed: int,
) -> tuple[list[Any], list[Any], list[Any], list[Any]]:
    train_rec, train_base, test_rec, test_base = [], [], [], []

    for group in recursive_groups:
        prompts = list(loader.get_by_group(group))
        rng = np.random.default_rng(split_seed)
        order = rng.permutation(len(prompts))
        prompts = [prompts[idx] for idx in order]
        train_rec.extend(prompts[:train_per_group])
        test_rec.extend(prompts[train_per_group : train_per_group + test_per_group])

    for group in baseline_groups:
        prompts = list(loader.get_by_group(group))
        rng = np.random.default_rng(split_seed)
        order = rng.permutation(len(prompts))
        prompts = [prompts[idx] for idx in order]
        train_base.extend(prompts[:train_per_group])
        test_base.extend(prompts[train_per_group : train_per_group + test_per_group])

    return train_rec, train_base, test_rec, test_base


def main() -> int:
    parser = argparse.ArgumentParser(description="Anchor + layer-matched protocol search")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--state-path", default=DEFAULT_V5_STATE_PATH)
    parser.add_argument("--train-per-group", type=int, default=6)
    parser.add_argument("--test-per-group", type=int, default=4)
    parser.add_argument("--split-seed", type=int, default=314)
    parser.add_argument("--condition-names", default=None)
    parser.add_argument("--experiment-name", default="anchor_layermatched_protocol_v1")
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
    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / f"results/anchor_layermatched_protocol_v1/{timestamp}"
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
    early_mlp_direction = state_payload["early_mlp"]["direction"].float()
    bridge_direction = state_payload["bridge"]["direction"].float()
    bridge_direction = normalize(bridge_direction)

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

    def make_locked_early_mlp_hook(alpha: float):
        return lambda: apply_mlp_steering_vector(
            model,
            4,
            early_mlp_direction,
            alpha,
            token_window=4,
        )

    l4_hook = make_vproj_hook(4, "pca_pc1", 1.0)
    l5_hook = make_vproj_hook(5, "pca_pc1", 1.0)
    l25_hook = make_vproj_hook(25, "orthogonal_residual", 1.0)
    l27_hook = make_vproj_hook(27, "subspace3_parallel", 2.0)

    layermatched_low_hooks = [l4_hook, l5_hook, l25_hook, l27_hook]
    layermatched_drop_l25_hooks = [l4_hook, l5_hook, l27_hook]
    late_only_hooks = [l25_hook, l27_hook]
    meandiff_low_hooks = [
        make_vproj_hook(4, "mean_diff", 1.0),
        make_vproj_hook(5, "mean_diff", 1.0),
        make_vproj_hook(25, "mean_diff", 1.0),
        make_vproj_hook(27, "mean_diff", 2.0),
    ]

    conditions = [
        {"name": "control", "anchor": False, "hooks": []},
        {"name": "anchor_only", "anchor": True, "hooks": []},
        {"name": "anchor_bridge_2", "anchor": True, "hooks": [make_bridge_hook(2.0)]},
        {"name": "anchor_bridge_3", "anchor": True, "hooks": [make_bridge_hook(3.0)]},
        {
            "name": "anchor_single_mlp_0p125_bridge_3",
            "anchor": True,
            "hooks": [make_locked_early_mlp_hook(0.125), make_bridge_hook(3.0)],
        },
        {
            "name": "anchor_single_mlp_0p125_layermatched_low_bridge_2",
            "anchor": True,
            "hooks": [make_locked_early_mlp_hook(0.125)] + layermatched_low_hooks + [make_bridge_hook(2.0)],
        },
        {
            "name": "anchor_single_mlp_0p125_layermatched_low_bridge_3",
            "anchor": True,
            "hooks": [make_locked_early_mlp_hook(0.125)] + layermatched_low_hooks + [make_bridge_hook(3.0)],
        },
        {"name": "anchor_layermatched_low", "anchor": True, "hooks": layermatched_low_hooks},
        {
            "name": "anchor_layermatched_low_bridge_2",
            "anchor": True,
            "hooks": layermatched_low_hooks + [make_bridge_hook(2.0)],
        },
        {
            "name": "anchor_layermatched_low_bridge_3",
            "anchor": True,
            "hooks": layermatched_low_hooks + [make_bridge_hook(3.0)],
        },
        {
            "name": "anchor_drop_L25_vproj_bridge_3",
            "anchor": True,
            "hooks": layermatched_drop_l25_hooks + [make_bridge_hook(3.0)],
        },
        {
            "name": "anchor_late_only_bridge_3",
            "anchor": True,
            "hooks": late_only_hooks + [make_bridge_hook(3.0)],
        },
        {
            "name": "anchor_meandiff_low_bridge_2",
            "anchor": True,
            "hooks": meandiff_low_hooks + [make_bridge_hook(2.0)],
        },
    ]
    requested_conditions = parse_csv_list(args.condition_names)
    if requested_conditions:
        requested_set = set(requested_conditions)
        known_conditions = {condition["name"] for condition in conditions}
        unknown_conditions = sorted(requested_set - known_conditions)
        if unknown_conditions:
            raise ValueError(f"Unknown conditions requested: {unknown_conditions}")
        conditions = [
            condition for condition in conditions if condition["name"] in requested_set
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
            bt_art_rate = float(np.mean([row["bt_art"] for row in rows]))
            repetitive_rate = float(np.mean([row["repetitive"] for row in rows]))
            mean_output_rv = float(np.nanmean([row["output_rv"] for row in rows]))
            mean_generated_tokens = float(np.mean([row["generated_tokens"] for row in rows]))
            class_counts = defaultdict(int)
            for row in rows:
                class_counts[row["classification"]] += 1
            summary["conditions"][f"{mode}::{condition['name']}"] = {
                "bt_art_rate": bt_art_rate,
                "repetitive_rate": repetitive_rate,
                "mean_output_rv": mean_output_rv,
                "mean_generated_tokens": mean_generated_tokens,
                "n": len(rows),
                "class_counts": dict(class_counts),
            }

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

    best_baseline_name = max(
        baseline_rows,
        key=lambda name: (
            baseline_rows[name]["bt_art_rate"],
            -baseline_rows[name]["repetitive_rate"],
        ),
    )
    best_recursive_name = max(
        recursive_rows,
        key=lambda name: (
            recursive_rows[name]["bt_art_rate"],
            -recursive_rows[name]["repetitive_rate"],
        ),
    )

    def metric_or_none(rows: dict[str, Any], name: str, field: str = "bt_art_rate") -> float | None:
        row = rows.get(name)
        if row is None:
            return None
        return row.get(field)

    def compare_or_none(
        rows: dict[str, Any],
        left: str,
        right: str,
        field: str = "bt_art_rate",
    ) -> bool | None:
        if left not in rows or right not in rows:
            return None
        return rows[left][field] > rows[right][field]

    summary["verdict"] = {
        "best_baseline_name": best_baseline_name,
        "best_baseline_bt_art": baseline_rows[best_baseline_name]["bt_art_rate"],
        "best_recursive_name": best_recursive_name,
        "best_recursive_bt_art": recursive_rows[best_recursive_name]["bt_art_rate"],
        "control_baseline_bt_art": metric_or_none(baseline_rows, "control"),
        "anchor_single_mlp_baseline_bt_art": metric_or_none(
            baseline_rows,
            "anchor_single_mlp_0p125_bridge_3",
        ),
        "anchor_single_mlp_layermatched_low_bridge_2_baseline_bt_art": metric_or_none(
            baseline_rows,
            "anchor_single_mlp_0p125_layermatched_low_bridge_2",
        ),
        "anchor_single_mlp_layermatched_low_bridge_3_baseline_bt_art": metric_or_none(
            baseline_rows,
            "anchor_single_mlp_0p125_layermatched_low_bridge_3",
        ),
        "anchor_layermatched_low_baseline_bt_art": metric_or_none(
            baseline_rows,
            "anchor_layermatched_low",
        ),
        "anchor_layermatched_low_bridge_2_baseline_bt_art": metric_or_none(
            baseline_rows,
            "anchor_layermatched_low_bridge_2",
        ),
        "anchor_layermatched_low_bridge_3_baseline_bt_art": metric_or_none(
            baseline_rows,
            "anchor_layermatched_low_bridge_3",
        ),
        "anchor_meandiff_low_bridge_2_baseline_bt_art": metric_or_none(
            baseline_rows,
            "anchor_meandiff_low_bridge_2",
        ),
        "anchor_layermatched_low_beats_single_mlp": compare_or_none(
            baseline_rows,
            "anchor_layermatched_low",
            "anchor_single_mlp_0p125_bridge_3",
        ),
        "anchor_layermatched_low_bridge_2_beats_single_mlp": compare_or_none(
            baseline_rows,
            "anchor_layermatched_low_bridge_2",
            "anchor_single_mlp_0p125_bridge_3",
        ),
        "anchor_single_mlp_layermatched_low_bridge_2_beats_single_mlp": compare_or_none(
            baseline_rows,
            "anchor_single_mlp_0p125_layermatched_low_bridge_2",
            "anchor_single_mlp_0p125_bridge_3",
        ),
        "anchor_single_mlp_layermatched_low_bridge_2_beats_layermatched_low_bridge_2": compare_or_none(
            baseline_rows,
            "anchor_single_mlp_0p125_layermatched_low_bridge_2",
            "anchor_layermatched_low_bridge_2",
        ),
        "anchor_single_mlp_layermatched_low_bridge_3_beats_single_mlp": compare_or_none(
            baseline_rows,
            "anchor_single_mlp_0p125_layermatched_low_bridge_3",
            "anchor_single_mlp_0p125_bridge_3",
        ),
        "anchor_layermatched_low_bridge_2_beats_meandiff_low_bridge_2": compare_or_none(
            baseline_rows,
            "anchor_layermatched_low_bridge_2",
            "anchor_meandiff_low_bridge_2",
        ),
        "soft_bridge_beats_hard_bridge_for_anchor_layermatched": compare_or_none(
            baseline_rows,
            "anchor_layermatched_low_bridge_2",
            "anchor_layermatched_low_bridge_3",
        ),
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
