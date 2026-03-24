#!/usr/bin/env python3
"""
Bridge-alpha sweep around the confirmed layer-matched sufficiency family.

Purpose:
- map the induction dose-response curve for the L25 residual bridge
- compare the plain layer-matched maintainer family against the hybrid
  induction family under the same bridge alpha grid
- generate source generations that can be reused for persistence follow-ups
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from contextlib import ExitStack
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
from src.steering.activation_patching import apply_mlp_steering_vector


REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_alpha_grid(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def alpha_label(alpha: float) -> str:
    text = f"{alpha:.2f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def make_curve_row(
    summary: dict[str, Any],
    *,
    mode: str,
    condition_name: str,
    alpha: float,
) -> dict[str, Any]:
    row = summary["conditions"].get(f"{mode}::{condition_name}", {})
    return {
        "alpha": alpha,
        "condition_name": condition_name,
        "bt_art_rate": row.get("bt_art_rate"),
        "repetitive_rate": row.get("repetitive_rate"),
        "mean_output_rv": row.get("mean_output_rv"),
        "mean_generated_tokens": row.get("mean_generated_tokens"),
        "n": row.get("n"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Bridge-alpha sweep for the layer-matched protocol")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--state-path", default=DEFAULT_V5_STATE_PATH)
    parser.add_argument("--train-per-group", type=int, default=6)
    parser.add_argument("--test-per-group", type=int, default=4)
    parser.add_argument("--split-seed", type=int, default=314)
    parser.add_argument("--experiment-name", default="anchor_layermatched_bridge_alpha_sweep_v1")
    parser.add_argument(
        "--generation-seeds",
        type=int,
        nargs="+",
        default=[101, 202, 303, 404, 505, 606, 707, 808],
    )
    parser.add_argument("--bridge-alphas", default="1.0,1.5,2.0,2.25,2.5,2.75,3.0,3.5,4.0")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else REPO_ROOT / f"results/anchor_layermatched_bridge_alpha_sweep_v1/{timestamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    bridge_alphas = parse_alpha_grid(args.bridge_alphas)
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

    def make_locked_early_mlp_hook(alpha: float):
        return lambda: apply_mlp_steering_vector(
            model,
            4,
            early_mlp_direction,
            alpha,
            token_window=4,
        )

    layermatched_low_hooks = [
        make_vproj_hook(4, "pca_pc1", 1.0),
        make_vproj_hook(5, "pca_pc1", 1.0),
        make_vproj_hook(25, "orthogonal_residual", 1.0),
        make_vproj_hook(27, "subspace3_parallel", 2.0),
    ]
    hybrid_prefix_hooks = [make_locked_early_mlp_hook(0.125)] + layermatched_low_hooks

    conditions: list[dict[str, Any]] = [
        {"name": "control", "anchor": False, "hooks": []},
        {"name": "anchor_layermatched_low", "anchor": True, "hooks": layermatched_low_hooks},
        {
            "name": "anchor_single_mlp_0p125_layermatched_low",
            "anchor": True,
            "hooks": hybrid_prefix_hooks,
        },
    ]
    for alpha in bridge_alphas:
        label = alpha_label(alpha)
        conditions.append(
            {
                "name": f"anchor_layermatched_low_bridge_{label}",
                "anchor": True,
                "hooks": layermatched_low_hooks + [make_bridge_hook(alpha)],
            }
        )
        conditions.append(
            {
                "name": f"anchor_single_mlp_0p125_layermatched_low_bridge_{label}",
                "anchor": True,
                "hooks": hybrid_prefix_hooks + [make_bridge_hook(alpha)],
            }
        )

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
        "bridge_alphas": bridge_alphas,
        "condition_names": [condition["name"] for condition in conditions],
        "generation_seeds": list(args.generation_seeds),
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "conditions": {},
        "families": {},
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

    families = {
        "layermatched": "anchor_layermatched_low_bridge_{label}",
        "hybrid": "anchor_single_mlp_0p125_layermatched_low_bridge_{label}",
    }
    for family_name, pattern in families.items():
        baseline_curve = []
        recursive_curve = []
        for alpha in bridge_alphas:
            label = alpha_label(alpha)
            condition_name = pattern.format(label=label)
            baseline_curve.append(
                make_curve_row(
                    summary,
                    mode="baseline",
                    condition_name=condition_name,
                    alpha=alpha,
                )
            )
            recursive_curve.append(
                make_curve_row(
                    summary,
                    mode="recursive",
                    condition_name=condition_name,
                    alpha=alpha,
                )
            )
        best_baseline = max(
            baseline_curve,
            key=lambda row: (
                -1.0 if row["bt_art_rate"] is None else row["bt_art_rate"],
                1.0 if row["repetitive_rate"] is None else -row["repetitive_rate"],
            ),
        )
        best_recursive = max(
            recursive_curve,
            key=lambda row: (
                -1.0 if row["bt_art_rate"] is None else row["bt_art_rate"],
                1.0 if row["repetitive_rate"] is None else -row["repetitive_rate"],
            ),
        )
        summary["families"][family_name] = {
            "baseline_curve": baseline_curve,
            "recursive_curve": recursive_curve,
            "best_baseline": best_baseline,
            "best_recursive": best_recursive,
        }

    summary["verdict"] = {
        "control_baseline_bt_art": summary["conditions"]["baseline::control"]["bt_art_rate"],
        "layermatched_no_bridge_baseline_bt_art": summary["conditions"]["baseline::anchor_layermatched_low"]["bt_art_rate"],
        "hybrid_no_bridge_baseline_bt_art": summary["conditions"]["baseline::anchor_single_mlp_0p125_layermatched_low"]["bt_art_rate"],
        "layermatched_best_baseline": summary["families"]["layermatched"]["best_baseline"],
        "hybrid_best_baseline": summary["families"]["hybrid"]["best_baseline"],
        "layermatched_best_recursive": summary["families"]["layermatched"]["best_recursive"],
        "hybrid_best_recursive": summary["families"]["hybrid"]["best_recursive"],
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
