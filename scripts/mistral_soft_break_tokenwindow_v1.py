#!/usr/bin/env python3
"""
Token-window localization for the best anti-late soft break.

Uses the same anti-late bundle as the soft break sweep, but restricts the
intervention to the last k tokens so we can see whether the break acts as an
entry perturbation, an ongoing maintenance perturbation, or both.
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


def window_name(token_window: int | None) -> str:
    if token_window is None:
        return "anti_late_full"
    return f"anti_late_last{token_window}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Token-window localization for anti-late soft break")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--state-path", default=DEFAULT_V5_STATE_PATH)
    parser.add_argument("--train-per-group", type=int, default=6)
    parser.add_argument("--test-per-group", type=int, default=4)
    parser.add_argument("--split-seed", type=int, default=314)
    parser.add_argument("--experiment-name", default="mistral_soft_break_tokenwindow_v1")
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--token-windows", default="1,2,4,8,16")
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

    token_windows = [int(item.strip()) for item in args.token_windows.split(",") if item.strip()]

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

    def make_vproj_hook(layer_idx: int, method: str, alpha: float, token_window: int | None):
        vec = vproj_vectors[layer_idx][method]
        return lambda: apply_vproj_steering(model, layer_idx, vec, alpha, token_window=token_window)

    def make_bridge_hook(alpha: float, token_window: int | None):
        return lambda: apply_residual_steering(model, 25, bridge_direction, alpha, token_window=token_window)

    conditions = [{"name": "control", "token_window": None, "hooks": []}]
    conditions.append(
        {
            "name": window_name(None),
            "token_window": None,
            "hooks": [
                make_vproj_hook(25, "orthogonal_residual", BASE_L25 * args.scale, None),
                make_vproj_hook(27, "subspace3_parallel", BASE_L27 * args.scale, None),
                make_bridge_hook(BASE_BRIDGE * args.scale, None),
            ],
        }
    )
    for token_window in token_windows:
        conditions.append(
            {
                "name": window_name(token_window),
                "token_window": token_window,
                "hooks": [
                    make_vproj_hook(25, "orthogonal_residual", BASE_L25 * args.scale, token_window),
                    make_vproj_hook(27, "subspace3_parallel", BASE_L27 * args.scale, token_window),
                    make_bridge_hook(BASE_BRIDGE * args.scale, token_window),
                ],
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
                        "token_window": condition["token_window"],
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
        "scale": args.scale,
        "token_windows": token_windows,
        "generation_seeds": list(args.generation_seeds),
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
    best_name = None
    best_selectivity = -1e9
    ranking = []
    for condition in conditions:
        if condition["name"] == "control":
            continue
        rec = summary["conditions"][f"recursive::{condition['name']}"]
        base = summary["conditions"][f"baseline::{condition['name']}"]
        recursive_drop = control_recursive - rec["bt_art_rate"]
        baseline_drop = control_baseline - base["bt_art_rate"]
        selectivity = recursive_drop - baseline_drop - rec["malformed_rate"]
        ranking.append(
            {
                "condition": condition["name"],
                "token_window": condition["token_window"],
                "recursive_bt_art": rec["bt_art_rate"],
                "baseline_bt_art": base["bt_art_rate"],
                "recursive_drop_bt_art": recursive_drop,
                "baseline_drop_bt_art": baseline_drop,
                "recursive_malformed_rate": rec["malformed_rate"],
                "selectivity_score": selectivity,
            }
        )
        if selectivity > best_selectivity:
            best_selectivity = selectivity
            best_name = condition["name"]

    summary["verdict"] = {
        "control_recursive_bt_art": control_recursive,
        "control_baseline_bt_art": control_baseline,
        "best_condition": best_name,
        "best_selectivity_score": best_selectivity,
        "window_ranking": sorted(ranking, key=lambda row: row["selectivity_score"], reverse=True),
    }

    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary["verdict"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
