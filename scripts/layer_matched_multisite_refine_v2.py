#!/usr/bin/env python3
"""
Refinement of the layer-matched multisite experiment.

v1 found that low-dose layer-matched V_PROJ steering beat both control and the
mean-diff multisite comparison, while the additive L25 residual bridge hurt at
the original alpha=3.0 setting. This follow-up isolates that question:

- Does a softer bridge help the winning low-dose layer-matched bundle?
- Does layer-matched still beat mean-diff at the same low-dose geometry?
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from prompts.loader import PromptLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from scripts.layer_matched_multisite_steering import (
    apply_residual_steering,
    apply_vproj_steering,
    classify_output,
    compute_bridge_direction,
    compute_output_rv,
    compute_vproj_vectors,
    generate_with_steering,
)


REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303, 404, 505, 606, 707, 808])
    parser.add_argument("--train-per-group", type=int, default=6)
    parser.add_argument("--test-per-group", type=int, default=4)
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / f"results/layer_matched_multisite_refine_v2/{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    model_name = "mistralai/Mistral-7B-v0.1"

    print("=== Layer-Matched Multisite Refinement v2 ===")
    print(f"Output: {out_dir}")
    print(f"Device: {device}")
    print()

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    print(f"Model loaded: {model_name}")

    loader = PromptLoader()
    rec_groups = ["L3_deeper", "L4_full", "L5_refined"]
    base_groups = ["baseline_math", "baseline_factual", "baseline_creative"]

    train_rec, train_base, test_rec, test_base = [], [], [], []
    for group in rec_groups:
        prompts = loader.get_by_group(group)
        np.random.seed(314)
        np.random.shuffle(prompts)
        train_rec.extend(prompts[:args.train_per_group])
        test_rec.extend(prompts[args.train_per_group:args.train_per_group + args.test_per_group])
    for group in base_groups:
        prompts = loader.get_by_group(group)
        np.random.seed(314)
        np.random.shuffle(prompts)
        train_base.extend(prompts[:args.train_per_group])
        test_base.extend(prompts[args.train_per_group:args.train_per_group + args.test_per_group])

    train_rec_texts = [p["text"] if isinstance(p, dict) else p for p in train_rec]
    train_base_texts = [p["text"] if isinstance(p, dict) else p for p in train_base]

    print(f"Training: {len(train_rec_texts)} recursive, {len(train_base_texts)} baseline")
    print(f"Testing: {len(test_rec)} recursive, {len(test_base)} baseline")

    print("\n--- Computing layer-specific V_PROJ vectors ---")
    vproj_vectors = {}
    for layer_idx in [4, 5, 25, 27]:
        vproj_vectors[layer_idx] = compute_vproj_vectors(
            model,
            tokenizer,
            train_rec_texts,
            train_base_texts,
            layer_idx=layer_idx,
            window=16,
            device=device,
        )

    print("\n--- Computing L25 residual bridge direction ---")
    bridge_direction, bridge_norm, bridge_cosine = compute_bridge_direction(
        model,
        tokenizer,
        train_rec_texts,
        train_base_texts,
        layer_idx=25,
        window=32,
        device=device,
    )

    torch.save(
        {
            "vproj": {
                layer: {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in vecs.items()}
                for layer, vecs in vproj_vectors.items()
            },
            "bridge": {"direction": bridge_direction.cpu(), "norm": bridge_norm, "cosine": bridge_cosine},
        },
        out_dir / "vectors.pt",
    )

    def make_vproj_hook(layer_idx: int, method: str, alpha: float):
        vec = vproj_vectors[layer_idx][method]
        return lambda: apply_vproj_steering(model, layer_idx, vec, alpha)

    def make_bridge_hook(alpha: float):
        return lambda: apply_residual_steering(model, 25, bridge_direction, alpha)

    layermatched_low_hooks = [
        make_vproj_hook(4, "pca_pc1", 1.0),
        make_vproj_hook(5, "pca_pc1", 1.0),
        make_vproj_hook(25, "orthogonal_residual", 1.0),
        make_vproj_hook(27, "subspace3_parallel", 2.0),
    ]
    meandiff_low_hooks = [
        make_vproj_hook(4, "mean_diff", 1.0),
        make_vproj_hook(5, "mean_diff", 1.0),
        make_vproj_hook(25, "mean_diff", 1.0),
        make_vproj_hook(27, "mean_diff", 2.0),
    ]

    conditions = [
        {"name": "control", "hooks": []},
        {"name": "bridge_only_1", "hooks": [make_bridge_hook(1.0)]},
        {"name": "bridge_only_2", "hooks": [make_bridge_hook(2.0)]},
        {"name": "bridge_only_3", "hooks": [make_bridge_hook(3.0)]},
        {"name": "layermatched_low", "hooks": layermatched_low_hooks},
        {"name": "layermatched_low_bridge_1", "hooks": layermatched_low_hooks + [make_bridge_hook(1.0)]},
        {"name": "layermatched_low_bridge_2", "hooks": layermatched_low_hooks + [make_bridge_hook(2.0)]},
        {"name": "layermatched_low_bridge_3", "hooks": layermatched_low_hooks + [make_bridge_hook(3.0)]},
        {"name": "meandiff_low", "hooks": meandiff_low_hooks},
        {"name": "meandiff_low_bridge_1", "hooks": meandiff_low_hooks + [make_bridge_hook(1.0)]},
    ]

    test_prompts = []
    for prompt in test_rec:
        text = prompt["text"] if isinstance(prompt, dict) else prompt
        group = prompt.get("group", "recursive") if isinstance(prompt, dict) else "recursive"
        test_prompts.append({"text": text, "mode": "recursive", "group": group})
    for prompt in test_base:
        text = prompt["text"] if isinstance(prompt, dict) else prompt
        group = prompt.get("group", "baseline") if isinstance(prompt, dict) else "baseline"
        test_prompts.append({"text": text, "mode": "baseline", "group": group})

    n_total = len(test_prompts) * len(args.seeds) * len(conditions)
    print(f"\n=== Running {n_total} generations ({len(test_prompts)} prompts × {len(args.seeds)} seeds × {len(conditions)} conditions) ===\n")

    records = []
    done = 0
    t0 = time.time()

    for prompt_index, prompt_record in enumerate(test_prompts):
        for seed in args.seeds:
            for condition in conditions:
                set_seed(seed)
                generated_text, generated_tokens = generate_with_steering(
                    model,
                    tokenizer,
                    prompt_record["text"],
                    condition["hooks"],
                    max_new_tokens=128,
                    temperature=0.7,
                    top_p=0.95,
                    device=device,
                )
                classification = classify_output(generated_text)
                bt_art = classification in ("BREAKTHROUGH", "ARTICULATE")
                output_rv = compute_output_rv(model, tokenizer, generated_text, device=device)
                records.append(
                    {
                        "prompt_index": prompt_index,
                        "prompt_mode": prompt_record["mode"],
                        "prompt_group": prompt_record["group"],
                        "seed": seed,
                        "condition": condition["name"],
                        "classification": classification,
                        "bt_art": int(bt_art),
                        "generated_tokens": generated_tokens,
                        "output_rv": output_rv,
                        "generated_text": generated_text[:500],
                    }
                )
                done += 1
                if done % 50 == 0:
                    elapsed = time.time() - t0
                    rate = done / elapsed
                    remaining = (n_total - done) / rate if rate > 0 else 0.0
                    print(f"  [{done}/{n_total}] {rate:.1f} gen/s, ~{remaining/60:.0f} min remaining")

    with (out_dir / "benchmark_records.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    by_mode_condition = defaultdict(lambda: {"n": 0, "bt": 0, "rep": 0, "rv": []})
    for record in records:
        bucket = by_mode_condition[(record["prompt_mode"], record["condition"])]
        bucket["n"] += 1
        bucket["bt"] += record["bt_art"]
        bucket["rep"] += 1 if record["classification"] == "REPETITIVE" else 0
        if record["output_rv"] is not None:
            bucket["rv"].append(record["output_rv"])

    summary = {"timestamp": timestamp, "model": model_name, "conditions": {}, "verdict": {}}

    for mode in ["recursive", "baseline"]:
        print(f"--- {mode.upper()} prompts ---")
        print(f"{'Condition':32s} {'BT+ART':>8s} {'Rep%':>6s} {'RV':>8s} {'n':>5s}")
        rows = []
        for (bucket_mode, condition_name), stats in sorted(by_mode_condition.items()):
            if bucket_mode != mode:
                continue
            bt_rate = stats["bt"] / stats["n"] if stats["n"] else 0.0
            rep_rate = stats["rep"] / stats["n"] if stats["n"] else 0.0
            mean_rv = float(np.mean(stats["rv"])) if stats["rv"] else None
            rows.append((condition_name, bt_rate, rep_rate, mean_rv, stats["n"]))
            summary["conditions"][f"{mode}::{condition_name}"] = {
                "bt_art_rate": bt_rate,
                "repetitive_rate": rep_rate,
                "mean_output_rv": mean_rv,
                "n": stats["n"],
            }
        rows.sort(key=lambda row: -row[1])
        for condition_name, bt_rate, rep_rate, mean_rv, n in rows:
            rv_str = f"{mean_rv:.4f}" if mean_rv is not None else "N/A"
            print(f"  {condition_name:32s} {bt_rate:7.1%} {rep_rate:5.1%} {rv_str:>8s} {n:5d}")
        print()

    rec = lambda name: summary["conditions"][f"recursive::{name}"]["bt_art_rate"]
    base = lambda name: summary["conditions"][f"baseline::{name}"]["bt_art_rate"]

    bridge_sweep = {
        "control": rec("control"),
        "layermatched_low": rec("layermatched_low"),
        "bridge_only_1": rec("bridge_only_1"),
        "bridge_only_2": rec("bridge_only_2"),
        "bridge_only_3": rec("bridge_only_3"),
        "layermatched_low_bridge_1": rec("layermatched_low_bridge_1"),
        "layermatched_low_bridge_2": rec("layermatched_low_bridge_2"),
        "layermatched_low_bridge_3": rec("layermatched_low_bridge_3"),
    }

    best_recursive_name = max(bridge_sweep, key=bridge_sweep.get)
    summary["verdict"] = {
        "control_recursive_bt_art": rec("control"),
        "control_baseline_bt_art": base("control"),
        "best_recursive_name": best_recursive_name,
        "best_recursive_bt_art": bridge_sweep[best_recursive_name],
        "layermatched_low_recursive_bt_art": rec("layermatched_low"),
        "layermatched_low_baseline_bt_art": base("layermatched_low"),
        "best_bridge_coupled_name": max(
            ["layermatched_low_bridge_1", "layermatched_low_bridge_2", "layermatched_low_bridge_3"],
            key=rec,
        ),
        "best_bridge_coupled_bt_art": max(
            rec("layermatched_low_bridge_1"),
            rec("layermatched_low_bridge_2"),
            rec("layermatched_low_bridge_3"),
        ),
        "soft_bridge_beats_no_bridge": max(
            rec("layermatched_low_bridge_1"),
            rec("layermatched_low_bridge_2"),
            rec("layermatched_low_bridge_3"),
        ) > rec("layermatched_low"),
        "layermatched_beats_meandiff_low": rec("layermatched_low") > rec("meandiff_low"),
        "meandiff_low_recursive_bt_art": rec("meandiff_low"),
        "meandiff_low_baseline_bt_art": base("meandiff_low"),
    }

    print("=== VERDICT ===")
    for key, value in summary["verdict"].items():
        print(f"  {key}: {value}")

    with (out_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, default=str)

    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
