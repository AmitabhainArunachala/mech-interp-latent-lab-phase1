#!/usr/bin/env python3
"""
Compare steering inside the learned recursive subspace vs its orthogonal residual.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.pca_vs_mean_steering import (
    aggregate,
    build_split,
    compute_vectors,
    generate_with_optional_steering,
    parse_csv_list,
    parse_float_list,
    parse_int_list,
)
from prompts.loader import PromptLoader
from src.core.models import load_model, set_seed
from src.metrics.behavior_strict import score_behavior_strict
from src.metrics.rv import compute_rv_with_components
from src.utils.persistent_patching_classification import classify_output


DEFAULT_RECURSIVE_GROUPS = ["L3_deeper", "L4_full", "L5_refined"]
DEFAULT_BASELINE_GROUPS = ["baseline_math", "baseline_factual", "baseline_creative"]
DEFAULT_SEEDS = [101, 202, 303]
DEFAULT_ALPHAS = [2.0, 3.0, 4.0]


def normalize(vec: torch.Tensor) -> torch.Tensor:
    return vec / (vec.norm() + 1e-8)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare in-subspace vs orthogonal-residual steering")
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--layer", type=int, default=27)
    parser.add_argument("--early-layer", type=int, default=5)
    parser.add_argument("--late-layer", type=int, default=27)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--recursive-groups", default=",".join(DEFAULT_RECURSIVE_GROUPS))
    parser.add_argument("--baseline-groups", default=",".join(DEFAULT_BASELINE_GROUPS))
    parser.add_argument("--train-per-group", type=int, default=6)
    parser.add_argument("--test-per-group", type=int, default=6)
    parser.add_argument("--generation-seeds", default=",".join(str(x) for x in DEFAULT_SEEDS))
    parser.add_argument("--alphas", default=",".join(str(x) for x in DEFAULT_ALPHAS))
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=314)
    parser.add_argument("--output-dir", default="results/subspace_component_steering_v1")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    recursive_groups = parse_csv_list(args.recursive_groups)
    baseline_groups = parse_csv_list(args.baseline_groups)
    generation_seeds = parse_int_list(args.generation_seeds)
    alphas = parse_float_list(args.alphas)

    set_seed(args.seed)
    model, tokenizer = load_model(args.model, device=args.device)
    model.eval()
    loader = PromptLoader()

    split = build_split(
        loader,
        recursive_groups=recursive_groups,
        baseline_groups=baseline_groups,
        train_per_group=args.train_per_group,
        test_per_group=args.test_per_group,
        seed=args.seed,
    )

    vectors, vector_meta = compute_vectors(
        model=model,
        tokenizer=tokenizer,
        train_recursive=split.train_recursive,
        train_baseline=split.train_baseline,
        layer_idx=args.layer,
        device=args.device,
        window=args.window,
    )

    mean_diff = vectors["mean_diff"]
    parallel = vectors["pca_subspace3_meanproj"]
    orthogonal = mean_diff - parallel
    orth_norm = float(orthogonal.norm().cpu())
    if orth_norm > 1e-6:
        orthogonal = normalize(orthogonal)
    else:
        orthogonal = torch.zeros_like(mean_diff)

    method_vectors: dict[str, torch.Tensor | None] = {
        "control": None,
        "mean_diff": mean_diff,
        "subspace3_parallel": parallel,
        "orthogonal_residual": orthogonal,
        "pca_pc1": vectors["pca_pc1"],
    }

    torch.save(
        {k: (v.cpu() if v is not None else None) for k, v in method_vectors.items()},
        out_dir / "vectors.pt",
    )

    records: list[dict[str, Any]] = []
    prompt_modes = [
        ("baseline", split.test_baseline),
        ("recursive", split.test_recursive),
    ]

    methods: list[tuple[str, torch.Tensor | None, float]] = [("control", None, 0.0)]
    for method_name, vector in method_vectors.items():
        if method_name == "control":
            continue
        for alpha in alphas:
            methods.append((method_name, vector, alpha))

    total_jobs = sum(len(prompts) for _, prompts in prompt_modes) * len(generation_seeds) * len(methods)
    job_idx = 0

    for prompt_mode, prompts in prompt_modes:
        for prompt_idx, prompt in enumerate(prompts):
            for generation_seed in generation_seeds:
                for method_name, vector, alpha in methods:
                    job_idx += 1
                    print(
                        f"[{job_idx}/{total_jobs}] mode={prompt_mode} prompt={prompt_idx+1}/{len(prompts)} "
                        f"seed={generation_seed} method={method_name} alpha={alpha:+.3f}",
                        flush=True,
                    )
                    generated = generate_with_optional_steering(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=prompt,
                        layer_idx=args.layer,
                        steering_vector=vector,
                        alpha=alpha,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        device=args.device,
                        seed=generation_seed,
                    )
                    output_rv, _, _ = compute_rv_with_components(
                        model,
                        tokenizer,
                        generated,
                        early=args.early_layer,
                        late=args.late_layer,
                        window=args.window,
                        device=args.device,
                    )
                    classification = classify_output(generated, output_rv)
                    strict = score_behavior_strict(generated)
                    records.append(
                        {
                            "prompt_mode": prompt_mode,
                            "prompt_index": prompt_idx,
                            "generation_seed": generation_seed,
                            "method": method_name,
                            "alpha": alpha,
                            "prompt_text": prompt,
                            "generated_text": generated,
                            "output_rv": output_rv,
                            "classification": classification,
                            "strict": strict.to_dict(),
                        }
                    )

    (out_dir / "records.json").write_text(json.dumps(records, indent=2), encoding="utf-8")

    summary: dict[str, Any] = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "experiment": "subspace_component_steering_v1",
        "model": args.model,
        "layer": args.layer,
        "early_layer": args.early_layer,
        "late_layer": args.late_layer,
        "window": args.window,
        "recursive_groups": recursive_groups,
        "baseline_groups": baseline_groups,
        "train_per_group": args.train_per_group,
        "test_per_group": args.test_per_group,
        "generation_seeds": generation_seeds,
        "alphas": alphas,
        "vector_metadata": vector_meta,
        "decomposition": {
            "parallel_norm": float(parallel.norm().cpu()),
            "orthogonal_norm": orth_norm,
            "parallel_fraction_of_mean": float(torch.dot(mean_diff, parallel).cpu()),
            "orthogonal_cosine_to_mean": float(torch.dot(mean_diff, orthogonal).cpu()) if orth_norm > 1e-6 else 0.0,
        },
        "by_mode_method_alpha": {},
        "effects_vs_control": {},
        "winners": {},
    }

    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        key = f"{record['prompt_mode']}::{record['method']}::{record['alpha']}"
        grouped.setdefault(key, []).append(record)

    for key, group_records in grouped.items():
        summary["by_mode_method_alpha"][key] = aggregate(group_records)

    for method_name, vector in method_vectors.items():
        if method_name == "control":
            continue
        for alpha in alphas:
            base_key = f"baseline::{method_name}::{alpha}"
            rec_key = f"recursive::{method_name}::{alpha}"
            base_control = summary["by_mode_method_alpha"]["baseline::control::0.0"]
            rec_control = summary["by_mode_method_alpha"]["recursive::control::0.0"]
            base = summary["by_mode_method_alpha"][base_key]
            rec = summary["by_mode_method_alpha"][rec_key]
            summary["effects_vs_control"][f"{method_name}::{alpha}"] = {
                "baseline_bt_art_delta": base["bt_art_rate"] - base_control["bt_art_rate"],
                "recursive_bt_art_delta": rec["bt_art_rate"] - rec_control["bt_art_rate"],
                "baseline_rv_delta": base["mean_output_rv"] - base_control["mean_output_rv"],
                "recursive_rv_delta": rec["mean_output_rv"] - rec_control["mean_output_rv"],
            }

    for prompt_mode in ("baseline", "recursive"):
        candidates = []
        for method_name, _vector, alpha in methods:
            key = f"{prompt_mode}::{method_name}::{alpha}"
            agg = summary["by_mode_method_alpha"].get(key) or {}
            if agg:
                candidates.append((method_name, alpha, agg))
        candidates.sort(
            key=lambda item: (
                item[2].get("bt_art_rate", 0.0),
                item[2].get("strict_pass_rate", 0.0),
                -(item[2].get("malformed_rate", 1.0)),
                -(item[2].get("repetitive_rate", 1.0)),
            ),
            reverse=True,
        )
        if candidates:
            method_name, alpha, agg = candidates[0]
            summary["winners"][prompt_mode] = {
                "method": method_name,
                "alpha": alpha,
                **agg,
            }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["winners"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
