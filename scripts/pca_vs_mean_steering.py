#!/usr/bin/env python3
"""
PCA vs mean-difference steering on held-out Mistral prompts.

Goal:
- compare the old mean-difference steering vector against PCA-derived alternatives
- test whether a small learned subspace steers baseline prompts more cleanly

This is a canonical follow-up to the subspace probe and eigenstate analysis.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from prompts.loader import PromptLoader
from src.core.models import load_model, set_seed
from src.core.patching import extract_v_activation
from src.metrics.behavior_strict import score_behavior_strict
from src.metrics.rv import compute_rv_with_components
from src.pipelines.archive.steering import SteeringVectorPatcher
from src.utils.persistent_patching_classification import classify_output


DEFAULT_RECURSIVE_GROUPS = ["L3_deeper", "L4_full", "L5_refined"]
DEFAULT_BASELINE_GROUPS = ["baseline_math", "baseline_factual", "baseline_creative"]
DEFAULT_SEEDS = [101, 202, 303]
DEFAULT_ALPHAS = [1.0, 2.0, 3.0]


@dataclass
class SplitPrompts:
    train_recursive: list[str]
    test_recursive: list[str]
    train_baseline: list[str]
    test_baseline: list[str]
    recursive_counts: dict[str, dict[str, int]]
    baseline_counts: dict[str, dict[str, int]]


def parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def build_split(
    loader: PromptLoader,
    recursive_groups: list[str],
    baseline_groups: list[str],
    train_per_group: int,
    test_per_group: int,
    seed: int,
) -> SplitPrompts:
    def split_group(group_names: list[str]) -> tuple[list[str], list[str], dict[str, dict[str, int]]]:
        train: list[str] = []
        test: list[str] = []
        counts: dict[str, dict[str, int]] = {}
        for idx, group_name in enumerate(group_names):
            prompts = loader.get_by_group(
                group_name,
                limit=train_per_group + test_per_group,
                seed=seed + idx,
            )
            train_slice = prompts[:train_per_group]
            test_slice = prompts[train_per_group : train_per_group + test_per_group]
            train.extend(train_slice)
            test.extend(test_slice)
            counts[group_name] = {
                "train": len(train_slice),
                "test": len(test_slice),
            }
        return train, test, counts

    train_recursive, test_recursive, recursive_counts = split_group(recursive_groups)
    train_baseline, test_baseline, baseline_counts = split_group(baseline_groups)
    return SplitPrompts(
        train_recursive=train_recursive,
        test_recursive=test_recursive,
        train_baseline=train_baseline,
        test_baseline=test_baseline,
        recursive_counts=recursive_counts,
        baseline_counts=baseline_counts,
    )


def mean_tail_v(
    model: Any,
    tokenizer: Any,
    text: str,
    layer_idx: int,
    device: str,
    window: int,
) -> torch.Tensor:
    v = extract_v_activation(model, tokenizer, text, layer_idx=layer_idx, device=device)
    if v.dim() == 3:
        v = v[0]
    w = min(window, v.shape[0])
    return v[-w:, :].float().mean(dim=0)


def normalize(vec: torch.Tensor) -> torch.Tensor:
    return vec / (vec.norm() + 1e-8)


def compute_vectors(
    *,
    model: Any,
    tokenizer: Any,
    train_recursive: list[str],
    train_baseline: list[str],
    layer_idx: int,
    device: str,
    window: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    rec_reps = torch.stack(
        [mean_tail_v(model, tokenizer, text, layer_idx, device, window) for text in train_recursive]
    )
    base_reps = torch.stack(
        [mean_tail_v(model, tokenizer, text, layer_idx, device, window) for text in train_baseline]
    )

    rec_mean = rec_reps.mean(dim=0)
    base_mean = base_reps.mean(dim=0)
    mean_diff = normalize(rec_mean - base_mean)

    n_pairs = min(rec_reps.shape[0], base_reps.shape[0])
    diffs = rec_reps[:n_pairs] - base_reps[:n_pairs]
    diffs_centered = diffs - diffs.mean(dim=0, keepdim=True)
    _, svals, vh = torch.linalg.svd(diffs_centered, full_matrices=False)
    pc1 = vh[0].float()
    if torch.dot(pc1, mean_diff) < 0:
        pc1 = -pc1
    pc1 = normalize(pc1)

    basis2 = vh[:2].T.float()
    proj2 = basis2 @ (basis2.T @ mean_diff)
    proj2 = normalize(proj2)

    basis3 = vh[:3].T.float()
    proj3 = basis3 @ (basis3.T @ mean_diff)
    proj3 = normalize(proj3)

    vectors = {
        "mean_diff": mean_diff,
        "pca_pc1": pc1,
        "pca_subspace2_meanproj": proj2,
        "pca_subspace3_meanproj": proj3,
    }
    metadata = {
        "n_train_recursive": int(rec_reps.shape[0]),
        "n_train_baseline": int(base_reps.shape[0]),
        "n_diff_pairs": int(n_pairs),
        "singular_values_top5": svals[:5].cpu().tolist(),
        "vector_norms": {name: float(vec.norm().cpu()) for name, vec in vectors.items()},
        "vector_cosines": {
            "mean_to_pc1": float(torch.dot(vectors["mean_diff"], vectors["pca_pc1"]).cpu()),
            "mean_to_subspace2proj": float(torch.dot(vectors["mean_diff"], vectors["pca_subspace2_meanproj"]).cpu()),
            "mean_to_subspace3proj": float(torch.dot(vectors["mean_diff"], vectors["pca_subspace3_meanproj"]).cpu()),
            "pc1_to_subspace2proj": float(torch.dot(vectors["pca_pc1"], vectors["pca_subspace2_meanproj"]).cpu()),
            "pc1_to_subspace3proj": float(torch.dot(vectors["pca_pc1"], vectors["pca_subspace3_meanproj"]).cpu()),
            "subspace2proj_to_subspace3proj": float(torch.dot(vectors["pca_subspace2_meanproj"], vectors["pca_subspace3_meanproj"]).cpu()),
        },
    }
    return vectors, metadata


def generate_with_optional_steering(
    *,
    model: Any,
    tokenizer: Any,
    prompt: str,
    layer_idx: int,
    steering_vector: torch.Tensor | None,
    alpha: float,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: str,
    seed: int,
) -> str:
    set_seed(seed)
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    input_len = int(enc["input_ids"].shape[1])
    model_dtype = next(model.parameters()).dtype

    patcher = None
    try:
        if steering_vector is not None and abs(alpha) > 1e-9:
            steering_vector = steering_vector.to(device=device, dtype=model_dtype)
            patcher = SteeringVectorPatcher(model, steering_vector, alpha)
            patcher.register(layer_idx=layer_idx)
        with torch.no_grad():
            outputs = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0.0,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = outputs[0][input_len:]
        return tokenizer.decode(generated, skip_special_tokens=True).strip()
    finally:
        if patcher is not None:
            patcher.remove()


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {}
    bt_art_rate = sum(1 for r in records if r["classification"] in ("BREAKTHROUGH", "ARTICULATE")) / len(records)
    malformed_rate = sum(1 for r in records if r["classification"] == "MALFORMED") / len(records)
    repetitive_rate = sum(1 for r in records if r["classification"] == "REPETITIVE") / len(records)
    breakthrough_rate = sum(1 for r in records if r["classification"] == "BREAKTHROUGH") / len(records)
    articulate_rate = sum(1 for r in records if r["classification"] == "ARTICULATE") / len(records)
    strict_pass_rate = sum(1 for r in records if r["strict"]["passed_gates"]) / len(records)
    mean_final_score = float(np.mean([r["strict"]["final_score"] for r in records]))
    valid_rv = [r["output_rv"] for r in records if r["output_rv"] is not None and not np.isnan(r["output_rv"])]
    return {
        "n": len(records),
        "bt_art_rate": bt_art_rate,
        "malformed_rate": malformed_rate,
        "repetitive_rate": repetitive_rate,
        "breakthrough_rate": breakthrough_rate,
        "articulate_rate": articulate_rate,
        "strict_pass_rate": strict_pass_rate,
        "mean_final_score": mean_final_score,
        "mean_output_rv": float(np.mean(valid_rv)) if valid_rv else float("nan"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare PCA-based vs mean-difference steering")
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--layer", type=int, default=27)
    parser.add_argument("--early-layer", type=int, default=5)
    parser.add_argument("--late-layer", type=int, default=27)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--recursive-groups", default=",".join(DEFAULT_RECURSIVE_GROUPS))
    parser.add_argument("--baseline-groups", default=",".join(DEFAULT_BASELINE_GROUPS))
    parser.add_argument("--train-per-group", type=int, default=8)
    parser.add_argument("--test-per-group", type=int, default=4)
    parser.add_argument("--generation-seeds", default=",".join(str(x) for x in DEFAULT_SEEDS))
    parser.add_argument("--alphas", default=",".join(str(x) for x in DEFAULT_ALPHAS))
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=314)
    parser.add_argument("--output-dir", default="results/pca_vs_mean_steering")
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

    torch.save({k: v.cpu() for k, v in vectors.items()}, out_dir / "vectors.pt")

    records: list[dict[str, Any]] = []
    prompt_modes = [
        ("baseline", split.test_baseline),
        ("recursive", split.test_recursive),
    ]

    methods: list[tuple[str, torch.Tensor | None, float]] = [("control", None, 0.0)]
    for method_name, vector in vectors.items():
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
        "experiment": "pca_vs_mean_steering_v1",
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
        "split_counts": {
            "recursive": split.recursive_counts,
            "baseline": split.baseline_counts,
        },
        "vector_metadata": vector_meta,
        "by_mode_method_alpha": {},
        "winners": {},
    }

    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        key = f"{record['prompt_mode']}::{record['method']}::{record['alpha']}"
        grouped.setdefault(key, []).append(record)

    for key, group_records in grouped.items():
        summary["by_mode_method_alpha"][key] = aggregate(group_records)

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
