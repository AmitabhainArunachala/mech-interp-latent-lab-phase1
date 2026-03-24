#!/usr/bin/env python3
"""
Compare early-layer subspace steering methods when coupled to the L25 bridge.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

import torch

from prompts.loader import PromptLoader

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

PCA_MODULE_PATH = SCRIPT_DIR / "pca_vs_mean_steering.py"
PCA_SPEC = importlib.util.spec_from_file_location("pca_vs_mean_steering_local", PCA_MODULE_PATH)
if PCA_SPEC is None or PCA_SPEC.loader is None:
    raise RuntimeError(f"Unable to load helper module at {PCA_MODULE_PATH}")
PCA_MODULE = importlib.util.module_from_spec(PCA_SPEC)
PCA_SPEC.loader.exec_module(PCA_MODULE)

aggregate = PCA_MODULE.aggregate
build_split = PCA_MODULE.build_split
compute_vectors = PCA_MODULE.compute_vectors
parse_csv_list = PCA_MODULE.parse_csv_list
parse_int_list = PCA_MODULE.parse_int_list
from src.core.models import load_model, set_seed
from src.metrics.behavior_strict import score_behavior_strict
from src.metrics.rv import compute_rv_with_components
from src.pipelines.archive.steering import SteeringVectorPatcher
from src.utils.persistent_patching_classification import classify_output


DEFAULT_RECURSIVE_GROUPS = ["L3_deeper", "L4_full", "L5_refined"]
DEFAULT_BASELINE_GROUPS = ["baseline_math", "baseline_factual", "baseline_creative"]
DEFAULT_SEEDS = [101, 202, 303]


def normalize(vec: torch.Tensor) -> torch.Tensor:
    return vec / (vec.norm() + 1e-8)


def generate_with_dual_steering(
    *,
    model: Any,
    tokenizer: Any,
    prompt: str,
    early_layer: int,
    early_vector: torch.Tensor | None,
    early_alpha: float,
    bridge_layer: int,
    bridge_vector: torch.Tensor | None,
    bridge_alpha: float,
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

    patchers: list[SteeringVectorPatcher] = []
    try:
        if early_vector is not None and abs(early_alpha) > 1e-9:
            ev = early_vector.to(device=device, dtype=model_dtype)
            ep = SteeringVectorPatcher(model, ev, early_alpha)
            ep.register(layer_idx=early_layer)
            patchers.append(ep)
        if bridge_vector is not None and abs(bridge_alpha) > 1e-9:
            bv = bridge_vector.to(device=device, dtype=model_dtype)
            bp = SteeringVectorPatcher(model, bv, bridge_alpha)
            bp.register(layer_idx=bridge_layer)
            patchers.append(bp)
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
        for patcher in reversed(patchers):
            patcher.remove()


def main() -> int:
    parser = argparse.ArgumentParser(description="Bridge-coupled subspace steering on held-out prompts")
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--layer", type=int, default=5)
    parser.add_argument("--bridge-layer", type=int, default=25)
    parser.add_argument("--late-layer", type=int, default=27)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--recursive-groups", default=",".join(DEFAULT_RECURSIVE_GROUPS))
    parser.add_argument("--baseline-groups", default=",".join(DEFAULT_BASELINE_GROUPS))
    parser.add_argument("--train-per-group", type=int, default=6)
    parser.add_argument("--test-per-group", type=int, default=6)
    parser.add_argument("--generation-seeds", default=",".join(str(x) for x in DEFAULT_SEEDS))
    parser.add_argument("--bridge-alpha", type=float, default=3.0)
    parser.add_argument("--early-alpha", type=float, default=2.0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=314)
    parser.add_argument("--output-dir", default="results/bridge_coupled_subspace_steering_v1")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    recursive_groups = parse_csv_list(args.recursive_groups)
    baseline_groups = parse_csv_list(args.baseline_groups)
    generation_seeds = parse_int_list(args.generation_seeds)

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

    early_vectors, early_meta = compute_vectors(
        model=model,
        tokenizer=tokenizer,
        train_recursive=split.train_recursive,
        train_baseline=split.train_baseline,
        layer_idx=args.layer,
        device=args.device,
        window=args.window,
    )
    bridge_vectors, bridge_meta = compute_vectors(
        model=model,
        tokenizer=tokenizer,
        train_recursive=split.train_recursive,
        train_baseline=split.train_baseline,
        layer_idx=args.bridge_layer,
        device=args.device,
        window=args.window,
    )

    parallel = early_vectors["pca_subspace3_meanproj"]
    orthogonal = early_vectors["mean_diff"] - parallel
    orth_norm = float(orthogonal.norm().cpu())
    if orth_norm > 1e-6:
        orthogonal = normalize(orthogonal)
    else:
        orthogonal = torch.zeros_like(early_vectors["mean_diff"])

    method_vectors: dict[str, tuple[torch.Tensor | None, float, torch.Tensor | None, float]] = {
        "control": (None, 0.0, None, 0.0),
        "bridge_only_3": (None, 0.0, bridge_vectors["mean_diff"], args.bridge_alpha),
        "mean_diff_bridge_3": (early_vectors["mean_diff"], args.early_alpha, bridge_vectors["mean_diff"], args.bridge_alpha),
        "pca_pc1_bridge_3": (early_vectors["pca_pc1"], args.early_alpha, bridge_vectors["mean_diff"], args.bridge_alpha),
        "subspace3_parallel_bridge_3": (parallel, args.early_alpha, bridge_vectors["mean_diff"], args.bridge_alpha),
        "orthogonal_residual_bridge_3": (orthogonal, args.early_alpha, bridge_vectors["mean_diff"], args.bridge_alpha),
    }

    records: list[dict[str, Any]] = []
    prompt_modes = [
        ("baseline", split.test_baseline),
        ("recursive", split.test_recursive),
    ]
    total_jobs = sum(len(prompts) for _, prompts in prompt_modes) * len(generation_seeds) * len(method_vectors)
    job_idx = 0

    for prompt_mode, prompts in prompt_modes:
        for prompt_idx, prompt in enumerate(prompts):
            for generation_seed in generation_seeds:
                for method_name, (early_vec, early_alpha, bridge_vec, bridge_alpha) in method_vectors.items():
                    job_idx += 1
                    print(
                        f"[{job_idx}/{total_jobs}] mode={prompt_mode} prompt={prompt_idx+1}/{len(prompts)} "
                        f"seed={generation_seed} method={method_name}",
                        flush=True,
                    )
                    generated = generate_with_dual_steering(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=prompt,
                        early_layer=args.layer,
                        early_vector=early_vec,
                        early_alpha=early_alpha,
                        bridge_layer=args.bridge_layer,
                        bridge_vector=bridge_vec,
                        bridge_alpha=bridge_alpha,
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
                        early=args.layer,
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
        "experiment": "bridge_coupled_subspace_steering_v1",
        "model": args.model,
        "layer": args.layer,
        "bridge_layer": args.bridge_layer,
        "late_layer": args.late_layer,
        "window": args.window,
        "recursive_groups": recursive_groups,
        "baseline_groups": baseline_groups,
        "train_per_group": args.train_per_group,
        "test_per_group": args.test_per_group,
        "generation_seeds": generation_seeds,
        "early_alpha": args.early_alpha,
        "bridge_alpha": args.bridge_alpha,
        "early_vector_metadata": early_meta,
        "bridge_vector_metadata": bridge_meta,
        "by_mode_method": {},
        "effects_vs_control": {},
        "winners": {},
    }

    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        key = f"{record['prompt_mode']}::{record['method']}"
        grouped.setdefault(key, []).append(record)
    for key, group_records in grouped.items():
        summary["by_mode_method"][key] = aggregate(group_records)

    for method_name in method_vectors:
        if method_name == "control":
            continue
        base = summary["by_mode_method"][f"baseline::{method_name}"]
        rec = summary["by_mode_method"][f"recursive::{method_name}"]
        base_control = summary["by_mode_method"]["baseline::control"]
        rec_control = summary["by_mode_method"]["recursive::control"]
        summary["effects_vs_control"][method_name] = {
            "baseline_bt_art_delta": base["bt_art_rate"] - base_control["bt_art_rate"],
            "recursive_bt_art_delta": rec["bt_art_rate"] - rec_control["bt_art_rate"],
            "baseline_rv_delta": base["mean_output_rv"] - base_control["mean_output_rv"],
            "recursive_rv_delta": rec["mean_output_rv"] - rec_control["mean_output_rv"],
        }

    for prompt_mode in ("baseline", "recursive"):
        candidates = []
        for method_name in method_vectors:
            key = f"{prompt_mode}::{method_name}"
            agg = summary["by_mode_method"].get(key) or {}
            if agg:
                candidates.append((method_name, agg))
        candidates.sort(
            key=lambda item: (
                item[1].get("bt_art_rate", 0.0),
                item[1].get("strict_pass_rate", 0.0),
                -(item[1].get("malformed_rate", 1.0)),
                -(item[1].get("repetitive_rate", 1.0)),
            ),
            reverse=True,
        )
        if candidates:
            method_name, agg = candidates[0]
            summary["winners"][prompt_mode] = {"method": method_name, **agg}

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
