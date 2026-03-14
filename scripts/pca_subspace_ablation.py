#!/usr/bin/env python3
"""
Projection-ablation follow-up for the recursive control subspace.

Goal:
- test whether removing the learned PCA-style subspace degrades recursive behavior
- compare mean-difference removal against 1D / 2D / 3D PCA subspace removal
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
from src.utils.persistent_patching_classification import classify_output


DEFAULT_RECURSIVE_GROUPS = ["L3_deeper", "L4_full", "L5_refined"]
DEFAULT_BASELINE_GROUPS = ["baseline_math", "baseline_factual", "baseline_creative"]
DEFAULT_SEEDS = [101, 202, 303]
DEFAULT_ALPHAS = [0.5, 1.0]


@dataclass
class SplitPrompts:
    train_recursive: list[str]
    test_recursive: list[str]
    train_baseline: list[str]
    test_baseline: list[str]


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
    def split_group(group_names: list[str]) -> tuple[list[str], list[str]]:
        train: list[str] = []
        test: list[str] = []
        for idx, group_name in enumerate(group_names):
            prompts = loader.get_by_group(
                group_name,
                limit=train_per_group + test_per_group,
                seed=seed + idx,
            )
            train.extend(prompts[:train_per_group])
            test.extend(prompts[train_per_group : train_per_group + test_per_group])
        return train, test

    train_recursive, test_recursive = split_group(recursive_groups)
    train_baseline, test_baseline = split_group(baseline_groups)
    return SplitPrompts(
        train_recursive=train_recursive,
        test_recursive=test_recursive,
        train_baseline=train_baseline,
        test_baseline=test_baseline,
    )


def mean_tail_v(model: Any, tokenizer: Any, text: str, layer_idx: int, device: str, window: int) -> torch.Tensor:
    v = extract_v_activation(model, tokenizer, text, layer_idx=layer_idx, device=device)
    if v.dim() == 3:
        v = v[0]
    w = min(window, v.shape[0])
    return v[-w:, :].float().mean(dim=0)


def normalize(vec: torch.Tensor) -> torch.Tensor:
    return vec / (vec.norm() + 1e-8)


def compute_bases(
    *,
    model: Any,
    tokenizer: Any,
    train_recursive: list[str],
    train_baseline: list[str],
    layer_idx: int,
    device: str,
    window: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    rec_reps = torch.stack([mean_tail_v(model, tokenizer, text, layer_idx, device, window) for text in train_recursive])
    base_reps = torch.stack([mean_tail_v(model, tokenizer, text, layer_idx, device, window) for text in train_baseline])

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

    basis2 = torch.linalg.qr(vh[:2].T.float(), mode="reduced").Q
    basis3 = torch.linalg.qr(vh[:3].T.float(), mode="reduced").Q

    bases = {
        "mean_diff_1d": mean_diff.unsqueeze(1),
        "pca_pc1_1d": pc1.unsqueeze(1),
        "pca_subspace2": basis2,
        "pca_subspace3": basis3,
    }
    meta = {
        "singular_values_top5": svals[:5].cpu().tolist(),
        "basis_dims": {k: int(v.shape[1]) for k, v in bases.items()},
    }
    return bases, meta


class ProjectionAblationPatcher:
    def __init__(self, model: Any, basis: torch.Tensor, alpha: float):
        self.model = model
        self.basis = basis
        self.alpha = alpha
        self.handle: torch.utils.hooks.RemovableHandle | None = None

    def register(self, layer_idx: int) -> None:
        layer = self.model.model.layers[layer_idx].self_attn
        basis = self.basis.to(device=self.model.device, dtype=next(self.model.parameters()).dtype)

        def hook_fn(_module, _inp, out):
            flat = out.reshape(-1, out.shape[-1])
            coeff = flat @ basis
            proj = coeff @ basis.transpose(0, 1)
            patched = flat - self.alpha * proj
            return patched.reshape_as(out)

        self.handle = layer.v_proj.register_forward_hook(hook_fn)

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def generate_with_optional_ablation(
    *,
    model: Any,
    tokenizer: Any,
    prompt: str,
    layer_idx: int,
    basis: torch.Tensor | None,
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

    patcher = None
    try:
        if basis is not None and abs(alpha) > 1e-9:
            patcher = ProjectionAblationPatcher(model, basis, alpha)
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
    repetitive_rate = sum(1 for r in records if r["classification"] == "REPETITIVE") / len(records)
    malformed_rate = sum(1 for r in records if r["classification"] == "MALFORMED") / len(records)
    strict_pass_rate = sum(1 for r in records if r["strict"]["passed_gates"]) / len(records)
    valid_rv = [r["output_rv"] for r in records if r["output_rv"] is not None and not np.isnan(r["output_rv"])]
    return {
        "n": len(records),
        "bt_art_rate": bt_art_rate,
        "repetitive_rate": repetitive_rate,
        "malformed_rate": malformed_rate,
        "strict_pass_rate": strict_pass_rate,
        "mean_output_rv": float(np.mean(valid_rv)) if valid_rv else float("nan"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Projection-ablation comparison for recursive subspace objects")
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
    parser.add_argument("--output-dir", default="results/pca_subspace_ablation")
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

    bases, base_meta = compute_bases(
        model=model,
        tokenizer=tokenizer,
        train_recursive=split.train_recursive,
        train_baseline=split.train_baseline,
        layer_idx=args.layer,
        device=args.device,
        window=args.window,
    )

    torch.save({k: v.cpu() for k, v in bases.items()}, out_dir / "bases.pt")

    prompt_modes = [
        ("baseline", split.test_baseline),
        ("recursive", split.test_recursive),
    ]
    methods: list[tuple[str, torch.Tensor | None, float]] = [("control", None, 0.0)]
    for method_name, basis in bases.items():
        for alpha in alphas:
            methods.append((method_name, basis, alpha))

    total_jobs = sum(len(prompts) for _, prompts in prompt_modes) * len(generation_seeds) * len(methods)
    records: list[dict[str, Any]] = []
    job_idx = 0

    for prompt_mode, prompts in prompt_modes:
        for prompt_idx, prompt in enumerate(prompts):
            for generation_seed in generation_seeds:
                for method_name, basis, alpha in methods:
                    job_idx += 1
                    print(
                        f"[{job_idx}/{total_jobs}] mode={prompt_mode} prompt={prompt_idx+1}/{len(prompts)} "
                        f"seed={generation_seed} method={method_name} alpha={alpha:+.3f}",
                        flush=True,
                    )
                    generated = generate_with_optional_ablation(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=prompt,
                        layer_idx=args.layer,
                        basis=basis,
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
                            "generated_text": generated,
                            "output_rv": output_rv,
                            "classification": classification,
                            "strict": strict.to_dict(),
                        }
                    )

    (out_dir / "records.json").write_text(json.dumps(records, indent=2), encoding="utf-8")

    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
      key = f"{record['prompt_mode']}::{record['method']}::{record['alpha']}"
      grouped.setdefault(key, []).append(record)

    by_mode_method_alpha = {key: aggregate(group) for key, group in grouped.items()}
    control_recursive = by_mode_method_alpha.get("recursive::control::0.0", {})
    control_baseline = by_mode_method_alpha.get("baseline::control::0.0", {})

    effects_vs_control = {}
    for method_name, _basis, alpha in methods:
        if method_name == "control":
            continue
        rec_key = f"recursive::{method_name}::{alpha}"
        bas_key = f"baseline::{method_name}::{alpha}"
        rec = by_mode_method_alpha.get(rec_key, {})
        bas = by_mode_method_alpha.get(bas_key, {})
        effects_vs_control[f"{method_name}::{alpha}"] = {
            "recursive_bt_art_delta": rec.get("bt_art_rate", 0.0) - control_recursive.get("bt_art_rate", 0.0),
            "baseline_bt_art_delta": bas.get("bt_art_rate", 0.0) - control_baseline.get("bt_art_rate", 0.0),
            "recursive_rv_delta": rec.get("mean_output_rv", float("nan")) - control_recursive.get("mean_output_rv", float("nan")),
            "baseline_rv_delta": bas.get("mean_output_rv", float("nan")) - control_baseline.get("mean_output_rv", float("nan")),
        }

    winners = sorted(
        effects_vs_control.items(),
        key=lambda item: (
            -(item[1]["recursive_bt_art_delta"]),
            item[1]["baseline_bt_art_delta"],
            item[1]["recursive_rv_delta"],
        ),
    )

    summary = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "experiment": "pca_subspace_ablation_v1",
        "model": args.model,
        "layer": args.layer,
        "recursive_groups": recursive_groups,
        "baseline_groups": baseline_groups,
        "train_per_group": args.train_per_group,
        "test_per_group": args.test_per_group,
        "generation_seeds": generation_seeds,
        "alphas": alphas,
        "basis_metadata": base_meta,
        "by_mode_method_alpha": by_mode_method_alpha,
        "effects_vs_control": effects_vs_control,
        "winner_by_recursive_drop": winners[:5],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["winner_by_recursive_drop"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
