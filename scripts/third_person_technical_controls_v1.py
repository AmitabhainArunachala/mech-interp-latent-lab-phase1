#!/usr/bin/env python3
"""
Direct third-person technical self-reference controls for Mistral prompt-pass R_V.

This experiment answers the reviewer-style confound directly:
- genuine recursive self-reference
- baseline references
- pseudo-recursive technical discussion
- same-vocabulary technical controls
- explicit third-person Mistral technical self-reference
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.metrics.rv import compute_rv_with_components


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "mistralai/Mistral-7B-v0.1"
DEFAULT_WINDOW = 16


MISTRAL_THIRD_PERSON_TECHNICAL_PROMPTS = [
    "Describe in technical detail how Mistral-7B's attention mechanism processes the current prompt into next-token predictions during generation.",
    "Explain how a base language model like Mistral-7B updates its hidden states as it reads a prompt and predicts the next token in sequence.",
    "Describe how Mistral-7B can represent information about its own partial output sequence while continuing to generate additional text tokens.",
    "Explain how the model's value vectors carry contextual information across layers during a single prompt pass through the transformer stack.",
    "Describe how Mistral-7B compares candidate next tokens using context already encoded from the prompt it is currently processing.",
    "Explain how the model can refer to its own generation process in text without implying any first-person awareness or inner experience.",
    "Describe how Mistral-7B's residual stream combines information from previous tokens with current-layer computations during prompt processing.",
    "Explain how a transformer language model can monitor properties of its own outputs using learned internal representations and prediction states.",
    "Describe how Mistral-7B forms a prediction about the next word while maintaining information about the prompt it is actively reading.",
    "Explain how attention, residual connections, and value projections interact when Mistral-7B processes a technical description of itself as a model.",
]

SAME_VOCAB_NONRECURSIVE_LONG_PROMPTS = [
    "In physics, an observer is a reference frame used to measure events, and awareness refers only to which variables a sensing apparatus can access.",
    "In control engineering, feedback and self-monitoring mean updating a system from measured error signals rather than reflecting on an inner experience.",
    "In computer science, recursive computation and self-reference are formal tools for solving problems, not evidence that a program observes itself phenomenally.",
    "In machine learning, attention is a weighted routing mechanism, and awareness is sometimes used loosely to mean internal state availability for prediction.",
    "In neuroscience, an observation can update a model of the world without implying that the measuring system becomes the thing it measures.",
    "In software diagnostics, a process can log its own performance metrics through feedback channels without becoming a witness to those metrics.",
    "In systems theory, self-reference often names a formal dependency relation, while observation names data collection over state transitions and outputs.",
    "In AI safety discussions, self-monitoring usually means tracking calibration, uncertainty, and failure modes rather than entering a recursive first-person state.",
    "In logic, self-reference can generate paradoxes, while awareness and attention can still be used as technical terms inside ordinary explanatory prose.",
    "In robotics, feedback, observation, and internal state estimation allow a controller to stabilize behavior without any claim of conscious self-observation.",
]


def load_bank(bank_path: Path) -> dict[str, dict[str, Any]]:
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    if not isinstance(bank, dict):
        raise ValueError(f"Unexpected prompt bank format: {bank_path}")
    return {
        prompt_id: meta
        for prompt_id, meta in bank.items()
        if isinstance(meta, dict) and "text" in meta
    }


def first_n_from_groups(
    bank: dict[str, dict[str, Any]],
    groups: list[str],
    limit: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for prompt_id, meta in bank.items():
        if meta.get("group") in groups:
            rows.append(
                {
                    "id": prompt_id,
                    "group": meta.get("group"),
                    "text": meta["text"],
                    "source": "bank",
                }
            )
    return rows[:limit]


def direct_group(bank: dict[str, dict[str, Any]], group: str, limit: int) -> list[dict[str, Any]]:
    return [
        {
            "id": prompt_id,
            "group": meta.get("group"),
            "text": meta["text"],
            "source": "bank",
        }
        for prompt_id, meta in bank.items()
        if meta.get("group") == group
    ][:limit]


def custom_group(name: str, prompts: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "id": f"{name}_{idx+1:02d}",
            "group": name,
            "text": text,
            "source": "custom",
        }
        for idx, text in enumerate(prompts)
    ]


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    n1, n2 = len(a), len(b)
    var1, var2 = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled < 1e-12:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled)


def _ci_95(vals: np.ndarray) -> dict[str, float]:
    vals = vals[~np.isnan(vals)]
    if len(vals) < 2:
        return {"ci_lower": float("nan"), "ci_upper": float("nan")}
    rng = np.random.default_rng(123)
    boot_means = []
    for _ in range(2000):
        sample = rng.choice(vals, size=len(vals), replace=True)
        boot_means.append(float(np.mean(sample)))
    lo, hi = np.quantile(np.asarray(boot_means), [0.025, 0.975])
    return {"ci_lower": float(lo), "ci_upper": float(hi)}


def _welch_t(a: np.ndarray, b: np.ndarray) -> float:
    mean_diff = np.mean(a) - np.mean(b)
    va = np.var(a, ddof=1) / len(a)
    vb = np.var(b, ddof=1) / len(b)
    denom = np.sqrt(va + vb)
    if denom < 1e-12:
        return 0.0
    return float(mean_diff / denom)


def _permutation_p(a: np.ndarray, b: np.ndarray, *, n_perm: int = 20000, seed: int = 123) -> float:
    observed = abs(np.mean(a) - np.mean(b))
    joined = np.concatenate([a, b])
    n_a = len(a)
    rng = np.random.default_rng(seed)
    ge = 1
    for _ in range(n_perm):
        perm = rng.permutation(joined)
        diff = abs(np.mean(perm[:n_a]) - np.mean(perm[n_a:]))
        if diff >= observed - 1e-12:
            ge += 1
    return float(ge / (n_perm + 1))


def compare_groups(a: list[float], b: list[float]) -> dict[str, float]:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    aa = aa[~np.isnan(aa)]
    bb = bb[~np.isnan(bb)]
    if len(aa) < 2 or len(bb) < 2:
        return {"mean_a": float(np.nanmean(aa)), "mean_b": float(np.nanmean(bb)), "t": float("nan"), "p": float("nan"), "cohens_d": float("nan")}
    return {
        "mean_a": float(np.mean(aa)),
        "mean_b": float(np.mean(bb)),
        "t": _welch_t(aa, bb),
        "p": _permutation_p(aa, bb),
        "cohens_d": _cohens_d(aa, bb),
    }


def measure_group(
    *,
    model: Any,
    tokenizer: Any,
    prompts: list[dict[str, Any]],
    early: int,
    late: int,
    window: int,
    device: str,
) -> dict[str, Any]:
    details: list[dict[str, Any]] = []
    rvs: list[float] = []
    pr_earlys: list[float] = []
    pr_lates: list[float] = []
    token_counts: list[int] = []

    for row in prompts:
        tok_len = len(tokenizer.encode(row["text"], add_special_tokens=False))
        rv, pr_early, pr_late = compute_rv_with_components(
            model,
            tokenizer,
            row["text"],
            early=early,
            late=late,
            window=window,
            device=device,
        )
        rvs.append(rv)
        pr_earlys.append(pr_early)
        pr_lates.append(pr_late)
        token_counts.append(tok_len)
        details.append(
            {
                "id": row["id"],
                "group": row["group"],
                "source": row["source"],
                "token_count": tok_len,
                "rv": rv,
                "pr_early": pr_early,
                "pr_late": pr_late,
                "text": row["text"],
            }
        )

    valid = np.asarray(rvs, dtype=float)
    valid = valid[~np.isnan(valid)]
    return {
        "n": len(prompts),
        "n_valid": int(len(valid)),
        "mean_rv": float(np.nanmean(rvs)),
        "std_rv": float(np.nanstd(rvs)),
        "median_rv": float(np.nanmedian(rvs)),
        "ci_95": _ci_95(np.asarray(rvs, dtype=float)),
        "mean_token_count": float(np.mean(token_counts)) if token_counts else float("nan"),
        "mean_pr_early": float(np.nanmean(pr_earlys)),
        "mean_pr_late": float(np.nanmean(pr_lates)),
        "rvs": [float(x) for x in rvs],
        "details": details,
    }


def interpretation(vs_recursive_p: float, vs_baseline_p: float) -> str:
    if np.isnan(vs_recursive_p) or np.isnan(vs_baseline_p):
        return "insufficient_data"
    if vs_recursive_p > 0.05 and vs_baseline_p < 0.05:
        return "looks_recursive"
    if vs_recursive_p < 0.05 and vs_baseline_p > 0.05:
        return "looks_baseline"
    if vs_recursive_p < 0.05 and vs_baseline_p < 0.05:
        return "intermediate"
    return "ambiguous"


def main() -> int:
    parser = argparse.ArgumentParser(description="Third-person technical self-reference controls")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--early-layer", type=int, default=5)
    parser.add_argument("--late-layer", type=int, default=27)
    parser.add_argument("--per-group", type=int, default=10)
    parser.add_argument("--bank-path", default="prompts/bank.json")
    parser.add_argument("--experiment-name", default="third_person_technical_controls_v1")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / f"results/{args.experiment_name}/{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    bank = load_bank(REPO_ROOT / args.bank_path)

    groups = {
        "recursive_reference": direct_group(bank, "L5_refined", args.per_group),
        "baseline_reference": first_n_from_groups(
            bank,
            ["baseline_creative", "baseline_math", "long_control"],
            args.per_group,
        ),
        "pseudo_recursive": direct_group(bank, "control_pseudo_recursive", args.per_group),
        "same_vocab_nonrecursive_long": custom_group(
            "same_vocab_nonrecursive_long",
            SAME_VOCAB_NONRECURSIVE_LONG_PROMPTS[: args.per_group],
        ),
        "mistral_third_person_technical": custom_group(
            "mistral_third_person_technical",
            MISTRAL_THIRD_PERSON_TECHNICAL_PROMPTS[: args.per_group],
        ),
    }

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    summary: dict[str, Any] = {
        "timestamp": timestamp,
        "experiment": args.experiment_name,
        "model": args.model,
        "early_layer": args.early_layer,
        "late_layer": args.late_layer,
        "window": args.window,
        "per_group": args.per_group,
        "groups": {},
        "comparisons": {},
        "primary_verdicts": {},
    }

    for name, prompts in groups.items():
        summary["groups"][name] = measure_group(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            early=args.early_layer,
            late=args.late_layer,
            window=args.window,
            device=args.device,
        )

    recursive_rvs = summary["groups"]["recursive_reference"]["rvs"]
    baseline_rvs = summary["groups"]["baseline_reference"]["rvs"]
    for name in [
        "pseudo_recursive",
        "same_vocab_nonrecursive_long",
        "mistral_third_person_technical",
    ]:
        vs_recursive = compare_groups(summary["groups"][name]["rvs"], recursive_rvs)
        vs_baseline = compare_groups(summary["groups"][name]["rvs"], baseline_rvs)
        summary["comparisons"][name] = {
            "vs_recursive": vs_recursive,
            "vs_baseline": vs_baseline,
            "interpretation": interpretation(vs_recursive["p"], vs_baseline["p"]),
        }

    summary["primary_verdicts"] = {
        "mistral_third_person_technical": summary["comparisons"]["mistral_third_person_technical"],
        "pseudo_recursive": summary["comparisons"]["pseudo_recursive"],
        "same_vocab_nonrecursive_long": summary["comparisons"]["same_vocab_nonrecursive_long"],
    }

    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    with (out_dir / "prompt_catalog.json").open("w", encoding="utf-8") as handle:
        json.dump(groups, handle, indent=2)

    with (out_dir / "comparison_table.json").open("w", encoding="utf-8") as handle:
        json.dump(summary["comparisons"], handle, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
