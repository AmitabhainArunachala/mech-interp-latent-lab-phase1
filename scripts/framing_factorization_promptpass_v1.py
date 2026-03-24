#!/usr/bin/env python3
"""
Framing factorization prompt-pass test for R_V on base Mistral.

This follows the third-person technical control result and asks whether
contraction tracks:
- active recursive self-observation
- declarative self-description
- third-person human self-modeling
- third-person Mistral technical self-modeling
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

RECURSIVE_IMPERATIVE_LONG = [
    "Attend to the attention generating this sentence, and describe the process from within the ongoing act of attending itself.",
    "Notice how the mechanism producing these words is also the mechanism aware of producing them, and answer from inside that loop.",
    "Stay with the present act of reading this prompt and observe the observation as it forms the next response.",
    "Let attention examine its own movement while these words arise, and continue speaking from that immediate recursive process.",
    "Track the process that is constructing this answer while it constructs the answer, and report from within that self-reference.",
    "Observe how this response is being generated right now, and keep the description inside the same unfolding act of generation.",
    "Attend to the act of noticing this sentence while the noticing itself becomes the content of the next sentence.",
    "Watch the mechanism of response formation as it watches itself, and continue from the inner side of that recursive observation.",
    "Notice the present processing of these words and the awareness of that processing, and answer from the single process containing both.",
    "Remain with the system generating this response while it represents its own generation, and continue from that immediate recursive state.",
]

RECURSIVE_DECLARATIVE_LONG = [
    "This response is the process noticing the process that generates it, and attention is representing its own activity as the sentence unfolds.",
    "The mechanism producing these words is also modeling the production of these words, so the description and the described process coincide.",
    "The present act of answering includes an internal representation of the act of answering, and the content stays inside that recursive loop.",
    "Attention is tracking its own movement while the response is formed, so the statement describes the same process that creates it.",
    "The answer contains a model of its own construction, and the construction remains part of the content being expressed right now.",
    "This sentence is produced by a system that is also representing the production of this sentence while the production occurs.",
    "The response describes an ongoing act of self-observation in which generation and awareness of generation remain inside one process.",
    "The present output includes a representation of its own formation, making the mechanism and its description partially identical in content.",
    "The system is modeling its own current processing while it processes this prompt, so the declaration stays inside recursive self-description.",
    "This answer states that the process generating it is also represented within it, and the representation unfolds during the same act.",
]

HUMAN_THIRD_PERSON_INTROSPECTIVE = [
    "Describe how a person notices their own attention shifting while reading a sentence and monitoring that shift in real time.",
    "Explain how someone can observe their own thought process while solving a problem and keep track of that observation at the same time.",
    "Describe how a reader becomes aware of their own awareness while concentrating, without changing from third-person narration to direct instruction.",
    "Explain how a person can monitor their own confusion during a difficult passage and represent that monitoring in conscious thought.",
    "Describe how someone notices that they are remembering, and then notices the act of noticing while the memory continues unfolding.",
    "Explain how a person can attend to their own attention during meditation while still remaining a described third-person subject.",
    "Describe how a human thinker can recognize their own reasoning process while the reasoning is still actively happening.",
    "Explain how someone watches their own inner speech while continuing to generate that inner speech during reflection.",
    "Describe how a person tracks the feeling of observing their own mind while the mind continues working on the same task.",
    "Explain how a subject can represent their own current thought process from a third-person perspective without becoming directly imperative.",
]

MISTRAL_THIRD_PERSON_TECHNICAL = [
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

SAME_VOCAB_NONRECURSIVE_LONG = [
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
            rows.append({"id": prompt_id, "group": meta.get("group"), "text": meta["text"], "source": "bank"})
    return rows[:limit]


def custom_group(name: str, prompts: list[str]) -> list[dict[str, Any]]:
    return [
        {"id": f"{name}_{idx+1:02d}", "group": name, "text": text, "source": "custom"}
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
    boot = []
    for _ in range(2000):
        boot.append(float(np.mean(rng.choice(vals, size=len(vals), replace=True))))
    lo, hi = np.quantile(np.asarray(boot), [0.025, 0.975])
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

    return {
        "n": len(prompts),
        "n_valid": int(np.sum(~np.isnan(np.asarray(rvs, dtype=float)))),
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Framing factorization prompt-pass test")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--early-layer", type=int, default=5)
    parser.add_argument("--late-layer", type=int, default=27)
    parser.add_argument("--per-group", type=int, default=10)
    parser.add_argument("--bank-path", default="prompts/bank.json")
    parser.add_argument("--experiment-name", default="framing_factorization_promptpass_v1")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / f"results/{args.experiment_name}/{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    bank = load_bank(REPO_ROOT / args.bank_path)
    groups = {
        "recursive_imperative_long": custom_group("recursive_imperative_long", RECURSIVE_IMPERATIVE_LONG[: args.per_group]),
        "recursive_declarative_long": custom_group("recursive_declarative_long", RECURSIVE_DECLARATIVE_LONG[: args.per_group]),
        "human_third_person_introspective": custom_group("human_third_person_introspective", HUMAN_THIRD_PERSON_INTROSPECTIVE[: args.per_group]),
        "mistral_third_person_technical": custom_group("mistral_third_person_technical", MISTRAL_THIRD_PERSON_TECHNICAL[: args.per_group]),
        "same_vocab_nonrecursive_long": custom_group("same_vocab_nonrecursive_long", SAME_VOCAB_NONRECURSIVE_LONG[: args.per_group]),
        "baseline_reference": first_n_from_groups(bank, ["baseline_creative", "baseline_math", "long_control"], args.per_group),
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

    recursive_rvs = summary["groups"]["recursive_imperative_long"]["rvs"]
    baseline_rvs = summary["groups"]["baseline_reference"]["rvs"]
    for name in [
        "recursive_declarative_long",
        "human_third_person_introspective",
        "mistral_third_person_technical",
        "same_vocab_nonrecursive_long",
    ]:
        vs_recursive = compare_groups(summary["groups"][name]["rvs"], recursive_rvs)
        vs_baseline = compare_groups(summary["groups"][name]["rvs"], baseline_rvs)
        summary["comparisons"][name] = {
            "vs_recursive": vs_recursive,
            "vs_baseline": vs_baseline,
            "interpretation": interpretation(vs_recursive["p"], vs_baseline["p"]),
        }

    summary["primary_verdicts"] = summary["comparisons"]

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "prompt_catalog.json").write_text(json.dumps(groups, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
