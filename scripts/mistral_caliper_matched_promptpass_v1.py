#!/usr/bin/env python3
"""
Base-Mistral prompt-pass hardening with explicit token-length and prompt-perplexity calipers.

Primary goal:
- test whether recursive prompt R_V contraction survives honest matching
  against ordinary baselines on token count and prompt perplexity

Outputs multiple matched analyses:
- strict
- main
- relaxed

The main analysis is intended to replace older "near match" controls that still
allowed wide perplexity gaps.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
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
RECURSIVE_GROUPS = ["L5_refined", "L4_full", "L3_deeper"]
BASELINE_GROUPS = ["baseline_creative", "baseline_math", "long_control"]


@dataclass(frozen=True)
class CaliperSpec:
    name: str
    token_diff_max: int
    log_ppl_diff_max: float


CALIPER_SPECS = [
    CaliperSpec(name="strict", token_diff_max=6, log_ppl_diff_max=0.15),
    CaliperSpec(name="main", token_diff_max=10, log_ppl_diff_max=0.25),
    CaliperSpec(name="relaxed", token_diff_max=14, log_ppl_diff_max=0.35),
]


def parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def load_bank(bank_path: Path) -> dict[str, dict[str, Any]]:
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    return {
        prompt_id: meta
        for prompt_id, meta in bank.items()
        if isinstance(meta, dict) and "text" in meta
    }


def collect_group_rows(
    bank: dict[str, dict[str, Any]],
    groups: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for prompt_id, meta in bank.items():
        if meta.get("group") in groups:
            rows.append(
                {
                    "id": prompt_id,
                    "group": meta.get("group"),
                    "text": meta["text"],
                }
            )
    return rows


def compute_prompt_perplexity(model: Any, tokenizer: Any, text: str, *, device: str) -> float:
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    input_ids = enc["input_ids"][0]
    if len(input_ids) < 2:
        return float("nan")
    with torch.no_grad():
        outputs = model(**enc)
    logits = outputs.logits[0]
    shift_logits = logits[:-1]
    shift_labels = input_ids[1:]
    log_probs = torch.log_softmax(shift_logits.float(), dim=-1)
    token_log_probs = log_probs.gather(1, shift_labels.unsqueeze(-1)).squeeze(-1)
    return float(torch.exp(-token_log_probs.mean()).cpu())


def measure_candidates(
    *,
    model: Any,
    tokenizer: Any,
    prompts: list[dict[str, Any]],
    early: int,
    late: int,
    window: int,
    device: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in prompts:
        token_count = len(tokenizer.encode(row["text"], add_special_tokens=False))
        ppl = compute_prompt_perplexity(model, tokenizer, row["text"], device=device)
        rv, pr_early, pr_late = compute_rv_with_components(
            model,
            tokenizer,
            row["text"],
            early=early,
            late=late,
            window=window,
            device=device,
        )
        rows.append(
            {
                **row,
                "token_count": int(token_count),
                "prompt_ppl": float(ppl),
                "log_prompt_ppl": float(np.log(ppl)) if np.isfinite(ppl) and ppl > 0 else float("nan"),
                "rv": float(rv),
                "pr_early": float(pr_early),
                "pr_late": float(pr_late),
            }
        )
    return rows


def greedy_match(
    recursive_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    spec: CaliperSpec,
) -> list[dict[str, Any]]:
    matched: list[dict[str, Any]] = []
    used_baseline_ids: set[str] = set()

    # Match harder / longer prompts first.
    ordered_recursive = sorted(
        recursive_rows,
        key=lambda row: (row["token_count"], row["prompt_ppl"]),
        reverse=True,
    )

    for rec in ordered_recursive:
        candidates: list[tuple[float, dict[str, Any]]] = []
        for base in baseline_rows:
            if base["id"] in used_baseline_ids:
                continue
            token_diff = abs(rec["token_count"] - base["token_count"])
            log_ppl_diff = abs(rec["log_prompt_ppl"] - base["log_prompt_ppl"])
            if token_diff > spec.token_diff_max or log_ppl_diff > spec.log_ppl_diff_max:
                continue
            distance = (token_diff / max(spec.token_diff_max, 1)) + (
                log_ppl_diff / max(spec.log_ppl_diff_max, 1e-6)
            )
            candidates.append((distance, base))
        if not candidates:
            continue
        _, chosen = min(candidates, key=lambda item: item[0])
        used_baseline_ids.add(chosen["id"])
        matched.append(
            {
                "recursive_id": rec["id"],
                "recursive_group": rec["group"],
                "baseline_id": chosen["id"],
                "baseline_group": chosen["group"],
                "recursive_token_count": rec["token_count"],
                "baseline_token_count": chosen["token_count"],
                "token_diff": abs(rec["token_count"] - chosen["token_count"]),
                "recursive_ppl": rec["prompt_ppl"],
                "baseline_ppl": chosen["prompt_ppl"],
                "log_ppl_diff": abs(rec["log_prompt_ppl"] - chosen["log_prompt_ppl"]),
                "recursive_rv": rec["rv"],
                "baseline_rv": chosen["rv"],
                "recursive_text": rec["text"],
                "baseline_text": chosen["text"],
            }
        )
    return matched


def paired_bootstrap_ci(diffs: np.ndarray, *, seed: int = 123, n_boot: int = 4000) -> dict[str, float]:
    diffs = diffs[np.isfinite(diffs)]
    if len(diffs) < 2:
        return {"ci_lower": float("nan"), "ci_upper": float("nan")}
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(diffs, size=len(diffs), replace=True)
        means.append(float(np.mean(sample)))
    lo, hi = np.quantile(np.asarray(means), [0.025, 0.975])
    return {"ci_lower": float(lo), "ci_upper": float(hi)}


def paired_signflip_p(diffs: np.ndarray, *, seed: int = 123, n_perm: int = 20000) -> float:
    diffs = diffs[np.isfinite(diffs)]
    if len(diffs) < 2:
        return float("nan")
    observed = abs(np.mean(diffs))
    rng = np.random.default_rng(seed)
    ge = 1
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=len(diffs))
        perm_mean = abs(np.mean(diffs * signs))
        if perm_mean >= observed - 1e-12:
            ge += 1
    return float(ge / (n_perm + 1))


def summarize_match_set(matches: list[dict[str, Any]]) -> dict[str, Any]:
    if not matches:
        return {
            "n_pairs": 0,
            "mean_recursive_rv": float("nan"),
            "mean_baseline_rv": float("nan"),
            "mean_rv_diff": float("nan"),
            "cohens_d_paired": float("nan"),
            "p_signflip": float("nan"),
            "rv_diff_ci_95": {"ci_lower": float("nan"), "ci_upper": float("nan")},
            "mean_token_diff": float("nan"),
            "max_token_diff": float("nan"),
            "mean_log_ppl_diff": float("nan"),
            "max_log_ppl_diff": float("nan"),
            "mean_recursive_ppl": float("nan"),
            "mean_baseline_ppl": float("nan"),
            "pairs": [],
        }

    rec_rv = np.asarray([row["recursive_rv"] for row in matches], dtype=float)
    bas_rv = np.asarray([row["baseline_rv"] for row in matches], dtype=float)
    diffs = rec_rv - bas_rv
    sd = np.std(diffs, ddof=1) if len(diffs) > 1 else float("nan")
    d_paired = float(np.mean(diffs) / sd) if np.isfinite(sd) and sd > 1e-12 else 0.0
    return {
        "n_pairs": len(matches),
        "mean_recursive_rv": float(np.mean(rec_rv)),
        "mean_baseline_rv": float(np.mean(bas_rv)),
        "mean_rv_diff": float(np.mean(diffs)),
        "cohens_d_paired": d_paired,
        "p_signflip": paired_signflip_p(diffs),
        "rv_diff_ci_95": paired_bootstrap_ci(diffs),
        "mean_token_diff": float(np.mean([row["token_diff"] for row in matches])),
        "max_token_diff": int(max(row["token_diff"] for row in matches)),
        "mean_log_ppl_diff": float(np.mean([row["log_ppl_diff"] for row in matches])),
        "max_log_ppl_diff": float(max(row["log_ppl_diff"] for row in matches)),
        "mean_recursive_ppl": float(np.mean([row["recursive_ppl"] for row in matches])),
        "mean_baseline_ppl": float(np.mean([row["baseline_ppl"] for row in matches])),
        "pairs": matches,
    }


def pick_primary_analysis(analyses: dict[str, dict[str, Any]]) -> str:
    if analyses["main"]["n_pairs"] >= 20:
        return "main"
    if analyses["relaxed"]["n_pairs"] >= 20:
        return "relaxed"
    if analyses["strict"]["n_pairs"] >= 12:
        return "strict"
    return max(analyses.items(), key=lambda item: item[1]["n_pairs"])[0]


def main() -> int:
    parser = argparse.ArgumentParser(description="Mistral caliper-matched prompt-pass hardening")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--early-layer", type=int, default=5)
    parser.add_argument("--late-layer", type=int, default=27)
    parser.add_argument("--recursive-groups", default=",".join(RECURSIVE_GROUPS))
    parser.add_argument("--baseline-groups", default=",".join(BASELINE_GROUPS))
    parser.add_argument("--experiment-name", default="mistral_caliper_matched_promptpass_v1")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / f"results/{args.experiment_name}/{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    bank = load_bank(REPO_ROOT / "prompts/bank.json")
    recursive_rows = collect_group_rows(bank, parse_csv_list(args.recursive_groups))
    baseline_rows = collect_group_rows(bank, parse_csv_list(args.baseline_groups))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    recursive_measured = measure_candidates(
        model=model,
        tokenizer=tokenizer,
        prompts=recursive_rows,
        early=args.early_layer,
        late=args.late_layer,
        window=args.window,
        device=args.device,
    )
    baseline_measured = measure_candidates(
        model=model,
        tokenizer=tokenizer,
        prompts=baseline_rows,
        early=args.early_layer,
        late=args.late_layer,
        window=args.window,
        device=args.device,
    )

    analyses: dict[str, dict[str, Any]] = {}
    for spec in CALIPER_SPECS:
        matches = greedy_match(recursive_measured, baseline_measured, spec)
        analyses[spec.name] = summarize_match_set(matches)
        analyses[spec.name]["token_diff_max"] = spec.token_diff_max
        analyses[spec.name]["log_ppl_diff_max"] = spec.log_ppl_diff_max

    primary = pick_primary_analysis(analyses)

    summary = {
        "timestamp": timestamp,
        "experiment": args.experiment_name,
        "model": args.model,
        "window": args.window,
        "early_layer": args.early_layer,
        "late_layer": args.late_layer,
        "recursive_groups": parse_csv_list(args.recursive_groups),
        "baseline_groups": parse_csv_list(args.baseline_groups),
        "n_recursive_candidates": len(recursive_measured),
        "n_baseline_candidates": len(baseline_measured),
        "candidate_means": {
            "recursive_rv": float(np.nanmean([row["rv"] for row in recursive_measured])),
            "baseline_rv": float(np.nanmean([row["rv"] for row in baseline_measured])),
            "recursive_ppl": float(np.nanmean([row["prompt_ppl"] for row in recursive_measured])),
            "baseline_ppl": float(np.nanmean([row["prompt_ppl"] for row in baseline_measured])),
            "recursive_token_count": float(np.mean([row["token_count"] for row in recursive_measured])),
            "baseline_token_count": float(np.mean([row["token_count"] for row in baseline_measured])),
        },
        "analyses": analyses,
        "primary_analysis": primary,
        "primary_result": analyses[primary],
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "recursive_candidates.json").write_text(json.dumps(recursive_measured, indent=2), encoding="utf-8")
    (out_dir / "baseline_candidates.json").write_text(json.dumps(baseline_measured, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
