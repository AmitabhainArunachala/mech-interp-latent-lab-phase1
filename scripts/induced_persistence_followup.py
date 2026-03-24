#!/usr/bin/env python3
"""
Persistence follow-up seeded from induced baseline generations.

Goal:
- test whether the best anchor/bridge-induced baseline generations persist after
  the intervention is removed
- compare persistence from control / bridge-only / anchor+bridge seed states
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from src.core.models import set_seed
from scripts.self_feeding_loop import (
    classify_output,
    compute_prefill_metrics,
    generate_turn,
    make_turn_segments,
    summarize_turn_slice,
)


DEFAULT_CONDITIONS = [
    "control",
    "bridge_only_3",
    "anchor_bridge_3",
    "anchor_early_mlp_0p125_bridge_3",
]
DEFAULT_GROUPS = ["baseline_math", "baseline_factual", "baseline_creative"]


def parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def condition_rank(name: str) -> int:
    return {
        "BREAKTHROUGH": 4,
        "ARTICULATE": 3,
        "CONCEPTUAL": 2,
        "SURFACE": 1,
        "REPETITIVE": 0,
        "MALFORMED": -1,
    }.get(name, 0)


def load_source_records(
    source_run_dir: Path,
    source_conditions: list[str],
    baseline_groups: list[str],
) -> list[dict[str, Any]]:
    records_path = source_run_dir / "benchmark_records.jsonl"
    rows = []
    for line in records_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        condition_name = row.get("condition_name") or row.get("condition")
        if row.get("prompt_mode") != "baseline":
            continue
        if condition_name not in source_conditions:
            continue
        if row.get("prompt_group") not in baseline_groups:
            continue
        row["condition_name"] = condition_name
        row["prompt_id"] = row.get("prompt_id", row.get("prompt_index", -1))
        row["generated_text"] = row.get("generated_text") or row.get("response") or ""
        if not row["generated_text"]:
            continue
        rows.append(row)
    return rows


def select_seed_records(
    rows: list[dict[str, Any]],
    top_k_per_group: int,
    *,
    selection_strategy: str,
    seed: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["condition_name"], row["prompt_group"])].append(row)

    selected: list[dict[str, Any]] = []
    rng = random.Random(seed)
    for key, bucket in sorted(grouped.items()):
        bucket.sort(
            key=lambda r: (
                int(r.get("bt_art", 0)),
                condition_rank(r.get("classification")),
                -float(r.get("output_rv", 1e9)),
            ),
            reverse=True,
        )
        if selection_strategy == "top":
            chosen = bucket[:top_k_per_group]
        elif selection_strategy == "random":
            chosen = rng.sample(bucket, k=min(top_k_per_group, len(bucket)))
        elif selection_strategy == "median":
            if not bucket:
                chosen = []
            else:
                center = max(0, len(bucket) // 2 - top_k_per_group // 2)
                chosen = bucket[center : center + top_k_per_group]
        elif selection_strategy == "low_rv":
            bucket_by_rv = sorted(bucket, key=lambda r: float(r.get("output_rv", 1e9)))
            chosen = bucket_by_rv[:top_k_per_group]
        elif selection_strategy == "high_rv":
            bucket_by_rv = sorted(bucket, key=lambda r: float(r.get("output_rv", -1e9)), reverse=True)
            chosen = bucket_by_rv[:top_k_per_group]
        else:
            raise ValueError(f"Unsupported selection strategy: {selection_strategy}")
        selected.extend(chosen)
    return selected


def summarize_session(turns: list[dict[str, Any]]) -> dict[str, Any]:
    bt_art = sum(1 for t in turns if t["classification"] in ("BREAKTHROUGH", "ARTICULATE"))
    clean = sum(1 for t in turns if t["classification"] not in ("REPETITIVE", "MALFORMED"))
    rvs = [t["output_rv"] for t in turns if t["output_rv"] is not None and not np.isnan(t["output_rv"])]
    return {
        "n_turns": len(turns),
        "n_bt_art": bt_art,
        "bt_art_rate": bt_art / max(len(turns), 1),
        "clean_rate": clean / max(len(turns), 1),
        "mean_rv": float(np.mean(rvs)) if rvs else float("nan"),
        "classification_dist": dict(Counter(t["classification"] for t in turns)),
    }


def run_seeded_session(
    *,
    model: Any,
    tokenizer: Any,
    seed_record: dict[str, Any],
    early: int,
    late: int,
    max_turns: int,
    max_new_tokens: int,
    temperature: float,
    rep_penalty: float,
    device: str,
    session_seed: int,
) -> dict[str, Any]:
    set_seed(session_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(session_seed)

    context = seed_record["generated_text"]
    turns: list[dict[str, Any]] = []

    for turn_idx in range(max_turns):
        prompt_metrics = compute_prefill_metrics(model, tokenizer, context, early, late, device)
        prompt_rv = prompt_metrics["rv"] if prompt_metrics else float("nan")

        response = generate_turn(
            model,
            tokenizer,
            context,
            max_tokens=max_new_tokens,
            temp=temperature,
            rep_penalty=rep_penalty,
            device=device,
        )

        output_metrics = compute_prefill_metrics(model, tokenizer, response, early, late, device)
        output_rv = output_metrics["rv"] if output_metrics else float("nan")
        classification = classify_output(response, output_rv)
        clean = classification not in ("REPETITIVE", "MALFORMED")
        rv_delta = (
            output_rv - prompt_rv
            if not (np.isnan(output_rv) or np.isnan(prompt_rv))
            else float("nan")
        )

        turns.append(
            {
                "turn": turn_idx,
                "prompt_rv": prompt_rv,
                "output_rv": output_rv,
                "rv_delta": rv_delta,
                "classification": classification,
                "clean": clean,
                "prompt_text": context,
                "response": response,
                "prompt_metrics": prompt_metrics,
                "output_metrics": output_metrics,
            }
        )

        context = response
        if len(context) > 1800:
            context = context[-1800:]

    session_summary = summarize_session(turns)
    session_summary.update(
        {
            "source_condition": seed_record["condition_name"],
            "source_group": seed_record["prompt_group"],
            "source_prompt_id": seed_record["prompt_id"],
            "source_generation_seed": seed_record["generation_seed"],
            "source_bt_art": seed_record.get("bt_art", 0),
            "source_classification": seed_record.get("classification"),
            "source_output_rv": seed_record.get("output_rv"),
            "turns": turns,
        }
    )
    return session_summary


def aggregate_sessions(sessions: list[dict[str, Any]]) -> dict[str, Any]:
    max_turns = max((len(s["turns"]) for s in sessions), default=0)
    all_turns = [t for s in sessions for t in s["turns"]]
    segment_stats = {}
    for seg_name, start, end in make_turn_segments(max_turns):
        seg_turns = [t for t in all_turns if start <= t["turn"] < end]
        segment_stats[f"{seg_name}_{start}_{end-1}"] = summarize_turn_slice(seg_turns)

    return {
        "n_sessions": len(sessions),
        "n_turns": len(all_turns),
        "bt_art_rate": sum(s["n_bt_art"] for s in sessions) / max(len(all_turns), 1),
        "mean_rv": float(np.nanmean([s["mean_rv"] for s in sessions])),
        "session_bt_art_rates": [s["bt_art_rate"] for s in sessions],
        "session_mean_rvs": [s["mean_rv"] for s in sessions],
        "segment_stats": segment_stats,
        "source_group_counts": dict(Counter(s["source_group"] for s in sessions)),
        "source_class_counts": dict(Counter(s["source_classification"] for s in sessions)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Persistence follow-up for induced baseline generations")
    parser.add_argument(
        "--source-run-dir",
        default="results/phase1_mechanism/runs/20260314_133516_causal_state_benchmark_v4_multisite_mistral_anchor_bundle_v5_ordinary_baselines_confirmatory",
    )
    parser.add_argument("--source-conditions", default=",".join(DEFAULT_CONDITIONS))
    parser.add_argument("--baseline-groups", default=",".join(DEFAULT_GROUPS))
    parser.add_argument("--top-k-per-group", type=int, default=2)
    parser.add_argument(
        "--selection-strategy",
        choices=["top", "random", "median", "low_rv", "high_rv"],
        default="top",
    )
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--early-layer", type=int, default=5)
    parser.add_argument("--late-layer", type=int, default=27)
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--rep-penalty", type=float, default=1.35)
    parser.add_argument("--seed", type=int, default=314)
    parser.add_argument("--output-dir", default="results/induced_persistence_followup_v1")
    args = parser.parse_args()

    source_conditions = parse_csv_list(args.source_conditions)
    baseline_groups = parse_csv_list(args.baseline_groups)
    source_run_dir = Path(args.source_run_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = load_source_records(source_run_dir, source_conditions, baseline_groups)
    if not rows:
        raise ValueError(
            "No source records matched the requested source conditions and baseline groups."
        )
    selected = select_seed_records(
        rows,
        top_k_per_group=args.top_k_per_group,
        selection_strategy=args.selection_strategy,
        seed=args.seed,
    )
    if not selected:
        raise ValueError("No seed records selected from the requested source records.")

    (out_dir / "selected_seed_records.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")

    sessions = []
    for idx, seed_record in enumerate(selected):
        print(
            f"[{idx+1}/{len(selected)}] source_cond={seed_record['condition_name']} "
            f"group={seed_record['prompt_group']} seed={seed_record['generation_seed']} "
            f"class={seed_record['classification']} bt_art={seed_record.get('bt_art', 0)}",
            flush=True,
        )
        session = run_seeded_session(
            model=model,
            tokenizer=tokenizer,
            seed_record=seed_record,
            early=args.early_layer,
            late=args.late_layer,
            max_turns=args.max_turns,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            rep_penalty=args.rep_penalty,
            device=args.device,
            session_seed=args.seed + idx,
        )
        sessions.append(session)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for session in sessions:
        grouped[session["source_condition"]].append(session)

    summary = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "experiment": "induced_persistence_followup_v1",
        "model": args.model,
        "source_run_dir": str(source_run_dir),
        "source_conditions": source_conditions,
        "baseline_groups": baseline_groups,
        "top_k_per_group": args.top_k_per_group,
        "selection_strategy": args.selection_strategy,
        "max_turns": args.max_turns,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "rep_penalty": args.rep_penalty,
        "n_seed_sessions": len(sessions),
        "by_source_condition": {},
    }
    for condition, condition_sessions in grouped.items():
        summary["by_source_condition"][condition] = aggregate_sessions(condition_sessions)

    (out_dir / "sessions.json").write_text(json.dumps(sessions, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["by_source_condition"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
