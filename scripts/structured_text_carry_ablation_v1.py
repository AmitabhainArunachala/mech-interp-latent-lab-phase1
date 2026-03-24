#!/usr/bin/env python3
"""
Structured text-carry ablation seeded from selected baseline inductions.

Goal:
- test whether the strongest selected-seed persistence signal depends on exact
  prior surface text, shorter carry, or merely the fixed turn schedule
- keep the dialogue scaffold fixed while varying how much of the previous
  response is carried into the next turn
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.induced_persistence_followup import load_source_records, select_seed_records
from scripts.induced_persistence_unselected_seed_v1 import TURN_SCHEDULE_V1, build_prompt
from scripts.self_feeding_loop import (
    classify_output,
    compute_prefill_metrics,
    generate_turn,
    make_turn_segments,
    summarize_turn_slice,
)
from src.core.models import set_seed


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE_RUN_DIR = "results/anchor_layermatched_protocol_confirm_v1/20260316_092017"
DEFAULT_TARGET_CONDITION = "anchor_layermatched_low_bridge_3"
DEFAULT_CONTROL_CONDITION = "control"
DEFAULT_BASELINE_GROUPS = ["baseline"]


def parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def first_sentence(text: str) -> str:
    pieces = re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)
    return pieces[0].strip() if pieces and pieces[0].strip() else text.strip()


def scramble_words(text: str, seed: int) -> str:
    words = text.split()
    if len(words) < 4:
        return text
    rng = random.Random(seed)
    rng.shuffle(words)
    return " ".join(words)


def transform_context(mode: str, previous_response: str, *, seed: int) -> str:
    previous_response = previous_response.strip()
    if mode == "exact":
        return previous_response
    if mode == "last256":
        return previous_response[-256:].strip()
    if mode == "first_sentence":
        return first_sentence(previous_response)
    if mode == "scramble_words":
        return scramble_words(previous_response, seed)
    if mode == "instruction_only":
        return ""
    raise ValueError(f"Unsupported carry mode: {mode}")


def summarize_sessions(sessions: list[dict[str, Any]]) -> dict[str, Any]:
    all_turns = [turn for session in sessions for turn in session["turns"]]
    max_turns = max((len(session["turns"]) for session in sessions), default=0)
    prompt_category_counts = Counter(turn["turn_prompt_category"] for turn in all_turns)
    segment_stats = {}
    for seg_name, start, end in make_turn_segments(max_turns):
        seg_turns = [turn for turn in all_turns if start <= turn["turn"] < end]
        segment_stats[f"{seg_name}_{start}_{end-1}"] = summarize_turn_slice(seg_turns)

    return {
        "n_sessions": len(sessions),
        "n_turns": len(all_turns),
        "bt_art_rate": float(np.mean([turn["bt_art"] for turn in all_turns])) if all_turns else 0.0,
        "repetitive_rate": float(np.mean([turn["repetitive"] for turn in all_turns])) if all_turns else 0.0,
        "clean_rate": float(np.mean([turn["clean"] for turn in all_turns])) if all_turns else 0.0,
        "mean_prompt_rv": float(np.nanmean([turn["prompt_rv"] for turn in all_turns])) if all_turns else float("nan"),
        "mean_output_rv": float(np.nanmean([turn["output_rv"] for turn in all_turns])) if all_turns else float("nan"),
        "session_bt_art_rates": [
            float(np.mean([turn["bt_art"] for turn in session["turns"]]))
            for session in sessions
        ],
        "segment_stats": segment_stats,
        "turn_prompt_category_counts": dict(prompt_category_counts),
        "source_group_counts": dict(Counter(session["source_group"] for session in sessions)),
        "source_condition_counts": dict(Counter(session["source_condition"] for session in sessions)),
    }


def run_seeded_session(
    *,
    model: Any,
    tokenizer: Any,
    seed_record: dict[str, Any],
    carry_mode: str,
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

    raw_context = seed_record["generated_text"]
    turns: list[dict[str, Any]] = []

    for turn_idx in range(max_turns):
        carry_context = transform_context(carry_mode, raw_context, seed=session_seed + turn_idx)
        prompt_text, prompt_meta = build_prompt(carry_context, turn_idx)
        if not prompt_text.strip():
            prompt_text = TURN_SCHEDULE_V1[turn_idx][1]

        prompt_metrics = compute_prefill_metrics(model, tokenizer, prompt_text, 5, 27, device)
        prompt_rv = prompt_metrics["rv"] if prompt_metrics else float("nan")

        response = generate_turn(
            model,
            tokenizer,
            prompt_text,
            max_tokens=max_new_tokens,
            temp=temperature,
            rep_penalty=rep_penalty,
            device=device,
        )

        output_metrics = compute_prefill_metrics(model, tokenizer, response, 5, 27, device)
        output_rv = output_metrics["rv"] if output_metrics else float("nan")
        classification = classify_output(response, output_rv)
        rv_delta = (
            output_rv - prompt_rv
            if not (np.isnan(output_rv) or np.isnan(prompt_rv))
            else float("nan")
        )

        turns.append(
            {
                "turn": turn_idx,
                "carry_mode": carry_mode,
                "prompt_rv": prompt_rv,
                "output_rv": output_rv,
                "rv_delta": rv_delta,
                "classification": classification,
                "clean": int(classification not in ("REPETITIVE", "MALFORMED")),
                "bt_art": int(classification in ("BREAKTHROUGH", "ARTICULATE")),
                "repetitive": int(classification == "REPETITIVE"),
                "prompt_text": prompt_text,
                "carry_context": carry_context,
                "response": response,
                **prompt_meta,
            }
        )

        raw_context = response if response.strip() else raw_context
        if len(raw_context) > 1800:
            raw_context = raw_context[-1800:]

    return {
        "carry_mode": carry_mode,
        "source_condition": seed_record["condition_name"],
        "source_group": seed_record["prompt_group"],
        "source_prompt_id": seed_record["prompt_id"],
        "source_generation_seed": seed_record["generation_seed"],
        "turns": turns,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Structured text-carry ablation")
    parser.add_argument("--source-run-dir", default=DEFAULT_SOURCE_RUN_DIR)
    parser.add_argument("--target-condition", default=DEFAULT_TARGET_CONDITION)
    parser.add_argument("--control-condition", default=DEFAULT_CONTROL_CONDITION)
    parser.add_argument("--baseline-groups", default=",".join(DEFAULT_BASELINE_GROUPS))
    parser.add_argument("--top-k-per-group", type=int, default=8)
    parser.add_argument("--selection-strategy", choices=["top", "random", "median"], default="median")
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-turns", type=int, default=len(TURN_SCHEDULE_V1))
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--rep-penalty", type=float, default=1.35)
    parser.add_argument("--seed", type=int, default=314)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--experiment-name", default="structured_text_carry_ablation_v1")
    args = parser.parse_args()

    if args.max_turns > len(TURN_SCHEDULE_V1):
        raise ValueError(
            f"max_turns={args.max_turns} exceeds schedule length {len(TURN_SCHEDULE_V1)}"
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else REPO_ROOT / f"results/structured_text_carry_ablation_v1/{timestamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    source_run_dir = REPO_ROOT / args.source_run_dir
    baseline_groups = parse_csv_list(args.baseline_groups)

    rows = load_source_records(
        source_run_dir=source_run_dir,
        source_conditions=[args.control_condition, args.target_condition],
        baseline_groups=baseline_groups,
    )

    target_rows = [row for row in rows if row["condition_name"] == args.target_condition]
    control_rows = [row for row in rows if row["condition_name"] == args.control_condition]

    target_seed_rows = select_seed_records(
        target_rows,
        args.top_k_per_group,
        selection_strategy=args.selection_strategy,
        seed=args.seed,
    )
    control_seed_rows = select_seed_records(
        control_rows,
        args.top_k_per_group,
        selection_strategy=args.selection_strategy,
        seed=args.seed,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    condition_specs = [
        ("control_exact", control_seed_rows, "exact"),
        ("target_exact", target_seed_rows, "exact"),
        ("target_last256", target_seed_rows, "last256"),
        ("target_first_sentence", target_seed_rows, "first_sentence"),
        ("target_scramble_words", target_seed_rows, "scramble_words"),
        ("target_instruction_only", target_seed_rows, "instruction_only"),
    ]

    session_records: list[dict[str, Any]] = []
    all_turn_rows: list[dict[str, Any]] = []

    for condition_name, seed_rows, carry_mode in condition_specs:
        for idx, seed_record in enumerate(seed_rows):
            session = run_seeded_session(
                model=model,
                tokenizer=tokenizer,
                seed_record=seed_record,
                carry_mode=carry_mode,
                max_turns=args.max_turns,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                rep_penalty=args.rep_penalty,
                device=args.device,
                session_seed=args.seed + idx,
            )
            session["condition_name"] = condition_name
            session_records.append(session)
            for turn in session["turns"]:
                all_turn_rows.append(
                    {
                        "condition_name": condition_name,
                        "source_condition": session["source_condition"],
                        "source_group": session["source_group"],
                        "source_prompt_id": session["source_prompt_id"],
                        "source_generation_seed": session["source_generation_seed"],
                        **turn,
                    }
                )

    with (out_dir / "turn_records.jsonl").open("w", encoding="utf-8") as handle:
        for row in all_turn_rows:
            handle.write(json.dumps(row) + "\n")

    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for session in session_records:
        by_condition[session["condition_name"]].append(session)

    summary = {
        "timestamp": timestamp,
        "experiment": args.experiment_name,
        "model": args.model,
        "source_run_dir": args.source_run_dir,
        "target_condition": args.target_condition,
        "control_condition": args.control_condition,
        "selection_strategy": args.selection_strategy,
        "top_k_per_group": args.top_k_per_group,
        "max_turns": args.max_turns,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "rep_penalty": args.rep_penalty,
        "conditions": {},
    }
    for condition_name, sessions in by_condition.items():
        summary["conditions"][condition_name] = summarize_sessions(sessions)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
