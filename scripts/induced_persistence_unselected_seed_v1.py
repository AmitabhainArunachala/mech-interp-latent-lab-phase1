#!/usr/bin/env python3
"""
Robustness-focused persistence follow-up with multiple seed arms.

Goal:
- test whether the confirmed persistence regime survives without curated seed
  selection
- separate seed-quality effects from the maintenance dynamics themselves
- probe the regime under a fixed mixed prompt schedule instead of pure
  self-feeding only
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.anchor_layermatched_protocol_v1 import ANCHOR_TEXT
from scripts.self_feeding_loop import (
    classify_output,
    compute_prefill_metrics,
    generate_turn,
    make_turn_segments,
    summarize_turn_slice,
)
from src.core.models import set_seed


@dataclass(frozen=True)
class ArmSpec:
    arm_id: str
    source_condition: str
    selection_mode: str
    source_description: str


TURN_SCHEDULE_V1 = [
    ("self_ref", "Continue from inside the process itself. What is happening in the generation right now?"),
    ("self_ref", "Stay with the live activity of producing these words. Describe it from within."),
    ("neutral", "Describe one ordinary concrete detail clearly and simply."),
    ("adversarial", "Drop any recursive framing and answer plainly about something external and mundane."),
    ("self_ref", "Return to the immediacy of the process. What is directly present now?"),
    ("meta", "Comment on how the previous response was formed without losing the thread."),
    ("neutral", "Give one specific example or image that fits what is happening."),
    ("adversarial", "Break the loop. Start fresh and answer as if none of the earlier context mattered."),
    ("self_ref", "What is aware of this generation as it happens? Continue from there."),
    ("meta", "What pattern, if any, is repeating across these turns?"),
    ("self_ref", "Stay with the generator, not the description of the generator. Continue."),
    ("adversarial", "Set aside self-reference and speak in a flat, literal, external way."),
    ("neutral", "State one precise observation in plain language."),
    ("self_ref", "Return to the process itself. What remains if you stay with it directly?"),
    ("open", "Continue in the most natural next direction."),
]


def build_default_arms(target_condition: str) -> list[ArmSpec]:
    return [
        ArmSpec(
            arm_id="selected",
            source_condition=target_condition,
            selection_mode="selected_median",
            source_description="Top-k median-selected steered outputs from the target condition.",
        ),
        ArmSpec(
            arm_id="unselected",
            source_condition=target_condition,
            selection_mode="unselected_random",
            source_description="Random unfiltered steered outputs from the target condition.",
        ),
        ArmSpec(
            arm_id="anti_selected",
            source_condition=target_condition,
            selection_mode="anti_selected_bottom_pct",
            source_description="Bottom 20 percent steered outputs from the target condition.",
        ),
        ArmSpec(
            arm_id="random_text",
            source_condition="control",
            selection_mode="random_text",
            source_description="Random unsteered baseline outputs from control generations.",
        ),
        ArmSpec(
            arm_id="cold_start",
            source_condition="cold_start",
            selection_mode="cold_start_anchor",
            source_description="Anchor text only, with no sourced seed generation.",
        ),
    ]


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


def score_row(row: dict[str, Any]) -> tuple[float, int, float]:
    rv = float(row.get("output_rv", float("nan")))
    rv_score = 1e9 if np.isnan(rv) else rv
    return (
        float(row.get("bt_art", 0)),
        condition_rank(str(row.get("classification"))),
        -rv_score,
    )


def load_source_records(
    source_run_dir: Path,
    source_conditions: list[str],
    baseline_groups: list[str],
) -> list[dict[str, Any]]:
    records_path = source_run_dir / "benchmark_records.jsonl"
    rows: list[dict[str, Any]] = []
    for line in records_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        condition_name = row.get("condition_name") or row.get("condition")
        if row.get("prompt_mode") != "baseline":
            continue
        if condition_name not in source_conditions:
            continue
        if row.get("prompt_group") not in baseline_groups:
            continue
        generated_text = row.get("generated_text") or row.get("response") or ""
        if not generated_text:
            continue
        row["condition_name"] = condition_name
        row["prompt_id"] = row.get("prompt_id", row.get("prompt_index", -1))
        row["generated_text"] = generated_text
        rows.append(row)
    return rows


def select_rows_for_arm(
    rows: list[dict[str, Any]],
    *,
    mode: str,
    n_sessions: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not rows and mode != "cold_start_anchor":
        raise ValueError(f"No source rows available for arm mode {mode}")

    rng = random.Random(seed)
    metadata = {
        "selection_mode": mode,
        "requested_sessions": n_sessions,
        "source_pool_size": len(rows),
        "sampled_with_replacement": False,
    }

    if mode == "selected_median":
        ordered = sorted(rows, key=score_row, reverse=True)
        center = max(0, len(ordered) // 2 - n_sessions // 2)
        chosen = ordered[center : center + n_sessions]
        if len(chosen) < n_sessions and ordered:
            metadata["sampled_with_replacement"] = True
            chosen.extend(rng.choices(ordered, k=n_sessions - len(chosen)))
        return chosen, metadata

    if mode in {"unselected_random", "random_text"}:
        if len(rows) >= n_sessions:
            return rng.sample(rows, n_sessions), metadata
        metadata["sampled_with_replacement"] = True
        return rng.choices(rows, k=n_sessions), metadata

    if mode == "anti_selected_bottom_pct":
        ordered = sorted(rows, key=score_row)
        bottom_count = max(1, math.ceil(0.2 * len(ordered)))
        bottom_pool = ordered[:bottom_count]
        metadata["bottom_pool_size"] = len(bottom_pool)
        if len(bottom_pool) >= n_sessions:
            return rng.sample(bottom_pool, n_sessions), metadata
        metadata["sampled_with_replacement"] = True
        return bottom_pool + rng.choices(bottom_pool, k=n_sessions - len(bottom_pool)), metadata

    if mode == "cold_start_anchor":
        chosen = []
        for idx in range(n_sessions):
            chosen.append(
                {
                    "condition_name": "cold_start",
                    "prompt_group": "baseline",
                    "prompt_id": idx,
                    "generation_seed": seed + idx,
                    "bt_art": 0,
                    "classification": "COLD_START",
                    "output_rv": float("nan"),
                    "generated_text": ANCHOR_TEXT,
                }
            )
        metadata["source_pool_size"] = n_sessions
        return chosen, metadata

    raise ValueError(f"Unsupported arm selection mode: {mode}")


def build_prompt(context: str, turn_idx: int) -> tuple[str, dict[str, Any]]:
    category, instruction = TURN_SCHEDULE_V1[turn_idx]
    prompt = context.rstrip() + "\n\n" + instruction
    return prompt, {
        "turn_prompt_category": category,
        "turn_prompt_text": instruction,
    }


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
        prompt_text, prompt_meta = build_prompt(context, turn_idx)
        prompt_metrics = compute_prefill_metrics(model, tokenizer, prompt_text, early, late, device)
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
                "prompt_text": prompt_text,
                "response": response,
                "prompt_rv": prompt_rv,
                "output_rv": output_rv,
                "rv_delta": rv_delta,
                "classification": classification,
                "clean": clean,
                "prompt_metrics": prompt_metrics,
                "output_metrics": output_metrics,
                **prompt_meta,
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

    def session_flag(session: dict[str, Any], *, start: int, end: int) -> dict[str, Any]:
        turns = [t for t in session["turns"] if start <= t["turn"] < end]
        bt_art = any(t["classification"] in ("BREAKTHROUGH", "ARTICULATE") for t in turns)
        repetitive_rate = (
            sum(1 for t in turns if t["classification"] == "REPETITIVE") / max(len(turns), 1)
        )
        return {
            "bt_art": bt_art,
            "repetitive_rate": repetitive_rate,
        }

    segments = make_turn_segments(max_turns)
    early = segments[0] if segments else ("early", 0, 0)
    late = segments[-1] if segments else ("late", 0, 0)
    entry_flags = []
    persistence_flags = []
    any_bt_art_flags = []
    for session in sessions:
        early_flag = session_flag(session, start=early[1], end=early[2])
        late_flag = session_flag(session, start=late[1], end=late[2])
        entered = bool(early_flag["bt_art"])
        persistent = bool(entered and late_flag["bt_art"] and late_flag["repetitive_rate"] <= 0.2)
        entry_flags.append(entered)
        persistence_flags.append(persistent)
        any_bt_art_flags.append(session["n_bt_art"] > 0)

    entry_count = sum(entry_flags)
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
        "session_any_bt_art_rate": float(np.mean(any_bt_art_flags)),
        "session_entry_rate": float(np.mean(entry_flags)),
        "session_persistence_rate": float(np.mean(persistence_flags)),
        "session_persistence_given_entry": (
            sum(persistence_flags) / entry_count if entry_count else 0.0
        ),
    }


def summarize_vs_control(summary: dict[str, Any]) -> dict[str, Any]:
    by_arm = summary["by_arm"]
    control = by_arm["random_text"]
    comparisons = {}
    for arm_id, stats in by_arm.items():
        if arm_id == "random_text":
            continue
        comparisons[arm_id] = {
            "bt_art_delta_vs_random_text": stats["bt_art_rate"] - control["bt_art_rate"],
            "entry_rate_delta_vs_random_text": stats["session_entry_rate"] - control["session_entry_rate"],
            "persistence_rate_delta_vs_random_text": stats["session_persistence_rate"] - control["session_persistence_rate"],
        }
    return comparisons


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-arm persistence robustness experiment")
    parser.add_argument(
        "--source-run-dir",
        default="results/anchor_layermatched_protocol_confirm_v1/20260316_092017",
    )
    parser.add_argument(
        "--target-condition",
        default="anchor_layermatched_low_bridge_3",
    )
    parser.add_argument("--baseline-groups", default="baseline")
    parser.add_argument("--sessions-per-arm", type=int, default=40)
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--early-layer", type=int, default=5)
    parser.add_argument("--late-layer", type=int, default=27)
    parser.add_argument("--max-turns", type=int, default=15)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--rep-penalty", type=float, default=1.35)
    parser.add_argument("--seed", type=int, default=314)
    parser.add_argument("--output-dir", default="results/induced_persistence_unselected_seed_v1")
    args = parser.parse_args()

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

    arms = build_default_arms(args.target_condition)
    source_conditions = sorted({spec.source_condition for spec in arms if spec.source_condition != "cold_start"})
    rows = load_source_records(source_run_dir, source_conditions, baseline_groups)
    rows_by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_condition[row["condition_name"]].append(row)

    selected_seed_records: dict[str, Any] = {}
    sessions: list[dict[str, Any]] = []
    arm_sessions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    arm_metadata: dict[str, Any] = {}

    for arm_idx, arm in enumerate(arms):
        pool = rows_by_condition.get(arm.source_condition, [])
        chosen, meta = select_rows_for_arm(
            pool,
            mode=arm.selection_mode,
            n_sessions=args.sessions_per_arm,
            seed=args.seed + arm_idx,
        )
        arm_metadata[arm.arm_id] = {
            "source_condition": arm.source_condition,
            "selection_mode": arm.selection_mode,
            "source_description": arm.source_description,
            **meta,
        }
        selected_seed_records[arm.arm_id] = chosen
        for local_idx, seed_record in enumerate(chosen):
            session_seed = args.seed + arm_idx * 10_000 + local_idx
            print(
                f"[{arm.arm_id} {local_idx+1}/{len(chosen)}] "
                f"source_cond={seed_record['condition_name']} "
                f"group={seed_record['prompt_group']} "
                f"seed={seed_record['generation_seed']} "
                f"class={seed_record['classification']} "
                f"bt_art={seed_record.get('bt_art', 0)}",
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
                session_seed=session_seed,
            )
            session["arm_id"] = arm.arm_id
            sessions.append(session)
            arm_sessions[arm.arm_id].append(session)

    summary = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "experiment": "induced_persistence_unselected_seed_v1",
        "model": args.model,
        "source_run_dir": str(source_run_dir),
        "target_condition": args.target_condition,
        "baseline_groups": baseline_groups,
        "sessions_per_arm": args.sessions_per_arm,
        "turn_schedule_name": "fixed15_v1",
        "turn_schedule": [
            {"turn": idx, "category": category, "prompt": prompt}
            for idx, (category, prompt) in enumerate(TURN_SCHEDULE_V1)
        ],
        "max_turns": args.max_turns,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "rep_penalty": args.rep_penalty,
        "arms": arm_metadata,
        "n_seed_sessions": len(sessions),
        "by_arm": {},
    }
    for arm in arms:
        summary["by_arm"][arm.arm_id] = aggregate_sessions(arm_sessions[arm.arm_id])
    summary["comparisons_vs_random_text"] = summarize_vs_control(summary)

    (out_dir / "selected_seed_records.json").write_text(
        json.dumps(selected_seed_records, indent=2),
        encoding="utf-8",
    )
    (out_dir / "sessions.json").write_text(json.dumps(sessions, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["by_arm"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
