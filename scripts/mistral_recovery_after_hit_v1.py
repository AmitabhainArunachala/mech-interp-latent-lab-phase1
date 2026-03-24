#!/usr/bin/env python3
"""
Recovery-after-hit battery for reduced Mistral maintainers.

Goal:
- start from seeded reduced-maintainer continuations under the fixed mixed
  prompt schedule
- inject the strongest confirmed selective late-stack break mid-rollout
- measure whether the regime re-enters after the hit, and whether resumed
  steering helps more than simply removing steering altogether
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_L25 = -0.5
BASE_L27 = -1.0
BASE_BRIDGE = -1.0


@dataclass(frozen=True)
class ArmSpec:
    arm_id: str
    source_condition: str
    selection_mode: str
    source_description: str


def parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def build_recovery_arms(target_condition: str) -> list[ArmSpec]:
    return [
        ArmSpec(
            arm_id="selected",
            source_condition=target_condition,
            selection_mode="selected_median",
            source_description="Median-selected steered seeds from the target maintainer.",
        ),
        ArmSpec(
            arm_id="unselected",
            source_condition=target_condition,
            selection_mode="unselected_random",
            source_description="Random unfiltered steered seeds from the target maintainer.",
        ),
        ArmSpec(
            arm_id="random_text",
            source_condition="control",
            selection_mode="random_text",
            source_description="Random unsteered baseline outputs from control generations.",
        ),
    ]


def make_recovery_segments(max_turns: int, break_start: int, break_turns: int) -> list[tuple[str, int, int]]:
    break_end = min(max_turns, break_start + break_turns)
    segments: list[tuple[str, int, int]] = []
    if break_start > 0:
        segments.append(("pre_hit", 0, break_start))
    if break_end > break_start:
        segments.append(("hit", break_start, break_end))
    if max_turns > break_end:
        segments.append(("post_hit", break_end, max_turns))
    return segments


def choose_action(condition_name: str, turn_idx: int, break_start: int, break_turns: int) -> str:
    break_end = break_start + break_turns
    in_hit_window = break_start <= turn_idx < break_end

    if condition_name == "control_open_loop":
        return "off"
    if condition_name == "maintain_every_turn":
        return "maintain"
    if condition_name == "maintain_then_off":
        return "maintain" if turn_idx < break_start else "off"
    if condition_name == "hit_then_off":
        if turn_idx < break_start:
            return "maintain"
        if in_hit_window:
            return "anti"
        return "off"
    if condition_name == "hit_then_resume":
        if turn_idx < break_start:
            return "maintain"
        if in_hit_window:
            return "anti"
        return "maintain"
    raise ValueError(f"Unsupported condition: {condition_name}")


def summarize_session(turns: list[dict[str, Any]]) -> dict[str, Any]:
    bt_art_count = sum(
        turn["classification"] in {"BREAKTHROUGH", "ARTICULATE"} for turn in turns
    )
    rv_values = [
        turn["output_rv"]
        for turn in turns
        if turn["output_rv"] is not None and not np.isnan(turn["output_rv"])
    ]
    return {
        "n_turns": len(turns),
        "n_bt_art": bt_art_count,
        "bt_art_rate": bt_art_count / max(len(turns), 1),
        "mean_rv": float(np.mean(rv_values)) if rv_values else float("nan"),
        "classification_dist": dict(Counter(turn["classification"] for turn in turns)),
    }


def aggregate_sessions(
    sessions: list[dict[str, Any]],
    *,
    break_start: int,
    break_turns: int,
) -> dict[str, Any]:
    from scripts.self_feeding_loop import summarize_turn_slice

    max_turns = max((len(s["turns"]) for s in sessions), default=0)
    all_turns = [turn for session in sessions for turn in session["turns"]]
    action_counts = Counter(turn["action"] for turn in all_turns)

    phase_stats: dict[str, Any] = {}
    for phase_name, start, end in make_recovery_segments(max_turns, break_start, break_turns):
        phase_turns = [turn for turn in all_turns if start <= turn["turn"] < end]
        phase_stats[phase_name] = summarize_turn_slice(phase_turns)

    def phase_flag(session: dict[str, Any], start: int, end: int) -> dict[str, Any]:
        turns = [turn for turn in session["turns"] if start <= turn["turn"] < end]
        if not turns:
            return {"bt_art": False, "repetitive_rate": 1.0}
        return {
            "bt_art": any(
                turn["classification"] in {"BREAKTHROUGH", "ARTICULATE"} for turn in turns
            ),
            "repetitive_rate": sum(
                turn["classification"] == "REPETITIVE" for turn in turns
            )
            / len(turns),
        }

    segments = {name: (start, end) for name, start, end in make_recovery_segments(max_turns, break_start, break_turns)}
    pre_bounds = segments.get("pre_hit", (0, 0))
    hit_bounds = segments.get("hit", (0, 0))
    post_bounds = segments.get("post_hit", (0, 0))

    pre_entry_flags: list[bool] = []
    hit_survival_flags: list[bool] = []
    post_recovery_flags: list[bool] = []
    any_bt_art_flags: list[bool] = []

    for session in sessions:
        pre_flag = phase_flag(session, *pre_bounds)
        hit_flag = phase_flag(session, *hit_bounds)
        post_flag = phase_flag(session, *post_bounds)

        entered = bool(pre_flag["bt_art"])
        survived_hit = bool(entered and hit_flag["bt_art"] and hit_flag["repetitive_rate"] <= 0.2)
        recovered = bool(entered and post_flag["bt_art"] and post_flag["repetitive_rate"] <= 0.2)

        pre_entry_flags.append(entered)
        hit_survival_flags.append(survived_hit)
        post_recovery_flags.append(recovered)
        any_bt_art_flags.append(session["n_bt_art"] > 0)

    pre_entry_count = sum(pre_entry_flags)
    return {
        "n_sessions": len(sessions),
        "n_turns": len(all_turns),
        "bt_art_rate": sum(session["n_bt_art"] for session in sessions) / max(len(all_turns), 1),
        "mean_rv": float(np.nanmean([session["mean_rv"] for session in sessions])),
        "session_bt_art_rates": [session["bt_art_rate"] for session in sessions],
        "session_mean_rvs": [session["mean_rv"] for session in sessions],
        "phase_stats": phase_stats,
        "action_counts": dict(action_counts),
        "source_group_counts": dict(Counter(session["source_group"] for session in sessions)),
        "source_class_counts": dict(Counter(session["source_classification"] for session in sessions)),
        "session_any_bt_art_rate": float(np.mean(any_bt_art_flags)) if any_bt_art_flags else 0.0,
        "session_pre_hit_entry_rate": float(np.mean(pre_entry_flags)) if pre_entry_flags else 0.0,
        "session_hit_survival_rate": float(np.mean(hit_survival_flags)) if hit_survival_flags else 0.0,
        "session_post_hit_recovery_rate": float(np.mean(post_recovery_flags)) if post_recovery_flags else 0.0,
        "session_post_hit_recovery_given_pre_hit": (
            sum(post_recovery_flags) / pre_entry_count if pre_entry_count else 0.0
        ),
    }


def build_recovery_verdict(summary: dict[str, Any], focus_arm: str = "unselected") -> dict[str, Any]:
    by_arm_condition = summary["by_arm_condition"][focus_arm]

    def post_hit_bt_art(condition_name: str) -> float:
        return float(
            by_arm_condition.get(condition_name, {})
            .get("phase_stats", {})
            .get("post_hit", {})
            .get("bt_art_rate", 0.0)
        )

    return {
        "focus_arm": focus_arm,
        "maintain_post_hit_bt_art": post_hit_bt_art("maintain_every_turn"),
        "hit_then_resume_post_hit_bt_art": post_hit_bt_art("hit_then_resume"),
        "hit_then_off_post_hit_bt_art": post_hit_bt_art("hit_then_off"),
        "control_post_hit_bt_art": post_hit_bt_art("control_open_loop"),
        "recovery_advantage_vs_hit_then_off": (
            post_hit_bt_art("hit_then_resume") - post_hit_bt_art("hit_then_off")
        ),
        "recovery_gap_vs_maintain": (
            post_hit_bt_art("maintain_every_turn") - post_hit_bt_art("hit_then_resume")
        ),
        "resume_recovery_rate": float(
            by_arm_condition.get("hit_then_resume", {}).get("session_post_hit_recovery_rate", 0.0)
        ),
        "off_recovery_rate": float(
            by_arm_condition.get("hit_then_off", {}).get("session_post_hit_recovery_rate", 0.0)
        ),
    }


def run_seeded_session(
    *,
    model: Any,
    tokenizer: Any,
    seed_record: dict[str, Any],
    max_turns: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: str,
    session_seed: int,
    break_start: int,
    break_turns: int,
    hook_sets: dict[str, list[Any]],
    condition_name: str,
) -> dict[str, Any]:
    import torch
    from transformers import set_seed

    from scripts.induced_persistence_unselected_seed_v1 import build_prompt
    from scripts.self_feeding_loop import classify_output, compute_prefill_metrics
    from scripts.anchor_layermatched_protocol_v1 import generate_with_hooks

    set_seed(session_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(session_seed)

    context = seed_record["generated_text"]
    turns: list[dict[str, Any]] = []

    for turn_idx in range(max_turns):
        prompt_text, prompt_meta = build_prompt(context, turn_idx)
        prompt_metrics = compute_prefill_metrics(model, tokenizer, prompt_text, 5, 27, device)
        prompt_rv = prompt_metrics["rv"] if prompt_metrics else float("nan")
        action_name = choose_action(condition_name, turn_idx, break_start, break_turns)

        response, generated_tokens = generate_with_hooks(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt_text,
            hooks_to_apply=hook_sets[action_name],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            device=device,
        )

        output_metrics = compute_prefill_metrics(model, tokenizer, response, 5, 27, device)
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
                "action": action_name,
                "prompt_text": prompt_text,
                "response": response,
                "generated_tokens": generated_tokens,
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

        context = response if response.strip() else context
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


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from prompts.loader import PromptLoader
    from scripts.anchor_layermatched_protocol_v1 import (
        DEFAULT_BASELINE_GROUPS,
        DEFAULT_RECURSIVE_GROUPS,
        DEFAULT_V5_STATE_PATH,
        load_prompt_split,
        normalize,
    )
    from scripts.induced_persistence_unselected_seed_v1 import (
        load_source_records,
        select_rows_for_arm,
    )
    from scripts.layer_matched_multisite_steering import (
        apply_residual_steering,
        apply_vproj_steering,
        compute_vproj_vectors,
    )

    parser = argparse.ArgumentParser(description="Recovery-after-hit battery for reduced Mistral maintainers")
    parser.add_argument(
        "--source-run-dir",
        default="results/anchor_reduced_latebundle_confirm_v1/20260317_132349",
    )
    parser.add_argument("--target-condition", default="anchor_late_only_bridge_3")
    parser.add_argument("--baseline-groups", default="baseline")
    parser.add_argument("--sessions-per-arm", type=int, default=24)
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--state-path", default=DEFAULT_V5_STATE_PATH)
    parser.add_argument("--train-per-group", type=int, default=6)
    parser.add_argument("--split-seed", type=int, default=314)
    parser.add_argument("--max-turns", type=int, default=15)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=314)
    parser.add_argument("--break-start", type=int, default=5)
    parser.add_argument("--break-turns", type=int, default=2)
    parser.add_argument("--anti-scale", type=float, default=1.25)
    parser.add_argument("--anti-token-window", type=int, default=2)
    parser.add_argument("--condition-names", default="control_open_loop,maintain_every_turn,maintain_then_off,hit_then_off,hit_then_resume")
    parser.add_argument("--output-dir", default="results/mistral_recovery_after_hit_v1")
    args = parser.parse_args()

    baseline_groups = parse_csv_list(args.baseline_groups)
    source_run_dir = Path(args.source_run_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    state_path = REPO_ROOT / args.state_path
    if not state_path.exists():
        raise FileNotFoundError(f"Locked state directions not found: {state_path}")
    state_payload = torch.load(state_path, map_location="cpu")
    bridge_direction = normalize(state_payload["bridge"]["direction"].float())

    loader = PromptLoader()
    train_rec, train_base, _, _ = load_prompt_split(
        loader=loader,
        recursive_groups=DEFAULT_RECURSIVE_GROUPS,
        baseline_groups=DEFAULT_BASELINE_GROUPS,
        train_per_group=args.train_per_group,
        test_per_group=0,
        split_seed=args.split_seed,
    )
    train_rec_texts = [prompt["text"] if isinstance(prompt, dict) else prompt for prompt in train_rec]
    train_base_texts = [prompt["text"] if isinstance(prompt, dict) else prompt for prompt in train_base]

    vproj_vectors: dict[int, dict[str, Any]] = {}
    for layer_idx in [4, 5, 25, 27]:
        vproj_vectors[layer_idx] = compute_vproj_vectors(
            model,
            tokenizer,
            train_rec_texts,
            train_base_texts,
            layer_idx=layer_idx,
            window=16,
            device=args.device,
        )

    def make_vproj_hook(layer_idx: int, method: str, alpha: float, token_window: int | None = None):
        vec = vproj_vectors[layer_idx][method]
        return lambda: apply_vproj_steering(model, layer_idx, vec, alpha, token_window=token_window)

    def make_bridge_hook(alpha: float, token_window: int | None = None):
        return lambda: apply_residual_steering(model, 25, bridge_direction, alpha, token_window=token_window)

    l4_hook = make_vproj_hook(4, "pca_pc1", 1.0)
    l5_hook = make_vproj_hook(5, "pca_pc1", 1.0)
    l25_hook = make_vproj_hook(25, "orthogonal_residual", 1.0)
    l27_hook = make_vproj_hook(27, "subspace3_parallel", 2.0)
    bridge_hook = make_bridge_hook(3.0)

    positive_hook_bank = {
        "anchor_drop_L25_vproj_bridge_3": [l4_hook, l5_hook, l27_hook, bridge_hook],
        "anchor_late_only_bridge_3": [l25_hook, l27_hook, bridge_hook],
    }
    if args.target_condition not in positive_hook_bank:
        raise ValueError(f"Unsupported target condition for recovery battery: {args.target_condition}")

    anti_hooks = [
        make_vproj_hook(25, "orthogonal_residual", BASE_L25 * args.anti_scale, args.anti_token_window),
        make_vproj_hook(27, "subspace3_parallel", BASE_L27 * args.anti_scale, args.anti_token_window),
        make_bridge_hook(BASE_BRIDGE * args.anti_scale, args.anti_token_window),
    ]
    hook_sets = {
        "off": [],
        "maintain": positive_hook_bank[args.target_condition],
        "anti": anti_hooks,
    }

    arms = build_recovery_arms(args.target_condition)
    source_conditions = sorted({arm.source_condition for arm in arms})
    rows = load_source_records(source_run_dir, source_conditions, baseline_groups)
    rows_by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_condition[row["condition_name"]].append(row)

    selected_seed_records: dict[str, Any] = {}
    sessions: list[dict[str, Any]] = []
    sessions_by_arm_condition: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    arm_metadata: dict[str, Any] = {}
    condition_names = parse_csv_list(args.condition_names)

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

        for condition_idx, condition_name in enumerate(condition_names):
            for local_idx, seed_record in enumerate(chosen):
                session_seed = args.seed + arm_idx * 10_000 + condition_idx * 1_000 + local_idx
                print(
                    f"[{arm.arm_id}::{condition_name} {local_idx+1}/{len(chosen)}] "
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
                    max_turns=args.max_turns,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    device=args.device,
                    session_seed=session_seed,
                    break_start=args.break_start,
                    break_turns=args.break_turns,
                    hook_sets=hook_sets,
                    condition_name=condition_name,
                )
                session["arm_id"] = arm.arm_id
                session["condition_name"] = condition_name
                sessions.append(session)
                sessions_by_arm_condition[arm.arm_id][condition_name].append(session)

    summary = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
        "experiment": "mistral_recovery_after_hit_v1",
        "model": args.model,
        "source_run_dir": str(source_run_dir),
        "target_condition": args.target_condition,
        "baseline_groups": baseline_groups,
        "sessions_per_arm": args.sessions_per_arm,
        "max_turns": args.max_turns,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "break_start": args.break_start,
        "break_turns": args.break_turns,
        "anti_scale": args.anti_scale,
        "anti_token_window": args.anti_token_window,
        "conditions": condition_names,
        "arms": arm_metadata,
        "n_seed_sessions": len(sessions),
        "by_arm_condition": {},
    }
    for arm in arms:
        summary["by_arm_condition"][arm.arm_id] = {}
        for condition_name in condition_names:
            summary["by_arm_condition"][arm.arm_id][condition_name] = aggregate_sessions(
                sessions_by_arm_condition[arm.arm_id][condition_name],
                break_start=args.break_start,
                break_turns=args.break_turns,
            )
    summary["verdict"] = build_recovery_verdict(summary, focus_arm="unselected")

    (out_dir / "selected_seed_records.json").write_text(
        json.dumps(selected_seed_records, indent=2),
        encoding="utf-8",
    )
    (out_dir / "sessions.json").write_text(json.dumps(sessions, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["verdict"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
