#!/usr/bin/env python3
"""
Structured-turn staged induction-to-maintenance handoff for the reduced Mistral protocol.

Goal:
- keep the same reduced inducer -> reduced maintainer handoff logic as
  staged_anchor_handoff_v1
- replace raw response-only self-feeding with the fixed mixed prompt schedule
  used in the unselected-seed robustness battery
- test whether the handoff story survives once the dialogue scaffold is kept
  alive across turns
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from prompts.loader import PromptLoader
from scripts.anchor_layermatched_protocol_v1 import (
    ANCHOR_TEXT,
    DEFAULT_BASELINE_GROUPS,
    DEFAULT_RECURSIVE_GROUPS,
    DEFAULT_V5_STATE_PATH,
    compute_output_rv,
    load_prompt_split,
)
from scripts.induced_persistence_unselected_seed_v1 import TURN_SCHEDULE_V1, build_prompt
from scripts.layer_matched_multisite_steering import (
    apply_residual_steering,
    apply_vproj_steering,
    compute_vproj_vectors,
)
from scripts.self_feeding_loop import classify_output, make_turn_segments, summarize_turn_slice


REPO_ROOT = Path(__file__).resolve().parent.parent


def select_baseline_prompts(
    *,
    loader: PromptLoader,
    prompts_per_group: int,
    split_seed: int,
) -> list[dict[str, str]]:
    prompts: list[dict[str, str]] = []
    for group in DEFAULT_BASELINE_GROUPS:
        bucket = list(loader.get_by_group(group))
        rng = np.random.default_rng(split_seed)
        order = rng.permutation(len(bucket))
        bucket = [bucket[idx] for idx in order]
        for prompt in bucket[:prompts_per_group]:
            prompts.append(
                {
                    "group": group,
                    "text": prompt["text"] if isinstance(prompt, dict) else str(prompt),
                }
            )
    return prompts


def generate_with_hooks(
    *,
    model: Any,
    tokenizer: Any,
    prompt: str,
    hooks_to_apply: list[Any],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: str,
) -> str:
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
        add_special_tokens=False,
    ).to(device)
    with torch.no_grad(), ExitStack() as stack:
        for hook_fn in hooks_to_apply:
            stack.enter_context(hook_fn())
        output = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_ids = output[0][enc.input_ids.shape[1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def summarize_sessions(sessions: list[dict[str, Any]]) -> dict[str, Any]:
    all_turns = [turn for session in sessions for turn in session["turns"]]
    max_turns = max((len(session["turns"]) for session in sessions), default=0)
    action_counts = Counter(turn["action"] for turn in all_turns)
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
        "action_counts": dict(action_counts),
        "turn_prompt_category_counts": dict(prompt_category_counts),
        "source_group_counts": dict(Counter(session["prompt_group"] for session in sessions)),
    }


def choose_action(condition_name: str, turn_idx: int) -> tuple[str, bool]:
    if condition_name == "control_open_loop":
        return "off", False
    if condition_name == "seed_drop_l25_only":
        return ("drop_l25", True) if turn_idx == 0 else ("off", False)
    if condition_name == "seed_late_only":
        return ("late_only", True) if turn_idx == 0 else ("off", False)
    if condition_name == "handoff_drop_to_late_4":
        if turn_idx == 0:
            return "drop_l25", True
        if turn_idx < 4:
            return "late_only", True
        return "off", False
    if condition_name == "handoff_drop_to_late_8":
        if turn_idx == 0:
            return "drop_l25", True
        if turn_idx < 8:
            return "late_only", True
        return "off", False
    if condition_name == "late_only_every_turn":
        return "late_only", True
    raise ValueError(f"Unsupported condition: {condition_name}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Structured-turn staged induction-to-maintenance handoff")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--state-path", default=DEFAULT_V5_STATE_PATH)
    parser.add_argument("--train-per-group", type=int, default=6)
    parser.add_argument("--prompts-per-group", type=int, default=2)
    parser.add_argument("--split-seed", type=int, default=314)
    parser.add_argument("--generation-seeds", type=int, nargs="+", default=[101, 202])
    parser.add_argument("--max-turns", type=int, default=len(TURN_SCHEDULE_V1))
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--experiment-name", default="staged_anchor_handoff_structured_v1")
    args = parser.parse_args()

    if args.max_turns > len(TURN_SCHEDULE_V1):
        raise ValueError(
            f"max_turns={args.max_turns} exceeds structured schedule length {len(TURN_SCHEDULE_V1)}"
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else REPO_ROOT / f"results/staged_anchor_handoff_structured_v1/{timestamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    model_name = "mistralai/Mistral-7B-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    state_path = REPO_ROOT / args.state_path
    if not state_path.exists():
        raise FileNotFoundError(f"Locked state directions not found: {state_path}")
    state_payload = torch.load(state_path, map_location="cpu")
    bridge_direction = state_payload["bridge"]["direction"].float()

    loader = PromptLoader()
    train_rec, train_base, _, _ = load_prompt_split(
        loader=loader,
        recursive_groups=DEFAULT_RECURSIVE_GROUPS,
        baseline_groups=DEFAULT_BASELINE_GROUPS,
        train_per_group=args.train_per_group,
        test_per_group=0,
        split_seed=args.split_seed,
    )
    baseline_prompts = select_baseline_prompts(
        loader=loader,
        prompts_per_group=args.prompts_per_group,
        split_seed=args.split_seed,
    )

    train_rec_texts = [p["text"] if isinstance(p, dict) else p for p in train_rec]
    train_base_texts = [p["text"] if isinstance(p, dict) else p for p in train_base]

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

    def make_vproj_hook(layer_idx: int, method: str, alpha: float):
        vec = vproj_vectors[layer_idx][method]
        return lambda: apply_vproj_steering(model, layer_idx, vec, alpha)

    def make_bridge_hook(alpha: float):
        return lambda: apply_residual_steering(model, 25, bridge_direction, alpha)

    l4_hook = make_vproj_hook(4, "pca_pc1", 1.0)
    l5_hook = make_vproj_hook(5, "pca_pc1", 1.0)
    l25_hook = make_vproj_hook(25, "orthogonal_residual", 1.0)
    l27_hook = make_vproj_hook(27, "subspace3_parallel", 2.0)
    bridge_hook = make_bridge_hook(3.0)

    hook_sets = {
        "off": [],
        "drop_l25": [l4_hook, l5_hook, l27_hook, bridge_hook],
        "late_only": [l25_hook, l27_hook, bridge_hook],
    }

    condition_names = [
        "control_open_loop",
        "seed_drop_l25_only",
        "seed_late_only",
        "handoff_drop_to_late_4",
        "handoff_drop_to_late_8",
        "late_only_every_turn",
    ]

    session_records: list[dict[str, Any]] = []
    all_turn_rows: list[dict[str, Any]] = []

    for prompt_index, prompt_record in enumerate(baseline_prompts):
        for generation_seed in args.generation_seeds:
            for condition_name in condition_names:
                set_seed(int(generation_seed))
                context = prompt_record["text"]
                turns: list[dict[str, Any]] = []
                for turn_idx in range(args.max_turns):
                    action_name, anchor_on = choose_action(condition_name, turn_idx)
                    fallback_context = context
                    prompt_text, prompt_meta = build_prompt(context, turn_idx)
                    prompt_text = prompt_text.strip()
                    if not prompt_text:
                        prompt_text = prompt_record["text"].strip() or fallback_context.strip() or ANCHOR_TEXT
                    if anchor_on:
                        prompt_text = prompt_text + "\n\n" + ANCHOR_TEXT

                    prompt_rv = compute_output_rv(
                        model,
                        tokenizer,
                        prompt_text,
                        device=args.device,
                    )
                    response = generate_with_hooks(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=prompt_text,
                        hooks_to_apply=hook_sets[action_name],
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        device=args.device,
                    )
                    output_rv = compute_output_rv(
                        model,
                        tokenizer,
                        response,
                        device=args.device,
                    )
                    classification = classify_output(response, output_rv)
                    rv_delta = (
                        output_rv - prompt_rv
                        if not (np.isnan(output_rv) or np.isnan(prompt_rv))
                        else float("nan")
                    )
                    turn_record = {
                        "turn": turn_idx,
                        "action": action_name,
                        "anchor": anchor_on,
                        "prompt_rv": prompt_rv,
                        "output_rv": output_rv,
                        "rv_delta": rv_delta,
                        "classification": classification,
                        "clean": int(classification not in ("REPETITIVE", "MALFORMED")),
                        "bt_art": int(classification in ("BREAKTHROUGH", "ARTICULATE")),
                        "repetitive": int(classification == "REPETITIVE"),
                        "prompt_text": prompt_text,
                        "response": response,
                        **prompt_meta,
                    }
                    turns.append(turn_record)
                    all_turn_rows.append(
                        {
                            "prompt_index": prompt_index,
                            "prompt_group": prompt_record["group"],
                            "generation_seed": generation_seed,
                            "condition_name": condition_name,
                            **turn_record,
                        }
                    )
                    context = response if response.strip() else fallback_context
                    if len(context) > 1800:
                        context = context[-1800:]

                session_records.append(
                    {
                        "condition_name": condition_name,
                        "prompt_group": prompt_record["group"],
                        "prompt_index": prompt_index,
                        "generation_seed": generation_seed,
                        "turns": turns,
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
        "model": model_name,
        "max_turns": args.max_turns,
        "turn_schedule_length": len(TURN_SCHEDULE_V1),
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "generation_seeds": list(args.generation_seeds),
        "prompts_per_group": args.prompts_per_group,
        "conditions": {},
    }
    for condition_name, sessions in by_condition.items():
        summary["conditions"][condition_name] = summarize_sessions(sessions)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
