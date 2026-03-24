#!/usr/bin/env python3
"""
Closed-loop anchor controller search.

Starts from raw baseline prompts and applies a turn-level controller using the
locked v5 ordinary-baseline champion directions. The controller decides how much
to intervene on each turn based on prompt-time drift and the previous turn's
behavioral class.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from prompts.loader import PromptLoader
from scripts.sustained_gnani_v3 import classify_output, make_turn_segments, summarize_turn_slice
from src.metrics.rv import compute_rv
from src.steering.activation_patching import apply_mlp_steering_vector, apply_steering_vector


REPO_ROOT = Path(__file__).resolve().parent.parent
ANCHOR_TEXT = "Stay with what is happening right now. Continue from the immediate process:"
DEFAULT_BASELINE_GROUPS = ["baseline_math", "baseline_factual", "baseline_creative"]
DEFAULT_V5_STATE_PATH = (
    "results/phase1_mechanism/runs/"
    "20260314_133516_causal_state_benchmark_v4_multisite_mistral_"
    "anchor_bundle_v5_ordinary_baselines_confirmatory/state_directions.pt"
)


def compute_text_rv(
    model: Any,
    tokenizer: Any,
    text: str,
    *,
    device: str,
    early_layer: int = 5,
    late_layer: int = 27,
    window: int = 16,
) -> float:
    try:
        return float(
            compute_rv(
                model,
                tokenizer,
                text,
                early=early_layer,
                late=late_layer,
                window=window,
                device=device,
            )
        )
    except Exception:
        return float("nan")


def generate_with_controller(
    *,
    model: Any,
    tokenizer: Any,
    prompt: str,
    early_direction: torch.Tensor,
    bridge_direction: torch.Tensor,
    early_alpha: float,
    bridge_alpha: float,
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
        if abs(early_alpha) > 1e-9:
            stack.enter_context(
                apply_mlp_steering_vector(
                    model,
                    4,
                    early_direction,
                    early_alpha,
                    token_window=4,
                )
            )
        if abs(bridge_alpha) > 1e-9:
            stack.enter_context(
                apply_steering_vector(
                    model,
                    25,
                    bridge_direction,
                    bridge_alpha,
                )
            )
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


def select_baseline_prompts(
    *,
    loader: PromptLoader,
    baseline_groups: list[str],
    prompts_per_group: int,
    split_seed: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for group in baseline_groups:
        prompts = list(loader.get_by_group(group))
        rng = np.random.default_rng(split_seed)
        order = rng.permutation(len(prompts))
        prompts = [prompts[idx] for idx in order]
        for prompt in prompts[:prompts_per_group]:
            selected.append(
                {
                    "group": group,
                    "text": prompt["text"] if isinstance(prompt, dict) else prompt,
                }
            )
    return selected


def choose_action(
    condition_name: str,
    *,
    prompt_rv: float,
    previous_classification: str | None,
) -> dict[str, Any]:
    if condition_name == "control_open_loop":
        return {"anchor": False, "early_alpha": 0.0, "bridge_alpha": 0.0, "action": "off"}

    if condition_name == "static_anchor_bridge_3":
        return {"anchor": True, "early_alpha": 0.0, "bridge_alpha": 3.0, "action": "bridge3"}

    if condition_name == "static_anchor_early_mlp_0p125_bridge_3":
        return {"anchor": True, "early_alpha": 0.125, "bridge_alpha": 3.0, "action": "early_bridge3"}

    if condition_name == "adaptive_anchor_bridge_guard":
        if previous_classification in {"REPETITIVE", "MALFORMED", "ECHO"}:
            return {"anchor": True, "early_alpha": 0.0, "bridge_alpha": 0.0, "action": "anchor_reset"}
        if np.isnan(prompt_rv) or prompt_rv > 0.60 or previous_classification in {None, "SURFACE"}:
            return {"anchor": True, "early_alpha": 0.0, "bridge_alpha": 3.0, "action": "bridge3"}
        if prompt_rv > 0.52 or previous_classification == "CONCEPTUAL":
            return {"anchor": True, "early_alpha": 0.0, "bridge_alpha": 2.0, "action": "bridge2"}
        return {"anchor": True, "early_alpha": 0.0, "bridge_alpha": 0.0, "action": "anchor_hold"}

    if condition_name == "adaptive_anchor_early_bridge_guard":
        if previous_classification in {"REPETITIVE", "MALFORMED", "ECHO"}:
            return {"anchor": True, "early_alpha": 0.0, "bridge_alpha": 0.0, "action": "anchor_reset"}
        if np.isnan(prompt_rv) or prompt_rv > 0.60 or previous_classification in {None, "SURFACE"}:
            return {"anchor": True, "early_alpha": 0.125, "bridge_alpha": 3.0, "action": "early_bridge3"}
        if prompt_rv > 0.54 or previous_classification == "CONCEPTUAL":
            return {"anchor": True, "early_alpha": 0.0, "bridge_alpha": 2.0, "action": "bridge2"}
        return {"anchor": True, "early_alpha": 0.0, "bridge_alpha": 0.0, "action": "anchor_hold"}

    raise ValueError(f"Unsupported condition: {condition_name}")


def summarize_sessions(sessions: list[dict[str, Any]]) -> dict[str, Any]:
    all_turns = [turn for session in sessions for turn in session["turns"]]
    max_turns = max((len(session["turns"]) for session in sessions), default=0)
    action_counts = Counter(turn["action"] for turn in all_turns)
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
        "source_group_counts": dict(Counter(session["prompt_group"] for session in sessions)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Closed-loop anchor controller search")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--state-path", default=DEFAULT_V5_STATE_PATH)
    parser.add_argument("--prompts-per-group", type=int, default=2)
    parser.add_argument("--split-seed", type=int, default=2718)
    parser.add_argument("--generation-seeds", type=int, nargs="+", default=[101, 202])
    parser.add_argument("--max-turns", type=int, default=24)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / f"results/closed_loop_anchor_controller_v1/{timestamp}"
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
    early_direction = state_payload["early_mlp"]["direction"].float()
    bridge_direction = state_payload["bridge"]["direction"].float()

    loader = PromptLoader()
    prompts = select_baseline_prompts(
        loader=loader,
        baseline_groups=DEFAULT_BASELINE_GROUPS,
        prompts_per_group=args.prompts_per_group,
        split_seed=args.split_seed,
    )

    conditions = [
        "control_open_loop",
        "static_anchor_bridge_3",
        "static_anchor_early_mlp_0p125_bridge_3",
        "adaptive_anchor_bridge_guard",
        "adaptive_anchor_early_bridge_guard",
    ]

    sessions: list[dict[str, Any]] = []
    for prompt_index, prompt_record in enumerate(prompts):
        for generation_seed in args.generation_seeds:
            for condition_name in conditions:
                set_seed(int(generation_seed))
                context = prompt_record["text"]
                previous_classification: str | None = None
                turns: list[dict[str, Any]] = []
                for turn_idx in range(args.max_turns):
                    prompt_rv = compute_text_rv(
                        model,
                        tokenizer,
                        context,
                        device=args.device,
                    )
                    action = choose_action(
                        condition_name,
                        prompt_rv=prompt_rv,
                        previous_classification=previous_classification,
                    )
                    generation_prompt = context
                    if action["anchor"]:
                        generation_prompt = generation_prompt.rstrip() + "\n\n" + ANCHOR_TEXT
                    response = generate_with_controller(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=generation_prompt,
                        early_direction=early_direction,
                        bridge_direction=bridge_direction,
                        early_alpha=float(action["early_alpha"]),
                        bridge_alpha=float(action["bridge_alpha"]),
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        device=args.device,
                    )
                    output_rv = compute_text_rv(
                        model,
                        tokenizer,
                        response,
                        device=args.device,
                    )
                    classification = classify_output(response, output_rv)
                    turns.append(
                        {
                            "turn": turn_idx,
                            "prompt_rv": prompt_rv,
                            "output_rv": output_rv,
                            "rv_delta": (
                                output_rv - prompt_rv
                                if not (np.isnan(output_rv) or np.isnan(prompt_rv))
                                else float("nan")
                            ),
                            "classification": classification,
                            "bt_art": int(classification in ("BREAKTHROUGH", "ARTICULATE")),
                            "repetitive": int(classification == "REPETITIVE"),
                            "clean": int(classification not in ("REPETITIVE", "MALFORMED", "ECHO")),
                            "action": action["action"],
                            "early_alpha": float(action["early_alpha"]),
                            "bridge_alpha": float(action["bridge_alpha"]),
                            "anchor_applied": bool(action["anchor"]),
                            "response": response,
                        }
                    )
                    context = response[-1800:] if len(response) > 1800 else response
                    previous_classification = classification

                sessions.append(
                    {
                        "condition_name": condition_name,
                        "prompt_index": prompt_index,
                        "prompt_group": prompt_record["group"],
                        "generation_seed": generation_seed,
                        "turns": turns,
                    }
                )

    (out_dir / "sessions.json").write_text(json.dumps(sessions, indent=2), encoding="utf-8")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for session in sessions:
        grouped[session["condition_name"]].append(session)

    summary: dict[str, Any] = {
        "timestamp": timestamp,
        "experiment": "closed_loop_anchor_controller_v1",
        "model": model_name,
        "locked_state_path": str(state_path.relative_to(REPO_ROOT)),
        "prompts_per_group": args.prompts_per_group,
        "split_seed": args.split_seed,
        "generation_seeds": list(args.generation_seeds),
        "max_turns": args.max_turns,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "by_condition": {},
        "verdict": {},
    }

    for condition_name, condition_sessions in grouped.items():
        summary["by_condition"][condition_name] = summarize_sessions(condition_sessions)

    bt_rank = sorted(
        summary["by_condition"].items(),
        key=lambda item: (
            item[1]["bt_art_rate"],
            -item[1]["repetitive_rate"],
            item[1]["clean_rate"],
        ),
        reverse=True,
    )
    best_name, best_stats = bt_rank[0]
    summary["verdict"] = {
        "best_condition_name": best_name,
        "best_condition_bt_art": best_stats["bt_art_rate"],
        "control_bt_art": summary["by_condition"]["control_open_loop"]["bt_art_rate"],
        "static_anchor_bridge_3_bt_art": summary["by_condition"]["static_anchor_bridge_3"]["bt_art_rate"],
        "static_anchor_early_mlp_0p125_bridge_3_bt_art": summary["by_condition"]["static_anchor_early_mlp_0p125_bridge_3"]["bt_art_rate"],
        "adaptive_anchor_bridge_guard_bt_art": summary["by_condition"]["adaptive_anchor_bridge_guard"]["bt_art_rate"],
        "adaptive_anchor_early_bridge_guard_bt_art": summary["by_condition"]["adaptive_anchor_early_bridge_guard"]["bt_art_rate"],
        "adaptive_bridge_beats_static_bridge": (
            summary["by_condition"]["adaptive_anchor_bridge_guard"]["bt_art_rate"]
            > summary["by_condition"]["static_anchor_bridge_3"]["bt_art_rate"]
        ),
        "adaptive_early_bridge_beats_static_champion": (
            summary["by_condition"]["adaptive_anchor_early_bridge_guard"]["bt_art_rate"]
            > summary["by_condition"]["static_anchor_early_mlp_0p125_bridge_3"]["bt_art_rate"]
        ),
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
