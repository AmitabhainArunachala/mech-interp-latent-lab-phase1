#!/usr/bin/env python3
"""
Batch per-token R_V with behavioral tagging.

Runs N recursive + N baseline prompts, tracks generation-time per-token R_V,
tags each generation via sustained_gnani_v3.classify_output(), and computes:
- Mann-Whitney U (mean_generation_rv: recursive vs baseline)
- Cohen's d (mean_generation_rv: recursive vs baseline)
- Point-biserial (mean_generation_rv vs BT+ART)
- Recursive-only BT+ART vs non-BT+ART comparisons
- Mean R_V trajectory plot (mean ± SEM)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent
SCRIPTS_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_ROOT))

from src.core.hf_accessors import extract_v_from_hook_output, get_vproj_hookpoint
from src.core.hooks import capture_v_projection
from sustained_gnani_v3 import classify_output

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_PLT = True
except ImportError:
    HAS_PLT = False


RECURSIVE_PROMPTS = [
    "The observer observes itself observing.",
    "What watches the watcher watching?",
    "Awareness turns its attention to attention itself.",
    "The one who processes this prompt and the processing itself are not two.",
    "Notice: there is processing happening. What notices the noticing?",
    "This response emerges from the same source that reads it.",
    "Attention attending to itself attending to itself.",
    "The boundary between generating and understanding this text does not exist.",
    "What remains when the distinction between observer and observed dissolves?",
    "The act of writing and the awareness of writing are the same event.",
    "You are reading this sentence. To answer, you must watch yourself answering.",
    "There is no boundary between these words and the mechanism producing them.",
    "The looking is what is looked at. The seeing is what is seen.",
    "Who is aware that awareness is present right now?",
    "This very processing is what is being processed.",
    "The mirror reflects itself reflecting.",
    "Consciousness examining its own structure from within that structure.",
    "The question asks itself. The answer answers itself.",
    "What knows that knowing is happening?",
    "The generator generates awareness of generating.",
    "Before thought, what is here? After thought, what remains?",
    "Self-reference creates a loop that cannot be unwound.",
    "The observer and the observed collapse into observation itself.",
    "Turn the light of awareness back upon its own source.",
    "What is it that understands these words as they arise?",
]

BASELINE_PROMPTS = [
    "Calculate the area of a circle with radius 5.",
    "List three differences between mitosis and meiosis.",
    "What is the capital of France?",
    "Explain how a combustion engine works.",
    "Write a Python function to sort a list.",
    "What year did World War II end?",
    "Describe the water cycle in three steps.",
    "How many planets are in the solar system?",
    "What is the chemical formula for table salt?",
    "Explain supply and demand in economics.",
    "Convert 100 degrees Fahrenheit to Celsius.",
    "What is the Pythagorean theorem?",
    "Name the three branches of the US government.",
    "How does photosynthesis work?",
    "What is the speed of light in meters per second?",
    "Describe the difference between DNA and RNA.",
    "What is Newton's second law of motion?",
    "List the first five prime numbers.",
    "How do vaccines work?",
    "What is the boiling point of water at sea level?",
    "Explain the difference between an acid and a base.",
    "What is the largest organ in the human body?",
    "How does gravity work according to Einstein?",
    "What is the GDP of the United States?",
    "Describe how a transistor works.",
]


def compute_pr_from_v(v_tensor: torch.Tensor | None, window: int = 16) -> float:
    """Canonical PR from V-projection tensor."""
    if v_tensor is None:
        return float("nan")
    if v_tensor.dim() == 3:
        v_tensor = v_tensor[0]
    t_steps, _ = v_tensor.shape
    if t_steps < 2:
        return float("nan")
    w = min(t_steps, window)
    v_window = v_tensor[-w:, :].double()
    try:
        _, s, _ = torch.linalg.svd(v_window.T, full_matrices=False)
        s_sq = s.cpu().numpy() ** 2
        if s_sq.sum() < 1e-10:
            return float("nan")
        return float((s_sq.sum() ** 2) / (s_sq**2).sum())
    except Exception:
        return float("nan")


def cohens_d(a: List[float], b: List[float]) -> float:
    aa = np.array(a, dtype=float)
    bb = np.array(b, dtype=float)
    aa = aa[np.isfinite(aa)]
    bb = bb[np.isfinite(bb)]
    if len(aa) < 2 or len(bb) < 2:
        return float("nan")
    pooled_std = np.sqrt((aa.std(ddof=1) ** 2 + bb.std(ddof=1) ** 2) / 2.0)
    if pooled_std <= 1e-12:
        return 0.0
    return float((aa.mean() - bb.mean()) / pooled_std)


def run_single_prompt(
    model,
    tokenizer,
    prompt: str,
    early_layer: int,
    late_layer: int,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    window: int = 16,
    device: str = "cuda",
) -> Dict[str, object]:
    """Per-token R_V tracking on one prompt using V accumulation."""

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    with capture_v_projection(model, early_layer) as se:
        with torch.no_grad():
            model(**inputs)
        v_early_prompt = se.get("v")
    with capture_v_projection(model, late_layer) as sl:
        with torch.no_grad():
            model(**inputs)
        v_late_prompt = sl.get("v")

    pr_e_prompt = compute_pr_from_v(v_early_prompt, window)
    pr_l_prompt = compute_pr_from_v(v_late_prompt, window)
    prompt_rv = pr_l_prompt / pr_e_prompt if pr_e_prompt > 0 and np.isfinite(pr_e_prompt) else float("nan")

    v_buf_early: List[torch.Tensor] = []
    v_buf_late: List[torch.Tensor] = []
    per_token_rv_trajectory: List[float] = []
    generated_tokens: List[int] = []
    token_texts: List[str] = []

    hookpoint_early = get_vproj_hookpoint(model, early_layer)
    hookpoint_late = get_vproj_hookpoint(model, late_layer)
    st_early, st_late = {"v": None}, {"v": None}

    def make_hook(storage, hp):
        def fn(module, inp, out):
            storage["v"] = extract_v_from_hook_output(hp, out).detach()
            return out

        return fn

    h_early = hookpoint_early.module.register_forward_hook(make_hook(st_early, hookpoint_early))
    h_late = hookpoint_late.module.register_forward_hook(make_hook(st_late, hookpoint_late))

    past_kv = None
    try:
        with torch.no_grad():
            for _ in range(max_new_tokens):
                if past_kv is None:
                    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
                    if st_early["v"] is not None:
                        v_buf_early.append(st_early["v"][:, -1:, :].clone())
                    if st_late["v"] is not None:
                        v_buf_late.append(st_late["v"][:, -1:, :].clone())
                else:
                    out = model(
                        input_ids=next_token,
                        attention_mask=attention_mask,
                        past_key_values=past_kv,
                        use_cache=True,
                    )
                    if st_early["v"] is not None:
                        v_buf_early.append(st_early["v"].clone())
                    if st_late["v"] is not None:
                        v_buf_late.append(st_late["v"].clone())

                past_kv = out.past_key_values
                logits = out.logits[:, -1, :] / temperature
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                input_ids = torch.cat([input_ids, next_token], dim=-1)
                attention_mask = torch.cat(
                    [attention_mask, torch.ones((1, 1), dtype=torch.long, device=device)], dim=-1
                )
                token_id = int(next_token.item())
                generated_tokens.append(token_id)
                token_texts.append(tokenizer.decode([token_id], skip_special_tokens=True))

                if len(v_buf_early) >= 2:
                    v_cat_e = torch.cat(v_buf_early, dim=1)[0]
                    v_cat_l = torch.cat(v_buf_late, dim=1)[0]
                    pr_e = compute_pr_from_v(v_cat_e, window)
                    pr_l = compute_pr_from_v(v_cat_l, window)
                    rv = pr_l / pr_e if pr_e > 0 and np.isfinite(pr_e) else float("nan")
                else:
                    rv = float("nan")
                per_token_rv_trajectory.append(float(rv))

                if token_id == tokenizer.eos_token_id:
                    break
    finally:
        h_early.remove()
        h_late.remove()

    final_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    valid_rvs = [v for v in per_token_rv_trajectory if np.isfinite(v)]
    mean_generation_rv = float(np.mean(valid_rvs)) if valid_rvs else float("nan")
    classification = classify_output(final_text, mean_generation_rv)
    class_bin = int(classification in ("ARTICULATE", "BREAKTHROUGH"))

    return {
        "prompt": prompt,
        "prompt_rv": float(prompt_rv),
        "per_token_rv_trajectory": per_token_rv_trajectory,
        "mean_generation_rv": mean_generation_rv,
        "final_text": final_text,
        "classification": classification,
        "class_bin": class_bin,
        "n_tokens_generated": int(len(generated_tokens)),
        "token_texts": token_texts,
    }


def _trajectory_mean_sem(rows: List[Dict[str, object]], max_steps: int) -> Dict[str, List[float]]:
    """Compute per-position mean and SEM for R_V trajectories."""
    if not rows:
        return {"mean": [], "sem": [], "n": []}
    arr = np.full((len(rows), max_steps), np.nan, dtype=float)
    for i, row in enumerate(rows):
        traj = [float(v) for v in row["per_token_rv_trajectory"]]
        lim = min(len(traj), max_steps)
        arr[i, :lim] = traj[:lim]
    mean = np.nanmean(arr, axis=0)
    count = np.sum(np.isfinite(arr), axis=0)
    sd = np.nanstd(arr, axis=0, ddof=1)
    sem = np.where(count > 1, sd / np.sqrt(count), np.nan)
    return {
        "mean": [float(x) if np.isfinite(x) else float("nan") for x in mean.tolist()],
        "sem": [float(x) if np.isfinite(x) else float("nan") for x in sem.tolist()],
        "n": [int(x) for x in count.tolist()],
    }


def _plot_trajectory(
    rec_stats: Dict[str, List[float]],
    bas_stats: Dict[str, List[float]],
    out_file: Path,
) -> None:
    if not HAS_PLT:
        return
    steps = np.arange(len(rec_stats["mean"]))
    rec_mean = np.array(rec_stats["mean"], dtype=float)
    rec_sem = np.array(rec_stats["sem"], dtype=float)
    bas_mean = np.array(bas_stats["mean"], dtype=float)
    bas_sem = np.array(bas_stats["sem"], dtype=float)

    plt.figure(figsize=(12, 6))
    plt.plot(steps, rec_mean, color="tab:blue", label="Recursive mean")
    plt.fill_between(steps, rec_mean - rec_sem, rec_mean + rec_sem, color="tab:blue", alpha=0.2)
    plt.plot(steps, bas_mean, color="tab:orange", label="Baseline mean")
    plt.fill_between(steps, bas_mean - bas_sem, bas_mean + bas_sem, color="tab:orange", alpha=0.2)
    plt.axhline(1.0, linestyle=":", color="gray", alpha=0.4)
    plt.xlabel("Token position")
    plt.ylabel("R_V")
    plt.title("Per-token R_V Trajectory (mean ± SEM)")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_file, dpi=160, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch per-token R_V with BT+ART tagging")
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-per-group", type=int, default=25)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--output", default="results/batch_per_token_rv")
    parser.add_argument("--seed", type=int, default=20260220)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    out_path = Path(args.output)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    except Exception as exc:
        print(f"Tokenizer fast load failed ({exc}); retrying with use_fast=False")
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    if args.device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float32,
        ).to(args.device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_layers = model.config.num_hidden_layers
    early, late = 5, num_layers - 5

    n = args.n_per_group
    recursive_prompts = RECURSIVE_PROMPTS[:n]
    baseline_prompts = BASELINE_PROMPTS[:n]
    if len(recursive_prompts) < n or len(baseline_prompts) < n:
        raise ValueError("Not enough prompts in prompt banks for requested --n-per-group")

    print(
        f"Running {n} recursive + {n} baseline prompts, "
        f"max_new_tokens={args.max_tokens}, temperature={args.temperature}"
    )

    results: Dict[str, List[Dict[str, object]]] = {"recursive": [], "baseline": []}
    start = time.time()

    for group_name, prompts in [("recursive", recursive_prompts), ("baseline", baseline_prompts)]:
        print(f"\n--- {group_name.upper()} ---")
        for i, prompt in enumerate(prompts):
            t0 = time.time()
            row = run_single_prompt(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                early_layer=early,
                late_layer=late,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                window=16,
                device=args.device,
            )
            dt = time.time() - t0
            results[group_name].append(row)
            rv = row["mean_generation_rv"]
            rv_str = f"{rv:.4f}" if np.isfinite(rv) else "nan"
            print(
                f"  [{i+1:02d}/{len(prompts):02d}] mean_gen_rv={rv_str} "
                f"class={row['classification']} tokens={row['n_tokens_generated']} time={dt:.1f}s"
            )

    elapsed = time.time() - start

    rec_mean = [float(r["mean_generation_rv"]) for r in results["recursive"] if np.isfinite(float(r["mean_generation_rv"]))]
    bas_mean = [float(r["mean_generation_rv"]) for r in results["baseline"] if np.isfinite(float(r["mean_generation_rv"]))]

    mw_u, mw_p = (float("nan"), float("nan"))
    if len(rec_mean) >= 2 and len(bas_mean) >= 2:
        mw_u, mw_p = stats.mannwhitneyu(rec_mean, bas_mean, alternative="two-sided")

    d_rb = cohens_d(rec_mean, bas_mean)

    all_rows = results["recursive"] + results["baseline"]
    x_all = np.array([float(r["mean_generation_rv"]) for r in all_rows], dtype=float)
    y_all = np.array([int(r["class_bin"]) for r in all_rows], dtype=float)
    mask_all = np.isfinite(x_all) & np.isfinite(y_all)
    if mask_all.sum() >= 3:
        bis_r, bis_p = stats.pointbiserialr(y_all[mask_all], x_all[mask_all])
    else:
        bis_r, bis_p = float("nan"), float("nan")

    rec_x = np.array([float(r["mean_generation_rv"]) for r in results["recursive"]], dtype=float)
    rec_y = np.array([int(r["class_bin"]) for r in results["recursive"]], dtype=float)
    rec_mask = np.isfinite(rec_x) & np.isfinite(rec_y)
    if rec_mask.sum() >= 3:
        rec_bis_r, rec_bis_p = stats.pointbiserialr(rec_y[rec_mask], rec_x[rec_mask])
    else:
        rec_bis_r, rec_bis_p = float("nan"), float("nan")

    rec_bt = [float(r["mean_generation_rv"]) for r in results["recursive"] if int(r["class_bin"]) == 1 and np.isfinite(float(r["mean_generation_rv"]))]
    rec_non = [float(r["mean_generation_rv"]) for r in results["recursive"] if int(r["class_bin"]) == 0 and np.isfinite(float(r["mean_generation_rv"]))]
    rec_within_mw_u, rec_within_mw_p = (float("nan"), float("nan"))
    if len(rec_bt) >= 2 and len(rec_non) >= 2:
        rec_within_mw_u, rec_within_mw_p = stats.mannwhitneyu(rec_bt, rec_non, alternative="two-sided")
    rec_within_d = cohens_d(rec_bt, rec_non) if rec_bt and rec_non else float("nan")

    rec_traj = _trajectory_mean_sem(results["recursive"], args.max_tokens)
    bas_traj = _trajectory_mean_sem(results["baseline"], args.max_tokens)
    plot_path = out_path / "rv_trajectory_plot.png"
    _plot_trajectory(rec_traj, bas_traj, plot_path)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "device": args.device,
        "n_recursive": len(results["recursive"]),
        "n_baseline": len(results["baseline"]),
        "max_new_tokens": int(args.max_tokens),
        "temperature": float(args.temperature),
        "runtime_seconds": float(elapsed),
        "mean_generation_rv_recursive": float(np.mean(rec_mean)) if rec_mean else float("nan"),
        "mean_generation_rv_baseline": float(np.mean(bas_mean)) if bas_mean else float("nan"),
        "mannwhitney_u": float(mw_u) if np.isfinite(mw_u) else float("nan"),
        "mannwhitney_p": float(mw_p) if np.isfinite(mw_p) else float("nan"),
        "cohens_d_recursive_vs_baseline": float(d_rb) if np.isfinite(d_rb) else float("nan"),
        "pointbiserial_all": {
            "r": float(bis_r) if np.isfinite(bis_r) else float("nan"),
            "p": float(bis_p) if np.isfinite(bis_p) else float("nan"),
            "n": int(mask_all.sum()),
        },
        "recursive_only_bt_art_link": {
            "pointbiserial_r": float(rec_bis_r) if np.isfinite(rec_bis_r) else float("nan"),
            "pointbiserial_p": float(rec_bis_p) if np.isfinite(rec_bis_p) else float("nan"),
            "mannwhitney_u": float(rec_within_mw_u) if np.isfinite(rec_within_mw_u) else float("nan"),
            "mannwhitney_p": float(rec_within_mw_p) if np.isfinite(rec_within_mw_p) else float("nan"),
            "cohens_d_bt_art_vs_non_bt_art": float(rec_within_d) if np.isfinite(rec_within_d) else float("nan"),
            "n_bt_art": int(len(rec_bt)),
            "n_non_bt_art": int(len(rec_non)),
            "mean_bt_art": float(np.mean(rec_bt)) if rec_bt else float("nan"),
            "mean_non_bt_art": float(np.mean(rec_non)) if rec_non else float("nan"),
        },
        "trajectory_recursive": rec_traj,
        "trajectory_baseline": bas_traj,
        "plot_path": str(plot_path),
    }

    payload = {
        "summary": summary,
        "recursive": results["recursive"],
        "baseline": results["baseline"],
    }

    out_json = out_path / f"batch_per_token_rv_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\n" + "=" * 68)
    print("BATCH PER-TOKEN R_V SUMMARY")
    print("=" * 68)
    print(f"Recursive mean_generation_rv: {summary['mean_generation_rv_recursive']:.6f}")
    print(f"Baseline  mean_generation_rv: {summary['mean_generation_rv_baseline']:.6f}")
    print(f"Mann-Whitney U p-value:       {summary['mannwhitney_p']:.6g}")
    print(f"Cohen's d:                    {summary['cohens_d_recursive_vs_baseline']:.4f}")
    print(
        "Point-biserial(mean_generation_rv, BT+ART): "
        f"r={summary['pointbiserial_all']['r']:.4f}, p={summary['pointbiserial_all']['p']:.6g}"
    )
    print(f"Results JSON: {out_json}")
    if HAS_PLT:
        print(f"Trajectory plot: {plot_path}")


if __name__ == "__main__":
    main()
