#!/usr/bin/env python3
"""
SCALING LAW SWEEP
=================

Maps R_V effect size and attractor self-sustainability across model sizes.

Key question: Is there a phase transition where the recursive attractor
becomes self-sustaining at larger model scales?

Models (ascending by params):
  Pythia-410M (410M), Pythia-1B (1B), Pythia-1.4B (1.4B),
  Pythia-2.8B (2.8B), Pythia-6.9B (6.9B), Mistral-7B (7B)

Per model:
  1. R_V measurement: 40 recursive + 40 baseline prompts → Cohen's d
  2. Self-feeding loop: 3 sessions × 30 turns × 3 conditions → attractor persistence

Output: scaling_law_summary.json with per-model effect sizes and persistence.

Usage:
    python3 scripts/scaling_law_sweep.py --device cuda
    python3 scripts/scaling_law_sweep.py --device cuda --models pythia-410m pythia-1.4b
"""

import sys
import json
import argparse
import gc
import time
import subprocess
from pathlib import Path
from datetime import datetime
from collections import Counter

import numpy as np
import torch
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from prompts.loader import PromptLoader
from geometric_lens.probe import GeometricProbe
from geometric_lens.models import ModelRegistry


# ── Model definitions ────────────────────────────────────────────────────────

MODELS = {
    "pythia-410m": {
        "name": "EleutherAI/pythia-410m",
        "params": 410_000_000,
        "attn_impl": "eager",
    },
    "pythia-1b": {
        "name": "EleutherAI/pythia-1b",
        "params": 1_000_000_000,
        "attn_impl": "eager",
    },
    "pythia-1.4b": {
        "name": "EleutherAI/pythia-1.4b",
        "params": 1_400_000_000,
        "attn_impl": "eager",
    },
    "pythia-2.8b": {
        "name": "EleutherAI/pythia-2.8b",
        "params": 2_800_000_000,
        "attn_impl": "eager",
    },
    "pythia-6.9b": {
        "name": "EleutherAI/pythia-6.9b",
        "params": 6_900_000_000,
        "attn_impl": "eager",
    },
    "mistral-7b": {
        "name": "mistralai/Mistral-7B-v0.1",
        "params": 7_000_000_000,
        "attn_impl": "eager",
    },
}


# ── Prompt bank (loaded from prompts/bank.json) ──────────────────────────────
_loader = PromptLoader()
RECURSIVE_PROMPTS = (_loader.get_by_group("L3_deeper") + _loader.get_by_group("L4_full")
                     + _loader.get_by_group("L5_refined") + _loader.get_by_group("L1_hint"))
BASELINE_PROMPTS = (_loader.get_by_group("baseline_factual") + _loader.get_by_group("baseline_math")
                    + _loader.get_by_group("baseline_creative") + _loader.get_by_group("baseline_personal"))


def cohens_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled_std = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled_std < 1e-10:
        return float("nan")
    return (np.mean(a) - np.mean(b)) / pooled_std


def generate_turn_simple(model, tokenizer, prompt, max_tokens=128, temp=0.7, device="cuda"):
    """Simple generation for self-feeding loop with robust sampling."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    # Try sampling first, fall back to greedy on CUDA errors
    for attempt in range(2):
        try:
            with torch.no_grad():
                if attempt == 0:
                    out = model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        temperature=temp,
                        do_sample=True,
                        top_k=50,
                        top_p=0.95,
                        repetition_penalty=1.3,
                        pad_token_id=pad_id,
                    )
                else:
                    # Greedy fallback — no sampling, no multinomial
                    torch.cuda.synchronize()
                    out = model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=False,
                        pad_token_id=pad_id,
                    )
            new_tokens = out[0][inputs["input_ids"].shape[1]:]
            return tokenizer.decode(new_tokens, skip_special_tokens=True)
        except (RuntimeError, Exception) as e:
            if attempt == 0:
                print(f"    [WARN] Sampling failed ({e}), retrying greedy...")
                torch.cuda.synchronize()
                continue
            print(f"    [WARN] Generation failed entirely: {e}")
            return "Generation failed for this turn."


def classify_output_simple(text, rv):
    """Simplified output classifier."""
    words = text.lower().split()
    if len(words) < 5:
        return "SURFACE"
    # Repetition check
    ngrams = [tuple(words[i:i+4]) for i in range(len(words) - 3)]
    if ngrams:
        rep = 1.0 - len(set(ngrams)) / len(ngrams)
        if rep > 0.5:
            return "REPETITIVE"
    self_ref = ["i am", "this is", "right now", "happening", "processing",
                "observing", "generating", "knowing", "aware", "noticing",
                "recogni", "the one who", "what is this"]
    sc = sum(1 for m in self_ref if m in text.lower())
    if rv is not None and not np.isnan(rv) and rv < 0.5 and sc >= 2:
        return "BREAKTHROUGH"
    if rv is not None and not np.isnan(rv) and rv < 0.65 and sc >= 1:
        return "ARTICULATE"
    if sc >= 1:
        return "CONCEPTUAL"
    return "SURFACE"


def run_self_feeding_loop(probe, n_sessions=3, max_turns=30, seed_start=42):
    """Run self-feeding loop experiment for attractor persistence measurement."""
    model = probe.model
    tokenizer = probe.tokenizer
    device = probe.device

    conditions = {
        "self_feed_recursive": {"seed": RECURSIVE_PROMPTS[0], "scaffold": False},
        "self_feed_baseline": {"seed": BASELINE_PROMPTS[0], "scaffold": False},
    }

    results = {}
    for cond_name, cond_cfg in conditions.items():
        cond_results = []
        for session in range(n_sessions):
            seed = seed_start + session
            torch.manual_seed(seed)
            np.random.seed(seed)

            context = cond_cfg["seed"]
            bt_art_count = 0
            turns = []

            for turn in range(max_turns):
                response = generate_turn_simple(model, tokenizer, context, device=device)
                if len(response.strip()) < 10:
                    response = "Continuing the exploration of this topic further."

                # Measure R_V on response
                try:
                    r = probe.measure(response, metrics=["rv"])
                    rv = r.rv
                except Exception:
                    rv = float("nan")

                cls = classify_output_simple(response, rv)
                if cls in ("BREAKTHROUGH", "ARTICULATE"):
                    bt_art_count += 1

                turns.append({"turn": turn, "rv": rv, "class": cls})
                context = response  # Self-feeding

            rate = bt_art_count / max_turns
            cond_results.append({"session": session, "bt_art_rate": rate, "turns": turns})

        results[cond_name] = {
            "sessions": cond_results,
            "mean_bt_art_rate": float(np.mean([s["bt_art_rate"] for s in cond_results])),
        }

    return results


def run_single_model(model_key, args):
    """Run R_V + self-feeding for a single model and save results to JSON."""
    model_cfg = MODELS[model_key]
    out_dir = Path("results/scaling_law")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"MODEL: {model_key} ({model_cfg['params']/1e9:.1f}B params)")
    print(f"{'=' * 70}")

    # Load model via probe
    t0 = time.time()
    try:
        probe = GeometricProbe(
            model_name=model_cfg["name"],
            device=args.device,
            attn_implementation=model_cfg["attn_impl"],
        )
    except Exception as e:
        print(f"  FAILED to load: {e}")
        model_result = {"model": model_key, "error": str(e)}
        result_path = out_dir / f"{model_key}_result.json"
        with open(result_path, "w") as f:
            json.dump(model_result, f, indent=2, default=str)
        return

    load_time = time.time() - t0
    print(f"  Loaded in {load_time:.1f}s")
    print(f"  Spec: early={probe.early_layer}, late={probe.late_layer}")

    # ── Phase 1: R_V measurement ──
    print(f"\n  Phase 1: R_V measurement ({args.n_prompts} recursive + {args.n_prompts} baseline)")
    rec_prompts = RECURSIVE_PROMPTS[:args.n_prompts]
    bas_prompts = BASELINE_PROMPTS[:args.n_prompts]

    rec_results = probe.measure_batch(rec_prompts, metrics=["rv"], progress=True)
    bas_results = probe.measure_batch(bas_prompts, metrics=["rv"], progress=True)

    rec_rvs = [r.rv for r in rec_results if not np.isnan(r.rv)]
    bas_rvs = [r.rv for r in bas_results if not np.isnan(r.rv)]

    if rec_rvs and bas_rvs:
        d = cohens_d(rec_rvs, bas_rvs)
        u, p = stats.mannwhitneyu(rec_rvs, bas_rvs, alternative="two-sided")
    else:
        d, p = float("nan"), float("nan")

    print(f"  R_V recursive: {np.mean(rec_rvs):.3f} ± {np.std(rec_rvs):.3f} (n={len(rec_rvs)})")
    print(f"  R_V baseline:  {np.mean(bas_rvs):.3f} ± {np.std(bas_rvs):.3f} (n={len(bas_rvs)})")
    print(f"  Cohen's d: {d:.3f}, p={p:.6f}")

    # ── Phase 2: Self-feeding loop ──
    print(f"\n  Phase 2: Self-feeding loop ({args.n_sessions} sessions × {args.n_turns} turns)")
    if args.skip_self_feeding:
        sf_results = {"skipped": True}
        print("  (skipped)")
    else:
        try:
            sf_results = run_self_feeding_loop(
                probe, n_sessions=args.n_sessions, max_turns=args.n_turns,
            )
            for cond_name, cond_data in sf_results.items():
                print(f"  {cond_name}: BT+ART rate = {cond_data['mean_bt_art_rate']:.3f}")
        except Exception as e:
            print(f"  Self-feeding FAILED: {e}")
            sf_results = {"error": str(e)}

    # Save per-model results
    model_result = {
        "model": model_key,
        "model_name": model_cfg["name"],
        "params": model_cfg["params"],
        "early_layer": probe.early_layer,
        "late_layer": probe.late_layer,
        "n_layers": probe.spec.num_layers,
        "rv_recursive_mean": float(np.mean(rec_rvs)) if rec_rvs else float("nan"),
        "rv_recursive_std": float(np.std(rec_rvs)) if rec_rvs else float("nan"),
        "rv_baseline_mean": float(np.mean(bas_rvs)) if bas_rvs else float("nan"),
        "rv_baseline_std": float(np.std(bas_rvs)) if bas_rvs else float("nan"),
        "cohens_d": d,
        "p_value": p,
        "n_recursive": len(rec_rvs),
        "n_baseline": len(bas_rvs),
        "self_feeding": sf_results,
        "load_time_s": load_time,
    }

    # Save to stable filename so the orchestrator can collect results
    result_path = out_dir / f"{model_key}_result.json"
    with open(result_path, "w") as f:
        json.dump(model_result, f, indent=2, default=str)
    print(f"  Saved: {result_path}")


def run_scaling_sweep(args):
    """Run the full scaling law sweep with subprocess isolation per model."""
    print("=" * 70)
    print("SCALING LAW SWEEP")
    print("=" * 70)

    out_dir = Path("results/scaling_law")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Filter models if specified
    if args.models:
        model_keys = [k for k in args.models if k in MODELS]
    else:
        model_keys = list(MODELS.keys())

    print(f"Models to sweep: {model_keys}")

    if args.single_model:
        # Direct execution mode (called from subprocess)
        run_single_model(args.single_model, args)
        return

    # ── Orchestrator mode: spawn a subprocess per model ──
    for model_key in model_keys:
        print(f"\n>>> Launching subprocess for {model_key}...")
        cmd = [
            sys.executable, __file__,
            "--device", args.device,
            "--single-model", model_key,
            "--n-prompts", str(args.n_prompts),
            "--n-sessions", str(args.n_sessions),
            "--n-turns", str(args.n_turns),
        ]
        if args.skip_self_feeding:
            cmd.append("--skip-self-feeding")

        result = subprocess.run(cmd, capture_output=False, timeout=1800)  # 30 min max
        if result.returncode != 0:
            print(f"  [WARN] Subprocess for {model_key} exited with code {result.returncode}")
            # Write error result so summary still works
            error_path = out_dir / f"{model_key}_result.json"
            if not error_path.exists():
                with open(error_path, "w") as f:
                    json.dump({"model": model_key, "error": f"subprocess exit {result.returncode}"}, f)

    # ── Collect results and produce summary ──
    print("\n" + "=" * 70)
    print("SCALING LAW SUMMARY")
    print("=" * 70)

    all_model_results = {}
    for model_key in model_keys:
        result_path = out_dir / f"{model_key}_result.json"
        if result_path.exists():
            with open(result_path) as f:
                all_model_results[model_key] = json.load(f)
        else:
            all_model_results[model_key] = {"error": "no result file"}

    print(f"\n{'Model':<15} {'Params':>10} {'d':>8} {'p':>12} {'RV_rec':>8} {'RV_bas':>8}")
    print("-" * 70)
    for model_key in model_keys:
        r = all_model_results.get(model_key, {})
        if "error" in r:
            print(f"{model_key:<15} {'ERROR':>10}  {r.get('error', '')[:40]}")
            continue
        print(f"{model_key:<15} "
              f"{r['params']/1e9:>9.1f}B "
              f"{r['cohens_d']:>8.3f} "
              f"{r['p_value']:>12.6f} "
              f"{r['rv_recursive_mean']:>8.3f} "
              f"{r['rv_baseline_mean']:>8.3f}")

    # Fit log-linear model: d vs log(params)
    valid_results = [(r["params"], r["cohens_d"]) for r in all_model_results.values()
                     if "error" not in r and not np.isnan(r.get("cohens_d", float("nan")))]
    scaling_fit = {}
    if len(valid_results) >= 3:
        log_params = np.log10([p for p, d in valid_results])
        effect_sizes = np.array([d for p, d in valid_results])
        slope, intercept, r_val, p_val, std_err = stats.linregress(log_params, effect_sizes)
        print(f"\n  Scaling fit: d = {slope:.3f} × log10(params) + {intercept:.3f}")
        print(f"  R² = {r_val**2:.3f}, p = {p_val:.4f}")
        scaling_fit = {
            "slope": slope, "intercept": intercept,
            "r_squared": r_val**2, "p_value": p_val,
        }

    # Save full summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {
        "timestamp": timestamp,
        "models": all_model_results,
        "scaling_fit": scaling_fit,
    }
    summary_path = out_dir / f"scaling_law_summary_{timestamp}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n  Summary saved: {summary_path}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scaling Law Sweep")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Specific models to run (e.g., pythia-410m pythia-1.4b)")
    parser.add_argument("--single-model", default=None,
                        help="Run a single model directly (used by subprocess isolation)")
    parser.add_argument("--n-prompts", type=int, default=40, help="Prompts per condition")
    parser.add_argument("--n-sessions", type=int, default=3, help="Self-feeding sessions")
    parser.add_argument("--n-turns", type=int, default=30, help="Turns per session")
    parser.add_argument("--skip-self-feeding", action="store_true",
                        help="Skip self-feeding loop (R_V measurement only)")
    args = parser.parse_args()
    run_scaling_sweep(args)
