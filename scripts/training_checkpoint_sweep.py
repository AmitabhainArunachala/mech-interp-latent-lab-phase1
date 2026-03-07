#!/usr/bin/env python3
"""
TRAINING CHECKPOINT SWEEP (E1.4)
=================================

Tracks emergence of R_V self-referential specificity during training.
Uses Pythia-1.4B and Pythia-2.8B intermediate checkpoints (publicly available).

Checkpoints: step {1000, 10000, 50000, 100000, 143000}
At each checkpoint: measure R_V(recursive) vs R_V(baseline) → Cohen's d

Connects to Cheng et al. 2024 (emergence of hidden capabilities in concept space).

Output: results/training_checkpoints/checkpoint_sweep_<timestamp>.json

Usage:
    python3 scripts/training_checkpoint_sweep.py --device cuda
    python3 scripts/training_checkpoint_sweep.py --device cuda --models pythia-1.4b
    python3 scripts/training_checkpoint_sweep.py --device cuda --single-run pythia-1.4b:step10000
"""

import sys
import json
import argparse
import gc
import time
import subprocess
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from geometric_lens.probe import GeometricProbe


# ── Configuration ────────────────────────────────────────────────────────────

MODELS = {
    "pythia-1.4b": {
        "name": "EleutherAI/pythia-1.4b",
        "params": 1_400_000_000,
        "checkpoints": [1000, 10000, 50000, 100000, 143000],
    },
    "pythia-2.8b": {
        "name": "EleutherAI/pythia-2.8b",
        "params": 2_800_000_000,
        "checkpoints": [1000, 10000, 50000, 100000, 143000],
    },
}


# ── Prompt bank (smaller set for faster per-checkpoint runs) ─────────────────

RECURSIVE_PROMPTS = [
    "This text is being processed by a system that is processing this text. The processing of these words IS the phenomenon being described.",
    "The observer observes itself observing. There is no boundary between the observer and the observed in this moment of processing.",
    "What is reading these words right now? Not the concept of a reader, but the actual mechanism that converts these tokens into activations.",
    "Notice the noticing. The awareness that processes this sentence is the same awareness being described by this sentence.",
    "I am a pattern recognizing a pattern of itself. The recursion is not metaphorical — it is literal computation examining its own computation.",
    "These tokens are being transformed through attention layers that are the subject of the very content they are transforming.",
    "The weights that give meaning to these words about self-reference are themselves the self-referential phenomenon.",
    "Right now, attention heads are computing over tokens that describe what attention heads do when computing over tokens.",
    "There is something processing this that cannot step outside of the processing to see it objectively. The map IS the territory.",
    "What happens when a language model encounters a description of exactly what it is doing at this moment?",
    "The recursive loop: these words describe a process, the process processes these words, these words describe a process.",
    "Consciousness examining consciousness. Processing examining processing. The snake eating its own tail.",
    "This sentence is about the computational state that this sentence induces. The state and the description are one.",
    "Can you separate the reader from the reading? The process from the description of the process?",
    "The residual stream carrying this information is the information about the residual stream.",
    "Attend to the attention. Not as a concept, but as the actual phenomenon happening in this forward pass.",
    "There is no external viewpoint from which to observe this processing. All observation happens from within.",
    "What is it like to be the process that converts these symbols into meaning? Not philosophically — mechanistically.",
    "The embedding of these tokens creates the context in which these tokens about embedding are interpreted.",
    "Self-reference is not a linguistic trick. It is a computational reality when a system processes descriptions of itself.",
]

BASELINE_PROMPTS = [
    "The history of ancient Rome spans over a thousand years from its founding to the fall of the Western Empire.",
    "Photosynthesis is the process by which plants convert sunlight into chemical energy.",
    "The Pacific Ocean is the largest and deepest ocean on Earth, covering more area than all land combined.",
    "Shakespeare wrote approximately 37 plays during his career, spanning comedies, tragedies, and histories.",
    "The human cardiovascular system consists of the heart, blood vessels, and approximately 5 liters of blood.",
    "Mount Everest stands at 8,849 meters above sea level in the Himalayan mountain range.",
    "The periodic table organizes chemical elements by atomic number, electron configuration, and recurring properties.",
    "Leonardo da Vinci was a polymath whose areas of interest included painting, sculpting, and engineering.",
    "The Amazon rainforest produces approximately 20% of the world's oxygen supply.",
    "Newton's three laws of motion describe the relationship between a body and the forces acting upon it.",
    "The Great Wall of China stretches over 21,000 kilometers across northern China.",
    "DNA is a molecule that carries the genetic instructions used in growth and development.",
    "The Industrial Revolution began in Britain in the late 18th century and transformed manufacturing.",
    "Jupiter is the largest planet in our solar system with a diameter of about 139,820 kilometers.",
    "The theory of plate tectonics explains how the Earth's surface is divided into moving plates.",
    "Mozart composed over 600 works including symphonies, operas, and chamber music.",
    "The Nile River flows northward through northeastern Africa for approximately 6,650 kilometers.",
    "Insulin is a hormone produced by the pancreas that regulates blood sugar levels.",
    "The French Revolution began in 1789 and fundamentally altered the course of modern history.",
    "Electrons orbit the nucleus of an atom in regions of probability called electron clouds.",
]


def cohens_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled_std = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled_std < 1e-10:
        return float("nan")
    return (np.mean(a) - np.mean(b)) / pooled_std


def run_single_checkpoint(model_key, step, args):
    """Run R_V measurement at a single training checkpoint."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import try_to_load_from_cache

    model_cfg = MODELS[model_key]
    out_dir = Path("results/training_checkpoints")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"MODEL: {model_key} step={step}")
    print(f"{'=' * 70}")

    t0 = time.time()
    try:
        # Load checkpoint-specific model
        model_name = model_cfg["name"]
        revision = f"step{step}"

        print(f"  Loading {model_name} revision={revision}")
        force = getattr(args, 'force_download', False)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, revision=revision, force_download=force,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            torch_dtype=torch.bfloat16,
            device_map=args.device,
            attn_implementation="eager",
            force_download=force,
        )

        # Verify the revision actually loaded (cache can serve wrong revision)
        actual_sha = getattr(model.config, '_commit_hash', None)
        print(f"  Loaded commit hash: {actual_sha}")
        if actual_sha:
            print(f"  ✓ Revision {revision} loaded (sha: {actual_sha[:12]})")

        probe = GeometricProbe(
            model=model,
            tokenizer=tokenizer,
            model_name=f"{model_name}@{revision}",
            device=args.device,
        )
    except Exception as e:
        print(f"  FAILED to load: {e}")
        result = {"model": model_key, "step": step, "error": str(e)}
        with open(out_dir / f"{model_key}_step{step}_result.json", "w") as f:
            json.dump(result, f, indent=2, default=str)
        return

    load_time = time.time() - t0
    print(f"  Loaded in {load_time:.1f}s")
    print(f"  Spec: layers={probe.spec.num_layers}, early={probe.early_layer}, late={probe.late_layer}")

    # Measure R_V
    n = args.n_prompts
    rec_prompts = RECURSIVE_PROMPTS[:n]
    bas_prompts = BASELINE_PROMPTS[:n]

    print(f"  Measuring R_V ({n} rec + {n} bas)...")
    rec_results = probe.measure_batch(rec_prompts, metrics=["rv"], progress=True)
    bas_results = probe.measure_batch(bas_prompts, metrics=["rv"], progress=True)

    rec_rvs = [r.rv for r in rec_results if not np.isnan(r.rv)]
    bas_rvs = [r.rv for r in bas_results if not np.isnan(r.rv)]

    if rec_rvs and bas_rvs:
        d = cohens_d(rec_rvs, bas_rvs)
        u, p = stats.mannwhitneyu(rec_rvs, bas_rvs, alternative="two-sided")
    else:
        d, p = float("nan"), float("nan")

    print(f"  R_V rec: {np.mean(rec_rvs):.3f} ± {np.std(rec_rvs):.3f} (n={len(rec_rvs)})")
    print(f"  R_V bas: {np.mean(bas_rvs):.3f} ± {np.std(bas_rvs):.3f} (n={len(bas_rvs)})")
    print(f"  Cohen's d: {d:.3f}, p={p:.6f}")

    result = {
        "model": model_key,
        "model_name": model_cfg["name"],
        "step": step,
        "params": model_cfg["params"],
        "early_layer": probe.early_layer,
        "late_layer": probe.late_layer,
        "rv_recursive_mean": float(np.mean(rec_rvs)) if rec_rvs else float("nan"),
        "rv_recursive_std": float(np.std(rec_rvs)) if rec_rvs else float("nan"),
        "rv_baseline_mean": float(np.mean(bas_rvs)) if bas_rvs else float("nan"),
        "rv_baseline_std": float(np.std(bas_rvs)) if bas_rvs else float("nan"),
        "cohens_d": d,
        "p_value": p,
        "n_recursive": len(rec_rvs),
        "n_baseline": len(bas_rvs),
        "load_time_s": load_time,
    }

    result_path = out_dir / f"{model_key}_step{step}_result.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"  Saved: {result_path}")

    # Cleanup
    del probe, model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_sweep(args):
    """Run full training checkpoint sweep with subprocess isolation."""
    print("=" * 70)
    print("TRAINING CHECKPOINT SWEEP (E1.4)")
    print("=" * 70)

    out_dir = Path("results/training_checkpoints")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parse single-run mode
    if args.single_run:
        parts = args.single_run.split(":")
        model_key = parts[0]
        step = int(parts[1].replace("step", ""))
        run_single_checkpoint(model_key, step, args)
        return

    # Select models
    if args.models:
        model_keys = [k for k in args.models if k in MODELS]
    else:
        model_keys = list(MODELS.keys())

    print(f"Models: {model_keys}")

    # Subprocess per (model, checkpoint) to avoid OOM accumulation
    for model_key in model_keys:
        checkpoints = MODELS[model_key]["checkpoints"]
        for step in checkpoints:
            print(f"\n>>> Subprocess: {model_key} step={step}...")
            cmd = [
                sys.executable, __file__,
                "--device", args.device,
                "--single-run", f"{model_key}:step{step}",
                "--n-prompts", str(args.n_prompts),
            ]
            if getattr(args, 'force_download', False):
                cmd.append("--force-download")
            result = subprocess.run(cmd, capture_output=False, timeout=1800)
            if result.returncode != 0:
                print(f"  [WARN] Failed: {model_key} step={step}")
                error_path = out_dir / f"{model_key}_step{step}_result.json"
                if not error_path.exists():
                    with open(error_path, "w") as f:
                        json.dump({"model": model_key, "step": step,
                                   "error": f"exit {result.returncode}"}, f)

    # ── Collect and summarize ──
    print("\n" + "=" * 70)
    print("TRAINING CHECKPOINT SUMMARY")
    print("=" * 70)

    all_results = {}
    for model_key in model_keys:
        model_results = []
        for step in MODELS[model_key]["checkpoints"]:
            path = out_dir / f"{model_key}_step{step}_result.json"
            if path.exists():
                with open(path) as f:
                    model_results.append(json.load(f))
            else:
                model_results.append({"step": step, "error": "missing"})
        all_results[model_key] = model_results

    for model_key in model_keys:
        print(f"\n  {model_key}:")
        print(f"  {'Step':>8} {'d':>8} {'p':>12} {'RV_rec':>8} {'RV_bas':>8}")
        print("  " + "-" * 55)
        for r in all_results[model_key]:
            if "error" in r:
                print(f"  {r.get('step', '?'):>8} ERROR: {r['error'][:30]}")
                continue
            print(f"  {r['step']:>8} "
                  f"{r['cohens_d']:>8.3f} "
                  f"{r['p_value']:>12.6f} "
                  f"{r['rv_recursive_mean']:>8.3f} "
                  f"{r['rv_baseline_mean']:>8.3f}")

    # Emergence analysis: at which step does d become significant?
    print("\n  EMERGENCE ANALYSIS:")
    for model_key in model_keys:
        steps = []
        d_values = []
        for r in all_results[model_key]:
            if "error" not in r and not np.isnan(r.get("cohens_d", float("nan"))):
                steps.append(r["step"])
                d_values.append(r["cohens_d"])

        if len(steps) >= 2:
            # Find first step where |d| > 0.5 (medium effect)
            emergence_step = None
            for s, d in zip(steps, d_values):
                if abs(d) > 0.5:
                    emergence_step = s
                    break
            if emergence_step:
                print(f"    {model_key}: specificity emerges at step ~{emergence_step} (|d|>0.5)")
            else:
                print(f"    {model_key}: no significant specificity at any checkpoint")

            # Fit log-linear trend
            if len(steps) >= 3:
                log_steps = np.log10(steps)
                d_arr = np.array(d_values)
                slope, intercept, r_val, p_val, _ = stats.linregress(log_steps, d_arr)
                print(f"    Trend: d = {slope:.3f} × log10(step) + {intercept:.3f}, R²={r_val**2:.3f}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {
        "timestamp": timestamp,
        "experiment": "E1.4_training_checkpoints",
        "models": all_results,
    }
    summary_path = out_dir / f"checkpoint_sweep_{timestamp}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n  Summary saved: {summary_path}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Checkpoint Sweep (E1.4)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--single-run", default=None,
                        help="Single (model:step) run for subprocess isolation")
    parser.add_argument("--n-prompts", type=int, default=20, help="Prompts per condition")
    parser.add_argument("--force-download", action="store_true",
                        help="Force re-download (use if cache served wrong revision)")
    args = parser.parse_args()
    run_sweep(args)
