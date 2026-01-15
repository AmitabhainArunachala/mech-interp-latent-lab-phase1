"""
C2 + R_V Measurement Pipeline

Runs the proven C2 behavioral transfer config while measuring R_V at each generation step.
This bridges the geometry → behavior gap.

The C2 config:
- head_target: "h18_h26" (V_proj steering on H18 + H26)
- kv_strategy: "full" (Full KV swap from recursive prompt)
- residual_alphas: {26: 0.6} (Cascade at L26)
- vproj_alpha: 2.5

Expected outcome:
- C2 generation produces R_V < 0.55 (contracted)
- Baseline generation produces R_V > 0.70 (expanded)
- Behavioral output shifts to philosophical domain
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from prompts.loader import PromptLoader
from src.core.models import load_model, set_seed
from src.core.hooks import capture_v_projection
from src.metrics.rv import participation_ratio
from src.metrics.baseline_suite import BaselineMetricsSuite
from src.pipelines.archive.steering import compute_steering_vector, SteeringVectorPatcher
from src.pipelines.archive.surgical_sweep import (
    CascadeResidualSteeringPatcher,
    SplitBrainKVPatcher,
    compute_coherence,
    compute_on_topic,
    score_recursion_regex,
)
from src.pipelines.registry import ExperimentResult


# C2 config (proven to work in Dec 18 experiments)
C2_CONFIG = {
    "name": "C2_H18_H26_FullKV",
    "head_target": "h18_h26",
    "kv_strategy": "full",
    "residual_alphas": {26: 0.6},
    "vproj_alpha": 2.5,
}

# Baseline config (no intervention)
BASELINE_CONFIG = {
    "name": "Baseline_NoIntervention",
    "head_target": "none",
    "kv_strategy": "none",
    "residual_alphas": None,
    "vproj_alpha": 0.0,
}

# KV-only config (no steering, just KV swap)
KV_ONLY_CONFIG = {
    "name": "KV_Only_NoSteering",
    "head_target": "none",
    "kv_strategy": "full",
    "residual_alphas": None,
    "vproj_alpha": 0.0,
}


def classify_domain(text: str) -> Dict[str, Any]:
    """Classify generated text into domain (philosophical vs task-like)."""
    text_lower = text.lower()

    philosophical_markers = [
        "self", "awareness", "consciousness", "witness", "observer",
        "knowing", "being", "existence", "mind", "thought",
        "recursive", "reflection", "meta", "attention",
    ]

    task_markers = [
        "calculate", "answer", "result", "equals", "=",
        "therefore", "because", "since", "given",
    ]

    phil_count = sum(1 for m in philosophical_markers if m in text_lower)
    task_count = sum(1 for m in task_markers if m in text_lower)

    if phil_count > task_count + 2:
        domain = "philosophical"
    elif task_count > phil_count:
        domain = "task"
    else:
        domain = "mixed"

    return {
        "domain": domain,
        "philosophical_markers": phil_count,
        "task_markers": task_count,
    }


def generate_with_rv_tracking(
    model,
    tokenizer,
    prompt: str,
    recursive_prompt: str,
    config: Dict[str, Any],
    steering_vector: Optional[torch.Tensor],
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    early_layer: int = 5,
    late_layer: int = 27,
    rv_window: int = 16,
    device: str = "cuda",
) -> Tuple[str, List[float], Dict[str, Any]]:
    """
    Generate text with C2-style config while tracking R_V at each step.

    Returns:
        (generated_text, rv_trajectory, metadata)
        - rv_trajectory: List of R_V values at each generation step
        - metadata: Dict with coherence, domain, etc.
    """
    from src.core.head_specific_patching import HeadSpecificSteeringPatcher

    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)

    patchers = []
    rv_trajectory = []

    try:
        # Extract KV caches if needed
        baseline_kv = None
        recursive_kv = None

        if config.get("kv_strategy", "none") != "none":
            with torch.no_grad():
                rec_inputs = tokenizer(recursive_prompt, return_tensors="pt", add_special_tokens=False).to(device)
                rec_outputs = model(**rec_inputs, use_cache=True)
                recursive_kv = rec_outputs.past_key_values

                base_outputs = model(**inputs, use_cache=True)
                baseline_kv = base_outputs.past_key_values

        # Setup residual steering (cascade)
        if config.get("residual_alphas") and steering_vector is not None:
            # Convert string keys to int if needed (JSON loading can make keys strings)
            residual_alphas = config["residual_alphas"]
            if isinstance(residual_alphas, dict):
                residual_alphas = {int(k): float(v) for k, v in residual_alphas.items()}
            cascade_patcher = CascadeResidualSteeringPatcher(
                model, steering_vector, residual_alphas
            )
            cascade_patcher.register()
            patchers.append(cascade_patcher)

        # Setup V_PROJ steering (head-specific)
        if config.get("vproj_alpha", 0) > 0 and steering_vector is not None:
            head_target = config.get("head_target", "full")

            if head_target == "h18_h26":
                target_heads = [18, 26]
                v_steering_patcher = HeadSpecificSteeringPatcher(
                    model, steering_vector, target_heads, config["vproj_alpha"]
                )
                v_steering_patcher.register(layer_idx=27)
                patchers.append(v_steering_patcher)
            elif head_target == "full":
                v_steering_patcher = SteeringVectorPatcher(
                    model, steering_vector, config["vproj_alpha"]
                )
                v_steering_patcher.register(layer_idx=27)
                patchers.append(v_steering_patcher)
            # "none" = no steering

        # Setup KV cache
        kv_cache_to_use = None
        if config.get("kv_strategy") == "full":
            kv_cache_to_use = recursive_kv

        # Generate with R_V tracking
        with torch.no_grad():
            if kv_cache_to_use is not None:
                current_ids = inputs["input_ids"][:, -1:]
                current_past = kv_cache_to_use
                generated_ids = inputs["input_ids"].clone()

                # Collect V-projections for R_V
                v_early_buffer = []
                v_late_buffer = []

                for step in range(max_new_tokens):
                    # Capture V-projections during this step
                    with capture_v_projection(model, early_layer) as v_early_store:
                        with capture_v_projection(model, late_layer) as v_late_store:
                            outputs = model(
                                input_ids=current_ids,
                                past_key_values=current_past,
                                use_cache=True
                            )

                    # Store V-projections
                    v_e = v_early_store.get("v")
                    v_l = v_late_store.get("v")

                    if v_e is not None:
                        v_early_buffer.append(v_e.detach().cpu())
                    if v_l is not None:
                        v_late_buffer.append(v_l.detach().cpu())

                    # Compute R_V every rv_window steps (or at end)
                    if len(v_early_buffer) >= rv_window:
                        # Stack last rv_window V-projections
                        v_early_window = torch.cat(v_early_buffer[-rv_window:], dim=1)
                        v_late_window = torch.cat(v_late_buffer[-rv_window:], dim=1)

                        pr_early = participation_ratio(v_early_window[0], rv_window)
                        pr_late = participation_ratio(v_late_window[0], rv_window)

                        if pr_early > 0 and not np.isnan(pr_early) and not np.isnan(pr_late):
                            rv = pr_late / pr_early
                            rv_trajectory.append(rv)

                    # Sample next token
                    next_token_logits = outputs.logits[:, -1, :]
                    next_token = torch.multinomial(
                        torch.softmax(next_token_logits / temperature, dim=-1), 1
                    )
                    generated_ids = torch.cat([generated_ids, next_token], dim=1)

                    if next_token.item() == tokenizer.eos_token_id:
                        break

                    current_ids = next_token
                    current_past = outputs.past_key_values

                generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

            else:
                # No KV replacement - normal generation with R_V tracking
                generated_ids = inputs["input_ids"].clone()
                v_early_buffer = []
                v_late_buffer = []

                for step in range(max_new_tokens):
                    with capture_v_projection(model, early_layer) as v_early_store:
                        with capture_v_projection(model, late_layer) as v_late_store:
                            outputs = model(
                                input_ids=generated_ids,
                                use_cache=False  # No KV caching for clean measurement
                            )

                    v_e = v_early_store.get("v")
                    v_l = v_late_store.get("v")

                    if v_e is not None:
                        v_early_buffer.append(v_e[:, -1:, :].detach().cpu())
                    if v_l is not None:
                        v_late_buffer.append(v_l[:, -1:, :].detach().cpu())

                    if len(v_early_buffer) >= rv_window:
                        v_early_window = torch.cat(v_early_buffer[-rv_window:], dim=1)
                        v_late_window = torch.cat(v_late_buffer[-rv_window:], dim=1)

                        pr_early = participation_ratio(v_early_window[0], rv_window)
                        pr_late = participation_ratio(v_late_window[0], rv_window)

                        if pr_early > 0 and not np.isnan(pr_early) and not np.isnan(pr_late):
                            rv = pr_late / pr_early
                            rv_trajectory.append(rv)

                    next_token_logits = outputs.logits[:, -1, :]
                    next_token = torch.multinomial(
                        torch.softmax(next_token_logits / temperature, dim=-1), 1
                    )
                    generated_ids = torch.cat([generated_ids, next_token], dim=1)

                    if next_token.item() == tokenizer.eos_token_id:
                        break

                generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # Compute metadata
        coherence = compute_coherence(generated_text)
        domain_info = classify_domain(generated_text)
        recursion_score = score_recursion_regex(generated_text)

        metadata = {
            "coherence": coherence,
            "domain": domain_info["domain"],
            "philosophical_markers": domain_info["philosophical_markers"],
            "task_markers": domain_info["task_markers"],
            "recursion_score": recursion_score,
            "n_tokens_generated": len(generated_ids[0]) - len(inputs["input_ids"][0]),
        }

        return generated_text, rv_trajectory, metadata

    finally:
        for patcher in patchers:
            patcher.remove()


def run_c2_rv_measurement_from_config(cfg: Dict[str, Any], run_dir: Path) -> ExperimentResult:
    """
    Run C2 behavioral transfer experiment with R_V measurement.

    Tests three conditions:
    1. Baseline (no intervention)
    2. KV-only (full KV swap, no steering)
    3. C2 Full (KV + H18/H26 steering + cascade)
    """
    print("=" * 80)
    print("C2 + R_V MEASUREMENT EXPERIMENT")
    print("=" * 80)

    params = cfg.get("params", {})
    model_name = params.get("model", "mistralai/Mistral-7B-v0.1")
    n_prompts = params.get("n_prompts", 20)
    n_recursive = params.get("n_recursive", 10)
    max_new_tokens = params.get("max_new_tokens", 100)
    temperature = params.get("temperature", 0.7)
    early_layer = params.get("early_layer", 5)
    late_layer = params.get("late_layer", 27)
    rv_window = params.get("rv_window", 16)
    seed = params.get("seed", 42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(seed)

    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Prompts: {n_prompts}")
    print(f"Max tokens: {max_new_tokens}")

    # Load model
    model, tokenizer = load_model(model_name, device=device)
    model.eval()

    # Initialize BaselineMetricsSuite for Nanda-standard metrics
    baseline_suite = BaselineMetricsSuite(model, tokenizer, device=device)

    # Load prompts
    loader = PromptLoader()

    # Load baseline prompts from multiple baseline groups
    baseline_prompts = []
    for baseline_group in ["baseline_math", "baseline_factual", "baseline_creative"]:
        baseline_prompts.extend(loader.get_by_group(baseline_group, limit=n_prompts // 3 + 1))
    baseline_prompts = baseline_prompts[:n_prompts]

    # Load recursive prompts from L3-L5 groups
    recursive_prompts = []
    for group in ["L3_deeper", "L4_full", "L5_refined"]:
        recursive_prompts.extend(loader.get_by_group(group, limit=n_recursive // 3 + 1))
    recursive_prompts = recursive_prompts[:n_recursive]

    print(f"Loaded {len(baseline_prompts)} baseline prompts")
    print(f"Loaded {len(recursive_prompts)} recursive prompts")

    # Compute steering vector
    print("\nComputing steering vector...")
    steering_vector = compute_steering_vector(
        model, tokenizer, recursive_prompts, baseline_prompts[:n_recursive],
        layer_idx=late_layer, device=device
    )
    print(f"Steering vector norm: {steering_vector.norm().item():.4f}")

    # Test configs - check for ablation config override
    config_override = params.get("config_override", None)
    
    if config_override:
        # Ablation mode: test only the specified config
        ablation_name = cfg.get("ablation", "ablation")
        ablation_config = {
            "name": f"Ablation_{ablation_name}",
            "head_target": config_override.get("head_target", "none"),
            "kv_strategy": config_override.get("kv_strategy", "none"),
            "residual_alphas": config_override.get("residual_alphas"),
            "vproj_alpha": config_override.get("vproj_alpha", 0.0),
        }
        configs = {
            "baseline": BASELINE_CONFIG,
            ablation_name: ablation_config,
        }
    else:
        # Standard mode: test all three configs
        configs = {
            "baseline": BASELINE_CONFIG,
            "kv_only": KV_ONLY_CONFIG,
            "c2_full": C2_CONFIG,
        }

    results = []

    for config_name, config in configs.items():
        print(f"\n{'='*60}")
        print(f"Running: {config['name']}")
        print(f"{'='*60}")

        for prompt_idx, prompt in enumerate(tqdm(baseline_prompts, desc=f"{config_name}")):
            try:
                rec_prompt = recursive_prompts[prompt_idx % len(recursive_prompts)]

                generated_text, rv_trajectory, metadata = generate_with_rv_tracking(
                    model, tokenizer, prompt, rec_prompt, config, steering_vector,
                    max_new_tokens=max_new_tokens, temperature=temperature,
                    early_layer=early_layer, late_layer=late_layer,
                    rv_window=rv_window, device=device
                )

                # Compute R_V stats
                rv_mean = np.mean(rv_trajectory) if rv_trajectory else float("nan")
                rv_min = np.min(rv_trajectory) if rv_trajectory else float("nan")
                rv_max = np.max(rv_trajectory) if rv_trajectory else float("nan")
                rv_final = rv_trajectory[-1] if rv_trajectory else float("nan")

                # Compute baseline metrics on the generated text (post-generation analysis)
                # Note: This measures the final state, not during generation
                try:
                    baseline_metrics = baseline_suite.compute_all(generated_text, return_trajectories=False)
                    logit_diff = baseline_metrics.logit_diff
                    crystallization_layer = baseline_metrics.crystallization_layer
                    mode_score = baseline_metrics.mode_score_m
                    residual_norm_late = baseline_metrics.residual_norm_late
                except Exception as e:
                    print(f"  Warning: Baseline metrics failed: {e}")
                    logit_diff = float("nan")
                    crystallization_layer = None
                    mode_score = float("nan")
                    residual_norm_late = float("nan")

                results.append({
                    "config": config_name,
                    "config_full_name": config["name"],
                    "prompt_idx": prompt_idx,
                    "prompt": prompt[:100],  # Truncate for storage
                    "generated_text": generated_text,
                    "rv_mean": rv_mean,
                    "rv_min": rv_min,
                    "rv_max": rv_max,
                    "rv_final": rv_final,
                    "rv_trajectory_len": len(rv_trajectory),
                    "coherence": metadata["coherence"],
                    "domain": metadata["domain"],
                    "philosophical_markers": metadata["philosophical_markers"],
                    "task_markers": metadata["task_markers"],
                    "recursion_score": metadata["recursion_score"],
                    "n_tokens": metadata["n_tokens_generated"],
                    # Nanda-standard metrics
                    "logit_diff": logit_diff,
                    "crystallization_layer": crystallization_layer,
                    "mode_score_m": mode_score,
                    "residual_norm_late": residual_norm_late,
                })

            except Exception as e:
                print(f"  ERROR on prompt {prompt_idx}: {e}")
                results.append({
                    "config": config_name,
                    "config_full_name": config["name"],
                    "prompt_idx": prompt_idx,
                    "prompt": prompt[:100],
                    "generated_text": f"ERROR: {e}",
                    "rv_mean": float("nan"),
                    "rv_min": float("nan"),
                    "rv_max": float("nan"),
                    "rv_final": float("nan"),
                    "rv_trajectory_len": 0,
                    "coherence": 0.0,
                    "domain": "error",
                    "philosophical_markers": 0,
                    "task_markers": 0,
                    "recursion_score": 0.0,
                    "n_tokens": 0,
                    "logit_diff": float("nan"),
                    "crystallization_layer": None,
                    "mode_score_m": float("nan"),
                    "residual_norm_late": float("nan"),
                })

    # Save results
    df = pd.DataFrame(results)
    results_csv = run_dir / "c2_rv_measurement.csv"
    df.to_csv(results_csv, index=False)

    # Compute summary with statistics
    from scipy import stats
    
    summary_by_config = {}
    for config_name in configs.keys():
        config_df = df[df["config"] == config_name].copy()
        
        # R_V statistics
        rv_values = config_df["rv_mean"].dropna()
        rv_mean = float(rv_values.mean())
        rv_std = float(rv_values.std())
        rv_n = len(rv_values)
        
        # 95% CI for R_V
        if rv_n > 1:
            rv_ci = stats.t.interval(0.95, df=rv_n-1, loc=rv_mean, scale=stats.sem(rv_values))
            rv_ci_low, rv_ci_high = float(rv_ci[0]), float(rv_ci[1])
        else:
            rv_ci_low, rv_ci_high = float("nan"), float("nan")
        
        # Logit diff statistics
        logit_diff_values = config_df["logit_diff"].dropna()
        logit_diff_mean = float(logit_diff_values.mean()) if len(logit_diff_values) > 0 else float("nan")
        logit_diff_std = float(logit_diff_values.std()) if len(logit_diff_values) > 0 else float("nan")
        
        summary_by_config[config_name] = {
            "rv_mean": rv_mean,
            "rv_std": rv_std,
            "rv_ci_95_low": rv_ci_low,
            "rv_ci_95_high": rv_ci_high,
            "rv_min": float(config_df["rv_min"].mean()),
            "coherence": float(config_df["coherence"].mean()),
            "philosophical_pct": float((config_df["domain"] == "philosophical").mean() * 100),
            "task_pct": float((config_df["domain"] == "task").mean() * 100),
            "n_prompts": len(config_df),
            # Nanda-standard metrics
            "logit_diff_mean": logit_diff_mean,
            "logit_diff_std": logit_diff_std,
            "crystallization_layer_mean": float(config_df["crystallization_layer"].dropna().mean()) if config_df["crystallization_layer"].notna().any() else None,
            "mode_score_m_mean": float(config_df["mode_score_m"].dropna().mean()) if config_df["mode_score_m"].notna().any() else None,
        }
    
    # Paired t-test: C2_full vs baseline
    baseline_df = df[df["config"] == "baseline"]
    c2_df = df[df["config"] == "c2_full"]
    
    # Match by prompt_idx for paired test
    merged = baseline_df.merge(c2_df, on="prompt_idx", suffixes=("_baseline", "_c2"))
    rv_baseline = merged["rv_mean_baseline"].dropna()
    rv_c2 = merged["rv_mean_c2"].dropna()
    
    statistics = {}
    if len(rv_baseline) == len(rv_c2) and len(rv_baseline) > 1:
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(rv_baseline, rv_c2)
        
        # Cohen's d (effect size)
        pooled_std = np.sqrt((rv_baseline.std()**2 + rv_c2.std()**2) / 2)
        cohens_d = (rv_baseline.mean() - rv_c2.mean()) / pooled_std if pooled_std > 0 else 0
        
        statistics = {
            "baseline_vs_c2": {
                "n_pairs": len(rv_baseline),
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "cohens_d": float(cohens_d),
                "rv_baseline_mean": float(rv_baseline.mean()),
                "rv_c2_mean": float(rv_c2.mean()),
                "rv_delta_mean": float(rv_baseline.mean() - rv_c2.mean()),
            }
        }

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Config':<20} {'R_V Mean':<12} {'R_V Std':<12} {'Phil%':<10} {'Task%':<10}")
    print("-" * 80)
    for config_name, stats in summary_by_config.items():
        print(f"{config_name:<20} {stats['rv_mean']:<12.4f} {stats['rv_std']:<12.4f} "
              f"{stats['philosophical_pct']:<10.1f} {stats['task_pct']:<10.1f}")

    # Save full outputs
    outputs_dir = run_dir / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    for config_name in configs.keys():
        config_df = df[df["config"] == config_name]
        output_file = outputs_dir / f"{config_name}_outputs.txt"

        with open(output_file, "w") as f:
            f.write(f"CONFIG: {configs[config_name]['name']}\n")
            f.write("=" * 80 + "\n\n")

            for _, row in config_df.iterrows():
                f.write(f"PROMPT {row['prompt_idx']}:\n{row['prompt']}\n\n")
                f.write(f"GENERATED:\n{row['generated_text']}\n\n")
                f.write(f"R_V: mean={row['rv_mean']:.4f}, min={row['rv_min']:.4f}, "
                       f"final={row['rv_final']:.4f}\n")
                f.write(f"Domain: {row['domain']}, Coherence: {row['coherence']:.2f}\n")
                f.write("-" * 80 + "\n\n")

    summary = {
        "experiment": "c2_rv_measurement",
        "model": model_name,
        "n_prompts": n_prompts,
        "by_config": summary_by_config,
        "statistics": statistics,
        "artifacts": {
            "csv": str(results_csv),
            "outputs_dir": str(outputs_dir),
        },
    }

    # Save summary JSON
    summary_json = run_dir / "summary.json"
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    return ExperimentResult(summary=summary)


if __name__ == "__main__":
    from datetime import datetime

    cfg = {
        "params": {
            "model": "mistralai/Mistral-7B-v0.1",
            "n_prompts": 20,
            "n_recursive": 10,
            "max_new_tokens": 100,
            "temperature": 0.7,
            "early_layer": 5,
            "late_layer": 27,
            "rv_window": 16,
            "seed": 42,
        }
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("results/phase1_mechanism/runs") / f"{timestamp}_c2_rv_measurement"
    run_dir.mkdir(parents=True, exist_ok=True)

    result = run_c2_rv_measurement_from_config(cfg, run_dir)
    print(f"\nResults saved to: {run_dir}")
