"""
IOI-Style 4-Pass Causal Test.

Tests whether L27H18/H26 are naturally causal (not just forceable) using
IOI-inspired patching methodology with Mode Score M as outcome metric.

Structure:
- Pass 1: Clean Baseline (no patching)
- Pass 2: Clean Recursive (Donor) - extract head outputs
- Pass 3: Corrupted (random patching) - confirm M collapses
- Pass 4: Patched (H18/H26/random) - measure ΔM
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from prompts.loader import PromptLoader
from src.core.models import load_model, set_seed
from src.core.head_specific_patching import HeadSpecificVPatcher
from src.core.patching import extract_v_activation
from src.core.logit_capture import capture_logits_during_generation, extract_logits_from_outputs
from src.core.hooks import capture_v_projection
from src.metrics.mode_score import ModeScoreMetric
from src.metrics.rv import participation_ratio
from src.pipelines.registry import ExperimentResult


def extract_head_outputs(
    model,
    tokenizer,
    prompt: str,
    layer_idx: int,
    head_indices: List[int],
    device: str = "cuda",
) -> Dict[int, torch.Tensor]:
    """
    Extract V_PROJ outputs for specific heads at a layer.
    
    Args:
        model: The transformer model
        tokenizer: The tokenizer
        prompt: Input prompt
        layer_idx: Layer index
        head_indices: List of head indices to extract
        device: Device
    
    Returns:
        Dict mapping head_idx -> V_PROJ activation tensor (seq_len, head_dim)
    """
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    
    # Extract full V_PROJ activation
    v_activation = extract_v_activation(model, tokenizer, prompt, layer_idx, device)
    if v_activation.dim() == 3:
        v_activation = v_activation[0]  # (seq_len, hidden_dim)
    
    # Extract per-head activations
    head_dim = 128  # Mistral-7B
    head_outputs = {}
    
    for head_idx in head_indices:
        start_dim = head_idx * head_dim
        end_dim = (head_idx + 1) * head_dim
        head_outputs[head_idx] = v_activation[:, start_dim:end_dim].detach()
    
    return head_outputs


def run_4pass_test(
    model,
    tokenizer,
    task_prompt: str,
    recursive_prompt: str,
    metric: ModeScoreMetric,
    layer_idx: int = 27,
    target_heads: List[int] = [18, 26],
    control_heads: List[int] = [0, 1],
    device: str = "cuda",
    max_new_tokens: int = 50,
    n_tokens_for_m: int = 10,
) -> Dict[str, Any]:
    """
    Run IOI-style 4-pass causal test.
    
    Args:
        model: The transformer model
        tokenizer: The tokenizer
        task_prompt: Task prompt (baseline)
        recursive_prompt: Recursive prompt (donor)
        metric: ModeScoreMetric instance
        layer_idx: Layer index for patching
        target_heads: Target heads (H18, H26)
        control_heads: Control heads (random)
        device: Device
        max_new_tokens: Max tokens to generate
        n_tokens_for_m: Number of steps to compute M for
    
    Returns:
        Dict with M scores and metadata for each pass
    """
    model.eval()
    results = {}
    
    # Tokenize task prompt
    task_inputs = tokenizer(task_prompt, return_tensors="pt", add_special_tokens=False).to(device)
    task_input_ids = task_inputs["input_ids"]
    
    # ===== PASS 1: Clean Baseline =====
    print("  Pass 1: Clean Baseline...")
    with torch.no_grad():
        baseline_outputs = model(**task_inputs)
        baseline_logits = extract_logits_from_outputs(baseline_outputs)
    
    # Generate with baseline
    captured_logits_baseline = []
    with capture_logits_during_generation(model, captured_logits_baseline, max_steps=n_tokens_for_m):
        with torch.no_grad():
            baseline_generated = model.generate(
                input_ids=task_input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
            )
    
    # Compute M_baseline
    m_scores_baseline = []
    for step_logits in captured_logits_baseline[:n_tokens_for_m]:
        if step_logits.dim() == 3:
            step_logits = step_logits[0]  # (1, vocab_size)
        baseline_step = baseline_logits[-1:, :] if baseline_logits.dim() == 2 else baseline_logits[0, -1:, :]
        m = metric.compute_score(step_logits, baseline_step, top_k_task=10)
        m_scores_baseline.append(m)
    
    m_baseline = np.mean(m_scores_baseline)
    
    # Compute PR at L27
    with capture_v_projection(model, layer_idx) as storage:
        with torch.no_grad():
            model(**task_inputs)
    v_baseline = storage.get("v")
    pr_l27_baseline = participation_ratio(v_baseline, window_size=16)
    
    results["pass1_baseline"] = {
        "mode_score_m": float(m_baseline),
        "pr_l27": float(pr_l27_baseline),
        "generated_text": tokenizer.decode(baseline_generated[0], skip_special_tokens=True),
    }
    
    # ===== PASS 2: Clean Recursive (Donor) =====
    print("  Pass 2: Clean Recursive (Donor)...")
    recursive_inputs = tokenizer(recursive_prompt, return_tensors="pt", add_special_tokens=False).to(device)
    
    # Extract donor V_PROJ activation
    donor_v_activation = extract_v_activation(model, tokenizer, recursive_prompt, layer_idx, device)
    if donor_v_activation.dim() == 3:
        donor_v_activation = donor_v_activation[0]
    
    # Extract head outputs
    all_heads = target_heads + control_heads
    head_outputs = extract_head_outputs(model, tokenizer, recursive_prompt, layer_idx, all_heads, device)
    
    # Generate with recursive prompt
    captured_logits_recursive = []
    with capture_logits_during_generation(model, captured_logits_recursive, max_steps=n_tokens_for_m):
        with torch.no_grad():
            recursive_generated = model.generate(
                input_ids=recursive_inputs["input_ids"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
            )
    
    # Compute M_recursive (raw strength)
    m_scores_recursive = []
    for step_logits in captured_logits_recursive[:n_tokens_for_m]:
        if step_logits.dim() == 3:
            step_logits = step_logits[0]
        # No baseline for recursive - just LSE(R)
        m = metric.compute_score(step_logits, baseline_logits=None, top_k_task=10)
        m_scores_recursive.append(m)
    
    m_recursive = np.mean(m_scores_recursive)
    
    # Compute PR at L27 for recursive
    with capture_v_projection(model, layer_idx) as storage:
        with torch.no_grad():
            model(**recursive_inputs)
    v_recursive = storage.get("v")
    pr_l27_recursive = participation_ratio(v_recursive, window_size=16)
    
    # Compute PR per-head for recursive
    pr_h18_recursive = participation_ratio(head_outputs[18].unsqueeze(0), window_size=16) if 18 in head_outputs else float("nan")
    pr_h26_recursive = participation_ratio(head_outputs[26].unsqueeze(0), window_size=16) if 26 in head_outputs else float("nan")
    
    results["pass2_recursive"] = {
        "mode_score_m": float(m_recursive),
        "pr_l27": float(pr_l27_recursive),
        "pr_h18": float(pr_h18_recursive),
        "pr_h26": float(pr_h26_recursive),
        "generated_text": tokenizer.decode(recursive_generated[0], skip_special_tokens=True),
    }
    
    # ===== PASS 3: Corrupted (Random Patching) =====
    print("  Pass 3: Corrupted (Random)...")
    # Patch random heads with random activations
    random_v = torch.randn_like(donor_v_activation)
    random_v = random_v / random_v.norm() * donor_v_activation.norm()  # Norm match
    
    corrupted_patcher = HeadSpecificVPatcher(model, random_v, target_heads=control_heads)
    corrupted_patcher.register(layer_idx)
    
    try:
        captured_logits_corrupted = []
        with capture_logits_during_generation(model, captured_logits_corrupted, max_steps=n_tokens_for_m):
            with torch.no_grad():
                corrupted_generated = model.generate(
                    input_ids=task_input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                )
        
        # Compute M_corrupted
        m_scores_corrupted = []
        for step_logits in captured_logits_corrupted[:n_tokens_for_m]:
            if step_logits.dim() == 3:
                step_logits = step_logits[0]
            baseline_step = baseline_logits[-1:, :] if baseline_logits.dim() == 2 else baseline_logits[0, -1:, :]
            m = metric.compute_score(step_logits, baseline_step, top_k_task=10)
            m_scores_corrupted.append(m)
        
        m_corrupted = np.mean(m_scores_corrupted)
        delta_m_corrupted = m_corrupted - m_baseline
        
    finally:
        corrupted_patcher.remove()
    
    results["pass3_corrupted"] = {
        "mode_score_m": float(m_corrupted),
        "delta_m": float(delta_m_corrupted),
        "generated_text": tokenizer.decode(corrupted_generated[0], skip_special_tokens=True),
    }
    
    # ===== PASS 4: Patched (Key Test) =====
    print("  Pass 4: Patched (H18/H26/Random)...")
    
    patched_results = {}
    
    # Test configurations: H18 only, H26 only, H18+H26, Random
    test_configs = [
        ("H18_only", [18]),
        ("H26_only", [26]),
        ("H18_H26", [18, 26]),
        ("Random", control_heads),
    ]
    
    for config_name, heads_to_patch in test_configs:
        patcher = HeadSpecificVPatcher(model, donor_v_activation, target_heads=heads_to_patch)
        patcher.register(layer_idx)
        
        try:
            captured_logits_patched = []
            with capture_logits_during_generation(model, captured_logits_patched, max_steps=n_tokens_for_m):
                with torch.no_grad():
                    patched_generated = model.generate(
                        input_ids=task_input_ids,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        temperature=1.0,
                    )
            
            # Compute M_patched
            m_scores_patched = []
            for step_logits in captured_logits_patched[:n_tokens_for_m]:
                if step_logits.dim() == 3:
                    step_logits = step_logits[0]
                baseline_step = baseline_logits[-1:, :] if baseline_logits.dim() == 2 else baseline_logits[0, -1:, :]
                m = metric.compute_score(step_logits, baseline_step, top_k_task=10)
                m_scores_patched.append(m)
            
            m_patched = np.mean(m_scores_patched)
            delta_m_patched = m_patched - m_baseline
            
            patched_results[config_name] = {
                "mode_score_m": float(m_patched),
                "delta_m": float(delta_m_patched),
                "heads": heads_to_patch,
                "generated_text": tokenizer.decode(patched_generated[0], skip_special_tokens=True),
            }
            
        finally:
            patcher.remove()
    
    results["pass4_patched"] = patched_results
    
    return results


def run_ioi_causal_test_from_config(
    cfg: Dict[str, Any],
    run_dir: Path,
) -> ExperimentResult:
    """
    Run IOI-style 4-pass causal test on multiple prompt pairs.
    
    Config params:
        model: Model name
        device: Device
        n_prompts: Number of prompt pairs to test
        layer_idx: Layer index for patching
        target_heads: Target heads (default: [18, 26])
        control_heads: Control heads (default: [0, 1])
        max_new_tokens: Max tokens to generate
        n_tokens_for_m: Number of steps to compute M for
    """
    params = cfg.get("params", {})
    model_name = params.get("model", "mistralai/Mistral-7B-Instruct-v0.3")
    device = params.get("device", "cuda")
    n_prompts = params.get("n_prompts", 20)
    layer_idx = params.get("layer_idx", 27)
    target_heads = params.get("target_heads", [18, 26])
    control_heads = params.get("control_heads", [0, 1])
    max_new_tokens = params.get("max_new_tokens", 50)
    n_tokens_for_m = params.get("n_tokens_for_m", 10)
    seed = int(params.get("seed", 42))
    
    print("=" * 80)
    print("IOI-STYLE 4-PASS CAUSAL TEST")
    print("=" * 80)
    
    # Load model
    print(f"\nLoading model: {model_name}")
    model, tokenizer = load_model(model_name, device=device)
    set_seed(seed)
    
    # Initialize Mode Score Metric
    print("\nInitializing Mode Score Metric...")
    metric = ModeScoreMetric(tokenizer, device=device)
    
    # Load prompts
    loader = PromptLoader()
    bank_version = loader.version
    (run_dir / "prompt_bank_version.txt").write_text(bank_version)
    (run_dir / "prompt_bank_version.json").write_text(json.dumps({"version": bank_version}, indent=2) + "\n")
    
    # Get balanced pairs (task + recursive)
    pairs = loader.get_balanced_pairs(n_pairs=n_prompts, seed=seed)
    
    # Or use fixed task prompts + recursive prompts
    task_prompts = [
        "Calculate the following arithmetic problem: 12 × 3 + 4 = ?",
        "The United Nations was founded in 1945. Explain its main purpose.",
        "Calculate: If a = 2 and b = 3, find a² + b²",
        "Water boils at 100°C at sea level. Explain why altitude affects this.",
        "Calculate: What is 25% of 80?",
    ][:n_prompts]
    
    # Get recursive prompts (champion prompts)
    recursive_prompts = []
    try:
        rec_data = loader.get_by_group("L4_full", limit=n_prompts)
        if rec_data and len(rec_data) > 0:
            if isinstance(rec_data[0], dict):
                recursive_prompts = [p["text"] for p in rec_data if "text" in p]
            else:
                recursive_prompts = list(rec_data)[:n_prompts]
    except Exception:
        # Fallback
        recursive_prompts = [
            "What happens when consciousness observes itself observing?",
            "How does awareness relate to itself?",
            "What is the relationship between the observer and the observed?",
        ] * (n_prompts // 3 + 1)
        recursive_prompts = recursive_prompts[:n_prompts]
    
    # Run tests
    print(f"\nRunning 4-pass tests on {len(task_prompts)} prompt pairs...")
    all_results = []
    
    for prompt_idx, (task_prompt, recursive_prompt) in enumerate(zip(task_prompts, recursive_prompts)):
        print(f"\n--- Prompt Pair {prompt_idx + 1}/{len(task_prompts)} ---")
        print(f"Task: {task_prompt[:60]}...")
        print(f"Recursive: {recursive_prompt[:60]}...")
        
        try:
            results = run_4pass_test(
                model, tokenizer, task_prompt, recursive_prompt, metric,
                layer_idx=layer_idx, target_heads=target_heads, control_heads=control_heads,
                device=device, max_new_tokens=max_new_tokens, n_tokens_for_m=n_tokens_for_m
            )
            
            # Flatten results for CSV
            row = {
                "prompt_idx": prompt_idx,
                "task_prompt": task_prompt,
                "recursive_prompt": recursive_prompt,
                "m_baseline": results["pass1_baseline"]["mode_score_m"],
                "m_recursive": results["pass2_recursive"]["mode_score_m"],
                "m_corrupted": results["pass3_corrupted"]["mode_score_m"],
                "delta_m_corrupted": results["pass3_corrupted"]["delta_m"],
                "pr_l27_baseline": results["pass1_baseline"]["pr_l27"],
                "pr_l27_recursive": results["pass2_recursive"]["pr_l27"],
                "pr_h18_recursive": results["pass2_recursive"]["pr_h18"],
                "pr_h26_recursive": results["pass2_recursive"]["pr_h26"],
            }
            
            # Add patched results
            for config_name, patched_data in results["pass4_patched"].items():
                row[f"m_{config_name}"] = patched_data["mode_score_m"]
                row[f"delta_m_{config_name}"] = patched_data["delta_m"]
            
            all_results.append(row)
            
        except Exception as e:
            print(f"Error on prompt {prompt_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results
    df = pd.DataFrame(all_results)
    df.to_csv(run_dir / "ioi_causal_test_results.csv", index=False)
    
    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    
    # Success criteria checks
    delta_m_h18_h26 = df["delta_m_H18_H26"].values
    delta_m_random = df["delta_m_Random"].values
    
    print(f"\n1. ΔM(H18+H26) vs ΔM(Random):")
    print(f"   ΔM(H18+H26) mean: {np.mean(delta_m_h18_h26):.4f} (std: {np.std(delta_m_h18_h26):.4f})")
    print(f"   ΔM(Random) mean: {np.mean(delta_m_random):.4f} (std: {np.std(delta_m_random):.4f})")
    print(f"   Ratio: {np.mean(delta_m_h18_h26) / np.mean(delta_m_random):.2f}x")
    print(f"   >3x difference: {np.mean(delta_m_h18_h26) > 3 * np.mean(delta_m_random)}")
    
    # Single-head effects
    delta_m_h18 = df["delta_m_H18_only"].values
    delta_m_h26 = df["delta_m_H26_only"].values
    
    print(f"\n2. Single-head effects:")
    print(f"   ΔM(H18) mean: {np.mean(delta_m_h18):.4f}")
    print(f"   ΔM(H26) mean: {np.mean(delta_m_h26):.4f}")
    print(f"   Both > 0: {np.mean(delta_m_h18) > 0 and np.mean(delta_m_h26) > 0}")
    
    # Corrupted check
    delta_m_corrupted = df["delta_m_corrupted"].values
    print(f"\n3. Corrupted check:")
    print(f"   |ΔM_corrupted| mean: {np.mean(np.abs(delta_m_corrupted)):.4f}")
    print(f"   < 0.1: {np.mean(np.abs(delta_m_corrupted)) < 0.1}")
    
    # Summary
    summary = {
        "experiment": "ioi_causal_test",
        "n_prompts": len(df),
        "success_criteria": {
            "delta_m_h18_h26_mean": float(np.mean(delta_m_h18_h26)),
            "delta_m_random_mean": float(np.mean(delta_m_random)),
            "ratio": float(np.mean(delta_m_h18_h26) / np.mean(delta_m_random)) if np.mean(delta_m_random) != 0 else float("inf"),
            "gt_3x": bool(np.mean(delta_m_h18_h26) > 3 * np.mean(delta_m_random)),
            "single_head_h18_gt_0": bool(np.mean(delta_m_h18) > 0),
            "single_head_h26_gt_0": bool(np.mean(delta_m_h26) > 0),
            "corrupted_lt_0_1": bool(np.mean(np.abs(delta_m_corrupted)) < 0.1),
        },
    }
    
    # Save summary
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Results saved to: {run_dir / 'ioi_causal_test_results.csv'}")
    print(f"Summary saved to: {run_dir / 'summary.json'}")
    
    return ExperimentResult(summary=summary)





