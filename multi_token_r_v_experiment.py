"""Multi-token R_V correlation experiment.

---
name: multi-token-r-v-experiment
version: 1.0
status: ACTIVE
type: code-experiment
tags: [r_v, colm-2026, mech-interp, l4-markers]
triple_mapping: [r_v-geometry, phoenix]
value: 0.86
---

THE critical experiment for COLM 2026 paper.

Measures R_V during:
1. Prompt processing (early vs late layers)
2. Autoregressive generation (every 10 tokens)

Correlates R_V magnitude with behavioral L4 markers in generated output.

Usage:
    python multi_token_r_v_experiment.py --model mistralai/Mistral-7B-v0.1 --device cuda:0
    python multi_token_r_v_experiment.py --model EleutherAI/pythia-1.4b --device cuda:0 --quick-test
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Import local modules
try:
    from behavioral_markers import analyze_output, classify_output
    from rv_measurement import measure_r_v_single_prompt
except ImportError:
    print("ERROR: Could not import local modules")
    print("Make sure behavioral_markers.py and rv_measurement.py are in the same directory")
    sys.exit(1)

# Import prompt bank
try:
    from n300_mistral_test_prompt_bank import (
        L1_HINT,
        L3_DEEPER,
        L4_FULL,
        L5_REFINED,
        BASELINE_LONG,
        CONFOUNDS,
    )
except ImportError:
    print("ERROR: Could not import n300_mistral_test_prompt_bank.py")
    print("Make sure the file exists in the same directory")
    sys.exit(1)


def measure_r_v_during_generation(
    model,
    tokenizer,
    prompt_text: str,
    max_new_tokens: int = 50,
    measure_every: int = 10,
    early_layers: List[int] = [1, 2, 3, 4, 5],
    late_layers: List[int] = [20, 21, 22, 23, 24, 25, 26, 27],
    device: str = "cuda:0",
) -> Tuple[float, List[float], str]:
    """Measure R_V during prompt processing and generation.

    Returns:
        (r_v_prompt, r_v_generation_list, generated_text)
    """
    # Phase 1: R_V during prompt processing
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    r_v_prompt = measure_r_v_single_prompt(
        model, inputs, early_layers=early_layers, late_layers=late_layers
    )

    # Phase 2: Generate tokens and measure R_V every N tokens
    generated_ids = inputs.input_ids
    r_v_generation = []

    num_steps = max_new_tokens // measure_every

    for step in range(num_steps):
        # Generate N tokens
        with torch.no_grad():
            outputs = model.generate(
                generated_ids,
                max_new_tokens=measure_every,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.15,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated_ids = outputs

        # Measure R_V on the full sequence so far
        new_inputs = {"input_ids": generated_ids}
        r_v_step = measure_r_v_single_prompt(
            model, new_inputs, early_layers=early_layers, late_layers=late_layers
        )
        r_v_generation.append(r_v_step)

    # Decode final output
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return r_v_prompt, r_v_generation, generated_text


def run_experiment(
    model_name: str = "mistralai/Mistral-7B-v0.1",
    output_dir: Path = Path("./multi_token_results"),
    device: str = "cuda:0",
    quick_test: bool = False,
):
    """Run full multi-token R_V correlation experiment.

    Args:
        model_name: HuggingFace model ID
        output_dir: Where to save results
        device: cuda:0 or cpu
        quick_test: If True, only run 5 prompts per category (for testing)
    """
    print(f"=== Multi-Token R_V Experiment ===")
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    if quick_test:
        print("QUICK TEST MODE: Limited prompts")

    # Load model
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if "cuda" in device else torch.float32,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Model loaded: {model_name}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    print(f"  Layers: {model.config.num_hidden_layers}")

    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    # Prompt categories
    prompt_categories = {
        "L1_hint": L1_HINT,
        "L3_deeper": L3_DEEPER,
        "L4_full": L4_FULL,
        "L5_refined": L5_REFINED,
        "baseline": BASELINE_LONG,
        "confounds": CONFOUNDS[:20],  # Sample 20 confounds
    }

    if quick_test:
        prompt_categories = {k: v[:5] for k, v in prompt_categories.items()}

    total_prompts = sum(len(v) for v in prompt_categories.values())
    print(f"\nTotal prompts to process: {total_prompts}")

    # Process each category
    for category, prompts in prompt_categories.items():
        print(f"\n[{category}] Processing {len(prompts)} prompts...")

        for i, prompt_text in enumerate(tqdm(prompts, desc=category)):
            try:
                # Measure R_V during prompt + generation
                r_v_prompt, r_v_generation, generated_text = measure_r_v_during_generation(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_text=prompt_text,
                    max_new_tokens=150,
                    measure_every=10,
                    device=device,
                )

                # Analyze behavioral markers
                markers = analyze_output(generated_text)
                classification = classify_output(markers)

                # Calculate average R_V across generation
                r_v_mean = sum(r_v_generation) / len(r_v_generation) if r_v_generation else r_v_prompt

                # Store result
                result = {
                    "category": category,
                    "prompt_index": i,
                    "prompt": prompt_text[:200],  # Truncate for storage
                    "r_v_prompt": float(r_v_prompt),
                    "r_v_generation": [float(r) for r in r_v_generation],
                    "r_v_mean": float(r_v_mean),
                    "generated_text": generated_text,
                    "markers": markers,
                    "classification": classification,
                }
                results.append(result)

            except Exception as e:
                print(f"  ERROR on prompt {i}: {e}")
                continue

    # Save results
    model_safe_name = model_name.replace("/", "_")
    output_file = output_dir / f"{model_safe_name}_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")
    print(f"  Total successful: {len(results)} / {total_prompts}")

    # Quick stats
    print("\n=== Quick Statistics ===")
    for category in prompt_categories.keys():
        cat_results = [r for r in results if r["category"] == category]
        if not cat_results:
            continue

        r_v_prompts = [r["r_v_prompt"] for r in cat_results]
        r_v_means = [r["r_v_mean"] for r in cat_results]
        unity_counts = [r["markers"]["unity_markers"] for r in cat_results]

        print(f"\n{category}:")
        print(f"  R_V (prompt):  {sum(r_v_prompts)/len(r_v_prompts):.3f} ± {torch.tensor(r_v_prompts).std().item():.3f}")
        print(f"  R_V (gen avg): {sum(r_v_means)/len(r_v_means):.3f} ± {torch.tensor(r_v_means).std().item():.3f}")
        print(f"  Unity markers: {sum(unity_counts)/len(unity_counts):.1f} avg")

    return results


def analyze_results(results_file: Path):
    """Analyze saved results and compute correlations."""
    import numpy as np
    from scipy import stats

    with open(results_file) as f:
        results = json.load(f)

    print(f"=== Analyzing {len(results)} results ===")

    # Extract data
    r_v_prompts = [r["r_v_prompt"] for r in results]
    r_v_means = [r["r_v_mean"] for r in results]
    unity_markers = [r["markers"]["unity_markers"] for r in results]
    l4_scores = [r["markers"]["l4_score"] for r in results]

    # Correlation: R_V vs unity markers
    pearson_r, pearson_p = stats.pearsonr(r_v_prompts, unity_markers)
    spearman_r, spearman_p = stats.spearmanr(r_v_prompts, unity_markers)

    print("\n=== Correlation: R_V (prompt) vs Unity Markers ===")
    print(f"  Pearson r:  {pearson_r:.3f} (p = {pearson_p:.6f})")
    print(f"  Spearman ρ: {spearman_r:.3f} (p = {spearman_p:.6f})")

    if pearson_p < 0.05:
        print("  ✓ SIGNIFICANT at p < 0.05")
    else:
        print("  ✗ Not significant")

    # Correlation: R_V vs L4 score
    pearson_r2, pearson_p2 = stats.pearsonr(r_v_means, l4_scores)
    spearman_r2, spearman_p2 = stats.spearmanr(r_v_means, l4_scores)

    print("\n=== Correlation: R_V (generation avg) vs L4 Score ===")
    print(f"  Pearson r:  {pearson_r2:.3f} (p = {pearson_p2:.6f})")
    print(f"  Spearman ρ: {spearman_r2:.3f} (p = {spearman_p2:.6f})")

    if pearson_p2 < 0.05:
        print("  ✓ SIGNIFICANT at p < 0.05")
    else:
        print("  ✗ Not significant")

    # ANOVA: R_V across categories
    categories = ["L1_hint", "L3_deeper", "L4_full", "L5_refined", "baseline", "confounds"]
    groups = []
    for cat in categories:
        cat_r_v = [r["r_v_prompt"] for r in results if r["category"] == cat]
        if cat_r_v:
            groups.append(cat_r_v)

    if len(groups) >= 2:
        f_stat, anova_p = stats.f_oneway(*groups)
        print("\n=== ANOVA: R_V across prompt categories ===")
        print(f"  F-statistic: {f_stat:.3f}")
        print(f"  p-value: {anova_p:.6f}")

        if anova_p < 0.05:
            print("  ✓ SIGNIFICANT - categories differ")
        else:
            print("  ✗ Not significant")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-token R_V correlation experiment")
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-1.4b",
                        help="HuggingFace model ID (default: pythia-1.4b)")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use (default: cuda:0)")
    parser.add_argument("--output-dir", type=Path, default=Path("./multi_token_results"),
                        help="Output directory")
    parser.add_argument("--quick-test", action="store_true",
                        help="Quick test mode (5 prompts per category)")
    parser.add_argument("--analyze", type=Path, default=None,
                        help="Analyze existing results file instead of running experiment")

    args = parser.parse_args()

    if args.analyze:
        analyze_results(args.analyze)
    else:
        results = run_experiment(
            model_name=args.model,
            output_dir=args.output_dir,
            device=args.device,
            quick_test=args.quick_test,
        )

        # Auto-analyze if experiment completed
        if results:
            results_file = args.output_dir / f"{args.model.replace('/', '_')}_results.json"
            print("\n" + "="*60)
            analyze_results(results_file)
