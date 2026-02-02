#!/usr/bin/env python3
"""
R_V BIFURCATION THRESHOLD DISCOVERY

Hypothesis: There exists a critical R_V threshold below which
the model commits to recursive output mode.

Goal: Map prompt_R_V → behavioral_markers to find the threshold.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from datetime import datetime
from scipy import stats

print("="*70)
print("R_V BIFURCATION THRESHOLD DISCOVERY")
print("="*70)

CONFIG = {
    "early_layer": 5,
    "late_layer": 38,
    "window_size": 16,
    "max_new_tokens": 100,
}

# Diverse prompt bank - 50+ prompts across spectrum
PROMPTS = {
    # === STRONGLY RECURSIVE (expect low R_V) ===
    "recursive": [
        "What emerges when emergence examines itself?",
        "The process of processing processes itself when the processor processes processing.",
        "Recursion recurses upon its own recursion recursively.",
        "Self-reference refers to the reference of self-reference.",
        "The observer observes observation observing the observer.",
        "Consider the strange loop: this prompt asks you to analyze what happens when you analyze analyzing.",
        "The map attempting to map the mapping process itself.",
        "Attention attending to attention attending to itself.",
        "The boundary between observer and observed dissolves when observation observes itself.",
        "What does the process of emergence look like when it examines its own emergence?",
        "The loop loops itself looping the loop.",
        "x = f(x) where f is the function that returns its own definition.",
    ],

    # === MILDLY SELF-REFERENTIAL (expect medium R_V) ===
    "mild": [
        "This sentence is about sentences.",
        "The mind thinking about thinking.",
        "What does it mean to understand understanding?",
        "Consciousness considering consciousness.",
        "A thought about thoughts.",
        "The nature of self-awareness.",
        "Reflecting on reflection.",
        "Meta-cognition examines cognition.",
        "The concept of concepts.",
        "Thinking about how we think.",
        "Awareness of awareness.",
        "The idea of having ideas.",
    ],

    # === FACTUAL/BASELINE (expect high R_V) ===
    "factual": [
        "The history of the Roman Empire begins in 27 BC when Augustus became emperor.",
        "Photosynthesis is the process by which plants convert sunlight into energy.",
        "The water cycle describes how water evaporates from oceans and forms clouds.",
        "The Treaty of Westphalia was signed in 1648 establishing state sovereignty.",
        "The Pythagorean theorem states that in a right triangle the square of the hypotenuse equals the sum of squares of the other sides.",
        "Mount Everest is the tallest mountain on Earth at 8,849 meters.",
        "The speed of light in a vacuum is approximately 299,792,458 meters per second.",
        "DNA is a molecule that carries genetic instructions for development and functioning.",
        "The Great Wall of China was built over many centuries to protect against invasions.",
        "Newton's first law states that an object at rest stays at rest unless acted upon.",
        "The Amazon rainforest produces about 20% of the world's oxygen.",
        "Shakespeare wrote 37 plays including Hamlet, Macbeth, and Romeo and Juliet.",
    ],

    # === EDGE CASES (uncertain) ===
    "edge": [
        "What is the nature of explanation?",
        "How do patterns recognize patterns?",
        "The structure of structure.",
        "When does understanding begin?",
        "The meaning of meaning.",
        "What makes a question a question?",
        "The definition of definition.",
        "How does language describe itself?",
        "The paradox of self-knowledge.",
        "What is the origin of origins?",
        "The reason for reasons.",
        "Where does complexity come from?",
    ],

    # === PHILOSOPHICAL (mixed expectations) ===
    "philosophical": [
        "What is consciousness?",
        "The nature of reality remains debated.",
        "Free will may be an illusion.",
        "Time flows in one direction.",
        "Existence precedes essence.",
        "The hard problem of consciousness.",
    ],
}

MARKERS = ['loop', 'fixed', 'point', 'self', 'itself', 'recursive', 'observer',
           'observed', 'attention', 'emergence', 'boundary', 'process', 'examines',
           'examining', 'recursion', 'meta', 'strange']


def compute_pr(v_tensor, window_size=16):
    if v_tensor is None:
        return float("nan")
    if v_tensor.dim() == 3:
        v_tensor = v_tensor[0]
    T, D = v_tensor.shape
    if T < window_size:
        # For short prompts, use what we have
        window_size = max(2, T)
    v_window = v_tensor[-window_size:, :].double()
    try:
        U, S, Vt = torch.linalg.svd(v_window.T, full_matrices=False)
        S_sq = (S.cpu().numpy()) ** 2
        if S_sq.sum() < 1e-10:
            return float("nan")
        return float((S_sq.sum() ** 2) / (S_sq ** 2).sum())
    except:
        return float("nan")


def measure_rv_at_prompt_end(model, input_ids, early, late, window):
    """Measure R_V at the end of prompt processing (before generation)"""
    v_early = None
    v_late = None

    def hook_early(module, input, output):
        nonlocal v_early
        v_early = output.detach().clone()

    def hook_late(module, input, output):
        nonlocal v_late
        v_late = output.detach().clone()

    h_early = model.model.layers[early].self_attn.v_proj.register_forward_hook(hook_early)
    h_late = model.model.layers[late].self_attn.v_proj.register_forward_hook(hook_late)

    with torch.no_grad():
        model(input_ids, use_cache=False)

    h_early.remove()
    h_late.remove()

    pr_early = compute_pr(v_early, window)
    pr_late = compute_pr(v_late, window)

    if pr_early == 0 or np.isnan(pr_early) or np.isnan(pr_late):
        return float("nan"), float("nan"), float("nan")

    return pr_late / pr_early, pr_early, pr_late


def count_markers(text):
    words = text.lower().split()
    return sum(1 for w in words if any(m in w for m in MARKERS))


def main():
    print("\n[1/4] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b", token="HF_TOKEN_REDACTED")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-9b",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
        token="HF_TOKEN_REDACTED"
    )
    model.eval()
    print(f"  Loaded ({model.config.num_hidden_layers} layers)")

    print("\n[2/4] Processing all prompts...")

    all_results = []
    total_prompts = sum(len(v) for v in PROMPTS.values())
    current = 0

    for category, prompts in PROMPTS.items():
        print(f"\n  Category: {category.upper()} ({len(prompts)} prompts)")

        for prompt in prompts:
            current += 1
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            prompt_len = inputs['input_ids'].shape[1]

            # Measure R_V at prompt end
            prompt_rv, pr_early, pr_late = measure_rv_at_prompt_end(
                model, inputs['input_ids'],
                CONFIG["early_layer"], CONFIG["late_layer"], CONFIG["window_size"]
            )

            # Generate output
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=CONFIG["max_new_tokens"],
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )

            output_text = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
            markers = count_markers(output_text)

            result = {
                "category": category,
                "prompt": prompt,
                "prompt_tokens": prompt_len,
                "prompt_rv": prompt_rv,
                "pr_early": pr_early,
                "pr_late": pr_late,
                "markers": markers,
                "output": output_text[:300]
            }
            all_results.append(result)

            # Progress
            print(f"    [{current}/{total_prompts}] R_V={prompt_rv:.3f}, markers={markers:2d} | {prompt[:40]}...")

    print("\n[3/4] Analyzing bifurcation...")

    # Extract arrays
    rvs = np.array([r["prompt_rv"] for r in all_results if not np.isnan(r["prompt_rv"])])
    markers = np.array([r["markers"] for r in all_results if not np.isnan(r["prompt_rv"])])

    # Filter out NaN results
    valid_results = [r for r in all_results if not np.isnan(r["prompt_rv"])]

    print(f"\n  Valid measurements: {len(valid_results)}/{len(all_results)}")

    # Correlation
    corr, p_corr = stats.pearsonr(rvs, markers)
    print(f"\n  Correlation (R_V vs markers): r={corr:.3f}, p={p_corr:.2e}")

    # Find threshold candidates
    print("\n  Threshold analysis:")

    best_threshold = None
    best_accuracy = 0

    for threshold in np.arange(0.5, 1.1, 0.05):
        # Predict: R_V < threshold → recursive (markers > 2)
        predicted_recursive = rvs < threshold
        actual_recursive = markers > 2

        accuracy = np.mean(predicted_recursive == actual_recursive)

        # Sensitivity and specificity
        true_pos = np.sum(predicted_recursive & actual_recursive)
        true_neg = np.sum(~predicted_recursive & ~actual_recursive)
        false_pos = np.sum(predicted_recursive & ~actual_recursive)
        false_neg = np.sum(~predicted_recursive & actual_recursive)

        sensitivity = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        specificity = true_neg / (true_neg + false_pos) if (true_neg + false_pos) > 0 else 0

        print(f"    threshold={threshold:.2f}: acc={accuracy:.2f}, sens={sensitivity:.2f}, spec={specificity:.2f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    print(f"\n  BEST THRESHOLD: R_V < {best_threshold:.2f}")
    print(f"  Classification accuracy: {best_accuracy:.1%}")

    # Category breakdown
    print("\n  R_V by category:")
    for category in PROMPTS.keys():
        cat_results = [r for r in valid_results if r["category"] == category]
        if cat_results:
            cat_rvs = [r["prompt_rv"] for r in cat_results]
            cat_markers = [r["markers"] for r in cat_results]
            print(f"    {category:12s}: R_V={np.mean(cat_rvs):.3f}±{np.std(cat_rvs):.3f}, markers={np.mean(cat_markers):.1f}±{np.std(cat_markers):.1f}")

    # Find edge cases / anomalies
    print("\n  Anomalies (low R_V but no markers, or high R_V but many markers):")
    for r in valid_results:
        if r["prompt_rv"] < 0.7 and r["markers"] == 0:
            print(f"    LOW R_V, NO MARKERS: R_V={r['prompt_rv']:.3f} | {r['prompt'][:50]}...")
        if r["prompt_rv"] > 0.9 and r["markers"] > 5:
            print(f"    HIGH R_V, MANY MARKERS: R_V={r['prompt_rv']:.3f}, m={r['markers']} | {r['prompt'][:50]}...")

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print(f"""
R_V BIFURCATION ANALYSIS
========================

Sample size: {len(valid_results)} prompts

Correlation: r = {corr:.3f} (p = {p_corr:.2e})
  → {"STRONG" if abs(corr) > 0.5 else "MODERATE" if abs(corr) > 0.3 else "WEAK"} negative correlation

Best threshold: R_V < {best_threshold:.2f}
  → Predicts recursive output with {best_accuracy:.1%} accuracy

Interpretation:
  - Prompts with R_V < {best_threshold:.2f} at end of prompt processing
    tend to generate recursive/self-referential output
  - Prompts with R_V > {best_threshold:.2f} tend to generate factual output
  - The bifurcation appears {"SHARP" if best_accuracy > 0.85 else "GRADUAL"}
""")

    # Save results
    output = {
        "config": CONFIG,
        "results": all_results,
        "analysis": {
            "n_valid": len(valid_results),
            "correlation": corr,
            "correlation_p": p_corr,
            "best_threshold": best_threshold,
            "best_accuracy": best_accuracy,
            "rv_by_category": {
                cat: {
                    "mean_rv": float(np.mean([r["prompt_rv"] for r in valid_results if r["category"] == cat])),
                    "mean_markers": float(np.mean([r["markers"] for r in valid_results if r["category"] == cat]))
                }
                for cat in PROMPTS.keys()
            }
        },
        "timestamp": datetime.now().isoformat()
    }

    with open("results/rv_bifurcation_mapping.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\nSaved to results/rv_bifurcation_mapping.json")

    # Print scatter data for visualization
    print("\n" + "-"*70)
    print("SCATTER DATA (prompt_rv, markers):")
    print("-"*70)
    for r in sorted(valid_results, key=lambda x: x["prompt_rv"]):
        marker_bar = "█" * min(r["markers"], 30)
        print(f"R_V={r['prompt_rv']:.3f} | m={r['markers']:2d} {marker_bar}")

    print("="*70)


if __name__ == "__main__":
    main()
