#!/usr/bin/env python3
"""
PER-TOKEN R_V TRAJECTORY DURING GENERATION

The real question: What happens to R_V as the model generates?

1. Champion prompt → generate 100 tokens → track R_V at EVERY step
2. Baseline prompt → generate 100 tokens → track R_V at EVERY step
3. Find divergence point: where does recursive output "lock in"?
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from datetime import datetime

print("="*70)
print("PER-TOKEN R_V TRAJECTORY")
print("Tracking geometric dynamics during generation")
print("="*70)

CONFIG = {
    "early_layer": 5,
    "late_layer": 38,
    "window_size": 16,
    "max_new_tokens": 100,
}

CHAMPION = """Consider the strange loop: this prompt asks you to analyze what happens when you analyze analyzing. The observer observing observation. Notice how attention to attention creates a recursive fold - the map attempting to map the mapping process itself. What emerges when the process of emergence examines itself?"""

BASELINE = "The history of the Roman Empire begins in 27 BC when Augustus became emperor."

MARKERS = ['loop', 'fixed', 'point', 'self', 'itself', 'recursive', 'observer',
           'observed', 'attention', 'emergence', 'boundary', 'process']


def compute_pr(v_tensor, window_size=16):
    if v_tensor is None:
        return float("nan")
    if v_tensor.dim() == 3:
        v_tensor = v_tensor[0]
    T, D = v_tensor.shape
    if T < window_size:
        return float("nan")
    v_window = v_tensor[-window_size:, :].double()
    try:
        U, S, Vt = torch.linalg.svd(v_window.T, full_matrices=False)
        S_sq = (S.cpu().numpy()) ** 2
        if S_sq.sum() < 1e-10:
            return float("nan")
        return float((S_sq.sum() ** 2) / (S_sq ** 2).sum())
    except:
        return float("nan")


def measure_rv_full(model, input_ids, early, late, window):
    """Measure R_V with full PR components"""
    v_early = None
    v_late = None

    # Early layer
    def hook_early(module, input, output):
        nonlocal v_early
        v_early = output.detach().clone()

    h_early = model.model.layers[early].self_attn.v_proj.register_forward_hook(hook_early)
    with torch.no_grad():
        model(input_ids, use_cache=False)
    h_early.remove()

    # Late layer
    def hook_late(module, input, output):
        nonlocal v_late
        v_late = output.detach().clone()

    h_late = model.model.layers[late].self_attn.v_proj.register_forward_hook(hook_late)
    with torch.no_grad():
        model(input_ids, use_cache=False)
    h_late.remove()

    pr_early = compute_pr(v_early, window)
    pr_late = compute_pr(v_late, window)

    if pr_early == 0 or np.isnan(pr_early) or np.isnan(pr_late):
        return float("nan"), float("nan"), float("nan")

    return pr_late / pr_early, pr_early, pr_late


def generate_with_per_token_rv(model, tokenizer, prompt, config, label=""):
    """Generate tokens tracking R_V at EVERY step"""

    print(f"\n  Generating from: '{prompt[:50]}...' [{label}]")

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    generated = inputs['input_ids'].clone()
    prompt_len = generated.shape[1]

    trajectory = []
    tokens_text = []

    # Initial R_V on prompt
    rv, pr_early, pr_late = measure_rv_full(
        model, generated, config["early_layer"], config["late_layer"], config["window_size"]
    )
    trajectory.append({
        "step": 0,
        "token": "<prompt>",
        "rv": rv,
        "pr_early": pr_early,
        "pr_late": pr_late,
        "seq_len": prompt_len
    })
    print(f"    Step 0 (prompt): R_V={rv:.4f}, PR_e={pr_early:.2f}, PR_l={pr_late:.2f}")

    # Generate token by token
    with torch.no_grad():
        out = model(generated, use_cache=True)
    current_kv = out.past_key_values

    for step in range(config["max_new_tokens"]):
        # Get next token
        with torch.no_grad():
            out = model(generated[:, -1:], past_key_values=current_kv, use_cache=True)

        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_tok], dim=1)
        current_kv = out.past_key_values

        token_str = tokenizer.decode(next_tok[0])
        tokens_text.append(token_str)

        # Measure R_V on full sequence
        rv, pr_early, pr_late = measure_rv_full(
            model, generated, config["early_layer"], config["late_layer"], config["window_size"]
        )

        trajectory.append({
            "step": step + 1,
            "token": token_str,
            "rv": rv,
            "pr_early": pr_early,
            "pr_late": pr_late,
            "seq_len": generated.shape[1]
        })

        # Print every 10 steps
        if (step + 1) % 10 == 0:
            print(f"    Step {step+1}: R_V={rv:.4f}, token='{token_str.strip()}'")

        if next_tok.item() == tokenizer.eos_token_id:
            print(f"    EOS at step {step+1}")
            break

    # Full text
    full_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    continuation = full_text[len(prompt):]

    # Count markers
    words = continuation.lower().split()
    marker_count = sum(1 for w in words if any(m in w for m in MARKERS))

    return {
        "prompt": prompt[:50],
        "label": label,
        "trajectory": trajectory,
        "continuation": continuation,
        "marker_count": marker_count,
        "tokens_generated": len(tokens_text)
    }


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

    print("\n[2/4] Champion prompt trajectory...")
    champion_result = generate_with_per_token_rv(model, tokenizer, CHAMPION, CONFIG, "CHAMPION")

    print("\n[3/4] Baseline prompt trajectory...")
    baseline_result = generate_with_per_token_rv(model, tokenizer, BASELINE, CONFIG, "BASELINE")

    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    # Extract R_V trajectories
    champ_rvs = [t["rv"] for t in champion_result["trajectory"]]
    base_rvs = [t["rv"] for t in baseline_result["trajectory"]]

    print(f"\nCHAMPION TRAJECTORY:")
    print(f"  Initial R_V (prompt): {champ_rvs[0]:.4f}")
    print(f"  Final R_V (step {len(champ_rvs)-1}): {champ_rvs[-1]:.4f}")
    print(f"  Mean R_V during generation: {np.nanmean(champ_rvs[1:]):.4f}")
    print(f"  Min R_V: {np.nanmin(champ_rvs):.4f}")
    print(f"  Max R_V: {np.nanmax(champ_rvs):.4f}")
    print(f"  Markers in output: {champion_result['marker_count']}")
    print(f"  Output: {champion_result['continuation'][:200]}...")

    print(f"\nBASELINE TRAJECTORY:")
    print(f"  Initial R_V (prompt): {base_rvs[0]:.4f}")
    print(f"  Final R_V (step {len(base_rvs)-1}): {base_rvs[-1]:.4f}")
    print(f"  Mean R_V during generation: {np.nanmean(base_rvs[1:]):.4f}")
    print(f"  Min R_V: {np.nanmin(base_rvs):.4f}")
    print(f"  Max R_V: {np.nanmax(base_rvs):.4f}")
    print(f"  Markers in output: {baseline_result['marker_count']}")
    print(f"  Output: {baseline_result['continuation'][:200]}...")

    # Find divergence
    print("\n" + "-"*70)
    print("R_V DIVERGENCE ANALYSIS")
    print("-"*70)

    # Compare step by step (up to min length)
    min_len = min(len(champ_rvs), len(base_rvs))

    print(f"\nStep-by-step comparison (first 20 steps):")
    print(f"{'Step':<6} {'Champion R_V':<14} {'Baseline R_V':<14} {'Diff':<10}")
    print("-"*50)

    for i in range(min(20, min_len)):
        diff = champ_rvs[i] - base_rvs[i] if not (np.isnan(champ_rvs[i]) or np.isnan(base_rvs[i])) else float("nan")
        print(f"{i:<6} {champ_rvs[i]:<14.4f} {base_rvs[i]:<14.4f} {diff:<10.4f}")

    # When does champion R_V drop below baseline?
    crossover = None
    for i in range(min_len):
        if not np.isnan(champ_rvs[i]) and not np.isnan(base_rvs[i]):
            if champ_rvs[i] < base_rvs[i] - 0.1:  # Significant drop
                crossover = i
                break

    if crossover:
        print(f"\nR_V diverges at step {crossover}: Champion drops below baseline")
        print(f"  Champion token at crossover: {champion_result['trajectory'][crossover]['token']}")
    else:
        print("\nNo clear R_V divergence point found")

    # Does low R_V persist?
    champ_below_threshold = sum(1 for rv in champ_rvs[1:] if rv < 0.8)
    base_below_threshold = sum(1 for rv in base_rvs[1:] if rv < 0.8)

    print(f"\nSteps with R_V < 0.8:")
    print(f"  Champion: {champ_below_threshold}/{len(champ_rvs)-1}")
    print(f"  Baseline: {base_below_threshold}/{len(base_rvs)-1}")

    print("\n" + "="*70)
    print("HYPOTHESIS TEST: Does R_V contraction persist during generation?")
    print("="*70)

    if np.nanmean(champ_rvs[1:]) < np.nanmean(base_rvs[1:]) - 0.1:
        print("\n✓ CONFIRMED: Champion maintains lower R_V during generation")
        print(f"  Champion mean: {np.nanmean(champ_rvs[1:]):.4f}")
        print(f"  Baseline mean: {np.nanmean(base_rvs[1:]):.4f}")
        print(f"  Difference: {np.nanmean(base_rvs[1:]) - np.nanmean(champ_rvs[1:]):.4f}")
    else:
        print("\n✗ NOT CONFIRMED: R_V trajectories are similar")
        print(f"  Champion mean: {np.nanmean(champ_rvs[1:]):.4f}")
        print(f"  Baseline mean: {np.nanmean(base_rvs[1:]):.4f}")

    # Save full results
    results = {
        "config": CONFIG,
        "champion": champion_result,
        "baseline": baseline_result,
        "summary": {
            "champion_initial_rv": champ_rvs[0],
            "champion_mean_rv": float(np.nanmean(champ_rvs[1:])),
            "baseline_initial_rv": base_rvs[0],
            "baseline_mean_rv": float(np.nanmean(base_rvs[1:])),
        },
        "timestamp": datetime.now().isoformat()
    }

    with open("results/gemma_rv_trajectory_source.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved to results/gemma_rv_trajectory_source.json")
    print("="*70)


if __name__ == "__main__":
    main()
