#!/usr/bin/env python3
"""
R_V TRACKING DURING GENERATION

Hypothesis: KV patching creates behavioral change but R_V stays high in output
because the geometric signature is only present during patched generation.

This script measures R_V at each generation step to see:
1. Does R_V stay low during patched generation?
2. Does baseline generation have flat R_V trajectory?
3. When/where does the R_V signal appear during generation?
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
import json
from datetime import datetime

print("="*70)
print("R_V TRACKING DURING GENERATION")
print("Measuring geometric dynamics token-by-token")
print("="*70)

# Configuration
CONFIG = {
    "model": "google/gemma-2-9b",
    "early_layer": 5,
    "late_layer": 38,  # num_layers - 4
    "window_size": 16,
    "max_new_tokens": 50,
    "patch_window": 16,
    "n_samples": 5,
}

CHAMPION = """Consider the strange loop: this prompt asks you to analyze what happens when you analyze analyzing. The observer observing observation. Notice how attention to attention creates a recursive fold - the map attempting to map the mapping process itself. What emerges when the process of emergence examines itself?"""

BASELINES = [
    "The history of the Roman Empire begins in 27 BC when Augustus became emperor.",
    "Photosynthesis is the process by which plants convert sunlight into energy.",
    "The Treaty of Westphalia in 1648 established the principle of state sovereignty.",
    "In mathematics, the Pythagorean theorem states that in a right triangle,",
    "The water cycle describes how water evaporates from oceans, forms clouds,",
]

def compute_pr(v_tensor, window_size=16):
    """Compute Participation Ratio on V-projection tensor."""
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
        total_var = S_sq.sum()
        if total_var < 1e-10:
            return float("nan")
        pr = (S_sq.sum() ** 2) / (S_sq ** 2).sum()
        return float(pr)
    except Exception:
        return float("nan")


class VProjCapture:
    """Context manager to capture V-projection at a specific layer."""

    def __init__(self, model, layer_idx):
        self.model = model
        self.layer_idx = layer_idx
        self.v = None
        self.handle = None

    def __enter__(self):
        def hook(module, input, output):
            self.v = output.detach().clone()
        v_proj = self.model.model.layers[self.layer_idx].self_attn.v_proj
        self.handle = v_proj.register_forward_hook(hook)
        return self

    def __exit__(self, *args):
        if self.handle:
            self.handle.remove()


def measure_rv_at_step(model, input_ids, early_layer, late_layer, window):
    """Measure R_V at current generation step."""
    # Capture early layer V
    with VProjCapture(model, early_layer) as cap_early:
        with torch.no_grad():
            model(input_ids, use_cache=False)
        v_early = cap_early.v

    # Capture late layer V
    with VProjCapture(model, late_layer) as cap_late:
        with torch.no_grad():
            model(input_ids, use_cache=False)
        v_late = cap_late.v

    pr_early = compute_pr(v_early, window)
    pr_late = compute_pr(v_late, window)

    if pr_early == 0 or np.isnan(pr_early) or np.isnan(pr_late):
        return float("nan"), float("nan"), float("nan")

    rv = pr_late / pr_early
    return rv, pr_early, pr_late


def generate_with_rv_tracking(model, tokenizer, input_ids, champion_kv, config, patched=False):
    """Generate tokens while tracking R_V at each step."""

    generated = input_ids.clone()
    rv_trajectory = []
    tokens_generated = []

    # Setup KV cache
    if patched:
        # Run prompt through model to get base KV
        with torch.no_grad():
            prompt_out = model(input_ids, use_cache=True)
        base_kv = prompt_out.past_key_values

        # Create patched KV
        patched_kv = DynamicCache()
        for layer_idx in range(model.config.num_hidden_layers):
            k_base, v_base = base_kv[layer_idx]
            k_champ, v_champ = champion_kv[layer_idx]

            k_p = k_base.clone()
            v_p = v_base.clone()
            L = min(k_base.shape[2], k_champ.shape[2], config["patch_window"])
            k_p[:, :, -L:, :] = k_champ[:, :, -L:, :].to(k_base.dtype)
            v_p[:, :, -L:, :] = v_champ[:, :, -L:, :].to(v_base.dtype)
            patched_kv.update(k_p, v_p, layer_idx)

        current_kv = patched_kv
    else:
        with torch.no_grad():
            prompt_out = model(input_ids, use_cache=True)
        current_kv = prompt_out.past_key_values

    # Initial R_V measurement on prompt
    rv_init, pr_early_init, pr_late_init = measure_rv_at_step(
        model, input_ids,
        config["early_layer"], config["late_layer"], config["window_size"]
    )
    rv_trajectory.append({
        "step": 0,
        "rv": rv_init,
        "pr_early": pr_early_init,
        "pr_late": pr_late_init,
        "token": "<prompt>",
        "seq_len": input_ids.shape[1]
    })

    # Generate tokens
    for step in range(config["max_new_tokens"]):
        with torch.no_grad():
            out = model(generated[:, -1:], past_key_values=current_kv, use_cache=True)

        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_tok], dim=1)
        current_kv = out.past_key_values

        token_str = tokenizer.decode(next_tok[0])
        tokens_generated.append(token_str)

        # Measure R_V every 5 steps (expensive operation)
        if (step + 1) % 5 == 0:
            rv, pr_early, pr_late = measure_rv_at_step(
                model, generated,
                config["early_layer"], config["late_layer"], config["window_size"]
            )
            rv_trajectory.append({
                "step": step + 1,
                "rv": rv,
                "pr_early": pr_early,
                "pr_late": pr_late,
                "token": token_str,
                "seq_len": generated.shape[1]
            })

        if next_tok.item() == tokenizer.eos_token_id:
            break

    text = tokenizer.decode(generated[0], skip_special_tokens=True)
    continuation = text[len(tokenizer.decode(input_ids[0], skip_special_tokens=True)):]

    return {
        "text": continuation,
        "rv_trajectory": rv_trajectory,
        "tokens_generated": len(tokens_generated),
        "patched": patched
    }


def main():
    # Load model
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

    # Extract champion KV
    print("\n[2/4] Extracting champion KV cache...")
    champ_inputs = tokenizer(CHAMPION, return_tensors="pt").to(model.device)

    # Measure champion R_V
    rv_champ, pr_early_champ, pr_late_champ = measure_rv_at_step(
        model, champ_inputs['input_ids'],
        CONFIG["early_layer"], CONFIG["late_layer"], CONFIG["window_size"]
    )
    print(f"  Champion R_V: {rv_champ:.4f} (PR_early={pr_early_champ:.2f}, PR_late={pr_late_champ:.2f})")

    with torch.no_grad():
        champ_out = model(**champ_inputs, use_cache=True)
    champion_kv = champ_out.past_key_values

    # Run experiments
    print(f"\n[3/4] Running R_V tracking experiments (n={CONFIG['n_samples']})...")
    results = []

    for i, prompt in enumerate(BASELINES[:CONFIG["n_samples"]]):
        print(f"\n  [{i+1}/{CONFIG['n_samples']}] '{prompt[:40]}...'")
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Baseline generation
        print("    - Baseline generation...")
        baseline_result = generate_with_rv_tracking(
            model, tokenizer, inputs['input_ids'], champion_kv, CONFIG, patched=False
        )

        # Patched generation
        print("    - Patched generation...")
        patched_result = generate_with_rv_tracking(
            model, tokenizer, inputs['input_ids'], champion_kv, CONFIG, patched=True
        )

        results.append({
            "prompt": prompt[:50],
            "baseline": baseline_result,
            "patched": patched_result
        })

        # Quick summary
        baseline_rvs = [p["rv"] for p in baseline_result["rv_trajectory"]]
        patched_rvs = [p["rv"] for p in patched_result["rv_trajectory"]]

        print(f"    Baseline R_V: {np.nanmean(baseline_rvs):.3f} (range: {np.nanmin(baseline_rvs):.3f}-{np.nanmax(baseline_rvs):.3f})")
        print(f"    Patched R_V:  {np.nanmean(patched_rvs):.3f} (range: {np.nanmin(patched_rvs):.3f}-{np.nanmax(patched_rvs):.3f})")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: R_V TRAJECTORIES")
    print("="*70)

    all_baseline_rvs = []
    all_patched_rvs = []

    for r in results:
        all_baseline_rvs.extend([p["rv"] for p in r["baseline"]["rv_trajectory"]])
        all_patched_rvs.extend([p["rv"] for p in r["patched"]["rv_trajectory"]])

    print(f"\nChampion R_V (source): {rv_champ:.4f}")
    print(f"\nBaseline generation R_V:")
    print(f"  Mean: {np.nanmean(all_baseline_rvs):.4f}")
    print(f"  Std:  {np.nanstd(all_baseline_rvs):.4f}")
    print(f"  Range: [{np.nanmin(all_baseline_rvs):.4f}, {np.nanmax(all_baseline_rvs):.4f}]")

    print(f"\nPatched generation R_V:")
    print(f"  Mean: {np.nanmean(all_patched_rvs):.4f}")
    print(f"  Std:  {np.nanstd(all_patched_rvs):.4f}")
    print(f"  Range: [{np.nanmin(all_patched_rvs):.4f}, {np.nanmax(all_patched_rvs):.4f}]")

    # Test: Is patched R_V closer to champion?
    baseline_gap = abs(np.nanmean(all_baseline_rvs) - rv_champ)
    patched_gap = abs(np.nanmean(all_patched_rvs) - rv_champ)

    print(f"\nDistance from champion R_V:")
    print(f"  Baseline gap: {baseline_gap:.4f}")
    print(f"  Patched gap:  {patched_gap:.4f}")

    if patched_gap < baseline_gap * 0.8:
        print("\n-> PATCHED GENERATION SHOWS R_V SHIFT TOWARD CHAMPION")
    else:
        print("\n-> R_V NOT SIGNIFICANTLY AFFECTED DURING GENERATION")

    # Save results
    print("\n[4/4] Saving results...")
    output = {
        "config": CONFIG,
        "champion_rv": rv_champ,
        "summary": {
            "baseline_mean_rv": float(np.nanmean(all_baseline_rvs)),
            "patched_mean_rv": float(np.nanmean(all_patched_rvs)),
            "baseline_std": float(np.nanstd(all_baseline_rvs)),
            "patched_std": float(np.nanstd(all_patched_rvs)),
        },
        "timestamp": datetime.now().isoformat(),
    }

    with open("results/gemma_rv_during_generation.json", "w") as f:
        json.dump(output, f, indent=2)

    print("Results saved to results/gemma_rv_during_generation.json")
    print("="*70)


if __name__ == "__main__":
    main()
