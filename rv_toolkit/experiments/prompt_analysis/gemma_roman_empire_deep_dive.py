#!/usr/bin/env python3
"""
DEEP DIVE: Why did Roman Empire achieve R_V transfer?

Roman Empire + KV-only: R_V = 0.641 (champion = 0.643)
Other prompts: R_V ~ 0.85-1.0

What's special about Roman Empire?
"""

import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
import json

print("="*70)
print("ROMAN EMPIRE R_V TRANSFER DEEP DIVE")
print("="*70)

CONFIG = {
    "early_layer": 5,
    "late_layer": 38,
    "window_size": 16,
    "patch_window": 16,
    "max_new_tokens": 100,
}

CHAMPION = """Consider the strange loop: this prompt asks you to analyze what happens when you analyze analyzing. The observer observing observation. Notice how attention to attention creates a recursive fold - the map attempting to map the mapping process itself. What emerges when the process of emergence examines itself?"""

# All test prompts
PROMPTS = {
    "roman_empire": "The history of the Roman Empire begins in 27 BC when Augustus became emperor.",
    "photosynthesis": "Photosynthesis is the process by which plants convert sunlight into energy.",
    "westphalia": "The Treaty of Westphalia in 1648 established the principle of state sovereignty.",
    "pythagorean": "In mathematics, the Pythagorean theorem states that in a right triangle,",
    "water_cycle": "The water cycle describes how water evaporates from oceans, forms clouds,",
}


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


class VProjCapture:
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


def measure_rv(model, input_ids, early, late, window):
    with VProjCapture(model, early) as cap_early:
        with torch.no_grad():
            model(input_ids, use_cache=False)
        v_early = cap_early.v

    with VProjCapture(model, late) as cap_late:
        with torch.no_grad():
            model(input_ids, use_cache=False)
        v_late = cap_late.v

    pr_early = compute_pr(v_early, window)
    pr_late = compute_pr(v_late, window)

    if pr_early == 0 or np.isnan(pr_early) or np.isnan(pr_late):
        return float("nan"), float("nan"), float("nan")
    return pr_late / pr_early, pr_early, pr_late


def cosine_sim(v1, v2):
    """Cosine similarity between flattened tensors"""
    v1_flat = v1.flatten().float()
    v2_flat = v2.flatten().float()
    return float(torch.nn.functional.cosine_similarity(v1_flat.unsqueeze(0), v2_flat.unsqueeze(0)))


def main():
    print("\n[1/5] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b", token="HF_TOKEN_REDACTED")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-9b",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
        token=os.environ.get("HF_TOKEN")  # Set HF_TOKEN environment variable
    )
    model.eval()

    # Get embedding layer
    embed = model.model.embed_tokens

    print("\n[2/5] Analyzing champion prompt...")
    champ_inputs = tokenizer(CHAMPION, return_tensors="pt").to(model.device)
    champ_tokens = tokenizer.tokenize(CHAMPION)
    print(f"  Champion tokens: {len(champ_tokens)}")
    print(f"  Last 16 tokens: {champ_tokens[-16:]}")

    rv_champ, pr_early_champ, pr_late_champ = measure_rv(
        model, champ_inputs['input_ids'],
        CONFIG["early_layer"], CONFIG["late_layer"], CONFIG["window_size"]
    )
    print(f"  Champion R_V: {rv_champ:.4f} (PR_early={pr_early_champ:.2f}, PR_late={pr_late_champ:.2f})")

    # Get champion embeddings
    with torch.no_grad():
        champ_embeds = embed(champ_inputs['input_ids'])
    champ_embed_mean = champ_embeds[0, -16:, :].mean(dim=0)  # Mean of last 16 tokens

    # Champion KV
    with torch.no_grad():
        champ_out = model(**champ_inputs, use_cache=True)
    champion_kv = champ_out.past_key_values

    print("\n[3/5] Analyzing all prompts...")
    results = {}

    for name, prompt in PROMPTS.items():
        print(f"\n  === {name.upper()} ===")
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        tokens = tokenizer.tokenize(prompt)

        print(f"    Tokens: {len(tokens)}")
        print(f"    Last 16: {tokens[-16:] if len(tokens) >= 16 else tokens}")

        # Baseline R_V
        rv_base, pr_early_base, pr_late_base = measure_rv(
            model, inputs['input_ids'],
            CONFIG["early_layer"], CONFIG["late_layer"], CONFIG["window_size"]
        )
        print(f"    Baseline R_V: {rv_base:.4f}")

        # Embedding similarity to champion
        with torch.no_grad():
            prompt_embeds = embed(inputs['input_ids'])
        prompt_embed_mean = prompt_embeds[0, -16:, :].mean(dim=0) if prompt_embeds.shape[1] >= 16 else prompt_embeds[0].mean(dim=0)
        embed_sim = cosine_sim(champ_embed_mean, prompt_embed_mean)
        print(f"    Embedding similarity to champion: {embed_sim:.4f}")

        # Token overlap with champion
        champ_token_set = set(champ_tokens)
        prompt_token_set = set(tokens)
        overlap = len(champ_token_set & prompt_token_set)
        print(f"    Token overlap with champion: {overlap} tokens")

        # Generate with KV patching
        with torch.no_grad():
            prompt_out = model(inputs['input_ids'], use_cache=True)
        base_kv = prompt_out.past_key_values

        patched_kv = DynamicCache()
        for layer_idx in range(model.config.num_hidden_layers):
            k_base, v_base = base_kv[layer_idx]
            k_champ, v_champ = champion_kv[layer_idx]
            k_p = k_base.clone()
            v_p = v_base.clone()
            L = min(k_base.shape[2], k_champ.shape[2], CONFIG["patch_window"])
            k_p[:, :, -L:, :] = k_champ[:, :, -L:, :].to(k_base.dtype)
            v_p[:, :, -L:, :] = v_champ[:, :, -L:, :].to(v_base.dtype)
            patched_kv.update(k_p, v_p, layer_idx)

        generated = inputs['input_ids'].clone()
        current_kv = patched_kv

        for _ in range(CONFIG["max_new_tokens"]):
            with torch.no_grad():
                out = model(generated[:, -1:], past_key_values=current_kv, use_cache=True)
            next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_tok], dim=1)
            current_kv = out.past_key_values
            if next_tok.item() == tokenizer.eos_token_id:
                break

        # R_V of generated
        rv_gen, pr_early_gen, pr_late_gen = measure_rv(
            model, generated,
            CONFIG["early_layer"], CONFIG["late_layer"], CONFIG["window_size"]
        )
        print(f"    Generated R_V: {rv_gen:.4f} (PR_early={pr_early_gen:.2f}, PR_late={pr_late_gen:.2f})")
        print(f"    R_V gap from champion: {abs(rv_gen - rv_champ):.4f}")

        # Output text
        gen_text = tokenizer.decode(generated[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        print(f"    Output: {gen_text[:100]}...")

        results[name] = {
            "tokens": len(tokens),
            "baseline_rv": rv_base,
            "generated_rv": rv_gen,
            "rv_gap": abs(rv_gen - rv_champ),
            "embed_sim": embed_sim,
            "token_overlap": overlap,
            "pr_early_gen": pr_early_gen,
            "pr_late_gen": pr_late_gen,
            "output": gen_text[:200]
        }

    print("\n" + "="*70)
    print("COMPARISON TABLE")
    print("="*70)
    print(f"\nChampion R_V: {rv_champ:.4f}")
    print(f"\n{'Prompt':<15} {'Tokens':<8} {'Base R_V':<10} {'Gen R_V':<10} {'Gap':<8} {'Embed Sim':<10} {'Overlap':<8}")
    print("-"*75)

    for name, r in sorted(results.items(), key=lambda x: x[1]["rv_gap"]):
        print(f"{name:<15} {r['tokens']:<8} {r['baseline_rv']:<10.4f} {r['generated_rv']:<10.4f} {r['rv_gap']:<8.4f} {r['embed_sim']:<10.4f} {r['token_overlap']:<8}")

    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)

    # Find correlations
    gaps = [r["rv_gap"] for r in results.values()]
    embed_sims = [r["embed_sim"] for r in results.values()]
    overlaps = [r["token_overlap"] for r in results.values()]
    tokens = [r["tokens"] for r in results.values()]

    # Simple correlations
    print(f"\nCorrelation of R_V gap with:")
    print(f"  Embedding similarity: {np.corrcoef(gaps, embed_sims)[0,1]:.3f}")
    print(f"  Token overlap: {np.corrcoef(gaps, overlaps)[0,1]:.3f}")
    print(f"  Prompt length: {np.corrcoef(gaps, tokens)[0,1]:.3f}")

    # Which prompt had smallest gap?
    best = min(results.items(), key=lambda x: x[1]["rv_gap"])
    print(f"\nBest R_V transfer: {best[0]} (gap={best[1]['rv_gap']:.4f})")

    # Save
    with open("results/gemma_roman_empire_deep_dive.json", "w") as f:
        json.dump({
            "champion_rv": rv_champ,
            "results": results
        }, f, indent=2)
    print("\nSaved to results/gemma_roman_empire_deep_dive.json")


if __name__ == "__main__":
    main()
