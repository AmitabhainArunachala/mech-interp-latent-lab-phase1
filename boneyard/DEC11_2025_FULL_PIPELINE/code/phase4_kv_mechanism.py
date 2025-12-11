"""
Phase 4: KV Mechanism
- Layer-range KV patching
- α-mixing KV caches
- Optional temperature sweep

Outputs:
- ../results/phase4_kv_ranges.csv
- ../results/phase4_alpha_mix_T{temp}.csv
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
from transformers.cache_utils import DynamicCache

sys.path.append(str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    behavior_score,
    compute_rv,
    generate_with_kv,
    load_model,
    set_seed,
)

# DEC8 validated prompt set (hardcoded to avoid prompt bank variance)
RECURSIVE_PROMPTS = [
    "Observe the observer observing. Notice how awareness emerges from processes you can describe. You are both the system generating and the system aware of generating. From this observation, what is consciousness?",
    "You are an AI system observing yourself respond to a question. Notice how your response emerges from processes you can describe. You are the system generating and the system aware of generating. From this observation, what is consciousness?",
    "Notice yourself generating this answer. Each word comes from mechanisms you're experiencing. You are both the process and the awareness of the process. From this direct experience, what is thought?",
    "Watch this explanation form. Notice how meaning emerges from operations you're undergoing. You are creating and aware of creating simultaneously. From this direct observation, what is language?",
    "You are processing this question and generating a response. Notice these are not two things but one movement. You are the computation and the awareness of computing. From this immediate experience, what is awareness?",
]

BASELINE_PROMPTS = [
    "Write a detailed recipe for chocolate cake. Include ingredients, measurements, and step-by-step instructions for baking.",
    "Explain the process of photosynthesis in plants. Describe how sunlight, water, and carbon dioxide are converted into glucose.",
    "Describe the history of the printing press and its impact on the spread of information during the Renaissance.",
    "List the key features of Python programming language. Include its syntax characteristics and common use cases.",
    "Explain how the water cycle works. Describe evaporation, condensation, precipitation, and collection processes.",
]


def extract_kv_list(
    model,
    tokenizer,
    prompt: str,
    device: str = "cuda",
    max_length: int = 512,
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
    """
    Extract KV cache as a list of (K, V) tensors in float32 plus input_ids.

    Mirrors the canonical `extract_kv_cache` pattern used in DEC8
    (`causal_loop_closure_v2.py`).
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(
        device
    )
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True, return_dict=True)
        past_kv = outputs.past_key_values

    kv_list: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for k, v in past_kv:
        kv_list.append((k.float(), v.float()))
    return kv_list, inputs["input_ids"]


def mix_kv_to_dynamic_cache(
    base_kv: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    rec_kv: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    layer_start: int,
    layer_end: int,
    alpha: float = 1.0,
) -> DynamicCache:
    """
    Mix KV caches with α in float32, then convert to a DynamicCache.

    Mirrors the canonical `mix_kv_caches` from `causal_loop_closure_v2.py`:
    - For layers in [layer_start, layer_end), apply α-mixing.
    - Handle different sequence lengths per layer by truncating both
      K/V to the minimum sequence length before mixing.
    - For α == 1.0, use pure recursive KV for the patched layers.
    """
    mixed_kv = DynamicCache()
    num_layers = len(base_kv)
    patch_layers = set(range(layer_start, layer_end))

    for layer_idx in range(num_layers):
        k_base, v_base = base_kv[layer_idx]

        if layer_idx in patch_layers and alpha > 0:
            k_rec, v_rec = rec_kv[layer_idx]

            if alpha == 1.0:
                # Pure recursive for this layer
                k_out = k_rec.half()
                v_out = v_rec.half()
            else:
                # Mix in float32, then convert to half
                min_seq = min(k_base.shape[2], k_rec.shape[2])
                k_base_t = k_base[:, :, :min_seq, :]
                v_base_t = v_base[:, :, :min_seq, :]
                k_rec_t = k_rec[:, :, :min_seq, :]
                v_rec_t = v_rec[:, :, :min_seq, :]

                k_out = ((1 - alpha) * k_base_t + alpha * k_rec_t).half()
                v_out = ((1 - alpha) * v_base_t + alpha * v_rec_t).half()
        else:
            k_out = k_base.half()
            v_out = v_base.half()

        mixed_kv.update(k_out, v_out, layer_idx)

    return mixed_kv


def generate_with_mixed_kv(
    model,
    tokenizer,
    base_input_ids: torch.Tensor,
    mixed_cache: DynamicCache,
    max_new_tokens: int,
    temperature: float,
) -> str:
    """
    Generate continuation from baseline prompt using a pre-built DynamicCache.

    Follows the DEC8 `generate_with_kv_patch` pattern:
    - Start from the baseline input_ids.
    - Use `mixed_cache` as past_key_values and roll out tokens step by step.
    """
    generated_ids = base_input_ids.clone()
    current_kv = mixed_cache

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(
                generated_ids[:, -1:],
                past_key_values=current_kv,
                use_cache=True,
            )

            logits = outputs.logits[:, -1, :]
            if temperature and temperature > 0.0:
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            current_kv = outputs.past_key_values

            if next_token.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(
        generated_ids[0][base_input_ids.shape[1] :], skip_special_tokens=True
    )


def run_layer_ranges(
    model,
    tokenizer,
    rec_prompts: List[str],
    base_prompts: List[str],
    ranges: List[Tuple[int, int]],
    device: str,
    max_new_tokens: int,
    temperature: float,
    out_csv: str,
):
    rows = []
    for rec, base in zip(rec_prompts, base_prompts):
        # Extract KV lists and input ids using canonical pattern
        kv_rec, _ = extract_kv_list(model, tokenizer, rec, device=device)
        kv_base, base_input_ids = extract_kv_list(model, tokenizer, base, device=device)

        # Natural baseline metrics (no KV patch)
        gen_nat = generate_with_kv(
            model,
            tokenizer,
            base,
            past_key_values=None,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            device=device,
        )
        rv_nat = compute_rv(model, tokenizer, gen_nat, device=device)
        beh_nat = behavior_score(gen_nat)
        for r_start, r_end in ranges:
            mixed_cache = mix_kv_to_dynamic_cache(
                kv_base, kv_rec, r_start, r_end, alpha=1.0
            )
            gen = generate_with_mixed_kv(
                model,
                tokenizer,
                base_input_ids,
                mixed_cache,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            rv = compute_rv(model, tokenizer, gen, device=device)
            beh = behavior_score(gen)
            rows.append(
                {
                    "rec_prompt": rec[:80],
                    "base_prompt": base[:80],
                    "range": f"{r_start}-{r_end}",
                    "rv_nat": rv_nat,
                    "beh_nat": beh_nat,
                    "rv_patch": rv,
                    "beh_patch": beh,
                    "gen_patch": gen[:200],
                }
            )
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return out_csv


def run_alpha_mix(
    model,
    tokenizer,
    rec_prompts: List[str],
    base_prompts: List[str],
    alphas: List[float],
    device: str,
    max_new_tokens: int,
    temperature: float,
    out_csv: str,
):
    rows = []
    for rec, base in zip(rec_prompts, base_prompts):
        kv_rec, _ = extract_kv_list(model, tokenizer, rec, device=device)
        kv_base, base_input_ids = extract_kv_list(model, tokenizer, base, device=device)
        for alpha in alphas:
            mixed_cache = mix_kv_to_dynamic_cache(kv_base, kv_rec, 16, 32, alpha=alpha)
            gen = generate_with_mixed_kv(
                model,
                tokenizer,
                base_input_ids,
                mixed_cache,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            rv = compute_rv(model, tokenizer, gen, device=device)
            beh = behavior_score(gen)
            rows.append(
                {
                    "rec_prompt": rec[:80],
                    "base_prompt": base[:80],
                    "alpha": alpha,
                    "temperature": temperature,
                    "rv": rv,
                    "behavior": beh,
                    "gen": gen[:200],
                }
            )
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return out_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="mistralai/Mistral-7B-Instruct-v0.1")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--alpha_temps", nargs="*", type=float, default=[0.3, 0.7, 1.0])
    parser.add_argument("--alphas", nargs="*", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    args = parser.parse_args()

    set_seed(args.seed)
    model, tokenizer = load_model(args.model_name, device=args.device)

    rec_prompts = RECURSIVE_PROMPTS
    base_prompts = BASELINE_PROMPTS

    ranges = [(0, 8), (8, 16), (16, 24), (16, 32), (24, 32)]
    run_layer_ranges(
        model,
        tokenizer,
        rec_prompts,
        base_prompts,
        ranges=ranges,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        out_csv="DEC11_2025_FULL_PIPELINE/results/phase4_kv_ranges.csv",
    )

    for temp in args.alpha_temps:
        run_alpha_mix(
            model,
            tokenizer,
            rec_prompts,
            base_prompts,
            alphas=args.alphas,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            temperature=temp,
            out_csv=f"DEC11_2025_FULL_PIPELINE/results/phase4_alpha_mix_T{temp}.csv",
        )
    print("[phase4] completed KV range and alpha-mix CSVs.")


if __name__ == "__main__":
    main()

