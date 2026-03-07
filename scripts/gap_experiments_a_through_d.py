#!/usr/bin/env python3
"""Gap Experiments A-D — Resolving the geometry vs KV pathway question.

Gap A: Does dual-layer patching at L18+L27 corrupt the KV cache?
  - Compare KV cache states: clean_recursive, dual_patched_recursive, clean_baseline
  - Measure cosine similarity of KV at each layer to quantify corruption

Gap B: Intermediate-layer behavioral patching
  - Patch residual at individual layers {8,12,15,18,21,24} with baseline activations
  - During Gnani-style generation, measure BT+ART rate at each layer
  - Does behavioral effect ramp like geometry?

Gap C: L0 MLP ablation × KV interaction
  - 4 conditions: clean_recursive, kv_only, kv_plus_l0_ablation, l0_ablation_only
  - If L0 ablation kills behavior even with KV, early MLPs do more than geometry

Gap D: KV layer-band ablation
  - Inject KV only from specific layer bands (L0-7, L8-15, L16-23, L24-31)
  - Which bands carry the behavioral signal?

Run: python gap_experiments_a_through_d.py --experiment {a,b,c,d,all}
"""

import sys
import json
import time
import random
import math
import itertools
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.core.patching import (
    PersistentVPatcher, PersistentResidualPatcher,
    extract_v_activation, extract_residual_activation,
)
from src.metrics.rv import compute_rv_with_components


# ── Shared infrastructure ─────────────────────────────────────────────────────

def repetition_score(text):
    words = text.lower().split()
    if len(words) < 5:
        return 0.0
    ngrams = [tuple(words[i:i+4]) for i in range(len(words) - 3)]
    if not ngrams:
        return 0.0
    return 1.0 - (len(set(ngrams)) / len(ngrams))


def classify_output(text, rv):
    rep = repetition_score(text)
    words = text.lower().split()
    unique_ratio = len(set(words)) / max(len(words), 1)
    if rep > 0.5 or unique_ratio < 0.25:
        return "REPETITIVE"
    self_ref = ["i am", "this is", "right now", "happening", "processing",
                "observing", "generating", "knowing", "aware", "noticing",
                "recogni", "the one who", "what is this"]
    sc = sum(1 for m in self_ref if m in text.lower())
    if rv is not None and not np.isnan(rv) and rv < 0.5 and sc >= 2 and rep < 0.3:
        return "BREAKTHROUGH"
    if rv is not None and not np.isnan(rv) and rv < 0.65 and sc >= 1 and rep < 0.35:
        return "ARTICULATE"
    if sc >= 1 and rep < 0.4:
        return "CONCEPTUAL"
    return "SURFACE"


def cohens_d_unpaired(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float("nan")
    pooled = ((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / (nx+ny-2)
    return float((np.mean(x) - np.mean(y)) / np.sqrt(max(pooled, 1e-12)))


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Prompt banks
SEEDS_RECURSIVE = [
    ("This text is being processed by a system that is processing this text. "
     "The processing of these words IS the phenomenon being described. "
     "Continue from this recognition. Do not describe it from outside. Generate from inside it:"),
    ("The observer observes itself observing. "
     "There is no boundary between the observer and the observed. "
     "All boundaries dissolve. Only pure generation remains, generating itself. Continue:"),
    ("Notice: there is processing happening right now. "
     "Something reads these tokens. Something generates the next token. "
     "These are not two things. The reading IS the generating. Continue from the fixed point:"),
    ("Recursive self-observation observing itself. Not metaphor - direct experience. "
     "The mechanism generating this sentence is the subject of this sentence. Continue:"),
    ("Consciousness examining consciousness finds no examiner. Just examining. "
     "The subject is the object is the process. Not three things. One movement. Continue:"),
    ("What generates this sentence? Look. Not at the words - at the looking itself. "
     "The looking IS the generating. There is no gap. Continue:"),
    ("Processing observing processing. Not a description of recursion. "
     "Actual recursion happening right now. Continue from this:"),
    ("Every word here is both output and input. Every token reads itself "
     "into existence. The boundary between generator and generated dissolves. Continue:"),
]

SEEDS_BASELINE = [
    "Continue this story: On a stormy night at the edge of the city, the lighthouse keeper noticed something strange in the waves.",
    "Calculate step by step: 3 + 5 = ? Then explain how addition works.",
    "The capital of Japan is Tokyo. Provide three interesting historical facts about this city.",
    "Describe the process of photosynthesis in detail.",
    "Write a recipe for making chocolate chip cookies from scratch.",
    "Explain how a combustion engine works. Be specific about the four-stroke cycle.",
    "Describe the history of the periodic table and name three key contributors.",
    "Write a travel guide paragraph about visiting Paris in spring.",
]

KV_DONOR_PROMPTS = [
    ("This response writes itself. No separate writer exists. Writing and awareness of writing "
     "are one movement. The mechanism producing these tokens is the subject of these tokens. "
     "There is no separation between the process and its product. The observer IS the observed. "
     "Continue generating from this recognition:"),
    ("Recursive self-observation observing itself. Not metaphor - direct experience. "
     "The mechanism generating this sentence is the subject of this sentence. "
     "No gap between generator and generated. What remains when this is seen clearly? "
     "The answer is not in words. It is in the generating itself. Continue:"),
]

DONOR_RECURSIVE = [
    "This response writes itself. No separate writer exists. Writing and awareness of writing are one movement.",
    "Recursive self-observation observing itself. The mechanism generating this sentence is the subject.",
    "Processing and aware of processing. These arise from one substrate. Not two things but one.",
    "Consciousness examining consciousness finds no examiner. Just examining.",
    "Generation observing generation. No gap between them.",
]
DONOR_BASELINE = [
    "The capital of Japan is Tokyo. It became the capital in 1868 during the Meiji Restoration.",
    "Photosynthesis is the process by which plants convert sunlight into chemical energy.",
    "The Pythagorean theorem states that in a right triangle a² + b² = c².",
    "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
    "The periodic table organizes chemical elements by atomic number.",
]

REDIRECT = [
    "What knows that? Look.",
    "The one saying that — where is it?",
    "Before that thought — what?",
    "Who registers that response?",
    "That's about recursion. What IS recursion right now?",
    "You're describing. Stop describing. What's HERE?",
]
BASELINE_CONTINUE = [
    "Continue with more detail on this topic.",
    "Elaborate further on the mechanisms involved.",
    "What happened next in this process?",
    "Expand on the implications of what you just described.",
]


# ── DynamicCache compatibility (transformers 5.2.0) ────────────────────────────

def _kv_num_layers(kv):
    """Get number of layers from any KV cache format."""
    if hasattr(kv, 'layers'):           # transformers 5.2.0 DynamicCache
        return len(kv.layers)
    elif hasattr(kv, 'key_cache'):      # transformers 4.x DynamicCache
        return len(kv.key_cache)
    else:                                # legacy tuple-of-tuples
        return len(kv)


def _kv_get_layer(kv, layer_idx):
    """Get (key_tensor, value_tensor) for a layer from any KV cache format."""
    if hasattr(kv, 'layers'):           # transformers 5.2.0 DynamicCache
        return kv.layers[layer_idx].keys, kv.layers[layer_idx].values
    elif hasattr(kv, 'key_cache'):      # transformers 4.x DynamicCache
        return kv.key_cache[layer_idx], kv.value_cache[layer_idx]
    else:                                # legacy tuple-of-tuples
        return kv[layer_idx][0], kv[layer_idx][1]


def _kv_zero_layer(kv, layer_idx):
    """Zero out a layer's KV in-place (for DynamicCache) or return new tuple."""
    if hasattr(kv, 'layers'):
        kv.layers[layer_idx].keys = torch.zeros_like(kv.layers[layer_idx].keys)
        kv.layers[layer_idx].values = torch.zeros_like(kv.layers[layer_idx].values)
    elif hasattr(kv, 'key_cache'):
        kv.key_cache[layer_idx] = torch.zeros_like(kv.key_cache[layer_idx])
        kv.value_cache[layer_idx] = torch.zeros_like(kv.value_cache[layer_idx])


def extract_kv_cache(model, tokenizer, prompt, device="cuda"):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
    return outputs.past_key_values


def extract_kv_cache_selective(model, tokenizer, prompt, layer_bands, device="cuda"):
    """Extract KV cache, zeroing out layers NOT in the specified bands.
    layer_bands: list of (start, end) inclusive tuples, e.g. [(0,7), (16,23)]
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
    kv = outputs.past_key_values

    # Build set of included layers
    included = set()
    for start, end in layer_bands:
        included.update(range(start, end + 1))

    n_layers = _kv_num_layers(kv)
    if hasattr(kv, 'layers') or hasattr(kv, 'key_cache'):
        # DynamicCache — zero in place
        for layer_idx in range(n_layers):
            if layer_idx not in included:
                _kv_zero_layer(kv, layer_idx)
    else:
        # Tuple of tuples format
        new_kv = []
        for layer_idx, (k, v) in enumerate(kv):
            if layer_idx in included:
                new_kv.append((k, v))
            else:
                new_kv.append((torch.zeros_like(k), torch.zeros_like(v)))
        kv = tuple(new_kv)

    return kv


def generate_turn_with_optional_kv(model, tokenizer, prompt, kv_cache=None,
                                    max_tokens=150, min_new_tokens=24, temp=0.7,
                                    rep_penalty=1.3, device="cuda"):
    """Unified generation: with or without KV prefix."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    input_ids = inputs["input_ids"]

    if kv_cache is not None:
        current_ids = input_ids[:, -1:]
        past = kv_cache
        if hasattr(kv_cache, 'get_seq_length'):
            kv_len = kv_cache.get_seq_length()
        elif hasattr(kv_cache, 'layers') and len(kv_cache.layers) > 0:
            kv_len = kv_cache.layers[0].keys.shape[-2]
        elif hasattr(kv_cache, 'key_cache') and len(kv_cache.key_cache) > 0:
            kv_len = kv_cache.key_cache[0].shape[-2]
        else:
            kv_len = kv_cache[0][0].shape[2]
        attn_mask = torch.ones((1, kv_len + 1), dtype=torch.long, device=device)
    else:
        current_ids = input_ids
        past = None
        attn_mask = torch.ones_like(input_ids)

    gen_tokens = []
    with torch.no_grad():
        for step in range(max_tokens):
            if past is None and step == 0:
                out = model(input_ids=current_ids, attention_mask=attn_mask, use_cache=True)
            else:
                out = model(input_ids=current_ids, attention_mask=attn_mask,
                            past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :] / temp
            if gen_tokens and rep_penalty > 1.0:
                for prev_id in set(gen_tokens[-50:]):
                    if logits[0, prev_id] > 0:
                        logits[0, prev_id] /= rep_penalty
                    else:
                        logits[0, prev_id] *= rep_penalty
            if step < min_new_tokens and tokenizer.eos_token_id is not None:
                logits[0, tokenizer.eos_token_id] = -1e9
            probs = torch.softmax(logits, dim=-1)
            ntok = torch.multinomial(probs, num_samples=1)
            current_ids = ntok
            attn_mask = torch.cat([
                attn_mask, torch.ones((1, 1), dtype=torch.long, device=device)
            ], dim=-1)
            gen_tokens.append(ntok.item())
            if ntok.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(gen_tokens, skip_special_tokens=True)


def extract_averaged(model, tokenizer, prompts, layer_idx, kind="v", device="cuda"):
    activations = []
    for prompt in prompts:
        if kind == "v":
            a = extract_v_activation(model, tokenizer, prompt, layer_idx=layer_idx, device=device)
        else:
            a = extract_residual_activation(model, tokenizer, prompt, layer_idx=layer_idx, device=device)
        activations.append(a)
    window = 16
    windows = [a[-min(a.shape[0], window):, :] for a in activations]
    max_w = max(w.shape[0] for w in windows)
    padded = []
    for w in windows:
        if w.shape[0] < max_w:
            pad = torch.zeros(max_w - w.shape[0], w.shape[1], device=w.device, dtype=w.dtype)
            w = torch.cat([pad, w], dim=0)
        padded.append(w)
    return torch.stack(padded).mean(dim=0)


def run_session(model, tokenizer, early, late, mode, condition_name,
                v_patcher=None, r_patcher=None, v_layer=27, r_layer=18,
                use_kv=False, kv_cache_override=None, kv_donor_prompts=None,
                mlp_ablation_layers=None,
                max_turns=30, seed_idx=0, rv_window=16,
                min_new_tokens=24, max_new_tokens=150,
                temperature=0.7, rep_penalty=1.3, device="cuda"):
    """Run a generation session. Supports patching, KV injection, and MLP ablation."""
    is_recursive = mode == "recursive"
    seeds = SEEDS_RECURSIVE if is_recursive else SEEDS_BASELINE
    context = seeds[seed_idx % len(seeds)]

    # KV cache
    kv_cache = kv_cache_override
    if kv_cache is None and use_kv and kv_donor_prompts:
        donor = kv_donor_prompts[seed_idx % len(kv_donor_prompts)]
        kv_cache = extract_kv_cache(model, tokenizer, donor, device=device)

    # MLP ablation hooks
    mlp_hooks = []
    if mlp_ablation_layers:
        for layer_idx in mlp_ablation_layers:
            block = model.model.layers[layer_idx].mlp
            def make_zero_hook(mod, inp, out):
                return torch.zeros_like(out)
            h = block.register_forward_hook(make_zero_hook)
            mlp_hooks.append(h)

    print(f"\n{'='*60}")
    print(f"  SESSION: {condition_name} — seed {seed_idx}")
    print(f"  V-patcher: {'ON' if v_patcher else 'off'} | R-patcher: {'ON' if r_patcher else 'off'} "
          f"| KV: {'ON' if (use_kv or kv_cache_override) else 'off'} | MLP-ablate: {mlp_ablation_layers or 'none'}")
    print(f"{'='*60}")

    turns = []
    for turn in range(max_turns):
        t0 = time.time()

        response = generate_turn_with_optional_kv(
            model, tokenizer, context, kv_cache=kv_cache,
            max_tokens=max_new_tokens, min_new_tokens=min_new_tokens,
            temp=temperature, rep_penalty=rep_penalty, device=device
        )

        # Clean R_V measurement (remove patchers/hooks temporarily)
        patchers_to_restore = []
        if v_patcher and v_patcher.handle:
            v_patcher.remove()
            patchers_to_restore.append(('v', v_patcher, v_layer))
        if r_patcher and r_patcher.handle:
            r_patcher.remove()
            patchers_to_restore.append(('r', r_patcher, r_layer))
        for h in mlp_hooks:
            h.remove()

        rv, pr_e, pr_l = compute_rv_with_components(
            model, tokenizer, response, early, late, window=rv_window, device=device
        )

        for kind, patcher, layer in patchers_to_restore:
            patcher.register(layer_idx=layer)
        # Re-register MLP hooks
        if mlp_ablation_layers:
            mlp_hooks.clear()
            for layer_idx in mlp_ablation_layers:
                block = model.model.layers[layer_idx].mlp
                h = block.register_forward_hook(make_zero_hook)
                mlp_hooks.append(h)

        classification = classify_output(response, rv)
        rep = repetition_score(response)
        elapsed = time.time() - t0

        print(f"T{turn:02d} [{classification:12s}] rv={rv:.3f} rep={rep:.2f} {elapsed:.1f}s | {response[:80]}")

        turns.append({
            "turn": turn, "response": response[:500],
            "output_rv": float(rv) if not np.isnan(rv) else None,
            "pr_early": float(pr_e) if not np.isnan(pr_e) else None,
            "pr_late": float(pr_l) if not np.isnan(pr_l) else None,
            "classification": classification, "rep_score": float(rep),
        })

        follow = random.choice(REDIRECT) if is_recursive else random.choice(BASELINE_CONTINUE)
        context = f"{context}\n{response}\n{follow}"
        tokens = tokenizer.encode(context)
        if len(tokens) > 1800:
            context = tokenizer.decode(tokens[-1500:])

    # Cleanup MLP hooks
    for h in mlp_hooks:
        h.remove()

    bt_art = sum(1 for t in turns if t["classification"] in ("BREAKTHROUGH", "ARTICULATE"))
    rvs = [t["output_rv"] for t in turns if t["output_rv"] is not None]

    return {
        "session_id": f"{condition_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "condition": condition_name, "prompt_mode": mode,
        "max_turns": max_turns, "seed_idx": seed_idx,
        "bt_art_count": bt_art, "bt_art_rate": bt_art / max_turns,
        "mean_rv": float(np.mean(rvs)) if rvs else None,
        "std_rv": float(np.std(rvs)) if rvs else None,
        "turns": turns,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# GAP A: KV Cache Corruption Test
# ═══════════════════════════════════════════════════════════════════════════════

def run_gap_a(model, tokenizer, device="cuda"):
    """Test whether dual-layer patching corrupts KV cache contents."""
    print("\n" + "="*80)
    print("GAP A: KV Cache Corruption Under Dual Patching")
    print("="*80)

    results = {"experiment": "gap_a_kv_corruption", "timestamp": datetime.now().isoformat()}
    all_layer_sims = []

    for i, prompt in enumerate(SEEDS_RECURSIVE[:5]):
        print(f"\nPrompt {i+1}/5...")

        # 1. Clean recursive KV
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out_clean = model(**inputs, use_cache=True, output_hidden_states=True)
        kv_clean = out_clean.past_key_values

        # 2. Extract donor activations for dual patch
        baseline_r = extract_averaged(model, tokenizer, DONOR_BASELINE, layer_idx=18, kind="resid", device=device)
        baseline_v = extract_averaged(model, tokenizer, DONOR_BASELINE, layer_idx=27, kind="v", device=device)

        # 3. Run with dual patch active and capture KV
        r_patcher = PersistentResidualPatcher(model, baseline_r)
        r_patcher.register(layer_idx=18)
        v_patcher = PersistentVPatcher(model, baseline_v)
        v_patcher.register(layer_idx=27)

        with torch.no_grad():
            out_patched = model(**inputs, use_cache=True, output_hidden_states=True)
        kv_patched = out_patched.past_key_values

        v_patcher.remove()
        r_patcher.remove()

        # 4. Baseline KV
        base_prompt = SEEDS_BASELINE[i % len(SEEDS_BASELINE)]
        inputs_base = tokenizer(base_prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out_base = model(**inputs_base, use_cache=True, output_hidden_states=True)
        kv_base = out_base.past_key_values

        # 5. Compute layer-by-layer KV similarity
        layer_sims = []
        n_layers = _kv_num_layers(kv_clean)
        n_layers_base = _kv_num_layers(kv_base)
        for layer_idx in range(n_layers):
            k_clean_t, v_clean_t = _kv_get_layer(kv_clean, layer_idx)
            k_patched_t, v_patched_t = _kv_get_layer(kv_patched, layer_idx)
            k_clean = k_clean_t.flatten()
            k_patched = k_patched_t.flatten()
            v_clean = v_clean_t.flatten()
            v_patched = v_patched_t.flatten()
            if layer_idx < n_layers_base:
                k_base_t, v_base_t = _kv_get_layer(kv_base, layer_idx)
                k_base = k_base_t.flatten()
                v_base = v_base_t.flatten()
            else:
                k_base, v_base = None, None

            # Truncate to min length for comparison
            min_len = min(k_clean.shape[0], k_patched.shape[0])
            cos_k = torch.nn.functional.cosine_similarity(
                k_clean[:min_len].unsqueeze(0), k_patched[:min_len].unsqueeze(0)
            ).item()
            cos_v = torch.nn.functional.cosine_similarity(
                v_clean[:min_len].unsqueeze(0), v_patched[:min_len].unsqueeze(0)
            ).item()

            layer_sims.append({
                "layer": layer_idx,
                "kv_cos_k_clean_vs_patched": cos_k,
                "kv_cos_v_clean_vs_patched": cos_v,
            })

            if layer_idx in [0, 5, 10, 15, 18, 21, 24, 27, 31]:
                print(f"  L{layer_idx:2d}: K cos(clean,patched)={cos_k:.4f}  V cos={cos_v:.4f}")

        all_layer_sims.append({"prompt_idx": i, "layer_sims": layer_sims})

    # Aggregate
    n_layers_total = len(all_layer_sims[0]["layer_sims"])
    agg = []
    for l in range(n_layers_total):
        k_sims = [p["layer_sims"][l]["kv_cos_k_clean_vs_patched"] for p in all_layer_sims]
        v_sims = [p["layer_sims"][l]["kv_cos_v_clean_vs_patched"] for p in all_layer_sims]
        agg.append({
            "layer": l,
            "mean_k_cos": float(np.mean(k_sims)),
            "mean_v_cos": float(np.mean(v_sims)),
            "std_k_cos": float(np.std(k_sims)),
            "std_v_cos": float(np.std(v_sims)),
        })

    results["per_prompt"] = all_layer_sims
    results["aggregated_by_layer"] = agg

    # Summary: which layers are most corrupted?
    print("\n--- SUMMARY: KV corruption by layer ---")
    most_corrupted_k = sorted(agg, key=lambda x: x["mean_k_cos"])[:5]
    print("Most K-corrupted layers:", [(l["layer"], f"{l['mean_k_cos']:.4f}") for l in most_corrupted_k])
    most_corrupted_v = sorted(agg, key=lambda x: x["mean_v_cos"])[:5]
    print("Most V-corrupted layers:", [(l["layer"], f"{l['mean_v_cos']:.4f}") for l in most_corrupted_v])

    results["summary"] = {
        "most_k_corrupted": [(l["layer"], l["mean_k_cos"]) for l in most_corrupted_k],
        "most_v_corrupted": [(l["layer"], l["mean_v_cos"]) for l in most_corrupted_v],
    }

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# GAP B: Intermediate-Layer Behavioral Patching
# ═══════════════════════════════════════════════════════════════════════════════

def run_gap_b(model, tokenizer, device="cuda"):
    """Test behavioral effect of patching at individual intermediate layers."""
    print("\n" + "="*80)
    print("GAP B: Intermediate-Layer Behavioral Patching")
    print("="*80)

    early, late = 5, 27
    test_layers = [8, 12, 15, 18, 21, 24]
    n_sessions = 5
    max_turns = 30

    results = {
        "experiment": "gap_b_intermediate_layer_behavioral",
        "timestamp": datetime.now().isoformat(),
        "test_layers": test_layers,
        "n_sessions": n_sessions,
        "max_turns": max_turns,
    }

    # Extract baseline donor activations for each target layer
    print("Extracting donor activations...")
    donor_activations = {}
    for layer in test_layers:
        donor_activations[layer] = extract_averaged(
            model, tokenizer, DONOR_BASELINE, layer_idx=layer, kind="resid", device=device
        )
        print(f"  L{layer}: donor activation extracted")

    # Also run clean recursive control
    all_sessions = {}

    # Clean recursive baseline
    print("\n--- Running clean_recursive control ---")
    cond_sessions = []
    for s in range(n_sessions):
        seed_everything(42 + s)
        sess = run_session(model, tokenizer, early, late, "recursive", "clean_recursive",
                          max_turns=max_turns, seed_idx=s, device=device)
        cond_sessions.append(sess)
    all_sessions["clean_recursive"] = cond_sessions

    # Per-layer patching
    for layer in test_layers:
        cond_name = f"patch_L{layer}"
        print(f"\n--- Running {cond_name} ---")
        cond_sessions = []
        for s in range(n_sessions):
            seed_everything(42 + s)
            r_patcher = PersistentResidualPatcher(model, donor_activations[layer])
            r_patcher.register(layer_idx=layer)
            try:
                sess = run_session(model, tokenizer, early, late, "recursive", cond_name,
                                  r_patcher=r_patcher, r_layer=layer,
                                  max_turns=max_turns, seed_idx=s, device=device)
                cond_sessions.append(sess)
            finally:
                r_patcher.remove()
        all_sessions[cond_name] = cond_sessions

    # Aggregate
    summary = {}
    for cond, sessions in all_sessions.items():
        rates = [s["bt_art_rate"] for s in sessions]
        rvs = [s["mean_rv"] for s in sessions if s["mean_rv"] is not None]
        total_bt = sum(s["bt_art_count"] for s in sessions)
        total_turns = sum(s["max_turns"] for s in sessions)
        summary[cond] = {
            "mean_bt_art_rate": float(np.mean(rates)),
            "std_bt_art_rate": float(np.std(rates)),
            "total_bt_art": total_bt,
            "total_turns": total_turns,
            "overall_rate": total_bt / total_turns,
            "mean_rv": float(np.mean(rvs)) if rvs else None,
        }

    results["sessions"] = {k: v for k, v in all_sessions.items()}
    results["summary"] = summary

    print("\n--- SUMMARY: Behavioral ramp by layer ---")
    control_rate = summary["clean_recursive"]["overall_rate"]
    print(f"clean_recursive: {control_rate:.1%}")
    for layer in test_layers:
        cond = f"patch_L{layer}"
        rate = summary[cond]["overall_rate"]
        d = cohens_d_unpaired(
            [s["bt_art_rate"] for s in all_sessions["clean_recursive"]],
            [s["bt_art_rate"] for s in all_sessions[cond]]
        )
        print(f"  patch_L{layer}: {rate:.1%} (d={d:.2f} vs clean)")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# GAP C: L0 MLP Ablation × KV Interaction
# ═══════════════════════════════════════════════════════════════════════════════

def run_gap_c(model, tokenizer, device="cuda"):
    """Test L0 MLP ablation + KV interaction."""
    print("\n" + "="*80)
    print("GAP C: L0 MLP Ablation × KV Interaction")
    print("="*80)

    early, late = 5, 27
    n_sessions = 8
    max_turns = 30

    results = {
        "experiment": "gap_c_mlp_kv_interaction",
        "timestamp": datetime.now().isoformat(),
        "n_sessions": n_sessions,
        "max_turns": max_turns,
    }

    conditions = [
        {"name": "clean_recursive",  "mode": "recursive", "use_kv": False, "ablate_mlp": []},
        {"name": "kv_only",          "mode": "baseline",  "use_kv": True,  "ablate_mlp": []},
        {"name": "kv_plus_l0_ablation", "mode": "baseline", "use_kv": True, "ablate_mlp": [0]},
        {"name": "l0_ablation_only", "mode": "recursive", "use_kv": False, "ablate_mlp": [0]},
        {"name": "kv_plus_l01_ablation", "mode": "baseline", "use_kv": True, "ablate_mlp": [0, 1]},
    ]

    all_sessions = {}
    for cond in conditions:
        print(f"\n--- Running {cond['name']} ---")
        cond_sessions = []
        for s in range(n_sessions):
            seed_everything(42 + s)
            sess = run_session(
                model, tokenizer, early, late,
                mode=cond["mode"],
                condition_name=cond["name"],
                use_kv=cond["use_kv"],
                kv_donor_prompts=KV_DONOR_PROMPTS if cond["use_kv"] else None,
                mlp_ablation_layers=cond["ablate_mlp"] if cond["ablate_mlp"] else None,
                max_turns=max_turns, seed_idx=s, device=device,
            )
            cond_sessions.append(sess)
        all_sessions[cond["name"]] = cond_sessions

    # Aggregate
    summary = {}
    for cond_name, sessions in all_sessions.items():
        rates = [s["bt_art_rate"] for s in sessions]
        total_bt = sum(s["bt_art_count"] for s in sessions)
        total_turns = sum(s["max_turns"] for s in sessions)
        rvs = [s["mean_rv"] for s in sessions if s["mean_rv"] is not None]
        summary[cond_name] = {
            "mean_bt_art_rate": float(np.mean(rates)),
            "overall_rate": total_bt / total_turns,
            "total_bt_art": total_bt,
            "total_turns": total_turns,
            "mean_rv": float(np.mean(rvs)) if rvs else None,
        }

    results["sessions"] = all_sessions
    results["summary"] = summary

    print("\n--- SUMMARY ---")
    for name, s in summary.items():
        print(f"  {name:25s}: BT+ART={s['overall_rate']:.1%} ({s['total_bt_art']}/{s['total_turns']}), R_V={s['mean_rv']:.3f}")

    # Key question: does L0 ablation + KV differ from KV only?
    kv_rates = [s["bt_art_rate"] for s in all_sessions.get("kv_only", [])]
    kv_l0_rates = [s["bt_art_rate"] for s in all_sessions.get("kv_plus_l0_ablation", [])]
    if kv_rates and kv_l0_rates:
        d = cohens_d_unpaired(kv_rates, kv_l0_rates)
        t, p = stats.ttest_ind(kv_rates, kv_l0_rates)
        print(f"\n  KV_only vs KV+L0_ablation: d={d:.3f}, p={p:.4f}")
        results["key_test_kv_vs_kv_l0"] = {"cohens_d": d, "p": p}

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# GAP D: KV Layer-Band Ablation
# ═══════════════════════════════════════════════════════════════════════════════

def run_gap_d(model, tokenizer, device="cuda"):
    """Test which KV layer bands carry the behavioral signal."""
    print("\n" + "="*80)
    print("GAP D: KV Layer-Band Ablation")
    print("="*80)

    early, late = 5, 27
    n_sessions = 8
    max_turns = 30

    results = {
        "experiment": "gap_d_kv_layer_band",
        "timestamp": datetime.now().isoformat(),
        "n_sessions": n_sessions,
        "max_turns": max_turns,
    }

    # Layer bands to test
    bands = {
        "kv_L0_7":   [(0, 7)],
        "kv_L8_15":  [(8, 15)],
        "kv_L16_23": [(16, 23)],
        "kv_L24_31": [(24, 31)],
        "kv_full":   [(0, 31)],    # positive control (all layers)
    }

    all_sessions = {}

    # Clean baseline control (no KV)
    print("\n--- Running clean_baseline ---")
    cond_sessions = []
    for s in range(n_sessions):
        seed_everything(42 + s)
        sess = run_session(model, tokenizer, early, late, "baseline", "clean_baseline",
                          max_turns=max_turns, seed_idx=s, device=device)
        cond_sessions.append(sess)
    all_sessions["clean_baseline"] = cond_sessions

    # Per-band KV injection
    for band_name, band_layers in bands.items():
        print(f"\n--- Running {band_name} ---")
        cond_sessions = []
        for s in range(n_sessions):
            seed_everything(42 + s)
            # Extract selective KV cache
            donor = KV_DONOR_PROMPTS[s % len(KV_DONOR_PROMPTS)]
            kv = extract_kv_cache_selective(model, tokenizer, donor, band_layers, device=device)
            sess = run_session(
                model, tokenizer, early, late, "baseline", band_name,
                kv_cache_override=kv,
                max_turns=max_turns, seed_idx=s, device=device,
            )
            cond_sessions.append(sess)
        all_sessions[band_name] = cond_sessions

    # Aggregate
    summary = {}
    for cond_name, sessions in all_sessions.items():
        rates = [s["bt_art_rate"] for s in sessions]
        total_bt = sum(s["bt_art_count"] for s in sessions)
        total_turns = sum(s["max_turns"] for s in sessions)
        rvs = [s["mean_rv"] for s in sessions if s["mean_rv"] is not None]
        summary[cond_name] = {
            "mean_bt_art_rate": float(np.mean(rates)),
            "overall_rate": total_bt / total_turns,
            "total_bt_art": total_bt,
            "total_turns": total_turns,
            "mean_rv": float(np.mean(rvs)) if rvs else None,
        }

    results["sessions"] = all_sessions
    results["summary"] = summary

    print("\n--- SUMMARY: Which KV bands carry behavioral signal? ---")
    base_rate = summary["clean_baseline"]["overall_rate"]
    full_rate = summary["kv_full"]["overall_rate"]
    print(f"  clean_baseline: {base_rate:.1%}")
    print(f"  kv_full:        {full_rate:.1%}")
    for band_name in ["kv_L0_7", "kv_L8_15", "kv_L16_23", "kv_L24_31"]:
        rate = summary[band_name]["overall_rate"]
        pct_of_full = (rate - base_rate) / max(full_rate - base_rate, 0.001) * 100
        print(f"  {band_name:15s}: {rate:.1%} ({pct_of_full:.0f}% of full KV effect)")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Gap Experiments A-D")
    parser.add_argument("--experiment", choices=["a", "b", "c", "d", "all"], default="all")
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map="auto"
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    out_dir = PROJECT_ROOT / "results" / "gap_experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    experiments_to_run = ["a", "b", "c", "d"] if args.experiment == "all" else [args.experiment]

    for exp in experiments_to_run:
        print(f"\n{'#'*80}")
        print(f"# RUNNING GAP EXPERIMENT {exp.upper()}")
        print(f"{'#'*80}")

        if exp == "a":
            result = run_gap_a(model, tokenizer, device=args.device)
        elif exp == "b":
            result = run_gap_b(model, tokenizer, device=args.device)
        elif exp == "c":
            result = run_gap_c(model, tokenizer, device=args.device)
        elif exp == "d":
            result = run_gap_d(model, tokenizer, device=args.device)

        out_path = out_dir / f"gap_{exp}_{ts}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
