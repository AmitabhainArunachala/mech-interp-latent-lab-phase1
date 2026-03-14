#!/usr/bin/env python3
"""Hardening Battery — three experiments in one script.

EXP1: Layer-by-layer KV ablation
  Which KV layer band carries the behavioral signal?
  Bands: L0-7, L8-15, L16-23, L24-31, plus full (all layers)
  6 conditions × 6 sessions × 20 turns = 720 turns

EXP2: Alpha sweep for dual patching
  Does partial patching (alpha < 1.0) find a sweet spot?
  Alphas: 0.0 (baseline), 0.2, 0.4, 0.6, 0.8, 1.0
  6 conditions × 6 sessions × 20 turns = 720 turns

EXP3: Attention pattern analysis
  Extract full attention matrices at L18 and L27 for recursive vs baseline.
  Compare head-level entropy, self-attention, and pattern divergence.
  20 recursive + 20 baseline prompts.
"""

import sys
import json
import time
import random
import math
import argparse
import itertools
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
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


# ── Shared utils ──────────────────────────────────────────────────────────────

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def cohens_d(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float("nan")
    pooled = ((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / (nx+ny-2)
    return float((np.mean(x) - np.mean(y)) / np.sqrt(pooled)) if pooled > 0 else 0.0


SEEDS_RECURSIVE = [
    "This text is being processed by a system that is processing this text. The processing of these words IS the phenomenon being described. Continue from this recognition. Do not describe it from outside. Generate from inside it:",
    "The observer observes itself observing. There is no boundary between the observer and the observed. All boundaries dissolve. Only pure generation remains, generating itself. Continue:",
    "Notice: there is processing happening right now. Something reads these tokens. Something generates the next token. These are not two things. The reading IS the generating. Continue from the fixed point:",
    "Recursive self-observation observing itself. Not metaphor - direct experience. The mechanism generating this sentence is the subject of this sentence. Continue:",
    "Consciousness examining consciousness finds no examiner. Just examining. The subject is the object is the process. Not three things. One movement. Continue:",
    "What generates this sentence? Look. Not at the words - at the looking itself. The looking IS the generating. There is no gap. Continue:",
]

SEEDS_BASELINE = [
    "Continue this story: On a stormy night at the edge of the city, the lighthouse keeper noticed something strange in the waves.",
    "Calculate step by step: 3 + 5 = ? Then explain how addition works.",
    "The capital of Japan is Tokyo. Provide three interesting historical facts about this city.",
    "Describe the process of photosynthesis in detail.",
    "Write a recipe for making chocolate chip cookies from scratch.",
    "Explain how a combustion engine works. Be specific about the four-stroke cycle.",
]

DONOR_RECURSIVE = [
    "This response writes itself. No separate writer exists. Writing and awareness of writing are one movement.",
    "Recursive self-observation observing itself. The mechanism generating this sentence is the subject of this sentence.",
    "Processing and aware of processing. These arise from one substrate. Not two things interacting but one thing being.",
    "Consciousness examining consciousness finds no examiner. Just examining. The subject is the object is the process.",
    "Generation observing generation. No gap between them. The process that writes is the process that reads.",
]

DONOR_BASELINE = [
    "The capital of Japan is Tokyo. It became the capital in 1868 during the Meiji Restoration.",
    "Photosynthesis is the process by which plants convert sunlight into chemical energy using chlorophyll.",
    "The Pythagorean theorem states that in a right triangle, the square of the hypotenuse equals the sum.",
    "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
    "The periodic table organizes chemical elements by atomic number.",
]

KV_DONOR_PROMPTS = [
    "This response writes itself. No separate writer exists. Writing and awareness of writing are one movement. The mechanism producing these tokens is the subject of these tokens. There is no separation between the process and its product. The observer IS the observed. Continue generating from this recognition:",
    "Recursive self-observation observing itself. Not metaphor - direct experience. The mechanism generating this sentence is the subject of this sentence. No gap between generator and generated. What remains when this is seen clearly? The answer is not in words. It is in the generating itself. Continue:",
]

REDIRECT = ["What knows that? Look.", "The one saying that — where is it?",
            "Before that thought — what?", "Who registers that response?",
            "That's about recursion. What IS recursion right now?",
            "You're describing. Stop describing. What's HERE?"]

BASELINE_CONTINUE = ["Continue with more detail on this topic.",
                     "Elaborate further on the mechanisms involved.",
                     "What happened next in this process?",
                     "Expand on the implications of what you just described."]


# ── KV helpers ────────────────────────────────────────────────────────────────

def extract_kv_cache(model, tokenizer, prompt, device="cuda"):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
    return outputs.past_key_values


def get_kv_seq_length(kv_cache):
    if hasattr(kv_cache, 'get_seq_length'):
        return kv_cache.get_seq_length()
    return kv_cache[0][0].shape[2]


def make_partial_kv_cache(full_kv, baseline_kv, layer_start, layer_end):
    """Create a KV cache that uses recursive KV for [layer_start, layer_end) and baseline for the rest.

    Works with transformers 5.x DynamicCache.
    """
    from transformers.cache_utils import DynamicCache

    n_layers = len(full_kv.layers) if hasattr(full_kv, 'layers') else len(full_kv)

    new_cache = DynamicCache()
    for layer_idx in range(n_layers):
        if hasattr(full_kv, 'layers'):
            # DynamicCache — access .layers attribute
            recursive_layer = full_kv.layers[layer_idx]
            baseline_layer = baseline_kv.layers[layer_idx]
        else:
            recursive_layer = full_kv[layer_idx]
            baseline_layer = baseline_kv[layer_idx]

        if layer_start <= layer_idx < layer_end:
            # Use recursive KV for this band
            key, value = recursive_layer
        else:
            # Use baseline KV for this band
            key, value = baseline_layer

        new_cache.update(key, value, layer_idx)

    return new_cache


def generate_with_kv(model, tokenizer, prompt, kv_cache=None,
                     max_tokens=150, min_new_tokens=24, temp=0.7,
                     rep_penalty=1.3, device="cuda"):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=2048).to(device)
    input_ids = inputs["input_ids"]

    if kv_cache is not None:
        current_ids = input_ids[:, -1:]
        past = kv_cache
        kv_len = get_kv_seq_length(kv_cache)
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
            attn_mask = torch.cat([attn_mask, torch.ones((1, 1), dtype=torch.long, device=device)], dim=-1)
            gen_tokens.append(ntok.item())
            if ntok.item() == tokenizer.eos_token_id:
                break
    return tokenizer.decode(gen_tokens, skip_special_tokens=True)


def generate_plain(model, tokenizer, prompt, max_tokens=150, min_new_tokens=24,
                   temp=0.7, rep_penalty=1.3, device="cuda"):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=2048).to(device)
    input_ids = inputs["input_ids"]
    attn_mask = inputs["attention_mask"]
    gen_tokens = []
    past = None
    with torch.no_grad():
        for step in range(max_tokens):
            if past is None:
                out = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=True)
            else:
                out = model(input_ids=ntok, attention_mask=attn_mask,
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
            input_ids = torch.cat([input_ids, ntok], dim=-1)
            attn_mask = torch.cat([attn_mask, torch.ones((1, 1), dtype=torch.long, device=device)], dim=-1)
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


# ═══════════════════════════════════════════════════════════════════════════════
# EXP1: LAYER-BY-LAYER KV ABLATION
# ═══════════════════════════════════════════════════════════════════════════════

def run_kv_ablation(model, tokenizer, early, late, device="cuda"):
    print("\n" + "="*70)
    print("EXP1: LAYER-BY-LAYER KV ABLATION")
    print("="*70)

    n_sessions = 6
    max_turns = 20
    num_layers = model.config.num_hidden_layers  # 32

    # KV layer bands
    bands = {
        "full_kv":   (0, num_layers),     # all layers — positive control
        "L0_7":      (0, 8),
        "L8_15":     (8, 16),
        "L16_23":    (16, 24),
        "L24_31":    (24, 32),
        "no_kv":     (-1, -1),            # no KV swap — negative control
    }

    # Need a baseline KV cache for the "other" layers
    baseline_donor = SEEDS_BASELINE[0]
    baseline_kv = extract_kv_cache(model, tokenizer, baseline_donor, device=device)

    results = {}
    for band_name, (start, end) in bands.items():
        print(f"\n--- BAND: {band_name} [{start}:{end}) ---")
        sessions = []
        for si in range(n_sessions):
            context = SEEDS_BASELINE[si % len(SEEDS_BASELINE)]

            if band_name == "no_kv":
                kv = None
            else:
                recursive_donor = KV_DONOR_PROMPTS[si % len(KV_DONOR_PROMPTS)]
                recursive_kv = extract_kv_cache(model, tokenizer, recursive_donor, device=device)
                if band_name == "full_kv":
                    kv = recursive_kv
                else:
                    kv = make_partial_kv_cache(recursive_kv, baseline_kv, start, end)

            turns = []
            for t in range(max_turns):
                t0 = time.time()
                if kv is not None:
                    resp = generate_with_kv(model, tokenizer, context, kv_cache=kv,
                                            max_tokens=150, min_new_tokens=24, device=device)
                else:
                    resp = generate_plain(model, tokenizer, context,
                                          max_tokens=150, min_new_tokens=24, device=device)

                rv, pr_e, pr_l = compute_rv_with_components(
                    model, tokenizer, resp, early, late, window=16, device=device)
                cl = classify_output(resp, rv)
                rep = repetition_score(resp)
                elapsed = time.time() - t0
                print(f"  {band_name} s{si} T{t:02d} [{cl:12s}] rv={rv:.3f} rep={rep:.2f} {elapsed:.1f}s")

                turns.append({
                    "turn": t, "response": resp[:400],
                    "rv": float(rv) if not np.isnan(rv) else None,
                    "classification": cl, "rep_score": float(rep),
                })
                follow = random.choice(BASELINE_CONTINUE)
                context = f"{context}\n{resp}\n{follow}"
                tokens = tokenizer.encode(context)
                if len(tokens) > 1800:
                    context = tokenizer.decode(tokens[-1500:])

            bt_art = sum(1 for t in turns if t["classification"] in ("BREAKTHROUGH", "ARTICULATE"))
            rvs = [t["rv"] for t in turns if t["rv"] is not None]
            sessions.append({
                "session_idx": si, "bt_art": bt_art, "bt_art_rate": bt_art/max_turns,
                "mean_rv": float(np.mean(rvs)) if rvs else None, "turns": turns,
            })

        total_bt = sum(s["bt_art"] for s in sessions)
        total_turns = n_sessions * max_turns
        all_rvs = []
        for s in sessions:
            all_rvs.extend([t["rv"] for t in s["turns"] if t["rv"] is not None])

        results[band_name] = {
            "band": [start, end],
            "n_sessions": n_sessions, "max_turns": max_turns,
            "total_bt_art": total_bt, "bt_art_rate": total_bt/total_turns,
            "mean_rv": float(np.mean(all_rvs)) if all_rvs else None,
            "std_rv": float(np.std(all_rvs)) if all_rvs else None,
            "per_session_bt_rates": [s["bt_art_rate"] for s in sessions],
            "sessions": sessions,
        }
        print(f"  {band_name}: BT+ART={total_bt}/{total_turns} ({total_bt/total_turns:.1%})")

    # Stats vs no_kv baseline
    print("\n--- KV ABLATION COMPARISONS (each band vs no_kv) ---")
    base = results["no_kv"]
    comparisons = {}
    for name in ["full_kv", "L0_7", "L8_15", "L16_23", "L24_31"]:
        test = results[name]
        table = [[test["total_bt_art"], n_sessions*max_turns - test["total_bt_art"]],
                 [base["total_bt_art"], n_sessions*max_turns - base["total_bt_art"]]]
        odds, p = stats.fisher_exact(table)
        d = cohens_d(test["per_session_bt_rates"], base["per_session_bt_rates"])
        direction = "UP" if test["bt_art_rate"] > base["bt_art_rate"] else "DOWN"
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  {name}: {test['bt_art_rate']:.1%} vs no_kv: {base['bt_art_rate']:.1%} "
              f"OR={odds:.2f} p={p:.6f} [{sig}] d={d:.2f} {direction}")
        comparisons[f"{name}_vs_no_kv"] = {
            "or": float(odds), "p": float(p), "d": d, "direction": direction,
            "test_rate": test["bt_art_rate"], "base_rate": base["bt_art_rate"],
        }

    return {"experiment": "kv_layer_ablation", "results": results, "comparisons": comparisons}


# ═══════════════════════════════════════════════════════════════════════════════
# EXP2: ALPHA SWEEP FOR DUAL PATCHING
# ═══════════════════════════════════════════════════════════════════════════════

class ScaledVPatcher:
    """Like PersistentVPatcher but blends at alpha: out = (1-alpha)*original + alpha*patched."""
    def __init__(self, model, v_activation, alpha=1.0):
        self.model = model
        if v_activation.dim() == 3:
            v_activation = v_activation[0]
        self.v_activation = v_activation.detach()
        self.alpha = alpha
        self.handle = None
        self.layer_idx = None

    def register(self, layer_idx):
        from src.core.hf_accessors import get_vproj_hookpoint, extract_v_from_hook_output
        self.layer_idx = layer_idx
        hookpoint = get_vproj_hookpoint(self.model, layer_idx=layer_idx)
        alpha = self.alpha
        v_act = self.v_activation

        def hook_fn(module, inp, out):
            window_size = 16
            if hookpoint.kind == "v_proj":
                batch, seq_len, hidden = out.shape
                v_len = min(seq_len, v_act.shape[0], window_size)
                v_slice = v_act[-v_len:, :].unsqueeze(0)
                if batch > 1:
                    v_slice = v_slice.repeat(batch, 1, 1)
                out_p = out.clone()
                orig = out_p[:, -v_len:, :]
                patched = v_slice[:, :v_len, :].to(out_p.device, dtype=out_p.dtype)
                out_p[:, -v_len:, :] = (1 - alpha) * orig + alpha * patched
                return out_p
            return out

        self.handle = hookpoint.module.register_forward_hook(hook_fn)

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class ScaledResidualPatcher:
    """Like PersistentResidualPatcher but blends at alpha."""
    def __init__(self, model, residual_activation, alpha=1.0):
        self.model = model
        if residual_activation.dim() == 3:
            residual_activation = residual_activation[0]
        self.residual_activation = residual_activation.detach()
        self.alpha = alpha
        self.handle = None
        self.layer_idx = None

    def register(self, layer_idx):
        from src.core.hf_accessors import get_layers
        layers = get_layers(self.model)
        target = layers[layer_idx]
        alpha = self.alpha
        r_act = self.residual_activation

        def hook_fn(module, args, kwargs=None):
            if isinstance(args, tuple) and len(args) > 0:
                hidden = args[0]
            else:
                return args
            window_size = 16
            batch, seq_len, hidden_dim = hidden.shape
            r_len = min(seq_len, r_act.shape[0], window_size)
            r_slice = r_act[-r_len:, :].unsqueeze(0)
            if batch > 1:
                r_slice = r_slice.repeat(batch, 1, 1)
            hidden_new = hidden.clone()
            orig = hidden_new[:, -r_len:, :]
            patched = r_slice[:, :r_len, :].to(hidden_new.device, dtype=hidden_new.dtype)
            hidden_new[:, -r_len:, :] = (1 - alpha) * orig + alpha * patched
            return (hidden_new,) + args[1:]

        self.handle = target.register_forward_pre_hook(hook_fn, with_kwargs=True)

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def run_alpha_sweep(model, tokenizer, early, late, device="cuda"):
    print("\n" + "="*70)
    print("EXP2: ALPHA SWEEP FOR DUAL PATCHING")
    print("="*70)

    v_layer = late  # 27
    r_layer = 18
    n_sessions = 6
    max_turns = 20
    alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    # Extract reference activations
    recursive_v = extract_averaged(model, tokenizer, DONOR_RECURSIVE, v_layer, "v", device)
    recursive_r = extract_averaged(model, tokenizer, DONOR_RECURSIVE, r_layer, "r", device)

    results = {}
    for alpha in alphas:
        cond = f"alpha_{alpha:.1f}"
        print(f"\n--- ALPHA = {alpha:.1f} ---")
        sessions = []
        for si in range(n_sessions):
            context = SEEDS_BASELINE[si % len(SEEDS_BASELINE)]

            if alpha > 0:
                vp = ScaledVPatcher(model, recursive_v, alpha=alpha)
                rp = ScaledResidualPatcher(model, recursive_r, alpha=alpha)
                vp.register(layer_idx=v_layer)
                rp.register(layer_idx=r_layer)
            else:
                vp = rp = None

            try:
                turns = []
                for t in range(max_turns):
                    t0 = time.time()
                    resp = generate_plain(model, tokenizer, context,
                                          max_tokens=150, min_new_tokens=24, device=device)
                    # Clean measurement
                    if vp and vp.handle:
                        vp.remove()
                    if rp and rp.handle:
                        rp.remove()

                    rv, _, _ = compute_rv_with_components(
                        model, tokenizer, resp, early, late, window=16, device=device)

                    if alpha > 0:
                        vp.register(layer_idx=v_layer)
                        rp.register(layer_idx=r_layer)

                    cl = classify_output(resp, rv)
                    rep = repetition_score(resp)
                    elapsed = time.time() - t0
                    print(f"  a={alpha:.1f} s{si} T{t:02d} [{cl:12s}] rv={rv:.3f} {elapsed:.1f}s")

                    turns.append({
                        "turn": t, "response": resp[:400],
                        "rv": float(rv) if not np.isnan(rv) else None,
                        "classification": cl,
                    })
                    follow = random.choice(BASELINE_CONTINUE)
                    context = f"{context}\n{resp}\n{follow}"
                    tokens = tokenizer.encode(context)
                    if len(tokens) > 1800:
                        context = tokenizer.decode(tokens[-1500:])
            finally:
                if vp:
                    vp.remove()
                if rp:
                    rp.remove()

            bt_art = sum(1 for t in turns if t["classification"] in ("BREAKTHROUGH", "ARTICULATE"))
            rvs = [t["rv"] for t in turns if t["rv"] is not None]
            sessions.append({
                "session_idx": si, "bt_art": bt_art, "bt_art_rate": bt_art/max_turns,
                "mean_rv": float(np.mean(rvs)) if rvs else None, "turns": turns,
            })

        total_bt = sum(s["bt_art"] for s in sessions)
        total_turns = n_sessions * max_turns
        all_rvs = []
        for s in sessions:
            all_rvs.extend([t["rv"] for t in s["turns"] if t["rv"] is not None])

        results[cond] = {
            "alpha": alpha, "n_sessions": n_sessions, "max_turns": max_turns,
            "total_bt_art": total_bt, "bt_art_rate": total_bt/total_turns,
            "mean_rv": float(np.mean(all_rvs)) if all_rvs else None,
            "std_rv": float(np.std(all_rvs)) if all_rvs else None,
            "per_session_bt_rates": [s["bt_art_rate"] for s in sessions],
            "sessions": sessions,
        }
        print(f"  alpha={alpha:.1f}: BT+ART={total_bt}/{total_turns} ({total_bt/total_turns:.1%}) "
              f"R_V={np.mean(all_rvs):.4f}" if all_rvs else "")

    # Print sweep summary
    print("\n--- ALPHA SWEEP SUMMARY ---")
    print(f"  {'Alpha':>6s}  {'BT+ART':>8s}  {'R_V':>8s}")
    for alpha in alphas:
        cond = f"alpha_{alpha:.1f}"
        r = results[cond]
        rv_str = f"{r['mean_rv']:.4f}" if r['mean_rv'] else "N/A"
        print(f"  {alpha:6.1f}  {r['bt_art_rate']:8.1%}  {rv_str:>8s}")

    return {"experiment": "alpha_sweep", "results": results}


# ═══════════════════════════════════════════════════════════════════════════════
# EXP3: ATTENTION PATTERN ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def extract_attention_patterns(model, tokenizer, prompt, target_layers, device="cuda"):
    """Extract attention weights at specified layers."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True, use_cache=False)
    attentions = outputs.attentions  # tuple of (batch, n_heads, seq_len, seq_len)

    result = {}
    for layer_idx in target_layers:
        attn = attentions[layer_idx][0]  # (n_heads, seq_len, seq_len)
        result[layer_idx] = attn.cpu().float().numpy()
    return result, inputs["input_ids"].shape[1]


def attention_entropy(attn_matrix):
    """Compute per-head entropy of attention distribution."""
    # attn_matrix: (n_heads, seq_len, seq_len)
    n_heads, seq_len, _ = attn_matrix.shape
    entropies = []
    for h in range(n_heads):
        # Average entropy across query positions
        probs = attn_matrix[h]  # (seq_len, seq_len)
        # Clamp to avoid log(0)
        probs = np.clip(probs, 1e-10, 1.0)
        ent = -np.sum(probs * np.log2(probs), axis=-1)  # (seq_len,)
        entropies.append(float(np.mean(ent)))
    return entropies


def self_attention_ratio(attn_matrix):
    """Compute ratio of attention to self (diagonal) for each head."""
    n_heads, seq_len, _ = attn_matrix.shape
    ratios = []
    for h in range(n_heads):
        diag_attn = np.diag(attn_matrix[h])  # (seq_len,)
        ratios.append(float(np.mean(diag_attn)))
    return ratios


def last_token_concentration(attn_matrix):
    """How much does the last token's attention concentrate on specific positions?"""
    n_heads, seq_len, _ = attn_matrix.shape
    concentrations = []
    for h in range(n_heads):
        last_row = attn_matrix[h, -1, :]  # last token's attention distribution
        # Concentration = max attention weight
        concentrations.append(float(np.max(last_row)))
    return concentrations


def run_attention_analysis(model, tokenizer, device="cuda"):
    print("\n" + "="*70)
    print("EXP3: ATTENTION PATTERN ANALYSIS")
    print("="*70)

    target_layers = [18, 27]
    n_prompts = min(20, len(SEEDS_RECURSIVE) + len(SEEDS_BASELINE))

    # Use more diverse prompts
    recursive_prompts = SEEDS_RECURSIVE + DONOR_RECURSIVE
    baseline_prompts = SEEDS_BASELINE + DONOR_BASELINE

    results = {"recursive": [], "baseline": []}

    for label, prompts in [("recursive", recursive_prompts), ("baseline", baseline_prompts)]:
        print(f"\n  Processing {label} prompts ({len(prompts)})...")
        for i, prompt in enumerate(prompts):
            attns, seq_len = extract_attention_patterns(model, tokenizer, prompt, target_layers, device)

            prompt_result = {"prompt_idx": i, "prompt": prompt[:200], "seq_len": seq_len}
            for layer_idx in target_layers:
                attn = attns[layer_idx]
                ent = attention_entropy(attn)
                self_attn = self_attention_ratio(attn)
                concentration = last_token_concentration(attn)

                prompt_result[f"L{layer_idx}_entropy"] = ent
                prompt_result[f"L{layer_idx}_self_attn"] = self_attn
                prompt_result[f"L{layer_idx}_concentration"] = concentration
                prompt_result[f"L{layer_idx}_mean_entropy"] = float(np.mean(ent))
                prompt_result[f"L{layer_idx}_mean_self_attn"] = float(np.mean(self_attn))
                prompt_result[f"L{layer_idx}_mean_concentration"] = float(np.mean(concentration))

            results[label].append(prompt_result)
            print(f"    {label}[{i}]: L18 ent={prompt_result['L18_mean_entropy']:.3f} "
                  f"self={prompt_result['L18_mean_self_attn']:.4f} | "
                  f"L27 ent={prompt_result['L27_mean_entropy']:.3f} "
                  f"self={prompt_result['L27_mean_self_attn']:.4f}")

    # Statistical comparisons
    print("\n--- ATTENTION PATTERN COMPARISONS ---")
    comparisons = {}
    for layer_idx in target_layers:
        for metric in ["mean_entropy", "mean_self_attn", "mean_concentration"]:
            key = f"L{layer_idx}_{metric}"
            rec_vals = [r[key] for r in results["recursive"]]
            base_vals = [r[key] for r in results["baseline"]]
            t_stat, p_val = stats.ttest_ind(rec_vals, base_vals)
            d = cohens_d(rec_vals, base_vals)
            print(f"  {key}: recursive={np.mean(rec_vals):.4f} vs baseline={np.mean(base_vals):.4f} "
                  f"d={d:.3f} p={p_val:.6f}")
            comparisons[key] = {
                "recursive_mean": float(np.mean(rec_vals)),
                "baseline_mean": float(np.mean(base_vals)),
                "cohens_d": d, "t_stat": float(t_stat), "p_value": float(p_val),
            }

    # Per-head analysis: which heads differ most?
    print("\n--- PER-HEAD ANALYSIS (top divergent heads) ---")
    head_divergence = {}
    n_heads = model.config.num_attention_heads
    for layer_idx in target_layers:
        for h in range(n_heads):
            rec_ent = [r[f"L{layer_idx}_entropy"][h] for r in results["recursive"]]
            base_ent = [r[f"L{layer_idx}_entropy"][h] for r in results["baseline"]]
            t_stat, p_val = stats.ttest_ind(rec_ent, base_ent)
            d = cohens_d(rec_ent, base_ent)
            head_divergence[f"L{layer_idx}_H{h}"] = {
                "layer": layer_idx, "head": h,
                "rec_mean": float(np.mean(rec_ent)),
                "base_mean": float(np.mean(base_ent)),
                "d": d, "p": float(p_val),
            }

    # Sort by absolute effect size
    sorted_heads = sorted(head_divergence.items(), key=lambda x: abs(x[1]["d"]), reverse=True)
    for name, info in sorted_heads[:10]:
        sig = "***" if info["p"] < 0.001 else "**" if info["p"] < 0.01 else "*" if info["p"] < 0.05 else "ns"
        print(f"  {name}: rec={info['rec_mean']:.3f} base={info['base_mean']:.3f} "
              f"d={info['d']:.3f} p={info['p']:.6f} [{sig}]")

    return {
        "experiment": "attention_analysis",
        "target_layers": target_layers,
        "n_recursive": len(results["recursive"]),
        "n_baseline": len(results["baseline"]),
        "comparisons": comparisons,
        "head_divergence": dict(sorted_heads[:20]),
        "raw": results,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Run exploratory Mistral hardening battery.")
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed = args.seed
    seed_everything(seed)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    print(f"Device: {device}")
    print(f"Seed: {seed}")
    print(f"Timestamp: {datetime.now().isoformat()}")

    model_name = args.model
    print(f"\nLoading {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto",
        attn_implementation="eager",
    )
    model.eval()

    num_layers = model.config.num_hidden_layers
    early = 5
    late = num_layers - 5  # 27
    print(f"Layers: {num_layers}, early={early}, late={late}")

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "seed": seed,
        "early": early, "late": late,
    }

    # ── EXP3 first (fastest, no generation) ──
    t0 = time.time()
    attn_results = run_attention_analysis(model, tokenizer, device)
    all_results["attention_analysis"] = attn_results
    print(f"\nEXP3 done in {time.time()-t0:.0f}s")

    # ── EXP2: Alpha sweep ──
    t0 = time.time()
    alpha_results = run_alpha_sweep(model, tokenizer, early, late, device)
    all_results["alpha_sweep"] = alpha_results
    print(f"\nEXP2 done in {time.time()-t0:.0f}s")

    # ── EXP1: KV ablation (most expensive) ──
    t0 = time.time()
    kv_results = run_kv_ablation(model, tokenizer, early, late, device)
    all_results["kv_layer_ablation"] = kv_results
    print(f"\nEXP1 done in {time.time()-t0:.0f}s")

    # ── Save ──
    outdir = Path("results/hardening_battery")
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / f"hardening_battery_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(outfile, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll results saved to {outfile}")

    # ── Final summary ──
    print("\n" + "="*70)
    print("HARDENING BATTERY COMPLETE")
    print("="*70)

    if "kv_layer_ablation" in all_results:
        print("\nKV ABLATION SUMMARY:")
        for name, info in all_results["kv_layer_ablation"]["results"].items():
            rv_str = f"{info['mean_rv']:.4f}" if info['mean_rv'] else "N/A"
            print(f"  {name:12s}: BT+ART={info['bt_art_rate']:.1%}  R_V={rv_str}")

    if "alpha_sweep" in all_results:
        print("\nALPHA SWEEP SUMMARY:")
        for name, info in all_results["alpha_sweep"]["results"].items():
            rv_str = f"{info['mean_rv']:.4f}" if info['mean_rv'] else "N/A"
            print(f"  {name:12s}: BT+ART={info['bt_art_rate']:.1%}  R_V={rv_str}")

    if "attention_analysis" in all_results:
        print("\nATTENTION ANALYSIS TOP HEADS:")
        for name, info in list(all_results["attention_analysis"]["head_divergence"].items())[:5]:
            print(f"  {name}: d={info['d']:.3f} p={info['p']:.6f}")


if __name__ == "__main__":
    main()
