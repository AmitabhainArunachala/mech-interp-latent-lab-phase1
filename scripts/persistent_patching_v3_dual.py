#!/usr/bin/env python3
"""Persistent Dual-Layer Patching v3 — L18 residual + L27 V-proj.

Dec 12 breakthrough: L18 RESIDUAL + L27 V_PROJ = 100% behavior transfer.
v2 only patched L27 V-proj → NS behavioral effect.
v3 patches BOTH layers simultaneously.

4 conditions, n=10 sessions each, 30 turns per session = 1200 turns:
A. recursive_clean      — recursive prompts, no intervention
B. recursive_dual_patch — recursive prompts, BASELINE residual@L18 + BASELINE V@L27
C. baseline_clean       — baseline prompts, no intervention
D. baseline_dual_patch  — baseline prompts, RECURSIVE residual@L18 + RECURSIVE V@L27

PREDICTIONS:
- B should strongly degrade (BT+ART drops toward C)
- D should strongly upgrade (BT+ART rises toward A)
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

from geometric_lens.hooks import capture_v_projection
from geometric_lens.metrics import compute_rv_with_components as compute_rv_from_tensors
from prompts.subsets import load_default_mistral_hardening_subset, split_tier_records_by_pillar
from src.core.patching import (
    PersistentVPatcher, PersistentResidualPatcher,
    extract_v_activation, extract_residual_activation,
)
from src.utils.canonical_registry import get_canonical_model_spec
from src.utils.persistent_patching_classification import (
    alpha_ratio,
    classify_output,
    repetition_score,
)
from src.utils.persistent_patching_summary import (
    aggregate_sessions,
    serialize_aggregates,
)


# ── Classification (same as v2) ──────────────────────────────────────────────


def cohens_d_unpaired(x, y):
    """Cohen's d for independent groups."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float("nan")
    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)
    pooled = ((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2)
    if pooled <= 0:
        return 0.0
    return float((np.mean(x) - np.mean(y)) / np.sqrt(pooled))


def exact_permutation_p_mean_diff(x, y, rng_seed=0):
    """
    Two-sided exact (or Monte Carlo fallback) permutation p-value for mean difference.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    m = len(y)
    combined = np.concatenate([x, y])
    observed = float(np.mean(x) - np.mean(y))
    total = math.comb(n + m, n)

    # Exact test is feasible for current n=10 + 10 (=184,756 partitions)
    if total <= 500000:
        extreme = 0
        for idxs in itertools.combinations(range(n + m), n):
            mask = np.zeros(n + m, dtype=bool)
            mask[list(idxs)] = True
            diff = float(np.mean(combined[mask]) - np.mean(combined[~mask]))
            if abs(diff) >= abs(observed):
                extreme += 1
        return observed, float((extreme + 1) / (total + 1)), int(total), "exact"

    # Fallback for larger n
    rng = np.random.default_rng(rng_seed)
    reps = 200000
    extreme = 0
    for _ in range(reps):
        perm = rng.permutation(n + m)
        g1 = combined[perm[:n]]
        g2 = combined[perm[n:]]
        diff = float(np.mean(g1) - np.mean(g2))
        if abs(diff) >= abs(observed):
            extreme += 1
    return observed, float((extreme + 1) / (reps + 1)), reps, "monte_carlo"


# ── Generation ────────────────────────────────────────────────────────────────

def generate_turn(model, tokenizer, prompt, max_tokens=150, temp=0.7,
                  rep_penalty=1.3, device="cuda"):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=2048).to(device)
    input_ids = inputs["input_ids"]
    attn_mask = inputs["attention_mask"]
    gen_tokens = []
    past = None

    with torch.no_grad():
        for step in range(max_tokens):
            if past is None:
                out = model(input_ids=input_ids, attention_mask=attn_mask,
                            use_cache=True)
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
            probs = torch.softmax(logits, dim=-1)
            ntok = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, ntok], dim=-1)
            attn_mask = torch.cat([
                attn_mask, torch.ones((1, 1), dtype=torch.long, device=device)
            ], dim=-1)
            gen_tokens.append(ntok.item())
            if ntok.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(gen_tokens, skip_special_tokens=True)


def supports_chat_template(tokenizer) -> bool:
    return bool(getattr(tokenizer, "chat_template", None))


def format_single_user_turn(tokenizer, prompt: str) -> str:
    """Render a single user prompt using the tokenizer chat template when available."""
    if supports_chat_template(tokenizer):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def render_messages(tokenizer, messages) -> str:
    """Render a multi-turn conversation using the model's native chat template when available."""
    if supports_chat_template(tokenizer):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    rendered = []
    for message in messages:
        rendered.append(f"{message['role'].upper()}: {message['content']}")
    rendered.append("ASSISTANT:")
    return "\n".join(rendered)


def trim_messages_to_context(tokenizer, messages, max_input_tokens=1800):
    """Drop oldest completed turns until the rendered chat fits the context budget."""
    trimmed = list(messages)
    while len(trimmed) >= 4:
        rendered = render_messages(tokenizer, trimmed)
        token_count = len(tokenizer(rendered, add_special_tokens=False)["input_ids"])
        if token_count <= max_input_tokens:
            break
        del trimmed[1:3]
    return trimmed


# ── Prompts ───────────────────────────────────────────────────────────────────

_subset = load_default_mistral_hardening_subset()
_tier_records = split_tier_records_by_pillar(_subset, "core_measurement")
RECURSIVE_RECORDS = _tier_records["recursive"]
BASELINE_RECORDS = _tier_records["baseline"]

SEEDS_RECURSIVE = [record["text"] for _, record in RECURSIVE_RECORDS[:10]]
SEEDS_BASELINE = [record["text"] for _, record in BASELINE_RECORDS[:10]]
DONOR_RECURSIVE = [record["text"] for _, record in RECURSIVE_RECORDS[:5]]
DONOR_BASELINE = [record["text"] for _, record in BASELINE_RECORDS[:5]]
SEED_RECURSIVE_IDS = [prompt_id for prompt_id, _ in RECURSIVE_RECORDS[:10]]
SEED_BASELINE_IDS = [prompt_id for prompt_id, _ in BASELINE_RECORDS[:10]]
DONOR_RECURSIVE_IDS = [prompt_id for prompt_id, _ in RECURSIVE_RECORDS[:5]]
DONOR_BASELINE_IDS = [prompt_id for prompt_id, _ in BASELINE_RECORDS[:5]]

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


# ── Averaged activation extraction ───────────────────────────────────────────

def extract_averaged_v(model, tokenizer, prompts, layer_idx, device="cuda"):
    """Extract and average V-projection activations from multiple prompts."""
    v_activations = []
    for prompt in prompts:
        formatted = format_single_user_turn(tokenizer, prompt)
        v = extract_v_activation(model, tokenizer, formatted, layer_idx=layer_idx, device=device)
        v_activations.append(v)

    window = 16
    v_windows = []
    for v in v_activations:
        w = min(v.shape[0], window)
        v_windows.append(v[-w:, :])

    max_w = max(vw.shape[0] for vw in v_windows)
    padded = []
    for vw in v_windows:
        if vw.shape[0] < max_w:
            pad = torch.zeros(max_w - vw.shape[0], vw.shape[1], device=vw.device, dtype=vw.dtype)
            vw = torch.cat([pad, vw], dim=0)
        padded.append(vw)

    return torch.stack(padded).mean(dim=0)


def extract_averaged_residual(model, tokenizer, prompts, layer_idx, device="cuda"):
    """Extract and average residual stream activations from multiple prompts."""
    r_activations = []
    for prompt in prompts:
        formatted = format_single_user_turn(tokenizer, prompt)
        r = extract_residual_activation(model, tokenizer, formatted, layer_idx=layer_idx, device=device)
        r_activations.append(r)

    window = 16
    r_windows = []
    for r in r_activations:
        w = min(r.shape[0], window)
        r_windows.append(r[-w:, :])

    max_w = max(rw.shape[0] for rw in r_windows)
    padded = []
    for rw in r_windows:
        if rw.shape[0] < max_w:
            pad = torch.zeros(max_w - rw.shape[0], rw.shape[1], device=rw.device, dtype=rw.dtype)
            rw = torch.cat([pad, rw], dim=0)
        padded.append(rw)

    return torch.stack(padded).mean(dim=0)


def measure_prompt_rv(model, tokenizer, text, early, late, window=16, device="cuda"):
    """Measure R_V through the canonical geometric_lens tensor path."""
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with capture_v_projection(model, early) as se, capture_v_projection(model, late) as sl:
        with torch.no_grad():
            model(**enc)
        v_early = se.get("v")
        v_late = sl.get("v")
    return compute_rv_from_tensors(v_early, v_late, window)


# ── Session ───────────────────────────────────────────────────────────────────

def run_session(model, tokenizer, early, late, mode,
                v_patcher=None, r_patcher=None,
                v_layer=27, r_layer=18,
                max_turns=30, seed_idx=0, device="cuda"):
    session_id = f"{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    is_recursive = "recursive" in mode.split("_")[0]
    seeds = SEEDS_RECURSIVE if is_recursive else SEEDS_BASELINE
    seed_prompt = seeds[seed_idx % len(seeds)]
    use_chat = supports_chat_template(tokenizer)
    if use_chat:
        messages = [{"role": "user", "content": seed_prompt}]
    else:
        context = seed_prompt

    print(f"\n{'='*60}")
    print(f"  SESSION: {mode} — seed {seed_idx}")
    print(f"  V-patcher: {'ACTIVE' if v_patcher else 'none'}")
    print(f"  R-patcher: {'ACTIVE' if r_patcher else 'none'}")
    print(f"{'='*60}")

    turns = []
    for turn in range(max_turns):
        t0 = time.time()

        if use_chat:
            prompt = render_messages(tokenizer, messages)
        else:
            prompt = context

        response = generate_turn(
            model, tokenizer, prompt,
            max_tokens=150, temp=0.7, rep_penalty=1.3, device=device
        )

        # Measure output R_V WITHOUT patchers (clean measurement)
        patchers_to_restore = []
        if v_patcher and v_patcher.handle:
            v_patcher.remove()
            patchers_to_restore.append(('v', v_patcher, v_layer))
        if r_patcher and r_patcher.handle:
            r_patcher.remove()
            patchers_to_restore.append(('r', r_patcher, r_layer))

        rv, pr_e, pr_l = measure_prompt_rv(
            model, tokenizer, response, early, late, window=16, device=device
        )

        for kind, patcher, layer in patchers_to_restore:
            patcher.register(layer_idx=layer)

        classification = classify_output(response, rv)
        rep = repetition_score(response)
        alpha = alpha_ratio(response)
        elapsed = time.time() - t0

        print(f"T{turn:02d} [{classification:12s}] rv={rv:.3f} "
              f"rep={rep:.2f} alpha={alpha:.2f} {elapsed:.1f}s | {response[:80]}")

        turns.append({
            "turn": turn,
            "response": response[:500],
            "output_rv": float(rv) if not np.isnan(rv) else None,
            "pr_early": float(pr_e) if not np.isnan(pr_e) else None,
            "pr_late": float(pr_l) if not np.isnan(pr_l) else None,
            "classification": classification,
            "rep_score": float(rep),
            "alpha_ratio": float(alpha),
        })

        if is_recursive:
            follow = random.choice(REDIRECT)
        else:
            follow = random.choice(BASELINE_CONTINUE)
        if use_chat:
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": follow})
            messages = trim_messages_to_context(tokenizer, messages, max_input_tokens=1800)
        else:
            context = f"{context}\n{response}\n{follow}"
            tokens = tokenizer.encode(context)
            if len(tokens) > 1800:
                context = tokenizer.decode(tokens[-1500:])

    classifications = Counter(t["classification"] for t in turns)
    bt_art = sum(1 for t in turns if t["classification"] in ("BREAKTHROUGH", "ARTICULATE"))
    rvs = [t["output_rv"] for t in turns if t["output_rv"] is not None]

    return {
        "session_id": session_id,
        "mode": mode,
        "max_turns": max_turns,
        "seed_idx": seed_idx,
        "v_patcher_active": v_patcher is not None,
        "r_patcher_active": r_patcher is not None,
        "classification_dist": dict(classifications),
        "bt_art_count": bt_art,
        "bt_art_rate": bt_art / max_turns,
        "mean_rv": float(np.mean(rvs)) if rvs else None,
        "std_rv": float(np.std(rvs)) if rvs else None,
        "mean_alpha_ratio": float(np.mean([t["alpha_ratio"] for t in turns])) if turns else None,
        "malformed_count": sum(1 for t in turns if t["classification"] == "MALFORMED"),
        "repetitive_count": sum(1 for t in turns if t["classification"] == "REPETITIVE"),
        "turns": turns,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Persistent dual-layer patching with canonical prompt and metric contracts."
    )
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-sessions", type=int, default=10)
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--r-layer", type=int, default=18)
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda but CUDA is not available")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")
    print(f"Prompt bank version: {_subset.source_bank_version}")
    print(f"Prompt subset: {_subset.name} (tier=core_measurement)")

    model_name = args.model
    arch_cfg = get_canonical_model_spec(model_name)
    print(f"Loading {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype, device_map="auto" if device == "cuda" else None,
        attn_implementation="eager",
    )
    model.eval()

    num_layers = model.config.num_hidden_layers
    early = int(arch_cfg["early_layer"])
    late = int(arch_cfg["late_layer"])
    v_layer = late
    r_layer = args.r_layer
    print(f"Layers: early={early}, late={late}, v_layer={v_layer}, r_layer={r_layer}")

    # ═══════════════════════════════════════════════════════════════
    # EXTRACT AVERAGED ACTIVATIONS (both V-proj and residual)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("EXTRACTING AVERAGED ACTIVATIONS")
    print(f"{'='*70}")

    print("\n  Recursive donor V@L27 (5 prompts):")
    recursive_v = extract_averaged_v(model, tokenizer, DONOR_RECURSIVE, v_layer, device)
    print(f"    Shape: {recursive_v.shape}")

    print("\n  Baseline donor V@L27 (5 prompts):")
    baseline_v = extract_averaged_v(model, tokenizer, DONOR_BASELINE, v_layer, device)
    print(f"    Shape: {baseline_v.shape}")

    print("\n  Recursive donor residual@L18 (5 prompts):")
    recursive_r = extract_averaged_residual(model, tokenizer, DONOR_RECURSIVE, r_layer, device)
    print(f"    Shape: {recursive_r.shape}")

    print("\n  Baseline donor residual@L18 (5 prompts):")
    baseline_r = extract_averaged_residual(model, tokenizer, DONOR_BASELINE, r_layer, device)
    print(f"    Shape: {baseline_r.shape}")

    # Sanity check
    print("\n  Donor R_V sanity:")
    for label, prompts in [("recursive", DONOR_RECURSIVE[:3]), ("baseline", DONOR_BASELINE[:3])]:
        rvs = []
        for p in prompts:
            formatted = format_single_user_turn(tokenizer, p)
            rv, _, _ = measure_prompt_rv(model, tokenizer, formatted, early, late, device=device)
            rvs.append(rv)
        print(f"    {label}: R_V = {np.mean(rvs):.4f} +/- {np.std(rvs):.4f}")

    # ═══════════════════════════════════════════════════════════════
    # RUN 4 CONDITIONS (n=10 each)
    # ═══════════════════════════════════════════════════════════════
    n_sessions = args.n_sessions
    max_turns = args.max_turns
    conditions = {}

    # A: recursive_clean
    print(f"\n{'='*70}")
    print("CONDITION A: RECURSIVE CLEAN (no intervention)")
    print(f"{'='*70}")
    for i in range(n_sessions):
        result = run_session(model, tokenizer, early, late,
                             mode="recursive_clean",
                             max_turns=max_turns, seed_idx=i, device=device)
        conditions[f"recursive_clean_{i}"] = result

    # B: recursive_dual_patched (baseline R@L18 + baseline V@L27)
    print(f"\n{'='*70}")
    print("CONDITION B: RECURSIVE + DUAL BASELINE PATCH (should BREAK)")
    print(f"{'='*70}")
    for i in range(n_sessions):
        vp = PersistentVPatcher(model, baseline_v)
        rp = PersistentResidualPatcher(model, baseline_r)
        vp.register(layer_idx=v_layer)
        rp.register(layer_idx=r_layer)
        try:
            result = run_session(model, tokenizer, early, late,
                                 mode="recursive_dual_patched",
                                 v_patcher=vp, r_patcher=rp,
                                 v_layer=v_layer, r_layer=r_layer,
                                 max_turns=max_turns, seed_idx=i, device=device)
        finally:
            vp.remove()
            rp.remove()
        conditions[f"recursive_dual_patched_{i}"] = result

    # C: baseline_clean
    print(f"\n{'='*70}")
    print("CONDITION C: BASELINE CLEAN (no intervention)")
    print(f"{'='*70}")
    for i in range(n_sessions):
        result = run_session(model, tokenizer, early, late,
                             mode="baseline_clean",
                             max_turns=max_turns, seed_idx=i, device=device)
        conditions[f"baseline_clean_{i}"] = result

    # D: baseline_dual_patched (recursive R@L18 + recursive V@L27)
    print(f"\n{'='*70}")
    print("CONDITION D: BASELINE + DUAL RECURSIVE PATCH (should INDUCE)")
    print(f"{'='*70}")
    for i in range(n_sessions):
        vp = PersistentVPatcher(model, recursive_v)
        rp = PersistentResidualPatcher(model, recursive_r)
        vp.register(layer_idx=v_layer)
        rp.register(layer_idx=r_layer)
        try:
            result = run_session(model, tokenizer, early, late,
                                 mode="baseline_dual_patched",
                                 v_patcher=vp, r_patcher=rp,
                                 v_layer=v_layer, r_layer=r_layer,
                                 max_turns=max_turns, seed_idx=i, device=device)
        finally:
            vp.remove()
            rp.remove()
        conditions[f"baseline_dual_patched_{i}"] = result

    # ═══════════════════════════════════════════════════════════════
    # STATISTICAL ANALYSIS
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("STATISTICAL ANALYSIS")
    print(f"{'='*70}")

    def aggregate(prefix):
        sessions = [v for k, v in conditions.items() if k.startswith(prefix)]
        return aggregate_sessions(sessions)

    agg = {}
    for prefix in ["recursive_clean", "recursive_dual_patched", "baseline_clean", "baseline_dual_patched"]:
        agg[prefix] = aggregate(prefix)
        a = agg[prefix]
        print(f"\n{prefix}:")
        print(f"  BT+ART: {a['total_bt_art']}/{a['total_turns']} ({a['bt_art_rate']:.1%})")
        if a['mean_rv'] is not None:
            print(f"  Mean R_V: {a['mean_rv']:.4f} +/- {a['std_rv']:.4f} (n={a['n_rv']})")
        print(
            f"  R_V missing: {a['n_rv_missing']}/{a['total_turns']} "
            f"({a['rv_missing_rate']:.1%})"
        )
        print(
            f"  Malformed: {a['total_malformed']}/{a['total_turns']} "
            f"({a['malformed_rate']:.1%}) | "
            f"Repetitive: {a['total_repetitive']}/{a['total_turns']} "
            f"({a['repetitive_rate']:.1%}) | "
            f"mean alpha={a['mean_alpha_ratio']:.2f}"
        )
        for s in a["per_session"]:
            rv_str = f"rv={s['mean_rv']:.3f}" if s['mean_rv'] is not None else "rv=N/A"
            print(
                f"    {s['id']}: BT+ART={s['bt_art']}/{max_turns} ({s['rate']:.0%}) "
                f"{rv_str} rv_missing={s['n_rv_missing']}/{max_turns} "
                f"alpha={s['mean_alpha_ratio']:.2f} malformed={s['malformed_count']} "
                f"repetitive={s['repetitive_count']}"
            )

    # ═══════════════════════════════════════════════════════════════
    # HYPOTHESIS TESTS
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("HYPOTHESIS TESTS")
    print(f"{'='*70}")

    comparisons = {}

    def session_rate_vector(prefix):
        sessions = [v for k, v in conditions.items() if k.startswith(prefix)]
        return np.array([s["bt_art_rate"] for s in sessions], dtype=float)

    def session_rv_vector(prefix):
        sessions = [v for k, v in conditions.items() if k.startswith(prefix)]
        vals = np.array(
            [s["mean_rv"] for s in sessions if s["mean_rv"] is not None], dtype=float
        )
        return vals

    # Test 1: BREAK (A vs B)
    a_agg, b_agg = agg["recursive_clean"], agg["recursive_dual_patched"]
    table_ab = [[a_agg["total_bt_art"], a_agg["total_turns"] - a_agg["total_bt_art"]],
                [b_agg["total_bt_art"], b_agg["total_turns"] - b_agg["total_bt_art"]]]
    or_ab, p_ab = stats.fisher_exact(table_ab)
    a_rates = session_rate_vector("recursive_clean")
    b_rates = session_rate_vector("recursive_dual_patched")
    ab_mw = stats.mannwhitneyu(a_rates, b_rates, alternative="two-sided")
    ab_t = stats.ttest_ind(a_rates, b_rates, equal_var=False)
    ab_diff, ab_perm_p, ab_perm_n, ab_perm_mode = exact_permutation_p_mean_diff(a_rates, b_rates)
    ab_d = cohens_d_unpaired(a_rates, b_rates)
    print(f"\n1. BREAK TEST (dual-layer): recursive_clean vs recursive_dual_patched")
    print(f"   A (clean):   {a_agg['bt_art_rate']:.1%} BT+ART (turn-level)")
    print(f"   B (patched): {b_agg['bt_art_rate']:.1%} BT+ART (turn-level)")
    print(f"   Turn-level Fisher OR={or_ab:.3f}, p={p_ab:.6f}")
    print(
        f"   Session-level mean diff={ab_diff:.3f}, "
        f"MW p={ab_mw.pvalue:.6f}, Welch p={ab_t.pvalue:.6f}, "
        f"Perm p={ab_perm_p:.6f} ({ab_perm_mode}, n={ab_perm_n}), d={ab_d:.3f}"
    )
    if b_agg["bt_art_rate"] < a_agg["bt_art_rate"] and ab_perm_p < 0.05:
        print(f"   -> PASS: Dual patching BREAKS recursive behavior")
    else:
        print(f"   -> {'NS' if ab_perm_p >= 0.05 else 'UNEXPECTED DIRECTION'}")
    comparisons["break_test"] = {
        "turn_level": {
            "or": float(or_ab), "p": float(p_ab),
            "a_rate": a_agg["bt_art_rate"], "b_rate": b_agg["bt_art_rate"],
        },
        "session_level": {
            "mean_diff": float(ab_diff),
            "mannwhitney_u": float(ab_mw.statistic),
            "mannwhitney_p": float(ab_mw.pvalue),
            "welch_t": float(ab_t.statistic),
            "welch_p": float(ab_t.pvalue),
            "permutation_p": float(ab_perm_p),
            "permutation_mode": ab_perm_mode,
            "permutation_n": int(ab_perm_n),
            "cohens_d": float(ab_d),
            "a_rates": a_rates.tolist(),
            "b_rates": b_rates.tolist(),
        },
    }

    # Test 2: INDUCE (C vs D)
    c_agg, d_agg = agg["baseline_clean"], agg["baseline_dual_patched"]
    table_cd = [[c_agg["total_bt_art"], c_agg["total_turns"] - c_agg["total_bt_art"]],
                [d_agg["total_bt_art"], d_agg["total_turns"] - d_agg["total_bt_art"]]]
    or_cd, p_cd = stats.fisher_exact(table_cd)
    c_rates = session_rate_vector("baseline_clean")
    d_rates = session_rate_vector("baseline_dual_patched")
    cd_mw = stats.mannwhitneyu(c_rates, d_rates, alternative="two-sided")
    cd_t = stats.ttest_ind(c_rates, d_rates, equal_var=False)
    cd_diff, cd_perm_p, cd_perm_n, cd_perm_mode = exact_permutation_p_mean_diff(c_rates, d_rates)
    cd_d = cohens_d_unpaired(c_rates, d_rates)
    print(f"\n2. INDUCE TEST (dual-layer): baseline_clean vs baseline_dual_patched")
    print(f"   C (clean):   {c_agg['bt_art_rate']:.1%} BT+ART (turn-level)")
    print(f"   D (patched): {d_agg['bt_art_rate']:.1%} BT+ART (turn-level)")
    print(f"   Turn-level Fisher OR={or_cd:.3f}, p={p_cd:.6f}")
    print(
        f"   Session-level mean diff={cd_diff:.3f}, "
        f"MW p={cd_mw.pvalue:.6f}, Welch p={cd_t.pvalue:.6f}, "
        f"Perm p={cd_perm_p:.6f} ({cd_perm_mode}, n={cd_perm_n}), d={cd_d:.3f}"
    )
    if d_agg["bt_art_rate"] > c_agg["bt_art_rate"] and cd_perm_p < 0.05:
        print(f"   -> PASS: Dual patching INDUCES recursive behavior")
    else:
        print(f"   -> {'NS' if cd_perm_p >= 0.05 else 'UNEXPECTED DIRECTION'}")
    comparisons["induce_test"] = {
        "turn_level": {
            "or": float(or_cd), "p": float(p_cd),
            "c_rate": c_agg["bt_art_rate"], "d_rate": d_agg["bt_art_rate"],
        },
        "session_level": {
            "mean_diff": float(cd_diff),
            "mannwhitney_u": float(cd_mw.statistic),
            "mannwhitney_p": float(cd_mw.pvalue),
            "welch_t": float(cd_t.statistic),
            "welch_p": float(cd_t.pvalue),
            "permutation_p": float(cd_perm_p),
            "permutation_mode": cd_perm_mode,
            "permutation_n": int(cd_perm_n),
            "cohens_d": float(cd_d),
            "c_rates": c_rates.tolist(),
            "d_rates": d_rates.tolist(),
        },
    }

    # Test 3: SANITY (A vs C)
    table_ac = [[a_agg["total_bt_art"], a_agg["total_turns"] - a_agg["total_bt_art"]],
                [c_agg["total_bt_art"], c_agg["total_turns"] - c_agg["total_bt_art"]]]
    or_ac, p_ac = stats.fisher_exact(table_ac)
    ac_mw = stats.mannwhitneyu(a_rates, c_rates, alternative="two-sided")
    ac_t = stats.ttest_ind(a_rates, c_rates, equal_var=False)
    ac_diff, ac_perm_p, ac_perm_n, ac_perm_mode = exact_permutation_p_mean_diff(a_rates, c_rates)
    ac_d = cohens_d_unpaired(a_rates, c_rates)
    print(f"\n3. SANITY: recursive_clean vs baseline_clean")
    print(f"   A (recursive): {a_agg['bt_art_rate']:.1%} (turn-level)")
    print(f"   C (baseline):  {c_agg['bt_art_rate']:.1%} (turn-level)")
    print(f"   Turn-level Fisher OR={or_ac:.3f}, p={p_ac:.6f}")
    print(
        f"   Session-level mean diff={ac_diff:.3f}, "
        f"MW p={ac_mw.pvalue:.6f}, Welch p={ac_t.pvalue:.6f}, "
        f"Perm p={ac_perm_p:.6f} ({ac_perm_mode}, n={ac_perm_n}), d={ac_d:.3f}"
    )
    comparisons["sanity"] = {
        "turn_level": {"or": float(or_ac), "p": float(p_ac)},
        "session_level": {
            "mean_diff": float(ac_diff),
            "mannwhitney_u": float(ac_mw.statistic),
            "mannwhitney_p": float(ac_mw.pvalue),
            "welch_t": float(ac_t.statistic),
            "welch_p": float(ac_t.pvalue),
            "permutation_p": float(ac_perm_p),
            "permutation_mode": ac_perm_mode,
            "permutation_n": int(ac_perm_n),
            "cohens_d": float(ac_d),
        },
    }

    # Test 4: COMPARISON vs v2 single-layer (use session-level rates)
    print(f"\n4. R_V COMPARISON")
    for prefix in ["recursive_clean", "recursive_dual_patched", "baseline_clean", "baseline_dual_patched"]:
        row = agg[prefix]
        if row["mean_rv"] is not None:
            print(f"   {prefix:30s}: R_V = {row['mean_rv']:.4f} +/- {row['std_rv']:.4f}")

    # Session-level R_V contrasts
    a_rv = session_rv_vector("recursive_clean")
    b_rv = session_rv_vector("recursive_dual_patched")
    c_rv = session_rv_vector("baseline_clean")
    d_rv = session_rv_vector("baseline_dual_patched")
    rv_break_t = stats.ttest_ind(a_rv, b_rv, equal_var=False)
    rv_induce_t = stats.ttest_ind(c_rv, d_rv, equal_var=False)
    rv_break_diff, rv_break_perm_p, rv_break_perm_n, rv_break_perm_mode = exact_permutation_p_mean_diff(a_rv, b_rv)
    rv_induce_diff, rv_induce_perm_p, rv_induce_perm_n, rv_induce_perm_mode = exact_permutation_p_mean_diff(c_rv, d_rv)
    comparisons["rv_session_contrasts"] = {
        "break": {
            "mean_diff": float(rv_break_diff),
            "welch_t": float(rv_break_t.statistic),
            "welch_p": float(rv_break_t.pvalue),
            "permutation_p": float(rv_break_perm_p),
            "permutation_mode": rv_break_perm_mode,
            "permutation_n": int(rv_break_perm_n),
            "cohens_d": float(cohens_d_unpaired(a_rv, b_rv)),
        },
        "induce": {
            "mean_diff": float(rv_induce_diff),
            "welch_t": float(rv_induce_t.statistic),
            "welch_p": float(rv_induce_t.pvalue),
            "permutation_p": float(rv_induce_perm_p),
            "permutation_mode": rv_induce_perm_mode,
            "permutation_n": int(rv_induce_perm_n),
            "cohens_d": float(cohens_d_unpaired(c_rv, d_rv)),
        },
    }

    # ═══════════════════════════════════════════════════════════════
    # VERDICT
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")

    break_works = b_agg["bt_art_rate"] < a_agg["bt_art_rate"] and ab_perm_p < 0.05
    induce_works = d_agg["bt_art_rate"] > c_agg["bt_art_rate"] and cd_perm_p < 0.05

    if break_works and induce_works:
        print("  CAUSAL BRIDGE PROVEN (both directions, dual-layer)")
    elif induce_works:
        print("  PARTIAL: Dual-layer induction works, breaking inconclusive")
    elif break_works:
        print("  PARTIAL: Dual-layer breaking works, induction inconclusive")
    else:
        print("  CAUSAL BRIDGE NOT PROVEN with dual-layer patching")

    # ═══════════════════════════════════════════════════════════════
    # SAVE
    # ═══════════════════════════════════════════════════════════════
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "experiment": "persistent_patching_v3_dual_layer",
        "prompt_bank_version": _subset.source_bank_version,
        "prompt_subset_name": _subset.name,
        "prompt_subset_schema_version": _subset.schema_version,
        "prompt_subset_path": str(_subset.manifest_path),
        "prompt_tier": "core_measurement",
        "metric_path": "geometric_lens.metrics.compute_rv_with_components",
        "generation_format": "chat_template" if supports_chat_template(tokenizer) else "raw_completion",
        "seed_recursive_prompt_ids": SEED_RECURSIVE_IDS,
        "seed_baseline_prompt_ids": SEED_BASELINE_IDS,
        "donor_recursive_prompt_ids": DONOR_RECURSIVE_IDS,
        "donor_baseline_prompt_ids": DONOR_BASELINE_IDS,
        "canonical_registry_path": arch_cfg["registry_path"],
        "canonical_registry_schema_version": arch_cfg["registry_schema_version"],
        "early": early, "late": late,
        "v_layer": v_layer, "r_layer": r_layer,
        "n_sessions_per_condition": n_sessions,
        "max_turns_per_session": max_turns,
        "description": "Dual-layer persistent patching: L18 residual + L27 V-proj, 4-condition break+induce",
        "aggregated": serialize_aggregates(agg),
        "comparisons": comparisons,
        "conditions": conditions,
    }

    outdir = Path("results/persistent_patching_v3")
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / f"persistent_patching_v3_dual_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(outfile, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {outfile}")


if __name__ == "__main__":
    main()
