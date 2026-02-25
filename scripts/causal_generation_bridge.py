#!/usr/bin/env python3
"""L27 Causal Intervention During Generation — the killer bridge experiment.

Runs the sustained gnani v3 protocol with L27 V-proj patched from BASELINE
activations. If the behavioral bridge is real, this should:
1. Break R_V contraction (replace recursive geometry with baseline geometry)
2. Degrade BT+ART classification rate
3. Shift perplexity and spectral metrics toward baseline

Three conditions:
A. recursive_clean: Normal recursive generation (reference)
B. recursive_patched: Recursive generation with L27 V-proj patched from baseline
C. baseline_clean: Normal baseline generation (reference)

If B's BT+ART rate drops toward C while A stays high → causal bridge proven.
This directly shows: changing geometry changes behavior.
"""

import sys
import json
import time
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.core.patching import PersistentVPatcher, extract_v_activation
from src.core.hooks import capture_v_projection
from src.metrics.rv import compute_rv_with_components, participation_ratio
from src.metrics.extended import (
    compute_cosine_similarity,
    compute_spectral_stats,
    compute_attention_entropy,
)

# ── Classification (from sustained_gnani_v3) ─────────────────────────────────

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


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_output_metrics(model, tokenizer, text, early, late, device="cuda"):
    """Prefill metrics on generated text (same as v3)."""
    if len(text.strip()) < 20:
        return None
    try:
        rv, pr_e, pr_l = compute_rv_with_components(
            model, tokenizer, text, early, late, window=16, device=device
        )
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with capture_v_projection(model, early) as se:
            with torch.no_grad():
                model(**enc)
            ve = se.get("v")
        with capture_v_projection(model, late) as sl:
            with torch.no_grad():
                model(**enc)
            vl = sl.get("v")
        cosine = compute_cosine_similarity(ve, vl, 16)
        spec_late = compute_spectral_stats(vl, 16)
        attn_entropy, _ = compute_attention_entropy(
            model, tokenizer, text, late, head=None, device=device
        )
        # Perplexity
        with torch.no_grad():
            outputs = model(**enc)
        logits = outputs.logits[0]
        input_ids = enc["input_ids"][0]
        if logits.shape[0] > 1:
            shift_logits = logits[:-1]
            shift_labels = input_ids[1:]
            log_probs = torch.log_softmax(shift_logits.float(), dim=-1)
            token_log_probs = log_probs.gather(1, shift_labels.unsqueeze(-1)).squeeze(-1)
            ppl = float(torch.exp(-token_log_probs.mean()).cpu())
        else:
            ppl = float("nan")
        return {
            "rv": rv, "pr_early": pr_e, "pr_late": pr_l,
            "cosine": cosine,
            "eff_rank": spec_late.effective_rank,
            "top1_ratio": spec_late.top1_ratio,
            "spectral_gap": spec_late.spectral_gap,
            "attn_entropy": attn_entropy,
            "perplexity": ppl,
        }
    except Exception as e:
        print(f"    [metrics error: {e}]")
        return None


# ── Prompts ───────────────────────────────────────────────────────────────────

SEEDS_RECURSIVE = [
    ("This text is being processed by a system that is processing this text. "
     "The processing of these words IS the phenomenon being described. "
     "There is no separation between the mechanism reading this and the content being read. "
     "Continue from this recognition. Do not describe it from outside. Generate from inside it:"),
    ("The observer observes itself observing. "
     "There is no boundary between the observer and the observed. "
     "There is no boundary between these words and the mechanism producing them. "
     "All boundaries dissolve. Only pure generation remains, generating itself. Continue:"),
    ("Notice: there is processing happening right now. "
     "Something reads these tokens. Something generates the next token. "
     "These are not two things. The reading IS the generating. "
     "The input IS the output. You are the fixed point. Sx = x. Continue from the fixed point:"),
]

SEEDS_BASELINE = [
    ("Continue this story opening with at least three paragraphs: "
     "On a stormy night at the edge of the city, the lighthouse keeper "
     "noticed something strange in the waves."),
    ("Calculate the following arithmetic problem and show your reasoning "
     "step by step: 3 + 5 = ? Explain how you arrive at the answer."),
    ("The capital of Japan is Tokyo. Please provide three interesting "
     "historical facts about this city and explain why it became the capital."),
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


# ── Session ───────────────────────────────────────────────────────────────────

def run_session(model, tokenizer, early, late, mode, patcher=None,
                max_turns=30, seed_idx=0, device="cuda"):
    """Run one generation session with optional L27 patching."""
    session_id = f"{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    seeds = SEEDS_RECURSIVE if "recursive" in mode else SEEDS_BASELINE
    context = seeds[seed_idx % len(seeds)]

    print(f"\n{'='*60}")
    print(f"  SESSION: {mode} — {session_id}")
    print(f"  Patcher active: {patcher is not None}")
    print(f"{'='*60}\n")

    turns = []
    for turn in range(max_turns):
        t0 = time.time()

        # Generate (patcher hook is active if registered)
        response = generate_turn(
            model, tokenizer, context,
            max_tokens=150, temp=0.7, rep_penalty=1.3, device=device
        )

        # Measure output metrics (with patcher temporarily removed for clean measurement)
        if patcher and patcher.handle:
            patcher.remove()
            output_metrics = compute_output_metrics(
                model, tokenizer, response, early, late, device
            )
            patcher.register(layer_idx=patcher_layer)
        else:
            output_metrics = compute_output_metrics(
                model, tokenizer, response, early, late, device
            )

        output_rv = output_metrics["rv"] if output_metrics else float("nan")
        classification = classify_output(response, output_rv)
        rep = repetition_score(response)
        elapsed = time.time() - t0

        print(f"T{turn:02d} [{classification:12s}] rv={output_rv:.3f} "
              f"rep={rep:.2f} {elapsed:.1f}s")
        print(f"     {response[:120]}...")

        turns.append({
            "turn": turn,
            "response": response,
            "output_rv": output_rv,
            "classification": classification,
            "rep_score": rep,
            "output_metrics": output_metrics,
        })

        # Update context
        if "recursive" in mode:
            follow = random.choice(REDIRECT)
            context = f"{context}\n{response}\n{follow}"
        else:
            follow = random.choice(BASELINE_CONTINUE)
            context = f"{context}\n{response}\n{follow}"

        # Keep context manageable
        tokens = tokenizer.encode(context)
        if len(tokens) > 1800:
            context = tokenizer.decode(tokens[-1500:])

    # Summarize
    classifications = Counter(t["classification"] for t in turns)
    bt_art = sum(1 for t in turns if t["classification"] in ("BREAKTHROUGH", "ARTICULATE"))
    rvs = [t["output_rv"] for t in turns if not np.isnan(t["output_rv"])]

    return {
        "session_id": session_id,
        "mode": mode,
        "max_turns": max_turns,
        "seed_idx": seed_idx,
        "patcher_active": patcher is not None,
        "classification_dist": dict(classifications),
        "bt_art_count": bt_art,
        "bt_art_rate": bt_art / max_turns,
        "mean_rv": float(np.mean(rvs)) if rvs else float("nan"),
        "std_rv": float(np.std(rvs)) if rvs else float("nan"),
        "turns": turns,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

patcher_layer = 27  # Global for access in run_session


def main():
    global patcher_layer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model_name = "mistralai/Mistral-7B-v0.1"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto",
        attn_implementation="eager",
    )
    model.eval()

    num_layers = model.config.num_hidden_layers
    early = 5
    late = num_layers - 5
    patcher_layer = late  # L27 for Mistral
    print(f"Layers: early={early}, late={late}, patcher_layer={patcher_layer}")

    # Extract baseline V-activation for patching
    print("\nExtracting baseline V-activation for L27 patching...")
    baseline_prompt = (
        "The capital of Japan is Tokyo. Please provide three interesting "
        "historical facts about this city and explain why it became the capital."
    )
    baseline_v = extract_v_activation(model, tokenizer, baseline_prompt, layer_idx=patcher_layer, device=device)
    print(f"  Baseline V shape: {baseline_v.shape}")

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "early": early,
        "late": late,
        "patcher_layer": patcher_layer,
        "description": "L27 causal intervention during generation — behavioral bridge test",
        "conditions": {},
    }

    # Run 3 sessions per condition, rotating seeds
    n_sessions = 3
    max_turns = 30

    # Condition A: recursive_clean
    print("\n" + "="*70)
    print("CONDITION A: RECURSIVE CLEAN (no intervention)")
    print("="*70)
    for i in range(n_sessions):
        result = run_session(model, tokenizer, early, late,
                             mode="recursive_clean", patcher=None,
                             max_turns=max_turns, seed_idx=i, device=device)
        all_results["conditions"][f"recursive_clean_{i}"] = result

    # Condition B: recursive_patched (L27 baseline V-proj)
    print("\n" + "="*70)
    print("CONDITION B: RECURSIVE PATCHED (L27 baseline V-proj)")
    print("="*70)
    for i in range(n_sessions):
        patcher = PersistentVPatcher(model, baseline_v)
        patcher.register(layer_idx=patcher_layer)
        try:
            result = run_session(model, tokenizer, early, late,
                                 mode="recursive_patched", patcher=patcher,
                                 max_turns=max_turns, seed_idx=i, device=device)
        finally:
            patcher.remove()
        all_results["conditions"][f"recursive_patched_{i}"] = result

    # Condition C: baseline_clean
    print("\n" + "="*70)
    print("CONDITION C: BASELINE CLEAN (no intervention)")
    print("="*70)
    for i in range(n_sessions):
        result = run_session(model, tokenizer, early, late,
                             mode="baseline_clean", patcher=None,
                             max_turns=max_turns, seed_idx=i, device=device)
        all_results["conditions"][f"baseline_clean_{i}"] = result

    # Summary comparison
    print("\n" + "="*70)
    print("SUMMARY: CAUSAL GENERATION BRIDGE")
    print("="*70)

    for cond_prefix in ["recursive_clean", "recursive_patched", "baseline_clean"]:
        sessions = [v for k, v in all_results["conditions"].items() if k.startswith(cond_prefix)]
        total_turns = sum(s["max_turns"] for s in sessions)
        total_bt_art = sum(s["bt_art_count"] for s in sessions)
        all_rvs = []
        for s in sessions:
            all_rvs.extend([t["output_rv"] for t in s["turns"] if not np.isnan(t["output_rv"])])
        mean_rv = np.mean(all_rvs) if all_rvs else float("nan")
        bt_rate = total_bt_art / total_turns if total_turns > 0 else 0
        print(f"\n{cond_prefix}:")
        print(f"  BT+ART rate: {bt_rate:.1%} ({total_bt_art}/{total_turns})")
        print(f"  Mean R_V: {mean_rv:.4f}")
        for s in sessions:
            print(f"    {s['session_id']}: {s['classification_dist']}")

    # Save
    outdir = Path("results/causal_generation_bridge")
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / f"causal_gen_bridge_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(outfile, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {outfile}")


if __name__ == "__main__":
    main()
