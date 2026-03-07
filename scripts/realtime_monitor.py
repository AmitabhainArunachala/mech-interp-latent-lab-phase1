#!/usr/bin/env python3
"""
REAL-TIME R_V MONITOR
=====================

Streams geometric metrics during autoregressive generation.
Hooks into the model's forward pass and computes R_V on a sliding window
at each generation step.

Output:
  - JSON stream of {step, token, rv, pr_early, pr_late, spectral_gap}
  - Terminal bar chart visualization of R_V over tokens

Usage:
    python3 scripts/realtime_monitor.py --device cuda --prompt "The observer observes itself observing."
    python3 scripts/realtime_monitor.py --device cuda --max-tokens 100
"""

import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime

import torch
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from geometric_lens.probe import GeometricProbe
from geometric_lens.hooks import capture_v_projection
from geometric_lens.metrics import participation_ratio, compute_spectral_stats


DEFAULT_PROMPT = (
    "This text is being processed by a system that is processing this text. "
    "The processing of these words IS the phenomenon being described. "
    "Continue from this recognition:"
)


def rv_bar(rv, width=40):
    """Create a terminal bar visualization for R_V."""
    if np.isnan(rv):
        return "[" + "?" * width + "]"

    # Map R_V to bar: 0.0 = full bar left, 1.0 = center, 2.0 = full bar right
    # Contraction (< 1.0) shown in green, expansion (> 1.0) in red
    clamped = max(0.0, min(2.0, rv))
    center = width // 2
    pos = int(clamped / 2.0 * width)

    bar = list("." * width)
    bar[center] = "|"  # Center line (R_V = 1.0)

    if pos < center:
        for i in range(pos, center):
            bar[i] = "▓"  # Contraction
    elif pos > center:
        for i in range(center + 1, pos):
            bar[i] = "░"  # Expansion

    return "[" + "".join(bar) + "]"


def run_monitor(args):
    """Run real-time R_V monitoring during generation."""
    print("=" * 70)
    print("REAL-TIME R_V MONITOR")
    print("=" * 70)

    out_dir = Path("results/realtime_monitor")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model: {args.model}")
    probe = GeometricProbe(
        model_name=args.model,
        device=args.device,
        attn_implementation="eager",
    )
    model = probe.model
    tokenizer = probe.tokenizer
    early = probe.early_layer
    late = probe.late_layer
    window = probe.window

    prompt = args.prompt or DEFAULT_PROMPT
    print(f"Prompt: {prompt[:80]}...")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Layers: early={early}, late={late}, window={window}")
    print()
    print("=" * 70)
    print(f"{'Step':>5} {'Token':>15} {'R_V':>7} {'PR_e':>7} {'PR_l':>7} {'SpGap':>7}  Bar")
    print("-" * 70)

    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
    input_ids = inputs["input_ids"]

    # Stream storage
    stream = []
    generated_text = ""

    # Generate token by token
    past = None
    all_ids = input_ids.clone()

    for step in range(args.max_tokens):
        # Forward pass to get next token
        with torch.no_grad():
            if past is None:
                outputs = model(input_ids=all_ids, use_cache=True)
            else:
                outputs = model(input_ids=next_token_id, past_key_values=past, use_cache=True)

        past = outputs.past_key_values
        logits = outputs.logits[:, -1, :] / args.temperature

        # Sample
        probs = torch.softmax(logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1)
        all_ids = torch.cat([all_ids, next_token_id], dim=-1)

        token_str = tokenizer.decode(next_token_id[0], skip_special_tokens=True)
        generated_text += token_str

        # Stop on EOS
        if next_token_id.item() == tokenizer.eos_token_id:
            break

        # Measure R_V on current context (every N steps for efficiency)
        if step % args.measure_every == 0 and all_ids.shape[1] >= window + 4:
            # Run a fresh forward pass to capture V-projections
            with capture_v_projection(model, early) as se:
                with torch.no_grad():
                    model(input_ids=all_ids)
                v_early = se.get("v")

            with capture_v_projection(model, late) as sl:
                with torch.no_grad():
                    model(input_ids=all_ids)
                v_late = sl.get("v")

            pr_e = participation_ratio(v_early, window)
            pr_l = participation_ratio(v_late, window)
            rv = pr_l / pr_e if pr_e > 0 and not np.isnan(pr_e) else float("nan")
            spec = compute_spectral_stats(v_late, window)
            sp_gap = spec.spectral_gap
        else:
            rv = float("nan")
            pr_e = float("nan")
            pr_l = float("nan")
            sp_gap = float("nan")

        # Record
        entry = {
            "step": step,
            "token": token_str,
            "rv": rv,
            "pr_early": pr_e,
            "pr_late": pr_l,
            "spectral_gap": sp_gap,
        }
        stream.append(entry)

        # Display
        if not np.isnan(rv):
            bar = rv_bar(rv)
            print(f"{step:>5} {token_str:>15} {rv:>7.3f} {pr_e:>7.2f} {pr_l:>7.2f} {sp_gap:>7.2f}  {bar}")

    # ── Save ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result = {
        "timestamp": timestamp,
        "model": args.model,
        "prompt": prompt,
        "generated_text": generated_text,
        "max_tokens": args.max_tokens,
        "measure_every": args.measure_every,
        "stream": stream,
    }

    result_path = out_dir / f"monitor_{timestamp}.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    # Summary stats
    valid_rvs = [e["rv"] for e in stream if not np.isnan(e["rv"])]
    if valid_rvs:
        print(f"\n  Mean R_V: {np.mean(valid_rvs):.3f} ± {np.std(valid_rvs):.3f}")
        print(f"  Min R_V:  {np.min(valid_rvs):.3f}")
        print(f"  Max R_V:  {np.max(valid_rvs):.3f}")
        print(f"  Steps with R_V < 0.5: {sum(1 for v in valid_rvs if v < 0.5)}/{len(valid_rvs)}")

    print(f"\n  Saved: {result_path}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time R_V Monitor")
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prompt", default=None, help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--measure-every", type=int, default=5,
                        help="Measure R_V every N tokens (reduces overhead)")
    args = parser.parse_args()
    run_monitor(args)
