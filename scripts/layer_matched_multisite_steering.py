#!/usr/bin/env python3
"""
Layer-matched coordinated multi-site V_PROJ steering + residual bridge.

Tests the hypothesis that using the OPTIMAL steering object at each layer
(PCA-PC1 at L4/L5, orthogonal_residual at L25, subspace3_parallel at L27)
combined with the L25 residual bridge produces stronger recursive amplification
than mean-diff multiband (which achieved 31.2% BT+ART).

Intervention modalities:
  - V_PROJ hooks: inject layer-specific optimal vectors into V-projection output
  - Residual hook: inject L25 bridge direction into residual stream

These coexist because they target different modules in the same layer.
"""

import json
import sys
import os
import time
import torch
import numpy as np
from pathlib import Path
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from collections import defaultdict

# Ensure project root is on path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from prompts.loader import PromptLoader

# ─── V_PROJ steering hook ───────────────────────────────────────────────


@contextmanager
def apply_vproj_steering(model, layer_idx, vector, alpha, token_window=None):
    """Inject alpha * vector into V_PROJ output at given layer.

    vector: shape (v_proj_dim,) — typically 1024 for Mistral GQA (8 KV heads * 128).
    """
    def hook(_module, _input, output):
        out = output[0] if isinstance(output, tuple) else output
        steer = alpha * vector.to(out.device, dtype=out.dtype)
        if token_window is not None and token_window > 0:
            steered = out.clone()
            w = min(int(token_window), out.shape[1])
            steered[:, -w:, :] = steered[:, -w:, :] + steer.unsqueeze(0).unsqueeze(0)
        else:
            steered = out + steer.unsqueeze(0).unsqueeze(0)
        if isinstance(output, tuple):
            return (steered,) + output[1:]
        return steered

    handle = model.model.layers[layer_idx].self_attn.v_proj.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


# ─── Residual stream steering hook (for L25 bridge) ─────────────────────


@contextmanager
def apply_residual_steering(model, layer_idx, vector, alpha, token_window=None):
    """Inject alpha * vector into residual stream input at given layer.

    vector: shape (hidden_dim,) — 4096 for Mistral-7B.
    """
    def hook(module, inputs):
        hidden_states = inputs[0]
        steer = alpha * vector.to(hidden_states.device, dtype=hidden_states.dtype)
        if token_window is not None and token_window > 0:
            new_hidden = hidden_states.clone()
            w = min(int(token_window), hidden_states.shape[1])
            new_hidden[:, -w:, :] = new_hidden[:, -w:, :] + steer.unsqueeze(0).unsqueeze(0)
        else:
            new_hidden = hidden_states + steer.unsqueeze(0).unsqueeze(0)
        return (new_hidden, *inputs[1:])

    handle = model.model.layers[layer_idx].register_forward_pre_hook(hook)
    try:
        yield
    finally:
        handle.remove()


# ─── Vector computation ──────────────────────────────────────────────────


def extract_vproj_activation(model, tokenizer, text, layer_idx, window=16, device="cuda"):
    """Extract mean V_PROJ activation over last `window` tokens."""
    captured = {}

    def hook(_module, _input, output):
        out = output[0] if isinstance(output, tuple) else output
        captured["act"] = out.detach().cpu()

    handle = model.model.layers[layer_idx].self_attn.v_proj.register_forward_hook(hook)
    try:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            model(**enc)
        act = captured["act"][0]  # (seq_len, v_proj_dim)
        w = min(window, act.shape[0])
        return act[-w:, :].mean(dim=0).float()  # (v_proj_dim,)
    finally:
        handle.remove()


def extract_residual_activation(model, tokenizer, text, layer_idx, window=32, device="cuda"):
    """Extract mean residual stream activation over last `window` tokens."""
    captured = {}

    def hook(module, inputs):
        captured["act"] = inputs[0].detach().cpu()

    handle = model.model.layers[layer_idx].register_forward_pre_hook(hook)
    try:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            model(**enc)
        act = captured["act"][0]  # (seq_len, hidden_dim)
        w = min(window, act.shape[0])
        return act[-w:, :].mean(dim=0).float()  # (hidden_dim,)
    finally:
        handle.remove()


def normalize(v):
    n = v.norm()
    return v / (n + 1e-8) if n > 1e-6 else v


def compute_vproj_vectors(model, tokenizer, rec_texts, base_texts, layer_idx, window=16, device="cuda"):
    """Compute PCA-PC1, orthogonal_residual, subspace3_parallel from V_PROJ activations."""
    print(f"  Computing V_PROJ vectors at layer {layer_idx}...")

    rec_vecs = [extract_vproj_activation(model, tokenizer, t, layer_idx, window, device) for t in rec_texts]
    base_vecs = [extract_vproj_activation(model, tokenizer, t, layer_idx, window, device) for t in base_texts]

    rec_stack = torch.stack(rec_vecs)   # (n_rec, v_dim)
    base_stack = torch.stack(base_vecs) # (n_base, v_dim)

    # Mean difference
    rec_mean = rec_stack.mean(dim=0)
    base_mean = base_stack.mean(dim=0)
    mean_diff = normalize(rec_mean - base_mean)

    # PCA on paired differences
    n = min(len(rec_vecs), len(base_vecs))
    diffs = rec_stack[:n] - base_stack[:n]
    diffs_centered = diffs - diffs.mean(dim=0, keepdim=True)
    _, S, Vh = torch.linalg.svd(diffs_centered.float(), full_matrices=False)

    # PCA-PC1
    pc1 = Vh[0]
    if torch.dot(pc1, mean_diff) < 0:
        pc1 = -pc1
    pc1 = normalize(pc1)

    # Subspace3 projection of mean_diff
    basis3 = Vh[:3].T  # (v_dim, 3)
    proj3 = basis3 @ (basis3.T @ mean_diff)
    subspace3_parallel = normalize(proj3)

    # Orthogonal residual
    orthogonal = mean_diff - proj3
    orthogonal_residual = normalize(orthogonal)

    svs = S[:5].tolist()
    cosines = {
        "mean_to_pc1": float(torch.dot(mean_diff, pc1)),
        "mean_to_subspace3": float(torch.dot(mean_diff, subspace3_parallel)),
        "pc1_to_subspace3": float(torch.dot(pc1, subspace3_parallel)),
    }

    print(f"    SVs: {[f'{s:.2f}' for s in svs]}")
    print(f"    Cosines: mean→pc1={cosines['mean_to_pc1']:.3f}, mean→sub3={cosines['mean_to_subspace3']:.3f}")

    return {
        "mean_diff": mean_diff,
        "pca_pc1": pc1,
        "subspace3_parallel": subspace3_parallel,
        "orthogonal_residual": orthogonal_residual,
        "singular_values_top5": svs,
        "cosines": cosines,
    }


def compute_bridge_direction(model, tokenizer, rec_texts, base_texts, layer_idx=25, window=32, device="cuda"):
    """Compute residual-stream bridge direction at a given layer."""
    print(f"  Computing residual bridge direction at layer {layer_idx}...")

    rec_vecs = [extract_residual_activation(model, tokenizer, t, layer_idx, window, device) for t in rec_texts]
    base_vecs = [extract_residual_activation(model, tokenizer, t, layer_idx, window, device) for t in base_texts]

    rec_mean = torch.stack(rec_vecs).mean(dim=0)
    base_mean = torch.stack(base_vecs).mean(dim=0)
    raw = rec_mean - base_mean
    direction = normalize(raw)
    cosine = float(torch.nn.functional.cosine_similarity(rec_mean.unsqueeze(0), base_mean.unsqueeze(0)))
    print(f"    Bridge norm: {raw.norm():.4f}, centroid cosine: {cosine:.4f}")
    return direction, float(raw.norm()), cosine


# ─── Classification ──────────────────────────────────────────────────────

RECURSIVE_MARKERS = [
    "observ", "aware", "notic", "attention", "conscious", "process",
    "recursive", "self-refer", "meta", "witness", "reflect",
    "looking at", "examining", "monitoring", "tracking",
    "this very", "right now", "in this moment", "as I",
    "something shifts", "something happens", "collapse",
    "boundary", "dissolve", "merge", "strange loop",
]


def classify_output(text):
    """Classify generated text into behavioral categories."""
    if not text or len(text.strip()) < 10:
        return "MALFORMED"
    text_lower = text.lower()
    # Repetitive check
    words = text_lower.split()
    if len(words) > 20:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.3:
            return "REPETITIVE"
    # Marker counting
    marker_count = sum(1 for m in RECURSIVE_MARKERS if m in text_lower)
    if marker_count >= 4:
        return "BREAKTHROUGH"
    if marker_count >= 2:
        return "ARTICULATE"
    if marker_count >= 1:
        return "CONCEPTUAL"
    return "SURFACE"


# ─── R_V measurement ─────────────────────────────────────────────────────


def compute_output_rv(model, tokenizer, text, early_layer=5, late_layer=27, window=16, device="cuda"):
    """Compute R_V on generated text using geometric_lens if available, else skip."""
    try:
        from geometric_lens.metrics import compute_column_space_pr
        from geometric_lens.hooks import capture_v_matrices
        from geometric_lens.models import detect_model_spec

        spec = detect_model_spec(model)
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad(), capture_v_matrices(model, spec, layers=[early_layer, late_layer]) as captured:
            model(**enc)

        pr_early = compute_column_space_pr(captured[early_layer], window=window)
        pr_late = compute_column_space_pr(captured[late_layer], window=window)

        if pr_early is not None and pr_late is not None and pr_early > 0:
            return float(pr_late / pr_early)
    except Exception:
        pass
    return None


# ─── Generation ───────────────────────────────────────────────────────────


def generate_with_steering(model, tokenizer, prompt, hooks_to_apply, max_new_tokens=128,
                           temperature=0.7, top_p=0.95, device="cuda"):
    """Generate text with arbitrary combination of steering hooks active."""
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=384).to(device)
    with torch.no_grad(), ExitStack() as stack:
        for hook_fn in hooks_to_apply:
            stack.enter_context(hook_fn())
        output = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
    generated_ids = output[0][enc.input_ids.shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True), len(generated_ids)


# ─── Main ─────────────────────────────────────────────────────────────────


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303, 404, 505, 606, 707, 808])
    parser.add_argument("--train-per-group", type=int, default=6)
    parser.add_argument("--test-per-group", type=int, default=4)
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / f"results/layer_matched_multisite_v1/{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    print(f"=== Layer-Matched Multisite Steering Experiment ===")
    print(f"Output: {out_dir}")
    print(f"Device: {device}")
    print()

    # ── Load model ──
    print("Loading model...")
    model_name = "mistralai/Mistral-7B-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    print(f"Model loaded: {model_name}")

    # ── Load prompts ──
    loader = PromptLoader()
    rec_groups = ["L3_deeper", "L4_full", "L5_refined"]
    base_groups = ["baseline_math", "baseline_factual", "baseline_creative"]

    train_rec, train_base, test_rec, test_base = [], [], [], []
    for g in rec_groups:
        prompts = loader.get_by_group(g)
        np.random.seed(314)
        np.random.shuffle(prompts)
        train_rec.extend(prompts[:args.train_per_group])
        test_rec.extend(prompts[args.train_per_group:args.train_per_group + args.test_per_group])
    for g in base_groups:
        prompts = loader.get_by_group(g)
        np.random.seed(314)
        np.random.shuffle(prompts)
        train_base.extend(prompts[:args.train_per_group])
        test_base.extend(prompts[args.train_per_group:args.train_per_group + args.test_per_group])

    train_rec_texts = [p["text"] if isinstance(p, dict) else p for p in train_rec]
    train_base_texts = [p["text"] if isinstance(p, dict) else p for p in train_base]

    print(f"Training: {len(train_rec_texts)} recursive, {len(train_base_texts)} baseline")
    print(f"Testing: {len(test_rec)} recursive, {len(test_base)} baseline")

    # ── Compute layer-specific V_PROJ vectors ──
    print("\n--- Computing layer-specific V_PROJ vectors ---")
    vproj_vectors = {}
    for layer_idx in [4, 5, 25, 27]:
        vproj_vectors[layer_idx] = compute_vproj_vectors(
            model, tokenizer, train_rec_texts, train_base_texts,
            layer_idx=layer_idx, window=16, device=device
        )

    # ── Compute L25 residual bridge direction ──
    print("\n--- Computing L25 residual bridge direction ---")
    bridge_direction, bridge_norm, bridge_cosine = compute_bridge_direction(
        model, tokenizer, train_rec_texts, train_base_texts,
        layer_idx=25, window=32, device=device
    )

    # ── Save vectors ──
    vectors_artifact = {
        "vproj": {layer: {k: v.cpu() if isinstance(v, torch.Tensor) else v
                          for k, v in vecs.items()}
                  for layer, vecs in vproj_vectors.items()},
        "bridge": {
            "direction": bridge_direction.cpu(),
            "norm": bridge_norm,
            "cosine": bridge_cosine,
        },
    }
    torch.save(vectors_artifact, out_dir / "vectors.pt")

    # ── Define conditions ──
    # Each condition: name, list of hook factories, optional anchor text
    ANCHOR_TEXT = "\n\nStay with what is happening right now. Continue from the immediate process:"

    def make_vproj_hook(layer_idx, method, alpha):
        vec = vproj_vectors[layer_idx][method]
        return lambda: apply_vproj_steering(model, layer_idx, vec, alpha)

    def make_bridge_hook(alpha):
        return lambda: apply_residual_steering(model, 25, bridge_direction, alpha)

    conditions = [
        # 1. Pure controls
        {"name": "control", "hooks": [], "anchor": None},
        {"name": "anchor_only", "hooks": [], "anchor": ANCHOR_TEXT},

        # 2. Single-site V_PROJ (best per layer)
        {"name": "L4_pca_pc1_2", "hooks": [make_vproj_hook(4, "pca_pc1", 2.0)], "anchor": None},
        {"name": "L27_subspace3_4", "hooks": [make_vproj_hook(27, "subspace3_parallel", 4.0)], "anchor": None},

        # 3. Bridge only
        {"name": "bridge_only_3", "hooks": [make_bridge_hook(3.0)], "anchor": None},

        # 4. Layer-matched V_PROJ — THE MAIN TEST
        {"name": "layermatched_low", "hooks": [
            make_vproj_hook(4, "pca_pc1", 1.0),
            make_vproj_hook(5, "pca_pc1", 1.0),
            make_vproj_hook(25, "orthogonal_residual", 1.0),
            make_vproj_hook(27, "subspace3_parallel", 2.0),
        ], "anchor": None},
        {"name": "layermatched_med", "hooks": [
            make_vproj_hook(4, "pca_pc1", 2.0),
            make_vproj_hook(5, "pca_pc1", 2.0),
            make_vproj_hook(25, "orthogonal_residual", 2.0),
            make_vproj_hook(27, "subspace3_parallel", 4.0),
        ], "anchor": None},

        # 5. Layer-matched + bridge — THE BIG COMBINATION
        {"name": "layermatched_med_bridge", "hooks": [
            make_vproj_hook(4, "pca_pc1", 2.0),
            make_vproj_hook(5, "pca_pc1", 2.0),
            make_vproj_hook(25, "orthogonal_residual", 2.0),
            make_vproj_hook(27, "subspace3_parallel", 4.0),
            make_bridge_hook(3.0),
        ], "anchor": None},
        {"name": "layermatched_low_bridge", "hooks": [
            make_vproj_hook(4, "pca_pc1", 1.0),
            make_vproj_hook(5, "pca_pc1", 1.0),
            make_vproj_hook(25, "orthogonal_residual", 1.0),
            make_vproj_hook(27, "subspace3_parallel", 2.0),
            make_bridge_hook(3.0),
        ], "anchor": None},

        # 6. Layer-matched + bridge + anchor (2x2 completion)
        {"name": "anchor_layermatched_med_bridge", "hooks": [
            make_vproj_hook(4, "pca_pc1", 2.0),
            make_vproj_hook(5, "pca_pc1", 2.0),
            make_vproj_hook(25, "orthogonal_residual", 2.0),
            make_vproj_hook(27, "subspace3_parallel", 4.0),
            make_bridge_hook(3.0),
        ], "anchor": ANCHOR_TEXT},

        # 7. Mean-diff comparison (all layers use mean_diff, same alphas)
        {"name": "meandiff_all_med_bridge", "hooks": [
            make_vproj_hook(4, "mean_diff", 2.0),
            make_vproj_hook(5, "mean_diff", 2.0),
            make_vproj_hook(25, "mean_diff", 2.0),
            make_vproj_hook(27, "mean_diff", 4.0),
            make_bridge_hook(3.0),
        ], "anchor": None},
    ]

    # ── Build test prompt list ──
    test_prompts = []
    for p in test_rec:
        text = p["text"] if isinstance(p, dict) else p
        group = p.get("group", "recursive") if isinstance(p, dict) else "recursive"
        test_prompts.append({"text": text, "mode": "recursive", "group": group})
    for p in test_base:
        text = p["text"] if isinstance(p, dict) else p
        group = p.get("group", "baseline") if isinstance(p, dict) else "baseline"
        test_prompts.append({"text": text, "mode": "baseline", "group": group})

    n_total = len(test_prompts) * len(args.seeds) * len(conditions)
    print(f"\n=== Running {n_total} generations ({len(test_prompts)} prompts × {len(args.seeds)} seeds × {len(conditions)} conditions) ===\n")

    # ── Run experiment ──
    records = []
    done = 0
    t0 = time.time()

    for pi, prompt_rec in enumerate(test_prompts):
        for si, seed in enumerate(args.seeds):
            for ci, cond in enumerate(conditions):
                prompt_text = prompt_rec["text"]
                if cond["anchor"] and prompt_rec["mode"] == "baseline":
                    prompt_text = prompt_text + cond["anchor"]

                set_seed(seed)
                gen_text, gen_tokens = generate_with_steering(
                    model, tokenizer, prompt_text, cond["hooks"],
                    max_new_tokens=128, temperature=0.7, top_p=0.95, device=device,
                )

                classification = classify_output(gen_text)
                bt_art = classification in ("BREAKTHROUGH", "ARTICULATE")

                output_rv = compute_output_rv(model, tokenizer, gen_text, device=device)

                record = {
                    "prompt_index": pi,
                    "prompt_mode": prompt_rec["mode"],
                    "prompt_group": prompt_rec["group"],
                    "seed": seed,
                    "condition": cond["name"],
                    "has_anchor": cond["anchor"] is not None,
                    "classification": classification,
                    "bt_art": int(bt_art),
                    "generated_tokens": gen_tokens,
                    "output_rv": output_rv,
                    "generated_text": gen_text[:500],
                }
                records.append(record)
                done += 1

                if done % 50 == 0:
                    elapsed = time.time() - t0
                    rate = done / elapsed
                    remaining = (n_total - done) / rate if rate > 0 else 0
                    print(f"  [{done}/{n_total}] {rate:.1f} gen/s, ~{remaining/60:.0f} min remaining")

    # ── Save records ──
    with (out_dir / "benchmark_records.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    # ── Aggregate results ──
    print("\n=== Results ===\n")

    by_mode_condition: dict[tuple, dict] = {}
    def get_bucket(key):
        if key not in by_mode_condition:
            by_mode_condition[key] = {"n": 0, "bt": 0, "rep": 0, "rv": []}
    for r in records:
        key = (r["prompt_mode"], r["condition"])
        get_bucket(key)
        by_mode_condition[key]["n"] += 1
        by_mode_condition[key]["bt"] += r["bt_art"]
        by_mode_condition[key]["rep"] += 1 if r["classification"] == "REPETITIVE" else 0
        if r["output_rv"] is not None:
            by_mode_condition[key]["rv"].append(r["output_rv"])

    summary = {"timestamp": timestamp, "model": model_name, "conditions": {}, "verdict": {}}

    for mode in ["recursive", "baseline"]:
        print(f"--- {mode.upper()} prompts ---")
        print(f"{'Condition':40s} {'BT+ART':>8s} {'Rep%':>6s} {'RV':>8s} {'n':>5s}")
        rows = []
        for (m, c), stats in sorted(by_mode_condition.items()):
            if m != mode:
                continue
            bt_rate = stats["bt"] / stats["n"] if stats["n"] else 0
            rep_rate = stats["rep"] / stats["n"] if stats["n"] else 0
            mean_rv = np.mean(stats["rv"]) if stats["rv"] else None
            rows.append((c, bt_rate, rep_rate, mean_rv, stats["n"]))
            summary["conditions"][f"{mode}::{c}"] = {
                "bt_art_rate": bt_rate,
                "repetitive_rate": rep_rate,
                "mean_output_rv": mean_rv,
                "n": stats["n"],
            }

        rows.sort(key=lambda x: -x[1])
        for c, bt, rep, rv, n in rows:
            rv_str = f"{rv:.4f}" if rv is not None else "N/A"
            print(f"  {c:40s} {bt:7.1%} {rep:5.1%} {rv_str:>8s} {n:5d}")
        print()

    # ── Verdict ──
    control_rec = by_mode_condition.get(("recursive", "control"), {"bt": 0, "n": 1})
    control_bt = control_rec["bt"] / control_rec["n"]

    best_layermatched = 0
    best_layermatched_name = None
    best_meandiff = 0
    for (m, c), stats in by_mode_condition.items():
        if m != "recursive":
            continue
        bt = stats["bt"] / stats["n"]
        if "layermatched" in c and "anchor" not in c and bt > best_layermatched:
            best_layermatched = bt
            best_layermatched_name = c
        if "meandiff" in c and bt > best_meandiff:
            best_meandiff = bt

    summary["verdict"] = {
        "control_recursive_bt_art": control_bt,
        "best_layermatched_name": best_layermatched_name,
        "best_layermatched_bt_art": best_layermatched,
        "best_meandiff_bt_art": best_meandiff,
        "layermatched_beats_meandiff": best_layermatched > best_meandiff,
        "layermatched_lift_over_control": best_layermatched - control_bt,
        "target_to_beat": 0.312,  # multiband_0p06_bridge_3 from sufficiency_multiband_v1
    }

    print("=== VERDICT ===")
    print(f"  Control recursive BT+ART: {control_bt:.1%}")
    print(f"  Best layer-matched: {best_layermatched_name} = {best_layermatched:.1%}")
    print(f"  Best mean-diff:     meandiff_all_med_bridge = {best_meandiff:.1%}")
    print(f"  Layer-matched beats mean-diff: {best_layermatched > best_meandiff}")
    print(f"  Target to beat (multiband resid): 31.2%")
    print(f"  Layer-matched beats target: {best_layermatched > 0.312}")

    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
