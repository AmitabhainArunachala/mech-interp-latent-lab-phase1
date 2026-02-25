#!/usr/bin/env python3
"""
PER-TOKEN R_V DIAGNOSTIC: R_V timeseries during autoregressive generation.

Hooks V-projections at early (L5) and late (L27) layers during generation.
Computes sliding-window PR using head-wise SVD (matching the head decomposition).

Head-wise PR computation:
  - Reshape V-proj window (16, 4096) → (num_heads, head_dim, 16)
  - SVD on each head's (head_dim, 16) matrix
  - PR = (sum(S²))² / sum(S⁴) per head
  - Average PR across heads

Produces diagnostic plot:
  - X-axis: generation step (token 16 to max_tokens)
  - Y-axis: R_V at that step
  - Color tokens: red for self-referential, blue otherwise
  - Horizontal lines: R_V=1.0 (no contraction) and mean champion R_V (~0.52)
  - Overlay recursive vs baseline on same plot
"""
import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.hooks import capture_v_projection
from src.core.hf_accessors import get_vproj_hookpoint, extract_v_from_hook_output
from prompts.loader import PromptLoader


SELF_REF_TOKENS = {
    "self", "itself", "this", "observe", "aware", "know", "process",
    "recursion", "recursive", "I", "me", "my", "observ", "generat",
    "recogni", "notic", "happen", "mechanism", "attention", "aware",
}


def is_self_referential(token_str: str) -> bool:
    """Check if a decoded token contains self-referential language."""
    t = token_str.lower().strip()
    for sr in SELF_REF_TOKENS:
        if sr in t:
            return True
    return False


def compute_pr_headwise(v_window: torch.Tensor, num_heads: int) -> float:
    """
    Head-wise PR: reshape (W, D) → (num_heads, head_dim, W), SVD each, average PR.

    Args:
        v_window: (W, D) tensor of V-projections
        num_heads: number of attention heads

    Returns:
        Average PR across heads.
    """
    W, D = v_window.shape
    head_dim = D // num_heads

    v_64 = v_window.double()
    # Reshape: (W, num_heads, head_dim) → (num_heads, head_dim, W)
    reshaped = v_64.view(W, num_heads, head_dim).permute(1, 2, 0)  # (H, d, W)

    prs = []
    for h in range(num_heads):
        head_mat = reshaped[h]  # (head_dim, W)
        try:
            U, S, Vt = torch.linalg.svd(head_mat, full_matrices=False)
            S_np = S.cpu().numpy()
            S_sq = S_np ** 2
            total = S_sq.sum()
            if total < 1e-10:
                continue
            pr = (total ** 2) / (S_sq ** 2).sum()
            prs.append(float(pr))
        except Exception:
            continue

    return float(np.mean(prs)) if prs else float("nan")


def run_generation_with_rv(
    model, tokenizer, prompt: str,
    early_layer: int = 5, late_layer: int = 27,
    max_new_tokens: int = 64, window: int = 16,
    device: str = "cuda",
):
    """
    Generate tokens autoregressively while tracking per-token R_V.

    Returns:
        dict with 'tokens', 'rv_timeseries', 'is_self_ref', 'pr_early_ts', 'pr_late_ts'
    """
    num_heads = model.config.num_attention_heads

    # Tokenize prompt
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    input_ids = enc["input_ids"]
    prompt_len = input_ids.shape[1]

    # Compute prompt-time R_V (reference)
    with capture_v_projection(model, early_layer) as se:
        with torch.no_grad():
            model(**enc)
        v_early_prompt = se.get("v")
    with capture_v_projection(model, late_layer) as sl:
        with torch.no_grad():
            model(**enc)
        v_late_prompt = sl.get("v")

    if v_early_prompt is not None and v_late_prompt is not None:
        ve = v_early_prompt[0] if v_early_prompt.dim() == 3 else v_early_prompt
        vl = v_late_prompt[0] if v_late_prompt.dim() == 3 else v_late_prompt
        if ve.shape[0] >= window and vl.shape[0] >= window:
            pr_e_prompt = compute_pr_headwise(ve[-window:], num_heads)
            pr_l_prompt = compute_pr_headwise(vl[-window:], num_heads)
            rv_prompt = pr_l_prompt / pr_e_prompt if pr_e_prompt > 0 else float("nan")
        else:
            rv_prompt = float("nan")
    else:
        rv_prompt = float("nan")

    # Autoregressive generation with V-proj hooks
    early_hookpoint = get_vproj_hookpoint(model, early_layer)
    late_hookpoint = get_vproj_hookpoint(model, late_layer)

    v_buffer_early = []
    v_buffer_late = []
    generated_tokens = []
    rv_timeseries = []
    pr_early_ts = []
    pr_late_ts = []
    is_self_ref = []

    # Capture storage
    v_early_last = {"v": None}
    v_late_last = {"v": None}

    def hook_early(module, inp, out):
        v = extract_v_from_hook_output(early_hookpoint, out)
        v_early_last["v"] = v.detach()
        return out

    def hook_late(module, inp, out):
        v = extract_v_from_hook_output(late_hookpoint, out)
        v_late_last["v"] = v.detach()
        return out

    h_early = early_hookpoint.module.register_forward_hook(hook_early)
    h_late = late_hookpoint.module.register_forward_hook(hook_late)

    try:
        with torch.no_grad():
            for step in range(max_new_tokens):
                # Forward pass
                outputs = model(input_ids=input_ids, use_cache=False)
                logits = outputs.logits

                # Grab V-projections for the last token position
                ve = v_early_last["v"]
                vl = v_late_last["v"]

                if ve is not None:
                    ve_tok = ve[0, -1, :] if ve.dim() == 3 else ve[-1, :]
                    v_buffer_early.append(ve_tok.cpu())
                if vl is not None:
                    vl_tok = vl[0, -1, :] if vl.dim() == 3 else vl[-1, :]
                    v_buffer_late.append(vl_tok.cpu())

                # Compute R_V when we have enough tokens
                if len(v_buffer_early) >= window and len(v_buffer_late) >= window:
                    v_win_e = torch.stack(v_buffer_early[-window:])  # (W, D)
                    v_win_l = torch.stack(v_buffer_late[-window:])   # (W, D)
                    pr_e = compute_pr_headwise(v_win_e, num_heads)
                    pr_l = compute_pr_headwise(v_win_l, num_heads)
                    rv = pr_l / pr_e if pr_e > 0 and not np.isnan(pr_e) else float("nan")
                    rv_timeseries.append(rv)
                    pr_early_ts.append(pr_e)
                    pr_late_ts.append(pr_l)
                else:
                    rv_timeseries.append(float("nan"))
                    pr_early_ts.append(float("nan"))
                    pr_late_ts.append(float("nan"))

                # Sample next token (greedy)
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                tok_str = tokenizer.decode(next_token[0])
                generated_tokens.append(tok_str)
                is_self_ref.append(is_self_referential(tok_str))

                # Append to input
                input_ids = torch.cat([input_ids, next_token], dim=-1)

                if next_token.item() == tokenizer.eos_token_id:
                    break

    finally:
        h_early.remove()
        h_late.remove()

    return {
        "prompt": prompt,
        "prompt_len": prompt_len,
        "rv_prompt": rv_prompt,
        "tokens": generated_tokens,
        "rv_timeseries": rv_timeseries,
        "pr_early_ts": pr_early_ts,
        "pr_late_ts": pr_late_ts,
        "is_self_ref": is_self_ref,
    }


def make_diagnostic_plot(results_list, output_path):
    """Create the per-token R_V diagnostic figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(results_list), 1,
                              figsize=(16, 5 * len(results_list)),
                              squeeze=False)

    for i, res in enumerate(results_list):
        ax = axes[i, 0]
        rvs = res["rv_timeseries"]
        tokens = res["tokens"]
        is_sr = res["is_self_ref"]
        mode = res.get("mode", "unknown")

        # Valid indices (where R_V is not NaN)
        valid_idx = [j for j, rv in enumerate(rvs) if not np.isnan(rv)]
        valid_rvs = [rvs[j] for j in valid_idx]

        if not valid_rvs:
            ax.text(0.5, 0.5, "No valid R_V values", transform=ax.transAxes, ha="center")
            continue

        # Color by self-referential
        colors = ["red" if is_sr[j] else "steelblue" for j in valid_idx]
        ax.scatter(valid_idx, valid_rvs, c=colors, s=30, zorder=3, alpha=0.8)
        ax.plot(valid_idx, valid_rvs, "k-", alpha=0.3, linewidth=0.8)

        # Reference lines
        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="R_V = 1.0 (no contraction)")
        ax.axhline(y=0.52, color="green", linestyle="--", alpha=0.5, label="Champion mean (0.52)")
        ax.axhline(y=res.get("rv_prompt", 0.52), color="purple", linestyle=":",
                    alpha=0.5, label=f"Prompt R_V ({res.get('rv_prompt', 0):.3f})")

        # Token annotations (every 4th)
        for j in valid_idx[::4]:
            if j < len(tokens):
                tok = tokens[j].strip()[:8]
                ax.annotate(tok, (j, rvs[j]), fontsize=6, rotation=45,
                           ha="left", va="bottom", alpha=0.7)

        ax.set_ylabel("R_V (head-wise)", fontsize=11)
        ax.set_title(f"{mode.upper()} — Per-Token R_V During Generation "
                     f"(red=self-ref, blue=other)", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(max(valid_rvs) * 1.2, 1.5))

    axes[-1, 0].set_xlabel("Generation Step", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Diagnostic plot saved: {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Per-Token R_V Diagnostic")
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--output", default="results/per_token_rv")
    args = parser.parse_args()

    from src.core.models import load_model, set_seed
    set_seed(42)

    print(f"Loading {args.model} with attn_implementation='eager' ...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_layers = model.config.num_hidden_layers
    early, late = 5, num_layers - 5

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Get one L5 recursive and one baseline prompt
    loader = PromptLoader()
    rec_prompts = loader.get_by_group("L5_refined", limit=3, seed=42)
    bas_prompts = loader.get_by_group("baseline_creative", limit=1, seed=42) + \
                  loader.get_by_group("baseline_math", limit=1, seed=42) + \
                  loader.get_by_group("baseline_factual", limit=1, seed=42)

    all_results = []

    # Run recursive prompts
    for i, prompt in enumerate(rec_prompts[:2]):
        print(f"\n{'='*60}")
        print(f"  RECURSIVE PROMPT {i+1}")
        print(f"  {prompt[:100]}...")
        print(f"{'='*60}\n")

        res = run_generation_with_rv(
            model, tokenizer, prompt,
            early_layer=early, late_layer=late,
            max_new_tokens=args.max_tokens, window=args.window,
            device=args.device,
        )
        res["mode"] = "recursive"
        all_results.append(res)

        valid = [rv for rv in res["rv_timeseries"] if not np.isnan(rv)]
        print(f"  R_V prompt: {res['rv_prompt']:.3f}")
        if valid:
            print(f"  R_V generation: {np.mean(valid):.3f} ± {np.std(valid):.3f}")
            print(f"  Min/Max: {min(valid):.3f} / {max(valid):.3f}")
        sr_count = sum(res["is_self_ref"])
        print(f"  Self-ref tokens: {sr_count}/{len(res['tokens'])}")
        print(f"  Generated: {''.join(res['tokens'][:30])}...")

    # Run baseline prompts
    for i, prompt in enumerate(bas_prompts[:2]):
        print(f"\n{'='*60}")
        print(f"  BASELINE PROMPT {i+1}")
        print(f"  {prompt[:100]}...")
        print(f"{'='*60}\n")

        res = run_generation_with_rv(
            model, tokenizer, prompt,
            early_layer=early, late_layer=late,
            max_new_tokens=args.max_tokens, window=args.window,
            device=args.device,
        )
        res["mode"] = "baseline"
        all_results.append(res)

        valid = [rv for rv in res["rv_timeseries"] if not np.isnan(rv)]
        print(f"  R_V prompt: {res['rv_prompt']:.3f}")
        if valid:
            print(f"  R_V generation: {np.mean(valid):.3f} ± {np.std(valid):.3f}")
        print(f"  Generated: {''.join(res['tokens'][:30])}...")

    # Save data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_data = []
    for res in all_results:
        save_data.append({
            "mode": res["mode"],
            "prompt": res["prompt"][:200],
            "rv_prompt": res["rv_prompt"],
            "tokens": res["tokens"],
            "rv_timeseries": res["rv_timeseries"],
            "pr_early_ts": res["pr_early_ts"],
            "pr_late_ts": res["pr_late_ts"],
            "is_self_ref": res["is_self_ref"],
        })
    with open(out_dir / f"per_token_rv_{timestamp}.json", "w") as f:
        json.dump(save_data, f, indent=2, default=str)

    # Plot
    make_diagnostic_plot(all_results, out_dir / "per_token_rv_diagnostic.png")

    # Summary comparison
    print(f"\n{'='*60}")
    print("  SUMMARY: RECURSIVE vs BASELINE per-token R_V")
    print(f"{'='*60}")
    for mode in ["recursive", "baseline"]:
        mode_res = [r for r in all_results if r["mode"] == mode]
        all_valid = []
        for r in mode_res:
            all_valid.extend([rv for rv in r["rv_timeseries"] if not np.isnan(rv)])
        if all_valid:
            print(f"  {mode.upper()}: mean={np.mean(all_valid):.3f} ± {np.std(all_valid):.3f} "
                  f"(n={len(all_valid)} tokens)")

    print(f"\nAll results saved to {out_dir}/")


if __name__ == "__main__":
    main()
