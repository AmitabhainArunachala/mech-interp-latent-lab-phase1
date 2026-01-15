#!/usr/bin/env python3
"""
EIGENSTATE DIRECTION FINDER
===========================

1. Find the TRUE steering direction using PCA/SVD (not just mean difference)
2. Measure eigenstate emergence across layers
3. Track eigenvalue dominance (λ₁/Σλ)

Key insight: The mean difference might not be the causal direction.
The TRUE direction should be the one that:
- Explains the most variance in the recursive-vs-baseline difference
- Shows progressive emergence across layers
- Has consistent sign/orientation
"""

from __future__ import annotations

import gc
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.models import load_model, set_seed
from prompts.loader import PromptLoader


def compute_eigenstate_metrics(
    hidden_states: torch.Tensor,
    *,
    window: int = 16,
    window_mode: str = "tail",  # "tail" | "random"
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """
    Compute eigenstate metrics from hidden states.
    
    Returns:
        - participation_ratio: PR = (Σλ)² / Σ(λ²) - effective dimensionality
        - eigenvalue_dominance: λ₁/Σλ - how dominant is the top eigenvalue
        - top_eigenvalues: top 5 eigenvalues
        - effective_rank: exp(entropy of normalized eigenvalues)
    """
    if hidden_states.dim() == 3:
        hidden_states = hidden_states[0]
    T, D = hidden_states.shape
    W = int(min(window, T))
    if W <= 0:
        return {"participation_ratio": np.nan, "eigenvalue_dominance": np.nan, "effective_rank": np.nan}

    if window_mode not in ("tail", "random"):
        raise ValueError(f"Unknown window_mode: {window_mode}")

    if window_mode == "tail":
        start = T - W
    else:
        if rng is None:
            rng = np.random.default_rng(0)
        start = int(rng.integers(0, T - W + 1))

    h_window = hidden_states[start : start + W, :].float()
    
    try:
        # SVD of the hidden states
        U, S, Vt = torch.linalg.svd(h_window.T, full_matrices=False)
        S_np = S.cpu().numpy()
        
        # Eigenvalues are S²
        eigenvalues = S_np ** 2
        total = eigenvalues.sum()
        
        if total < 1e-10:
            return {"participation_ratio": np.nan, "eigenvalue_dominance": np.nan}
        
        # Participation ratio
        pr = (total ** 2) / (eigenvalues ** 2).sum()
        
        # Eigenvalue dominance (how much does λ₁ explain?)
        dominance = eigenvalues[0] / total
        
        # Effective rank (entropy-based)
        p = eigenvalues / total
        p_nonzero = p[p > 1e-10]
        entropy = -np.sum(p_nonzero * np.log(p_nonzero))
        effective_rank = np.exp(entropy)
        
        return {
            "participation_ratio": float(pr),
            "eigenvalue_dominance": float(dominance),
            "effective_rank": float(effective_rank),
            "top_5_eigenvalues": eigenvalues[:5].tolist(),
            "top_eigenvalue": float(eigenvalues[0]),
            "T": int(T),
            "W": int(W),
            "window_mode": window_mode,
            "window_start": int(start),
        }
    except Exception as e:
        return {"participation_ratio": np.nan, "eigenvalue_dominance": np.nan, "error": str(e)}


def extract_pca_direction(
    recursive_hidden: torch.Tensor,
    baseline_hidden: torch.Tensor,
    n_components: int = 5
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Extract the principal directions that distinguish recursive from baseline.
    
    Instead of just taking mean difference, use PCA on the paired differences
    to find the direction that explains the most variance.
    """
    if recursive_hidden.dim() == 3:
        recursive_hidden = recursive_hidden[0]
    if baseline_hidden.dim() == 3:
        baseline_hidden = baseline_hidden[0]
    
    # Take last W tokens (use minimum of both)
    W = min(16, recursive_hidden.shape[0], baseline_hidden.shape[0])
    rec_h = recursive_hidden[-W:, :].float()
    base_h = baseline_hidden[-W:, :].float()
    
    # Compute difference
    diff = rec_h - base_h  # (W, D)
    
    # Center the differences
    diff_centered = diff - diff.mean(dim=0)
    
    # SVD to get principal directions
    U, S, Vt = torch.linalg.svd(diff_centered.T, full_matrices=False)
    
    # The first column of Vt is the principal direction
    # But we want directions in the hidden space, so use U
    # U: (D, k) - columns are the principal components
    
    # Explained variance ratio
    explained_variance = (S ** 2).cpu().numpy()
    total_var = explained_variance.sum()
    explained_ratio = explained_variance / total_var if total_var > 0 else np.zeros_like(explained_variance)
    
    return U[:, :n_components], explained_ratio[:n_components]


def run_eigenstate_analysis(
    model_name: str = "mistralai/Mistral-7B-v0.1",
    device: str = "cuda",
    output_dir: Optional[Path] = None,
    window: int = 16,
    min_tokens: int = 64,
    window_mode: str = "tail",  # "tail" | "random"
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run eigenstate emergence analysis and find true steering direction.
    """
    print("=" * 80)
    print("EIGENSTATE DIRECTION FINDER")
    print("=" * 80)
    
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"results/phase3_attention/runs/{timestamp}_eigenstate_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("\n[1/5] Loading model...")
    set_seed(seed)
    model, tokenizer = load_model(model_name, device=device)
    model.eval()
    num_layers = model.config.num_hidden_layers
    rng = np.random.default_rng(seed + 1337)
    
    # Define prompts.
    #
    # IMPORTANT: short baseline prompts can artifactually look ~rank-1 if you
    # only examine the last <=16 tokens. To avoid this, we draw *long* prompts
    # from the repo prompt bank (fallback to the short list if needed).
    loader = PromptLoader()
    bank = loader.prompts
    bank_version = loader.version
    (output_dir / "prompt_bank_version.txt").write_text(bank_version)
    (output_dir / "prompt_bank_version.json").write_text(
        json.dumps({"version": bank_version}, indent=2) + "\n"
    )

    def _pick_long_prompts(groups: List[str], n: int) -> List[str]:
        cands = []
        for k, v in bank.items():
            if v.get("group") in groups:
                text = v.get("text") or ""
                if not text:
                    continue
                tok_len = len(tokenizer.encode(text))
                if tok_len >= min_tokens:
                    cands.append(text)
        rng.shuffle(cands)
        return cands[:n]

    # Prefer long prompts from the bank (these are stable + comparable to other pipelines).
    recursive_prompts = _pick_long_prompts(["L5_refined", "L4_full", "L3_deeper"], n=8)
    baseline_prompts = _pick_long_prompts(["long_control", "baseline_creative", "baseline_math"], n=8)

    # DEC15 hygiene: no hardcoded fallback prompts. If too few long prompts, widen selection
    # to any prompts from the same groups (still from the bank), then fail loudly if still insufficient.
    def _pick_any_prompts(groups: List[str], n: int) -> List[str]:
        cands = []
        for _k, v in bank.items():
            if v.get("group") in groups:
                text = v.get("text") or ""
                if text:
                    cands.append(text)
        rng.shuffle(cands)
        return cands[:n]

    if len(recursive_prompts) < 5:
        recursive_prompts = _pick_any_prompts(["L5_refined", "L4_full", "L3_deeper"], n=8)
    if len(baseline_prompts) < 5:
        baseline_prompts = _pick_any_prompts(["long_control", "baseline_creative", "baseline_math"], n=8)

    if len(recursive_prompts) < 3 or len(baseline_prompts) < 3:
        raise RuntimeError("Prompt bank does not contain enough prompts for eigenstate analysis (need >=3 per side).")
    
    results = {
        "prompt_bank_version": bank_version,
        "prompts": {"recursive": recursive_prompts, "baseline": baseline_prompts},
    }
    results["params"] = {"window": int(window), "min_tokens": int(min_tokens), "window_mode": window_mode, "seed": int(seed)}
    
    # ==========================================================================
    # EXPERIMENT 1: Eigenstate metrics across layers for each prompt type
    # ==========================================================================
    print("\n[2/5] Computing eigenstate metrics across layers...")
    
    recursive_metrics = {l: [] for l in range(num_layers + 1)}
    baseline_metrics = {l: [] for l in range(num_layers + 1)}
    recursive_hidden_all = {l: [] for l in range(num_layers + 1)}
    baseline_hidden_all = {l: [] for l in range(num_layers + 1)}
    
    def _token_len(text: str) -> int:
        return int(len(tokenizer.encode(text)))

    # Filter out too-short prompts so W is truly window-sized and comparable.
    recursive_prompts_f = [p for p in recursive_prompts if _token_len(p) >= min_tokens]
    baseline_prompts_f = [p for p in baseline_prompts if _token_len(p) >= min_tokens]
    results["prompt_token_lengths"] = {
        "recursive": [{"tokens": _token_len(p), "text": p[:120]} for p in recursive_prompts],
        "baseline": [{"tokens": _token_len(p), "text": p[:120]} for p in baseline_prompts],
        "recursive_filtered_n": len(recursive_prompts_f),
        "baseline_filtered_n": len(baseline_prompts_f),
    }

    if len(recursive_prompts_f) < 3 or len(baseline_prompts_f) < 3:
        print(f"[warn] Not enough prompts >= min_tokens={min_tokens}. Falling back to unfiltered prompts.")
        recursive_prompts_f = recursive_prompts
        baseline_prompts_f = baseline_prompts

    for prompt in recursive_prompts_f:
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**enc, output_hidden_states=True)
        
        for l in range(num_layers + 1):
            h = outputs.hidden_states[l]
            metrics = compute_eigenstate_metrics(h, window=window, window_mode=window_mode, rng=rng)
            recursive_metrics[l].append(metrics)
            recursive_hidden_all[l].append(h.detach().cpu())
    
    for prompt in baseline_prompts_f:
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**enc, output_hidden_states=True)
        
        for l in range(num_layers + 1):
            h = outputs.hidden_states[l]
            metrics = compute_eigenstate_metrics(h, window=window, window_mode=window_mode, rng=rng)
            baseline_metrics[l].append(metrics)
            baseline_hidden_all[l].append(h.detach().cpu())
    
    # Aggregate metrics
    eigenstate_emergence = []
    for l in range(num_layers + 1):
        rec_pr = np.mean([m["participation_ratio"] for m in recursive_metrics[l] if not np.isnan(m["participation_ratio"])])
        base_pr = np.mean([m["participation_ratio"] for m in baseline_metrics[l] if not np.isnan(m["participation_ratio"])])
        rec_dom = np.mean([m["eigenvalue_dominance"] for m in recursive_metrics[l] if not np.isnan(m.get("eigenvalue_dominance", np.nan))])
        base_dom = np.mean([m["eigenvalue_dominance"] for m in baseline_metrics[l] if not np.isnan(m.get("eigenvalue_dominance", np.nan))])
        
        eigenstate_emergence.append({
            "layer": l,
            "recursive_pr": float(rec_pr),
            "baseline_pr": float(base_pr),
            "pr_ratio": float(rec_pr / base_pr) if base_pr > 0 else np.nan,
            "recursive_dominance": float(rec_dom),
            "baseline_dominance": float(base_dom),
        })
    
    results["eigenstate_emergence"] = eigenstate_emergence
    
    print("  Layer | Rec PR | Base PR | Ratio | Rec Dom | Base Dom")
    for e in eigenstate_emergence[::4]:  # Every 4th layer
        print(f"    L{e['layer']:2d}: {e['recursive_pr']:6.2f} | {e['baseline_pr']:6.2f} | {e['pr_ratio']:.3f} | {e['recursive_dominance']:.3f} | {e['baseline_dominance']:.3f}")
    
    # ==========================================================================
    # EXPERIMENT 2: Find TRUE steering direction using PCA
    # ==========================================================================
    print("\n[3/5] Extracting PCA-based steering directions...")
    
    pca_directions = {}
    
    for l in [8, 12, 16, 20, 24, 27]:
        # Use first pair for direction extraction
        rec_h = recursive_hidden_all[l][0].to(device)
        base_h = baseline_hidden_all[l][0].to(device)
        
        principal_dirs, explained_ratio = extract_pca_direction(rec_h, base_h, n_components=5)
        
        # Also compute mean difference direction
        W = min(16, rec_h.shape[1], base_h.shape[1])
        rec_mean = rec_h[0, -W:, :].float().mean(dim=0)
        base_mean = base_h[0, -W:, :].float().mean(dim=0)
        mean_diff_dir = rec_mean - base_mean
        mean_diff_dir = mean_diff_dir / (mean_diff_dir.norm() + 1e-8)
        
        # Cosine similarity between PCA direction and mean difference
        pc1 = principal_dirs[:, 0]
        cos_sim = float(F.cosine_similarity(pc1.unsqueeze(0), mean_diff_dir.unsqueeze(0)).cpu())
        
        pca_directions[l] = {
            "explained_ratio": explained_ratio.tolist(),
            "pc1_explains": float(explained_ratio[0]),
            "cos_pc1_to_mean_diff": cos_sim,
            "mean_diff_norm": float(mean_diff_dir.norm().cpu()),
        }
        
        print(f"  L{l}: PC1 explains {explained_ratio[0]*100:.1f}%, cos(PC1, mean_diff) = {cos_sim:.3f}")
    
    results["pca_directions"] = pca_directions
    
    # ==========================================================================
    # EXPERIMENT 3: Track direction consistency across prompts
    # ==========================================================================
    print("\n[4/5] Testing direction consistency across prompts...")
    
    direction_consistency = {}
    
    for l in [12, 20, 27]:
        directions = []
        
        for i in range(min(len(recursive_prompts), len(baseline_prompts))):
            rec_h = recursive_hidden_all[l][i].to(device)
            base_h = baseline_hidden_all[l][i].to(device)
            
            W = min(16, rec_h.shape[1], base_h.shape[1])
            rec_mean = rec_h[0, -W:, :].float().mean(dim=0)
            base_mean = base_h[0, -W:, :].float().mean(dim=0)
            direction = rec_mean - base_mean
            direction = direction / (direction.norm() + 1e-8)
            directions.append(direction)
        
        # Compute pairwise cosine similarities
        cos_sims = []
        for i in range(len(directions)):
            for j in range(i+1, len(directions)):
                cos = float(F.cosine_similarity(directions[i].unsqueeze(0), directions[j].unsqueeze(0)).cpu())
                cos_sims.append(cos)
        
        direction_consistency[l] = {
            "mean_cos_sim": float(np.mean(cos_sims)),
            "min_cos_sim": float(np.min(cos_sims)),
            "max_cos_sim": float(np.max(cos_sims)),
            "std_cos_sim": float(np.std(cos_sims)),
        }
        
        print(f"  L{l}: mean cos = {np.mean(cos_sims):.3f}, min = {np.min(cos_sims):.3f}, max = {np.max(cos_sims):.3f}")
    
    results["direction_consistency"] = direction_consistency
    
    # ==========================================================================
    # EXPERIMENT 4: Eigenstate emergence profile
    # ==========================================================================
    print("\n[5/5] Computing eigenstate emergence profile...")
    
    # Find where eigenstate "emerges" (where dominance spikes for recursive)
    emergence_profile = []
    
    for l in range(num_layers + 1):
        # Get mean hidden state across recursive prompts  
        # Use minimum window size to handle different lengths
        min_len_rec = min(h.shape[1] for h in recursive_hidden_all[l])
        min_len_base = min(h.shape[1] for h in baseline_hidden_all[l])
        W = min(16, min_len_rec, min_len_base)
        
        rec_hiddens = [h[0, -W:, :] for h in recursive_hidden_all[l]]
        base_hiddens = [h[0, -W:, :] for h in baseline_hidden_all[l]]
        
        rec_stack = torch.stack(rec_hiddens).float()  # (n_prompts, W, D)
        base_stack = torch.stack(base_hiddens).float()
        
        # Compute variance across prompts (lower = more consistent = more "eigenstate-like")
        rec_var = rec_stack.var(dim=0).mean().item()
        base_var = base_stack.var(dim=0).mean().item()
        
        emergence_profile.append({
            "layer": l,
            "recursive_variance": float(rec_var),
            "baseline_variance": float(base_var),
            "variance_ratio": float(rec_var / base_var) if base_var > 0 else np.nan,
        })
    
    results["emergence_profile"] = emergence_profile
    
    # ==========================================================================
    # GENERATE OUTPUTS
    # ==========================================================================
    print("\n" + "=" * 80)
    print("GENERATING OUTPUTS")
    print("=" * 80)
    
    # 1. Eigenstate emergence plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    layers = [e["layer"] for e in eigenstate_emergence]
    
    ax = axes[0, 0]
    ax.plot(layers, [e["recursive_pr"] for e in eigenstate_emergence], 'b-o', label='Recursive')
    ax.plot(layers, [e["baseline_pr"] for e in eigenstate_emergence], 'r-o', label='Baseline')
    ax.set_xlabel("Layer")
    ax.set_ylabel("Participation Ratio")
    ax.set_title("PR Across Layers")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.plot(layers, [e["pr_ratio"] for e in eigenstate_emergence], 'g-o')
    ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("PR Ratio (Rec/Base)")
    ax.set_title("PR Ratio = R_V Analog")
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    ax.plot(layers, [e["recursive_dominance"] for e in eigenstate_emergence], 'b-o', label='Recursive')
    ax.plot(layers, [e["baseline_dominance"] for e in eigenstate_emergence], 'r-o', label='Baseline')
    ax.set_xlabel("Layer")
    ax.set_ylabel("Eigenvalue Dominance (λ₁/Σλ)")
    ax.set_title("Top Eigenvalue Dominance")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    ax.plot(layers, [e["recursive_variance"] for e in emergence_profile], 'b-o', label='Recursive')
    ax.plot(layers, [e["baseline_variance"] for e in emergence_profile], 'r-o', label='Baseline')
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cross-Prompt Variance")
    ax.set_title("Eigenstate Consistency (lower = more consistent)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "eigenstate_emergence.png", dpi=150)
    plt.close()
    print(f"  Saved: eigenstate_emergence.png")
    
    # 2. Summary JSON
    with open(output_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"  Saved: summary.json")
    
    # ==========================================================================
    # KEY FINDINGS
    # ==========================================================================
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    
    # Find the layer where PR ratio drops most dramatically
    pr_ratios = [e["pr_ratio"] for e in eigenstate_emergence]
    min_ratio_layer = eigenstate_emergence[np.argmin(pr_ratios)]["layer"]
    
    print(f"\n1. EIGENSTATE EMERGENCE:")
    print(f"   PR ratio drops to minimum at L{min_ratio_layer}")
    print(f"   Minimum PR ratio: {min(pr_ratios):.3f}")
    
    print(f"\n2. DIRECTION ANALYSIS:")
    for l, data in pca_directions.items():
        print(f"   L{l}: PC1 explains {data['pc1_explains']*100:.1f}%, cos(PC1,mean)={data['cos_pc1_to_mean_diff']:.3f}")
    
    print(f"\n3. DIRECTION CONSISTENCY:")
    for l, data in direction_consistency.items():
        print(f"   L{l}: mean_cos={data['mean_cos_sim']:.3f} ± {data['std_cos_sim']:.3f}")
    
    print("\n" + "=" * 80)
    print(f"[OK] Analysis complete. Run dir: {output_dir}")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Eigenstate Direction Finder")
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1", help="Model name")
    parser.add_argument("--device", default="cuda", help="Device")
    args = parser.parse_args()
    
    try:
        run_eigenstate_analysis(model_name=args.model, device=args.device)
    finally:
        print("\n[cleanup] Clearing GPU memory...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print("[cleanup] GPU memory cleared.")

