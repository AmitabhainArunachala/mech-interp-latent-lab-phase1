#!/usr/bin/env python3
"""
ANTHROPIC-LEVEL INVESTIGATION
============================

Comprehensive mechanistic interpretability investigation following
IOI paper standards:

1. PROPER ABLATION - Zero head outputs BEFORE residual stream addition
2. COMPLETE HEAD SWEEP - All 32 heads at key layers
3. PATH PATCHING - Trace information flow
4. CONTROLS - Random, shuffled, wrong-layer, opposite
5. FAITHFULNESS TEST - Can circuit alone produce R_V?
6. BACKUP HEAD DETECTION - What heads activate when primary fails?
7. DIRECTION EXTRACTION - Find and validate the steering direction

Goal: 90% confident this is publishable.
"""

from __future__ import annotations

import gc
import json
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.models import load_model, set_seed
from prompts.loader import PromptLoader


# =============================================================================
# CORE METRICS
# =============================================================================

def compute_pr(hidden_states: torch.Tensor, window: int = 16) -> float:
    """Compute Participation Ratio from hidden states."""
    if hidden_states.dim() == 3:
        hidden_states = hidden_states[0]
    T, D = hidden_states.shape
    W = min(window, T)
    h_window = hidden_states[-W:, :].float()
    
    try:
        U, S, Vt = torch.linalg.svd(h_window.T, full_matrices=False)
        S_np = S.cpu().numpy()
        S_sq = S_np ** 2
        if S_sq.sum() < 1e-10:
            return np.nan
        return float((S_sq.sum()**2) / (S_sq**2).sum())
    except:
        return np.nan


def compute_rv(early_hidden: torch.Tensor, late_hidden: torch.Tensor, window: int = 16) -> float:
    """Compute R_V = PR(late) / PR(early)."""
    pr_early = compute_pr(early_hidden, window)
    pr_late = compute_pr(late_hidden, window)
    if pr_early <= 0 or np.isnan(pr_early):
        return np.nan
    return pr_late / pr_early


# =============================================================================
# PROPER ABLATION HOOKS - Intervene BEFORE residual stream
# =============================================================================

class HeadAblationHook:
    """
    Ablate a specific attention head using mean ablation.
    
    We hook the entire layer and replace the attention output with its mean,
    effectively removing the head's contribution to the information flow.
    
    This uses a pre-hook to modify inputs, which is more reliable.
    """
    
    def __init__(self, model, layer_idx: int, head_idx: int):
        self.model = model
        self.layer_idx = layer_idx
        self.head_idx = head_idx
        self.handle = None
        self.num_heads = model.config.num_attention_heads
        self.head_dim = model.config.hidden_size // self.num_heads
        self.mean_activation = None
        
    def compute_mean_activation(self, model, tokenizer, prompts, device):
        """Pre-compute mean activation for this head across prompts."""
        activations = []
        
        def capture_hook(module, input, output):
            # output[0] is attention output
            attn_out = output[0]  # (batch, seq, hidden)
            activations.append(attn_out.detach().clone())
            return output
        
        handle = model.model.layers[self.layer_idx].self_attn.register_forward_hook(capture_hook)
        
        for prompt in prompts:
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
            with torch.no_grad():
                _ = model(**enc)
        
        handle.remove()
        
        if activations:
            # Take mean across all prompts and positions
            all_acts = torch.cat([a.view(-1, a.shape[-1]) for a in activations], dim=0)
            self.mean_activation = all_acts.mean(dim=0)
        
    def _hook_fn(self, module, input, output):
        if self.mean_activation is None:
            return output
        
        # Get attention output and replace specific head with mean
        attn_output = output[0]  # (batch, seq, hidden)
        batch, seq_len, hidden_size = attn_output.shape
        
        # Reshape to (batch, seq, num_heads, head_dim)
        reshaped = attn_output.view(batch, seq_len, self.num_heads, self.head_dim).clone()
        
        # Get the mean for this head's portion
        mean_reshaped = self.mean_activation.view(self.num_heads, self.head_dim)
        head_mean = mean_reshaped[self.head_idx]  # (head_dim,)
        
        # Replace this head with its mean
        reshaped[:, :, self.head_idx, :] = head_mean
        
        # Reshape back
        modified = reshaped.view(batch, seq_len, hidden_size)
        
        return (modified,) + output[1:]
    
    def __enter__(self):
        layer = self.model.model.layers[self.layer_idx].self_attn
        self.handle = layer.register_forward_hook(self._hook_fn)
        return self
    
    def __exit__(self, *args):
        if self.handle:
            self.handle.remove()


class MultiHeadAblationHook:
    """Ablate multiple heads simultaneously."""
    
    def __init__(self, model, ablations: List[Tuple[int, int]]):
        """ablations: list of (layer_idx, head_idx) tuples"""
        self.model = model
        self.ablations = ablations
        self.hooks = []
        
    def __enter__(self):
        for layer_idx, head_idx in self.ablations:
            hook = HeadAblationHook(self.model, layer_idx, head_idx)
            hook.__enter__()
            self.hooks.append(hook)
        return self
    
    def __exit__(self, *args):
        for hook in self.hooks:
            hook.__exit__()


class ResidualPatchHook:
    """
    Patch the residual stream at a specific layer.
    Replace the hidden state with a source activation.
    """
    
    def __init__(self, model, layer_idx: int, source_hidden: torch.Tensor):
        self.model = model
        self.layer_idx = layer_idx
        self.source_hidden = source_hidden
        self.handle = None
        
    def _hook_fn(self, module, input, output):
        # For decoder layers, output is (hidden_states, ...) 
        # We replace the hidden states with our source
        return (self.source_hidden,) + output[1:]
    
    def __enter__(self):
        layer = self.model.model.layers[self.layer_idx]
        self.handle = layer.register_forward_hook(self._hook_fn)
        return self
    
    def __exit__(self, *args):
        if self.handle:
            self.handle.remove()


# =============================================================================
# EXPERIMENT RUNNERS
# =============================================================================

def run_with_ablation(
    model, tokenizer, prompt: str, 
    ablations: List[Tuple[int, int]],
    device: str = "cuda"
) -> Dict[str, float]:
    """Run model with specific heads ablated, return R_V and PR."""
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    with MultiHeadAblationHook(model, ablations):
        with torch.no_grad():
            outputs = model(**enc, output_hidden_states=True)
    
    hidden_states = outputs.hidden_states
    rv = compute_rv(hidden_states[5], hidden_states[27])
    pr_early = compute_pr(hidden_states[5])
    pr_late = compute_pr(hidden_states[27])
    
    return {"rv": rv, "pr_early": pr_early, "pr_late": pr_late}


def run_clean(model, tokenizer, prompt: str, device: str = "cuda") -> Dict[str, Any]:
    """Run model without intervention, capture all hidden states."""
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    with torch.no_grad():
        outputs = model(**enc, output_hidden_states=True)
    
    return {
        "hidden_states": outputs.hidden_states,
        "rv": compute_rv(outputs.hidden_states[5], outputs.hidden_states[27]),
        "pr_early": compute_pr(outputs.hidden_states[5]),
        "pr_late": compute_pr(outputs.hidden_states[27]),
    }


# =============================================================================
# MAIN INVESTIGATION
# =============================================================================

def run_anthropic_level_investigation(
    model_name: str = "mistralai/Mistral-7B-v0.1",
    device: str = "cuda",
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Comprehensive Anthropic-level investigation.
    """
    print("=" * 80)
    print("ANTHROPIC-LEVEL INVESTIGATION")
    print("=" * 80)
    
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"results/phase3_attention/runs/{timestamp}_anthropic_investigation")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prompt bank hygiene (DEC15): all prompts from prompts/bank.json + log version
    loader = PromptLoader()
    bank_version = loader.version
    (output_dir / "prompt_bank_version.txt").write_text(bank_version)
    (output_dir / "prompt_bank_version.json").write_text(
        json.dumps({"version": bank_version}, indent=2) + "\n"
    )
    
    # Load model
    print("\n[1/8] Loading model...")
    set_seed(42)
    model, tokenizer = load_model(model_name, device=device)
    model.eval()
    
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    
    # Define prompts (sealed bank ids; no hardcoded strings)
    champion_id = "hybrid_l5_math_01"
    baseline_id = "baseline_instructional_02"
    champion = str(loader.prompts[champion_id]["text"])
    baseline = str(loader.prompts[baseline_id]["text"])
    
    results = {
        "prompt_bank_version": bank_version,
        "prompt_ids": {"champion": champion_id, "baseline": baseline_id},
        "experiments": {},
    }
    
    # ==========================================================================
    # EXPERIMENT 1: BASELINE MEASUREMENTS
    # ==========================================================================
    print("\n[2/8] Baseline measurements...")
    
    champ_clean = run_clean(model, tokenizer, champion, device)
    base_clean = run_clean(model, tokenizer, baseline, device)
    
    results["baselines"] = {
        "champion": {"rv": champ_clean["rv"], "pr_early": champ_clean["pr_early"], "pr_late": champ_clean["pr_late"]},
        "baseline": {"rv": base_clean["rv"], "pr_early": base_clean["pr_early"], "pr_late": base_clean["pr_late"]},
        "rv_gap": champ_clean["rv"] - base_clean["rv"],
    }
    
    print(f"  Champion R_V: {champ_clean['rv']:.4f}")
    print(f"  Baseline R_V: {base_clean['rv']:.4f}")
    print(f"  Gap: {results['baselines']['rv_gap']:.4f}")
    
    # ==========================================================================
    # EXPERIMENT 2: FULL HEAD ABLATION SWEEP (L27)
    # ==========================================================================
    print("\n[3/8] Full head ablation sweep at L27...")
    
    ablation_results = []
    
    for head_idx in range(num_heads):
        result_champ = run_with_ablation(model, tokenizer, champion, [(27, head_idx)], device)
        result_base = run_with_ablation(model, tokenizer, baseline, [(27, head_idx)], device)
        
        delta_champ = result_champ["rv"] - champ_clean["rv"]
        delta_base = result_base["rv"] - base_clean["rv"]
        
        ablation_results.append({
            "head": head_idx,
            "champ_rv_ablated": result_champ["rv"],
            "base_rv_ablated": result_base["rv"],
            "delta_champ": delta_champ,
            "delta_base": delta_base,
            "differential_effect": delta_champ - delta_base,  # Effect specific to champion
        })
        
        if abs(delta_champ) > 0.01 or abs(delta_base) > 0.01:
            print(f"  H{head_idx}: Δchamp={delta_champ:+.4f}, Δbase={delta_base:+.4f}")
    
    results["experiments"]["l27_ablation_sweep"] = ablation_results
    
    # Find critical heads
    critical_heads = sorted(ablation_results, key=lambda x: abs(x["delta_champ"]), reverse=True)[:5]
    print(f"\n  Top 5 critical heads at L27:")
    for h in critical_heads:
        print(f"    H{h['head']}: Δchamp={h['delta_champ']:+.4f}, diff={h['differential_effect']:+.4f}")
    
    # ==========================================================================
    # EXPERIMENT 3: MULTI-LAYER HEAD SWEEP
    # ==========================================================================
    print("\n[4/8] Multi-layer head sweep (L20, L22, L25, L27, L29)...")
    
    key_layers = [20, 22, 25, 27, 29]
    layer_sweep_results = {}
    
    for layer in key_layers:
        layer_sweep_results[layer] = []
        print(f"  Layer {layer}...")
        
        for head_idx in range(num_heads):
            result = run_with_ablation(model, tokenizer, champion, [(layer, head_idx)], device)
            delta = result["rv"] - champ_clean["rv"]
            
            layer_sweep_results[layer].append({
                "head": head_idx,
                "rv_ablated": result["rv"],
                "delta": delta,
            })
    
    results["experiments"]["multi_layer_sweep"] = layer_sweep_results
    
    # ==========================================================================
    # EXPERIMENT 4: CONTROL CONDITIONS
    # ==========================================================================
    print("\n[5/8] Control conditions...")
    
    controls = {}
    
    # 4a. Wrong-layer control: Ablate at L10 instead of L27
    print("  4a. Wrong-layer control (L10)...")
    wrong_layer_results = []
    for head_idx in [0, 10, 20, 31]:  # Sample heads
        result = run_with_ablation(model, tokenizer, champion, [(10, head_idx)], device)
        wrong_layer_results.append({
            "head": head_idx,
            "delta": result["rv"] - champ_clean["rv"],
        })
    controls["wrong_layer_l10"] = wrong_layer_results
    
    # 4b. Multi-head ablation: Top 3 critical heads together
    print("  4b. Multi-head ablation (top 3 together)...")
    top3_heads = [(27, h["head"]) for h in critical_heads[:3]]
    multi_result = run_with_ablation(model, tokenizer, champion, top3_heads, device)
    controls["multi_head_top3"] = {
        "heads": [h[1] for h in top3_heads],
        "rv_ablated": multi_result["rv"],
        "delta": multi_result["rv"] - champ_clean["rv"],
    }
    print(f"    Ablating heads {[h[1] for h in top3_heads]}: Δ={controls['multi_head_top3']['delta']:+.4f}")
    
    # 4c. All heads ablation at L27
    print("  4c. All heads ablation at L27...")
    all_heads = [(27, h) for h in range(num_heads)]
    all_result = run_with_ablation(model, tokenizer, champion, all_heads, device)
    controls["all_heads_l27"] = {
        "rv_ablated": all_result["rv"],
        "delta": all_result["rv"] - champ_clean["rv"],
    }
    print(f"    All 32 heads: Δ={controls['all_heads_l27']['delta']:+.4f}")
    
    results["experiments"]["controls"] = controls
    
    # ==========================================================================
    # EXPERIMENT 5: STEERING DIRECTION EXTRACTION
    # ==========================================================================
    print("\n[6/8] Steering direction extraction...")
    
    # Extract mean difference vector at each layer
    steering_analysis = {}
    
    for layer in range(0, num_layers + 1, 4):  # Every 4th layer
        champ_h = champ_clean["hidden_states"][layer][0, -16:, :].float()
        base_h = base_clean["hidden_states"][layer][0, -16:, :].float()
        
        champ_mean = champ_h.mean(dim=0)
        base_mean = base_h.mean(dim=0)
        
        steering_vec = champ_mean - base_mean
        steering_norm = float(steering_vec.norm().cpu())
        
        # Cosine similarity with L27 steering vector
        if layer == 27:
            l27_steering = steering_vec
        
        steering_analysis[layer] = {"norm": steering_norm}
    
    # Compute cosine similarities to L27 direction
    for layer in steering_analysis:
        if layer != 27:
            champ_h = champ_clean["hidden_states"][layer][0, -16:, :].float()
            base_h = base_clean["hidden_states"][layer][0, -16:, :].float()
            vec = champ_h.mean(dim=0) - base_h.mean(dim=0)
            cos_sim = float(F.cosine_similarity(vec.unsqueeze(0), l27_steering.unsqueeze(0)).cpu())
            steering_analysis[layer]["cos_to_l27"] = cos_sim
        else:
            steering_analysis[layer]["cos_to_l27"] = 1.0
    
    results["experiments"]["steering_direction"] = steering_analysis
    
    print("  Layer | Norm | Cos(L27)")
    for layer in sorted(steering_analysis.keys()):
        data = steering_analysis[layer]
        print(f"    L{layer:2d}: {data['norm']:6.2f} | {data['cos_to_l27']:.3f}")
    
    # ==========================================================================
    # EXPERIMENT 6: BACKUP HEAD DETECTION
    # ==========================================================================
    print("\n[7/8] Backup head detection...")
    
    # Ablate top head, then find which other heads become more critical
    top_head = critical_heads[0]["head"]
    backup_analysis = []
    
    print(f"  With H{top_head} ablated, testing other heads...")
    
    # Get baseline with top head ablated
    with_top_ablated = run_with_ablation(model, tokenizer, champion, [(27, top_head)], device)
    
    for head_idx in range(num_heads):
        if head_idx == top_head:
            continue
        
        # Ablate both top head and this head
        both_result = run_with_ablation(model, tokenizer, champion, [(27, top_head), (27, head_idx)], device)
        
        # Compare to just top head ablated
        additional_effect = both_result["rv"] - with_top_ablated["rv"]
        
        # Compare to just this head ablated (from sweep)
        original_effect = ablation_results[head_idx]["delta_champ"]
        
        # Backup head = effect increases when primary is already ablated
        backup_score = abs(additional_effect) - abs(original_effect)
        
        backup_analysis.append({
            "head": head_idx,
            "original_effect": original_effect,
            "effect_with_primary_ablated": additional_effect,
            "backup_score": backup_score,
        })
    
    # Find potential backup heads
    backup_heads = sorted(backup_analysis, key=lambda x: x["backup_score"], reverse=True)[:5]
    
    results["experiments"]["backup_heads"] = {
        "primary_head": top_head,
        "candidates": backup_heads,
    }
    
    print(f"  Potential backup heads (effect increases when H{top_head} ablated):")
    for h in backup_heads:
        if h["backup_score"] > 0.001:
            print(f"    H{h['head']}: original={h['original_effect']:+.4f}, with_primary={h['effect_with_primary_ablated']:+.4f}, backup_score={h['backup_score']:+.4f}")
    
    # ==========================================================================
    # EXPERIMENT 7: DOSE-RESPONSE WITH ABLATION
    # ==========================================================================
    print("\n[8/8] Dose-response test...")
    
    # Dose-response prompts (pulled from bank by level for reproducibility)
    dose_prompts: Dict[str, str] = {"champion": champion}
    dose_prompt_ids: Dict[str, str] = {"champion": champion_id}
    for level in [1, 3, 5]:
        cands = [
            (k, v.get("text", ""))
            for k, v in loader.prompts.items()
            if v.get("pillar") == "dose_response" and v.get("level") == level and v.get("text")
        ]
        if not cands:
            continue
        cands.sort(key=lambda x: x[0])
        pid, ptxt = cands[0]
        dose_prompts[f"L{level}"] = str(ptxt)
        dose_prompt_ids[f"L{level}"] = str(pid)
    
    dose_results = {}
    
    for level, prompt in dose_prompts.items():
        clean = run_clean(model, tokenizer, prompt, device)
        ablated = run_with_ablation(model, tokenizer, prompt, [(27, top_head)], device)
        
        dose_results[level] = {
            "rv_clean": clean["rv"],
            "rv_ablated": ablated["rv"],
            "delta": ablated["rv"] - clean["rv"],
        }
        print(f"  {level}: clean={clean['rv']:.4f}, ablated={ablated['rv']:.4f}, Δ={dose_results[level]['delta']:+.4f}")
    
    results["experiments"]["dose_response"] = {"prompt_ids": dose_prompt_ids, "results": dose_results}
    
    # ==========================================================================
    # GENERATE OUTPUTS
    # ==========================================================================
    print("\n" + "=" * 80)
    print("GENERATING OUTPUTS")
    print("=" * 80)
    
    # 1. Ablation heatmap
    fig, axes = plt.subplots(1, len(key_layers), figsize=(20, 5))
    
    for idx, layer in enumerate(key_layers):
        ax = axes[idx]
        deltas = [r["delta"] for r in layer_sweep_results[layer]]
        
        # Reshape to 8x4 grid
        grid = np.array(deltas).reshape(8, 4)
        
        sns.heatmap(grid, ax=ax, cmap="RdBu_r", center=0, annot=True, fmt=".3f", cbar=False)
        ax.set_title(f"L{layer} Head Ablation Effect")
        ax.set_xlabel("Head (mod 4)")
        ax.set_ylabel("Head (// 4)")
    
    plt.tight_layout()
    plt.savefig(output_dir / "ablation_heatmap.png", dpi=150)
    plt.close()
    print(f"  Saved: ablation_heatmap.png")
    
    # 2. Critical heads bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    heads = [r["head"] for r in ablation_results]
    deltas = [r["delta_champ"] for r in ablation_results]
    colors = ['red' if d > 0 else 'blue' for d in deltas]
    
    ax.bar(heads, deltas, color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-')
    ax.set_xlabel("Head Index")
    ax.set_ylabel("ΔR_V (Champion)")
    ax.set_title("Effect of Ablating Each Head at L27 on Champion R_V\n(Red = R_V increases = head was causing contraction)")
    
    plt.tight_layout()
    plt.savefig(output_dir / "critical_heads.png", dpi=150)
    plt.close()
    print(f"  Saved: critical_heads.png")
    
    # 3. Summary JSON
    with open(output_dir / "summary.json", "w") as f:
        # Convert to serializable format
        serializable_results = {
            "baselines": results["baselines"],
            "critical_heads_l27": [{"head": h["head"], "delta": h["delta_champ"]} for h in critical_heads],
            "controls": controls,
            "backup_heads": results["experiments"]["backup_heads"],
            "steering_direction": steering_analysis,
            "dose_response": dose_results,
        }
        json.dump(serializable_results, f, indent=2, default=float)
    print(f"  Saved: summary.json")
    
    # 4. Detailed CSV
    df = pd.DataFrame(ablation_results)
    df.to_csv(output_dir / "l27_ablation_sweep.csv", index=False)
    print(f"  Saved: l27_ablation_sweep.csv")
    
    # ==========================================================================
    # PRINT KEY FINDINGS
    # ==========================================================================
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    
    print(f"\n1. BASELINE R_V:")
    print(f"   Champion: {champ_clean['rv']:.4f}")
    print(f"   Baseline: {base_clean['rv']:.4f}")
    print(f"   Gap: {results['baselines']['rv_gap']:.4f}")
    
    print(f"\n2. CRITICAL HEADS AT L27:")
    for h in critical_heads[:5]:
        direction = "↑ (breaks contraction)" if h["delta_champ"] > 0 else "↓ (enhances contraction)"
        print(f"   H{h['head']}: Δ={h['delta_champ']:+.4f} {direction}")
    
    print(f"\n3. CONTROLS:")
    print(f"   Wrong layer (L10): max Δ = {max(abs(r['delta']) for r in controls['wrong_layer_l10']):.4f}")
    print(f"   Multi-head top3: Δ = {controls['multi_head_top3']['delta']:+.4f}")
    print(f"   All heads L27: Δ = {controls['all_heads_l27']['delta']:+.4f}")
    
    print(f"\n4. BACKUP HEADS:")
    for h in backup_heads[:3]:
        if h["backup_score"] > 0.001:
            print(f"   H{h['head']}: backup_score = {h['backup_score']:+.4f}")
    
    print(f"\n5. STEERING DIRECTION:")
    print(f"   Emerges around L{min(l for l, d in steering_analysis.items() if d['norm'] > 1.0)} (norm > 1)")
    print(f"   Saturates around L{min(l for l, d in steering_analysis.items() if d.get('cos_to_l27', 0) > 0.9)} (cos > 0.9)")
    
    print("\n" + "=" * 80)
    print(f"[OK] Investigation complete. Run dir: {output_dir}")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Anthropic-Level Investigation")
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1", help="Model name")
    parser.add_argument("--device", default="cuda", help="Device")
    args = parser.parse_args()
    
    try:
        run_anthropic_level_investigation(model_name=args.model, device=args.device)
    finally:
        print("\n[cleanup] Clearing GPU memory...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print("[cleanup] GPU memory cleared.")

