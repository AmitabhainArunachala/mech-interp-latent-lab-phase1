"""
Position-Specific MLP Ablation: Test which token positions drive L0 effect.

Tests:
- BOS only (position 0)
- First 4 tokens (positions 0-3)
- Last 16 tokens (matches R_V window)
- All tokens (current baseline)

Goal: Determine if L0 effect is BOS-driven, token-distributed, or diffuse.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import stats
from tqdm import tqdm

from prompts.loader import PromptLoader
from src.core.models import load_model, set_seed
from src.metrics.behavior_strict import score_behavior_strict
from src.metrics.mode_score import ModeScoreMetric
from src.metrics.rv import compute_rv
from src.pipelines.registry import ExperimentResult


class PositionSpecificMLPAblationHook:
    """Zero out MLP output at specified layer and token positions."""
    
    def __init__(self, model, layer_idx: int, positions: Optional[List[int]] = None):
        """
        Args:
            model: The model
            layer_idx: Layer to ablate
            positions: List of token positions to zero. None = all positions.
                       Examples:
                       - [0] = BOS only
                       - [0, 1, 2, 3] = First 4 tokens
                       - [-16, -15, ..., -1] = Last 16 tokens
                       - None = All positions
        """
        self.model = model
        self.layer_idx = layer_idx
        self.positions = positions  # None = all, list = specific positions
        self.handle: Optional[torch.utils.hooks.RemovableHandle] = None
    
    def register(self):
        """Register forward hook to zero MLP output at specified positions."""
        if self.handle is not None:
            raise RuntimeError("Hook already registered. Call remove() first.")
        
        mlp = self.model.model.layers[self.layer_idx].mlp
        
        def hook_fn(module, inp, out):
            """Zero out MLP output at specified positions."""
            if isinstance(out, tuple):
                out_tensor = out[0]
            else:
                out_tensor = out
            
            # Clone to avoid modifying in-place
            out_ablated = out_tensor.clone()
            
            if self.positions is None:
                # Zero all positions
                out_ablated.zero_()
            else:
                # Zero only specified positions
                batch_size, seq_len, hidden_dim = out_ablated.shape
                for pos in self.positions:
                    # Handle negative indices (from end)
                    if pos < 0:
                        pos = seq_len + pos
                    if 0 <= pos < seq_len:
                        out_ablated[:, pos, :] = 0
            
            if isinstance(out, tuple):
                return (out_ablated,) + out[1:]
            return out_ablated
        
        self.handle = mlp.register_forward_hook(hook_fn)
    
    def remove(self):
        """Remove the forward hook."""
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
    
    def __enter__(self):
        self.register()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove()


def run_position_specific_ablation_from_config(cfg: Dict[str, Any], run_dir: Path) -> ExperimentResult:
    """Run position-specific MLP ablation test."""
    params = cfg.get("params", {})
    model_name = params.get("model", "mistralai/Mistral-7B-v0.1")
    layer_idx = params.get("layer", 0)
    n_pairs = params.get("n_pairs", 30)
    window_size = params.get("window_size", 16)
    max_new_tokens = params.get("max_new_tokens", 200)
    seed = int(params.get("seed", 42))
    
    # Position configurations to test
    position_configs = {
        "bos_only": [0],  # BOS only
        "first_4": [0, 1, 2, 3],  # First 4 tokens
        "last_16": list(range(-16, 0)),  # Last 16 tokens (negative indices)
        "all_tokens": None,  # All tokens (baseline)
    }
    
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model: {model_name}")
    model, tokenizer = load_model(model_name, device=device)
    model.eval()
    
    loader = PromptLoader()
    bank_version = loader.version
    (run_dir / "prompt_bank_version.txt").write_text(bank_version)
    (run_dir / "prompt_bank_version.json").write_text(json.dumps({"version": bank_version}, indent=2) + "\n")
    mode_metric = ModeScoreMetric(tokenizer, device)
    
    # Get prompt pairs
    pairs = loader.get_balanced_pairs(n_pairs=n_pairs, seed=seed)
    
    print(f"\n{'='*60}")
    print(f"POSITION-SPECIFIC L{layer_idx} MLP ABLATION TEST")
    print(f"{'='*60}")
    print(f"Layer: L{layer_idx}")
    print(f"Pairs: {len(pairs)}")
    print(f"Position configs: {list(position_configs.keys())}")
    print(f"{'='*60}\n")
    
    results = []
    
    for pair_idx, (rec_text, base_text) in enumerate(tqdm(pairs, desc="Testing pairs")):
        # Get baseline prompt for comparison
        inputs_base = tokenizer(base_text, return_tensors="pt", add_special_tokens=False).to(device)
        with torch.no_grad():
            out_base = model(**inputs_base)
        
        inputs_rec = tokenizer(rec_text, return_tensors="pt", add_special_tokens=False).to(device)
        
        # BASELINE: Generate text without ablation
        with torch.no_grad():
            inputs_gen_baseline = tokenizer(rec_text, return_tensors="pt", add_special_tokens=False).to(device)
            outputs_baseline = model.generate(
                **inputs_gen_baseline,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text_baseline = tokenizer.decode(outputs_baseline[0], skip_special_tokens=True)
        
        # Compute baseline R_V
        rv_rec_baseline = compute_rv(model, tokenizer, generated_text_baseline, early=5, late=27, window=window_size, device=device)
        
        # Compute baseline mode score
        with torch.no_grad():
            out_rec_baseline = model(**inputs_rec)
        
        try:
            mode_rec_baseline = mode_metric.compute_score(out_rec_baseline.logits, baseline_logits=out_base.logits)
        except Exception as e:
            print(f"  Error computing baseline mode score: {e}")
            mode_rec_baseline = float("nan")
        
        # Test each position configuration
        for config_name, positions in position_configs.items():
            ablation_hook = PositionSpecificMLPAblationHook(model, layer_idx, positions)
            
            try:
                # Generate text with ablation
                with ablation_hook:
                    with torch.no_grad():
                        inputs_gen = tokenizer(rec_text, return_tensors="pt", add_special_tokens=False).to(device)
                        outputs_ablated = model.generate(
                            **inputs_gen,
                            max_new_tokens=max_new_tokens,
                            temperature=0.0,
                            do_sample=False,
                            pad_token_id=tokenizer.eos_token_id
                        )
                
                generated_text = tokenizer.decode(outputs_ablated[0], skip_special_tokens=True)
                
                # Compute R_V with ablation
                rv_rec_ablated = compute_rv(model, tokenizer, generated_text, early=5, late=27, window=window_size, device=device)
                
                # Compute mode score with ablation
                with ablation_hook:
                    with torch.no_grad():
                        out_rec_ablated = model(**inputs_rec)
                
                try:
                    mode_rec_ablated = mode_metric.compute_score(out_rec_ablated.logits, baseline_logits=out_base.logits)
                except Exception as e:
                    print(f"  Error computing ablated mode score ({config_name}): {e}")
                    mode_rec_ablated = float("nan")
                
                # Compute coherence
                try:
                    behavior_score = score_behavior_strict(generated_text)
                    coherence = behavior_score.coherence_score
                    recursion_score = behavior_score.recursion_score
                except Exception as e:
                    print(f"  Error computing behavior score ({config_name}): {e}")
                    coherence = 0.0
                    recursion_score = 0.0
            
            except Exception as e:
                print(f"  Error during ablation ({config_name}) at pair {pair_idx}: {e}")
                rv_rec_ablated = float("nan")
                mode_rec_ablated = float("nan")
                generated_text = ""
                coherence = 0.0
                recursion_score = 0.0
            
            # Compute deltas
            rv_delta = rv_rec_ablated - rv_rec_baseline
            mode_delta = mode_rec_ablated - mode_rec_baseline if not np.isnan(mode_rec_ablated) else float("nan")
            
            results.append({
                "pair_idx": pair_idx,
                "recursive_text": rec_text,
                "baseline_text": base_text,
                "position_config": config_name,
                "positions": str(positions) if positions else "all",
                "generated_text_baseline": generated_text_baseline,
                "generated_text_ablated": generated_text,
                "layer": layer_idx,
                "rv_baseline": rv_rec_baseline,
                "rv_ablated": rv_rec_ablated,
                "rv_delta": rv_delta,
                "mode_baseline": mode_rec_baseline,
                "mode_ablated": mode_rec_ablated,
                "mode_delta": mode_delta,
                "coherence": coherence,
                "recursion_score": recursion_score,
            })
    
    # Save results
    df = pd.DataFrame(results)
    csv_path = run_dir / "position_specific_ablation.csv"
    df.to_csv(csv_path, index=False)
    
    # Statistical analysis by position config
    summary_by_config = {}
    
    for config_name in position_configs.keys():
        config_df = df[df["position_config"] == config_name]
        
        rv_deltas = config_df["rv_delta"].dropna().values
        mode_deltas = config_df["mode_delta"].dropna().values
        
        # One-sample t-test
        rv_stat = None
        rv_pvalue = None
        rv_significant = None
        if len(rv_deltas) >= 3:
            t_stat, p_val = stats.ttest_1samp(rv_deltas, 0.0)
            rv_stat = float(t_stat)
            rv_pvalue = float(p_val)
            rv_significant = p_val < 0.01
        
        mode_stat = None
        mode_pvalue = None
        mode_significant = None
        if len(mode_deltas) >= 3:
            t_stat, p_val = stats.ttest_1samp(mode_deltas, 0.0)
            mode_stat = float(t_stat)
            mode_pvalue = float(p_val)
            mode_significant = p_val < 0.01
        
        # Effect sizes
        rv_cohens_d = float(np.mean(rv_deltas) / np.std(rv_deltas)) if len(rv_deltas) >= 2 and np.std(rv_deltas) > 0 else None
        mode_cohens_d = float(np.mean(mode_deltas) / np.std(mode_deltas)) if len(mode_deltas) >= 2 and np.std(mode_deltas) > 0 else None
        
        summary_by_config[config_name] = {
            "rv_delta_mean": float(config_df["rv_delta"].mean()),
            "rv_delta_std": float(config_df["rv_delta"].std()),
            "rv_t_statistic": rv_stat,
            "rv_pvalue": rv_pvalue,
            "rv_significant": rv_significant,
            "rv_cohens_d": rv_cohens_d,
            "mode_delta_mean": float(config_df["mode_delta"].mean()) if config_df["mode_delta"].notna().any() else None,
            "mode_t_statistic": mode_stat,
            "mode_pvalue": mode_pvalue,
            "mode_significant": mode_significant,
            "mode_cohens_d": mode_cohens_d,
        }
    
    # Overall summary
    summary = {
        "experiment": "position_specific_ablation",
        "layer": layer_idx,
        "n_pairs": len(pairs),
        "position_configs": summary_by_config,
    }
    
    summary_path = run_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print("RESULTS BY POSITION CONFIG")
    print(f"{'='*60}")
    for config_name, config_summary in summary_by_config.items():
        print(f"\n{config_name.upper()}:")
        print(f"  R_V delta: {config_summary['rv_delta_mean']:.4f} ± {config_summary['rv_delta_std']:.4f}")
        if config_summary['rv_significant']:
            print(f"    p-value: {config_summary['rv_pvalue']:.4e} (SIGNIFICANT)")
        print(f"  Mode delta: {config_summary['mode_delta_mean']:.4f}" if config_summary['mode_delta_mean'] is not None else "  Mode delta: N/A")
    
    print(f"\n✅ Results saved to: {csv_path}")
    print(f"✅ Summary saved to: {summary_path}")
    
    return ExperimentResult(summary=summary)

