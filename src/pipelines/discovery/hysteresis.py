"""
Hysteresis Pipeline (Gold Standard Pipeline 7)

Tests for "One-Way Door" dynamics / Attractor Depth.
Does it require more energy (intervention strength) to exit the recursive state than to enter it?

Implements the Asymmetry metric:
Asymmetry = Efficiency(Base->Rec) - Efficiency(Rec->Base)

Positive asymmetry indicates the recursive state is an attractor (harder to leave).
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from prompts.loader import PromptLoader
from src.core.models import load_model, set_seed
from src.core.hooks import capture_v_projection
from src.metrics.rv import participation_ratio
from src.pipelines.registry import ExperimentResult


@dataclass
class HysteresisPair:
    pair_idx: int
    rec_text: str
    base_text: str
    rv_base: float
    rv_rec: float


def _extract_residual(
    model,
    tokenizer,
    text: str,
    layer_idx: int,
    device: str,
) -> Optional[torch.Tensor]:
    """Extract residual stream output at specific layer."""
    inputs = tokenizer(text, return_tensors="pt").to(device)
    storage = {"resid": None}
    
    def hook(module, inp, out):
        if isinstance(out, tuple):
            storage["resid"] = out[0].detach().clone()
        else:
            storage["resid"] = out.detach().clone()
        return out
        
    handle = model.model.layers[layer_idx].register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(**inputs)
    except Exception:
        return None
    finally:
        handle.remove()
        
    return storage["resid"]


def _patch_and_measure(
    model,
    tokenizer,
    target_text: str,
    patch_resid: torch.Tensor,
    layer_idx: int,
    window: int,
    early_layer: int,
    late_layer: int,
    device: str,
) -> float:
    """
    Patch residual into target text at layer_idx, measure R_V.
    Uses forward_pre_hook on the layer to inject the patch into the stream.
    """
    inputs = tokenizer(target_text, return_tensors="pt").to(device)
    
    def pre_hook(module, args):
        hidden_states = args[0] # (B, T, D)
        
        # Align dimensions
        if patch_resid.dim() == 2:
            patch = patch_resid.unsqueeze(0)
        else:
            patch = patch_resid
            
        B, T, D = hidden_states.shape
        T_patch = patch.shape[1]
        
        # Patch last W tokens (min of available)
        W = min(window, T, T_patch)
        if W <= 0:
            return args
            
        # Inject patch
        patched = hidden_states.clone()
        patched[:, -W:, :] = patch[:, -W:, :].to(hidden_states.device)
        
        return (patched,) + args[1:]
        
    handle = model.model.layers[layer_idx].register_forward_pre_hook(pre_hook)
    
    try:
        # Capture V for R_V measurement
        v_early = None
        v_late = None
        
        with capture_v_projection(model, early_layer) as store_early:
            with capture_v_projection(model, late_layer) as store_late:
                with torch.no_grad():
                    model(**inputs)
        
        v_early = store_early.get("v")
        v_late = store_late.get("v")
        
        if v_early is None or v_late is None:
            return float('nan')
            
        if v_early.dim() == 3: v_early = v_early[0]
        if v_late.dim() == 3: v_late = v_late[0]
        
        pr_early = participation_ratio(v_early, window)
        pr_late = participation_ratio(v_late, window)
        
        if pr_early == 0 or np.isnan(pr_early) or np.isnan(pr_late):
            return float('nan')
            
        return float(pr_late / pr_early)
        
    finally:
        handle.remove()


def run_hysteresis_from_config(cfg: Dict[str, Any], run_dir: Path) -> ExperimentResult:
    """Run hysteresis / one-way door validation."""
    model_cfg = cfg.get("model") or {}
    params = cfg.get("params") or {}
    
    seed = int(cfg.get("seed") or 42)
    device = str(model_cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_name = str(model_cfg.get("name") or "mistralai/Mistral-7B-v0.1")
    
    n_pairs = int(params.get("n_pairs") or 50)
    window = int(params.get("window") or 16)
    early_layer = int(params.get("early_layer") or 5)
    late_layer = int(params.get("late_layer") or 27)
    test_layers = params.get("test_layers") or [24, 26, 28]
    
    set_seed(seed)
    model, tokenizer = load_model(model_name, device=device)
    
    loader = PromptLoader()
    bank_version = loader.version
    (run_dir / "prompt_bank_version.txt").write_text(bank_version)
    (run_dir / "prompt_bank_version.json").write_text(
        json.dumps({"version": bank_version}, indent=2) + "\n"
    )
    
    # Get pairs
    # Use simple balanced pairs for now
    recursive = loader.get_by_pillar("dose_response", limit=n_pairs, seed=seed)
    baseline = loader.get_by_pillar("baselines", limit=n_pairs, seed=seed)
    
    # Prune shorts
    pairs = []
    for r, b in zip(recursive, baseline):
        if len(tokenizer.encode(r)) >= window and len(tokenizer.encode(b)) >= window:
            pairs.append((r, b))
            
    # Pre-compute baselines
    pair_data = []
    print(f"Computing baselines for {len(pairs)} pairs...")
    for i, (rec, base) in enumerate(tqdm(pairs)):
        # Measure R_V for rec - pass a dummy patch (zeros) and layer_idx=0 (valid)
        # But we need to ensure the pre_hook handles window=0 or similar to do nothing
        # Actually, _patch_and_measure is designed to patch.
        # We should just use compute_rv from src.metrics.rv for baselines.
        
        from src.metrics.rv import compute_rv
        rv_rec = compute_rv(model, tokenizer, rec, early_layer, late_layer, window, device)
        rv_base = compute_rv(model, tokenizer, base, early_layer, late_layer, window, device)
        
        pair_data.append(HysteresisPair(i, rec, base, rv_base, rv_rec))
        
    results = []
    
    print(f"Testing hysteresis on {len(test_layers)} layers...")
    for layer in test_layers:
        for p in tqdm(pair_data, desc=f"Layer {layer}"):
            if np.isnan(p.rv_base) or np.isnan(p.rv_rec):
                continue
                
            gap = p.rv_base - p.rv_rec
            if abs(gap) < 0.05: # Skip small gaps
                continue
                
            # 1. Forward: Base + Rec_Resid -> ? (Should contract)
            rec_resid = _extract_residual(model, tokenizer, p.rec_text, layer, device)
            if rec_resid is None: continue
            
            rv_forward = _patch_and_measure(
                model, tokenizer, p.base_text, rec_resid, layer, window, early_layer, late_layer, device
            )
            
            # 2. Reverse: Rec + Base_Resid -> ? (Should NOT expand back fully)
            base_resid = _extract_residual(model, tokenizer, p.base_text, layer, device)
            if base_resid is None: continue
            
            rv_reverse = _patch_and_measure(
                model, tokenizer, p.rec_text, base_resid, layer, window, early_layer, late_layer, device
            )
            
            # Efficiencies (0 to 1 scale)
            # Forward: How much did we close the gap? (Base -> Rec)
            eff_fwd = (p.rv_base - rv_forward) / gap
            
            # Reverse: How much did we re-open the gap? (Rec -> Base)
            eff_rev = (rv_reverse - p.rv_rec) / gap
            
            # Asymmetry > 0 means Fwd > Rev (Easier to enter than leave)
            asymmetry = eff_fwd - eff_rev
            
            results.append({
                "layer": layer,
                "pair_idx": p.pair_idx,
                "rv_base": p.rv_base,
                "rv_rec": p.rv_rec,
                "rv_forward": rv_forward,
                "rv_reverse": rv_reverse,
                "eff_fwd": eff_fwd,
                "eff_rev": eff_rev,
                "asymmetry": asymmetry
            })
            
    # Save results
    out_csv = run_dir / "hysteresis_results.csv"
    with open(out_csv, "w", newline="") as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
            
    # Summary
    summary = {
        "experiment": "hysteresis",
        "model_name": model_name,
        "n_pairs": len(pairs),
        "prompt_bank_version": bank_version,
        "layers": {}
    }
    
    for layer in test_layers:
        layer_res = [r for r in results if r["layer"] == layer]
        if not layer_res: continue
        
        asyms = [r["asymmetry"] for r in layer_res if not np.isnan(r["asymmetry"])]
        
        summary["layers"][str(layer)] = {
            "mean_asymmetry": float(np.mean(asyms)),
            "std_asymmetry": float(np.std(asyms)),
            "n": len(asyms),
            "is_attractor": float(np.mean(asyms)) > 0.1 # Threshold for "significant"
        }
        
    summary["prompt_bank_version"] = bank_version
    return ExperimentResult(summary=summary)

