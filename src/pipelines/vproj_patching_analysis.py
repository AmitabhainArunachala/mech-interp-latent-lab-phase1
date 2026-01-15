"""
V_proj Patching Analysis Pipeline.

Patches V_proj from recursive prompt into baseline prompt during generation,
then analyzes semantic domain shifts in the generated outputs.

Key finding from Dec 2025: V_proj patching at L27 produces dramatic behavioral
shifts (task prompts → philosophical outputs).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from tqdm import tqdm

from prompts.loader import PromptLoader
from src.core.models import load_model, set_seed
from src.core.patching import PersistentVPatcher, extract_v_activation
from src.metrics.rv import compute_rv
from src.pipelines.registry import ExperimentResult
from src.utils.run_metadata import get_run_metadata, save_metadata, append_to_run_index


# Domain classification keywords
PHILOSOPHICAL_KEYWORDS = [
    "consciousness", "awareness", "self", "itself", "observer", "witness",
    "process", "being", "emptiness", "unity", "non-dual", "presence",
    "realization", "awakening", "meditation", "mindfulness", "essence",
    "transcendence", "void", "nothingness", "suchness", "thusness",
]

TASK_KEYWORDS = [
    "calculate", "answer", "result", "equation", "solve", "compute",
    "solution", "equals", "sum", "product", "total", "number",
]

NARRATIVE_KEYWORDS = [
    "story", "character", "said", "went", "happened", "told",
    "narrator", "protagonist", "scene", "dialogue", "plot",
]


def analyze_generation_domain(text: str) -> Dict[str, Any]:
    """
    Classify semantic domain of generated text.
    
    Returns:
        Dict with domain classification and marker counts
    """
    text_lower = text.lower()
    
    # Count domain markers
    phil_count = sum(1 for w in PHILOSOPHICAL_KEYWORDS if w in text_lower)
    task_count = sum(1 for w in TASK_KEYWORDS if w in text_lower)
    narr_count = sum(1 for w in NARRATIVE_KEYWORDS if w in text_lower)
    
    # Classify primary domain
    if phil_count > task_count and phil_count > narr_count:
        domain = "philosophical"
    elif task_count > narr_count:
        domain = "task"
    elif narr_count > 0:
        domain = "narrative"
    else:
        domain = "other"
    
    return {
        "domain": domain,
        "philosophical_markers": phil_count,
        "task_markers": task_count,
        "narrative_markers": narr_count,
        "total_markers": phil_count + task_count + narr_count,
    }


def run_vproj_patching_analysis_from_config(cfg: Dict[str, Any], run_dir: Path) -> ExperimentResult:
    """
    Run V_proj patching analysis with full generation.
    
    For each prompt pair:
    1. Extract V_proj from recursive prompt
    2. Generate baseline output (control)
    3. Generate transfer output (with V_proj patching)
    4. Generate recursive output (control)
    5. Analyze domain shifts
    6. Compute R_V on generated texts
    """
    model_cfg = cfg.get("model", {})
    params = cfg.get("params", {})
    
    seed = int(cfg.get("seed", 42))
    device = str(model_cfg.get("device", "cuda"))
    model_name = str(model_cfg.get("name", "mistralai/Mistral-7B-v0.1"))
    
    n_pairs = int(params.get("n_pairs", 20))
    patch_layer = int(params.get("patch_layer", 27))
    max_new_tokens = int(params.get("max_new_tokens", 100))
    temperature = float(params.get("temperature", 0.7))
    
    set_seed(seed)
    print(f"Loading model: {model_name}")
    model, tokenizer = load_model(model_name, device=device)
    model.eval()
    
    # Load prompts
    loader = PromptLoader()
    bank_version = loader.version
    (run_dir / "prompt_bank_version.txt").write_text(bank_version)
    
    pairs_with_ids = loader.get_balanced_pairs_with_ids(n_pairs=n_pairs, seed=seed)
    
    results = []
    
    for rec_id, base_id, rec_text, base_text in tqdm(pairs_with_ids, desc="V_proj patching"):
        
        try:
            # === Step 1: Extract V_proj from recursive prompt ===
            v_activation = extract_v_activation(
                model, tokenizer, rec_text, layer_idx=patch_layer, device=device
            )
            
            # === Step 2: Generate baseline output (control) ===
            base_inputs = tokenizer(base_text, return_tensors="pt", add_special_tokens=False).to(device)
            with torch.no_grad():
                base_outputs = model.generate(
                    **base_inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0.0,
                    pad_token_id=tokenizer.eos_token_id,
                )
            baseline_text = tokenizer.decode(base_outputs[0], skip_special_tokens=True)
            baseline_domain = analyze_generation_domain(baseline_text)
            
            # === Step 3: Generate transfer output (with V_proj patching) ===
            # Re-tokenize for fresh generation
            transfer_inputs = tokenizer(base_text, return_tensors="pt", add_special_tokens=False).to(device)
            
            # Create patcher and generate with it active
            patcher = PersistentVPatcher(model, v_activation)
            patcher.register(layer_idx=patch_layer)
            
            try:
                with torch.no_grad():
                    transfer_outputs = model.generate(
                        **transfer_inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        do_sample=temperature > 0.0,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                transfer_text = tokenizer.decode(transfer_outputs[0], skip_special_tokens=True)
                transfer_domain = analyze_generation_domain(transfer_text)
            finally:
                patcher.remove()
            
            # === Step 4: Generate recursive output (control) ===
            rec_inputs = tokenizer(rec_text, return_tensors="pt", add_special_tokens=False).to(device)
            with torch.no_grad():
                rec_outputs = model.generate(
                    **rec_inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0.0,
                    pad_token_id=tokenizer.eos_token_id,
                )
            recursive_text = tokenizer.decode(rec_outputs[0], skip_special_tokens=True)
            recursive_domain = analyze_generation_domain(recursive_text)
            
            # === Step 5: Compute R_V on generated texts ===
            rv_baseline = compute_rv(model, tokenizer, baseline_text, early=5, late=27, window=16, device=device)
            rv_recursive = compute_rv(model, tokenizer, recursive_text, early=5, late=27, window=16, device=device)
            rv_transfer = compute_rv(model, tokenizer, transfer_text, early=5, late=27, window=16, device=device)
            
            # === Step 6: Analyze domain shift ===
            domain_shift = baseline_domain["domain"] != transfer_domain["domain"]
            shift_to_philosophical = (
                baseline_domain["domain"] != "philosophical" and
                transfer_domain["domain"] == "philosophical"
            )
            
            # Compile results
            row = {
                "recursive_prompt_id": rec_id,
                "baseline_prompt_id": base_id,
                
                # Generated texts
                "baseline_text": baseline_text,
                "transfer_text": transfer_text,
                "recursive_text": recursive_text,
                
                # Domain analysis
                "baseline_domain": baseline_domain["domain"],
                "transfer_domain": transfer_domain["domain"],
                "recursive_domain": recursive_domain["domain"],
                
                "baseline_phil_markers": baseline_domain["philosophical_markers"],
                "transfer_phil_markers": transfer_domain["philosophical_markers"],
                "recursive_phil_markers": recursive_domain["philosophical_markers"],
                
                "baseline_task_markers": baseline_domain["task_markers"],
                "transfer_task_markers": transfer_domain["task_markers"],
                "recursive_task_markers": recursive_domain["task_markers"],
                
                # Domain shift metrics
                "domain_shift": domain_shift,
                "shift_to_philosophical": shift_to_philosophical,
                "phil_marker_increase": transfer_domain["philosophical_markers"] - baseline_domain["philosophical_markers"],
                
                # R_V metrics
                "rv_baseline": rv_baseline,
                "rv_recursive": rv_recursive,
                "rv_transfer": rv_transfer,
                "rv_transfer_delta": rv_transfer - rv_baseline,
                "rv_transfer_restoration": (rv_transfer - rv_baseline) / (rv_recursive - rv_baseline) if (rv_recursive - rv_baseline) != 0 else None,
            }
            
            results.append(row)
            
        except Exception as e:
            print(f"Error processing pair {rec_id}/{base_id}: {e}")
            continue
        
        # Clear cache periodically
        if len(results) % 5 == 0:
            torch.cuda.empty_cache()
    
    # Save CSV
    df = pd.DataFrame(results)
    csv_path = run_dir / "vproj_patching_analysis.csv"
    df.to_csv(csv_path, index=False)
    
    # Compute summary statistics
    summary = {
        "experiment": "vproj_patching_analysis",
        "n_pairs": len(df),
        "patch_layer": patch_layer,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        
        # Domain shift rates
        "domain_shift_rate": float(df["domain_shift"].mean()) * 100.0,
        "shift_to_philosophical_rate": float(df["shift_to_philosophical"].mean()) * 100.0,
        
        # Domain distribution
        "baseline_domain_distribution": df["baseline_domain"].value_counts().to_dict(),
        "transfer_domain_distribution": df["transfer_domain"].value_counts().to_dict(),
        "recursive_domain_distribution": df["recursive_domain"].value_counts().to_dict(),
        
        # Philosophical markers
        "baseline_phil_markers_mean": float(df["baseline_phil_markers"].mean()),
        "transfer_phil_markers_mean": float(df["transfer_phil_markers"].mean()),
        "recursive_phil_markers_mean": float(df["recursive_phil_markers"].mean()),
        "phil_marker_increase_mean": float(df["phil_marker_increase"].mean()),
        
        # R_V metrics
        "rv_baseline_mean": float(df["rv_baseline"].mean()),
        "rv_recursive_mean": float(df["rv_recursive"].mean()),
        "rv_transfer_mean": float(df["rv_transfer"].mean()),
        "rv_transfer_delta_mean": float(df["rv_transfer_delta"].mean()),
        "rv_transfer_restoration_mean": float(df["rv_transfer_restoration"].dropna().mean()) if df["rv_transfer_restoration"].notna().any() else None,
        
        "prompt_bank_version": bank_version,
        "artifacts": {"csv": str(csv_path)},
    }
    
    # Save summary
    summary_path = run_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    # Save metadata
    metadata = get_run_metadata(
        cfg,
        prompt_ids=pairs_with_ids,
        eval_window=16,
        intervention_scope=f"v_proj_L{patch_layer}",
        behavior_metric="domain_shift",
    )
    save_metadata(run_dir, metadata)
    
    # Append to run index
    append_to_run_index(run_dir, summary)
    
    print(f"\n{'='*60}")
    print("V_PROJ PATCHING ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Domain shift rate: {summary['domain_shift_rate']:.1f}%")
    print(f"Shift to philosophical: {summary['shift_to_philosophical_rate']:.1f}%")
    print(f"Philosophical markers increase: {summary['phil_marker_increase_mean']:.1f}")
    print(f"R_V transfer mean: {summary['rv_transfer_mean']:.3f}")
    print(f"R_V transfer delta: {summary['rv_transfer_delta_mean']:.3f}")
    print(f"\nResults saved to: {run_dir}")
    
    return ExperimentResult(summary=summary)
