#!/usr/bin/env python3
"""
FULL CIRCUIT MAPPING EXPERIMENT
===============================

Master orchestration script that:
1. Maps complete L0 → L_late circuit on Mistral-7B
2. Runs multi-turn eigenstate tracking sessions
3. Logs EVERYTHING to results/circuit_mapping/

Storage structure:
    results/circuit_mapping/
    ├── {timestamp}_run_manifest.json     # Full run metadata
    ├── {timestamp}_hardware.json          # GPU, CUDA, driver, dtype
    ├── anatomy/
    │   ├── recursive/                     # Per-prompt anatomies
    │   ├── baseline/
    │   └── aggregate_stats.json
    ├── eigenstate_sessions/
    │   ├── session_{id}.json              # Full session data
    │   └── session_{id}_report.md         # Human-readable
    └── plots/
        ├── pr_trajectory_recursive.png
        ├── pr_trajectory_baseline.png
        └── rv_vs_depth_comparison.png

Usage:
    # Full experiment (needs GPU)
    python experiments/full_circuit_mapping.py --model mistralai/Mistral-7B-Instruct-v0.2
    
    # Quick validation run
    python experiments/full_circuit_mapping.py --quick --n-prompts 5

Environment:
    - Requires: torch, transformers, sentence-transformers, lancedb
    - GPU: ~14GB VRAM for Mistral-7B
    - Runtime: ~2-4 hours full, ~15 min quick
"""

import argparse
import json
import logging
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import torch
import numpy as np

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.models import load_model, set_seed
from src.analysis.circuit_anatomizer import CircuitAnatomizer, run_anatomy_experiment
from src.pipelines.experimental.multi_turn_eigenstate import EigenstateTracker, DIALOGUE_PROTOCOLS
from prompts.loader import PromptLoader


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(output_dir: Path, run_id: str) -> logging.Logger:
    """Configure logging to both file and console."""
    log_file = output_dir / f"{run_id}_experiment.log"
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return logging.getLogger(__name__)


# ============================================================================
# HARDWARE LOGGING
# ============================================================================

def get_hardware_info() -> Dict[str, Any]:
    """Capture complete hardware/environment info."""
    info = {
        "timestamp": datetime.now().isoformat(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "python_version": sys.version,
        },
        "torch": {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        }
    }
    
    if torch.cuda.is_available():
        info["gpu"] = {
            "name": torch.cuda.get_device_name(0),
            "memory_total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "device_count": torch.cuda.device_count(),
        }
        
        # Try to get driver version
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                info["gpu"]["driver_version"] = result.stdout.strip()
        except Exception:
            pass
    
    return info


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_full_circuit_mapping(
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
    n_prompts: int = 30,
    layer_step: int = 1,
    run_eigenstate: bool = True,
    quick_mode: bool = False,
    seed: int = 42,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Run complete circuit mapping experiment.
    
    Args:
        model_name: Model to analyze
        n_prompts: Number of prompts per category
        layer_step: Sample every N layers (1 = all, 2 = every other)
        run_eigenstate: Whether to run eigenstate tracking sessions
        quick_mode: Reduced prompts and layers for validation
        seed: Random seed
        device: cuda or cpu
    
    Returns:
        Complete results dict
    """
    # ========== SETUP ==========
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / "results" / "circuit_mapping" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Subdirectories
    (output_dir / "anatomy" / "recursive").mkdir(parents=True, exist_ok=True)
    (output_dir / "anatomy" / "baseline").mkdir(parents=True, exist_ok=True)
    (output_dir / "eigenstate_sessions").mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(output_dir, run_id)
    logger.info(f"Starting Full Circuit Mapping Experiment: {run_id}")
    logger.info(f"Output directory: {output_dir}")
    
    # Quick mode adjustments
    if quick_mode:
        n_prompts = min(n_prompts, 5)
        layer_step = 2
        logger.info("QUICK MODE: Reduced prompts and layer sampling")
    
    # ========== HARDWARE LOGGING ==========
    logger.info("Capturing hardware info...")
    hardware_info = get_hardware_info()
    
    with open(output_dir / f"{run_id}_hardware.json", "w") as f:
        json.dump(hardware_info, f, indent=2)
    
    logger.info(f"GPU: {hardware_info.get('gpu', {}).get('name', 'CPU only')}")
    if 'gpu' in hardware_info:
        logger.info(f"CUDA: {hardware_info['gpu'].get('cuda_version')}")
        logger.info(f"VRAM: {hardware_info['gpu'].get('memory_total_gb', 0):.1f} GB")
    
    # ========== MODEL LOADING ==========
    logger.info(f"Loading model: {model_name}")
    set_seed(seed)
    
    model, tokenizer = load_model(model_name, device=device)
    num_layers = model.config.num_hidden_layers
    late_layer = num_layers - 5  # 84% depth rule
    
    logger.info(f"Model loaded: {num_layers} layers, late_layer = L{late_layer}")
    
    # ========== PROMPT LOADING ==========
    logger.info("Loading prompts...")
    loader = PromptLoader()
    
    recursive_prompts = loader.get_by_group("L5_refined")[:n_prompts]
    baseline_prompts = loader.get_by_group("baseline_factual")[:n_prompts]
    
    logger.info(f"Loaded {len(recursive_prompts)} recursive, {len(baseline_prompts)} baseline prompts")
    
    # ========== CIRCUIT ANATOMY ==========
    logger.info("="*60)
    logger.info("PHASE 1: CIRCUIT ANATOMY")
    logger.info("="*60)
    
    anatomizer = CircuitAnatomizer(
        model=model,
        tokenizer=tokenizer,
        late_layer=late_layer,
        device=device,
    )
    
    anatomy_results = {
        "recursive": [],
        "baseline": [],
        "aggregate": {},
    }
    
    # Analyze recursive prompts
    logger.info(f"Analyzing {len(recursive_prompts)} recursive prompts...")
    for i, text in enumerate(recursive_prompts):
        logger.debug(f"Recursive prompt {i+1}/{len(recursive_prompts)}")
        
        anatomy = anatomizer.full_anatomy(text, step=layer_step)
        anatomy_results["recursive"].append(anatomy.to_dict())
        
        # Save individual anatomy
        with open(output_dir / "anatomy" / "recursive" / f"anatomy_{i:03d}.json", "w") as f:
            json.dump(anatomy.to_dict(), f, indent=2)
        
        # Plot first few
        if i < 3:
            try:
                anatomy.plot(output_dir / "plots" / f"recursive_{i}_trajectory.png")
            except Exception as e:
                logger.warning(f"Plot failed: {e}")
        
        if (i + 1) % 10 == 0:
            logger.info(f"  Completed {i+1}/{len(recursive_prompts)} recursive")
    
    # Analyze baseline prompts
    logger.info(f"Analyzing {len(baseline_prompts)} baseline prompts...")
    for i, text in enumerate(baseline_prompts):
        logger.debug(f"Baseline prompt {i+1}/{len(baseline_prompts)}")
        
        anatomy = anatomizer.full_anatomy(text, step=layer_step)
        anatomy_results["baseline"].append(anatomy.to_dict())
        
        with open(output_dir / "anatomy" / "baseline" / f"anatomy_{i:03d}.json", "w") as f:
            json.dump(anatomy.to_dict(), f, indent=2)
        
        if i < 3:
            try:
                anatomy.plot(output_dir / "plots" / f"baseline_{i}_trajectory.png")
            except Exception as e:
                logger.warning(f"Plot failed: {e}")
        
        if (i + 1) % 10 == 0:
            logger.info(f"  Completed {i+1}/{len(baseline_prompts)} baseline")
    
    # Aggregate statistics
    rec_rvs = [a["rv"] for a in anatomy_results["recursive"] if not np.isnan(a["rv"])]
    base_rvs = [a["rv"] for a in anatomy_results["baseline"] if not np.isnan(a["rv"])]
    
    anatomy_results["aggregate"] = {
        "n_recursive": len(recursive_prompts),
        "n_baseline": len(baseline_prompts),
        "rv_recursive_mean": float(np.mean(rec_rvs)) if rec_rvs else None,
        "rv_recursive_std": float(np.std(rec_rvs)) if rec_rvs else None,
        "rv_baseline_mean": float(np.mean(base_rvs)) if base_rvs else None,
        "rv_baseline_std": float(np.std(base_rvs)) if base_rvs else None,
        "rv_delta": float(np.mean(rec_rvs) - np.mean(base_rvs)) if rec_rvs and base_rvs else None,
    }
    
    # Find critical layers
    if anatomy_results["recursive"]:
        all_max_drops = [a["max_drop_layer"] for a in anatomy_results["recursive"]]
        from collections import Counter
        drop_counts = Counter(all_max_drops)
        anatomy_results["aggregate"]["critical_layers"] = drop_counts.most_common(5)
    
    with open(output_dir / "anatomy" / "aggregate_stats.json", "w") as f:
        json.dump(anatomy_results["aggregate"], f, indent=2)
    
    logger.info(f"ANATOMY COMPLETE:")
    logger.info(f"  R_V recursive: {anatomy_results['aggregate']['rv_recursive_mean']:.3f} ± {anatomy_results['aggregate']['rv_recursive_std']:.3f}")
    logger.info(f"  R_V baseline:  {anatomy_results['aggregate']['rv_baseline_mean']:.3f} ± {anatomy_results['aggregate']['rv_baseline_std']:.3f}")
    logger.info(f"  Delta:         {anatomy_results['aggregate']['rv_delta']:.3f}")
    logger.info(f"  Critical layers: {anatomy_results['aggregate'].get('critical_layers', [])}")
    
    # ========== EIGENSTATE TRACKING ==========
    eigenstate_results = {}
    
    if run_eigenstate:
        logger.info("="*60)
        logger.info("PHASE 2: EIGENSTATE TRACKING")
        logger.info("="*60)
        
        tracker = EigenstateTracker(
            model=model,
            tokenizer=tokenizer,
            late_layer=late_layer,
            device=device,
        )
        
        for protocol_name in ["guided_descent", "phenomenological_probing"]:
            logger.info(f"Running protocol: {protocol_name}")
            
            try:
                session = tracker.run_protocol(protocol_name, verbose=False)
                
                # Save session
                session_file = output_dir / "eigenstate_sessions" / f"session_{session.session_id}_{protocol_name}.json"
                with open(session_file, "w") as f:
                    json.dump(session.to_dict(), f, indent=2)
                
                # Save report
                report_file = output_dir / "eigenstate_sessions" / f"session_{session.session_id}_{protocol_name}_report.md"
                tracker.save_session(output_dir / "eigenstate_sessions")
                
                eigenstate_results[protocol_name] = {
                    "session_id": session.session_id,
                    "baseline_rv": session.baseline_rv,
                    "min_rv": session.min_rv,
                    "eigenstate_turn": session.eigenstate_turn,
                    "genuine_l4_turns": session.genuine_l4_turns,
                    "rv_trajectory": session.rv_trajectory,
                }
                
                logger.info(f"  {protocol_name}:")
                logger.info(f"    Baseline R_V: {session.baseline_rv:.3f}")
                logger.info(f"    Min R_V:      {session.min_rv:.3f}")
                logger.info(f"    Eigenstate:   Turn {session.eigenstate_turn}")
                logger.info(f"    L4 turns:     {session.genuine_l4_turns}")
                
            except Exception as e:
                logger.error(f"Eigenstate tracking failed for {protocol_name}: {e}")
                eigenstate_results[protocol_name] = {"error": str(e)}
    
    # ========== FINAL MANIFEST ==========
    manifest = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model_name": model_name,
            "n_prompts": n_prompts,
            "layer_step": layer_step,
            "quick_mode": quick_mode,
            "seed": seed,
            "num_layers": num_layers,
            "late_layer": late_layer,
        },
        "hardware": hardware_info,
        "anatomy_summary": anatomy_results["aggregate"],
        "eigenstate_summary": eigenstate_results,
        "output_paths": {
            "anatomy": str(output_dir / "anatomy"),
            "eigenstate": str(output_dir / "eigenstate_sessions"),
            "plots": str(output_dir / "plots"),
            "log": str(output_dir / f"{run_id}_experiment.log"),
        },
    }
    
    with open(output_dir / f"{run_id}_run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    
    logger.info("="*60)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("="*60)
    logger.info(f"All results saved to: {output_dir}")
    logger.info(f"Manifest: {output_dir / f'{run_id}_run_manifest.json'}")
    
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Full Circuit Mapping Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.2",
                       help="Model to analyze")
    parser.add_argument("--n-prompts", type=int, default=30,
                       help="Number of prompts per category")
    parser.add_argument("--layer-step", type=int, default=1,
                       help="Sample every N layers (1=all)")
    parser.add_argument("--no-eigenstate", action="store_true",
                       help="Skip eigenstate tracking")
    parser.add_argument("--quick", action="store_true",
                       help="Quick validation mode (fewer prompts)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device (cuda or cpu)")
    
    args = parser.parse_args()
    
    manifest = run_full_circuit_mapping(
        model_name=args.model,
        n_prompts=args.n_prompts,
        layer_step=args.layer_step,
        run_eigenstate=not args.no_eigenstate,
        quick_mode=args.quick,
        seed=args.seed,
        device=args.device,
    )
    
    print(f"\n✅ Experiment complete! Results: {manifest['output_paths']['anatomy']}")


if __name__ == "__main__":
    main()
