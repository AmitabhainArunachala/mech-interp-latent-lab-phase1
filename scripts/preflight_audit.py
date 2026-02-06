#!/usr/bin/env python3
"""
Pre-Flight Audit Script — Gold Standard Compliance Checker

Run this BEFORE any GPU experiment to verify alignment with gold standard.

Usage:
    python scripts/preflight_audit.py --config configs/canonical/rv_l27_causal_validation.json
    python scripts/preflight_audit.py --model mixtral-8x7b --target-layer 27
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load and validate config file."""
    if not config_path.exists():
        return {"error": f"Config not found: {config_path}"}
    
    with open(config_path) as f:
        return json.load(f)


def check_gold_standard_alignment(cfg: Dict[str, Any]) -> List[Tuple[str, bool, str]]:
    """Check config against gold standard requirements."""
    checks = []
    
    # 1. Experiment field
    exp = cfg.get("experiment", "")
    checks.append(("experiment_field", bool(exp), f"experiment: {exp or 'MISSING'}"))
    
    # 2. Params object
    params = cfg.get("params", {})
    checks.append(("params_object", bool(params), "params present" if params else "params MISSING"))
    
    # 3. Sample size
    n = params.get("n_pairs", params.get("n_samples", 0))
    checks.append(("sample_size", n >= 50, f"n={n} (need ≥50)"))
    
    # 4. Controls
    controls = params.get("controls", [])
    required_controls = {"random", "shuffled", "wrong_layer", "orthogonal"}
    has_all = set(controls) >= required_controls
    checks.append(("controls", has_all, f"controls: {controls}"))
    
    # 5. Seed specified
    seed = params.get("seed", params.get("random_seed"))
    checks.append(("seed", seed is not None, f"seed: {seed}"))
    
    # 6. Device specified
    device = params.get("device", "auto")
    checks.append(("device", device in ("auto", "cuda", "mps"), f"device: {device}"))
    
    return checks


def check_prompt_bank() -> Tuple[bool, str]:
    """Check prompt bank exists and has sufficient prompts."""
    bank_path = Path(__file__).parent.parent / "prompts" / "bank.json"
    
    if not bank_path.exists():
        return False, "Prompt bank not found"
    
    with open(bank_path) as f:
        bank = json.load(f)
    
    count = len(bank)
    return count >= 100, f"{count} prompts in bank"


def check_registry_alignment(experiment: str) -> Tuple[bool, str]:
    """Check experiment is registered (local or external)."""
    # External experiments (run by OpenClawd mi-experimenter)
    external_experiments = {
        "rv_causal_validation",
        "cross_architecture_suite",
        "mlp_ablation",
    }
    
    if experiment in external_experiments:
        return True, f"external (OpenClawd mi-experimenter)"
    
    try:
        from src.pipelines.registry import get_registry
        registry = get_registry()
        return experiment in registry, f"{'registered' if experiment in registry else 'NOT registered'}"
    except ImportError:
        # Can't import without torch, check file directly
        registry_path = Path(__file__).parent.parent / "src" / "pipelines" / "registry.py"
        if registry_path.exists():
            content = registry_path.read_text()
            return f'"{experiment}"' in content, "registry check (file-based)"
        return False, "registry not found"


def run_preflight_audit(config_path: Path = None, model: str = None, target_layer: int = None):
    """Run complete pre-flight audit."""
    print("=" * 70)
    print("PRE-FLIGHT AUDIT — Gold Standard Compliance")
    print("=" * 70)
    
    all_passed = True
    
    # 1. Config check
    print("\n[1/4] Config Validation")
    if config_path:
        cfg = load_config(config_path)
        if "error" in cfg:
            print(f"  ✗ {cfg['error']}")
            all_passed = False
        else:
            checks = check_gold_standard_alignment(cfg)
            for name, passed, msg in checks:
                status = "✓" if passed else "✗"
                print(f"  {status} {name}: {msg}")
                if not passed:
                    all_passed = False
    else:
        print("  ⚠ No config provided, using CLI args")
        if model:
            print(f"  - model: {model}")
        if target_layer:
            print(f"  - target_layer: {target_layer}")
    
    # 2. Prompt bank
    print("\n[2/4] Prompt Bank")
    passed, msg = check_prompt_bank()
    status = "✓" if passed else "✗"
    print(f"  {status} {msg}")
    if not passed:
        all_passed = False
    
    # 3. Registry
    print("\n[3/4] Experiment Registry")
    if config_path:
        cfg = load_config(config_path)
        exp = cfg.get("experiment", "")
        if exp:
            passed, msg = check_registry_alignment(exp)
            status = "✓" if passed else "✗"
            print(f"  {status} {exp}: {msg}")
            if not passed:
                all_passed = False
    else:
        print("  ⚠ Skipped (no config)")
    
    # 4. Hardware
    print("\n[4/4] Hardware Check")
    try:
        import torch
        cuda = torch.cuda.is_available()
        if cuda:
            gpu = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  ✓ GPU: {gpu} ({mem:.1f}GB)")
        else:
            print("  ⚠ No CUDA GPU available")
        
        mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        if mps and not cuda:
            print(f"  ✓ MPS available (Apple Silicon)")
    except ImportError:
        print("  ⚠ PyTorch not installed, cannot check hardware")
    
    # Summary
    print("\n" + "=" * 70)
    if all_passed:
        print("PRE-FLIGHT AUDIT: ✓ PASSED")
        print("Ready to proceed with GPU experiment.")
    else:
        print("PRE-FLIGHT AUDIT: ✗ FAILED")
        print("Fix issues above before proceeding.")
    print("=" * 70)
    
    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Pre-flight audit for GPU experiments")
    parser.add_argument("--config", type=Path, help="Path to config file")
    parser.add_argument("--model", type=str, help="Model name (if no config)")
    parser.add_argument("--target-layer", type=int, help="Target layer (if no config)")
    
    args = parser.parse_args()
    
    passed = run_preflight_audit(
        config_path=args.config,
        model=args.model,
        target_layer=args.target_layer
    )
    
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
