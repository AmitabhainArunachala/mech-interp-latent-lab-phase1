#!/usr/bin/env python3
"""Verify repo is research-ready.

Comprehensive checks:
1. Core imports work (rv.py, model_physics.py, registry.py)
2. Registry loads all experiments AND each is callable
3. Prompt bank accessible with correct count
4. Config files reference valid experiments
5. SVD uses float64 precision
6. Deprecated pipelines are not in registry

Run from repo root:
    python scripts/verify_research_ready.py
"""
import json
import sys
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))


def main():
    errors = []
    warnings = []

    print("=" * 60)
    print("RESEARCH READY VERIFICATION (v2)")
    print("=" * 60)
    print()

    # 1. Check core imports work
    print("1. Core imports...")
    try:
        from src.metrics.rv import compute_rv, participation_ratio
        from src.core.model_physics import get_model_physics, ModelPhysics
        from src.pipelines.registry import get_registry, ExperimentResult
        print("   [OK] Core imports successful")
    except Exception as e:
        errors.append(f"Import error: {e}")
        print(f"   [FAIL] Core imports: {e}")
        # Can't continue without imports
        print(f"\n{len(errors)} ERRORS - NOT RESEARCH READY")
        sys.exit(1)

    # 2. Check registry loads all experiments AND they're callable
    print("\n2. Registry validation...")
    try:
        reg = get_registry()
        exp_count = len(reg)
        print(f"   [OK] Registry: {exp_count} experiments loaded")

        # Verify each is callable
        non_callable = [k for k, v in reg.items() if not callable(v)]
        if non_callable:
            errors.append(f"Non-callable experiments: {non_callable}")
            print(f"   [FAIL] Non-callable: {non_callable}")

        # Verify deprecated pipeline is NOT in registry
        if "mlp_ablation_necessity" in reg:
            errors.append("Deprecated mlp_ablation_necessity still in registry")
            print("   [FAIL] Deprecated mlp_ablation_necessity still registered")
        else:
            print("   [OK] Deprecated mlp_ablation_necessity removed from registry")

        # Verify key experiments exist
        required = ["rv_l27_causal_validation", "mlp_ablation_necessity_prompt_pass", "behavioral_grounding"]
        missing = [exp for exp in required if exp not in reg]
        if missing:
            errors.append(f"Missing required experiments: {missing}")
            print(f"   [FAIL] Missing: {missing}")
    except Exception as e:
        errors.append(f"Registry error: {e}")
        print(f"   [FAIL] Registry: {e}")

    # 3. Check prompt bank accessible with EXACT count and version
    print("\n3. Prompt bank...")
    # Canonical values - update these when bank.json is intentionally modified
    EXPECTED_PROMPT_COUNT = 754
    EXPECTED_VERSION_PREFIX = "75e7c1b8"  # First 8 chars of SHA256 hash

    try:
        from prompts.loader import PromptLoader
        loader = PromptLoader()
        count = len(loader._prompts) if loader._prompts else 0
        version = getattr(loader, 'version', 'unknown')

        # Check exact count
        if count == EXPECTED_PROMPT_COUNT:
            print(f"   [OK] Prompt bank: {count} prompts (exact match)")
        elif count > EXPECTED_PROMPT_COUNT:
            warnings.append(f"Prompt count increased: {count} > {EXPECTED_PROMPT_COUNT}")
            print(f"   [WARN] Prompt bank: {count} prompts (expected {EXPECTED_PROMPT_COUNT})")
        else:
            errors.append(f"Prompt count decreased: {count} < {EXPECTED_PROMPT_COUNT}")
            print(f"   [FAIL] Prompt bank: {count} prompts (expected {EXPECTED_PROMPT_COUNT})")

        # Check version hash prefix
        if version.startswith(EXPECTED_VERSION_PREFIX):
            print(f"   [OK] Version hash: {version[:8]}... (matches expected)")
        else:
            warnings.append(f"Version hash changed: {version[:8]} != {EXPECTED_VERSION_PREFIX}")
            print(f"   [WARN] Version hash: {version[:8]} != expected {EXPECTED_VERSION_PREFIX}")
            print(f"         (If bank.json was intentionally updated, update EXPECTED_* in this script)")
    except Exception as e:
        errors.append(f"Prompt bank error: {e}")
        print(f"   [FAIL] Prompt bank: {e}")

    # 4. Check config files reference valid experiments
    print("\n4. Config validation...")

    # Known exceptions: meta-experiments or intentionally removed pipelines
    KNOWN_EXCEPTIONS = {
        "batch_run",  # Meta-runner, not a pipeline
        "mlp_ablation_necessity",  # Removed from registry - contract violation (measures R_V on generated text)
        "p10_advanced_steering",  # Old experiment name, archived
    }

    config_dirs = [
        repo_root / "configs" / "canonical",
        repo_root / "configs" / "discovery",
        repo_root / "configs" / "smoke_test",
        repo_root / "configs" / "gold",
    ]
    orphaned_configs = []
    known_exception_configs = []

    for config_dir in config_dirs:
        if not config_dir.exists():
            continue
        for config_file in config_dir.glob("**/*.json"):
            try:
                with open(config_file) as f:
                    cfg = json.load(f)
                exp_name = cfg.get("experiment")
                if exp_name and exp_name not in reg:
                    if exp_name in KNOWN_EXCEPTIONS:
                        known_exception_configs.append(f"{config_file.relative_to(repo_root)}: {exp_name}")
                    else:
                        orphaned_configs.append(f"{config_file.relative_to(repo_root)}: {exp_name}")
            except Exception as e:
                warnings.append(f"Config parse error {config_file.name}: {e}")

    if orphaned_configs:
        errors.append(f"Configs reference missing experiments: {len(orphaned_configs)}")
        print(f"   [FAIL] {len(orphaned_configs)} configs reference missing experiments:")
        for oc in orphaned_configs[:5]:
            print(f"      - {oc}")
        if len(orphaned_configs) > 5:
            print(f"      ... and {len(orphaned_configs) - 5} more")
    elif known_exception_configs:
        warnings.append(f"{len(known_exception_configs)} configs use known exceptions (batch_run, deprecated pipelines)")
        print(f"   [WARN] {len(known_exception_configs)} configs reference known exceptions:")
        for oc in known_exception_configs[:3]:
            print(f"      - {oc}")
        if len(known_exception_configs) > 3:
            print(f"      ... and {len(known_exception_configs) - 3} more")
    else:
        print("   [OK] All configs reference valid experiments")

    # 5. Check model_physics is usable
    print("\n5. Model physics...")
    try:
        physics = get_model_physics("mistralai/Mistral-7B-v0.1")
        print(f"   [OK] Mistral-7B: early={physics.early_layer}, late={physics.late_layer}")
    except Exception as e:
        errors.append(f"Model physics error: {e}")
        print(f"   [FAIL] Model physics: {e}")

    # 6. Check rv.py uses double precision
    print("\n6. SVD precision check...")
    try:
        import inspect
        from src.metrics import rv
        source = inspect.getsource(rv.participation_ratio)
        if ".double()" in source:
            print("   [OK] rv.py uses float64 (double precision)")
        elif ".float()" in source:
            errors.append("rv.py still uses float32")
            print("   [FAIL] rv.py uses float32 (should be double)")
        else:
            warnings.append("Could not verify SVD precision")
            print("   [WARN] Could not verify precision")
    except Exception as e:
        warnings.append(f"Precision check error: {e}")
        print(f"   [WARN] Could not verify: {e}")

    # 7. Check requirements files
    print("\n7. Requirements validation...")

    # 7a. Check requirements.txt exists (development)
    req_txt = repo_root / "requirements.txt"
    if not req_txt.exists():
        errors.append("requirements.txt missing")
        print("   [FAIL] requirements.txt not found")
    else:
        print("   [OK] requirements.txt exists (development)")

    # 7b. Check requirements.lock exists and has exact pins (reproducibility)
    req_lock = repo_root / "requirements.lock"
    if not req_lock.exists():
        errors.append("requirements.lock missing (needed for reproducibility)")
        print("   [FAIL] requirements.lock not found")
    else:
        lock_content = req_lock.read_text()
        # Check that non-comment lines use ==
        lock_lines = [
            line.strip() for line in lock_content.split('\n')
            if line.strip() and not line.strip().startswith('#')
        ]
        has_exact_pins = all("==" in line for line in lock_lines if line)
        if has_exact_pins:
            print(f"   [OK] requirements.lock uses exact pins ({len(lock_lines)} packages)")
        else:
            bad_lines = [l for l in lock_lines if "==" not in l]
            errors.append(f"requirements.lock has non-exact pins: {bad_lines[:3]}")
            print(f"   [FAIL] requirements.lock missing exact pins: {bad_lines[:3]}")

    # Summary
    print()
    print("=" * 60)
    if errors:
        print(f"RESULT: {len(errors)} ERRORS, {len(warnings)} WARNINGS - NOT RESEARCH READY")
        print("=" * 60)
        print("\nErrors:")
        for err in errors:
            print(f"  - {err}")
        if warnings:
            print("\nWarnings:")
            for warn in warnings:
                print(f"  - {warn}")
        sys.exit(1)
    elif warnings:
        print(f"RESULT: RESEARCH READY ({len(warnings)} warnings)")
        print("=" * 60)
        print("\nWarnings:")
        for warn in warnings:
            print(f"  - {warn}")
        sys.exit(0)
    else:
        print("RESULT: RESEARCH READY")
        print("=" * 60)
        sys.exit(0)


if __name__ == "__main__":
    main()
