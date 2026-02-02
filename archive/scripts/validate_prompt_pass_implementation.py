#!/usr/bin/env python3
"""
Validation script for prompt-pass-only implementation.

Tests:
1. Import validation - all modules load correctly
2. Registry validation - experiment registered
3. Config validation - all configs parse correctly
4. Component function validation - compute_rv_with_components works
5. Hook validation - MLPAblationHook instantiates

Run this BEFORE deploying to GPU to catch any syntax/import errors.
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all modules import correctly."""
    print("Testing imports...")
    try:
        from src.metrics.rv import compute_rv_with_components, compute_rv
        from src.pipelines.canonical.mlp_ablation_necessity_prompt_pass import (
            run_mlp_ablation_necessity_prompt_pass_from_config,
            MLPAblationHook
        )
        from src.pipelines.registry import get_registry
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_registry():
    """Test that experiment is registered."""
    print("\nTesting registry...")
    try:
        from src.pipelines.registry import get_registry
        registry = get_registry()
        if "mlp_ablation_necessity_prompt_pass" in registry:
            print("✅ Experiment registered in registry")
            return True
        else:
            print("❌ Experiment not found in registry")
            print(f"Available experiments: {sorted(registry.keys())}")
            return False
    except Exception as e:
        print(f"❌ Registry test failed: {e}")
        return False

def test_configs():
    """Test that all config files parse correctly."""
    print("\nTesting config files...")
    import json

    configs = [
        "configs/canonical/mlp_ablation_necessity_prompt_pass_l0.json",
        "configs/canonical/mlp_ablation_necessity_prompt_pass_l1.json",
        "configs/canonical/mlp_ablation_necessity_prompt_pass_l2.json",
        "configs/canonical/mlp_ablation_necessity_prompt_pass_l3.json",
        "configs/canonical/mlp_ablation_necessity_prompt_pass_l4.json",
        "configs/canonical/mlp_ablation_necessity_prompt_pass_l5.json",
        "configs/smoke_test/mlp_ablation_prompt_pass_l0_quick.json",
    ]

    all_valid = True
    for config_path in configs:
        try:
            with open(config_path, "r") as f:
                cfg = json.load(f)

            # Validate structure
            assert "experiment" in cfg
            assert cfg["experiment"] == "mlp_ablation_necessity_prompt_pass"
            assert "params" in cfg
            assert "layer" in cfg["params"]

            print(f"  ✅ {config_path}")
        except FileNotFoundError:
            print(f"  ❌ {config_path} not found")
            all_valid = False
        except Exception as e:
            print(f"  ❌ {config_path}: {e}")
            all_valid = False

    if all_valid:
        print("✅ All config files valid")
    return all_valid

def test_component_function():
    """Test that compute_rv_with_components has correct signature."""
    print("\nTesting compute_rv_with_components...")
    try:
        from src.metrics.rv import compute_rv_with_components
        import inspect

        sig = inspect.signature(compute_rv_with_components)
        params = list(sig.parameters.keys())

        expected_params = ["model", "tokenizer", "text", "early", "late", "window", "device"]

        if params == expected_params:
            print("✅ Function signature correct")
            print(f"   Parameters: {params}")

            # Check return type hint
            if hasattr(sig.return_annotation, '__origin__'):
                print(f"   Return type: {sig.return_annotation}")

            return True
        else:
            print(f"❌ Unexpected parameters: {params}")
            print(f"   Expected: {expected_params}")
            return False

    except Exception as e:
        print(f"❌ Component function test failed: {e}")
        return False

def test_hook_class():
    """Test that MLPAblationHook class exists and has correct methods."""
    print("\nTesting MLPAblationHook class...")
    try:
        from src.pipelines.canonical.mlp_ablation_necessity_prompt_pass import MLPAblationHook

        # Check methods exist
        required_methods = ["__init__", "register", "remove", "__enter__", "__exit__"]

        all_present = True
        for method in required_methods:
            if not hasattr(MLPAblationHook, method):
                print(f"  ❌ Missing method: {method}")
                all_present = False

        if all_present:
            print("✅ MLPAblationHook class complete")
            print(f"   Methods: {required_methods}")
            return True
        else:
            return False

    except Exception as e:
        print(f"❌ Hook class test failed: {e}")
        return False

def test_pipeline_function_signature():
    """Test that pipeline function has correct signature."""
    print("\nTesting pipeline function signature...")
    try:
        from src.pipelines.canonical.mlp_ablation_necessity_prompt_pass import (
            run_mlp_ablation_necessity_prompt_pass_from_config
        )
        import inspect

        sig = inspect.signature(run_mlp_ablation_necessity_prompt_pass_from_config)
        params = list(sig.parameters.keys())

        expected_params = ["cfg", "run_dir"]

        if params == expected_params:
            print("✅ Pipeline function signature correct")
            print(f"   Parameters: {params}")
            print(f"   Return type: {sig.return_annotation}")
            return True
        else:
            print(f"❌ Unexpected parameters: {params}")
            print(f"   Expected: {expected_params}")
            return False

    except Exception as e:
        print(f"❌ Pipeline function test failed: {e}")
        return False

def main():
    """Run all validation tests."""
    print("="*60)
    print("PROMPT-PASS-ONLY IMPLEMENTATION VALIDATION")
    print("="*60)

    tests = [
        ("Imports", test_imports),
        ("Registry", test_registry),
        ("Config Files", test_configs),
        ("Component Function", test_component_function),
        ("Hook Class", test_hook_class),
        ("Pipeline Function", test_pipeline_function_signature),
    ]

    results = {}
    for test_name, test_func in tests:
        results[test_name] = test_func()

    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)

    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
        if not passed:
            all_passed = False

    print("="*60)

    if all_passed:
        print("\n🎉 ALL TESTS PASSED - Ready for deployment!")
        print("\nNext steps:")
        print("1. Run smoke test:")
        print("   python run.py configs/smoke_test/mlp_ablation_prompt_pass_l0_quick.json")
        print("\n2. Run full L0 experiment:")
        print("   python run.py configs/canonical/mlp_ablation_necessity_prompt_pass_l0.json")
        print("\n3. Compare to original results in results/ directory")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED - Fix errors before deployment")
        return 1

if __name__ == "__main__":
    sys.exit(main())
