#!/usr/bin/env python3
"""
CLI wrapper for multi-model config generation.

Usage:
    python scripts/generate_model_configs.py --model meta-llama/Meta-Llama-3-8B
    python scripts/generate_model_configs.py --model mistralai/Mistral-7B-v0.1 --short mistral7b

This wraps src/utils/multi_model_discovery.py with a command-line interface.
"""

import argparse
import json
import sys
from pathlib import Path

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.multi_model_discovery import (
    generate_discovery_configs,
    validate_registry_compatibility,
    validate_config_fields,
)


def get_model_config_from_hf(model_name: str) -> dict:
    """Extract model config from HuggingFace without loading weights."""
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        return {
            "num_layers": config.num_hidden_layers,
            "num_heads": config.num_attention_heads,
            "hidden_size": config.hidden_size,
            "num_kv_heads": getattr(config, "num_key_value_heads", config.num_attention_heads),
        }
    except Exception as e:
        print(f"Warning: Could not load config from HF: {e}")
        print("Please provide --layers and --heads manually.")
        return None


def model_name_to_short(model_name: str) -> str:
    """Convert model name to short identifier."""
    # meta-llama/Meta-Llama-3-8B -> llama3_8b
    # mistralai/Mistral-7B-v0.1 -> mistral_7b_v0.1
    short = model_name.split("/")[-1]
    short = short.lower().replace("-", "_").replace(".", "_")
    # Clean up common prefixes
    for prefix in ["meta_", "llama_"]:
        if short.startswith(prefix + prefix):
            short = short[len(prefix):]
    return short


def main():
    parser = argparse.ArgumentParser(
        description="Generate multi-model discovery configs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Auto-detect config from HuggingFace
    python scripts/generate_model_configs.py --model meta-llama/Meta-Llama-3-8B

    # Manual config (if HF fails)
    python scripts/generate_model_configs.py --model custom/Model --layers 32 --heads 32 --short custom_model

    # Dry run (show what would be generated)
    python scripts/generate_model_configs.py --model mistralai/Mistral-7B-v0.1 --dry-run
"""
    )

    parser.add_argument("--model", required=True, help="HuggingFace model name")
    parser.add_argument("--short", help="Short name for config files (auto-generated if not provided)")
    parser.add_argument("--output-dir", default="configs/discovery", help="Output directory")
    parser.add_argument("--layers", type=int, help="Number of layers (auto-detected if not provided)")
    parser.add_argument("--heads", type=int, help="Number of heads (auto-detected if not provided)")
    parser.add_argument("--late-layer", type=int, help="Override late layer (default: num_layers - 5)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be generated without writing")
    parser.add_argument("--validate-only", action="store_true", help="Validate existing configs")

    args = parser.parse_args()

    # Get model config
    if args.layers and args.heads:
        model_config = {
            "num_layers": args.layers,
            "num_heads": args.heads,
        }
    else:
        print(f"Fetching config for {args.model}...")
        model_config = get_model_config_from_hf(args.model)
        if model_config is None:
            print("ERROR: Could not auto-detect model config. Use --layers and --heads.")
            sys.exit(1)

    # Add optional overrides
    if args.late_layer:
        model_config["late_layer"] = args.late_layer

    # Generate short name
    model_short = args.short or model_name_to_short(args.model)

    print(f"\nModel: {args.model}")
    print(f"Short name: {model_short}")
    print(f"Layers: {model_config['num_layers']}")
    print(f"Heads: {model_config['num_heads']}")
    print(f"Late layer: {model_config.get('late_layer', model_config['num_layers'] - 5)}")
    print(f"Output: {args.output_dir}/{model_short}/")

    # Generate configs
    configs = generate_discovery_configs(
        model_name=args.model,
        model_config=model_config,
        out_dir=args.output_dir,
        model_short=model_short,
        write_files=not args.dry_run,
    )

    print(f"\nGenerated {len(configs)} configs:")
    for cfg in configs:
        phase = cfg.get("results", {}).get("phase", "unknown")
        print(f"  - {cfg['run_name']}")

    # Validate
    print("\nValidating registry compatibility...")
    missing = validate_registry_compatibility(configs)
    if missing:
        print(f"WARNING: Missing experiments in registry: {missing}")
    else:
        print("✓ All experiments registered")

    field_errors = validate_config_fields(configs)
    if field_errors:
        print(f"WARNING: Field validation errors: {field_errors}")
    else:
        print("✓ All required fields present")

    if args.dry_run:
        print("\n[DRY RUN] No files written. Remove --dry-run to generate files.")
    else:
        print(f"\n✓ Configs written to {args.output_dir}/{model_short}/")
        print(f"\nNext steps:")
        print(f"  1. Run Phase 2: python -m src.pipelines.run --config {args.output_dir}/{model_short}/01_baseline_rv.json")
        print(f"  2. Or use batch runner: ./scripts/run_model_discovery.sh {model_short}")


if __name__ == "__main__":
    main()
