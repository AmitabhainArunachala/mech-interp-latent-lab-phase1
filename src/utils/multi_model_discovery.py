"""
Multi-model discovery config generator.

Creates standardized config sets for Phase 2–6 of the multi-model discovery
protocol, with prompt bank versioning and registry compatibility checks.

Merged from GPT5.2 (validation, versioning) + Gemini 3.0 (GQA detection, CLI).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from prompts.loader import PromptLoader


def extract_model_config(model_name: str) -> Dict[str, Any]:
    """
    Extract model configuration from HuggingFace without loading weights.

    Uses AutoConfig which only downloads config.json (~1KB), not the model.

    Args:
        model_name: HuggingFace model identifier (e.g., "meta-llama/Meta-Llama-3-8B")

    Returns:
        Dict with num_layers, num_heads, hidden_size, num_kv_heads, is_gqa, head_dim
    """
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

        num_heads = config.num_attention_heads
        num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
        hidden_size = config.hidden_size
        head_dim = getattr(config, "head_dim", hidden_size // num_heads)

        return {
            "num_layers": config.num_hidden_layers,
            "num_heads": num_heads,
            "hidden_size": hidden_size,
            "num_kv_heads": num_kv_heads,
            "is_gqa": num_kv_heads < num_heads,
            "head_dim": head_dim,
            "architecture": getattr(config, "architectures", ["unknown"])[0] if hasattr(config, "architectures") else "unknown",
        }
    except Exception as e:
        raise ValueError(f"Could not extract config from {model_name}: {e}")


def model_name_to_short(model_name: str) -> str:
    """
    Convert HuggingFace model name to short identifier.

    Examples:
        meta-llama/Meta-Llama-3-8B -> llama3_8b
        mistralai/Mistral-7B-v0.1 -> mistral_7b_v0_1
        google/gemma-2-9b -> gemma_2_9b
    """
    short = model_name.split("/")[-1]
    short = short.lower().replace("-", "_").replace(".", "_")
    # Clean up redundant prefixes
    for prefix in ["meta_llama_", "meta_"]:
        if short.startswith(prefix):
            short = short[len(prefix):]
    return short


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _results_phase(results_phase: str, model_short: str, stage: str) -> str:
    return f"{results_phase}/{model_short}/{stage}"


def generate_discovery_configs(
    model_name: str,
    model_config: Optional[Dict[str, Any]] = None,
    out_dir: Union[str, Path] = "configs/discovery",
    model_short: Optional[str] = None,
    results_phase: str = "phase2_generalization",
    seed: int = 42,
    device: str = "cuda",
    write_files: bool = True,
) -> List[Dict[str, Any]]:
    """
    Generate Phase 2–6 discovery configs for a new model.

    Args:
        model_name: HuggingFace model identifier.
        model_config: Dict with model info. If None, auto-extracted from HuggingFace.
                      Required keys: num_layers, num_heads.
                      Optional keys: num_kv_heads, is_gqa, hidden_size, late_layer.
        out_dir: Base directory to write configs (e.g., configs/discovery).
        model_short: Short name (e.g., "llama3_8b"). Auto-generated if None.
        results_phase: Phase root for results (default: phase2_generalization).
        seed: Random seed for reproducibility.
        device: Compute device for model config blocks.
        write_files: If True, write JSON configs to disk.

    Returns:
        List of config dicts (in generated order).

    Example:
        # Auto-extract everything from HuggingFace
        configs = generate_discovery_configs("meta-llama/Meta-Llama-3-8B")

        # Or provide manual config
        configs = generate_discovery_configs(
            "custom/model",
            model_config={"num_layers": 32, "num_heads": 32},
            model_short="custom_model"
        )
    """
    # Auto-extract config if not provided
    if model_config is None:
        model_config = extract_model_config(model_name)

    # Auto-generate short name if not provided
    if model_short is None:
        model_short = model_name_to_short(model_name)

    num_layers = int(model_config["num_layers"])
    num_heads = int(model_config["num_heads"])
    num_kv_heads = int(model_config.get("num_kv_heads", num_heads))
    is_gqa = model_config.get("is_gqa", num_kv_heads < num_heads)
    hidden_size = int(model_config.get("hidden_size", 4096))
    head_dim = int(model_config.get("head_dim", hidden_size // num_heads))

    # Layer calculations with defensive bounds (handles tiny models)
    late_layer = int(model_config.get("late_layer", max(1, num_layers - 5)))
    wrong_layer = int(model_config.get("wrong_layer", max(0, late_layer - 6)))
    early_layer = min(5, num_layers - 2) if num_layers <= 7 else 5

    loader = PromptLoader()
    prompt_bank_version = loader.version

    base_dir = Path(out_dir) / model_short
    configs: List[Dict[str, Any]] = []

    # Model metadata block (included in all configs for traceability)
    model_metadata = {
        "name": model_name,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "is_gqa": is_gqa,
        "hidden_size": hidden_size,
        "head_dim": head_dim,
        "late_layer": late_layer,
        "early_layer": early_layer,
    }

    def add_config(name: str, cfg: Dict[str, Any]) -> None:
        cfg["prompt_bank_version"] = prompt_bank_version
        cfg["model_metadata"] = model_metadata
        configs.append(cfg)
        if write_files:
            _write_json(base_dir / name, cfg)

    # Phase 2: Baseline R_V separation (cross-architecture validation)
    add_config(
        "01_baseline_rv.json",
        {
            "experiment": "cross_architecture_validation",
            "run_name": f"{model_short}_baseline_rv",
            "seed": seed,
            "results": {"root": "results", "phase": _results_phase(results_phase, model_short, "01_baseline_rv")},
            "params": {
                "model": model_name,
                "early_layer": early_layer,
                "late_layer": late_layer,
                "window": 16,
                "n_champions": 50,
                "n_length_matched": 50,
                "n_pseudo_recursive": 50,
                "seed": seed,
                "prompt_groups": {"recursive": "champions", "controls": ["length_matched", "pseudo_recursive"]},
            },
        },
    )

    # Phase 3: Source hunt (MLP ablation, L0–L8)
    for layer in range(min(9, num_layers)):
        add_config(
            f"02_source_hunt_mlp_ablation_l{layer}.json",
            {
                "experiment": "mlp_ablation_necessity",
                "run_name": f"{model_short}_ablation_l{layer}",
                "seed": seed,
                "results": {"root": "results", "phase": _results_phase(results_phase, model_short, "02_source_hunt")},
                "params": {
                    "model": model_name,
                    "layer": layer,
                    "n_pairs": 80,
                    "window_size": 16,
                    "max_new_tokens": 200,
                    "seed": seed,
                },
            },
        )

    # Phase 4: Transfer sweet spot (MLP steering, L0–L10)
    for layer in range(min(11, num_layers)):
        add_config(
            f"03_transfer_hunt_mlp_steer_l{layer}.json",
            {
                "experiment": "combined_mlp_sufficiency_test",
                "run_name": f"{model_short}_steer_l{layer}",
                "seed": seed,
                "results": {"root": "results", "phase": _results_phase(results_phase, model_short, "03_transfer_hunt")},
                "params": {
                    "model": model_name,
                    "layers": [layer],
                    "n_pairs": 30,
                    "window_size": 16,
                    "max_new_tokens": 200,
                    "seed": seed,
                },
            },
        )

    # Phase 5: Readout validation (four-way controls)
    add_config(
        "04_readout_validation.json",
        {
            "experiment": "rv_l27_causal_validation",
            "run_name": f"{model_short}_readout_validation",
            "seed": seed,
            "results": {"root": "results", "phase": _results_phase(results_phase, model_short, "04_readout_validation")},
            "model": {"name": model_name, "device": device},
            "params": {
                "early_layer": early_layer,
                "target_layer": late_layer,
                "wrong_layer": wrong_layer,
                "window": 16,
                "max_pairs": 80,
                "max_length": 512,
                "pairing": {
                    "recursive_groups": ["L5_refined", "L4_full", "L3_deeper"],
                    "baseline_groups": ["long_control", "baseline_creative", "baseline_math"],
                },
                "measure_target_after_wrong_patch": False,
            },
        },
    )

    # Phase 6: Head identification
    # Note: For GQA models, num_kv_heads < num_heads. The pipeline handles this dynamically.
    add_config(
        "05_head_identification.json",
        {
            "experiment": "head_ablation_validation",
            "run_name": f"{model_short}_head_ablation",
            "seed": seed,
            "results": {"root": "results", "phase": _results_phase(results_phase, model_short, "05_head_identification")},
            "model": {"name": model_name, "device": device},
            "params": {
                "early_layer": early_layer,
                "target_layer": late_layer,
                "control_layer": wrong_layer,
                "window": 16,
                "n_recursive": 50,
                "n_baseline": 50,
                "max_length": 512,
                "recursive_groups": ["champions", "L5_refined", "L4_full", "L3_deeper"],
                "baseline_groups": ["baseline_math", "baseline_factual", "baseline_creative"],
                # GQA info for head ablation (pipeline extracts dynamically, but useful for reference)
                "num_kv_heads_hint": num_kv_heads,
                "is_gqa_hint": is_gqa,
            },
        },
    )

    return configs


def validate_registry_compatibility(configs: Iterable[Dict[str, Any]]) -> List[str]:
    """
    Return a sorted list of experiment names missing from the registry.
    """
    from src.pipelines.registry import get_registry

    registry = get_registry()
    missing = set()
    for cfg in configs:
        exp = cfg.get("experiment")
        if not exp or exp not in registry:
            missing.add(str(exp))
    return sorted(missing)


def validate_config_fields(configs: Iterable[Dict[str, Any]]) -> List[str]:
    """
    Validate required fields for generated configs.
    """
    errors: List[str] = []
    for cfg in configs:
        exp = cfg.get("experiment", "unknown")
        params = cfg.get("params") or {}
        if "model" not in params and not cfg.get("model"):
            errors.append(f"{exp}: missing model name")
        if "seed" not in params and "seed" not in cfg:
            errors.append(f"{exp}: missing seed")
        if "window" in params and int(params.get("window", 0)) <= 0:
            errors.append(f"{exp}: invalid window")
    return errors
