"""
Canonical model registry helpers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_REGISTRY_PATH = REPO_ROOT / "configs" / "canonical_registry.json"


def load_canonical_registry(path: Path = CANONICAL_REGISTRY_PATH) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Canonical registry not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def get_canonical_model_spec(model_name: str, path: Path = CANONICAL_REGISTRY_PATH) -> Dict[str, Any]:
    registry = load_canonical_registry(path)
    models = registry.get("models", {})
    if model_name not in models:
        raise KeyError(f"Model '{model_name}' not found in canonical registry {path}")
    spec = dict(models[model_name])
    spec["name"] = model_name
    spec["registry_path"] = str(path)
    spec["registry_schema_version"] = registry.get("schema_version", "unknown")
    return spec
