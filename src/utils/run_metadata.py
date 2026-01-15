"""
Run Metadata Helper: Standardized metadata collection and run index management.

Industry-grade reproducibility: Every run logs:
- Git commit hash
- Prompt bank version
- Prompt IDs (for deterministic replay)
- Model version
- Experimental parameters
- Metric definitions
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from prompts.loader import PromptLoader


def get_git_commit() -> str:
    """
    Get current git commit hash.
    
    Returns:
        Commit hash string, or "not_a_git_repo" if not in git repo.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent.parent.parent,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "not_a_git_repo"


def get_run_metadata(
    cfg: Dict[str, Any],
    prompt_ids: Optional[List[tuple]] = None,
    eval_window: int = 16,
    intervention_scope: str = "all_tokens",
    behavior_metric: str = "mode_score_m",
) -> Dict[str, Any]:
    """
    Generate standardized run metadata.
    
    Args:
        cfg: Experiment config dictionary.
        prompt_ids: List of (rec_id, base_id, rec_text, base_text) tuples.
        eval_window: Measurement window size (default: 16 tokens).
        intervention_scope: Where intervention is applied ("all_tokens", "last_16", "BOS_only", etc.).
        behavior_metric: Primary behavior metric name (default: "mode_score_m").
    
    Returns:
        Standardized metadata dictionary.
    """
    loader = PromptLoader()
    
    metadata = {
        "git_commit": get_git_commit(),
        "prompt_bank_version": loader.version,
        "model_id": cfg.get("model", {}).get("name", cfg.get("params", {}).get("model", "unknown")),
        "seed": cfg.get("seed", cfg.get("params", {}).get("seed", 42)),
        "n_pairs": cfg.get("params", {}).get("n_pairs", None),
        "eval_window": eval_window,
        "intervention_scope": intervention_scope,
        "behavior_metric": behavior_metric,
    }
    
    # Add prompt IDs if provided
    if prompt_ids:
        recursive_ids = [rec_id for rec_id, _, _, _ in prompt_ids]
        baseline_ids = [base_id for _, base_id, _, _ in prompt_ids]
        metadata["prompt_ids"] = {
            "recursive": recursive_ids,
            "baseline": baseline_ids,
        }
    
    return metadata


def append_to_run_index(run_dir: Path, summary: Dict[str, Any]) -> None:
    """
    Append run metadata to centralized run index.
    
    Args:
        run_dir: Path to run directory.
        summary: Summary dictionary from experiment (will be merged with metadata).
    """
    index_path = Path(__file__).parent.parent.parent / "results" / "RUN_INDEX.jsonl"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    # GAP-001 Fix: Ensure file exists before append
    if not index_path.exists():
        index_path.touch()
    
    # Load existing metadata if available
    metadata_path = run_dir / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    
    # Create index entry
    # Handle both absolute and relative paths
    repo_root = Path(__file__).parent.parent.parent
    if run_dir.is_absolute():
        try:
            run_dir_rel = str(run_dir.relative_to(repo_root))
        except ValueError:
            # If not in repo, use absolute path
            run_dir_rel = str(run_dir)
    else:
        # Already relative, use as-is
        run_dir_rel = str(run_dir)
    
    entry = {
        "timestamp": run_dir.name.split("_")[0] if "_" in run_dir.name else "unknown",
        "run_dir": run_dir_rel,
        "experiment": summary.get("experiment", "unknown"),
        **metadata,
        **summary,  # Merge summary (may overwrite metadata keys)
    }
    
    # Append to JSONL file
    with open(index_path, "a") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")


def save_metadata(run_dir: Path, metadata: Dict[str, Any]) -> None:
    """
    Save metadata to run directory.
    
    Args:
        run_dir: Path to run directory.
        metadata: Metadata dictionary.
    """
    metadata_path = run_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

