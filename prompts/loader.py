"""
PromptLoader: Strict API to fetch balanced sets of prompts.

No ad-hoc lists in .py files. All prompts come from prompts/bank.json.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Try to import from REUSABLE_PROMPT_BANK as fallback
try:
    from REUSABLE_PROMPT_BANK import get_all_prompts as _get_all_prompts_legacy
    _HAS_LEGACY = True
except ImportError:
    _HAS_LEGACY = False


class PromptLoader:
    """
    Loader for prompt bank with schema validation.
    
    Prompts are tagged by:
    - pillar: "recursive" (dose_response), "baseline", "control", etc.
    - type: "recursive", "instructional", "completion", "creative"
    - group: Specific group within pillar (e.g., "L3_deeper", "baseline_math")
    """
    
    def __init__(self, bank_path: Optional[Path] = None):
        """
        Initialize the prompt loader.
        
        Args:
            bank_path: Path to prompts/bank.json. If None, tries default location
                      or falls back to REUSABLE_PROMPT_BANK.
        """
        if bank_path is None:
            bank_path = Path(__file__).parent / "bank.json"
        
        self.bank_path = bank_path
        self._prompts: Optional[Dict] = None
        
        if bank_path.exists():
            self._load_json()
        elif _HAS_LEGACY:
            self._load_legacy()
        else:
            raise FileNotFoundError(
                f"Prompt bank not found at {bank_path} and REUSABLE_PROMPT_BANK not available"
            )
    
    def _load_json(self):
        """Load prompts from JSON file."""
        with open(self.bank_path, "r", encoding="utf-8") as f:
            self._prompts = json.load(f)
    
    def _load_legacy(self):
        """Load prompts from REUSABLE_PROMPT_BANK (temporary migration path)."""
        self._prompts = _get_all_prompts_legacy()
    
    @property
    def prompts(self) -> Dict:
        """Get all prompts as a dictionary."""
        if self._prompts is None:
            raise RuntimeError("Prompts not loaded")
        return self._prompts
    
    def get_by_pillar(
        self,
        pillar: str,
        limit: Optional[int] = None,
        seed: int = 0,
    ) -> List[str]:
        """
        Get prompts by pillar.
        
        Args:
            pillar: Pillar name ("recursive", "baseline", "control", etc.).
            limit: Maximum number of prompts to return.
            seed: Random seed for shuffling.
        
        Returns:
            List of prompt text strings.
        """
        rng = random.Random(seed)
        filtered = [
            v["text"] for k, v in self.prompts.items()
            if v.get("pillar") == pillar
        ]
        rng.shuffle(filtered)
        return filtered[:limit] if limit else filtered
    
    def get_by_type(
        self,
        prompt_type: str,
        limit: Optional[int] = None,
        seed: int = 0,
    ) -> List[str]:
        """
        Get prompts by type.
        
        Args:
            prompt_type: Type ("recursive", "instructional", "completion", "creative").
            limit: Maximum number of prompts to return.
            seed: Random seed for shuffling.
        
        Returns:
            List of prompt text strings.
        """
        rng = random.Random(seed)
        filtered = [
            v["text"] for k, v in self.prompts.items()
            if v.get("type") == prompt_type
        ]
        rng.shuffle(filtered)
        return filtered[:limit] if limit else filtered
    
    def get_by_group(
        self,
        group: str,
        limit: Optional[int] = None,
        seed: int = 0,
    ) -> List[str]:
        """
        Get prompts by group.
        
        Args:
            group: Group name (e.g., "L3_deeper", "baseline_math").
            limit: Maximum number of prompts to return.
            seed: Random seed for shuffling.
        
        Returns:
            List of prompt text strings.
        """
        rng = random.Random(seed)
        filtered = [
            v["text"] for k, v in self.prompts.items()
            if v.get("group") == group
        ]
        rng.shuffle(filtered)
        return filtered[:limit] if limit else filtered
    
    def get_balanced_pairs(
        self,
        n_pairs: int = 30,
        recursive_groups: Optional[List[str]] = None,
        baseline_groups: Optional[List[str]] = None,
        seed: int = 42,
    ) -> List[Tuple[str, str]]:
        """
        Generate balanced recursive/baseline prompt pairs.
        
        Args:
            n_pairs: Number of pairs to generate.
            recursive_groups: List of recursive groups to sample from.
                             Default: ["L3_deeper", "L4_full", "L5_refined"].
            baseline_groups: List of baseline groups to sample from.
                             Default: ["baseline_math", "baseline_factual", "baseline_creative"].
            seed: Random seed.
        
        Returns:
            List of (recursive_prompt, baseline_prompt) tuples.
        """
        rng = random.Random(seed)
        
        if recursive_groups is None:
            recursive_groups = ["L3_deeper", "L4_full", "L5_refined"]
        if baseline_groups is None:
            baseline_groups = ["baseline_math", "baseline_factual", "baseline_creative"]
        
        recursive = []
        baseline = []
        
        for k, v in self.prompts.items():
            if v.get("group") in recursive_groups:
                recursive.append(v["text"])
            elif v.get("group") in baseline_groups:
                baseline.append(v["text"])
        
        n_rec = min(n_pairs, len(recursive))
        n_base = min(n_pairs, len(baseline))
        
        sampled_rec = rng.sample(recursive, n_rec)
        sampled_base = rng.sample(baseline, n_base)
        
        pairs = []
        for i in range(min(n_rec, n_base)):
            pairs.append((sampled_rec[i], sampled_base[i]))
        
        return pairs
    
    def get_validated_pairs(
        self,
        n_pairs: int = 5,
        seed: int = 0,
    ) -> List[Tuple[str, str]]:
        """
        Get DEC8-validated recursive/baseline pairs.
        
        Args:
            n_pairs: Number of pairs (max 5).
            seed: Random seed (not used, returns fixed pairs).
        
        Returns:
            List of (recursive_prompt, baseline_prompt) tuples.
        """
        # DEC8 validated prompts
        recursive_keys = [f"L3_deeper_DEC8_{i:02d}" for i in range(1, 6)]
        baseline_keys = [f"baseline_instructional_{i:02d}" for i in range(1, 6)]
        
        pairs = []
        for i in range(min(n_pairs, 5)):
            rec_key = recursive_keys[i]
            base_key = baseline_keys[i]
            
            if rec_key in self.prompts and base_key in self.prompts:
                pairs.append((
                    self.prompts[rec_key]["text"],
                    self.prompts[base_key]["text"]
                ))
        
        return pairs


# Global instance for convenience
_loader = None


def get_loader() -> PromptLoader:
    """Get the global PromptLoader instance."""
    global _loader
    if _loader is None:
        _loader = PromptLoader()
    return _loader


# Convenience functions matching the old API
def get_prompts_by_pillar(
    pillar: str,
    limit: Optional[int] = None,
    seed: int = 0,
) -> List[str]:
    """Convenience function matching old API."""
    return get_loader().get_by_pillar(pillar, limit, seed)


def get_prompts_by_type(
    prompt_type: str,
    limit: Optional[int] = None,
    seed: int = 0,
) -> List[str]:
    """Convenience function matching old API."""
    return get_loader().get_by_type(prompt_type, limit, seed)


def get_validated_pairs(n_pairs: int = 5, seed: int = 0) -> List[Tuple[str, str]]:
    """Convenience function matching old API."""
    return get_loader().get_validated_pairs(n_pairs, seed)

