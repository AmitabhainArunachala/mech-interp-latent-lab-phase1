"""
Frozen prompt subset utilities.

These helpers resolve a frozen subset manifest against prompts/bank.json and
fail closed if the bank hash no longer matches the frozen contract.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from prompts.loader import PromptLoader


DEFAULT_MISTRAL_HARDENING_SUBSET_PATH = Path(__file__).parent / "subsets" / "mistral_hardening_v1.json"
DEFAULT_MODE_ATLAS_SUBSET_PATH = Path(__file__).parent / "subsets" / "mode_atlas_v1.json"


@dataclass(frozen=True)
class FrozenPromptSubset:
    """
    Resolve a frozen prompt subset manifest against PromptLoader.

    The manifest is expected to contain:
    - source_bank_version
    - tiers with either selection_rules or explicit prompt-id lists
    """

    manifest_path: Path
    manifest: Dict[str, Any]
    loader: PromptLoader

    @classmethod
    def load(
        cls,
        manifest_path: Path | str,
        loader: Optional[PromptLoader] = None,
    ) -> "FrozenPromptSubset":
        path = Path(manifest_path)
        if not path.exists():
            raise FileNotFoundError(f"Frozen subset manifest not found: {path}")

        manifest = json.loads(path.read_text(encoding="utf-8"))
        prompt_loader = loader or PromptLoader()
        subset = cls(manifest_path=path, manifest=manifest, loader=prompt_loader)
        subset.validate_bank_version()
        return subset

    @property
    def name(self) -> str:
        return str(self.manifest.get("name", self.manifest_path.stem))

    @property
    def schema_version(self) -> str:
        return str(self.manifest.get("schema_version", "unknown"))

    @property
    def source_bank_version(self) -> str:
        return str(self.manifest.get("source_bank_version", "unknown"))

    def validate_bank_version(self) -> None:
        actual = self.loader.version
        expected = self.source_bank_version
        if actual != expected:
            raise ValueError(
                "Frozen prompt subset hash mismatch: "
                f"expected bank version {expected}, got {actual}. "
                f"Subset file: {self.manifest_path}"
            )

    def get_prompt_ids_for_tier(self, tier_name: str) -> List[str]:
        tiers = self.manifest.get("tiers", {})
        if tier_name not in tiers:
            raise KeyError(f"Unknown tier '{tier_name}' in {self.manifest_path}")

        tier = tiers[tier_name]
        prompt_ids: List[str] = []

        for rule in tier.get("selection_rules", []):
            prompt_ids.extend(self._resolve_selection_rule(rule))

        for key, value in tier.items():
            if key.endswith("_prompt_ids"):
                prompt_ids.extend(value)

        # Preserve order while removing duplicates
        deduped: List[str] = []
        seen = set()
        for prompt_id in prompt_ids:
            if prompt_id not in seen:
                deduped.append(prompt_id)
                seen.add(prompt_id)
        return deduped

    def get_records_for_tier(self, tier_name: str) -> List[Tuple[str, Dict[str, Any]]]:
        records: List[Tuple[str, Dict[str, Any]]] = []
        for prompt_id in self.get_prompt_ids_for_tier(tier_name):
            if prompt_id not in self.loader.prompts:
                raise KeyError(
                    f"Prompt id '{prompt_id}' from tier '{tier_name}' "
                    f"was not found in {self.loader.bank_path}"
                )
            records.append((prompt_id, self.loader.prompts[prompt_id]))
        return records

    def _resolve_selection_rule(self, rule: Dict[str, Any]) -> List[str]:
        group = rule.get("group")
        if not group:
            raise ValueError(f"Selection rule missing 'group': {rule}")

        exclude_ids = set(rule.get("exclude_prompt_ids", []))
        include_ids = rule.get("include_prompt_ids")

        if include_ids:
            selected = [prompt_id for prompt_id in include_ids if prompt_id not in exclude_ids]
        else:
            selected = [
                prompt_id
                for prompt_id in sorted(self.loader.prompts.keys())
                if self.loader.prompts[prompt_id].get("group") == group
                and prompt_id not in exclude_ids
            ]

        limit = rule.get("limit")
        if limit is not None:
            selected = selected[: int(limit)]
        return selected


def load_default_mistral_hardening_subset(
    loader: Optional[PromptLoader] = None,
) -> FrozenPromptSubset:
    """Load the frozen Mistral hardening prompt contract."""
    return FrozenPromptSubset.load(DEFAULT_MISTRAL_HARDENING_SUBSET_PATH, loader=loader)


def load_default_mode_atlas_subset(
    loader: Optional[PromptLoader] = None,
) -> FrozenPromptSubset:
    """Load the frozen bank-backed mode atlas contract."""
    return FrozenPromptSubset.load(DEFAULT_MODE_ATLAS_SUBSET_PATH, loader=loader)


def split_tier_records_by_pillar(
    subset: FrozenPromptSubset,
    tier_name: str,
    *,
    recursive_pillars: Tuple[str, ...] = ("dose_response",),
    baseline_pillars: Tuple[str, ...] = ("baselines",),
) -> Dict[str, List[Tuple[str, Dict[str, Any]]]]:
    """
    Split a frozen tier into recursive and baseline records using prompt-bank pillars.

    This keeps paper-facing scripts aligned on the same condition definitions instead
    of re-encoding group lists independently in each experiment file.
    """
    recursive_records: List[Tuple[str, Dict[str, Any]]] = []
    baseline_records: List[Tuple[str, Dict[str, Any]]] = []
    other_records: List[Tuple[str, Dict[str, Any]]] = []

    for prompt_id, record in subset.get_records_for_tier(tier_name):
        pillar = record.get("pillar")
        if pillar in recursive_pillars:
            recursive_records.append((prompt_id, record))
        elif pillar in baseline_pillars:
            baseline_records.append((prompt_id, record))
        else:
            other_records.append((prompt_id, record))

    return {
        "recursive": recursive_records,
        "baseline": baseline_records,
        "other": other_records,
    }
