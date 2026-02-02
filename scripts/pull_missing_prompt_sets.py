#!/usr/bin/env python3
"""
Pull missing prompt sets into prompts/bank.json with provenance.

Design goals:
- Never overwrite existing prompts
- De-duplicate by normalized text hash (casefold + whitespace squash)
- Keep canonical pillars/groups clean by putting legacy sources under dedicated pillars
- Create a timestamped backup before writing

Run:
  python3 scripts/pull_missing_prompt_sets.py
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


ROOT = Path(__file__).resolve().parent.parent
BANK_PATH = ROOT / "prompts" / "bank.json"


def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip()).casefold()


def _text_hash16(s: str) -> str:
    return hashlib.sha256(_norm_text(s).encode("utf-8")).hexdigest()[:16]


def _safe_slug(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_]+", "_", s.strip())
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:80] if s else "unknown"


def _extract_prompt_dict_from_py(py_path: Path, dict_var_name: str) -> Dict[str, Dict[str, Any]]:
    """
    Extract dict_var_name[\"key\"] = { ... } assignments (AST, no execution).
    Returns: {key: dict_literal}
    """
    src = py_path.read_text(encoding="utf-8", errors="ignore")
    tree = ast.parse(src)
    out: Dict[str, Dict[str, Any]] = {}

    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        tgt = node.targets[0]
        if not (
            isinstance(tgt, ast.Subscript)
            and isinstance(tgt.value, ast.Name)
            and tgt.value.id == dict_var_name
        ):
            continue
        if not (isinstance(tgt.slice, ast.Constant) and isinstance(tgt.slice.value, str)):
            continue
        key = tgt.slice.value
        if not isinstance(node.value, ast.Dict):
            continue
        try:
            d = ast.literal_eval(node.value)
        except Exception:
            continue
        if isinstance(d, dict):
            out[key] = d
    return out


def _extract_list_of_strings_from_py(py_path: Path, var_name: str) -> List[str]:
    """
    Extract `var_name = ["...", "..."]` (AST, no execution).
    """
    src = py_path.read_text(encoding="utf-8", errors="ignore")
    tree = ast.parse(src)
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        if not (isinstance(node.targets[0], ast.Name) and node.targets[0].id == var_name):
            continue
        if not isinstance(node.value, (ast.List, ast.Tuple)):
            continue
        out: List[str] = []
        for elt in node.value.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                s = elt.value.strip()
                if len(s) >= 1:
                    out.append(s)
        return out
    return []


@dataclass(frozen=True)
class ImportResult:
    added: int
    skipped_existing_text: int


def _add_prompts_to_bank(
    bank: Dict[str, Dict[str, Any]],
    prompts: Iterable[Tuple[str, Dict[str, Any]]],
    *,
    id_prefix: str,
    source_path: str,
    source_kind: str,
    default_pillar: Optional[str] = None,
    default_group: Optional[str] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> ImportResult:
    """
    prompts: (source_key, prompt_meta_with_text)
    """
    existing_hashes = {_text_hash16(v["text"]) for v in bank.values() if isinstance(v.get("text"), str)}
    added = 0
    skipped = 0
    ts = datetime.now().strftime("%Y-%m-%d")

    for source_key, meta in prompts:
        text = meta.get("text")
        if not isinstance(text, str) or len(text.strip()) == 0:
            continue
        th = _text_hash16(text)
        if th in existing_hashes:
            skipped += 1
            continue

        # Create stable-ish ID: prefix + slug(source_key) + hash
        prompt_id = f"{id_prefix}_{_safe_slug(source_key)}_{th}"
        if prompt_id in bank:
            prompt_id = f"{prompt_id}_{len(bank)}"

        new_meta = dict(meta)
        new_meta.setdefault("pillar", default_pillar)
        new_meta.setdefault("group", default_group)
        new_meta["source_file"] = source_path
        new_meta["source_key"] = source_key
        new_meta["source_kind"] = source_kind
        new_meta["imported_date"] = ts
        if extra_meta:
            for k, v in extra_meta.items():
                new_meta.setdefault(k, v)

        bank[prompt_id] = new_meta
        existing_hashes.add(th)
        added += 1

    return ImportResult(added=added, skipped_existing_text=skipped)


def main() -> None:
    if not BANK_PATH.exists():
        raise FileNotFoundError(f"Bank not found: {BANK_PATH}")

    bank: Dict[str, Dict[str, Any]] = json.loads(BANK_PATH.read_text(encoding="utf-8"))

    # Backup first
    backup_path = BANK_PATH.parent / f"bank.json.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_path.write_text(json.dumps(bank, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[bank] loaded {len(bank)} prompts")
    print(f"[bank] backup -> {backup_path.name}")

    total_added = 0

    # 1) Alternative self-reference taxonomy (Gödelian, strange loops, surrender, etc.)
    alt_path = ROOT / "REUSABLE_PROMPT_BANK" / "alternative_self_reference.py"
    alt_dict = _extract_prompt_dict_from_py(alt_path, "alternative_prompts")
    alt_items = list(alt_dict.items())
    res = _add_prompts_to_bank(
        bank,
        alt_items,
        id_prefix="altselfref",
        source_path=str(alt_path.relative_to(ROOT)),
        source_kind="dict:alternative_prompts",
        default_pillar="alternative_self_reference",
        default_group=None,
        extra_meta={"original_pillar": "REUSABLE_PROMPT_BANK.alternative_self_reference.pillar"},
    )
    print(
        f"[import] alternative_self_reference: added={res.added}, skipped_existing_text={res.skipped_existing_text}"
    )
    total_added += res.added

    # 2) Dose-response (legacy variants) - avoid contaminating canonical ladder
    dr_path = ROOT / "REUSABLE_PROMPT_BANK" / "dose_response.py"
    dr_dict = _extract_prompt_dict_from_py(dr_path, "dose_response_prompts")
    dr_items: List[Tuple[str, Dict[str, Any]]] = []
    for k, v in dr_dict.items():
        # Keep original group/level in metadata, but store under legacy pillar for selection safety.
        vv = dict(v)
        vv["original_group"] = v.get("group")
        vv["original_pillar"] = v.get("pillar")
        vv["pillar"] = "dose_response_legacy"
        vv["group"] = f"{v.get('group','dose_response')}_legacy"
        dr_items.append((k, vv))
    res = _add_prompts_to_bank(
        bank,
        dr_items,
        id_prefix="doselegacy",
        source_path=str(dr_path.relative_to(ROOT)),
        source_kind="dict:dose_response_prompts",
    )
    print(f"[import] dose_response_legacy: added={res.added}, skipped_existing_text={res.skipped_existing_text}")
    total_added += res.added

    # 3) comprehensive_circuit_test embedded prompt lists (historical exact prompts)
    cct_path = ROOT / "comprehensive_circuit_test.py"
    champ = _extract_list_of_strings_from_py(cct_path, "CHAMPION_PROMPTS")
    base = _extract_list_of_strings_from_py(cct_path, "BASELINE_PROMPTS")
    champ_items = [(f"CHAMPION_PROMPTS[{i}]", {"text": s, "pillar": "legacy", "group": "legacy_comprehensive_circuit_test_champions"}) for i, s in enumerate(champ)]
    base_items = [(f"BASELINE_PROMPTS[{i}]", {"text": s, "pillar": "legacy", "group": "legacy_comprehensive_circuit_test_baselines"}) for i, s in enumerate(base)]
    res = _add_prompts_to_bank(
        bank,
        champ_items,
        id_prefix="legacycct",
        source_path=str(cct_path.relative_to(ROOT)),
        source_kind="list:CHAMPION_PROMPTS",
    )
    print(f"[import] comprehensive_circuit_test champions: added={res.added}, skipped_existing_text={res.skipped_existing_text}")
    total_added += res.added
    res = _add_prompts_to_bank(
        bank,
        base_items,
        id_prefix="legacycct",
        source_path=str(cct_path.relative_to(ROOT)),
        source_kind="list:BASELINE_PROMPTS",
    )
    print(f"[import] comprehensive_circuit_test baselines: added={res.added}, skipped_existing_text={res.skipped_existing_text}")
    total_added += res.added

    # Write bank
    BANK_PATH.write_text(json.dumps(bank, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[bank] wrote {len(bank)} prompts (added {total_added})")


if __name__ == "__main__":
    main()










