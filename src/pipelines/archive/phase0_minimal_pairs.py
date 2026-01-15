"""
Phase 0 Pipeline A: Minimal pairs (semantics vs syntax) + champion ablations.

Goal:
- Estimate R_V sensitivity to surface form when "meaning" is held approximately constant.
- Provide a clean, reproducible CSV artifact under results/phase0_metric_validation/runs/...

This is intentionally small-N and surgical.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

from prompts.loader import PromptLoader
from src.core.models import load_model, set_seed
from src.metrics.rv import participation_ratio
from src.pipelines.registry import ExperimentResult


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class Variant:
    semantic_id: str
    variant_id: str
    prompt_id: str
    text: str


def _capture_vproj_multi(model, tokenizer, text: str, layer_idxs: Sequence[int], device: str, max_length: int) -> Dict[int, torch.Tensor | None]:
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    storage: Dict[int, torch.Tensor | None] = {i: None for i in layer_idxs}
    handles = []

    def make_hook(layer_idx: int):
        def hook_fn(_module, _inp, out):
            storage[layer_idx] = out.detach()[0]  # (seq, d)
            return out

        return hook_fn

    for idx in layer_idxs:
        layer = model.model.layers[idx].self_attn
        handles.append(layer.v_proj.register_forward_hook(make_hook(idx)))

    try:
        with torch.no_grad():
            model(**enc)
    finally:
        for h in handles:
            h.remove()

    return storage


def _compute_rv_from_v(v_early: torch.Tensor | None, v_late: torch.Tensor | None, window: int) -> float:
    pr_e = participation_ratio(v_early, window_size=window)
    pr_l = participation_ratio(v_late, window_size=window)
    if pr_e == 0 or np.isnan(pr_e) or np.isnan(pr_l):
        return float("nan")
    return float(pr_l / pr_e)


def _default_variants_from_bank(loader: PromptLoader, max_per_bucket: int = 6) -> List[Variant]:
    """
    DEC15 prompt hygiene: no hardcoded prompt strings.

    We build "minimal pairs" from existing bank metadata:
    - A paraphrase bucket: `hybrid_l5_math_01` + any prompts with `is_paraphrase_of == hybrid_l5_math_01`
    - A control bucket: a deterministic sample from a control group
    - A baseline bucket: a deterministic sample from an instructional baseline group
    """
    bank = loader.prompts

    def _by_group(group: str, n: int) -> List[str]:
        ids = [k for k, v in bank.items() if v.get("group") == group and v.get("text")]
        ids.sort()
        return ids[:n]

    out: List[Variant] = []

    # Bucket 1: paraphrases of a known champion
    champ_id = "hybrid_l5_math_01"
    paraphrase_ids = [champ_id]
    for k, v in bank.items():
        if v.get("is_paraphrase_of") == champ_id and v.get("text"):
            paraphrase_ids.append(k)
    paraphrase_ids = sorted(set(paraphrase_ids))[: int(max_per_bucket)]
    for i, pid in enumerate(paraphrase_ids):
        out.append(Variant("champion_paraphrases", f"v{i+1}", pid, str(bank[pid]["text"])))

    # Bucket 2: pseudo-recursive controls (deterministic sample)
    for i, pid in enumerate(_by_group("control_pseudo_recursive", max_per_bucket)):
        out.append(Variant("controls_pseudo_recursive", f"v{i+1}", pid, str(bank[pid]["text"])))

    # Bucket 3: instructional baselines (deterministic sample)
    for i, pid in enumerate(_by_group("baseline_instructional", max_per_bucket)):
        out.append(Variant("baselines_instructional", f"v{i+1}", pid, str(bank[pid]["text"])))

    if not out:
        raise RuntimeError("No variants could be constructed from prompts/bank.json (unexpected).")
    return out


def run_phase0_minimal_pairs_from_config(cfg: Dict[str, Any], run_dir: Path) -> ExperimentResult:
    model_cfg = cfg.get("model") or {}
    params = cfg.get("params") or {}

    seed = int(cfg.get("seed") or 0)
    device = str(model_cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_name = str(model_cfg.get("name") or "mistralai/Mistral-7B-v0.1")

    early_layer = int(params.get("early_layer") or 5)
    late_layers = params.get("late_layers") or [25, 27]
    late_layers = [int(x) for x in late_layers]
    window = int(params.get("window") or 16)
    max_length = int(params.get("max_length") or 512)
    n_repeats = int(params.get("n_repeats") or 1)

    set_seed(seed)
    model, tokenizer = load_model(model_name, device=device)

    loader = PromptLoader()
    bank_version = loader.version
    (run_dir / "prompt_bank_version.txt").write_text(bank_version)
    (run_dir / "prompt_bank_version.json").write_text(
        json.dumps({"version": bank_version}, indent=2) + "\n"
    )

    variants = _default_variants_from_bank(loader, max_per_bucket=int(params.get("max_per_bucket") or 6))

    rows: List[Dict[str, Any]] = []
    for v in variants:
        for rep in range(n_repeats):
            vs = _capture_vproj_multi(model, tokenizer, v.text, [early_layer, *late_layers], device=device, max_length=max_length)
            v_e = vs[early_layer]
            for late in late_layers:
                rv = _compute_rv_from_v(v_e, vs[late], window=window)
                rows.append(
                    {
                        "semantic_id": v.semantic_id,
                        "variant_id": v.variant_id,
                        "prompt_id": v.prompt_id,
                        "repeat": rep,
                        "late_layer": late,
                        "early_layer": early_layer,
                        "window": window,
                        "text_len_chars": len(v.text),
                        "text_sha": _sha(v.text),
                        "rv": rv,
                    }
                )

    out_csv = run_dir / "phase0_minimal_pairs.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Summaries: within-semantic variance and overall min/mean per layer
    df = rows
    by_sem: Dict[str, Dict[str, Any]] = {}
    for sem in sorted({r["semantic_id"] for r in df}):
        sem_rows = [r for r in df if r["semantic_id"] == sem]
        by_layer: Dict[str, Any] = {}
        for late in sorted({r["late_layer"] for r in sem_rows}):
            vals = np.asarray([r["rv"] for r in sem_rows if r["late_layer"] == late and not np.isnan(r["rv"])], dtype=np.float64)
            by_layer[str(late)] = {
                "n": int(vals.size),
                "mean": float(vals.mean()) if vals.size else float("nan"),
                "std": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
                "min": float(vals.min()) if vals.size else float("nan"),
                "max": float(vals.max()) if vals.size else float("nan"),
            }
        by_sem[sem] = {"by_late_layer": by_layer}

    summary = {
        "experiment": "phase0_minimal_pairs",
        "model_name": model_name,
        "device": device,
        "seed": seed,
        "prompt_bank_version": bank_version,
        "params": {
            "early_layer": early_layer,
            "late_layers": late_layers,
            "window": window,
            "max_length": max_length,
            "n_repeats": n_repeats,
            "max_per_bucket": int(params.get("max_per_bucket") or 6),
        },
        "n_rows": len(rows),
        "semantic_groups": by_sem,
        "artifacts": {"csv": str(out_csv)},
    }

    return ExperimentResult(summary=summary)


