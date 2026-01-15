"""
Behavioral grounding (batch): run many prompt pairs across multiple patch layers.

Goal: n=100+ to "see it in action" and quantify collapse/degeneracy per patch layer.

Design:
- Sample balanced (recursive, baseline) pairs from prompt bank
- For each pair and each patch_layer:
  - capture recursive residual at patch_layer, take last W tokens as patch
  - generate on baseline prompt with:
      (a) no patch
      (b) push-only residual patch during prompt pass, then generate unpatched
  - (optional) generate recursive prompt as reference (off by default)

Artifacts:
- behavioral_grounding_batch.jsonl
- behavioral_grounding_batch_summary.csv (one row per sample)
- summary.json includes aggregated stats by (patch_layer, condition)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch

from prompts.loader import PromptLoader
from src.core.models import load_model, set_seed
from src.pipelines.registry import ExperimentResult

# Reuse the proven helpers from the single-run pipeline
from src.pipelines.discovery.behavioral_grounding import (  # noqa: F401
    SELF_REF_MARKERS_DEFAULT,
    _capture_resid,
    _generate_with_optional_resid_patch,
    _make_patch,
    _repeat_ngram_frac,
    _self_ref_rate,
    _unique_word_ratio,
)


def run_behavioral_grounding_batch_from_config(cfg: Dict[str, Any], run_dir: Path) -> ExperimentResult:
    model_cfg = cfg.get("model") or {}
    params = cfg.get("params") or {}

    seed = int(cfg.get("seed") or 0)
    device = str(model_cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_name = str(model_cfg.get("name") or "mistralai/Mistral-7B-v0.1")

    set_seed(seed)
    model, tokenizer = load_model(model_name, device=device)

    # Generation params
    window = int(params.get("window") or 32)
    patch_layers = params.get("patch_layers") or [24, 25, 26, 27]
    patch_layers = [int(x) for x in patch_layers]
    max_new_tokens = int(params.get("max_new_tokens") or 160)
    max_length = int(params.get("max_length") or 512)
    do_sample = bool(params.get("do_sample", True))
    temperature = float(params.get("temperature") or 0.7)

    markers = list(params.get("markers") or SELF_REF_MARKERS_DEFAULT)

    # Pair sampling
    pairing = params.get("pairing") or {}
    n_pairs = int(pairing.get("n_pairs") or 100)
    recursive_groups = pairing.get("recursive_groups")  # may be None to use defaults
    baseline_groups = pairing.get("baseline_groups")  # may be None to use defaults

    include_recursive_generation = bool(params.get("include_recursive_generation", False))

    loader = PromptLoader()
    bank_version = loader.version
    (run_dir / "prompt_bank_version.txt").write_text(bank_version)
    (run_dir / "prompt_bank_version.json").write_text(
        json.dumps({"version": bank_version}, indent=2) + "\n"
    )
    pairs: List[Tuple[str, str]] = loader.get_balanced_pairs(
        n_pairs=n_pairs,
        recursive_groups=recursive_groups,
        baseline_groups=baseline_groups,
        seed=seed,
    )

    jsonl_path = run_dir / "behavioral_grounding_batch.jsonl"
    csv_path = run_dir / "behavioral_grounding_batch_summary.csv"

    rows: List[Dict[str, Any]] = []
    with jsonl_path.open("w", encoding="utf-8") as jf:
        for pair_idx, (rec_prompt, base_prompt) in enumerate(pairs):
            for patch_layer in patch_layers:
                resid_src = _capture_resid(
                    model, tokenizer, rec_prompt, layer_idx=patch_layer, device=device, max_length=max_length
                )
                patch = _make_patch(resid_src, window=window) if resid_src is not None else None

                conditions = [
                    ("baseline", base_prompt, None),
                    ("baseline_patched", base_prompt, patch),
                ]
                if include_recursive_generation:
                    conditions.append(("recursive", rec_prompt, None))

                for cond, prompt, p in conditions:
                    gen_text, gen_ids = _generate_with_optional_resid_patch(
                        model,
                        tokenizer,
                        prompt=prompt,
                        patch_layer=patch_layer,
                        patch=p,
                        window=window,
                        max_new_tokens=max_new_tokens,
                        device=device,
                        max_length=max_length,
                        do_sample=do_sample,
                        temperature=temperature,
                    )

                    metrics = {
                        "self_ref_rate": _self_ref_rate(gen_text, markers),
                        "unique_word_ratio": _unique_word_ratio(gen_text),
                        "repeat_4gram_frac": _repeat_ngram_frac(gen_text, n=4),
                        "gen_chars": int(len(gen_text)),
                        "gen_token_count": int(len(gen_ids)),
                    }

                    rec = {
                        "pair_idx": int(pair_idx),
                        "condition": cond,
                        "patch_layer": int(patch_layer),
                        "window": int(window),
                        "do_sample": bool(do_sample),
                        "temperature": float(temperature),
                        "rec_prompt_preview": rec_prompt[:200],
                        "base_prompt_preview": base_prompt[:200],
                        "gen_text": gen_text,
                        "metrics": metrics,
                    }
                    jf.write(json.dumps(rec, ensure_ascii=False) + "\n")

                    rows.append(
                        {
                            "pair_idx": int(pair_idx),
                            "condition": cond,
                            "patch_layer": int(patch_layer),
                            "window": int(window),
                            "self_ref_rate": float(metrics["self_ref_rate"]),
                            "unique_word_ratio": float(metrics["unique_word_ratio"]),
                            "repeat_4gram_frac": float(metrics["repeat_4gram_frac"]),
                            "gen_token_count": int(metrics["gen_token_count"]),
                        }
                    )

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    # Aggregate summary
    agg = (
        df.groupby(["patch_layer", "condition"])[
            ["self_ref_rate", "unique_word_ratio", "repeat_4gram_frac", "gen_token_count"]
        ]
        .mean()
        .reset_index()
    )
    agg_dict: Dict[str, Any] = {}
    for _, r in agg.iterrows():
        key = f"L{int(r['patch_layer'])}:{str(r['condition'])}"
        agg_dict[key] = {
            "self_ref_rate_mean": float(r["self_ref_rate"]),
            "unique_word_ratio_mean": float(r["unique_word_ratio"]),
            "repeat_4gram_frac_mean": float(r["repeat_4gram_frac"]),
            "gen_token_count_mean": float(r["gen_token_count"]),
        }

    summary = {
        "experiment": "behavioral_grounding_batch",
        "model_name": model_name,
        "device": device,
        "prompt_bank_version": bank_version,
        "n_pairs": int(len(pairs)),
        "patch_layers": patch_layers,
        "window": int(window),
        "do_sample": bool(do_sample),
        "temperature": float(temperature),
        "include_recursive_generation": bool(include_recursive_generation),
        "artifacts": {"jsonl": str(jsonl_path), "csv": str(csv_path)},
        "by_layer_condition": agg_dict,
    }

    return ExperimentResult(summary=summary)



