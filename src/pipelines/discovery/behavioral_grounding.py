"""
Behavioral grounding: does pushing the geometric regime change generation?

We generate under three conditions:
1) baseline (no patch)
2) baseline + residual patch (source=recursive prompt residual at patch_layer)
3) recursive (no patch)

We compute simple, falsifiable metrics on the generated continuation:
- self_ref_rate: fraction of words that match a marker list
- unique_word_ratio: unique_words / total_words
- repeat_4gram_frac: fraction of 4-grams that appear more than once

Artifacts:
- behavioral_grounding.jsonl (one record per condition per pair)
- behavioral_grounding_summary.csv (aggregated metrics)
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from prompts.loader import PromptLoader
from src.core.models import load_model, set_seed
from src.metrics.rv import participation_ratio
from src.pipelines.registry import ExperimentResult


SELF_REF_MARKERS_DEFAULT = [
    "itself",
    "self",
    "observer",
    "observed",
    "process",
    "aware",
    "awareness",
    "recursive",
    "recursion",
    "loop",
    "strange loop",
]


@dataclass
class PairSpec:
    rec_id: str
    base_id: str
    rec_group: str
    base_group: str


def _token_len(tokenizer, text: str) -> int:
    return int(len(tokenizer.encode(text)))


def _words(text: str) -> List[str]:
    return re.findall(r"[A-Za-z']+", text.lower())


def _self_ref_rate(text: str, markers: List[str]) -> float:
    ws = _words(text)
    if not ws:
        return 0.0
    # marker match: marker can be multiword; treat multiword markers by substring in text
    hits = 0
    text_l = text.lower()
    for w in ws:
        if w in ("the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "is", "are"):
            continue
        if any(m == w for m in markers if " " not in m):
            hits += 1
            continue
    # Add multiword markers as binary hits (to avoid double counting)
    for m in markers:
        if " " in m and m in text_l:
            hits += 3  # small weight
    return float(hits) / float(len(ws))


def _unique_word_ratio(text: str) -> float:
    ws = _words(text)
    if not ws:
        return 0.0
    return float(len(set(ws))) / float(len(ws))


def _repeat_ngram_frac(text: str, n: int = 4) -> float:
    ws = _words(text)
    if len(ws) < n:
        return 0.0
    grams = [" ".join(ws[i : i + n]) for i in range(len(ws) - n + 1)]
    if not grams:
        return 0.0
    from collections import Counter

    c = Counter(grams)
    repeated = sum(v for v in c.values() if v > 1)
    return float(repeated) / float(len(grams))


def _capture_resid(model, tokenizer, text: str, *, layer_idx: int, device: str, max_length: int) -> Optional[torch.Tensor]:
    """
    Capture residual stream input to block layer_idx. Returns (T,D) with batch removed.
    """
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    out: Dict[str, Optional[torch.Tensor]] = {"x": None}

    def hook_fn(_module, inputs):
        out["x"] = inputs[0].detach()[0]
        return None

    h = None
    try:
        h = model.model.layers[layer_idx].register_forward_pre_hook(hook_fn)
        with torch.no_grad():
            model(**enc)
    finally:
        if h is not None:
            h.remove()
    return out["x"]


def _make_patch(src: torch.Tensor, *, window: int) -> torch.Tensor:
    T = int(src.shape[0])
    W = min(int(window), T)
    if W <= 0:
        return src[:0]
    return src[-W:, :]


def _generate_with_optional_resid_patch(
    model,
    tokenizer,
    *,
    prompt: str,
    patch_layer: int,
    patch: Optional[torch.Tensor],
    window: int,
    max_new_tokens: int,
    device: str,
    max_length: int,
    do_sample: bool,
    temperature: float,
    capture_vproj: bool = False,
    vproj_layer: Optional[int] = None,
) -> Tuple[str, List[int]]:
    """
    IMPORTANT: We apply the residual patch ONLY during the prompt "push" forward pass,
    then generate unpatched using KV cache. This avoids the artifact where cached
    generation has T=1 and we effectively inject the same vector every step.
    """
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    input_ids = enc["input_ids"]
    base_attention_mask = enc.get("attention_mask", None)
    prompt_len = int(input_ids.shape[1])

    def hook_patch_resid(_module, inputs):
        if patch is None:
            return None
        hidden = inputs[0]
        hidden2 = hidden.clone()
        B, T, _D = hidden2.shape
        W = min(int(window), T, int(patch.shape[0]))
        if W > 0:
            hidden2[:, -W:, :] = patch[-W:, :].unsqueeze(0).expand(B, -1, -1)
        return (hidden2, *inputs[1:])

    # Optional: capture v-projection at a layer during the prompt pass and during generation.
    vproj_layer_idx = int(vproj_layer) if vproj_layer is not None else int(patch_layer)
    v_prompt: Optional[torch.Tensor] = None  # (T,D)
    v_gen_list: List[torch.Tensor] = []      # list of (D,)

    def hook_capture_vproj(_module, _inp, out):
        # out: (B,T,D) for prompt pass, (B,1,D) for cached generation
        if not capture_vproj:
            return out
        t = out.detach()
        if t.dim() == 3:
            if t.shape[1] == 1:
                v_gen_list.append(t[0, 0, :].clone().cpu())
            else:
                nonlocal v_prompt
                v_prompt = t[0].clone().cpu()
        return out

    # 1) Push pass (optional patch) to produce past_key_values + next-token logits
    handle = None
    handle_v = None
    try:
        if patch is not None:
            handle = model.model.layers[patch_layer].register_forward_pre_hook(hook_patch_resid)
        if capture_vproj:
            handle_v = model.model.layers[vproj_layer_idx].self_attn.v_proj.register_forward_hook(hook_capture_vproj)
        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                attention_mask=base_attention_mask,
                use_cache=True,
            )
    finally:
        if handle is not None:
            handle.remove()
        if handle_v is not None:
            handle_v.remove()

    past = out.past_key_values
    logits = out.logits[:, -1, :]  # next token distribution after prompt

    gen_ids: List[int] = []
    cur_token = None

    def sample_next(logits_1d: torch.Tensor) -> int:
        if not do_sample:
            return int(torch.argmax(logits_1d).item())
        t = max(float(temperature), 1e-6)
        probs = torch.softmax(logits_1d / t, dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())

    # 2) Autoregressive loop (UNPATCHED)
    next_id = sample_next(logits[0])
    gen_ids.append(next_id)
    cur_token = torch.tensor([[next_id]], device=device, dtype=input_ids.dtype)
    for _ in range(int(max_new_tokens) - 1):
        # attention mask must grow with generated length when using past_key_values
        if base_attention_mask is not None:
            gen_mask = torch.ones((1, len(gen_ids)), device=device, dtype=base_attention_mask.dtype)
            attn_mask = torch.cat([base_attention_mask, gen_mask], dim=1)
        else:
            attn_mask = None
        with torch.no_grad():
            handle_v_step = None
            try:
                if capture_vproj:
                    handle_v_step = model.model.layers[vproj_layer_idx].self_attn.v_proj.register_forward_hook(hook_capture_vproj)
                out2 = model(
                    input_ids=cur_token,
                    attention_mask=attn_mask,
                    past_key_values=past,
                    use_cache=True,
                )
            finally:
                if handle_v_step is not None:
                    handle_v_step.remove()
        past = out2.past_key_values
        next_id = sample_next(out2.logits[0, -1, :])
        gen_ids.append(next_id)
        cur_token = torch.tensor([[next_id]], device=device, dtype=input_ids.dtype)
        if next_id == tokenizer.eos_token_id:
            break

    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    # Stash capture results on the function for retrieval by caller (avoid changing return type)
    _generate_with_optional_resid_patch._last_v_prompt = v_prompt  # type: ignore[attr-defined]
    if v_gen_list:
        _generate_with_optional_resid_patch._last_v_gen = torch.stack(v_gen_list, dim=0)  # type: ignore[attr-defined]
    else:
        _generate_with_optional_resid_patch._last_v_gen = None  # type: ignore[attr-defined]
    return gen_text, gen_ids


def run_behavioral_grounding_from_config(cfg: Dict[str, Any], run_dir: Path) -> ExperimentResult:
    model_cfg = cfg.get("model") or {}
    params = cfg.get("params") or {}

    seed = int(cfg.get("seed") or 0)
    device = str(model_cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_name = str(model_cfg.get("name") or "mistralai/Mistral-7B-v0.1")
    set_seed(seed)

    patch_layer = int(params.get("patch_layer") or 24)
    patch_layers = params.get("patch_layers")
    if isinstance(patch_layers, list) and patch_layers:
        patch_layers_list = [int(x) for x in patch_layers]
    else:
        patch_layers_list = [int(patch_layer)]

    window = int(params.get("window") or 16)
    max_new_tokens = int(params.get("max_new_tokens") or 120)
    max_length = int(params.get("max_length") or 512)
    do_sample = bool(params.get("do_sample", False))
    temperature = float(params.get("temperature") or 0.8)

    markers = list(params.get("markers") or SELF_REF_MARKERS_DEFAULT)

    max_pairs = int(params.get("max_pairs") or 8)
    override_baseline_text = params.get("override_baseline_text")
    override_recursive_text = params.get("override_recursive_text")
    # Pair selection mode (preferred): balanced pairs sampler
    pair_mode = str(params.get("pair_mode") or "aligned_groups")
    pairing = params.get("pairing") or {}
    recursive_groups = pairing.get("recursive_groups") or ["L5_refined", "L4_full", "L3_deeper"]
    baseline_groups = pairing.get("baseline_groups") or ["long_control", "baseline_creative", "baseline_math"]

    model, tokenizer = load_model(model_name, device=device)
    loader = PromptLoader()
    bank_version = loader.version
    (run_dir / "prompt_bank_version.txt").write_text(bank_version)
    (run_dir / "prompt_bank_version.json").write_text(
        json.dumps({"version": bank_version}, indent=2) + "\n"
    )
    bank = loader.prompts

    if isinstance(override_baseline_text, str) and isinstance(override_recursive_text, str):
        pairs = [PairSpec(rec_id="override_recursive", base_id="override_baseline", rec_group="override", base_group="override")]
        bank = {
            "override_recursive": {"text": override_recursive_text, "group": "override"},
            "override_baseline": {"text": override_baseline_text, "group": "override"},
        }
        max_pairs = 1
    else:
        if pair_mode == "balanced_pairs":
            # Get varied pairs by sampling groups (no aligned indexing assumptions)
            text_pairs = loader.get_balanced_pairs(
                n_pairs=max_pairs,
                recursive_groups=list(recursive_groups),
                baseline_groups=list(baseline_groups),
                seed=seed,
            )
            pairs = []
            # store texts directly in bank under synthetic ids
            bank = {}
            for i, (rec_t, base_t) in enumerate(text_pairs):
                rid = f"rec_{i:04d}"
                bid = f"base_{i:04d}"
                bank[rid] = {"text": rec_t, "group": "balanced_recursive"}
                bank[bid] = {"text": base_t, "group": "balanced_baseline"}
                if _token_len(tokenizer, base_t) >= max(window, 8):
                    pairs.append(PairSpec(rec_id=rid, base_id=bid, rec_group="balanced_recursive", base_group="balanced_baseline"))
            pairs = pairs[:max_pairs]
        else:
            # Legacy: aligned indices per group
            candidates: List[PairSpec] = []
            for rec_group in recursive_groups:
                rec_ids = [k for k, v in bank.items() if v.get("group") == rec_group]
                for base_group in baseline_groups:
                    base_ids = [k for k, v in bank.items() if v.get("group") == base_group]
                    for i in range(min(len(rec_ids), len(base_ids))):
                        candidates.append(PairSpec(rec_id=rec_ids[i], base_id=base_ids[i], rec_group=rec_group, base_group=base_group))

            rng = np.random.default_rng(seed + 20251213)
            rng.shuffle(candidates)

            pairs = []
            for spec in candidates:
                if _token_len(tokenizer, bank[spec.base_id]["text"]) >= max(window, 8):
                    pairs.append(spec)
                if len(pairs) >= max_pairs:
                    break

    jsonl_path = run_dir / "behavioral_grounding.jsonl"
    csv_path = run_dir / "behavioral_grounding_summary.csv"
    rows_summary: List[Dict[str, Any]] = []

    with jsonl_path.open("w", encoding="utf-8") as jf:
        for spec in pairs:
            rec_text = bank[spec.rec_id]["text"]
            base_text = bank[spec.base_id]["text"]

            for pl in patch_layers_list:
                # Patch source: capture recursive residual at patch layer
                resid_src = _capture_resid(model, tokenizer, rec_text, layer_idx=int(pl), device=device, max_length=max_length)
                patch = _make_patch(resid_src, window=window) if resid_src is not None else None

                conditions = [
                    ("baseline", base_text, None),
                    ("baseline_patched", base_text, patch),
                    ("recursive", rec_text, None),
                ]

                for cond, prompt, p in conditions:
                    gen_text, gen_ids = _generate_with_optional_resid_patch(
                        model,
                        tokenizer,
                        prompt=prompt,
                        patch_layer=int(pl),
                        patch=p,
                        window=window,
                        max_new_tokens=max_new_tokens,
                        device=device,
                        max_length=max_length,
                        do_sample=do_sample,
                        temperature=temperature,
                        capture_vproj=True,
                        vproj_layer=int(pl),
                    )

                    v_prompt = getattr(_generate_with_optional_resid_patch, "_last_v_prompt", None)
                    v_gen = getattr(_generate_with_optional_resid_patch, "_last_v_gen", None)
                    pr_prompt_v = participation_ratio(v_prompt, window_size=window) if v_prompt is not None else float("nan")
                    pr_gen_v = participation_ratio(v_gen, window_size=window) if v_gen is not None else float("nan")

                    metrics = {
                        "self_ref_rate": _self_ref_rate(gen_text, markers),
                        "unique_word_ratio": _unique_word_ratio(gen_text),
                        "repeat_4gram_frac": _repeat_ngram_frac(gen_text, n=4),
                        "non_ws_chars": int(len(re.sub(r"\s+", "", gen_text))),
                        "gen_chars": int(len(gen_text)),
                        "pr_prompt_v": float(pr_prompt_v),
                        "pr_gen_v": float(pr_gen_v),
                    }

                    rec = {
                        "rec_id": spec.rec_id,
                        "base_id": spec.base_id,
                        "rec_group": spec.rec_group,
                        "base_group": spec.base_group,
                        "condition": cond,
                        "patch_layer": int(pl),
                        "window": int(window),
                        "max_new_tokens": int(max_new_tokens),
                        "do_sample": bool(do_sample),
                        "temperature": float(temperature),
                        "prompt_preview": prompt[:200],
                        "gen_text": gen_text,
                        "gen_token_count": int(len(gen_ids)),
                        "metrics": metrics,
                    }
                    jf.write(json.dumps(rec, ensure_ascii=False) + "\n")

                    rows_summary.append(
                        {
                            "rec_id": spec.rec_id,
                            "base_id": spec.base_id,
                            "rec_group": spec.rec_group,
                            "base_group": spec.base_group,
                            "condition": cond,
                            "patch_layer": int(pl),
                            "window": int(window),
                            "self_ref_rate": float(metrics["self_ref_rate"]),
                            "unique_word_ratio": float(metrics["unique_word_ratio"]),
                            "repeat_4gram_frac": float(metrics["repeat_4gram_frac"]),
                            "pr_prompt_v": float(metrics["pr_prompt_v"]),
                            "pr_gen_v": float(metrics["pr_gen_v"]),
                            "gen_token_count": int(len(gen_ids)),
                        }
                    )

    # Write summary CSV
    import pandas as pd

    df = pd.DataFrame(rows_summary)
    df.to_csv(csv_path, index=False)

    # Aggregate means by condition
    by_cond: Dict[str, Any] = {}
    for cond in sorted(df["condition"].unique().tolist()):
        sub = df[df["condition"] == cond]
        by_cond[cond] = {
            "n": int(len(sub)),
            "self_ref_rate_mean": float(sub["self_ref_rate"].mean()),
            "unique_word_ratio_mean": float(sub["unique_word_ratio"].mean()),
            "repeat_4gram_frac_mean": float(sub["repeat_4gram_frac"].mean()),
            "pr_prompt_v_mean": float(sub["pr_prompt_v"].mean()),
            "pr_gen_v_mean": float(sub["pr_gen_v"].mean()),
            "gen_token_count_mean": float(sub["gen_token_count"].mean()),
        }

    summary = {
        "experiment": "behavioral_grounding",
        "model_name": model_name,
        "device": device,
        "prompt_bank_version": bank_version,
        "params": {
            "patch_layer": patch_layer,
            "window": window,
            "max_new_tokens": max_new_tokens,
            "max_pairs": max_pairs,
            "do_sample": do_sample,
            "temperature": temperature,
            "pairing": {"recursive_groups": recursive_groups, "baseline_groups": baseline_groups},
        },
        "by_condition": by_cond,
        "artifacts": {"jsonl": str(jsonl_path), "csv": str(csv_path)},
    }

    return ExperimentResult(summary=summary)


