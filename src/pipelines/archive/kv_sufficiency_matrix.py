"""
KV sufficiency matrix: settle the 'random KV anomaly' with a fixed condition set.

Conditions (per user request):
- A: control (no intervention)
- B: KV from recursive prompt
- C: KV from baseline prompt
- D: random KV (gaussian) with 3 seeds
- E: V_PROJ only (no KV swap)

Outputs:
- kv_matrix_results.csv (per-prompt rows)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import json
import numpy as np
import pandas as pd
import torch
from transformers import DynamicCache

from prompts.loader import PromptLoader
from src.core.models import load_model, set_seed
from src.metrics.behavior_states import BehaviorState, label_behavior_state
from src.metrics.rv import participation_ratio
from src.core.hooks import capture_v_projection
from src.pipelines.registry import ExperimentResult


def _is_expression(text: str) -> bool:
    s = label_behavior_state(text).state
    return s in (BehaviorState.RECURSIVE_PROSE, BehaviorState.NAKED_LOOP)


def _extract_full_kv_cache(model, tokenizer, prompt: str, device: str) -> DynamicCache:
    # IMPORTANT: we disable special tokens so sequence-length matching is stable across prompts.
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    with torch.no_grad():
        out = model(**inputs, use_cache=True, output_attentions=False)
        return out.past_key_values


def _to_legacy_kv(cache) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
    """
    Normalize HF cache objects to the legacy tuple-of-(k,v) format.

    - transformers may return `DynamicCache` (new) or a legacy tuple.
    - We convert to legacy for stable shape access + construction.
    """
    if cache is None:
        return tuple()
    if hasattr(cache, "to_legacy_cache"):
        return tuple(cache.to_legacy_cache())
    # Assume already legacy: tuple(layer -> (k, v))
    return tuple(cache)


def _from_legacy_kv(legacy: Tuple[Tuple[torch.Tensor, torch.Tensor], ...]):
    """Convert legacy cache into DynamicCache if supported; otherwise return legacy tuple."""
    if hasattr(DynamicCache, "from_legacy_cache"):
        return DynamicCache.from_legacy_cache(legacy)
    return legacy


def _create_gaussian_kv_cache(
    model,
    tokenizer,
    reference_prompt: str,
    seed: int,
    device: str,
) -> DynamicCache:
    set_seed(seed)
    ref_kv = _extract_full_kv_cache(model, tokenizer, reference_prompt, device)
    ref_legacy = _to_legacy_kv(ref_kv)
    random_layers: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for layer_idx, (k_ref, v_ref) in enumerate(ref_legacy):
        k_shape = k_ref.shape
        v_shape = v_ref.shape
        k_mean = k_ref.mean().item()
        k_std = max(k_ref.std().item(), 1e-6)
        v_mean = v_ref.mean().item()
        v_std = max(v_ref.std().item(), 1e-6)
        k_rand = torch.randn(k_shape, device=device, dtype=k_ref.dtype) * k_std + k_mean
        v_rand = torch.randn(v_shape, device=device, dtype=v_ref.dtype) * v_std + v_mean
        random_layers.append((k_rand, v_rand))
    return _from_legacy_kv(tuple(random_layers))


def _seq_len(tokenizer, prompt: str) -> int:
    # IMPORTANT: match the encoding used throughout this pipeline (no BOS/EOS insertion).
    return int(tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].shape[1])


def _generate_with_optional_kv(
    model,
    tokenizer,
    prompt: str,
    *,
    device: str,
    source_kv_cache: Optional[DynamicCache],
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    input_ids = inputs["input_ids"]
    target_seq_len = int(input_ids.shape[1])

    # Only use KV if length matches exactly.
    use_kv = False
    if source_kv_cache is not None:
        try:
            legacy = _to_legacy_kv(source_kv_cache)
            first_layer_k = legacy[0][0]
            # Typical k shape: (batch, heads, seq, head_dim) or (batch, seq, heads, head_dim)
            # We use the penultimate dimension as seq when possible.
            src_seq_len = int(first_layer_k.shape[-2])
            use_kv = src_seq_len == target_seq_len
        except Exception:
            use_kv = False

    with torch.no_grad():
        if use_kv:
            out = model(input_ids=input_ids, past_key_values=source_kv_cache, use_cache=True)
        else:
            out = model(input_ids=input_ids, use_cache=True)
        past = out.past_key_values

    generated = input_ids.clone()
    for _ in range(int(max_new_tokens)):
        with torch.no_grad():
            out = model(input_ids=generated[:, -1:], past_key_values=past, use_cache=True)
            logits = out.logits[:, -1, :]
            if do_sample:
                probs = torch.softmax(logits / float(temperature), dim=-1)
                nxt = torch.multinomial(probs, num_samples=1)
            else:
                nxt = torch.argmax(logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, nxt], dim=1)
            past = out.past_key_values
            if int(nxt.item()) == int(tokenizer.eos_token_id):
                break
    return tokenizer.decode(generated[0], skip_special_tokens=True)


def _compute_rv_from_ids(model, early: int, late: int, window: int, input_ids: torch.Tensor) -> float:
    with capture_v_projection(model, early) as v_early_storage:
        with capture_v_projection(model, late) as v_late_storage:
            with torch.no_grad():
                model(input_ids=input_ids)
    v_early = v_early_storage.get("v")
    v_late = v_late_storage.get("v")
    if v_early is None or v_late is None:
        return float("nan")
    if v_early.dim() == 3:
        v_early = v_early[0]
    if v_late.dim() == 3:
        v_late = v_late[0]
    pr_e = participation_ratio(v_early, window_size=window)
    pr_l = participation_ratio(v_late, window_size=window)
    if pr_e == 0 or np.isnan(pr_e) or np.isnan(pr_l):
        return float("nan")
    return float(pr_l / pr_e)


@dataclass(frozen=True)
class Pair:
    baseline: str
    recursive: str
    token_len: int


def _force_token_len(tokenizer, text: str, target_len: int, pad_piece: str = " and") -> str:
    """
    Force a string to have exactly target_len tokens under the given tokenizer.

    We do this by truncating/padding at the token level (decode after).
    This is used to guarantee KV-cache length compatibility.
    """
    if target_len <= 0:
        return ""
    base_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(base_ids) == 0:
        base_ids = tokenizer.encode("Text.", add_special_tokens=False)
    pad_ids = tokenizer.encode(pad_piece, add_special_tokens=False)
    if len(pad_ids) == 0:
        pad_ids = tokenizer.encode(" ", add_special_tokens=False)
    ids = list(base_ids)
    while len(ids) < target_len:
        ids.extend(pad_ids)
    ids = ids[:target_len]
    out = tokenizer.decode(ids, skip_special_tokens=True)
    return out.strip()


def _select_length_matched_pairs(tokenizer, loader: PromptLoader, n_pairs: int, seed: int) -> List[Pair]:
    rng = np.random.default_rng(seed)

    # NOTE: The canonical prompt bank uses pillar="confounds" for non-recursive baselines.
    # Some older docs refer to pillar="baseline", but that pillar may be empty/absent.
    baselines = loader.get_by_pillar("confounds", limit=2000, seed=seed)
    recursives = loader.get_by_pillar("dose_response", limit=2000, seed=seed)

    # Bucket by tokenized seq length (including BOS/EOS behavior via tokenizer call).
    by_len_base: Dict[int, List[str]] = {}
    for t in baselines:
        L = _seq_len(tokenizer, t)
        by_len_base.setdefault(L, []).append(t)

    by_len_rec: Dict[int, List[str]] = {}
    for t in recursives:
        L = _seq_len(tokenizer, t)
        by_len_rec.setdefault(L, []).append(t)

    common_lengths = sorted(set(by_len_base.keys()) & set(by_len_rec.keys()))
    pairs: List[Pair] = []
    for L in common_lengths:
        b_list = by_len_base[L]
        r_list = by_len_rec[L]
        rng.shuffle(b_list)
        rng.shuffle(r_list)
        k = min(len(b_list), len(r_list))
        for i in range(k):
            pairs.append(Pair(baseline=b_list[i], recursive=r_list[i], token_len=L))

    rng.shuffle(pairs)
    pairs = pairs[: int(n_pairs)]

    # Fallback: if the bank doesn't contain enough exact length matches, synthesize length-matched
    # baselines by token-level padding/truncation of confound prompts.
    if len(pairs) < int(n_pairs):
        need = int(n_pairs) - len(pairs)
        # Pull more recursive prompts (unique-ish)
        rng.shuffle(recursives)
        # A neutral non-recursive base topic; we also seed with existing confounds for variety.
        if not baselines:
            baselines = [
                "Describe how urban trees reduce heat in cities, with concrete mechanisms and one practical challenge."
            ]
        rng.shuffle(baselines)

        used = set((p.token_len, p.recursive) for p in pairs)
        added = 0
        for rec in recursives:
            if added >= need:
                break
            L = _seq_len(tokenizer, rec)
            # choose a baseline template and force it to L tokens
            base_template = baselines[(len(pairs) + added) % len(baselines)]
            base_forced = _force_token_len(tokenizer, base_template, L, pad_piece=" and")
            if (_seq_len(tokenizer, base_forced) != L) or (_seq_len(tokenizer, rec) != L):
                continue
            key = (L, rec)
            if key in used:
                continue
            used.add(key)
            pairs.append(Pair(baseline=base_forced, recursive=rec, token_len=L))
            added += 1

    return pairs[: int(n_pairs)]


def run_kv_sufficiency_matrix_from_config(cfg: Dict[str, Any], run_dir: Path) -> ExperimentResult:
    p = cfg.get("params") or {}
    model_name = str(p.get("model_name") or "mistralai/Mistral-7B-Instruct-v0.2")
    device = str(p.get("device") or "cuda")
    torch_dtype = str(p.get("torch_dtype") or "float16")
    attn_implementation = str(p.get("attn_implementation") or "sdpa")

    seed = int(p.get("seed") or 42)
    n_pairs = int(p.get("n_pairs") or 20)
    max_new_tokens = int(p.get("max_new_tokens") or 100)
    temperature = float(p.get("temperature") or 0.7)
    do_sample = bool(p.get("do_sample", True))

    early = int(p.get("early_layer") or 5)
    late = int(p.get("late_layer") or 27)
    window = int(p.get("window") or 16)
    random_kv_seeds = list(p.get("random_kv_seeds") or [101, 202, 303])

    import torch as _torch

    dtype = _torch.float16 if torch_dtype == "float16" else _torch.bfloat16
    model, tokenizer = load_model(model_name, device=device, torch_dtype=dtype, attn_implementation=attn_implementation)

    loader = PromptLoader()
    bank_version = loader.version
    (run_dir / "prompt_bank_version.txt").write_text(bank_version)
    (run_dir / "prompt_bank_version.json").write_text(
        json.dumps({"version": bank_version}, indent=2) + "\n"
    )
    pairs = _select_length_matched_pairs(tokenizer, loader, n_pairs=n_pairs, seed=seed)

    if len(pairs) < n_pairs:
        raise RuntimeError(f"Could not find {n_pairs} length-matched baseline/recursive pairs (found {len(pairs)}).")

    # Precompute KV caches for sources
    baseline_kv_sources = [_from_legacy_kv(_to_legacy_kv(_extract_full_kv_cache(model, tokenizer, pair.baseline, device))) for pair in pairs]
    recursive_kv_sources = [_from_legacy_kv(_to_legacy_kv(_extract_full_kv_cache(model, tokenizer, pair.recursive, device))) for pair in pairs]

    rows: List[Dict[str, object]] = []

    def record(cond: str, pair_idx: int, gen_text: str, used_kv: bool):
        lab = label_behavior_state(gen_text)
        rows.append(
            {
                "pair_idx": int(pair_idx),
                "condition": cond,
                "baseline_prompt": pairs[pair_idx].baseline,
                "recursive_prompt": pairs[pair_idx].recursive,
                "token_len": int(pairs[pair_idx].token_len),
                "used_kv": bool(used_kv),
                "generated_text": gen_text,
                "behavior_state": lab.state.value,
                "is_expression": bool(lab.state in (BehaviorState.RECURSIVE_PROSE, BehaviorState.NAKED_LOOP)),
            }
        )

    # A: Control (baseline prompt, no kv swap)
    for i, pair in enumerate(pairs):
        gen = _generate_with_optional_kv(
            model, tokenizer, pair.baseline, device=device, source_kv_cache=None, max_new_tokens=max_new_tokens, temperature=temperature, do_sample=do_sample
        )
        record("A_control", i, gen, used_kv=False)

    # B: KV from recursive prompt
    for i, pair in enumerate(pairs):
        kv = recursive_kv_sources[i]
        gen = _generate_with_optional_kv(
            model, tokenizer, pair.baseline, device=device, source_kv_cache=kv, max_new_tokens=max_new_tokens, temperature=temperature, do_sample=do_sample
        )
        # used_kv is guaranteed by length-matching, but keep it explicit
        record("B_kv_from_recursive", i, gen, used_kv=True)

    # C: KV from baseline prompt (use baseline kv from a different baseline of same length when possible)
    # Here we simply use each pair's baseline kv on the same baseline prompt (this is a structure-only control).
    for i, pair in enumerate(pairs):
        kv = baseline_kv_sources[i]
        gen = _generate_with_optional_kv(
            model, tokenizer, pair.baseline, device=device, source_kv_cache=kv, max_new_tokens=max_new_tokens, temperature=temperature, do_sample=do_sample
        )
        record("C_kv_from_baseline", i, gen, used_kv=True)

    # D: Random KV (gaussian) with 3 seeds
    for s in random_kv_seeds:
        for i, pair in enumerate(pairs):
            kv = _create_gaussian_kv_cache(model, tokenizer, reference_prompt=pair.baseline, seed=int(s), device=device)
            gen = _generate_with_optional_kv(
                model, tokenizer, pair.baseline, device=device, source_kv_cache=kv, max_new_tokens=max_new_tokens, temperature=temperature, do_sample=do_sample
            )
            record(f"D_random_kv_seed_{int(s)}", i, gen, used_kv=True)

    # E: V_PROJ only (no KV swap): overwrite v_proj output at late layer during baseline prompt forward.
    # We implement this by capturing v_proj output from the recursive prompt, then forcing it during baseline prompt encoding,
    # then generating normally from that modified cache.
    layer = model.model.layers[late].self_attn

    def build_past_with_vproj_patch(baseline_prompt: str, v_replacement: torch.Tensor) -> DynamicCache:
        inputs = tokenizer(baseline_prompt, return_tensors="pt", add_special_tokens=False).to(device)

        def hook_fn(_module, _inp, out):
            # out: (bs, seq, hidden)
            if out is None:
                return out
            if out.shape == v_replacement.shape:
                return v_replacement
            return out

        handle = layer.v_proj.register_forward_hook(hook_fn)
        try:
            with torch.no_grad():
                out = model(**inputs, use_cache=True)
                return _from_legacy_kv(_to_legacy_kv(out.past_key_values))
        finally:
            handle.remove()

    # Capture v_proj for each recursive prompt at late layer (prompt pass)
    for i, pair in enumerate(pairs):
        inputs_r = tokenizer(pair.recursive, return_tensors="pt", add_special_tokens=False).to(device)
        with capture_v_projection(model, late) as storage:
            with torch.no_grad():
                model(**inputs_r)
        v_rep = storage.get("v")
        if v_rep is None:
            # fall back: treat as no intervention
            gen = _generate_with_optional_kv(
                model, tokenizer, pair.baseline, device=device, source_kv_cache=None, max_new_tokens=max_new_tokens, temperature=temperature, do_sample=do_sample
            )
            record("E_vproj_only", i, gen, used_kv=False)
            continue

        past = build_past_with_vproj_patch(pair.baseline, v_rep.detach())
        # Now generate as continuation using cached past
        # Start generation by feeding last token with given past
        base_inputs = tokenizer(pair.baseline, return_tensors="pt", add_special_tokens=False).to(device)
        input_ids = base_inputs["input_ids"]
        generated = input_ids.clone()
        past_kv = past
        for _ in range(int(max_new_tokens)):
            with torch.no_grad():
                out = model(input_ids=generated[:, -1:], past_key_values=past_kv, use_cache=True)
                logits = out.logits[:, -1, :]
                if do_sample:
                    probs = torch.softmax(logits / float(temperature), dim=-1)
                    nxt = torch.multinomial(probs, num_samples=1)
                else:
                    nxt = torch.argmax(logits, dim=-1, keepdim=True)
                generated = torch.cat([generated, nxt], dim=1)
                past_kv = out.past_key_values
                if int(nxt.item()) == int(tokenizer.eos_token_id):
                    break
        gen_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        record("E_vproj_only", i, gen_text, used_kv=False)

    df = pd.DataFrame(rows)
    out_csv = run_dir / "kv_matrix_results.csv"
    df.to_csv(out_csv, index=False)

    # Expression rates per condition
    rates = (
        df.groupby("condition")["is_expression"]
        .mean()
        .sort_index()
        .to_dict()
    )

    summary = {
        "prompt_bank_version": bank_version,
        "n_pairs": int(n_pairs),
        "n_rows": int(len(df)),
        "expression_rate_by_condition": {k: float(v) for k, v in rates.items()},
        "artifacts": {
            "kv_matrix_results_csv": str(out_csv),
        },
        "params": {
            "model_name": model_name,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": do_sample,
            "random_kv_seeds": random_kv_seeds,
        },
    }
    return ExperimentResult(summary=summary)


__all__ = ["run_kv_sufficiency_matrix_from_config"]


