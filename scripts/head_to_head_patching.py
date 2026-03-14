#!/usr/bin/env python3
"""
HEAD-TO-HEAD PATH PATCHING
===========================

Tests information flow between specific attention heads by patching individual
head V-projections. This identifies the *wiring* of the R_V circuit:
which heads carry recursive-specific information to which downstream layers.

Methodology (IOI-style, adapted for R_V):
  For each (source_head @ source_layer):
    1. Run RECURSIVE prompt → capture source head's V-proj output
    2. Run BASELINE prompt → capture source head's V-proj output
    3. Run RECURSIVE prompt with source head V-proj REPLACED by baseline version
    4. Measure R_V change → large ΔR_V means this head carries recursive signal

This is per-head path patching in the break direction: we replace the recursive
head output with baseline output and see if R_V (contraction) is destroyed.

Additionally tests head combinations (pairs/triples) for interaction effects:
if ablating heads A+B together produces a larger effect than sum of individual
effects, they are in the same sub-circuit.

Usage:
    python3 scripts/head_to_head_patching.py --device cuda
    python3 scripts/head_to_head_patching.py --device cuda --top-k 30 --n-prompts 30
    python3 scripts/head_to_head_patching.py --device cuda --head-sweep-file results/full_head_sweep/full_head_sweep_20260312_052013.json

Output: results/head_circuit/head_circuit_<timestamp>.json
"""

import sys
import json
import argparse
import gc
import time
import itertools
from pathlib import Path
from datetime import datetime
import torch
import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from prompts.subsets import load_default_mistral_hardening_subset, split_tier_records_by_pillar  # noqa: E402
from geometric_lens.probe import GeometricProbe  # noqa: E402
from geometric_lens.models import get_v_proj_module  # noqa: E402


# ── Frozen prompt contract ───────────────────────────────────────────────────
_subset = load_default_mistral_hardening_subset()
_tier_records = split_tier_records_by_pillar(_subset, "core_measurement")
RECURSIVE_RECORDS = _tier_records["recursive"]
BASELINE_RECORDS = _tier_records["baseline"]
RECURSIVE_PROMPT_IDS = [prompt_id for prompt_id, _ in RECURSIVE_RECORDS]
BASELINE_PROMPT_IDS = [prompt_id for prompt_id, _ in BASELINE_RECORDS]
RECURSIVE_PROMPTS = [record["text"] for _, record in RECURSIVE_RECORDS]
BASELINE_PROMPTS = [record["text"] for _, record in BASELINE_RECORDS]


def cohens_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled_std = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled_std < 1e-10:
        return float("nan")
    return (np.mean(a) - np.mean(b)) / pooled_std


def _align_seq_len(donor, target_len):
    """Pad or truncate donor activation along the seq dimension."""
    d_len = donor.shape[1]
    if d_len == target_len:
        return donor
    if d_len > target_len:
        return donor[:, :target_len]
    pad = donor[:, -1:].expand(-1, target_len - d_len, *donor.shape[2:])
    return torch.cat([donor, pad], dim=1)


def _metric_value(row, metric_name):
    value = row.get(metric_name, float("nan"))
    return abs(value) if value == value else 0.0


def load_top_heads(head_sweep_file, top_k=20, ranking_metric="entropy_d"):
    """Load top heads from a head sweep JSON file."""
    with open(head_sweep_file) as f:
        data = json.load(f)

    head_results = data.get("head_results", [])
    ranked = sorted(
        head_results,
        key=lambda r: (
            _metric_value(r, ranking_metric),
            _metric_value(r, "rank_d" if ranking_metric == "entropy_d" else "entropy_d"),
        ),
        reverse=True,
    )
    top = []
    for r in ranked[:top_k]:
        top.append({
            "layer": r["layer"],
            "head": r["head"],
            "entropy_d": r.get("entropy_d", float("nan")),
            "rank_d": r.get("rank_d", float("nan")),
        })
    return top


def parse_manual_heads(spec):
    heads = []
    if not spec:
        return heads

    for raw in spec.split(","):
        token = raw.strip()
        if not token:
            continue
        token = token.replace("L", "").replace("H", "").replace(".", ":")
        parts = [p for p in token.split(":") if p]
        if len(parts) != 2:
            raise ValueError(
                f"Invalid head spec '{raw}'. Use forms like '27:5' or 'L27.H5'."
            )
        layer, head = int(parts[0]), int(parts[1])
        heads.append({
            "layer": layer,
            "head": head,
            "entropy_d": float("nan"),
            "rank_d": float("nan"),
        })
    return heads


def dedupe_gqa_heads(top_heads, num_q_heads, num_kv_heads):
    """Collapse query-head rankings onto unique KV-head intervention sites for GQA models."""
    if num_q_heads == num_kv_heads:
        for head in top_heads:
            head["kv_head"] = head["head"]
            head["alias_query_heads"] = [head["head"]]
        return top_heads

    group_size = num_q_heads // num_kv_heads
    deduped = []
    by_site = {}

    for head in top_heads:
        kv_head = head["head"] // group_size
        site = (head["layer"], kv_head)
        if site in by_site:
            by_site[site]["alias_query_heads"].append(head["head"])
            continue

        entry = dict(head)
        entry["kv_head"] = kv_head
        entry["alias_query_heads"] = [head["head"]]
        deduped.append(entry)
        by_site[site] = entry

    return deduped


def find_matching_head_sweep(results_dir, model_name):
    """Pick the newest head sweep artifact whose stored model matches the requested model."""
    matches = []
    for path in sorted(results_dir.glob("full_head_sweep_*.json")):
        try:
            with open(path) as f:
                payload = json.load(f)
        except Exception:
            continue
        if payload.get("model") == model_name:
            matches.append(path)
    return matches[-1] if matches else None


def capture_head_v_activation(model, tokenizer, text, layer_idx, head_idx, head_dim, device):
    """Capture a single head's V-projection output."""
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    module, kind = get_v_proj_module(model, layer_idx)
    storage = {"act": None}

    def hook(mod, inp, out):
        storage["act"] = out.detach().clone()

    handle = module.register_forward_hook(hook)
    with torch.no_grad():
        model(**enc)
    handle.remove()

    v_full = storage["act"]  # (batch, seq, v_dim)
    if v_full is None:
        return None

    # Extract specific head's slice
    start = head_idx * head_dim
    end = start + head_dim
    if end > v_full.shape[-1]:
        return None

    return v_full, start, end


def patch_head_v(model, layer_idx, head_start, head_end, donor_head_act):
    """Register a hook that patches a specific head's V-projection."""
    module, kind = get_v_proj_module(model, layer_idx)

    def hook(mod, inp, out):
        out_patched = out.clone()
        aligned = _align_seq_len(donor_head_act, out.shape[1])
        out_patched[:, :, head_start:head_end] = aligned[:, :, head_start:head_end].to(
            out.device, dtype=out.dtype
        )
        return out_patched

    return module.register_forward_hook(hook)


def patch_multiple_heads(model, patches):
    """Register hooks for multiple head patches. Returns list of handles."""
    handles = []
    for layer_idx, head_start, head_end, donor_act in patches:
        h = patch_head_v(model, layer_idx, head_start, head_end, donor_act)
        handles.append(h)
    return handles


def measure_rv(probe, text, device):
    """Measure R_V for a single text."""
    result = probe.measure(text, metrics=["rv"])
    return result.rv


def run_head_circuit_analysis(args):
    run_start = time.time()
    print("=" * 70)
    print("HEAD-TO-HEAD PATH PATCHING (CIRCUIT WIRING)")
    print("=" * 70)

    out_dir = Path("results/head_circuit")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ──
    print(f"Loading model: {args.model}")
    probe = GeometricProbe(
        model_name=args.model,
        device=args.device,
        attn_implementation="eager",
    )
    model = probe.model
    tokenizer = probe.tokenizer
    spec = probe.spec
    n_layers = spec.num_layers
    num_kv_heads = getattr(spec, "num_kv_heads", spec.num_heads)
    gqa_group_size = max(1, spec.num_heads // num_kv_heads)
    head_dim = spec.head_dim

    print(f"Model: {spec.num_layers} layers, {spec.num_heads} Q-heads, "
          f"{num_kv_heads} KV-heads, head_dim={head_dim}")

    # ── Load top heads ──
    if args.head_sweep_file:
        sweep_path = Path(args.head_sweep_file)
        with open(sweep_path) as f:
            sweep_payload = json.load(f)
        sweep_model = sweep_payload.get("model")
        if sweep_model and sweep_model != args.model:
            raise ValueError(
                f"Head sweep model mismatch: requested {args.model}, "
                f"but {sweep_path} stores {sweep_model}"
            )
    else:
        # Find latest matching head sweep for the requested model.
        sweep_dir = Path("results/full_head_sweep")
        sweep_path = find_matching_head_sweep(sweep_dir, args.model)
        if sweep_path is None:
            print(
                f"ERROR: No head sweep file found for {args.model}. "
                "Run full_head_sweep.py for the same model first."
            )
            return

    if args.manual_heads:
        top_heads = parse_manual_heads(args.manual_heads)
        top_heads = dedupe_gqa_heads(top_heads, spec.num_heads, num_kv_heads)[:args.top_k]
        print(f"Using {len(top_heads)} manually specified heads")
    else:
        print(f"Loading head sweep: {sweep_path}")
        candidate_k = args.top_k * max(1, spec.num_heads // num_kv_heads)
        top_heads = load_top_heads(
            sweep_path,
            top_k=candidate_k,
            ranking_metric=args.ranking_metric,
        )
        top_heads = dedupe_gqa_heads(top_heads, spec.num_heads, num_kv_heads)[:args.top_k]
        print(f"Top {len(top_heads)} heads loaded using {args.ranking_metric}")

    # Print selected heads
    print("\n  Selected heads for circuit analysis:")
    print(f"  {'L.H':>6} {'KV':>4} {'d_entropy':>10} {'d_rank':>10} {'aliases':>12}")
    print("  " + "-" * 50)
    for h in top_heads:
        aliases = ",".join(f"{q:02d}" for q in h["alias_query_heads"])
        print(f"  L{h['layer']:02d}.H{h['head']:02d} "
              f"{h['kv_head']:>4d} {h['entropy_d']:>10.3f} {h['rank_d']:>10.3f} {aliases:>12}")

    n = min(args.n_prompts, len(RECURSIVE_PROMPTS), len(BASELINE_PROMPTS))
    rec_prompts = RECURSIVE_PROMPTS[:n]
    bas_prompts = BASELINE_PROMPTS[:n]

    print(f"\nPrompts: {n} per condition")
    print(f"Prompt contract: subset={_subset.name} bank={_subset.source_bank_version}")

    # ── Phase 1: Clean R_V baselines ──
    print("\n" + "=" * 70)
    print("PHASE 1: CLEAN R_V BASELINES")
    print("=" * 70)

    clean_rec_rvs = []
    clean_bas_rvs = []
    for i in range(n):
        rv = measure_rv(probe, rec_prompts[i], args.device)
        clean_rec_rvs.append(rv)
        rv = measure_rv(probe, bas_prompts[i], args.device)
        clean_bas_rvs.append(rv)

    clean_gap = np.nanmean(clean_rec_rvs) - np.nanmean(clean_bas_rvs)
    print(f"  Clean recursive R_V: {np.nanmean(clean_rec_rvs):.4f} ± {np.nanstd(clean_rec_rvs):.4f}")
    print(f"  Clean baseline R_V:  {np.nanmean(clean_bas_rvs):.4f} ± {np.nanstd(clean_bas_rvs):.4f}")
    print(f"  Clean R_V gap: {clean_gap:+.4f}")

    # ── Phase 2: Per-head path patching (break direction) ──
    print("\n" + "=" * 70)
    print("PHASE 2: PER-HEAD PATH PATCHING (BREAK DIRECTION)")
    print("=" * 70)
    print("  For each head: replace recursive V-proj with baseline V-proj")
    print("  Large ΔR_V = this head carries recursive-specific information\n")

    head_results = []
    t0 = time.time()

    for hi, head_info in enumerate(top_heads):
        layer_idx = head_info["layer"]
        head_idx = head_info["head"]
        kv_head = head_info["kv_head"]

        print(f"  [{hi+1}/{len(top_heads)}] L{layer_idx}.H{head_idx} (KV head {kv_head})...")

        patched_rvs = []
        for i in range(n):
            # Capture baseline V-proj output for this head
            result = capture_head_v_activation(
                model, tokenizer, bas_prompts[i], layer_idx, kv_head, head_dim, args.device
            )
            if result is None:
                patched_rvs.append(float("nan"))
                continue

            donor_full, h_start, h_end = result

            # Run recursive prompt with this head's V-proj replaced by baseline
            handle = patch_head_v(model, layer_idx, h_start, h_end, donor_full)
            try:
                rv = measure_rv(probe, rec_prompts[i], args.device)
                patched_rvs.append(rv)
            finally:
                handle.remove()

        valid_patched = [v for v in patched_rvs if not np.isnan(v)]
        valid_clean = [v for v in clean_rec_rvs if not np.isnan(v)]

        if valid_patched and valid_clean:
            d = cohens_d(valid_patched, valid_clean)
            delta_rv = np.nanmean(valid_patched) - np.nanmean(valid_clean)
            _, p_val = stats.mannwhitneyu(valid_patched, valid_clean, alternative="two-sided")
        else:
            d, delta_rv, p_val = float("nan"), float("nan"), float("nan")

        result = {
            "layer": layer_idx,
            "head": head_idx,
            "kv_head": kv_head,
            "alias_query_heads": head_info["alias_query_heads"],
            "entropy_d": head_info["entropy_d"],
            "rank_d": head_info["rank_d"],
            "patched_rv_mean": float(np.nanmean(valid_patched)) if valid_patched else float("nan"),
            "clean_rv_mean": float(np.nanmean(valid_clean)),
            "delta_rv": float(delta_rv),
            "cohens_d": float(d),
            "p_value": float(p_val),
            "n_valid": len(valid_patched),
            "direction": "break",
        }
        head_results.append(result)

        sig = "***" if abs(d) > 1.0 else " **" if abs(d) > 0.5 else "  *" if abs(d) > 0.2 else "   "
        print(f"    ΔR_V = {delta_rv:+.4f}, d = {d:+.3f}, p = {p_val:.4f} {sig}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"\n  Phase 2 complete in {elapsed/60:.1f} min")

    # ── Phase 3: Head pair interactions ──
    print("\n" + "=" * 70)
    print("PHASE 3: HEAD PAIR INTERACTIONS (TOP 10 PAIRS)")
    print("=" * 70)

    sorted_by_effect = sorted(
        head_results,
        key=lambda r: abs(r["cohens_d"]),
        reverse=True,
    )
    significant_heads = [r for r in sorted_by_effect if abs(r["cohens_d"]) > args.single_head_threshold]

    if args.pair_source == "significant":
        pair_pool = significant_heads[:args.pair_pool_size]
    else:
        pair_pool = sorted_by_effect[:args.pair_pool_size]
    pair_prompt_count = min(n, args.pair_prompt_limit)

    valid_clean = [v for v in clean_rec_rvs if not np.isnan(v)]

    pair_results = []
    if len(pair_pool) >= 2:
        pairs = list(itertools.combinations(range(len(pair_pool)), 2))[:args.max_pairs]
        print(
            f"  Testing {len(pairs)} head pairs from {args.pair_source} pool "
            f"(pool={len(pair_pool)}, prompts={pair_prompt_count})\n"
        )

        for pi, (idx_a, idx_b) in enumerate(pairs):
            ha = pair_pool[idx_a]
            hb = pair_pool[idx_b]

            la, headA, kvA = ha["layer"], ha["head"], ha["kv_head"]
            lb, headB, kvB = hb["layer"], hb["head"], hb["kv_head"]

            print(f"  [{pi+1}/{len(pairs)}] L{la}.H{headA} + L{lb}.H{headB}...")

            pair_patched_rvs = []
            for i in range(pair_prompt_count):
                # Capture baseline for both heads
                res_a = capture_head_v_activation(
                    model, tokenizer, bas_prompts[i], la, kvA, head_dim, args.device
                )
                res_b = capture_head_v_activation(
                    model, tokenizer, bas_prompts[i], lb, kvB, head_dim, args.device
                )
                if res_a is None or res_b is None:
                    pair_patched_rvs.append(float("nan"))
                    continue

                donor_a, sa, ea = res_a
                donor_b, sb, eb = res_b

                # Patch both heads simultaneously
                handle_a = patch_head_v(model, la, sa, ea, donor_a)
                handle_b = patch_head_v(model, lb, sb, eb, donor_b)
                try:
                    rv = measure_rv(probe, rec_prompts[i], args.device)
                    pair_patched_rvs.append(rv)
                finally:
                    handle_a.remove()
                    handle_b.remove()

            valid = [v for v in pair_patched_rvs if not np.isnan(v)]
            if valid and valid_clean:
                clean_pair = valid_clean[:len(valid)]
                d_pair = cohens_d(valid, clean_pair)
                delta_pair = np.nanmean(valid) - np.nanmean(clean_pair)
            else:
                d_pair, delta_pair = float("nan"), float("nan")

            # Interaction effect: is pair effect > sum of individual effects?
            d_a = ha["cohens_d"]
            d_b = hb["cohens_d"]
            d_additive = d_a + d_b
            interaction = d_pair - d_additive if not np.isnan(d_pair) else float("nan")
            interaction_excess = (
                abs(d_pair) - abs(d_additive) if not np.isnan(d_pair) else float("nan")
            )

            pair_result = {
                "head_a": {"layer": la, "head": headA, "d_individual": d_a},
                "head_b": {"layer": lb, "head": headB, "d_individual": d_b},
                "d_pair": float(d_pair),
                "delta_rv_pair": float(delta_pair),
                "d_additive_expected": float(d_additive),
                "interaction_effect": float(interaction),
                "interaction_excess_abs_d": float(interaction_excess),
                "superadditive": bool(interaction_excess > args.superadditive_margin)
                if not np.isnan(interaction_excess)
                else False,
                "n_valid": len(valid),
            }
            pair_results.append(pair_result)

            marker = " SUPER" if pair_result["superadditive"] else ""
            print(
                f"    d_pair={d_pair:+.3f} vs d_sum={d_additive:+.3f} "
                f"(interaction={interaction:+.3f}, excess={interaction_excess:+.3f}){marker}"
            )

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    else:
        print(f"  Fewer than 2 heads in {args.pair_source} pool — skipping pair analysis")

    # ── Phase 4: Circuit summary ──
    print("\n" + "=" * 70)
    print("CIRCUIT SUMMARY")
    print("=" * 70)

    # Rank heads by circuit importance (|d| from patching)
    ranked = sorted(head_results, key=lambda r: abs(r["cohens_d"]), reverse=True)
    circuit_nodes = [r for r in ranked if abs(r["cohens_d"]) > 0.3]

    # Classify into functional roles
    early_diversifiers = [r for r in circuit_nodes if r["delta_rv"] > 0 and r["layer"] < n_layers // 2]
    late_compressors = [r for r in circuit_nodes if r["delta_rv"] > 0 and r["layer"] >= n_layers // 2]
    early_compressors = [r for r in circuit_nodes if r["delta_rv"] < 0 and r["layer"] < n_layers // 2]
    late_diversifiers = [r for r in circuit_nodes if r["delta_rv"] < 0 and r["layer"] >= n_layers // 2]

    print("\n  Circuit nodes (|d| > 0.3):")
    print(f"  {'L.H':>6} {'ΔR_V':>8} {'d':>8} {'p':>10} {'Role':>20}")
    print("  " + "-" * 60)
    for r in circuit_nodes:
        if r["delta_rv"] > 0:
            role = "BREAKS contraction" if r["layer"] >= n_layers // 2 else "Early gate"
        else:
            role = "Maintains contraction" if r["layer"] >= n_layers // 2 else "Early source"
        print(f"  L{r['layer']:02d}.H{r['head']:02d} "
              f"{r['delta_rv']:>+8.4f} {r['cohens_d']:>+8.3f} "
              f"{r['p_value']:>10.4f} {role:>20}")

    print("\n  Functional classification:")
    print(f"    Early diversifiers (break early → R_V rises): {len(early_diversifiers)}")
    print(f"    Late compressors (break late → R_V rises):    {len(late_compressors)}")
    print(f"    Early compressors (break early → R_V drops):  {len(early_compressors)}")
    print(f"    Late diversifiers (break late → R_V drops):   {len(late_diversifiers)}")

    if pair_results:
        superadditive = [p for p in pair_results if p["superadditive"]]
        print(f"\n  Superadditive pairs (same sub-circuit): {len(superadditive)}/{len(pair_results)}")
        for p in superadditive:
            a, b = p["head_a"], p["head_b"]
            print(f"    L{a['layer']:02d}.H{a['head']:02d} + L{b['layer']:02d}.H{b['head']:02d}: "
                  f"d_pair={p['d_pair']:+.3f} > d_sum={p['d_additive_expected']:+.3f}")

    # ── Save results ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {
        "timestamp": timestamp,
        "experiment": "head_to_head_patching",
        "model": args.model,
        "model_num_layers": spec.num_layers,
        "model_num_query_heads": spec.num_heads,
        "model_num_kv_heads": num_kv_heads,
        "gqa_group_size": gqa_group_size,
        "prompt_bank_version": _subset.source_bank_version,
        "prompt_subset_name": _subset.name,
        "prompt_tier": "core_measurement",
        "n_prompts": n,
        "top_k_heads": args.top_k,
        "max_pairs": args.max_pairs,
        "single_head_threshold": args.single_head_threshold,
        "pair_source": args.pair_source,
        "pair_pool_size": args.pair_pool_size,
        "pair_prompt_limit": args.pair_prompt_limit,
        "superadditive_margin": args.superadditive_margin,
        "n_unique_intervention_sites": len(top_heads),
        "head_selection_mode": "manual" if args.manual_heads else "ranked",
        "ranking_metric": None if args.manual_heads else args.ranking_metric,
        "manual_heads": args.manual_heads,
        "head_sweep_source": str(sweep_path),
        "clean_recursive_rv": float(np.nanmean(clean_rec_rvs)),
        "clean_baseline_rv": float(np.nanmean(clean_bas_rvs)),
        "clean_rv_gap": float(clean_gap),
        "selected_intervention_sites": [
            {
                "layer": h["layer"],
                "head": h["head"],
                "kv_head": h["kv_head"],
                "entropy_d": h["entropy_d"],
                "rank_d": h["rank_d"],
                "alias_query_heads": h["alias_query_heads"],
            }
            for h in top_heads
        ],
        "pair_candidate_pool": [
            {
                "layer": r["layer"],
                "head": r["head"],
                "kv_head": r["kv_head"],
                "cohens_d": r["cohens_d"],
                "delta_rv": r["delta_rv"],
                "p_value": r["p_value"],
                "alias_query_heads": r["alias_query_heads"],
            }
            for r in pair_pool
        ],
        "per_head_results": head_results,
        "pair_interaction_results": pair_results,
        "circuit_summary": {
            "n_significant_heads": len(circuit_nodes),
            "n_early_diversifiers": len(early_diversifiers),
            "n_late_compressors": len(late_compressors),
            "n_superadditive_pairs": len([p for p in pair_results if p["superadditive"]]),
            "top_5_circuit_nodes": [
                {"layer": r["layer"], "head": r["head"], "d": r["cohens_d"], "delta_rv": r["delta_rv"]}
                for r in circuit_nodes[:5]
            ],
        },
        "recursive_prompt_ids": RECURSIVE_PROMPT_IDS[:n],
        "baseline_prompt_ids": BASELINE_PROMPT_IDS[:n],
    }

    summary_path = out_dir / f"head_circuit_{timestamp}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n  Results saved: {summary_path}")
    total_time = time.time() - run_start
    print(f"  Total time: {total_time/60:.1f} min")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Head-to-Head Path Patching (Circuit Wiring)")
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-prompts", type=int, default=20, help="Prompts per condition")
    parser.add_argument("--top-k", type=int, default=20, help="Top K heads from sweep to test")
    parser.add_argument("--max-pairs", type=int, default=15, help="Max head pairs to test")
    parser.add_argument(
        "--single-head-threshold",
        type=float,
        default=0.3,
        help="Single-head |d| threshold for calling a node individually significant",
    )
    parser.add_argument(
        "--pair-source",
        choices=["significant", "top_effect"],
        default="significant",
        help="Whether pair tests draw from individually significant heads or the top-effect pool",
    )
    parser.add_argument(
        "--pair-pool-size",
        type=int,
        default=10,
        help="How many heads to keep in the pair-testing candidate pool",
    )
    parser.add_argument(
        "--pair-prompt-limit",
        type=int,
        default=15,
        help="Max prompts to use during pair patching",
    )
    parser.add_argument(
        "--superadditive-margin",
        type=float,
        default=0.1,
        help="Minimum abs(d) excess above additive expectation to call a pair superadditive",
    )
    parser.add_argument(
        "--ranking-metric",
        choices=["entropy_d", "rank_d"],
        default="entropy_d",
        help="Which head-sweep metric to rank candidates by",
    )
    parser.add_argument("--head-sweep-file", type=str, default=None,
                        help="Path to head sweep JSON (default: latest)")
    parser.add_argument(
        "--manual-heads",
        type=str,
        default=None,
        help="Comma-separated heads like 'L27.H5,L18.H1,L5.H29' to override sweep ranking",
    )
    args = parser.parse_args()
    run_head_circuit_analysis(args)
