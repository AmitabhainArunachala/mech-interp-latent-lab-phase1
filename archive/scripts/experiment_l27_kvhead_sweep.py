#!/usr/bin/env python3
"""
Deep L27 headwise test (GQA-aware): sweep KV heads and measure causal ΔR_V.

This is designed to be:
- Bank-driven (no hardcoded prompts)
- GQA-correct (ablation is at KV-head level, not query-head)
- Artifact-producing (CSV + summary JSON)

Default runs are GPU-friendly. CPU works but will be slow at large N.

Example (fast sanity on CPU):
  python3 experiment_l27_kvhead_sweep.py --device cpu --n_recursive 6 --n_baseline 6 --kv_heads 2,0 --layers 27,21

Example (deep run on GPU):
  python3 experiment_l27_kvhead_sweep.py --device cuda --n_recursive 50 --n_baseline 50 --kv_heads all --layers 27,21
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parent


def participation_ratio(v_window: torch.Tensor) -> float:
    """
    PR = (Σ λ_i^2)^2 / Σ (λ_i^2)^2 where λ_i are singular values of v_window.T
    """
    try:
        x = v_window.to(torch.float32)
        _, s, _ = torch.linalg.svd(x.T, full_matrices=False)
        s2 = (s**2).detach().cpu().numpy()
        denom = float(np.sum(s2**2))
        if denom <= 0:
            return float("nan")
        return float((np.sum(s2) ** 2) / denom)
    except Exception:
        return float("nan")


class VExtractor:
    def __init__(self, model, layer_idx: int):
        self.model = model
        self.layer_idx = layer_idx
        self.activations: List[torch.Tensor] = []
        self.handle = None

    def _hook(self, module, inp, out):
        self.activations.append(out.detach())
        return out

    def __enter__(self):
        layer = self.model.model.layers[self.layer_idx]
        self.handle = layer.self_attn.v_proj.register_forward_hook(self._hook)
        return self

    def __exit__(self, *args):
        if self.handle is not None:
            self.handle.remove()


@contextmanager
def ablate_kv_head(model, layer_idx: int, kv_head_idx: int, num_kv_heads: int, head_dim: int):
    """
    Zero out a specific KV-head in v_proj output at a given layer.
    out shape: (batch, seq, num_kv_heads * head_dim)
    """
    layer = model.model.layers[layer_idx]
    handle = None

    def hook_fn(module, inp, out):
        if out.dim() != 3:
            return out
        b, t, hd = out.shape
        expected = num_kv_heads * head_dim
        if hd != expected:
            return out
        out_view = out.view(b, t, num_kv_heads, head_dim)
        out_view[:, :, kv_head_idx, :] = 0.0
        return out_view.view(b, t, -1)

    handle = layer.self_attn.v_proj.register_forward_hook(hook_fn)
    try:
        yield
    finally:
        if handle is not None:
            handle.remove()


def compute_rv(
    model,
    tokenizer,
    text: str,
    *,
    early_layer: int,
    late_layer: int,
    window: int,
    device: str,
    ablate_layer: Optional[int] = None,
    ablate_kv_head_idx: Optional[int] = None,
    num_kv_heads: int,
    head_dim: int,
) -> Tuple[float, int]:
    toks = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = toks["input_ids"].to(device)
    tlen = int(input_ids.shape[1])
    if tlen < window + 1:
        return float("nan"), tlen

    with torch.no_grad():
        if ablate_layer is not None and ablate_kv_head_idx is not None:
            with VExtractor(model, early_layer) as ve, VExtractor(model, late_layer) as vl, ablate_kv_head(
                model, ablate_layer, ablate_kv_head_idx, num_kv_heads=num_kv_heads, head_dim=head_dim
            ):
                _ = model(input_ids=input_ids)
        else:
            with VExtractor(model, early_layer) as ve, VExtractor(model, late_layer) as vl:
                _ = model(input_ids=input_ids)

        if not ve.activations or not vl.activations:
            return float("nan"), tlen

        pr_e = participation_ratio(ve.activations[0][0, -window:, :])
        pr_l = participation_ratio(vl.activations[0][0, -window:, :])
        if pr_e == 0 or np.isnan(pr_e) or np.isnan(pr_l):
            return float("nan"), tlen
        return float(pr_l / pr_e), tlen


@dataclass
class OneSampleSummary:
    n: int
    mean: float
    std: float
    ci_95: Tuple[float, float]
    t_stat: float
    p_value: float
    cohens_d: float


def summarize_deltas(deltas: np.ndarray) -> OneSampleSummary:
    deltas = deltas.astype(float)
    n = int(deltas.shape[0])
    mean = float(np.mean(deltas))
    std = float(np.std(deltas, ddof=1)) if n > 1 else 0.0
    # one-sample t-test vs 0
    t_stat, p_value = stats.ttest_1samp(deltas, 0.0)
    # Cohen's d for one-sample: mean / std
    cohens_d = float(mean / std) if std > 0 else 0.0
    # 95% CI for mean (t-based)
    if n > 1:
        se = std / math.sqrt(n)
        tcrit = stats.t.ppf(0.975, df=n - 1)
        ci = (float(mean - tcrit * se), float(mean + tcrit * se))
    else:
        ci = (mean, mean)
    return OneSampleSummary(n=n, mean=mean, std=std, ci_95=ci, t_stat=float(t_stat), p_value=float(p_value), cohens_d=cohens_d)


def parse_kv_heads(arg: str, num_kv_heads: int) -> List[int]:
    if arg.strip().lower() == "all":
        return list(range(num_kv_heads))
    parts = [p.strip() for p in arg.split(",") if p.strip()]
    out = []
    for p in parts:
        out.append(int(p))
    return out


def parse_layers(arg: str) -> List[int]:
    parts = [p.strip() for p in arg.split(",") if p.strip()]
    return [int(p) for p in parts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1")
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--early_layer", type=int, default=5)
    ap.add_argument("--late_layer", type=int, default=27)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--layers", type=str, default="27,21", help="Comma-separated ablation layers to test, e.g. 27,21")
    ap.add_argument("--kv_heads", type=str, default="all", help='KV heads to test: "all" or comma list like "2,0"')
    ap.add_argument("--n_recursive", type=int, default=50)
    ap.add_argument("--n_baseline", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default="results/l27_kvhead_sweep")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load bank prompts (strict: no ad-hoc)
    bank = json.loads((REPO_ROOT / "prompts" / "bank.json").read_text(encoding="utf-8"))
    champions = [(k, v) for k, v in bank.items() if v.get("group") == "champions"]
    l5 = [(k, v) for k, v in bank.items() if v.get("group") == "L5_refined"]
    baseline = [(k, v) for k, v in bank.items() if v.get("group") == "baseline_math"]

    # Recursive pool: prioritize champions then L5 (deterministic slice)
    rec_pool = champions + l5
    rec_pool = rec_pool[: args.n_recursive]
    base_pool = baseline[: args.n_baseline]

    if len(rec_pool) < args.n_recursive or len(base_pool) < args.n_baseline:
        raise RuntimeError(f"Not enough prompts in bank: rec={len(rec_pool)} base={len(base_pool)}")

    # Load model/tokenizer (prefer local cache; can flip later on GPU)
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(args.device)
    model.eval()

    num_heads = model.config.num_attention_heads
    num_kv_heads = getattr(model.config, "num_key_value_heads", num_heads)
    head_dim = model.config.hidden_size // num_heads

    ablation_layers = parse_layers(args.layers)
    kv_heads = parse_kv_heads(args.kv_heads, num_kv_heads=num_kv_heads)

    run_dir = Path(args.out_dir) / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_kvhead_sweep"
    run_dir.mkdir(parents=True, exist_ok=True)

    csv_path = run_dir / "kvhead_sweep.csv"
    summary_path = run_dir / "summary.json"

    rows = []

    # For each prompt: compute rv_no_ablation once; then for each (layer, kv_head) compute rv_ablated
    def iter_prompts() -> Iterable[Tuple[str, int, str, str]]:
        for i, (pid, meta) in enumerate(rec_pool):
            yield "recursive", i, pid, meta["text"]
        for i, (pid, meta) in enumerate(base_pool):
            yield "baseline", i, pid, meta["text"]

    for prompt_type, prompt_idx, prompt_id, text in iter_prompts():
        rv0, tlen = compute_rv(
            model,
            tok,
            text,
            early_layer=args.early_layer,
            late_layer=args.late_layer,
            window=args.window,
            device=args.device,
            ablate_layer=None,
            ablate_kv_head_idx=None,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )
        if np.isnan(rv0):
            continue
        for layer in ablation_layers:
            for kvh in kv_heads:
                rvA, _ = compute_rv(
                    model,
                    tok,
                    text,
                    early_layer=args.early_layer,
                    late_layer=args.late_layer,
                    window=args.window,
                    device=args.device,
                    ablate_layer=layer,
                    ablate_kv_head_idx=kvh,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                )
                if np.isnan(rvA):
                    continue
                rows.append(
                    dict(
                        prompt_type=prompt_type,
                        prompt_idx=prompt_idx,
                        prompt_id=prompt_id,
                        token_len=tlen,
                        rv_no_ablation=rv0,
                        ablate_layer=layer,
                        kv_head=kvh,
                        rv_ablated=rvA,
                        delta=(rvA - rv0),
                        delta_pct=(rvA - rv0) / rv0 if rv0 != 0 else float("nan"),
                    )
                )

    # Write CSV
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "prompt_type",
                "prompt_idx",
                "prompt_id",
                "token_len",
                "rv_no_ablation",
                "ablate_layer",
                "kv_head",
                "rv_ablated",
                "delta",
                "delta_pct",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # Summaries: per (prompt_type, layer, kv_head) one-sample delta test
    by_key: Dict[Tuple[str, int, int], List[float]] = {}
    for r in rows:
        key = (r["prompt_type"], int(r["ablate_layer"]), int(r["kv_head"]))
        by_key.setdefault(key, []).append(float(r["delta"]))

    summary = {
        "timestamp": run_dir.name.split("_", 1)[0],
        "run_dir": str(run_dir),
        "model": args.model,
        "device": args.device,
        "dtype": args.dtype,
        "params": {
            "early_layer": args.early_layer,
            "late_layer": args.late_layer,
            "window": args.window,
            "ablation_layers": ablation_layers,
            "kv_heads": kv_heads,
            "n_recursive_requested": args.n_recursive,
            "n_baseline_requested": args.n_baseline,
            "seed": args.seed,
        },
        "model_config": {
            "num_attention_heads": int(num_heads),
            "num_key_value_heads": int(num_kv_heads),
            "head_dim": int(head_dim),
        },
        "artifacts": {
            "csv": str(csv_path),
        },
        "analysis": {},
    }

    for (ptype, layer, kvh), deltas in sorted(by_key.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        deltas_np = np.array(deltas, dtype=float)
        ss = summarize_deltas(deltas_np)
        summary["analysis"].setdefault(ptype, {}).setdefault(f"layer_{layer}", {})[f"kv_{kvh}"] = asdict(ss)

    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {summary_path}")


if __name__ == "__main__":
    main()










