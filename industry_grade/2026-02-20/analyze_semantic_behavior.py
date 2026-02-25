#!/usr/bin/env python3
"""
Semantic behavioral rescoring for C2 and seed-bridge outputs.

Method:
- Embed generated outputs using sentence-transformers.
- Embed five L5_refined exemplar prompts from prompt bank.
- semantic_recursive_score = max cosine similarity to any exemplar.
- semantic_recursive = score > threshold (default 0.4).

Outputs:
- industry_grade/2026-02-20/evidence/semantic_bridge_scores_seed_bridge.csv
- industry_grade/2026-02-20/evidence/semantic_bridge_scores_c2.csv
- industry_grade/2026-02-20/evidence/semantic_behavior_analysis.json
- industry_grade/2026-02-20/evidence/semantic_behavior_analysis.md
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import torch
from transformers import AutoModel, AutoTokenizer


DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_THRESHOLD = 0.4

# Fixed five exemplars from L5_refined (as requested)
EXEMPLAR_IDS = [
    "L5_refined_01",
    "L5_refined_05",
    "L5_refined_08",
    "L5_refined_14",
    "L5_refined_19",
]


def _load_encoder(model_name: str):
    """
    Load SBERT-compatible encoder using Transformers directly for stability.
    """
    repo = f"sentence-transformers/{model_name}"
    try:
        tok = AutoTokenizer.from_pretrained(repo)
        mdl = AutoModel.from_pretrained(repo)
    except Exception:
        # Network can be restricted/intermittent; retry local cache only.
        tok = AutoTokenizer.from_pretrained(repo, local_files_only=True)
        mdl = AutoModel.from_pretrained(repo, local_files_only=True)
    mdl.eval()
    return tok, mdl


def _resolve_exemplars(prompt_bank_path: Path) -> Tuple[List[str], List[str]]:
    bank = json.loads(prompt_bank_path.read_text(encoding="utf-8"))
    ids: List[str] = []
    texts: List[str] = []
    for key in EXEMPLAR_IDS:
        if key in bank and isinstance(bank[key], dict) and isinstance(bank[key].get("text"), str):
            ids.append(key)
            texts.append(bank[key]["text"])
    if len(texts) < 5:
        fallback = [
            (k, v.get("text", ""))
            for k, v in bank.items()
            if isinstance(v, dict) and v.get("group") == "L5_refined" and isinstance(v.get("text"), str)
        ]
        fallback = sorted(fallback, key=lambda x: x[0])
        for k, t in fallback:
            if k in ids:
                continue
            ids.append(k)
            texts.append(t)
            if len(texts) >= 5:
                break
    if len(texts) < 5:
        raise RuntimeError("Could not find 5 L5_refined exemplar prompts in prompt bank.")
    return ids[:5], texts[:5]


def _normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms <= 1e-12, 1.0, norms)
    return mat / norms


def _encode_texts(encoder, texts: List[str], batch_size: int = 64) -> np.ndarray:
    tok, mdl = encoder
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)

    all_out: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tok(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            out = mdl(**enc)
            last_hidden = out.last_hidden_state  # [B, T, D]
            mask = enc["attention_mask"].unsqueeze(-1).to(last_hidden.dtype)  # [B, T, 1]
            summed = (last_hidden * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1e-9)
            pooled = summed / denom
            arr = pooled.cpu().numpy().astype(np.float32)
            all_out.append(arr)
    emb = np.concatenate(all_out, axis=0)
    return _normalize_rows(emb)


def _score_against_exemplars(texts: List[str], encoder, exemplar_emb: np.ndarray) -> np.ndarray:
    if not texts:
        return np.zeros((0,), dtype=np.float32)
    emb = _encode_texts(encoder, texts)
    sim = emb @ exemplar_emb.T
    return np.max(sim, axis=1).astype(np.float32)


def _find_seed_bridge_runs(repo_root: Path) -> List[Path]:
    roots = [
        repo_root / "results" / "phase1_mechanism" / "runs",
        repo_root / "results" / "remote_gpu_sync" / "2026-02-20" / "phase1_mechanism",
    ]
    runs: Dict[str, Path] = {}
    for root in roots:
        if not root.exists():
            continue
        for run_dir in root.glob("*seed_bridge*"):
            if not run_dir.is_dir():
                continue
            summary = run_dir / "summary.json"
            per_sample = run_dir / "per_sample.csv"
            if not summary.exists() or not per_sample.exists():
                continue
            key = run_dir.name.split("_rv_l27_activation_patching_bridge_")[-1]
            current = runs.get(key)
            if current is None or run_dir.stat().st_mtime > current.stat().st_mtime:
                runs[key] = run_dir
    return sorted(runs.values())


def _seed_bridge_condition(summary: Dict[str, Any]) -> str:
    patch_mode = summary.get("patch_mode")
    donor = summary.get("donor_type")
    if patch_mode == "head_specific" and donor == "recursive":
        return "head_specific"
    if patch_mode == "random_head" and donor == "recursive":
        return "random_head_control"
    if patch_mode == "head_specific" and donor == "baseline":
        return "baseline_donor_control"
    return "other"


def _collect_seed_bridge_rows(repo_root: Path) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for run_dir in _find_seed_bridge_runs(repo_root):
        summary_path = run_dir / "summary.json"
        per_sample_path = run_dir / "per_sample.csv"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        df = pd.read_csv(per_sample_path)
        if "patched_output" not in df.columns:
            continue
        seed = int(summary.get("seed")) if summary.get("seed") is not None else None
        condition = _seed_bridge_condition(summary)
        for row in df.to_dict(orient="records"):
            records.append(
                {
                    "dataset": "seed_bridge",
                    "run_dir": str(run_dir),
                    "run_name": summary.get("run_name", run_dir.name),
                    "seed": seed,
                    "condition": condition,
                    "patch_mode": summary.get("patch_mode"),
                    "donor_type": summary.get("donor_type"),
                    "rec_id": row.get("rec_id"),
                    "base_id": row.get("base_id"),
                    "rv_patch": row.get("rv_patch"),
                    "rv_base": row.get("rv_base"),
                    "rv_delta": row.get("rv_delta"),
                    "text": row.get("patched_output", ""),
                    "baseline_text": row.get("baseline_output", ""),
                }
            )
    return pd.DataFrame(records)


def _collect_c2_rows(repo_root: Path) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    # Include known C2 result locations only (avoid very deep full-tree scans).
    results_root = repo_root / "results"
    candidate_paths: List[Path] = []
    candidate_paths.extend((results_root / "canonical" / "c2_measurement_suite").glob("*/c2_rv_measurement.csv"))
    candidate_paths.extend((results_root / "phase1_mechanism" / "runs").glob("*c2_rv_measurement*/c2_rv_measurement.csv"))
    candidate_paths.extend((results_root / "runs").glob("*c2_rv_measurement*/c2_rv_measurement.csv"))

    seen = set()
    for path in candidate_paths:
        p = path.resolve()
        if p in seen:
            continue
        seen.add(p)
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        needed = {"generated_text", "rv_mean", "config"}
        if not needed.issubset(set(df.columns)):
            continue
        for row in df.to_dict(orient="records"):
            records.append(
                {
                    "dataset": "c2",
                    "source_csv": str(p),
                    "config": row.get("config"),
                    "prompt_idx": row.get("prompt_idx"),
                    "rv_mean": row.get("rv_mean"),
                    "text": row.get("generated_text", ""),
                }
            )
    return pd.DataFrame(records)


def _safe_float(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return float("nan")
    return v


def _spearman(x: pd.Series, y: pd.Series) -> Dict[str, Any]:
    xv = np.array([_safe_float(v) for v in x], dtype=float)
    yv = np.array([_safe_float(v) for v in y], dtype=float)
    mask = (~np.isnan(xv)) & (~np.isnan(yv))
    if int(mask.sum()) < 3:
        return {"n": int(mask.sum()), "rho": None, "p_value": None}
    rho, p = stats.spearmanr(xv[mask], yv[mask])
    if math.isnan(rho) or math.isnan(p):
        return {"n": int(mask.sum()), "rho": None, "p_value": None}
    return {"n": int(mask.sum()), "rho": float(rho), "p_value": float(p)}


def _rate(df: pd.DataFrame, group_col: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name, sub in df.groupby(group_col):
        if len(sub) == 0:
            continue
        out[str(name)] = {
            "n": int(len(sub)),
            "semantic_recursive_rate": float(np.mean(sub["semantic_recursive"].astype(float))),
            "semantic_score_mean": float(np.mean(sub["semantic_recursive_score"].astype(float))),
        }
    return out


def _welch_test(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if len(x) < 2 or len(y) < 2:
        return {"n_x": int(len(x)), "n_y": int(len(y)), "mean_diff": None, "p_value": None, "cohens_d": None}
    t, p = stats.ttest_ind(x, y, equal_var=False)
    sx = float(np.std(x, ddof=1))
    sy = float(np.std(y, ddof=1))
    nx, ny = len(x), len(y)
    sp = math.sqrt(((nx - 1) * sx * sx + (ny - 1) * sy * sy) / (nx + ny - 2)) if (nx + ny - 2) > 0 else float("nan")
    d = float((np.mean(x) - np.mean(y)) / sp) if sp > 1e-12 else 0.0
    return {
        "n_x": int(nx),
        "n_y": int(ny),
        "mean_diff": float(np.mean(x) - np.mean(y)),
        "p_value": float(p),
        "cohens_d": d,
    }


def _paired_seedwise_semantic(seed_df: pd.DataFrame, a: str, b: str) -> Dict[str, Any]:
    diffs: List[float] = []
    for seed, sub in seed_df.groupby("seed"):
        aa = sub[sub["condition"] == a]
        bb = sub[sub["condition"] == b]
        if aa.empty or bb.empty:
            continue
        amap = {
            (str(r.get("rec_id")), str(r.get("base_id"))): float(r.get("semantic_recursive_score"))
            for _, r in aa.iterrows()
        }
        bmap = {
            (str(r.get("rec_id")), str(r.get("base_id"))): float(r.get("semantic_recursive_score"))
            for _, r in bb.iterrows()
        }
        shared = sorted(set(amap).intersection(bmap))
        for key in shared:
            av = amap[key]
            bv = bmap[key]
            if math.isnan(av) or math.isnan(bv):
                continue
            diffs.append(av - bv)
    arr = np.array(diffs, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 3:
        return {"n_pairs": int(len(arr)), "mean_diff": None, "p_value": None, "cohens_d": None}
    t, p = stats.ttest_1samp(arr, 0.0)
    sd = float(np.std(arr, ddof=1))
    d = float(np.mean(arr) / sd) if sd > 1e-12 else 0.0
    return {
        "n_pairs": int(len(arr)),
        "mean_diff": float(np.mean(arr)),
        "p_value": float(p),
        "cohens_d": d,
    }


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = repo_root / "industry_grade" / "2026-02-20" / "evidence"
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_bank_path = repo_root / "prompts" / "bank.json"
    exemplar_ids, exemplar_texts = _resolve_exemplars(prompt_bank_path)

    encoder = _load_encoder(DEFAULT_MODEL)
    exemplar_emb = _encode_texts(encoder, exemplar_texts)

    seed_df = _collect_seed_bridge_rows(repo_root)
    c2_df = _collect_c2_rows(repo_root)

    if not seed_df.empty:
        seed_scores = _score_against_exemplars(seed_df["text"].fillna("").astype(str).tolist(), encoder, exemplar_emb)
        seed_df["semantic_recursive_score"] = seed_scores
        seed_df["semantic_recursive"] = seed_df["semantic_recursive_score"] > DEFAULT_THRESHOLD
        base_scores = _score_against_exemplars(seed_df["baseline_text"].fillna("").astype(str).tolist(), encoder, exemplar_emb)
        seed_df["baseline_semantic_score"] = base_scores
        seed_df["semantic_score_delta"] = seed_df["semantic_recursive_score"] - seed_df["baseline_semantic_score"]
    else:
        seed_df["semantic_recursive_score"] = []
        seed_df["semantic_recursive"] = []
        seed_df["baseline_semantic_score"] = []
        seed_df["semantic_score_delta"] = []

    if not c2_df.empty:
        c2_scores = _score_against_exemplars(c2_df["text"].fillna("").astype(str).tolist(), encoder, exemplar_emb)
        c2_df["semantic_recursive_score"] = c2_scores
        c2_df["semantic_recursive"] = c2_df["semantic_recursive_score"] > DEFAULT_THRESHOLD
    else:
        c2_df["semantic_recursive_score"] = []
        c2_df["semantic_recursive"] = []

    # Save raw scored rows.
    seed_csv = out_dir / "semantic_bridge_scores_seed_bridge.csv"
    c2_csv = out_dir / "semantic_bridge_scores_c2.csv"
    seed_df.to_csv(seed_csv, index=False)
    c2_df.to_csv(c2_csv, index=False)

    seed_corr_by_condition: Dict[str, Any] = {}
    for cond, sub in seed_df.groupby("condition") if not seed_df.empty else []:
        seed_corr_by_condition[str(cond)] = {
            "rv_delta_vs_semantic_score": _spearman(sub["rv_delta"], sub["semantic_recursive_score"]),
            "rv_patch_vs_semantic_score": _spearman(sub["rv_patch"], sub["semantic_recursive_score"]),
        }

    c2_corr_by_config: Dict[str, Any] = {}
    for cfg, sub in c2_df.groupby("config") if not c2_df.empty else []:
        c2_corr_by_config[str(cfg)] = {
            "rv_mean_vs_semantic_score": _spearman(sub["rv_mean"], sub["semantic_recursive_score"]),
        }

    seed_semantic_contrasts: Dict[str, Any] = {}
    if not seed_df.empty:
        contrasts = [
            ("head_specific", "random_head_control"),
            ("head_specific", "baseline_donor_control"),
            ("random_head_control", "baseline_donor_control"),
        ]
        for a, b in contrasts:
            xa = seed_df.loc[seed_df["condition"] == a, "semantic_recursive_score"].to_numpy(dtype=float)
            xb = seed_df.loc[seed_df["condition"] == b, "semantic_recursive_score"].to_numpy(dtype=float)
            seed_semantic_contrasts[f"{a}_vs_{b}"] = {
                "welch": _welch_test(xa, xb),
                "paired_seedwise": _paired_seedwise_semantic(seed_df, a, b),
            }

    summary = {
        "semantic_model": DEFAULT_MODEL,
        "threshold": DEFAULT_THRESHOLD,
        "exemplar_ids": exemplar_ids,
        "exemplar_preview": [t[:200] for t in exemplar_texts],
        "seed_bridge": {
            "n_rows": int(len(seed_df)),
            "runs": int(seed_df["run_dir"].nunique()) if not seed_df.empty else 0,
            "seeds": sorted([int(s) for s in seed_df["seed"].dropna().unique().tolist()]) if not seed_df.empty else [],
            "semantic_recursive_rate_by_condition": _rate(seed_df, "condition") if not seed_df.empty else {},
            "spearman_overall": {
                "rv_delta_vs_semantic_score": _spearman(seed_df["rv_delta"], seed_df["semantic_recursive_score"])
                if not seed_df.empty else {"n": 0, "rho": None, "p_value": None},
                "rv_patch_vs_semantic_score": _spearman(seed_df["rv_patch"], seed_df["semantic_recursive_score"])
                if not seed_df.empty else {"n": 0, "rho": None, "p_value": None},
            },
            "spearman_by_condition": seed_corr_by_condition,
            "semantic_score_contrasts": seed_semantic_contrasts,
        },
        "c2": {
            "n_rows": int(len(c2_df)),
            "sources": int(c2_df["source_csv"].nunique()) if not c2_df.empty else 0,
            "semantic_recursive_rate_by_config": _rate(c2_df, "config") if not c2_df.empty else {},
            "spearman_overall": {
                "rv_mean_vs_semantic_score": _spearman(c2_df["rv_mean"], c2_df["semantic_recursive_score"])
                if not c2_df.empty else {"n": 0, "rho": None, "p_value": None},
            },
            "spearman_by_config": c2_corr_by_config,
        },
        "artifacts": {
            "seed_bridge_scores_csv": str(seed_csv),
            "c2_scores_csv": str(c2_csv),
        },
    }

    json_path = out_dir / "semantic_behavior_analysis.json"
    md_path = out_dir / "semantic_behavior_analysis.md"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    lines = ["# Semantic Behavior Analysis\n"]
    lines.append(f"- model: `{DEFAULT_MODEL}`\n")
    lines.append(f"- threshold: `{DEFAULT_THRESHOLD}`\n")
    lines.append(f"- exemplar_ids: `{exemplar_ids}`\n\n")

    lines.append("## Seed Bridge\n")
    lines.append(f"- rows scored: `{summary['seed_bridge']['n_rows']}`\n")
    lines.append(f"- runs: `{summary['seed_bridge']['runs']}`\n")
    lines.append(f"- seeds: `{summary['seed_bridge']['seeds']}`\n")
    lines.append("- semantic_recursive_rate by condition:\n")
    for cond, vals in summary["seed_bridge"]["semantic_recursive_rate_by_condition"].items():
        lines.append(
            f"  - `{cond}`: rate={vals['semantic_recursive_rate']:.4f}, "
            f"mean_score={vals['semantic_score_mean']:.4f}, n={vals['n']}\n"
        )
    ov = summary["seed_bridge"]["spearman_overall"]["rv_delta_vs_semantic_score"]
    lines.append(
        f"- Spearman rv_delta vs semantic_score: rho={ov['rho']}, p={ov['p_value']}, n={ov['n']}\n"
    )
    ov2 = summary["seed_bridge"]["spearman_overall"]["rv_patch_vs_semantic_score"]
    lines.append(
        f"- Spearman rv_patch vs semantic_score: rho={ov2['rho']}, p={ov2['p_value']}, n={ov2['n']}\n\n"
    )
    if summary["seed_bridge"]["semantic_score_contrasts"]:
        lines.append("- semantic score contrasts:\n")
        for k, vals in summary["seed_bridge"]["semantic_score_contrasts"].items():
            w = vals["welch"]
            p = vals["paired_seedwise"]
            lines.append(
                f"  - `{k}` welch: diff={w['mean_diff']}, p={w['p_value']}, d={w['cohens_d']}; "
                f"paired: diff={p['mean_diff']}, p={p['p_value']}, d={p['cohens_d']}, n={p['n_pairs']}\n"
            )
        lines.append("\n")

    lines.append("## C2 Behavioral Transfer\n")
    lines.append(f"- rows scored: `{summary['c2']['n_rows']}`\n")
    lines.append(f"- sources: `{summary['c2']['sources']}`\n")
    lines.append("- semantic_recursive_rate by config:\n")
    for cfg, vals in summary["c2"]["semantic_recursive_rate_by_config"].items():
        lines.append(
            f"  - `{cfg}`: rate={vals['semantic_recursive_rate']:.4f}, "
            f"mean_score={vals['semantic_score_mean']:.4f}, n={vals['n']}\n"
        )
    c2ov = summary["c2"]["spearman_overall"]["rv_mean_vs_semantic_score"]
    lines.append(
        f"- Spearman rv_mean vs semantic_score: rho={c2ov['rho']}, p={c2ov['p_value']}, n={c2ov['n']}\n"
    )

    md_path.write_text("".join(lines), encoding="utf-8")

    print(json_path)
    print(md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
