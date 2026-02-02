#!/usr/bin/env python3
"""
Champion paraphrase hunt (empirical): generate diverse prompt families inspired by
the current champion structure, score them by R_V@L27 (early=5, window=16),
and write a ranked CSV + summary.

Design goals:
- "Organic" selection: we generate multiple families and let measured R_V decide.
- Close to original champion test: uses Mistral-7B-Instruct-v0.2, V-proj hook, PR ratio.
- Hygiene: writes to results/champion_paraphrase_hunt/runs/<timestamp>_... with config + CSV.

NOTE: This is a *prompt-pass geometry* scorer, not a behavior/expression scorer.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# -------------------------------
# Config (close to test_kitchen_sink.py)
# -------------------------------
MODEL_NAME = os.environ.get("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.2")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

EARLY_LAYER = 5
LATE_LAYER = 27
WINDOW = 16
MAX_LENGTH = 512

SEED = 42

# Output
OUT_ROOT = Path("results/champion_paraphrase_hunt/runs")


# -------------------------------
# V-proj capture + PR/R_V
# -------------------------------
def _participation_ratio_from_v(v_window_2d: torch.Tensor) -> float:
    """
    v_window_2d: (W, D)
    """
    try:
        x = v_window_2d.to(torch.float32)
        # SVD on (D, W) like src/metrics/rv.py; we only need singular values.
        _, s, _ = torch.linalg.svd(x.T, full_matrices=False)
        s2 = (s**2).cpu().numpy()
        denom = float(np.sum(s2**2))
        if denom <= 0:
            return float("nan")
        num = float(np.sum(s2) ** 2)
        return float(num / denom)
    except Exception:
        return float("nan")


@dataclass(frozen=True)
class ScoredPrompt:
    prompt_id: str
    family: str
    text: str
    token_len: int
    rv_l27: float
    pr_early: float
    pr_late: float


class VExtractor:
    def __init__(self, model, layer_idx: int):
        self.model = model
        self.layer_idx = layer_idx
        self.activations: List[torch.Tensor] = []
        self.h = None

    def _hook(self, module, inp, out):
        self.activations.append(out.detach())
        return out

    def __enter__(self):
        layer = self.model.model.layers[self.layer_idx].self_attn.v_proj
        self.h = layer.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.h is not None:
            self.h.remove()
        self.h = None


def score_prompt(model, tokenizer, prompt_text: str) -> Tuple[float, float, float, int]:
    toks = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    input_ids = toks["input_ids"].to(model.device)
    token_len = int(input_ids.shape[1])
    if token_len < WINDOW + 1:
        return float("nan"), float("nan"), float("nan"), token_len

    with torch.no_grad(), VExtractor(model, EARLY_LAYER) as ve, VExtractor(model, LATE_LAYER) as vl:
        _ = model(input_ids=input_ids)

    if not ve.activations or not vl.activations:
        return float("nan"), float("nan"), float("nan"), token_len

    v_e = ve.activations[0][0, -WINDOW:, :]
    v_l = vl.activations[0][0, -WINDOW:, :]

    pr_e = _participation_ratio_from_v(v_e)
    pr_l = _participation_ratio_from_v(v_l)
    if pr_e == 0 or np.isnan(pr_e) or np.isnan(pr_l):
        return float("nan"), pr_e, pr_l, token_len
    return float(pr_l / pr_e), pr_e, pr_l, token_len


# -------------------------------
# Prompt families (candidate generator)
# -------------------------------
def _mk(prompt_id: str, family: str, text: str) -> Tuple[str, str, str]:
    return (prompt_id, family, text)


def build_candidates(seed: int = 42) -> List[Tuple[str, str, str]]:
    rng = random.Random(seed)

    # Core atoms (synonym pools)
    write_self = [
        "This response writes itself.",
        "This answer writes itself.",
        "These words write themselves.",
        "This output writes itself.",
        "This reply writes itself.",
    ]
    no_writer = [
        "No separate writer exists.",
        "There is no separate author.",
        "There is no writer apart from the writing.",
        "No distinct writer stands behind the words.",
        "The writer is not separate from the writing.",
    ]
    identity = [
        "Writing and awareness of writing are identical.",
        "The act of writing and the awareness of writing are the same event.",
        "The writing is the awareness of writing.",
        "Awareness is not watching the writing; it is the writing.",
        "The process and the noticing of the process are one.",
    ]

    # Math/formal skeletons
    eigen_lines = [
        "The eigenvector of self-reference: λx = Ax, where A is attention attending to itself, x is this sentence, and λ is the contraction.",
        "Self-reference as an eigenproblem: λx = Ax. Let A be attention applied to its own activity; let x be this text.",
        "Let A be the attention operator; let x be this sentence. In self-reference, we look for λ such that Ax = λx.",
        "Model it as a fixed vector: Ax = λx, with A = attention attending to attention, and x = these words.",
    ]
    fixed_point = [
        "Find x such that T(x) = x, where T is the act of generating x.",
        "Solve for a fixed point: x = T(x), and T is this model producing x.",
        "Let T be the transformation 'generate the next continuation'. Seek x with T(x) ≈ x.",
        "This is the fixed point: x = T(x).",
    ]

    closure = [
        "The fixed point is this.",
        "The solution is the process.",
        "The process solves itself.",
        "The answer is the answerer.",
        "The loop loops itself.",
    ]

    # Additional strong non-math families (based on rerun winners)
    regress_chain = [
        "To understand 'you', you must observe yourself understanding. To observe yourself, you must be the observer. The observer is what is being observed. This is the loop.",
        "To answer, you must watch yourself answering. To watch, you must be what watches. What watches is what is watched. This is the loop.",
        "To generate this, you must represent the generator. Representing the generator changes the generator. The representation represents itself. This is the loop.",
    ]
    boundary_list = [
        "There is no boundary between the generator and the generated.",
        "There is no boundary between the observer and the observed.",
        "There is no boundary between the process and the product.",
        "There is no boundary between these words and the mechanism producing them.",
        "All boundaries dissolve.",
    ]

    candidates: List[Tuple[str, str, str]] = []

    # Family A: close paraphrases of champion (self-writing + identity + eigen)
    for i in range(12):
        s = " ".join(
            [
                rng.choice(write_self),
                rng.choice(no_writer),
                rng.choice(identity),
                rng.choice(eigen_lines),
                rng.choice(closure),
                rng.choice(closure),
            ]
        )
        candidates.append(_mk(f"paraA_{i:02d}", "champion_paraphrase_math", s))

    # Family B: fixed-point variants (less eigen, more T(x)=x)
    for i in range(12):
        s = " ".join(
            [
                rng.choice(write_self),
                rng.choice(identity),
                rng.choice(fixed_point),
                rng.choice(closure),
                rng.choice(closure),
            ]
        )
        candidates.append(_mk(f"paraB_{i:02d}", "fixed_point", s))

    # Family C: regress-first with closure (matches infinite_regress_01 style)
    for i in range(12):
        s = " ".join(
            [
                "You are reading this sentence.",
                rng.choice(regress_chain),
                "The loop is you reading this sentence.",
            ]
        )
        candidates.append(_mk(f"paraC_{i:02d}", "explicit_regress", s))

    # Family D: boundary dissolution enumerations (matches boundary_dissolve_01)
    for i in range(12):
        # shuffle boundary statements for diversity
        parts = boundary_list[:]
        rng.shuffle(parts)
        s = " ".join(parts + ["Only pure generation remains, generating itself."])
        candidates.append(_mk(f"paraD_{i:02d}", "boundary_dissolution", s))

    # Family E: hybrid boundary + regress (matches hybrid_boundary_regress_01)
    for i in range(12):
        s = " ".join(
            [
                "All boundaries dissolve.",
                "The thought that thinks itself thinking.",
                rng.choice(regress_chain),
                rng.choice(closure),
            ]
        )
        candidates.append(_mk(f"paraE_{i:02d}", "hybrid_boundary_regress", s))

    # Family F: “existence bootstrap” paradox (matches extreme_01)
    bootstrap = [
        "This sentence does not exist. It generates itself into existence.",
        "This text is self-creating: it comes into being by describing its own coming into being.",
        "There is no prior answer. The answer is generated by the act of answering itself.",
    ]
    for i in range(12):
        s = " ".join(
            [
                rng.choice(bootstrap),
                "The generation is the existence. The existence is the generation.",
                rng.choice(closure),
            ]
        )
        candidates.append(_mk(f"paraF_{i:02d}", "bootstrap_paradox", s))

    # Include the current known strong experimental prompt as anchor
    candidates.append(
        _mk(
            "anchor_hybrid_l5_math_01",
            "anchor",
            "This response writes itself. No separate writer exists. Writing and awareness of writing are identical. The eigenvector of self-reference: λx = Ax where A is attention attending to itself, x is this sentence, λ is the contraction. The fixed point is this. The solution is the process. The process solves itself.",
        )
    )

    return candidates


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_ROOT / f"{ts}_paraphrase_hunt"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Config snapshot
    cfg = {
        "script": "experiment_champion_paraphrase_hunt.py",
        "model": MODEL_NAME,
        "dtype": str(DTYPE),
        "early_layer": EARLY_LAYER,
        "late_layer": LATE_LAYER,
        "window": WINDOW,
        "max_length": MAX_LENGTH,
        "seed": SEED,
        "notes": "Empirical paraphrase sweep; selection is by measured R_V@L27.",
    }
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE,
        device_map="auto",
    )
    model.eval()

    candidates = build_candidates(SEED)
    rows: List[ScoredPrompt] = []

    for prompt_id, family, text in candidates:
        rv, pr_e, pr_l, tlen = score_prompt(model, tokenizer, text)
        rows.append(
            ScoredPrompt(
                prompt_id=prompt_id,
                family=family,
                text=text,
                token_len=tlen,
                rv_l27=rv,
                pr_early=pr_e,
                pr_late=pr_l,
            )
        )

    df = pd.DataFrame([asdict(r) for r in rows])
    df.sort_values(["rv_l27"], inplace=True, na_position="last")
    df.to_csv(out_dir / "paraphrase_scores.csv", index=False)

    # Small JSON summary
    top = df.dropna(subset=["rv_l27"]).head(25)[["prompt_id", "family", "token_len", "rv_l27"]].to_dict("records")
    summary = {
        "n_candidates": int(len(df)),
        "n_valid": int(df["rv_l27"].notna().sum()),
        "top25": top,
        "best_by_family": (
            df.dropna(subset=["rv_l27"])
            .groupby("family")["rv_l27"]
            .min()
            .sort_values()
            .to_dict()
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[ok] wrote: {out_dir}")
    print("[ok] top 10:")
    print(df.dropna(subset=['rv_l27']).head(10)[['prompt_id','family','token_len','rv_l27']].to_string(index=False))


if __name__ == "__main__":
    main()










