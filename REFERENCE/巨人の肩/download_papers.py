#!/usr/bin/env python3
"""Download core mech-interp papers (PDFs) into this repo.

Usage (from repo root, in a networked environment):

    cd REFERENCE/巨人の肩
    python download_papers.py

This script is intentionally simple: you can extend the `PAPERS` list
with more URLs or fix any that change. It will save PDFs under
`REFERENCE/巨人の肩/pdfs/` with safe filenames.
"""

import os
import pathlib
import textwrap
from urllib.parse import urlparse

import sys

try:
    import requests
except ImportError:
    print("[download_papers] Please `pip install requests` first.")
    sys.exit(1)

# Core set of ~15 high-priority papers.
# For arXiv entries, we use the direct PDF URL pattern.
# Some entries are non-arXiv and use their canonical PDF links.
# If any URL 404s, you can update it manually.

PAPERS = [
    # === CORE CIRCUIT ANALYSIS ===
    {
        "name": "IOI_Indirect_Object_Identification_Wang_2023",
        "url": "https://arxiv.org/pdf/2211.00593.pdf",  # Interpretability in the Wild (IOI circuit)
    },
    {
        "name": "ACDC_Automated_Circuit_Discovery_Conmy_2023",
        "url": "https://arxiv.org/pdf/2304.14997.pdf",
    },
    {
        "name": "In_Context_Learning_Induction_Heads_Olsson_2022",
        "url": "https://arxiv.org/pdf/2209.11895.pdf",
    },

    # === ACTIVATION PATCHING & CAUSAL METHODS ===
    {
        "name": "ROME_Locating_Editing_Factual_Associations_Meng_2022",
        "url": "https://arxiv.org/pdf/2202.05262.pdf",
    },
    {
        "name": "Best_Practices_Activation_Patching_Zhang_Nanda_2024",
        "url": "https://arxiv.org/pdf/2405.07852.pdf",
    },
    {
        "name": "Causal_Scrubbing_Chan_Geiger_2023",
        "url": "https://arxiv.org/pdf/2310.17682.pdf",
    },
    {
        "name": "Distributed_Alignment_Search_DAS_Geiger_2024",
        # Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations
        "url": "https://arxiv.org/pdf/2303.02536.pdf",
    },

    # === LINEAR REPRESENTATIONS & SUBSPACES ===
    {
        "name": "Linear_Representation_Hypothesis_Park_2024",
        "url": "https://arxiv.org/pdf/2310.10781.pdf",
    },
    {
        "name": "Attention_Low_Dimensional_Subspaces_Wang_2025",
        # Attention Layers Add Into Low-Dimensional Residual Subspaces
        "url": "https://arxiv.org/pdf/2508.16929.pdf",
    },
    {
        "name": "Function_Vectors_Todd_2024",
        # Function Vectors in Large Language Models (ICLR 2024)
        "url": "https://arxiv.org/pdf/2310.15213.pdf",
    },

    # === SPARSE AUTOENCODERS & FEATURES ===
    {
        "name": "Towards_Monosemanticity_Anthropic_2023",
        # Towards Monosemanticity: Decomposing Language Models With Dictionary Learning
        # Note: transformer-circuits.pub blocks direct PDF download; this is the HTML version
        "url": "https://transformer-circuits.pub/2023/monosemantic-features/index.html",
    },
    {
        "name": "Sparse_Feature_Circuits_Marks_2024",
        # Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs
        "url": "https://arxiv.org/pdf/2403.19647.pdf",
    },

    # === MEMORY & KNOWLEDGE IN TRANSFORMERS ===
    {
        "name": "Understanding_Factual_Recall_Associative_Memories_2024",
        # Understanding Factual Recall in Transformers via Associative Memories
        "url": "https://arxiv.org/pdf/2412.06538.pdf",
    },
    {
        "name": "Knowledge_Probing_BELIEF_Benchmark_2024",
        # What Matters in Memorizing and Recalling Facts? Multifaceted Benchmarks for Knowledge Probing
        "url": "https://arxiv.org/pdf/2406.12277.pdf",
    },

    # === INTERPRETABILITY CRITIQUE & RIGOR ===
    {
        "name": "Interpretability_Illusions_Makelov_2023",
        # Is This the Subspace You Are Looking for? An Interpretability Illusion
        "url": "https://arxiv.org/pdf/2311.17030.pdf",
    },
    {
        "name": "Reply_Interpretability_Illusions_2024",
        # A Reply to Makelov et al. (2023)'s "Interpretability Illusion" Arguments
        "url": "https://arxiv.org/pdf/2401.12631.pdf",
    },

    # === RECURSIVE SELF-IMPROVEMENT (closest to recursive self-reference) ===
    {
        "name": "Recursive_Introspection_RISE_2024",
        # Recursive Introspection: Teaching Language Model Agents How to Self-Improve
        "url": "https://arxiv.org/pdf/2407.18219.pdf",
    },
]


def safe_filename(name: str) -> str:
    # Replace spaces and problematic chars
    return "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in name)


def download(url: str, dest: pathlib.Path) -> None:
    print(f"[download] GET {url}")
    try:
        resp = requests.get(url, timeout=30)
    except Exception as e:
        print(f"  ! request failed: {e}")
        return
    if resp.status_code != 200:
        print(f"  ! HTTP {resp.status_code} for {url}")
        return
    dest.write_bytes(resp.content)
    print(f"  → saved to {dest}")


def main() -> None:
    root = pathlib.Path(__file__).resolve().parent
    pdf_dir = root / "pdfs"
    pdf_dir.mkdir(exist_ok=True)

    print("Downloading core mech-interp papers into:", pdf_dir)
    print("You can edit `PAPERS` in download_papers.py to fix URLs or add more.")

    for paper in PAPERS:
        name = safe_filename(paper["name"])
        url = paper["url"]
        # Derive extension from URL if present, else default to .pdf
        parsed = urlparse(url)
        ext = ".pdf" if not os.path.splitext(parsed.path)[1] else os.path.splitext(parsed.path)[1]
        dest = pdf_dir / f"{name}{ext}"
        if dest.exists():
            print(f"[skip] {dest.name} already exists")
            continue
        download(url, dest)


if __name__ == "__main__":
    main()
