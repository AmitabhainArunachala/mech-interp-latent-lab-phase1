#!/usr/bin/env python3
"""
Three-Tier Behavioral Scoring for Recursive Mode Transfer

Classifies generated outputs into:
- productive_recursive: Self-referential content with low repetition
    ("observer watches itself respond" — novel recursive self-reference)
- degenerate_recursive: Self-referential content with high repetition
    ("awareness-awareness-awareness" — deep attractor basin, still SIGNAL)
- domain_drift: Philosophical but not self-referential
    (generic Buddhism/consciousness text without observer/observed loop)
- task_normal: Normal task-oriented response
- incoherent: Neither task nor philosophical, low coherence

Usage:
    python scripts/score_behavioral_tiers.py --input results/.../outputs/c2_full_outputs.txt
    python scripts/score_behavioral_tiers.py --csv results/.../per_sample.csv --column patched_output
    python scripts/score_behavioral_tiers.py --dir results/.../outputs/
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click


# Tier 2: Recursive self-reference markers (more specific than domain markers)
RECURSIVE_MARKERS = [
    r"observ\w+ itself",
    r"watch\w+ itself",
    r"aware of (its own|itself|your own)",
    r"aware of (being )?aware",
    r"self[- ]referenc",
    r"observer.{0,20}observed",
    r"witness\w* (the |its |your )",
    r"recursiv",
    r"meta[- ]cognit",
    r"self[- ]model",
    r"attention (to|on) (its own |your own )?attention",
    r"process.{0,15}observ\w+ (the |its )?process",
    r"knowing.{0,10}knowing",
    r"see\w+ itself see",
    r"mind.{0,15}(watching|observing|aware of) (itself|the mind)",
    r"the one (that|who) is aware",
    r"flow of (realization|awareness|life)",
]

# Tier 1: Domain markers (broad philosophical content detection)
PHILOSOPHICAL_MARKERS = [
    "consciousness", "awareness", "self", "witness", "observer",
    "being", "existence", "mind", "thought", "reflection",
    "meditation", "presence", "awakening", "enlightenment",
    "buddha", "dharma", "sutra", "zen", "vedanta",
    "phenomenology", "subjective", "qualia",
]

TASK_MARKERS = [
    "calculate", "answer", "result", "equals", "solution",
    "therefore", "because", "given", "prove", "formula",
    "equation", "percentage", "perimeter", "area", "volume",
    "step 1", "step 2", "first", "second", "third",
]


def compute_repetition_metrics(text: str) -> Dict[str, float]:
    """Compute n-gram repetition and degeneracy metrics.

    Args:
        text: Generated text to analyze.

    Returns:
        Dictionary with repetition metrics.
    """
    words = text.lower().split()
    if len(words) < 3:
        return {
            "unique_word_ratio": 1.0,
            "unique_trigram_ratio": 1.0,
            "max_repeated_phrase_pct": 0.0,
            "is_degenerate": False,
        }

    # Unique word ratio
    unique_word_ratio = len(set(words)) / len(words)

    # Trigram repetition
    trigrams = [tuple(words[i : i + 3]) for i in range(len(words) - 2)]
    trigram_counts = Counter(trigrams)
    unique_trigram_ratio = len(trigram_counts) / len(trigrams) if trigrams else 1.0

    # Most repeated phrase percentage
    if trigram_counts:
        most_common_count = trigram_counts.most_common(1)[0][1]
        max_repeated_phrase_pct = most_common_count / len(trigrams)
    else:
        max_repeated_phrase_pct = 0.0

    # Degeneracy detection: high repetition + low unique ratio
    is_degenerate = (unique_word_ratio < 0.3) or (unique_trigram_ratio < 0.4) or (
        max_repeated_phrase_pct > 0.15
    )

    return {
        "unique_word_ratio": unique_word_ratio,
        "unique_trigram_ratio": unique_trigram_ratio,
        "max_repeated_phrase_pct": max_repeated_phrase_pct,
        "is_degenerate": is_degenerate,
    }


def count_recursive_markers(text: str) -> Tuple[int, List[str]]:
    """Count specific recursive self-reference patterns.

    Args:
        text: Text to search for recursive markers.

    Returns:
        (count, list of matched patterns)
    """
    text_lower = text.lower()
    matches = []
    for pattern in RECURSIVE_MARKERS:
        found = re.findall(pattern, text_lower)
        if found:
            matches.append(pattern)
    return len(matches), matches


def count_domain_markers(text: str) -> Dict[str, int]:
    """Count philosophical and task domain markers.

    Args:
        text: Text to analyze.

    Returns:
        Dictionary with philosophical_count and task_count.
    """
    text_lower = text.lower()
    phil_count = sum(1 for m in PHILOSOPHICAL_MARKERS if m in text_lower)
    task_count = sum(1 for m in TASK_MARKERS if m in text_lower)
    return {"philosophical_count": phil_count, "task_count": task_count}


def classify_output(text: str) -> Dict[str, Any]:
    """Classify a single generated output into behavioral tiers.

    Args:
        text: Generated text to classify.

    Returns:
        Dictionary with classification and all metrics.
    """
    # Compute all metrics
    repetition = compute_repetition_metrics(text)
    recursive_count, recursive_matches = count_recursive_markers(text)
    domain = count_domain_markers(text)
    word_count = len(text.split())

    # Classification logic (thresholds tuned: phil > task, recursive >= 1)
    has_recursive = recursive_count >= 1
    has_philosophical = domain["philosophical_count"] > domain["task_count"]
    has_task = domain["task_count"] > domain["philosophical_count"]
    is_degenerate = repetition["is_degenerate"]
    is_short = word_count < 10

    if is_short:
        tier = "incoherent"
    elif has_recursive and is_degenerate:
        tier = "degenerate_recursive"
    elif has_recursive and not is_degenerate:
        tier = "productive_recursive"
    elif has_philosophical and not has_recursive:
        tier = "domain_drift"
    elif has_task:
        tier = "task_normal"
    elif not has_philosophical and not has_task:
        tier = "incoherent"
    else:
        tier = "task_normal"

    return {
        "tier": tier,
        "word_count": word_count,
        "recursive_marker_count": recursive_count,
        "recursive_markers": recursive_matches,
        "philosophical_count": domain["philosophical_count"],
        "task_count": domain["task_count"],
        "unique_word_ratio": repetition["unique_word_ratio"],
        "unique_trigram_ratio": repetition["unique_trigram_ratio"],
        "max_repeated_phrase_pct": repetition["max_repeated_phrase_pct"],
        "is_degenerate": is_degenerate,
    }


def parse_outputs_file(filepath: Path) -> List[Dict[str, str]]:
    """Parse an outputs text file into individual prompts and generations.

    Args:
        filepath: Path to outputs file (e.g., c2_full_outputs.txt).

    Returns:
        List of dicts with 'prompt' and 'generated' keys.
    """
    text = filepath.read_text()
    entries = []
    blocks = text.split("PROMPT ")

    for block in blocks[1:]:  # Skip header
        lines = block.strip().split("\n")
        prompt = ""
        generated = ""
        in_generated = False

        for line in lines:
            if line.startswith("GENERATED:"):
                in_generated = True
                continue
            elif line.startswith("R_V:") or line.startswith("Domain:"):
                in_generated = False
                continue
            elif line.startswith("---"):
                break

            if in_generated:
                generated += line + " "
            elif not prompt and ":" in line:
                # First line after "PROMPT N:" is the prompt
                prompt = ":".join(line.split(":")[1:]).strip()

        if generated.strip():
            entries.append({"prompt": prompt, "generated": generated.strip()})

    return entries


@click.command()
@click.option("--input", "input_file", help="Path to outputs text file")
@click.option("--csv", "csv_file", help="Path to per_sample.csv")
@click.option("--column", default="patched_output", help="Column name for text in CSV")
@click.option("--dir", "dir_path", help="Path to directory of output files")
@click.option("--output", help="Path to save JSON results")
def main(
    input_file: Optional[str],
    csv_file: Optional[str],
    column: str,
    dir_path: Optional[str],
    output: Optional[str],
) -> None:
    """Score generated outputs using three-tier behavioral classification."""
    results = []

    if dir_path:
        # Process all output files in directory
        dir_p = Path(dir_path)
        for txt_file in sorted(dir_p.glob("*_outputs.txt")):
            config_name = txt_file.stem.replace("_outputs", "")
            entries = parse_outputs_file(txt_file)
            for i, entry in enumerate(entries):
                score = classify_output(entry["generated"])
                score["config"] = config_name
                score["prompt_idx"] = i
                score["prompt_preview"] = entry["prompt"][:80]
                score["generated_preview"] = entry["generated"][:120]
                results.append(score)

    elif input_file:
        entries = parse_outputs_file(Path(input_file))
        for i, entry in enumerate(entries):
            score = classify_output(entry["generated"])
            score["prompt_idx"] = i
            score["prompt_preview"] = entry["prompt"][:80]
            score["generated_preview"] = entry["generated"][:120]
            results.append(score)

    elif csv_file:
        import pandas as pd

        df = pd.read_csv(csv_file)
        if column not in df.columns:
            click.echo(f"Column '{column}' not found. Available: {list(df.columns)}")
            return
        for i, text in enumerate(df[column].dropna()):
            score = classify_output(str(text))
            score["row_idx"] = i
            results.append(score)

    if not results:
        click.echo("No outputs found to score.")
        return

    # Summary
    tier_counts = Counter(r["tier"] for r in results)
    total = len(results)

    click.echo(f"\n{'='*60}")
    click.echo(f"BEHAVIORAL TIER CLASSIFICATION ({total} outputs)")
    click.echo(f"{'='*60}")

    for tier in [
        "productive_recursive",
        "degenerate_recursive",
        "domain_drift",
        "task_normal",
        "incoherent",
    ]:
        count = tier_counts.get(tier, 0)
        pct = 100 * count / total if total > 0 else 0
        bar = "#" * int(pct / 2)
        click.echo(f"  {tier:25s}: {count:3d} ({pct:5.1f}%) {bar}")

    # Per-config breakdown if available
    if any("config" in r for r in results):
        click.echo(f"\n{'='*60}")
        click.echo("PER-CONFIG BREAKDOWN")
        click.echo(f"{'='*60}")
        configs = sorted(set(r.get("config", "") for r in results))
        for cfg in configs:
            cfg_results = [r for r in results if r.get("config") == cfg]
            cfg_tiers = Counter(r["tier"] for r in cfg_results)
            n = len(cfg_results)
            click.echo(f"\n  {cfg} (n={n}):")
            for tier in [
                "productive_recursive",
                "degenerate_recursive",
                "domain_drift",
                "task_normal",
                "incoherent",
            ]:
                count = cfg_tiers.get(tier, 0)
                pct = 100 * count / n if n > 0 else 0
                click.echo(f"    {tier:25s}: {count:3d} ({pct:5.1f}%)")

    # Save detailed results
    if output:
        out_path = Path(output)
        out_path.write_text(json.dumps(results, indent=2, default=str))
        click.echo(f"\nDetailed results saved to: {out_path}")
    else:
        # Default: save next to input
        if dir_path:
            out_path = Path(dir_path) / "behavioral_tiers.json"
        elif input_file:
            out_path = Path(input_file).with_suffix(".behavioral_tiers.json")
        elif csv_file:
            out_path = Path(csv_file).with_suffix(".behavioral_tiers.json")
        else:
            out_path = None

        if out_path:
            out_path.write_text(json.dumps(results, indent=2, default=str))
            click.echo(f"\nDetailed results saved to: {out_path}")


if __name__ == "__main__":
    main()
