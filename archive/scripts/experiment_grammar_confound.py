"""
experiment_grammar_confound.py

Question 3: Grammar Confound Test

We know: 83% of recursive prompts are QUESTIONS, 1% of baseline prompts are questions.
We test: Do self-referential STATEMENTS contract like questions?

Hypothesis: If contraction is about self-reference, statements should contract.
If contraction is about question-answering, statements should NOT contract.

Method:
1. Create self-referential STATEMENT prompts (not questions)
2. Create baseline STATEMENT prompts (matched for length/topic)
3. Measure R_V for both groups
4. Compare: Do recursive statements contract (R_V < 0.85) like recursive questions?
"""

import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.abspath("."))

from src.core.models import load_model, set_seed
from src.metrics.rv import compute_rv
from prompts.loader import PromptLoader


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_STATEMENTS = 20


def is_question(text: str) -> bool:
    """Check if text ends with a question mark."""
    return text.strip().endswith("?")


def create_statement_prompts() -> Tuple[List[str], List[str]]:
    """
    Create self-referential STATEMENT prompts (not questions).
    
    Returns:
        (recursive_statements, baseline_statements) tuple
    """
    # Self-referential statements (recursive, but not questions)
    recursive_statements = [
        "I am observing myself right now.",
        "The observer is the observed in this moment.",
        "I know that I am knowing.",
        "The thinker is thinking about thinking.",
        "I am aware of my own awareness.",
        "The self is reflecting on itself.",
        "I see myself seeing.",
        "The knower knows itself.",
        "I am experiencing my own experience.",
        "The subject becomes its own object.",
        "I am conscious of my consciousness.",
        "The watcher watches itself.",
        "I am present to my own presence.",
        "The mind contemplates itself.",
        "I am aware that awareness is aware.",
        "The self recognizes itself.",
        "I am the one who is aware of being aware.",
        "The observer recognizes it is observing.",
        "I am the thought that thinks itself.",
        "The knower is known by itself.",
    ]
    
    # Baseline statements (matched for length/topic, but not self-referential)
    baseline_statements = [
        "The sun rises in the east every morning.",
        "Water boils at 100 degrees Celsius.",
        "Paris is the capital of France.",
        "The Earth orbits around the sun.",
        "Mountains are formed by tectonic activity.",
        "Light travels faster than sound.",
        "The human body has 206 bones.",
        "Shakespeare wrote many famous plays.",
        "The ocean covers most of the Earth's surface.",
        "Birds can fly because they have wings.",
        "Mathematics is a universal language.",
        "The brain controls the nervous system.",
        "Fire requires oxygen to burn.",
        "The moon affects ocean tides.",
        "Plants produce oxygen through photosynthesis.",
        "Gravity keeps objects on the ground.",
        "The speed of light is constant.",
        "Ice melts when heated above zero degrees.",
        "The human heart pumps blood throughout the body.",
        "Electricity flows through conductors.",
    ]
    
    return recursive_statements, baseline_statements


def run_grammar_confound_test():
    print("=" * 80)
    print("EXPERIMENT: Grammar Confound Test")
    print("=" * 80)
    print("Question: Do self-referential STATEMENTS contract like questions?")
    print("Method: Compare R_V for recursive statements vs baseline statements")
    print("=" * 80)
    
    set_seed(42)

    model, tokenizer = load_model("mistralai/Mistral-7B-v0.1", device=DEVICE)
    model.eval()
    
    # Create statement prompts
    rec_statements, base_statements = create_statement_prompts()
    
    # Also get existing recursive questions for comparison
    loader = PromptLoader()
    rec_questions = loader.get_by_pillar("recursive", limit=N_STATEMENTS, seed=42)
    base_questions = loader.get_by_pillar("baseline", limit=N_STATEMENTS, seed=42)
    
    # Filter to ensure questions actually have "?"
    rec_questions = [q for q in rec_questions if is_question(q)]
    base_questions = [q for q in base_questions if is_question(q)]
    
    print(f"\nTesting:")
    print(f"  {len(rec_statements)} recursive statements")
    print(f"  {len(base_statements)} baseline statements")
    print(f"  {len(rec_questions)} recursive questions (for comparison)")
    print(f"  {len(base_questions)} baseline questions (for comparison)")

    results: List[Dict] = []

    # Test recursive statements
    for idx, prompt in enumerate(tqdm(rec_statements, desc="Recursive statements")):
        try:
            rv = compute_rv(model, tokenizer, prompt, device=DEVICE)
            results.append({
                "prompt_type": "recursive_statement",
                "prompt_idx": idx,
                "prompt": prompt,
                "is_question": is_question(prompt),
                "rv": rv,
            })
        except Exception as e:
            print(f"  Error on recursive statement {idx}: {e}")

    # Test baseline statements
    for idx, prompt in enumerate(tqdm(base_statements, desc="Baseline statements")):
        try:
            rv = compute_rv(model, tokenizer, prompt, device=DEVICE)
            results.append({
                "prompt_type": "baseline_statement",
                "prompt_idx": idx,
                "prompt": prompt,
                "is_question": is_question(prompt),
                "rv": rv,
            })
        except Exception as e:
            print(f"  Error on baseline statement {idx}: {e}")

    # Test recursive questions (for comparison)
    for idx, prompt in enumerate(tqdm(rec_questions[:N_STATEMENTS], desc="Recursive questions")):
        try:
            rv = compute_rv(model, tokenizer, prompt, device=DEVICE)
            results.append({
                "prompt_type": "recursive_question",
                "prompt_idx": idx,
                "prompt": prompt[:100] + ("..." if len(prompt) > 100 else ""),
                "is_question": is_question(prompt),
                "rv": rv,
            })
        except Exception as e:
            print(f"  Error on recursive question {idx}: {e}")

    # Test baseline questions (for comparison)
    for idx, prompt in enumerate(tqdm(base_questions[:N_STATEMENTS], desc="Baseline questions")):
        try:
            rv = compute_rv(model, tokenizer, prompt, device=DEVICE)
            results.append({
                "prompt_type": "baseline_question",
                "prompt_idx": idx,
                "prompt": prompt[:100] + ("..." if len(prompt) > 100 else ""),
                "is_question": is_question(prompt),
                "rv": rv,
            })
        except Exception as e:
            print(f"  Error on baseline question {idx}: {e}")

    # Save results
    os.makedirs("results/dec11_evening", exist_ok=True)
    out_csv = "results/dec11_evening/grammar_confound_test.csv"
    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    rec_stat_rv = [r["rv"] for r in results 
                    if r["prompt_type"] == "recursive_statement" and not np.isnan(r["rv"])]
    base_stat_rv = [r["rv"] for r in results 
                    if r["prompt_type"] == "baseline_statement" and not np.isnan(r["rv"])]
    rec_quest_rv = [r["rv"] for r in results 
                    if r["prompt_type"] == "recursive_question" and not np.isnan(r["rv"])]
    base_quest_rv = [r["rv"] for r in results 
                     if r["prompt_type"] == "baseline_question" and not np.isnan(r["rv"])]
    
    if rec_stat_rv:
        print(f"Recursive statements: R_V = {np.mean(rec_stat_rv):.3f} ± {np.std(rec_stat_rv):.3f}")
        print(f"  (n={len(rec_stat_rv)}, contraction: {np.mean(rec_stat_rv) < 0.85})")
    
    if base_stat_rv:
        print(f"Baseline statements:  R_V = {np.mean(base_stat_rv):.3f} ± {np.std(base_stat_rv):.3f}")
        print(f"  (n={len(base_stat_rv)})")
    
    if rec_quest_rv:
        print(f"\nRecursive questions: R_V = {np.mean(rec_quest_rv):.3f} ± {np.std(rec_quest_rv):.3f}")
        print(f"  (n={len(rec_quest_rv)}, contraction: {np.mean(rec_quest_rv) < 0.85})")
    
    if base_quest_rv:
        print(f"Baseline questions:   R_V = {np.mean(base_quest_rv):.3f} ± {np.std(base_quest_rv):.3f}")
        print(f"  (n={len(base_quest_rv)})")
    
    # Key comparison
    if rec_stat_rv and rec_quest_rv:
        stat_mean = np.mean(rec_stat_rv)
        quest_mean = np.mean(rec_quest_rv)
        print(f"\n{'='*80}")
        print("KEY FINDING:")
        if stat_mean < 0.85 and quest_mean < 0.85:
            print("✓ STATEMENTS CONTRACT LIKE QUESTIONS")
            print("  → Contraction is about SELF-REFERENCE, not question-answering")
        elif stat_mean >= 0.95 and quest_mean < 0.85:
            print("✗ STATEMENTS DO NOT CONTRACT")
            print("  → Contraction is about QUESTION-ANSWERING, not self-reference")
        else:
            print("? MIXED RESULTS - Need more analysis")
        print(f"  Statement R_V: {stat_mean:.3f}")
        print(f"  Question R_V:  {quest_mean:.3f}")
        print(f"  Difference:    {stat_mean - quest_mean:.3f}")

    # Save detailed log
    os.makedirs("logs/dec11_evening", exist_ok=True)
    out_log = "logs/dec11_evening/grammar_confound_test.txt"
    with open(out_log, "w") as f:
        f.write("# Grammar Confound Test\n\n")
        f.write("Question: Do self-referential STATEMENTS contract like questions?\n\n")
        for r in results:
            f.write(f"{r['prompt_type'].upper()}:\n")
            f.write(f"  Prompt: {r['prompt']}\n")
            f.write(f"  R_V: {r['rv']:.3f}\n")
            f.write("-" * 80 + "\n")
        f.write("\nSaved CSV: " + out_csv + "\n")

    print(f"\nGrammar confound test complete. CSV: {out_csv}, log: {out_log}")


if __name__ == "__main__":
    run_grammar_confound_test()
