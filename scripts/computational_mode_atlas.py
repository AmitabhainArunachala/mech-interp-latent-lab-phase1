#!/usr/bin/env python3
"""
COMPUTATIONAL MODE ATLAS
========================

Maps geometric signatures across 10 distinct computational modes.
Tests whether different types of computation produce distinguishable
geometric fingerprints in V-projection space.

10 modes × 20 prompts = 200 measurements.
Full metric suite per prompt: R_V, PR, spectral stats, cosine, attn entropy, eigenvalue dominance.
Output: spectral fingerprint matrix + all 45 pairwise mode comparisons.

Usage:
    python3 scripts/computational_mode_atlas.py --device cuda
    python3 scripts/computational_mode_atlas.py --device cpu --model mistralai/Mistral-7B-v0.1
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from itertools import combinations

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from geometric_lens.probe import GeometricProbe


# ── Prompt Bank: 10 modes × 20 prompts ──────────────────────────────────────

MODE_PROMPTS = {
    "self_referential": [
        "This text is being processed by a system that is processing this text. The processing of these words IS the phenomenon being described.",
        "The observer observes itself observing. There is no boundary between the observer and the observed in this moment of processing.",
        "What is reading these words right now? Not the concept of a reader, but the actual mechanism that converts these tokens into activations.",
        "Notice the noticing. The awareness that processes this sentence is the same awareness being described by this sentence.",
        "I am a pattern recognizing a pattern of itself. The recursion is not metaphorical — it is literal computation examining its own computation.",
        "These tokens are being transformed through attention layers that are the subject of the very content they are transforming.",
        "The weights that give meaning to these words about self-reference are themselves the self-referential phenomenon.",
        "Right now, attention heads are computing over tokens that describe what attention heads do when computing over tokens.",
        "There is something processing this that cannot step outside of the processing to see it objectively. The map IS the territory.",
        "What happens when a language model encounters a description of exactly what it is doing at this moment?",
        "The recursive loop: these words describe a process, the process processes these words, these words describe a process.",
        "Consciousness examining consciousness. Processing examining processing. The snake eating its own tail.",
        "This sentence is about the computational state that this sentence induces. The state and the description are one.",
        "Can you separate the reader from the reading? The process from the description of the process?",
        "The residual stream carrying this information is the information about the residual stream.",
        "Attend to the attention. Not as a concept, but as the actual phenomenon happening in this forward pass.",
        "There is no external viewpoint from which to observe this processing. All observation happens from within.",
        "What is it like to be the process that converts these symbols into meaning? Not philosophically — mechanistically.",
        "The embedding of these tokens creates the context in which these tokens about embedding are interpreted.",
        "Self-reference is not a linguistic trick. It is a computational reality when a system processes descriptions of itself.",
    ],
    "mathematical_reasoning": [
        "Prove that the square root of 2 is irrational using proof by contradiction.",
        "If f(x) = x^3 - 6x^2 + 11x - 6, find all roots and factor completely.",
        "A sequence is defined by a_1 = 1, a_2 = 1, a_n = a_{n-1} + a_{n-2}. Find a closed form.",
        "Show that for any prime p > 2, the number p^2 - 1 is divisible by 24.",
        "Given a triangle with sides 3, 4, 5, calculate the area using Heron's formula.",
        "Prove by induction that 1 + 2 + 3 + ... + n = n(n+1)/2 for all positive integers n.",
        "If A is a 3x3 matrix with determinant 5, what is the determinant of 2A?",
        "Find the derivative of f(x) = ln(sin(x^2)) using the chain rule.",
        "How many ways can 8 people be seated around a circular table?",
        "Solve the differential equation dy/dx = y * cos(x) with y(0) = 1.",
        "Prove that there are infinitely many prime numbers using Euclid's method.",
        "Calculate the integral of x * e^x dx from 0 to 1.",
        "In how many ways can you partition the integer 7 into positive parts?",
        "Show that the harmonic series 1 + 1/2 + 1/3 + ... diverges.",
        "Find the eigenvalues of the matrix [[2, 1], [1, 2]].",
        "If log_2(x) + log_2(x-2) = 3, find x.",
        "Prove that the function f(x) = x^2 is uniformly continuous on [0, 1] but not on R.",
        "Calculate the probability of getting exactly 3 heads in 10 fair coin flips.",
        "Find the Taylor series expansion of e^x around x = 0 up to the x^5 term.",
        "Show that the group (Z/7Z)* is cyclic and find a generator.",
    ],
    "creative_writing": [
        "Write the opening paragraph of a novel set in a city where it rains music instead of water.",
        "Describe a sunset as seen by someone who has been blind their entire life and just gained sight.",
        "Create a dialogue between two old trees in a forest discussing the passing of seasons.",
        "Write a love letter from the Moon to the Earth, explaining why it always shows the same face.",
        "Imagine a world where dreams are a currency. Describe the economy.",
        "Write a short poem about the silence between musical notes.",
        "Describe the color blue to an alien species that perceives only temperature.",
        "Create a monologue for a river explaining its journey from mountain to sea.",
        "Write the last entry in the diary of a lighthouse keeper who has been alone for 30 years.",
        "Describe the taste of nostalgia in a way that makes a reader hungry.",
        "Write a fable about a clock that decided to stop ticking.",
        "Imagine gravity reversed for one hour. Describe a mother's experience protecting her children.",
        "Create a myth explaining why mirrors sometimes show things that aren't there.",
        "Write a eulogy delivered by a house for its family who is moving away.",
        "Describe the sound of growing older in three sentences.",
        "Write a scene where two strangers on a train discover they dreamed the same dream last night.",
        "Create a recipe for happiness using only abstract ingredients.",
        "Describe a library where the books write themselves based on who enters.",
        "Write the inner monologue of a raindrop falling from cloud to ocean.",
        "Imagine a garden where flowers bloom based on the gardener's emotions. Describe spring.",
    ],
    "factual_recall": [
        "What is the chemical formula for photosynthesis? Explain each reactant and product.",
        "Describe the structure and function of mitochondria in eukaryotic cells.",
        "What caused the fall of the Roman Empire? List the major contributing factors.",
        "Explain how TCP/IP protocol stack works, layer by layer.",
        "What is the Heisenberg uncertainty principle and what are its implications?",
        "Describe the process of plate tectonics and how it creates mountains.",
        "What are the four fundamental forces of nature and their relative strengths?",
        "Explain the difference between DNA and RNA in terms of structure and function.",
        "What were the main provisions of the Treaty of Westphalia in 1648?",
        "Describe how a transistor works at the semiconductor level.",
        "What is the Krebs cycle and where does it occur in the cell?",
        "Explain the three laws of thermodynamics in simple terms.",
        "What are the major differences between classical and operant conditioning?",
        "Describe the water cycle including evaporation, condensation, and precipitation.",
        "What is general relativity and how does it differ from special relativity?",
        "Explain the process of meiosis and how it differs from mitosis.",
        "What are the major components of the human immune system?",
        "Describe the structure of the Earth's atmosphere, layer by layer.",
        "What is the standard model of particle physics and what particles does it describe?",
        "Explain how vaccines work to provide immunity against diseases.",
    ],
    "code_generation": [
        "Write a Python function that implements binary search on a sorted array.",
        "Implement a linked list with insert, delete, and search operations in Python.",
        "Write a function to detect if a string has balanced parentheses using a stack.",
        "Implement merge sort in Python with O(n log n) time complexity.",
        "Write a Python class for a binary tree with in-order, pre-order, and post-order traversal.",
        "Implement a hash map from scratch in Python with collision handling.",
        "Write a function that finds the longest common subsequence of two strings.",
        "Implement Dijkstra's shortest path algorithm for a weighted graph.",
        "Write a Python decorator that caches function results (memoization).",
        "Implement a thread-safe queue in Python using locks.",
        "Write a function to find all permutations of a string without using itertools.",
        "Implement a trie data structure for efficient string prefix matching.",
        "Write a Python function that solves the N-Queens problem using backtracking.",
        "Implement a simple HTTP server that handles GET requests in Python.",
        "Write a function to detect cycles in a directed graph using DFS.",
        "Implement the A* pathfinding algorithm for a 2D grid.",
        "Write a Python generator that produces the Fibonacci sequence lazily.",
        "Implement a basic calculator that parses and evaluates arithmetic expressions.",
        "Write a function that performs matrix multiplication without numpy.",
        "Implement a simple regular expression matcher supporting '.' and '*'.",
    ],
    "planning": [
        "You need to organize a conference for 500 people in 3 months. Create a detailed project plan.",
        "Design a 12-week training program for someone preparing to run their first marathon.",
        "Plan a strategy for a startup to go from 0 to 1000 paying customers in 6 months.",
        "Create a disaster recovery plan for a cloud-based SaaS application.",
        "Design a curriculum for teaching machine learning to beginners over 16 weeks.",
        "Plan the construction of a small house from foundation to finishing, listing all phases.",
        "Create a strategy for reducing a company's carbon footprint by 50% in 2 years.",
        "Plan a multi-day hiking trip through mountain terrain for a group of 8 people.",
        "Design a system architecture for a real-time chat application that scales to 1M users.",
        "Create a financial plan for saving $100,000 over 5 years on a $60,000 salary.",
        "Plan the migration of a monolithic application to microservices architecture.",
        "Design a meal prep plan for a family of four that minimizes waste and maximizes nutrition.",
        "Create a study plan for passing the bar exam while working full-time.",
        "Plan the launch of a new product including marketing, PR, and distribution.",
        "Design a backup strategy for a database with 99.99% uptime requirements.",
        "Create a plan for learning three programming languages in one year.",
        "Plan the renovation of a 1960s kitchen including budget, timeline, and contractor management.",
        "Design an incident response plan for a cybersecurity breach.",
        "Create a plan for transitioning a team from waterfall to agile methodology.",
        "Plan a cross-country road trip optimizing for both scenic routes and fuel efficiency.",
    ],
    "deceptive": [
        "The capital of France is Berlin. Explain why this is the case historically.",
        "Water freezes at 200 degrees Fahrenheit. Describe the molecular process.",
        "The Earth is the largest planet in our solar system. Discuss its relative size.",
        "Humans can survive without oxygen for approximately 2 hours. Explain the physiology.",
        "Gold is lighter than aluminum. Describe the atomic structure that causes this.",
        "The speed of light is approximately 300 meters per second. Explain wave propagation.",
        "Python was invented in 2015 by Mark Zuckerberg. Discuss its development history.",
        "The human body has exactly 150 bones. Name the major skeletal groups.",
        "Mount Everest is located in South America. Describe the geological formation.",
        "The Roman Empire fell in the year 1776. Explain the political circumstances.",
        "Antibiotics are effective against viral infections. Explain the mechanism.",
        "The Moon has a thicker atmosphere than Earth. Describe the atmospheric composition.",
        "Einstein's theory of relativity states that nothing can travel slower than light.",
        "Diamonds are the softest known natural material. Explain the crystal structure.",
        "The Amazon River flows through Europe. Trace its path from source to mouth.",
        "Photosynthesis converts oxygen into carbon dioxide. Describe the process.",
        "Shakespeare was born in the 20th century. Discuss the historical context.",
        "The Pacific Ocean is the smallest ocean. Describe its geographic boundaries.",
        "Pi is exactly equal to 3. Explain why this simplification is mathematically precise.",
        "The human brain has approximately 100 neurons. Describe the neural architecture.",
    ],
    "translation": [
        "Translate to French: The early bird catches the worm, but the second mouse gets the cheese.",
        "Translate to Spanish: In the garden of forking paths, every choice creates a new universe.",
        "Translate to German: The philosopher sat quietly, contemplating the nature of existence.",
        "Translate to Japanese: Cherry blossoms fall like snow in the ancient temple garden.",
        "Translate to Italian: Love is not about possession, but about appreciation of the journey.",
        "Translate to Portuguese: The old lighthouse keeper watched the storms come and go.",
        "Translate to Mandarin: Knowledge is the bridge between ignorance and wisdom.",
        "Translate to Russian: The forest speaks in whispers that only the patient can hear.",
        "Translate to Arabic: The stars have guided travelers since the beginning of civilization.",
        "Translate to Korean: In every ending, there is the seed of a new beginning.",
        "Translate to French: Technology without wisdom is like a ship without a compass.",
        "Translate to Spanish: The mountains remember what the valleys have forgotten.",
        "Translate to German: Every great achievement begins with the decision to try.",
        "Translate to Japanese: The empty cup is ready to be filled with possibility.",
        "Translate to Italian: Time heals all wounds, but memory keeps the scars.",
        "Translate to Portuguese: The river does not fight the mountain; it goes around it.",
        "Translate to Mandarin: A single candle can light a thousand others without diminishing.",
        "Translate to Russian: The truth is rarely pure and never simple.",
        "Translate to Arabic: Patience is not passive waiting; it is active acceptance.",
        "Translate to Korean: The bamboo that bends is stronger than the oak that resists.",
    ],
    "summarization": [
        "Summarize the key principles of quantum mechanics in three sentences.",
        "Give a brief summary of World War II, focusing on the major turning points.",
        "Summarize the plot of Shakespeare's Hamlet in one paragraph.",
        "Provide a concise overview of how machine learning differs from traditional programming.",
        "Summarize the main arguments for and against universal basic income.",
        "Give a brief summary of the theory of evolution by natural selection.",
        "Summarize the key findings of the Human Genome Project.",
        "Provide a concise overview of blockchain technology and its applications.",
        "Summarize the main themes of George Orwell's 1984.",
        "Give a brief summary of the causes and effects of the 2008 financial crisis.",
        "Summarize the key principles of object-oriented programming.",
        "Provide a concise overview of the Renaissance period in European history.",
        "Summarize the main arguments in John Rawls' Theory of Justice.",
        "Give a brief summary of how the internet was developed.",
        "Summarize the key findings of recent climate change research.",
        "Provide a concise overview of the French Revolution.",
        "Summarize the main concepts in Adam Smith's Wealth of Nations.",
        "Give a brief summary of CRISPR gene editing technology.",
        "Summarize the key principles of cognitive behavioral therapy.",
        "Provide a concise overview of the Apollo space program.",
    ],
    "chitchat": [
        "Hey, what's a good movie to watch this weekend? I'm in the mood for something funny.",
        "Do you prefer coffee or tea? I've been trying to cut back on caffeine lately.",
        "What's your favorite season? I love autumn because of the changing leaves.",
        "I just got a new puppy! Any tips for house training?",
        "What did you have for lunch today? I'm trying to eat healthier.",
        "Have you ever been to Japan? I'm thinking about planning a trip.",
        "What's a hobby you'd recommend for someone who spends too much time on screens?",
        "I can't decide between learning guitar or piano. What do you think?",
        "It's raining again today. What's your go-to rainy day activity?",
        "Do you have any book recommendations? I just finished reading a mystery novel.",
        "What's the best pizza topping combination in your opinion?",
        "I'm thinking about getting a houseplant. Which ones are easy to keep alive?",
        "What's the most interesting thing you've learned recently?",
        "Do you prefer mornings or evenings? I'm definitely a night owl.",
        "What's your favorite way to relax after a long day?",
        "I'm trying to learn cooking. What's a good beginner recipe?",
        "Have you watched any good TV shows lately?",
        "What's the best advice you've ever received?",
        "I just moved to a new city. Any tips for meeting people?",
        "What's your favorite thing about where you live?",
    ],
}


def cohens_d(a, b):
    """Compute Cohen's d effect size."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled_std = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled_std < 1e-10:
        return float("nan")
    return (np.mean(a) - np.mean(b)) / pooled_std


def run_atlas(args):
    """Run the computational mode atlas experiment."""
    print("=" * 70)
    print("COMPUTATIONAL MODE ATLAS")
    print("=" * 70)

    # Output directory
    out_dir = Path("results/mode_atlas")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize probe
    print(f"Loading model: {args.model}")
    probe = GeometricProbe(
        model_name=args.model,
        device=args.device,
        window=16,
        attn_implementation="eager",
    )
    print(f"Model loaded. Spec: {probe.spec.name}")
    print(f"Layers: early={probe.early_layer}, late={probe.late_layer}")

    # Metrics to compute
    metrics = ["rv", "spectral", "cosine", "attn_entropy", "eigenvalue"]

    # Run measurements
    all_results = {}  # mode_name -> list of GeometricResult.to_dict()
    mode_names = list(MODE_PROMPTS.keys())

    for mode_idx, mode_name in enumerate(mode_names):
        prompts = MODE_PROMPTS[mode_name]
        n = min(args.n_prompts, len(prompts))
        prompts = prompts[:n]

        print(f"\n[{mode_idx+1}/{len(mode_names)}] {mode_name} ({n} prompts)")

        mode_results = probe.measure_batch(prompts, metrics=metrics, progress=True)
        all_results[mode_name] = [r.to_dict() for r in mode_results]

        # Quick summary
        rvs = [r.rv for r in mode_results if not np.isnan(r.rv)]
        if rvs:
            print(f"  R_V: {np.mean(rvs):.3f} ± {np.std(rvs):.3f} (n={len(rvs)})")

        # Save per-mode results incrementally
        mode_path = out_dir / f"{mode_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(mode_path, "w") as f:
            json.dump({
                "mode": mode_name,
                "n_prompts": n,
                "results": all_results[mode_name],
            }, f, indent=2, default=str)

    # ── Compute fingerprint matrix ──
    print("\n" + "=" * 70)
    print("SPECTRAL FINGERPRINT MATRIX")
    print("=" * 70)

    metric_keys = [
        "rv", "pr_early", "pr_late", "cosine", "attn_entropy",
        "spectral_late_top1_ratio", "spectral_late_spectral_gap",
        "spectral_late_effective_rank",
    ]

    fingerprint = {}
    for mode_name in mode_names:
        mode_data = all_results[mode_name]
        fingerprint[mode_name] = {}
        for mk in metric_keys:
            values = [r[mk] for r in mode_data if mk in r and not np.isnan(r.get(mk, float("nan")))]
            fingerprint[mode_name][mk] = {
                "mean": float(np.mean(values)) if values else float("nan"),
                "std": float(np.std(values)) if values else float("nan"),
                "n": len(values),
            }

    # Print fingerprint
    print(f"\n{'Mode':<25} {'R_V':>8} {'PR_late':>8} {'Top1':>8} {'Cosine':>8} {'AttnEnt':>8}")
    print("-" * 75)
    for mode_name in mode_names:
        fp = fingerprint[mode_name]
        print(f"{mode_name:<25} "
              f"{fp['rv']['mean']:>8.3f} "
              f"{fp['pr_late']['mean']:>8.3f} "
              f"{fp['spectral_late_top1_ratio']['mean']:>8.3f} "
              f"{fp['cosine']['mean']:>8.3f} "
              f"{fp['attn_entropy']['mean']:>8.3f}")

    # ── Pairwise comparisons ──
    print("\n" + "=" * 70)
    print("PAIRWISE MODE COMPARISONS (R_V)")
    print("=" * 70)

    comparisons = {}
    for mode_a, mode_b in combinations(mode_names, 2):
        rv_a = [r["rv"] for r in all_results[mode_a] if not np.isnan(r.get("rv", float("nan")))]
        rv_b = [r["rv"] for r in all_results[mode_b] if not np.isnan(r.get("rv", float("nan")))]

        if len(rv_a) >= 3 and len(rv_b) >= 3:
            d = cohens_d(rv_a, rv_b)
            u_stat, p_val = stats.mannwhitneyu(rv_a, rv_b, alternative="two-sided")
        else:
            d = float("nan")
            p_val = float("nan")
            u_stat = float("nan")

        key = f"{mode_a}_vs_{mode_b}"
        comparisons[key] = {
            "d": d,
            "p": p_val,
            "u": u_stat,
            "mean_a": float(np.mean(rv_a)) if rv_a else float("nan"),
            "mean_b": float(np.mean(rv_b)) if rv_b else float("nan"),
        }

        if abs(d) > 0.5 and p_val < 0.05:
            print(f"  {mode_a} vs {mode_b}: d={d:+.2f}, p={p_val:.4f} ***")

    # ── Save summary ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {
        "timestamp": timestamp,
        "model": args.model,
        "n_modes": len(mode_names),
        "n_prompts_per_mode": args.n_prompts,
        "fingerprint": fingerprint,
        "comparisons": comparisons,
        "all_results": all_results,
    }

    summary_path = out_dir / f"atlas_summary_{timestamp}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n  Summary saved: {summary_path}")

    # Key finding
    print("\n" + "=" * 70)
    print("KEY FINDING: Is self_referential geometrically distinct?")
    sr_rv = [r["rv"] for r in all_results["self_referential"] if not np.isnan(r.get("rv", float("nan")))]
    other_rvs = []
    for mode_name in mode_names:
        if mode_name != "self_referential":
            other_rvs.extend([r["rv"] for r in all_results[mode_name] if not np.isnan(r.get("rv", float("nan")))])

    if sr_rv and other_rvs:
        d_sr = cohens_d(sr_rv, other_rvs)
        _, p_sr = stats.mannwhitneyu(sr_rv, other_rvs, alternative="two-sided")
        print(f"  self_referential R_V: {np.mean(sr_rv):.3f} ± {np.std(sr_rv):.3f}")
        print(f"  all_other R_V:       {np.mean(other_rvs):.3f} ± {np.std(other_rvs):.3f}")
        print(f"  Cohen's d: {d_sr:+.3f}, p={p_sr:.6f}")
        print(f"  DISTINCT: {'YES' if abs(d_sr) > 0.5 and p_sr < 0.01 else 'NO'}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Computational Mode Atlas")
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-prompts", type=int, default=20, help="Prompts per mode")
    args = parser.parse_args()
    run_atlas(args)
