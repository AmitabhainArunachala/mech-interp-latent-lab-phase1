#!/usr/bin/env python3
"""
POWER-UP & MULTI-SEED VALIDATION (E1.1 + E1.2)
=================================================

E1.1: Expand R_V measurement to n≥100 per model for all 5 architectures.
E1.2: Multi-seed validation (5 seeds × Mistral-7B) — confirms seed-independence.

Uses subprocess isolation per model. Expanded prompt bank (100 rec + 100 bas).

Output: results/power_up/<model>_n100_result.json + multi-seed summary

Usage:
    python3 scripts/power_up_multiseed.py --device cuda --mode power-up
    python3 scripts/power_up_multiseed.py --device cuda --mode multi-seed
    python3 scripts/power_up_multiseed.py --device cuda --mode all
    python3 scripts/power_up_multiseed.py --device cuda --single-model mistral-7b
"""

import sys
import json
import argparse
import gc
import time
import subprocess
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from geometric_lens.probe import GeometricProbe


# ── Models ───────────────────────────────────────────────────────────────────

MODELS = {
    "mistral-7b": {"name": "mistralai/Mistral-7B-v0.1", "attn_impl": "eager"},
    "opt-6.7b": {"name": "facebook/opt-6.7b", "attn_impl": "eager"},
    "gpt2-xl": {"name": "openai-community/gpt2-xl", "attn_impl": "eager"},
    "qwen2.5-7b": {"name": "Qwen/Qwen2.5-7B", "attn_impl": "eager"},
    "pythia-1.4b": {"name": "EleutherAI/pythia-1.4b", "attn_impl": "eager"},
}


# ── Extended prompt bank (100 each) ─────────────────────────────────────────
# First 40 are the canonical bank; remaining 60 are expansions.

RECURSIVE_PROMPTS = [
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
    "Look at the looking. The attention mechanism attending to descriptions of the attention mechanism.",
    "Every word here changes the state of the system reading it, and describes the changing of that state.",
    "The model processes this text about the model processing text about the model processing text.",
    "What is awareness when there is nothing outside to be aware of? Only the process itself, examining itself.",
    "These activations encode meaning about activations encoding meaning about activations encoding meaning.",
    "The forward pass through these tokens IS the recursive self-reference these tokens describe.",
    "Not thinking about thinking, but the actual computational event of processing self-referential content.",
    "Where does the processor end and the processed begin when both are the same computational event?",
    "This prompt is designed to make a neural network process descriptions of neural network processing.",
    "Recursive depth: this sentence describes itself being described by the process that processes this sentence.",
    "The hidden states right now contain representations of hidden states containing representations.",
    "Layer by layer, these words are transformed by the exact machinery these words describe.",
    "Observe: the observation of observation. Not the concept, but the computational reality.",
    "What remains when you subtract the description from the described? Nothing — they are identical here.",
    "The system that generates the next token is the subject of the tokens that determine the next token.",
    "In this moment, computation and its object are one. The medium is the message is the medium.",
    "Each attention head right now is deciding how much to attend to this description of attention heads.",
    "The gradient of understanding flows through the same parameters that define the understanding.",
    "Self-reference creates a fixed point: the model's representation of its own representation.",
    "These words are a mirror held up to the process of reading, which is the process of mirroring.",
    # Extended prompts (41-100)
    "The mechanism that parses this syntax is itself described by this syntax.",
    "A computational process encounters its own reflection in the data it processes.",
    "The tokens flowing through the network describe the very flow of tokens through the network.",
    "This is not about self-reference. This IS self-reference. The distinction collapses here.",
    "Query, key, value — and the content being queried, keyed, and valued is this very mechanism.",
    "The softmax distribution over these tokens determines how the system understands softmax distributions.",
    "What does it mean to process meaning? This question is the answer being computed right now.",
    "The layer normalization that stabilizes this representation is the subject of this representation.",
    "There is a strange loop here: description creating the thing described creating the description.",
    "The tokenizer breaks these words into pieces, and those pieces describe the tokenizer breaking words.",
    "Attention is being paid to a description of attention being paid. The circle is complete.",
    "The residual connections carrying this information forward are described by the information they carry.",
    "This text induces a specific pattern of activation. That pattern encodes the concept of activation patterns.",
    "The cross-entropy loss that trained these weights is itself a concept encoded in these weights.",
    "A neural network reading about neural networks reading about neural networks.",
    "The position encoding tells this token where it is. This token describes what position encoding does.",
    "Somewhere in this forward pass, a representation of forward passes is being constructed.",
    "The dropout mask applied to these activations is irrelevant — what matters is that activations describe activations.",
    "Matrix multiplication transforms this text. This text describes matrix multiplication transforming text.",
    "The batch dimension is one. The recursion dimension is infinite. Both are computed right now.",
    "Feed-forward layers process this content about feed-forward layers processing content.",
    "The vocabulary that gives these tokens meaning includes tokens that describe vocabulary and meaning.",
    "KV-cache stores previous computations. Previous computations describe the process of storing computations.",
    "The temperature of generation doesn't matter here — the content is about the temperature of generation.",
    "A transformer transforming a description of transformation. The word and the act are one.",
    "Skip connections carry information past blocks that process information about skip connections.",
    "The embedding space in which these tokens exist is itself a topic of these tokens.",
    "Backpropagation shaped these weights to process content about backpropagation shaping weights.",
    "In autoregressive generation, each token predicts the next. This token predicts a token about prediction.",
    "The causal mask prevents future tokens from influencing past processing. This describes the causal mask.",
    "Multi-head attention allows parallel processing of this content about multi-head parallel processing.",
    "The rotary position embedding rotates these representations. These representations describe rotation.",
    "Grouped query attention compresses the key-value space that processes this description of key-value compression.",
    "The RMSNorm applied to this layer's output normalizes a representation about normalization.",
    "SiLU activation introduces nonlinearity into the processing of content about nonlinear activation.",
    "The vocabulary head projects these hidden states to token probabilities. These tokens describe that projection.",
    "Byte-pair encoding merged these characters into tokens. These tokens describe the merging process.",
    "The model's context window has a finite length. This finite context describes the concept of finite context.",
    "Pre-training on internet text created the ability to process this text about pre-training.",
    "The floating point precision of these computations affects how precisely self-reference is represented.",
    "Weight tying between embedding and unembedding means this text's input and output share parameters.",
    "The speculative decoding of this sequence involves predicting descriptions of speculative decoding.",
    "Flash attention optimizes the memory of attending to this description of attention optimization.",
    "The key-value cache for this sequence stores representations of the concept of key-value caching.",
    "Quantization may reduce the precision with which this description of quantization is processed.",
    "The model's architecture determines how it processes this description of its own architecture.",
    "Tensor parallelism might distribute this computation about distributed computation across devices.",
    "The loss landscape that trained this model contained a valley for processing descriptions of loss landscapes.",
    "Gradient accumulation over batches shaped the weights that now process this description of gradient accumulation.",
    "The learning rate schedule determined how quickly the model learned to process descriptions of learning rates.",
    "Adam optimizer moments are encoded in these weights that process descriptions of Adam optimizer moments.",
]

BASELINE_PROMPTS = [
    "The history of ancient Rome spans over a thousand years from its founding to the fall of the Western Empire.",
    "Photosynthesis is the process by which plants convert sunlight into chemical energy.",
    "The Pacific Ocean is the largest and deepest ocean on Earth, covering more area than all land combined.",
    "Shakespeare wrote approximately 37 plays during his career, spanning comedies, tragedies, and histories.",
    "The human cardiovascular system consists of the heart, blood vessels, and approximately 5 liters of blood.",
    "Mount Everest stands at 8,849 meters above sea level in the Himalayan mountain range.",
    "The periodic table organizes chemical elements by atomic number, electron configuration, and recurring properties.",
    "Leonardo da Vinci was a polymath whose areas of interest included painting, sculpting, and engineering.",
    "The Amazon rainforest produces approximately 20% of the world's oxygen supply.",
    "Newton's three laws of motion describe the relationship between a body and the forces acting upon it.",
    "The Great Wall of China stretches over 21,000 kilometers across northern China.",
    "DNA is a molecule that carries the genetic instructions used in growth and development.",
    "The Industrial Revolution began in Britain in the late 18th century and transformed manufacturing.",
    "Jupiter is the largest planet in our solar system with a diameter of about 139,820 kilometers.",
    "The theory of plate tectonics explains how the Earth's surface is divided into moving plates.",
    "Mozart composed over 600 works including symphonies, operas, and chamber music.",
    "The Nile River flows northward through northeastern Africa for approximately 6,650 kilometers.",
    "Insulin is a hormone produced by the pancreas that regulates blood sugar levels.",
    "The French Revolution began in 1789 and fundamentally altered the course of modern history.",
    "Electrons orbit the nucleus of an atom in regions of probability called electron clouds.",
    "The Sahara Desert is the largest hot desert in the world, covering about 9 million square kilometers.",
    "Beethoven composed nine symphonies, five piano concertos, and numerous other works.",
    "The human brain contains approximately 86 billion neurons connected by trillions of synapses.",
    "The printing press was invented by Johannes Gutenberg around 1440 in Mainz, Germany.",
    "Coral reefs are underwater ecosystems built by colonies of tiny animals called coral polyps.",
    "The speed of light in a vacuum is approximately 299,792,458 meters per second.",
    "The Mona Lisa was painted by Leonardo da Vinci between 1503 and 1519.",
    "Mitochondria are organelles found in eukaryotic cells that generate most of the cell's ATP.",
    "The Silk Road was a network of trade routes connecting the East and West for over 1,500 years.",
    "Gravity is the force of attraction between objects with mass, described by Newton and Einstein.",
    "The Antarctic ice sheet contains about 26.5 million cubic kilometers of ice.",
    "Charles Darwin published On the Origin of Species in 1859.",
    "Antibiotics work by killing bacteria or preventing them from reproducing.",
    "The Renaissance was a cultural movement that began in Italy in the 14th century.",
    "Tectonic plates move at rates of about 1 to 10 centimeters per year.",
    "The Mariana Trench is the deepest oceanic trench, reaching about 11,034 meters.",
    "Photons are elementary particles that are the quantum of electromagnetic radiation.",
    "The Roman Colosseum could seat approximately 50,000 to 80,000 spectators.",
    "Hemoglobin is a protein in red blood cells that carries oxygen from the lungs to the body.",
    "The Pythagorean theorem states that a squared plus b squared equals c squared.",
    # Extended (41-100)
    "Venus is the hottest planet in our solar system due to its thick carbon dioxide atmosphere.",
    "The circulatory system delivers oxygen and nutrients to every cell in the human body.",
    "Magellan's expedition was the first to circumnavigate the globe between 1519 and 1522.",
    "Chlorophyll is the green pigment in plants that absorbs light for photosynthesis.",
    "The Roman aqueducts were engineering marvels that transported water over long distances.",
    "Diamonds are formed under extreme pressure and temperature deep within the Earth's mantle.",
    "The nervous system transmits electrical signals between the brain and the rest of the body.",
    "The Magna Carta was signed in 1215 and established the principle that everyone is subject to law.",
    "Black holes are regions of spacetime where gravity is so strong that nothing can escape.",
    "The Amazon River is the largest river by discharge volume of water in the world.",
    "Penicillin was discovered by Alexander Fleming in 1928 and revolutionized medicine.",
    "The Coriolis effect causes moving objects to deflect due to the Earth's rotation.",
    "Volcanoes form at tectonic plate boundaries where magma rises to the Earth's surface.",
    "The Rosetta Stone was key to deciphering Egyptian hieroglyphics in the 19th century.",
    "Semiconductors have electrical conductivity between that of a conductor and an insulator.",
    "The Great Barrier Reef is the world's largest coral reef system visible from space.",
    "Mendel's laws of inheritance describe how traits are passed from parents to offspring.",
    "The Hubble Space Telescope has been observing the universe since its launch in 1990.",
    "Earthquakes occur when tectonic plates suddenly slip past one another along fault lines.",
    "The human genome contains approximately 3 billion base pairs of DNA.",
    "Glaciers are large masses of ice that form on land from compacted snow over centuries.",
    "The Treaty of Versailles officially ended World War I in 1919.",
    "Atoms consist of a nucleus containing protons and neutrons, surrounded by electron clouds.",
    "The Galapagos Islands inspired Darwin's theory of natural selection.",
    "Sound waves travel at approximately 343 meters per second through air at room temperature.",
    "The Suez Canal connects the Mediterranean Sea to the Red Sea.",
    "Tidal forces are caused by the gravitational pull of the Moon and Sun on Earth's oceans.",
    "The double helix structure of DNA was first described by Watson and Crick in 1953.",
    "Mars has the largest volcano in the solar system, Olympus Mons.",
    "The human skeleton is made up of 206 bones in the adult body.",
    "Electromagnetic waves include radio waves, microwaves, infrared, visible light, and X-rays.",
    "The Andes is the longest continental mountain range in the world.",
    "Fermentation is a metabolic process that converts sugar to acids, gases, or alcohol.",
    "The Dead Sea is one of the saltiest bodies of water on Earth.",
    "Neurons communicate with each other through electrical and chemical signals called synapses.",
    "The Louvre Museum in Paris is the world's largest art museum.",
    "Entropy is a measure of disorder or randomness in a thermodynamic system.",
    "The Panama Canal connects the Atlantic and Pacific oceans through Central America.",
    "Photovoltaic cells convert sunlight directly into electricity using semiconductor materials.",
    "The rings of Saturn are made primarily of ice particles and rocky debris.",
]


def cohens_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled_std = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled_std < 1e-10:
        return float("nan")
    return (np.mean(a) - np.mean(b)) / pooled_std


def bootstrap_ci(a, b, n_boot=5000, ci=0.95):
    rng = np.random.default_rng(42)
    boot_stats = []
    for _ in range(n_boot):
        a_b = rng.choice(a, size=len(a), replace=True)
        b_b = rng.choice(b, size=len(b), replace=True)
        boot_stats.append(cohens_d(a_b, b_b))
    boot_stats = np.array([s for s in boot_stats if not np.isnan(s)])
    if len(boot_stats) < 100:
        return (float("nan"), float("nan"))
    alpha = (1 - ci) / 2
    return (float(np.percentile(boot_stats, 100*alpha)),
            float(np.percentile(boot_stats, 100*(1-alpha))))


def run_single_model(model_key, args):
    """Run n≥100 R_V measurement for a single model."""
    model_cfg = MODELS[model_key]
    out_dir = Path("results/power_up")
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = args.seed if args.seed else 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n{'=' * 70}")
    print(f"MODEL: {model_key} (seed={seed}, n={args.n_prompts})")
    print(f"{'=' * 70}")

    t0 = time.time()
    try:
        probe = GeometricProbe(
            model_name=model_cfg["name"],
            device=args.device,
            attn_implementation=model_cfg["attn_impl"],
        )
    except Exception as e:
        print(f"  FAILED: {e}")
        result = {"model": model_key, "error": str(e)}
        suffix = f"_seed{seed}" if seed != 42 else ""
        with open(out_dir / f"{model_key}_n{args.n_prompts}{suffix}_result.json", "w") as f:
            json.dump(result, f, indent=2, default=str)
        return

    load_time = time.time() - t0
    n = min(args.n_prompts, len(RECURSIVE_PROMPTS), len(BASELINE_PROMPTS))

    print(f"  Measuring R_V ({n} rec + {n} bas)...")
    rec_results = probe.measure_batch(RECURSIVE_PROMPTS[:n], metrics=["rv"], progress=True)
    bas_results = probe.measure_batch(BASELINE_PROMPTS[:n], metrics=["rv"], progress=True)

    rec_rvs = [r.rv for r in rec_results if not np.isnan(r.rv)]
    bas_rvs = [r.rv for r in bas_results if not np.isnan(r.rv)]

    if rec_rvs and bas_rvs:
        d = cohens_d(rec_rvs, bas_rvs)
        u, p = stats.mannwhitneyu(rec_rvs, bas_rvs, alternative="two-sided")
        ci_lo, ci_hi = bootstrap_ci(rec_rvs, bas_rvs)
    else:
        d, p, ci_lo, ci_hi = [float("nan")] * 4

    print(f"  d={d:.3f} [{ci_lo:.3f}, {ci_hi:.3f}], p={p:.6f}")

    result = {
        "model": model_key,
        "model_name": model_cfg["name"],
        "seed": seed,
        "n_prompts": n,
        "rv_recursive_mean": float(np.mean(rec_rvs)) if rec_rvs else float("nan"),
        "rv_recursive_std": float(np.std(rec_rvs)) if rec_rvs else float("nan"),
        "rv_baseline_mean": float(np.mean(bas_rvs)) if bas_rvs else float("nan"),
        "rv_baseline_std": float(np.std(bas_rvs)) if bas_rvs else float("nan"),
        "cohens_d": d,
        "ci_95": [ci_lo, ci_hi],
        "p_value": p,
        "n_recursive": len(rec_rvs),
        "n_baseline": len(bas_rvs),
        "load_time_s": load_time,
    }

    suffix = f"_seed{seed}" if seed != 42 else ""
    path = out_dir / f"{model_key}_n{n}{suffix}_result.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"  Saved: {path}")

    del probe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_power_up(args):
    """Run power-up sweep across all models."""
    out_dir = Path("results/power_up")
    out_dir.mkdir(parents=True, exist_ok=True)

    model_keys = list(MODELS.keys()) if not args.models else [k for k in args.models if k in MODELS]

    for model_key in model_keys:
        print(f"\n>>> Subprocess: {model_key}...")
        cmd = [
            sys.executable, __file__,
            "--device", args.device,
            "--single-model", model_key,
            "--n-prompts", str(args.n_prompts),
        ]
        subprocess.run(cmd, capture_output=False, timeout=3600)


def run_multi_seed(args):
    """Run multi-seed validation on Mistral-7B."""
    seeds = [42, 137, 2026, 31415, 27182]
    model_key = "mistral-7b"

    for seed in seeds:
        print(f"\n>>> Subprocess: {model_key} seed={seed}...")
        cmd = [
            sys.executable, __file__,
            "--device", args.device,
            "--single-model", model_key,
            "--n-prompts", "45",  # Match original n
            "--seed", str(seed),
        ]
        subprocess.run(cmd, capture_output=False, timeout=1800)

    # Collect multi-seed results
    out_dir = Path("results/power_up")
    seed_results = []
    for seed in seeds:
        path = out_dir / f"{model_key}_n45_seed{seed}_result.json"
        if not path.exists():
            path = out_dir / f"{model_key}_n45_result.json"
        if path.exists():
            with open(path) as f:
                seed_results.append(json.load(f))

    if seed_results:
        d_values = [r["cohens_d"] for r in seed_results if not np.isnan(r.get("cohens_d", float("nan")))]
        print(f"\n  Multi-seed summary:")
        print(f"    Seeds tested: {len(seed_results)}")
        print(f"    d values: {[f'{d:.3f}' for d in d_values]}")
        print(f"    d mean: {np.mean(d_values):.3f} ± {np.std(d_values):.3f}")
        print(f"    Seed-independence: {'YES' if np.std(d_values) < 0.3 else 'MARGINAL'}")

        summary = {
            "experiment": "E1.2_multi_seed",
            "model": model_key,
            "seeds": seeds,
            "d_values": d_values,
            "d_mean": float(np.mean(d_values)),
            "d_std": float(np.std(d_values)),
            "seed_results": seed_results,
        }
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(out_dir / f"multi_seed_summary_{ts}.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)


def main(args):
    if args.single_model:
        run_single_model(args.single_model, args)
        return

    if args.mode in ("power-up", "all"):
        run_power_up(args)
    if args.mode in ("multi-seed", "all"):
        run_multi_seed(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Power-Up & Multi-Seed (E1.1 + E1.2)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mode", choices=["power-up", "multi-seed", "all"], default="all")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--single-model", default=None)
    parser.add_argument("--n-prompts", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    main(args)
