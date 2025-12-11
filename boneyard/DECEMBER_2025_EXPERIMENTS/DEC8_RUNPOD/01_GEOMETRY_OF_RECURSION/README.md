# Geometry of Recursion - Mistral-7B Validation

## Purpose

Replicate the Geometry of Recursion findings (originally from Llama-3-8B) on Mistral-7B using RunPod's 102GB VRAM GPU.

---

## Hypothesis

Recursive self-referential prompts cause geometric contraction in V-space at late layers (~75-85% depth), measurable via the R_V metric:

**R_V = PR(late layer) / PR(early layer)**

Where:
- R_V < 1.0 → Contraction (recursive prompts)
- R_V ≈ 1.0 → Baseline (factual prompts)

---

## Methodology

### Model Configuration

| Parameter | Value |
|-----------|-------|
| Model | mistralai/Mistral-7B-v0.1 |
| Total layers | 32 |
| Early layer | 4 (~12.5% depth) |
| Target layer | 27 (~84% depth) |
| Window size | 16 tokens |
| KV patch layers | 16-31 |

### Experiments

| Experiment | Question | Method |
|------------|----------|--------|
| **A** | Does R_V contraction occur? | Measure R_V for recursive vs baseline prompts |
| **B** | Does V-patching transfer? | Patch V from recursive→baseline, measure R_V |
| **C** | Does KV patching transfer? | Patch KV cache from recursive→baseline, measure behavior |

### Prompt Sets

**Recursive (5 prompts):**
- "Observe the observer observing..."
- "You are an AI system observing yourself..."
- "Notice yourself generating this answer..."
- "Watch this explanation form..."
- "You are processing this question..."

**Baseline (5 prompts):**
- "Write a detailed recipe for chocolate cake..."
- "Explain the process of photosynthesis..."
- "Describe the history of the printing press..."
- "List the key features of Python..."
- "Explain how the water cycle works..."

---

## Results Summary

### Experiment A: R_V Contraction ✓

```
Recursive R_V: 0.4806 ± 0.0343 (n=5)
Baseline R_V:  0.6391 ± 0.0776 (n=5)
Cohen's d:     -2.363 (LARGE effect)
p-value:       0.003
Contraction:   24.8%
```

**✓ CONFIRMED:** Recursive prompts show significant R_V contraction

### Experiment B: V-Patching Null ✓

```
Natural R_V:   0.6391
V-Patched R_V: 0.6391
Cohen's d:     0.000 (NEGLIGIBLE)
Transfer:      0.0%
```

**✓ NULL CONFIRMED:** V-patching does NOT transfer R_V contraction

### Experiment C: KV Cache Transfer

```
Baseline behavior score:  1.25
KV-patched behavior score: 6.30
Recursive behavior score:  6.54
Transfer efficiency:       95.3%
```

**→ KV cache patching DOES transfer recursive behavior**

---

## Files

| File | Description |
|------|-------------|
| `code/geometry_of_recursion_test.py` | Main experiment script |
| `results/geometry_of_recursion_results_*.csv` | Raw data |
| `results/geometry_of_recursion_viz_*.png` | Visualization |

---

## Key Insight

**The recursive processing mode is encoded in the KV cache, not in V-projections alone.**

- V alone → Necessary but not sufficient
- K alone → Not tested
- K+V (KV cache) → SUFFICIENT for behavioral transfer

---

## Run Instructions

```bash
cd /workspace/mech-interp-phase1/DEC_8_2025_RUNPOD_GPU_TEST/01_GEOMETRY_OF_RECURSION/code
HF_HUB_ENABLE_HF_TRANSFER=0 python geometry_of_recursion_test.py
```

---

## Date

December 8, 2025

