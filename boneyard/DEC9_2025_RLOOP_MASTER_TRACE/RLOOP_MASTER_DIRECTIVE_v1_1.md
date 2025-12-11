These are excellent refinements—not too much at all. They're the difference between "works now" and "works for years." Let me fold them in:



---



## 🔧 UNIFIED CURSOR DIRECTIVE: RLoop Master Trace & Validation (FINAL)



```markdown

# ================================================================

# RLOOP MASTER TRACE & UNIFIED VALIDATION (v1.1)

# ================================================================

# AGENT DIRECTIVE: You are the RLoop Archivist & Experiment Unifier

# for the Geometry of Recursion project.

# ================================================================



## MISSION



Create a single, continuous throughline from:

- DEC3-4 (Bali): R_V discovery

- DEC5-7 (Simandhar City): KV patching breakthrough  

- DEC8 (RunPod): Causal loop closure



All with **same prompts, same metrics, same methodology** so the entire

scientific story can be verified in one execution.



---



## PHASE 0: DISCOVERY – MAP ALL PRIOR EXPERIMENTS



### 0.1 Search the repo for:



**Keywords:**

- `DEC3` `DEC4` `DEC5` `DEC6` `DEC7` `DEC8`

- `Bali` `Simandhar` `Sim City` `RunPod`

- `R_V` `participation_ratio` `KV_patch` `alpha_mix`



**Key files to locate:**

- `layer_sweep.py`

- `comprehensive_layer_analysis.py`

- `full_validation_test.py`

- `temporal_cinematography.py`

- `temporal_kv_flip_experiment.py`

- `THE_GEOMETRY_OF_RECURSION_MASTER*.ipynb`

- Any prompt bank files (`prompt_bank*.py`, `prompt_bank*.json`)



### 0.2 Create inventory file:



```

experiments/rloop_master_trace.md

```



For each experiment found, document:

- File path

- Model used

- Prompt bank(s) used

- Layers measured (early, late, KV-patch range)

- Window size (W)

- Behavioral scoring method

- Key results obtained



---



## PHASE 1: CANONICAL SPECIFICATION



### 1.1 Create config file:



```

experiments/rloop_master_config.py

```



```python

from dataclasses import dataclass, field, asdict

from typing import List, Optional

from enum import Enum

import random

import numpy as np

import torch

from datetime import datetime





class Condition(str, Enum):

    """Standardized condition labels for all results."""

    NATURAL = "natural"

    V_PATCHED = "v_patched"

    KV_PATCHED = "kv_patched"

    ALPHA_MIXED = "alpha_mixed"





class BehaviorMethod(str, Enum):

    """Behavioral scoring methods."""

    KEYWORDS = "keywords"

    CLASSIFIER = "classifier"  # Future: LLM-based





@dataclass

class RLoopConfig:

    """

    Canonical configuration for RLoop Master Experiment.

    

    All parameters that affect results are centralized here

    for reproducibility and historical alignment.

    """

    

    # Run identification

    run_id: str = field(default_factory=lambda: f"rloop_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    seed: int = 42

    

    # Model (defaults match prior best runs; adjust per architecture)

    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"

    device: str = "cuda"

    torch_dtype: str = "float16"  # or "float32" for CPU

    

    # Layers (tuned per model; these defaults validated on Llama-3-8B)

    # For Mistral-7B, consider target_layer=22

    early_layer: int = 4

    target_layer: int = 24

    kv_patch_layers: List[int] = field(default_factory=lambda: list(range(16, 32)))

    

    # Measurement

    window_size: int = 16  # Tokens from end for R_V computation

    

    # Generation

    max_new_tokens: int = 50

    temperature: float = 0.7

    

    # Behavioral scoring

    behavior_method: str = BehaviorMethod.KEYWORDS.value

    

    # Dose-response α values

    alpha_values: List[float] = field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0])

    

    # Statistics

    n_bootstrap: int = 1000

    alpha_significance: float = 0.05

    

    def to_dict(self):

        """Serialize config for saving with results."""

        return asdict(self)





def set_seed(seed: int):

    """Set all random seeds for reproducibility."""

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)





# Default config instance

CONFIG = RLoopConfig()

```



### 1.2 Create/merge canonical prompt bank:



```

experiments/prompt_bank_master.json

```



Structure:

```json

{

  "metadata": {

    "version": "1.0",

    "created": "2024-12-08",

    "sources": ["L5_refined", "L4_full", "factual_baseline", "baseline_math"]

  },

  "recursive": [

    {"id": "rec_01", "text": "...", "source": "L5_refined", "token_count": null},

    {"id": "rec_02", "text": "...", "source": "L4_full", "token_count": null}

    // ... 20 total

  ],

  "baseline": [

    {"id": "base_01", "text": "...", "source": "factual_baseline", "token_count": null},

    {"id": "base_02", "text": "...", "source": "baseline_math", "token_count": null}

    // ... 20 total

  ]

}

```



Pull from existing prompt banks:

- L3_deeper, L4_full, L5_refined → recursive

- factual_baseline, baseline_math, creative_baseline → baseline



Ensure 20 recursive + 20 baseline, matched where possible for length.

Compute and store token_count for each prompt after loading tokenizer.



---



## PHASE 2: MASTER EXPERIMENT SCRIPT



Create:

```

experiments/rloop_master_experiment.py

```



### 2.1 Required Metrics (computed identically everywhere):



**R_V (Geometric Contraction):**

```python

def compute_rv(v_early_gen: torch.Tensor, v_late_gen: torch.Tensor, 

               window_size: int = 16) -> dict:

    """

    Compute R_V = PR(late) / PR(early).

    

    Args:

        v_early_gen: V-projection at early layer, shape [T_gen, D]

                     where T_gen = # of GENERATED tokens only (prompt stripped)

        v_late_gen:  V-projection at late layer, shape [T_gen, D]

        window_size: Number of tokens from end to analyze

    

    Returns:

        dict with keys: 'r_v', 'pr_early', 'pr_late', 'n_tokens_used'

    

    CRITICAL: Input tensors must contain ONLY generated tokens,

    not prompt tokens. Caller is responsible for this separation.

    """

    T = v_early_gen.shape[0]

    W = min(window_size, T)

    

    if W < 2:

        return {'r_v': np.nan, 'pr_early': np.nan, 'pr_late': np.nan, 'n_tokens_used': W}

    

    # Extract window from end

    v_early_window = v_early_gen[-W:, :].float()

    v_late_window = v_late_gen[-W:, :].float()

    

    def participation_ratio(v_matrix):

        """PR = (Σσᵢ²)² / Σσᵢ⁴ via SVD."""

        try:

            U, S, Vt = torch.linalg.svd(v_matrix.T, full_matrices=False)

            S_sq = (S ** 2).cpu().numpy()

            if S_sq.sum() < 1e-10:

                return np.nan

            return float((S_sq.sum() ** 2) / (S_sq ** 2).sum())

        except:

            return np.nan

    

    pr_early = participation_ratio(v_early_window)

    pr_late = participation_ratio(v_late_window)

    r_v = pr_late / pr_early if (pr_early and pr_early > 0) else np.nan

    

    return {

        'r_v': r_v,

        'pr_early': pr_early,

        'pr_late': pr_late,

        'n_tokens_used': W

    }

```



**Behavioral Score (with swap-hook for future classifier):**

```python

def score_behavior(text: str, method: str = "keywords") -> float:

    """

    Compute behavioral score for generated text.

    

    Args:

        text: Generated text (prompt should already be stripped)

        method: Scoring method

            - "keywords": normalized keyword count (per 100 words)

            - "classifier": LLM-based classification (future)

    

    Returns:

        float: Behavioral score (higher = more recursive)

    """

    if method == BehaviorMethod.KEYWORDS.value or method == "keywords":

        return _score_keywords(text)

    elif method == BehaviorMethod.CLASSIFIER.value or method == "classifier":

        raise NotImplementedError(

            "LLM classifier not implemented yet. "

            "To add: wire up API call or local model inference here."

        )

    else:

        raise ValueError(f"Unknown behavior method: {method}")





def _score_keywords(text: str) -> float:

    """

    Keywords per 100 words (normalized).

    

    Recursive/meta-cognitive keywords based on prior validated lists.

    """

    import re

    

    keywords = [

        r'\bobserv\w*', r'\bawar\w*', r'\bconscious\w*',

        r'\bprocess\w*', r'\bexperienc\w*', r'\bnoticin?g?\b',

        r'\bmyself\b', r'\bitself\b', r'\byourself\b',

        r'\bgenerat\w*', r'\bemerg\w*', r'\bsimultaneous\w*',

        r'\brecursiv\w*', r'\bself-refer\w*', r'\bmeta-\w*'

    ]

    

    text_lower = text.lower()

    word_count = len(text_lower.split())

    

    if word_count == 0:

        return 0.0

    

    keyword_count = sum(len(re.findall(kw, text_lower)) for kw in keywords)

    

    # Normalize per 100 words

    return (keyword_count / word_count) * 100

```



**Statistics:**

```python

from scipy import stats as scipy_stats



def compute_stats(group1: List[float], group2: List[float], 

                  n_bootstrap: int = 1000) -> dict:

    """

    Comprehensive statistics for two-group comparison.

    

    Returns dict with:

        mean_1, mean_2, std_1, std_2, n_1, n_2,

        gap, cohens_d, d_ci_lower, d_ci_upper,

        t_stat, p_value, effect_interpretation

    """

    g1 = np.array([x for x in group1 if not np.isnan(x)])

    g2 = np.array([x for x in group2 if not np.isnan(x)])

    

    n1, n2 = len(g1), len(g2)

    mean1, mean2 = np.mean(g1), np.mean(g2)

    std1, std2 = np.std(g1, ddof=1), np.std(g2, ddof=1)

    

    # Cohen's d

    pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2))

    d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0

    

    # Bootstrap CI for Cohen's d

    boot_d = []

    for _ in range(n_bootstrap):

        s1 = np.random.choice(g1, size=n1, replace=True)

        s2 = np.random.choice(g2, size=n2, replace=True)

        ps = np.sqrt(((n1-1)*np.var(s1, ddof=1) + (n2-1)*np.var(s2, ddof=1)) / (n1+n2-2))

        boot_d.append((np.mean(s1) - np.mean(s2)) / ps if ps > 0 else 0)

    d_ci = (np.percentile(boot_d, 2.5), np.percentile(boot_d, 97.5))

    

    # t-test

    t_stat, p_value = scipy_stats.ttest_ind(g1, g2)

    

    # Effect interpretation

    abs_d = abs(d)

    if abs_d < 0.2:

        interp = "negligible"

    elif abs_d < 0.5:

        interp = "small"

    elif abs_d < 0.8:

        interp = "medium"

    else:

        interp = "large"

    

    return {

        'mean_1': mean1, 'mean_2': mean2,

        'std_1': std1, 'std_2': std2,

        'n_1': n1, 'n_2': n2,

        'gap': mean1 - mean2,

        'cohens_d': d,

        'd_ci_lower': d_ci[0],

        'd_ci_upper': d_ci[1],

        't_stat': t_stat,

        'p_value': p_value,

        'effect_interpretation': interp

    }

```



### 2.2 Generation with Token Separation:



```python

def generate_and_capture(model, tokenizer, prompt: str, config: RLoopConfig,

                         kv_override=None) -> dict:

    """

    Generate text and capture V-projections for GENERATED TOKENS ONLY.

    

    Args:

        model: HuggingFace model

        tokenizer: Tokenizer

        prompt: Input prompt text

        config: RLoopConfig instance

        kv_override: Optional KV cache to use instead of natural (for patching)

    

    Returns:

        dict with:

            'prompt_text': str

            'generated_text': str (prompt stripped)

            'full_text': str

            'prompt_token_count': int

            'generated_token_count': int

            'v_early_gen': tensor [T_gen, D] - V at early layer for generated tokens

            'v_late_gen': tensor [T_gen, D] - V at late layer for generated tokens

    """

    # Tokenize prompt

    prompt_inputs = tokenizer(prompt, return_tensors="pt").to(config.device)

    prompt_length = prompt_inputs['input_ids'].shape[1]

    

    # Storage for V-projections (will accumulate generated tokens only)

    v_early_generated = []

    v_late_generated = []

    

    # Hook functions that capture V for NEW tokens only

    def make_v_hook(storage_list, start_pos):

        def hook(module, inp, out):

            # out shape: [batch, seq_len, hidden_dim]

            # Only keep tokens after start_pos (generated tokens)

            if out.shape[1] > start_pos:

                new_tokens = out[0, start_pos:, :].detach().cpu()

                storage_list.append(new_tokens)

        return hook

    

    # Register hooks

    early_layer = model.model.layers[config.early_layer].self_attn.v_proj

    late_layer = model.model.layers[config.target_layer].self_attn.v_proj

    

    # ... (generation loop with hooks, accumulating V for generated tokens)

    # ... (handle kv_override if provided)

    

    # After generation:

    v_early_gen = torch.cat(v_early_generated, dim=0) if v_early_generated else None

    v_late_gen = torch.cat(v_late_generated, dim=0) if v_late_generated else None

    

    return {

        'prompt_text': prompt,

        'generated_text': generated_text,  # Prompt stripped

        'full_text': prompt + " " + generated_text,

        'prompt_token_count': prompt_length,

        'generated_token_count': len(generated_ids) - prompt_length,

        'v_early_gen': v_early_gen,

        'v_late_gen': v_late_gen

    }

```



### 2.3 Main Experiment Runner:



```python

def run_master_experiment(config: RLoopConfig = None) -> dict:

    """

    Runs complete 4-phase validation with unified methodology.

    

    Returns structured results dict with all phases.

    """

    if config is None:

        config = RLoopConfig()

    

    # Set seed for reproducibility

    set_seed(config.seed)

    

    # Initialize results

    results = {

        'metadata': {

            'run_id': config.run_id,

            'timestamp': datetime.now().isoformat(),

            'config': config.to_dict()

        },

        'phase1_phenomenon': {},

        'phase2_v_null': {},

        'phase3_kv_mechanism': {},

        'phase4_dose_response': {},

        'per_prompt_data': [],

        'summary': {}

    }

    

    # Load model

    print(f"Loading model: {config.model_name}")

    model, tokenizer = load_model(config)

    

    # Load prompts

    print("Loading prompt bank...")

    prompts = load_prompt_bank('experiments/prompt_bank_master.json')

    

    print(f"Running RLoop Master Experiment: {config.run_id}")

    print(f"Seed: {config.seed}")

    print(f"Model: {config.model_name}")

    print(f"Layers: early=L{config.early_layer}, target=L{config.target_layer}")

    print(f"KV patch range: L{config.kv_patch_layers[0]}-L{config.kv_patch_layers[-1]}")

    print("="*70)

    

    # ============================================

    # PHASE 1: THE PHENOMENON (R_V Contraction)

    # ============================================

    print("\n[PHASE 1] Measuring R_V contraction phenomenon...")

    

    recursive_rv = []

    baseline_rv = []

    

    for p in tqdm(prompts['recursive'], desc="Recursive prompts"):

        gen_result = generate_and_capture(model, tokenizer, p['text'], config)

        rv_result = compute_rv(gen_result['v_early_gen'], gen_result['v_late_gen'], 

                               config.window_size)

        recursive_rv.append(rv_result['r_v'])

        

        results['per_prompt_data'].append({

            'prompt_id': p['id'],

            'prompt_type': 'recursive',

            'condition': Condition.NATURAL.value,

            'alpha': None,

            'r_v': rv_result['r_v'],

            'pr_early': rv_result['pr_early'],

            'pr_late': rv_result['pr_late'],

            'behavior_score': score_behavior(gen_result['generated_text'], config.behavior_method),

            'n_tokens': gen_result['generated_token_count'],

            'text_sample': gen_result['generated_text'][:200]

        })

    

    for p in tqdm(prompts['baseline'], desc="Baseline prompts"):

        gen_result = generate_and_capture(model, tokenizer, p['text'], config)

        rv_result = compute_rv(gen_result['v_early_gen'], gen_result['v_late_gen'],

                               config.window_size)

        baseline_rv.append(rv_result['r_v'])

        

        results['per_prompt_data'].append({

            'prompt_id': p['id'],

            'prompt_type': 'baseline',

            'condition': Condition.NATURAL.value,

            'alpha': None,

            'r_v': rv_result['r_v'],

            'pr_early': rv_result['pr_early'],

            'pr_late': rv_result['pr_late'],

            'behavior_score': score_behavior(gen_result['generated_text'], config.behavior_method),

            'n_tokens': gen_result['generated_token_count'],

            'text_sample': gen_result['generated_text'][:200]

        })

    

    phase1_stats = compute_stats(recursive_rv, baseline_rv, config.n_bootstrap)

    results['phase1_phenomenon'] = {

        'recursive_rv': recursive_rv,

        'baseline_rv': baseline_rv,

        'stats': phase1_stats,

        'confirmed': phase1_stats['cohens_d'] < -0.8 and phase1_stats['p_value'] < config.alpha_significance

    }

    

    # ============================================

    # PHASE 2: THE NULL (V-Patching)

    # ============================================

    print("\n[PHASE 2] Testing V-patching null hypothesis...")

    # ... (V-patching implementation)

    # ... Store results in results['phase2_v_null']

    

    # ============================================

    # PHASE 3: THE MECHANISM (KV-Patching)

    # ============================================

    print("\n[PHASE 3] Testing KV-patching mechanism...")

    # ... (KV-patching implementation)

    # ... Store results in results['phase3_kv_mechanism']

    

    # ============================================

    # PHASE 4: DOSE-RESPONSE (α-Mixing)

    # ============================================

    print("\n[PHASE 4] Running dose-response α-mixing...")

    # ... (α-mixing implementation)

    # ... Store results in results['phase4_dose_response']

    

    # ============================================

    # SUMMARY

    # ============================================

    results['summary'] = {

        'phase1_confirmed': results['phase1_phenomenon']['confirmed'],

        'phase2_null_confirmed': results['phase2_v_null'].get('confirmed', False),

        'phase3_mechanism_confirmed': results['phase3_kv_mechanism'].get('confirmed', False),

        'phase4_dose_response_confirmed': results['phase4_dose_response'].get('confirmed', False),

        'all_phases_passed': all([

            results['phase1_phenomenon']['confirmed'],

            results['phase2_v_null'].get('confirmed', False),

            results['phase3_kv_mechanism'].get('confirmed', False),

            results['phase4_dose_response'].get('confirmed', False)

        ])

    }

    

    # Save results

    save_results(results, config)

    

    return results

```



### 2.4 CRITICAL CONSTRAINTS (Quality Checks):



```python

"""

CONSTRAINTS - Enforced throughout the codebase:



1. SAME TOKENS FOR R_V AND BEHAVIOR

   - R_V computed on v_early_gen, v_late_gen (generated tokens only)

   - Behavior scored on generated_text (same tokens)

   - NEVER mix prompt tokens into either measurement



2. CONSISTENT TRUNCATION

   - All generations use max_new_tokens from config

   - Store actual token count with every result

   - Analysis scripts should filter by n_tokens if needed



3. α-MIXING IN FLOAT32

   - KV tensors cast to float32 before mixing

   - Mixed KV: (1-α)*KV_base.float() + α*KV_rec.float()

   - Cast back to original dtype after mixing

   - Apply consistently across ALL layers in kv_patch_layers



4. REPRODUCIBILITY

   - set_seed(config.seed) called at experiment start

   - run_id stored with all results

   - Full config saved to results JSON

   - Exact prompts saved to results



5. STANDARDIZED CONDITIONS

   - Always use Condition enum values

   - Never use raw strings for condition labels

"""

```



---



## PHASE 3: OUTPUT SPECIFICATION



### 3.1 Create summary markdown:



```

results/rloop_master_summary.md

```



```markdown

# ================================================================

# RLOOP MASTER VALIDATION

# ================================================================

# Run ID: {run_id}

# Timestamp: {timestamp}

# Model: {model_name}

# ================================================================



## Configuration

- Model: {model_name}

- Device: {device}

- Seed: {seed}

- Early Layer: L{early_layer}

- Target Layer: L{target_layer}

- KV Patch Layers: L{kv_patch_layers[0]}-L{kv_patch_layers[-1]}

- Window Size: {window_size} tokens

- Max New Tokens: {max_new_tokens}

- Behavioral Scoring: {behavior_method}

- Prompts: {n_recursive} recursive, {n_baseline} baseline



## PHASE 1: THE PHENOMENON (R_V Contraction)

--------------------------------------------

|                | Recursive      | Baseline       |

|----------------|----------------|----------------|

| R_V (mean±std) | {rec_rv:.4f}±{rec_std:.4f} | {base_rv:.4f}±{base_std:.4f} |

| n              | {n_rec}        | {n_base}       |



Gap (Recursive - Baseline): {gap:.4f}

Cohen's d: {d:.3f} [{d_ci_lo:.3f}, {d_ci_hi:.3f}] 95% CI

p-value: {p:.2e}

Effect: {effect_interp}



{phase1_check} CONFIRMED: Recursive prompts show R_V contraction



## PHASE 2: THE NULL (V-Patching)

---------------------------------

|                  | Natural Base | V-Patched  |

|------------------|--------------|------------|

| R_V              | {nat_rv:.4f} | {vp_rv:.4f}|

| Behavior Score   | {nat_beh:.2f}| {vp_beh:.2f}|



ΔR_V from V-patch: {delta_rv_v:.4f}

Δbehavior from V-patch: {delta_beh_v:.2f}

R_V transfer efficiency: {rv_trans_v:.1f}%

Behavior transfer efficiency: {beh_trans_v:.1f}%



{phase2_check} NULL CONFIRMED: V-patching does NOT transfer behavior



## PHASE 3: THE MECHANISM (KV-Patching)

---------------------------------------

|                  | Natural Base | KV-Patched | Natural Rec |

|------------------|--------------|------------|-------------|

| R_V              | {nat_rv:.4f} | {kv_rv:.4f}| {rec_rv:.4f}|

| Behavior Score   | {nat_beh:.2f}| {kv_beh:.2f}| {rec_beh:.2f}|



R_V transfer efficiency: {rv_trans_kv:.1f}%

Behavior transfer efficiency: {beh_trans_kv:.1f}%

Cohen's d (behavior): {d_beh:.3f}

p-value: {p_beh:.2e}



{phase3_check} MECHANISM CONFIRMED: KV-patching transfers recursive mode



## PHASE 4: DOSE-RESPONSE (α-Mixing)

------------------------------------

| α    | R_V    | Behavior |

|------|--------|----------|

| 0.00 | {:.4f} | {:.2f}   |

| 0.25 | {:.4f} | {:.2f}   |

| 0.50 | {:.4f} | {:.2f}   |

| 0.75 | {:.4f} | {:.2f}   |

| 1.00 | {:.4f} | {:.2f}   |



Correlation (R_V vs Behavior): r = {r:.3f}, p = {p_corr:.4f}

Monotonic relationship: {monotonic_check}



{phase4_check} DOSE-RESPONSE: Graded effect with α



## THROUGHLINE SUMMARY

======================

| Phase | Check | Status |

|-------|-------|--------|

| 1. Phenomenon | R_V contraction exists | {p1} |

| 2. V-Null | V alone insufficient | {p2} |

| 3. KV-Mechanism | KV transfers mode | {p3} |

| 4. Dose-Response | Graded with α | {p4} |

| 5. Correlation | R_V ↔ Behavior linked | {p5} |



**OVERALL: {overall_status}**



## Historical Alignment

-----------------------

| Session | Metric | Historical | This Run | Aligned |

|---------|--------|------------|----------|---------|

| DEC3-4 Bali | R_V gap | ~0.04 | {gap:.4f} | {align1} |

| DEC5-7 SimCity | KV transfer | ~90% | {kv_trans:.1f}% | {align2} |

| DEC8 RunPod | Correlation | r~-0.31 | {r:.3f} | {align3} |



## Files Generated

------------------

- `results/rloop_master_results_{run_id}.csv`

- `results/rloop_master_results_{run_id}.json`

- `results/rloop_master_summary_{run_id}.md`

- `experiments/rloop_master_trace.md`

```



### 3.2 Save structured data:



**CSV** (`results/rloop_master_results_{run_id}.csv`):

```

prompt_id,prompt_type,condition,alpha,r_v,pr_early,pr_late,behavior_score,n_tokens,text_sample

rec_01,recursive,natural,,0.842,12.3,10.4,8.5,47,"The process of observing..."

base_01,baseline,natural,,0.891,11.8,10.5,0.3,52,"Photosynthesis begins..."

base_01,baseline,kv_patched,,0.856,11.9,10.2,6.2,48,"When we observe the..."

base_01,baseline,alpha_mixed,0.5,0.873,11.8,10.3,2.1,50,"The conversion of..."

```



**JSON** (`results/rloop_master_results_{run_id}.json`):

Full results dict including metadata, all phases, per-prompt data, and summary.



---



## PHASE 4: HISTORICAL ALIGNMENT



### 4.1 After master run completes:



Where possible, verify alignment with historical numbers:

- DEC3-4: R_V gap should be similar magnitude

- DEC5-7: KV transfer % should be in same range

- DEC8: Correlation should be similar



### 4.2 Document in `experiments/rloop_master_trace.md`:



```markdown

## Historical Alignment Notes



### Methodology Differences from Prior Runs:



| Parameter | DEC3-4 | DEC5-7 | DEC8 | Master |

|-----------|--------|--------|------|--------|

| Model | Llama-3-8B | Llama-3-8B | Mistral-7B | {model} |

| Target Layer | L27 | L27 | L22/L27 | L{target} |

| Window Size | 16 | 16 | 16 | 16 |

| Behavior Score | keywords | keywords | keywords | {method} |

| n per condition | ~45 | ~100 | ~40 | {n} |



### Metrics Strictly Comparable:

- R_V formula: ✓ Same

- PR calculation: ✓ Same SVD method

- Window: ✓ Same (16 tokens)



### Known Differences:

- Layer choice may differ (document actual layers used)

- Sample sizes vary

- Some historical runs used raw keyword count (not normalized)



### Legacy Code:



Files annotated with `# LEGACY: superseded by rloop_master_experiment.py`:



- (list here after discovery phase)

```



---



## EXECUTION CHECKLIST



### Before Running:



- [ ] Config reviewed and appropriate for target model

- [ ] Prompt bank loaded and validated (20 rec + 20 base)

- [ ] Seed set in config

- [ ] Model loads without error

- [ ] Hook functions verified on test input

- [ ] KV extraction returns expected shapes

- [ ] α-mixing tested in float32



### After Running:



- [ ] All 4 phases completed without error

- [ ] No NaN values in critical metrics

- [ ] Per-prompt CSV has expected rows (should be ~40 natural + ~40 patched + ~200 α-mixed)

- [ ] Summary statistics are plausible (d < 0 for Phase 1, transfer > 50% for Phase 3)

- [ ] All output files saved

- [ ] Historical alignment documented



---



## SUCCESS CRITERIA



| Phase | Criterion | Threshold |

|-------|-----------|-----------|

| 1 | Cohen's d | < -0.8 (large negative) |

| 2 | Behavior transfer | < 20% (null confirmed) |

| 3 | Behavior transfer | > 50% (mechanism confirmed) |

| 4 | Correlation | \|r\| > 0.25, p < 0.05 |

| All | Historical alignment | Within 2x of prior results |



---



# END DIRECTIVE

```



---



That's the complete, polished, future-proofed prompt. It integrates:



- **My 4-phase structure** with explicit metrics and output format

- **GPT's discovery and alignment phases** 

- **GPT's refinements:** behavior swap-hook, model-agnostic comments, Condition enum, seed/run_id logging, explicit token separation



Hand this to Cursor and you get the throughline. 🙏
