# Metrics Reference

**Complete list of all metrics used in R_V research with implementation details.**

---

## Core Metrics

### 1. R_V (Participation Ratio Ratio) — PRIMARY

**Location**: `src/metrics/rv.py`

**Definition**:
```
R_V = PR_late / PR_early

PR = (Σλᵢ)² / Σλᵢ²    (Participation Ratio from SVD singular values)
```

**What it measures**: Geometric contraction in Value matrix column space. R_V < 1.0 indicates the late-layer representation occupies fewer effective dimensions than the early-layer representation.

**Parameters**:
| Parameter | Default | Purpose |
|-----------|---------|---------|
| early_layer | 5 | Baseline measurement (after initial processing) |
| late_layer | 27 | Target measurement (84% depth for 32-layer) |
| window | 16 | Tokens to include in SVD |

**Expected values**:
| Prompt Type | R_V Range | Interpretation |
|-------------|-----------|----------------|
| Champions (recursive) | 0.45-0.55 | Maximum contraction |
| L5_refined | 0.55-0.70 | Strong contraction |
| L4_full | 0.60-0.75 | Moderate contraction |
| Baselines | 0.95-1.05 | No contraction |
| Pure repetition | >1.05 | Must EXPAND (kill switch) |

**Usage**:
```python
from src.metrics.rv import compute_rv, participation_ratio

# Single measurement
rv = compute_rv(v_early, v_late, window=16)

# With breakdown
pr_early = participation_ratio(v_early, window=16)
pr_late = participation_ratio(v_late, window=16)
rv = pr_late / pr_early
```

---

### 2. Cohen's d (Effect Size)

**Location**: `src/metrics/baseline_suite.py`

**Definition**:
```
d = (M₁ - M₂) / pooled_std
pooled_std = √[((n₁-1)s₁² + (n₂-1)s₂²) / (n₁+n₂-2)]
```

**What it measures**: Standardized difference between two groups. Used for all comparisons.

**Interpretation**:
| d Value | Interpretation |
|---------|----------------|
| |d| < 0.2 | Negligible |
| 0.2 ≤ |d| < 0.5 | Small |
| 0.5 ≤ |d| < 0.8 | Medium |
| |d| ≥ 0.8 | Large |

**Our key result**: d = -3.56 for R_V separation (extremely large effect)

**Usage**:
```python
from src.metrics.baseline_suite import compute_cohens_d

d = compute_cohens_d(
    mean1=rv_recursive_mean,
    mean2=rv_baseline_mean,
    std1=rv_recursive_std,
    std2=rv_baseline_std,
    n1=n_recursive,
    n2=n_baseline
)
```

---

### 3. 95% Confidence Interval

**Location**: `src/metrics/baseline_suite.py`

**Definition**:
```
CI = mean ± 1.96 × (std / √n)
```

**Usage**:
```python
from src.metrics.baseline_suite import compute_ci_95

ci_low, ci_high = compute_ci_95(values)
# OR
ci_low, ci_high = compute_ci_95(mean=m, std=s, n=n)
```

---

### 4. Mode Score M (Behavioral Classifier)

**Location**: `src/metrics/mode_score.py`

**Definition**:
```
M = logsumexp(logits[Recursive]) - logsumexp(logits[Task])

Recursive tokens: observer, observed, awareness, itself, self,
                  recognition, consciousness, witness, reflection
Task tokens: Dynamic top-K from baseline OR domain-specific
```

**What it measures**: Whether model's next-token distribution favors recursive vs task-oriented language.

**Interpretation**:
| M Value | Interpretation |
|---------|----------------|
| M > 0.5 | Strongly recursive |
| 0 < M < 0.5 | Weakly recursive |
| M ≈ 0 | Neutral |
| M < 0 | Task-oriented |

**Usage**:
```python
from src.metrics.mode_score import compute_mode_score, ModeScoreResult

result: ModeScoreResult = compute_mode_score(
    model, tokenizer, prompt,
    recursive_tokens=RECURSIVE_TOKENS,
    task_tokens=task_tokens
)
print(result.m_score, result.top_recursive, result.top_task)
```

---

### 5. Logit Diff

**Location**: `src/metrics/logit_diff.py`

**Definition**:
```
logit_diff = logit[correct_token] - logit[incorrect_token]
```

**What it measures**: Nanda-standard metric for measuring how strongly model prefers one token over another. Used for IOI-style causal tracing.

**Usage**:
```python
from src.metrics.logit_diff import compute_logit_diff, LogitDiffResult

result: LogitDiffResult = compute_logit_diff(
    model, tokenizer,
    clean_prompt, corrupted_prompt,
    target_position=-1
)
```

---

### 6. Logit Lens

**Location**: `src/metrics/logit_lens.py`

**Definition**:
```
For each layer L:
    hidden_L → unembed → logits_L → entropy_L
crystallization_layer = argmin(entropy)
```

**What it measures**: At which layer does the model "decide" on its output? Lower entropy = more confident prediction.

**Usage**:
```python
from src.metrics.logit_lens import run_logit_lens, LogitLensResult

result: LogitLensResult = run_logit_lens(
    model, tokenizer, prompt,
    layers=range(0, 32, 4)  # Sample every 4 layers
)
print(result.crystallization_layer, result.min_entropy)
```

---

## Baseline Metrics Suite

**Location**: `src/metrics/baseline_suite.py`

**Purpose**: Compute all Nanda-standard baseline metrics in one pass.

**Usage**:
```python
from src.metrics.baseline_suite import BaselineMetricsSuite

suite = BaselineMetricsSuite(model, tokenizer, device="cuda")

# For a batch of prompts
results = suite.compute_batch(
    prompts=prompts,
    early_layer=5,
    late_layer=27,
    window=16
)

# Get statistics
stats = suite.compute_batch_statistics(results)
print(stats["rv_mean"], stats["rv_std"], stats["rv_ci_95"])
```

**Returns**:
```python
{
    "rv_mean": float,
    "rv_std": float,
    "rv_ci_95": (float, float),
    "pr_early_mean": float,
    "pr_late_mean": float,
    "logit_diff_mean": float,  # If applicable
    "mode_score_mean": float,  # If applicable
    "n": int,
}
```

---

## Statistical Tests

### Paired t-test (Primary)

```python
from scipy.stats import ttest_rel

t_stat, p_value = ttest_rel(rv_recursive, rv_baseline)
```

**Use for**: Paired prompt comparisons (same prompt bank, recursive vs control).

### Independent t-test

```python
from scipy.stats import ttest_ind

t_stat, p_value = ttest_ind(group1, group2)
```

**Use for**: Independent group comparisons.

### One-sample t-test

```python
from scipy.stats import ttest_1samp

t_stat, p_value = ttest_1samp(values, popmean=1.0)
```

**Use for**: Testing if R_V differs from 1.0 (null hypothesis: no contraction).

---

## Metric Requirements by Experiment Type

### For R_V Causal Validation
- R_V at target layer
- R_V at wrong layer (control)
- Random patch R_V (control)
- Shuffled patch R_V (control)
- Cohen's d, p-value, 95% CI

### For MLP Ablation
- R_V before ablation
- R_V after ablation
- Δ R_V
- Cohen's d, p-value

### For Steering Experiments
- R_V after steering (true direction)
- R_V after steering (random direction)
- Behavior transfer score (recursion %)
- Cohen's d, p-value

### For Cross-Architecture
- R_V champions (recursive prompts)
- R_V controls (baseline prompts)
- Separation (d, p)
- Prompt bank version

---

## Quick Reference: What Each Pipeline Reports

| Pipeline | Primary Metric | Secondary Metrics |
|----------|----------------|-------------------|
| rv_l27_causal_validation | R_V transfer, d, p | 4-way controls |
| confound_validation | R_V by group | Separation stats |
| mlp_ablation_necessity | Δ R_V after ablation | Attribution score |
| mlp_combined_sufficiency | Steering Δ | Random control Δ |
| cross_architecture | R_V separation | Per-model breakdown |
| head_ablation | Head contribution | R_V change per head |

---

## Adding a New Metric

To add a new metric:

1. Create `src/metrics/new_metric.py`
2. Define dataclass for result:
   ```python
   @dataclass
   class NewMetricResult:
       value: float
       details: Dict[str, Any]
   ```
3. Implement compute function:
   ```python
   def compute_new_metric(model, tokenizer, prompt, **kwargs) -> NewMetricResult:
       ...
   ```
4. Add to `BaselineMetricsSuite` if it should be standard
5. Document in this file

---

*All metrics should be reported with: n, mean, std, 95% CI, effect size (d), and p-value.*
