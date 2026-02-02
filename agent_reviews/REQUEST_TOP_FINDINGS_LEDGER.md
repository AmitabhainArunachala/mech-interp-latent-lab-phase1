# Multi-Agent Request: TOP FINDINGS LEDGER (DNA → CELL → ANIMAL)

Write your response as a single markdown file and save it to:

`agent_reviews/responses/YYYYMMDD__MODELNAME__TOP_FINDINGS_LEDGER.md`

Your file MUST start with this header:

```markdown
Title: TOP FINDINGS LEDGER (DNA → CELL → ANIMAL)
Date: YYYY-MM-DD
Model: <model name + version>
Repo commit: <optional if available>
Prompt bank version: <PromptLoader.version if you ran code; otherwise "not checked">
```

## Task

You are auditing a mechanistic interpretability repo. Produce a single document titled:
**“TOP FINDINGS LEDGER (DNA → CELL → ANIMAL)”**

### Hard constraints
1. Only make claims supported by artifacts in the repo. For every claim, cite exact file paths.
2. For every claim: provide **(a)** status: VERIFIED / UNCERTAIN / CONTRADICTED, **(b)** sample size N, **(c)** key stats (means, effect sizes, p-values or CIs) if available, **(d)** what exact script/config produced it.
3. Identify inconsistencies in definitions/measurement (especially R_V) across scripts; list files where definitions differ.
4. Treat “behavior/expression” as potentially heuristic; explicitly state if it is single-sample, keyword-based, seed-sensitive, etc.
5. End with a prioritized “NEXT MOVES” section: what needs more trials (N), what needs 3× replication (seed runs), what needs a new control, what is a dead end.

### Deliverables

#### A) Canonical measurement contract check (DNA)
- What is R_V exactly? (formula, early layer, late layer, window size, PR definition, numerical stability)
- Is it implemented the same way everywhere? If not: list variants + which results depend on which variant.
- What are canonical generation parameters (temperature, do_sample, max_new_tokens) used in behavior tests?

#### B) Top 10–15 core findings ledger (sorted by leverage/importance)
For each finding include:
- Claim (one sentence)
- Scale tag: DNA / CELL / ORGAN / ANIMAL
- Status: VERIFIED / UNCERTAIN / CONTRADICTED
- Evidence: file paths to CSV/JSON/MD + scripts/configs
- Stats: N, means, deltas, effect sizes, p-values/CIs
- Replication: how many independent runs? same prompt-bank hash? different seeds?
- Confounds handled: length, keyword contamination, wrong-layer control, shuffled/random control, opposite control (if relevant)
- What would falsify it?

#### C) Layer story (CELL)
- Where does contraction begin? gradual vs sharp transition? peak layer(s)?
- Clarify whether “L27” is “num_layers-5” or a fixed index, and how it shifts across models.

#### D) Head/circuit story (ORGAN)
- What heads/KV-groups are implicated? Is GQA aliasing correctly accounted for?
- Which interventions are causal vs correlational (ablation, patching, KV transfer, wrong-layer controls)?

#### E) Behavior/attractor/one-way-door story (ANIMAL)
- Multi-token persistence: what’s the best evidence? N? thresholds?
- Hysteresis / one-way door: what’s verified vs aspirational?
- KV cache transfer: which claims held up? which failed? what was the confound?

#### F) Next moves (ranked)
- Propose 3–5 canonical pipelines that should exist (or be consolidated), and which findings each pipeline reproduces.
- Minimal “gold standard suite” to run 3× (seeds) with pass/fail thresholds.

### Output format
- Use headings
- Use bullet lists for the findings ledger
- Include file paths inline for every evidence item
- No speculation beyond what the repo supports


