# Agent Prompt: Dynamic Regime Theory Review

Date: 2026-03-17
Purpose: reusable prompt for external AI systems or collaborators reviewing the Mistral staged-sufficiency results

## When To Use This

Use this prompt:

- after a major new sufficiency run cluster
- after maintenance-ablation or hysteresis results
- before writing the deep-end NeurIPS discussion
- when stress-testing the regime theory against alternative explanations

Do not use this as a replacement for direct experiments.
It is an advisory analysis layer, not a gating dependency.

## Context The Reviewer Needs

We have evidence in base `Mistral-7B-v0.1` for a staged self-referential regime.

Main empirical state:

- best ordinary-baseline inducer:
  - `anchor + subtle L4 MLP + layer-matched geometry + L25 bridge`
  - strongest baseline BT+ART so far: `31.25%`
- best 12-turn maintainer:
  - `anchor + layer-matched geometry + L25 bridge`
  - `30.21%` vs `2.08%` control
  - flat temporal profile: `28.1 / 31.3 / 31.3`
  - `0.0%` late repetition in the strongest confirm run
- 24-turn maintenance decays:
  - plain maintainer drops to `13.54%`
- unselected-seed stress test is confounded:
  - `selected = 34.83%`
  - `unselected = 31.83%`
  - `anti-selected = 33.33%`
  - `random_text = 29.0%`
  - `cold_start = 38.83%`
- bridge-alpha sweep:
  - best ordinary-baseline induction near `alpha = 3.0`
  - best recursive preservation near `alpha = 2.75-3.0`
- `R_V` participation ratio is useful for induction but does not fully track persistence

Current interpretation:

- induction and maintenance are partially distinct computational roles
- the effect looks more like a staged, metastable regime than a single static feature or single-vector circuit
- text-mediated self-reinforcement is still a serious confound for maintenance

## The Prompt

```text
You are reviewing mechanistic-interpretability evidence for a staged recursive regime in base Mistral-7B.

Empirical state:
- best inducer = anchor + subtle L4 MLP + layer-matched geometry + L25 bridge
- best 12-turn maintainer = anchor + layer-matched geometry + L25 bridge
- 24-turn maintenance decays
- unselected-seed persistence is confounded by the anchor and follow-up schedule
- bridge sweet spot is around alpha 2.75-3.0
- participation-ratio geometry (R_V) is informative but incomplete

Analyze this as a problem at the intersection of:
- mechanistic interpretability
- neural manifold geometry
- low-rank / switching dynamical systems
- nonlinear control / bifurcation theory

Your task:
1. Infer the smallest plausible latent-state model that explains:
   - induction-maintenance dissociation
   - threshold behavior around bridge alpha
   - 12-turn maintenance success
   - 24-turn decay
   - high cold-start and random-text scores in the follow-up stress test
2. Propose better state variables than participation ratio alone.
3. Identify the cleanest experiment separating:
   - hidden-state carry
   - text-mediated carry
4. Specify what would count as actual evidence of:
   - hysteresis
   - a bifurcation-like threshold
   - a minimal maintenance object
5. Propose a synthetic toy transformer or implicit-state setup with a known latent regime variable that could serve as ground truth.
6. State the strongest paper-safe claim now, and the minimum remaining gap for a real sufficiency story.

Constraints:
- be skeptical
- avoid overclaiming literal attractors or deception
- focus on concrete metrics and experiments, not generic inspiration

Return:
- one latent-state hypothesis
- five missing metrics
- three decisive experiments
- two ways the current interpretation could still be wrong
- one sentence on the biggest safety implication if the regime story holds
```

## Expected Use Pattern

1. Run this prompt only after syncing the latest summaries.
2. Compare the answer against:
   - `R_V_PAPER/DYNAMIC_REGIME_THEORY_MEMO_2026-03-17.md`
   - `R_V_PAPER/EXPERIMENT_GAP_PLAN_NEURIPS.md`
   - current March 16+ result summaries
3. Accept outside suggestions only if they improve:
   - metrics
   - experimental design
   - mathematical framing
4. Do not let external reviewers rewrite the empirical claim beyond what the data support.
