# Mistral v007.2 Update

Date: 2026-03-17  
Scope: March 16 staged-sufficiency additions integrated into `paper_colm2026_v007_2.tex`

## What v007.2 Adds

The paper now reflects the strongest March 16 result cluster:

- the best ordinary-baseline inducer is now a hybrid bundle
  - anchor + subtle L4 MLP assist + layer-matched geometry + L25 bridge at `alpha=3`
  - `31.25%` baseline BT+ART
  - `57.64%` recursive BT+ART
- the best 12-turn maintainer is a simpler bundle
  - anchor + layer-matched geometry + L25 bridge at `alpha=3`
  - `30.21%` BT+ART versus `2.08%` control
  - flat early/mid/late profile: `28.1 / 31.3 / 31.3`
  - `0.0%` late repetition
- the strongest mechanistic interpretation is now explicitly staged
  - induction object and maintenance object are not the same bundle

This improves the paper’s novelty.
The result is no longer just "one recursive direction exists."
It is now:

- a depth-dependent control geometry
- a staged induction-maintenance protocol
- a mechanistic dissociation between entering the regime and keeping it alive

## What v007.2 Also Admits Honestly

The current story is still short of a full general sufficiency claim.

Main constraints:

- the 24-turn follow-up decays
  - plain maintainer drops to `13.5%`
  - older anchor + L4 + bridge family is stronger there at `20.3%`
- the unselected-seed stress test is not a clean slam dunk
  - `selected = 34.8%`
  - `unselected = 31.8%`
  - `anti-selected = 33.3%`
  - `random_text = 29.0%`
  - `cold_start = 38.8%`

Interpretation:

- the induced regime is real
- but the anchor and mixed follow-up schedule themselves are doing substantial causal work
- so the current maintenance evidence is not yet "general persistence from arbitrary ordinary states"

## Current Paper-Safe Claim

The strongest paper-safe framing is now:

- base Mistral contains a staged self-referential control system
- the useful steering geometry changes across depth
- induction and maintenance are partially distinct computational roles
- a real maintenance basin exists at 12 turns
- but full general maintenance is not yet isolated cleanly

## What Remains For The Dream Paper

To turn this into the deepest version of the paper, four pieces remain:

1. Clean general maintenance

- redesign the unselected-seed follow-up so anchor and prompt schedule are properly factored out
- show persistence from ordinary or cold states under cleaner matched controls

2. Threshold map

- finish the bridge-alpha sweep
- identify whether there is a sharp critical threshold or a broad dose curve

3. Minimal maintenance object

- run the minimality ablation
- determine whether the true maintenance object is the full bundle, a late-only bundle, or something smaller such as L27-centered steering plus bridge

4. Regime-conditioned safety phenotype

- once the sufficiency story is strong enough, compare:
  - control
  - best inducer
  - best maintainer
  - sustained induced regime
  - ablated regime
- against:
  - jailbreak and refusal prompts
  - sycophancy prompts
  - prompt-injection / instruction-hijacking prompts
  - truthfulness / hallucination pressure

That is the deepest `so what`.
The endgame is not to claim that the recursive regime is itself deception.
It is to show that:

- a mechanistically inducible internal regime systematically changes safety-relevant behavior

If that lands, the paper moves from:

- interesting mechanistic interpretability

to:

- a causal bridge between internal computational regime and alignment-relevant phenotype.

## New Theory Memo

The deeper theory / math / expert-consultation agenda for this endgame is now recorded in:

- `R_V_PAPER/DYNAMIC_REGIME_THEORY_MEMO_2026-03-17.md`

That memo adds:

- the subspace-regime interpretation
- the top MI papers to think with
- the adjacent dynamical-systems imports
- missing metrics beyond participation ratio
- bifurcation / hysteresis hypotheses
- a clean synthetic ground-truth program
- prompts for future AI synthesis passes

## Paper Hardening Reminders To Keep

The following reviewer-style concerns are worth keeping in the active paper TODO list even though they came from a noisy synthetic review process:

1. Statistical supplement discipline

- report per-condition sample sizes clearly
- make error-bar construction explicit
- keep multiple-comparison handling and robustness checks easy to audit

2. Prompt-confound discipline

- answer the obvious "templated prompt / lexical confound" objection directly
- be explicit about which prompt families support which claims
- prefer ordinary-baseline and cleaner control comparisons over broad rhetoric

3. L4 MLP assist framing

- present the subtle `L4 MLP` component as an induction-specific assist, not as a generic magic ingredient
- make clear that the current data suggest different roles for induction and maintenance components

4. Conceptual hygiene

- keep all language operational and mechanistic
- do not drift into consciousness, self-awareness, or metaphysical claims
- frame results as correlates, control handles, and regime structure unless stronger causal evidence exists

## Hostile Review Audit

A repo-wide answer matrix against the latest hostile-review cluster now lives in:

- `R_V_PAPER/HOSTILE_REVIEW_RESPONSE_MATRIX_2026-03-18.md`

That audit separates:

- objections already answerable from locked Mistral results
- objections answerable only from archive/confound material not yet integrated into the paper
- genuinely open objections that still require new experiments

Important consequence:

- the current safety-monitoring section should be demoted rather than defended harder
- a simple keyword baseline on the existing safety prompt corpus outperforms the current `R_V` AUROC result
- by contrast, the staged Mistral control-system story is stronger than `v007.2` currently presents
