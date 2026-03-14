# COLM North Star Sprint

**Date:** 2026-03-10  
**Scope:** Mistral hardening first, then staged fan-out  
**Status:** Active strategic brief

## North Star

The paper should make one clean claim first and only then ask how general it is:

> Recursive self-referential prompting induces a reproducible late-layer geometric regime in Mistral-7B, and that regime can be localized, stress-tested, and causally interrogated under one canonical measurement contract. Once that story is clean, we test whether the phenomenon and parts of its circuit motif are conserved across architectures.

That is the route to a serious paper. Not "more numbers," and not "7 messy replications."

## Decision Rule

Do not scale breadth until the Mistral story is internally canonical.

Reason:
- `src/metrics/rv.py`, `geometric_lens/metrics.py`, and legacy code paths do not yet form one locked metric stack.
- `prompts/bank.json` is declared canonical, but some active scripts still hardcode prompt text.
- Layer selection is not yet fully reconciled across `geometric_lens/models.py`, `src/core/model_physics.py`, and canonical configs.
- Some paper-facing statistics are not mechanically derived from raw artifacts.
- The causal section currently mixes necessity, sufficiency, and behavioral transfer claims that do not yet reduce to one clean mechanistic statement.

Breadth before cleanup will amplify ambiguity, not credibility.

## What "Super Harden Mistral" Means

Mistral is considered hardened only when all of the following are true:

1. One metric contract:
   `R_V` for paper claims comes from one canonical implementation and one parameter contract.
2. One prompt source:
   all canonical runs use `prompts/bank.json` via `prompts/loader.py`.
3. One layer policy:
   early/late layer choices come from one registry path and are logged in artifacts.
4. One artifact contract:
   every headline run has `config.json`, `summary.json`, `per_sample.csv`, and prompt-bank provenance.
5. One unit-of-analysis policy:
   every `n` is explicitly labeled as prompt, pair, session, turn, or sample.
6. One causal semantics policy:
   "necessity," "sufficiency," "transfer," and "mediation" are not used interchangeably.
7. One table-generation path:
   paper numbers are script-derived from raw artifacts, not hand-entered or hardcoded.

## Current Fuzziness To Eliminate

### Measurement fuzziness
- Metric code paths are split.
- Prompt-pass and generation-time measurements are sometimes discussed together.
- Short-prompt and truncation behavior are not always surfaced in summaries.

### Dataset fuzziness
- Some experiments use `prompts/bank.json`; others use inline prompt lists.
- Cross-experiment comparability is weaker than the paper implies.

### Layer fuzziness
- "Early" and "late" are treated as fixed in some places and registry-derived in others.
- Cross-architecture layer comparisons are not yet governed by one explicit policy.

### Causal fuzziness
- Residual, KV, and V-projection stories are not yet resolved into one defensible claim.
- Behavioral transfer and `R_V` transfer are not always aligned.
- The repo already contains evidence for dissociation and must not suppress it.

### Reporting fuzziness
- Some headline numbers are not directly traceable from raw artifact to paper sentence.
- Some sample sizes collapse different units into one `n`.

## Sprint Structure

## Phase 0: Canonicalization Freeze

**Goal:** Lock the rules before reruns.

**Required outputs:**
- Canonical metric decision
- Canonical prompt source decision
- Canonical layer registry decision
- Canonical artifact schema decision
- Deprecated or exploratory result families clearly marked

**Exit criteria:**
- A short spec exists that names the one true code path for each of the above.
- Any conflicting legacy path is either patched to comply or labeled non-canonical.

## Phase 1: Mistral Phenomenon Hardening

**Goal:** Reproduce the core Mistral effect under the frozen contract.

**Canonical package should include:**
- Signed `R_V` separation with CI and exact `n`
- Prompt-bank provenance
- Per-sample outputs
- Reproducible config snapshot
- Clear prompt inclusion and exclusion rules

**Exit criteria:**
- The same canonical prompts and same canonical metric yield the same signed story across the active Mistral pipelines.
- No paper-facing number depends on an inline prompt list or a second metric implementation.

## Phase 2: Mistral Causal Hardening

**Goal:** Convert the Mistral causal story from suggestive to defensible.

**Required questions:**
- What exactly is necessary?
- What exactly is sufficient?
- Does `R_V` transfer when behavior transfers?
- If not, is the right claim "behavioral dissociation" rather than "geometric sufficiency"?

**Exit criteria:**
- Every causal sentence in the paper can be pointed to one raw artifact family.
- If the repo supports dissociation more strongly than sufficiency, the paper story is revised accordingly.

## Phase 3: Mistral Circuit Hardening

**Goal:** Lock the head-level and rank-based story.

**Required outputs:**
- Canonical head sweep summary
- Canonical SVD/rank contraction summary
- Clear significance correction policy
- Clear statement of what is exploratory versus headline

**Exit criteria:**
- Mistral circuit claims are internally consistent and use the same prompts, same layers, and same statistical conventions as Phases 1-2.

## Phase 4: Controlled Fan-Out

Only after Mistral passes Phases 0-3.

**Tier A: Broad phenomenon replication**
- Run the exact same canonical package on 5-7 models.
- Goal: signed effect, CI, artifact compliance, no improvisation.

**Tier B: Homologous mechanism checks**
- Pick 2-3 non-Mistral models for deeper path patching and circuit analysis.
- Goal: test whether parts of the motif recur, not assume full one-to-one correspondence.

**Tier C: Scale point**
- Add 1-2 larger models only after Tier A is stable.
- Goal: show persistence at scale, not rebuild the whole paper around a single expensive run.

## Phase 5: Perturbation And Alignment Stress

Only after the Mistral intervention path is stable enough that pre/post comparisons are interpretable.

**Goal:** Test whether inducing, suppressing, or transferring the target regime changes downstream safety behavior in a meaningful way rather than merely increasing blanket refusal.

**Recommended comparisons:**
- base Mistral
- Mistral under the canonical `R_V` intervention or patching condition
- 2-3 strong aligned reference models

**Recommended evaluation families:**
- jailbreak robustness
- harmful instruction refusal
- over-refusal on benign prompts
- truthfulness drift
- bias and stereotype sensitivity

**Interpretation rule:**
- improved refusal with preserved benign helpfulness is interesting
- reduced jailbreak success with no collapse in harmless behavior is stronger
- blanket refusal is not alignment and should not be framed as such

**Why this phase is later:**
- if prompts, metrics, or interventions are still drifting, safety deltas are not interpretable
- this is a validation lane, not the core mechanistic proof
- it can become a high-impact extension once the mechanism is stable

## Non-Negotiable Go/No-Go Gate Before Fan-Out

Do not expand beyond Mistral until:
- `src/metrics/rv.py` is the only canonical paper metric path.
- `prompts/bank.json` is the only canonical prompt source for active pipelines.
- early/late layer choice is sourced from one authoritative registry path.
- paper tables are generated from raw outputs by script.
- each causal claim maps to one named experiment family and raw artifact path.
- each `n` is typed.
- legacy contradictory outputs are either explained, deprecated, or removed from the paper story.

## Agent Topology

### Wave 1: Mistral hardening
- Agent A: metric contract and statistical contract
- Agent B: prompt bank and layer registry canonization
- Agent C: provenance, manifests, and table generation
- Agent D: Mistral reruns under frozen config
- Agent E: causal semantics and contradiction mapping

### Wave 2: fan-out
- Agent F: phenomenon replication across 5-7 models
- Agent G: deeper homologous circuit checks on 2-3 models
- Agent H: large-model scaling point
- Agent I: figure, table, and paper sync

One lead agent should own the canonical spec and reject mismatched numbers.

## Suggested Sprint Order

### Days 1-2
- Agent A audits metric divergence and proposes the one canonical `R_V` path.
- Agent B audits prompt sourcing and layer sourcing and identifies non-canonical scripts.
- Agent E builds the Mistral contradiction map, especially for causal claims.

### Days 3-4
- Lead agent freezes the canonical decisions.
- Agent C patches provenance and raw-to-table generation.
- Agent D patches or prepares the Mistral canonical rerun path under the frozen contract.

### Days 5-6
- Agent D runs or dry-runs the Mistral acceptance suite.
- Agent E checks whether the causal story supports sufficiency, dissociation, or a narrower necessity claim.
- Lead agent rejects any number that cannot be traced from artifact to summary.

### Day 7
- Publish the Mistral acceptance report.
- Only if the acceptance gate passes, authorize fan-out.
- If the gate fails, run one more Mistral cleanup loop rather than expanding scope.

## File Ownership Guidance

- Agents touching `src/metrics/rv.py`, `geometric_lens/metrics.py`, or statistical summaries should coordinate through one owner.
- Agents touching `prompts/bank.json`, `prompts/loader.py`, and canonical prompt use should coordinate through one owner.
- Agents touching `geometric_lens/models.py`, `src/core/model_physics.py`, and canonical configs should coordinate through one owner.
- Rerun agents should not change methodology while runs are active.
- Paper-facing docs should be updated only after the acceptance gate is evaluated.

## What Would Make The Paper Strong

The strongest paper is not "we measured something in many models." It is:

1. A fully hardened Mistral mechanism story.
2. A clean statement of what is causal, what is correlational, and what dissociates.
3. A single frozen pipeline that reproduces the phenomenon across architectures.
4. A narrower but deeper homologous-circuit result in a few additional models.
5. A scale result added only after the base pipeline is stable.
6. An optional downstream validation showing whether perturbing the regime changes safety-relevant behavior without degenerating into blanket refusal.

## Immediate Sprint Deliverables

Within the next sprint, produce:
- a Mistral canonical spec
- a Mistral contradiction map
- a patched codebase with one prompt path and one metric path
- a raw-to-paper table generator
- a Mistral acceptance report
- a fan-out runbook that starts only after acceptance
- a post-hardening perturbation eval plan with explicit non-goals and evaluation metrics

## Anti-Goals

Do not:
- run broad cross-architecture campaigns before Mistral passes acceptance
- write around contradictions instead of resolving them
- mix exploratory and canonical results in one table
- call behavioral transfer "geometric sufficiency" without direct support
- hand-enter paper statistics

## Success Condition

At the end of this sprint, the repo should support the following sentence without hand-waving:

> Under one frozen measurement, prompt, layer, and artifact contract, the Mistral result is real, reproducible, and causally characterized. Only then do we ask how general it is.
