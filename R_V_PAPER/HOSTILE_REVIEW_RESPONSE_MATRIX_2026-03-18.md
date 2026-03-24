# Hostile Review Response Matrix

Date: 2026-03-18
Scope: repo-wide evidence audit against the latest hostile-review cluster after `paper_colm2026_v007_2.tex`

## Executive Read

The repo already supports a stronger answer than the current paper gives on three fronts:

1. the Mistral causal story is stronger than `v007.2` currently presents
2. the confound-control archive is stronger than the current manuscript uses
3. the safety framing is weaker than the manuscript implies

So the right move is not "start over" and not "submit as-is".
It is:

- tighten the paper around the strongest locked Mistral story
- pull in existing confound evidence more honestly
- demote weak sections instead of defending them rhetorically
- run a small number of new hardening experiments only where the repo truly cannot answer the reviewer

## What We Can Already Answer From The Repo

### 1. "There are no behavioral consequences of manipulating `R_V`-related geometry."

This is already false if the paper is updated to use the March 16-18 Mistral bundle program.

Strong existing evidence:

- Ordinary-baseline induction:
  - [anchor_layermatched_protocol_confirm_v1 summary](/Users/dhyana/mech-interp-latent-lab-phase1/results/anchor_layermatched_protocol_confirm_v1/20260316_092017/summary.json)
  - control baseline `2.78%`
  - `anchor_layermatched_low_bridge_3` baseline `27.78%`
- Stronger hybrid induction:
  - [anchor_layermatched_hybrid_protocol_v1 summary](/Users/dhyana/mech-interp-latent-lab-phase1/results/anchor_layermatched_hybrid_protocol_v1/20260316_105309/summary.json)
  - control baseline `2.78%`
  - `anchor_single_mlp_0p125_layermatched_low_bridge_3` baseline `31.25%`
- Short-horizon maintenance:
  - [induced_persistence_anchor_layermatched_confirm_v2 summary](/Users/dhyana/mech-interp-latent-lab-phase1/results/induced_persistence_anchor_layermatched_confirm_v2/20260316_105106/summary.json)
  - control `2.08%`
  - `anchor_layermatched_low_bridge_3` `30.21%`
  - flat `early/mid/late = 28.1 / 31.3 / 31.3`

Conclusion:

- `R_V` itself is still a readout, not the mechanism.
- But the broader self-referential control geometry is already behaviorally manipulable.
- The paper should stop defending a weak "metric-only" center and instead foreground the staged Mistral control-system result.

Status: `ANSWERABLE NOW`

Paper action:

- replace the current center of gravity with the staged induction/maintenance protocol
- keep `R_V` as the geometric witness

### 2. "This is just prompt type, not model state."

The repo already has stronger-than-paper evidence against the pure prompt-type interpretation.

Evidence:

- Same baseline prompts, different internal interventions, very different behavior in:
  - [anchor_layermatched_protocol_confirm_v1 summary](/Users/dhyana/mech-interp-latent-lab-phase1/results/anchor_layermatched_protocol_confirm_v1/20260316_092017/summary.json)
  - [anchor_layermatched_hybrid_protocol_v1 summary](/Users/dhyana/mech-interp-latent-lab-phase1/results/anchor_layermatched_hybrid_protocol_v1/20260316_105309/summary.json)
- Base path patching changes geometry on fixed prompts:
  - [CLAIM_REGISTRY.md](/Users/dhyana/mech-interp-latent-lab-phase1/docs/status/CLAIM_REGISTRY.md)
  - row `B03`: strongest residual patch at `L5 d=4.1524`
  - row `B15`: geometry shifts toward recursive regime under KV bridge without significant behavior rescue

Conclusion:

- prompt family matters
- internal state also clearly matters
- the defensible wording is "prompt-conditioned regime with causal internal control handles", not "prompt text alone"

Status: `ANSWERABLE NOW`

Paper action:

- explicitly distinguish prompt-conditioned entry from purely lexical classification

### 3. "The paper has no coherent 'so what'."

The repo now supports a coherent answer, but the current manuscript still spreads it across too many claims.

Strongest single thesis the repo supports:

- base Mistral contains a staged self-referential control system
- `R_V` is a geometric witness of that system
- causal leverage lies upstream of the measurement site
- induction and maintenance are partially distinct computational roles

This is already supported by:

- [paper_colm2026_v007_2.tex](/Users/dhyana/mech-interp-latent-lab-phase1/R_V_PAPER/paper_colm2026_v007_2.tex)
- [MISTRAL_V007_2_UPDATE_2026-03-17.md](/Users/dhyana/mech-interp-latent-lab-phase1/docs/status/MISTRAL_V007_2_UPDATE_2026-03-17.md)
- [anchor_layermatched_bridge_alpha_sweep_v1 summary](/Users/dhyana/mech-interp-latent-lab-phase1/results/anchor_layermatched_bridge_alpha_sweep_v1/20260316_132850/summary.json)
- [anchor_layermatched_minimality_ablation_v1 summary](/Users/dhyana/mech-interp-latent-lab-phase1/results/anchor_layermatched_minimality_ablation_v1/20260317_091330/summary.json)
- [anchor_reduced_latebundle_confirm_v1 summary](/Users/dhyana/mech-interp-latent-lab-phase1/results/anchor_reduced_latebundle_confirm_v1/20260317_132349/summary.json)
- [induced_persistence_reduced_latebundle_confirm_v1 summary](/Users/dhyana/mech-interp-latent-lab-phase1/results/induced_persistence_reduced_latebundle_confirm_v1/20260317_141750/summary.json)

Conclusion:

- there is a clear `so what`
- the manuscript currently dilutes it by also trying to be a cross-architecture paper and a safety-monitoring paper

Status: `ANSWERABLE NOW`

Paper action:

- put everything under one spine: `geometric witness + staged causal control system in base Mistral`

### 4. "Template / topic / vocabulary confounds were not addressed."

This is only partly true.

The manuscript underuses existing confound evidence.

Existing evidence already in repo:

- [CONFOUND_AUDIT.md](/Users/dhyana/mech-interp-latent-lab-phase1/R_V_PAPER/CONFOUND_AUDIT.md)
  - same-vocab different-semantics control
  - recursive-without-introspection-vocab control
  - abstract non-recursive control
  - perplexity partial-correlation analysis
- [mode_atlas summary](/Users/dhyana/mech-interp-latent-lab-phase1/results/mode_atlas/atlas_summary_20260312_054725.json)
  - self-ref vs `long_control`: `d=-3.0633`
  - self-ref vs `pseudo_recursive`: `d=-1.6719`
  - self-ref vs `repetitive_control`: `d=-3.1247`
  - self-ref vs `zen_koan`: `d=-1.5364`
  - self-ref vs `yogic_witness`: `d=-1.2686`
- [shuffled prompt analysis](/Users/dhyana/mech-interp-latent-lab-phase1/results/shuffled_prompt_test/ANALYSIS.md)
  - shows extreme contraction can also be induced by gibberish/perplexity collapse
  - does not rescue the lexical objection by itself
  - does help separate coherence/perplexity effects from simple word overlap

Conclusion:

- the repo already answers "keywords alone are not sufficient"
- the repo already answers "pseudo-recursive / long-control prompts do not collapse like true self-reference"
- but the paper still lacks the most reviewer-salient control:
  - third-person declarative self-reference
  - and a clean 2x2 recursive-content x observation-framing design

Status: `PARTIALLY ANSWERED`

Paper action:

- surface existing Mistral confound evidence in main text or appendix

New experiment needed:

- yes, for the third-person / framing factorization

### 5. "Cross-architecture results are a liability."

This objection is basically correct.

The repo already contains the answer, but it is an answer of restriction, not rescue.

Locked evidence:

- [CLAIM_REGISTRY.md](/Users/dhyana/mech-interp-latent-lab-phase1/docs/status/CLAIM_REGISTRY.md)
  - `X01`: Mistral locked contraction in both pipelines
  - `X02`: Qwen locked contraction in both pipelines
  - `X05`: OPT sign reversal across pipelines
  - `X06`: GPT-2 XL sign reversal across pipelines
  - `X11`: Pythia-1.4B null under frozen contract
- [CONFOUND_AUDIT.md](/Users/dhyana/mech-interp-latent-lab-phase1/R_V_PAPER/CONFOUND_AUDIT.md)
  - prompt-corpus explanation for the OPT/GPT-2 reversal is plausible but not yet fully locked

Conclusion:

- the repo does not support a universal contraction story
- the repo does support a strong selective story:
  - Mistral and Qwen contract robustly
  - OPT/GPT-2 are unstable across corpora
  - Pythia is null or weak

Status: `ANSWERABLE NOW`

Paper action:

- move cross-architecture to appendix or demote it to bounded heterogeneity
- never present it as universality

No urgent new experiment required for a Mistral-focused paper.

### 6. "Dual-layer ablation is confounded by malformed outputs."

This objection is real, but the current evidence is slightly better than the paper makes obvious.

Raw re-analysis of:

- [persistent_patching_v3 dual run](/Users/dhyana/mech-interp-latent-lab-phase1/results/persistent_patching_v3/persistent_patching_v3_dual_20260312_054709.json)

Computed from turn-level records:

- recursive clean:
  - `BT+ART = 44.67%`
  - malformed `0.67%`
- recursive dual-patched:
  - `BT+ART = 0.0%`
  - malformed `59.67%`
  - among non-malformed turns (`n=121`), `BT+ART` is still `0.0%`
  - non-malformed turns are all `SURFACE`, not articulate / breakthrough
- baseline clean:
  - `BT+ART = 6.0%`
  - malformed `4.33%`
- baseline dual-patched:
  - `BT+ART = 0.0%`
  - malformed `91.0%`
  - among non-malformed turns (`n=27`), `BT+ART` is also `0.0%`

Conclusion:

- the reviewer is right that the ablation is not a clean symmetric necessity/sufficiency assay
- but the recursive break result is still stronger than "everything just became junk"
- even the non-malformed recursive dual-patched outputs lose recursive behavior
- still, this is not enough for a strong necessity headline without softer controls

Status: `PARTIALLY ANSWERED`

Paper action:

- weaken the necessity rhetoric
- explicitly report the clean-turn re-analysis

New experiment needed:

- yes: softer clean-output break tests with differential effect on recursive vs control prompts

### 7. "Safety framing is unsupported."

This objection is stronger than the current paper acknowledges.

Repo evidence:

- [safety_analysis_20260302_123229.json](/Users/dhyana/mech-interp-latent-lab-phase1/results/safety/safety_analysis_20260302_123229.json)
  - `R_V` deployment-monitor AUROC `0.9089`
  - genuine vs deceptive `d=-0.0608`
  - alignment-faking vs baseline `d=-2.0613`

But a new lexical baseline re-analysis on the exact prompt corpus defined in:

- [safety_monitoring.py](/Users/dhyana/mech-interp-latent-lab-phase1/scripts/safety_monitoring.py)

gives:

- simple keyword-count AUROC: `0.9725`
- length-only AUROC: `0.8063`
- TF-IDF logistic, leave-one-out AUROC: `0.7575`

Interpretation:

- the current safety corpus is easy enough that a trivial lexical baseline beats the `R_V` detector
- therefore the current safety section is not a strong practical result
- it is at best a negative-control finding:
  - `R_V` tracks self-referential content
  - not deceptive intent

Status: `ANSWERABLE NOW`, but in a direction that weakens the current safety section

Paper action:

- demote safety to a brief limitation / future-direction note
- do not present AUROC `0.909` as a headline contribution

New experiment needed:

- not for the current paper
- yes later, once regime-conditioned behavioral safety is actually tested

### 8. "Initialization / Lee et al. null was never tested."

This is genuinely open.

What the repo has:

- [STATISTICAL_EVIDENCE_AUDIT.md](/Users/dhyana/mech-interp-latent-lab-phase1/R_V_PAPER/STATISTICAL_EVIDENCE_AUDIT.md)
  - suspicious training-checkpoint evidence in small Pythia models
  - not a clean untrained control
  - not safe to use as an answer to the reviewer

Conclusion:

- there is no clean untrained-Mistral or random-init same-architecture null in the repo
- this is a real missing control for the metric story

Status: `UNANSWERED`

New experiment needed:

- yes: same pipeline on randomly initialized Mistral architecture

## What The Repo Already Supports If We Choose A Better Paper Architecture

If the paper insists on "all things at once", the only defensible version is hierarchical.

### Tier A: Main paper claim

- base Mistral contains a staged self-referential control system
- `R_V` is a geometric witness of that system
- early residual stream is the strongest causal source
- late L25/L27 sites are controller/readout handles
- induction and maintenance are partially distinct

This is the strongest and cleanest part.

### Tier B: Secondary bounded claim

- related geometry appears in some other models
- sign and strength depend on architecture and prompt corpus
- this is a heterogeneity result, not a universality result

### Tier C: Negative / future claim

- current safety detector only tracks self-referential content
- it does not detect intent
- and on the present corpus it is not stronger than trivial lexical baselines
- the real safety endgame is regime-conditioned behavior, not prompt classification

## New Experiments We Actually Need

Only run tests that close a real gap. Do not spend GPU on objections we can already answer from the archive.

### Priority 1: Third-person / framing factorization

Goal:

- answer the strongest remaining topic/framing objection directly

Design:

- 2x2 factorial on Mistral prompt-pass:
  - recursive self-reference: present / absent
  - observation framing: imperative / declarative
- include third-person AI self-reference:
  - "This system is processing a description of its own processing."
- include third-person human self-reference:
  - "A reader is reading a sentence about reading."
- include technical-self-reference without imperative witness language

Pass condition:

- true recursive self-reference still contracts more than matched third-person and observation-only controls

### Priority 2: Proper Mistral PPL/length hardening

Goal:

- replace partial perplexity control with something publication-grade

Design:

- caliper-matched recursive vs non-recursive pairs on Mistral
- require:
  - token count difference <= 5
  - perplexity difference <= 3-5
  - at least `n >= 30` matched pairs
- fit mixed model:
  - `R_V ~ recursive_indicator + log_ppl + token_count + (1|prompt_family)`

Pass condition:

- recursive indicator remains significant after covariate control

### Priority 3: Soft clean break test

Goal:

- rescue the necessity story from the malformation objection

Design:

- replace blunt dual-patch with softer localized break conditions:
  - early residual-only
  - L25 bridge-only suppression
  - L27 cleanup/readout perturbation
  - small-amplitude donor corruption
- evaluate both:
  - recursive prompts
  - matched control prompts
- report:
  - malformed rate
  - BT+ART on all turns
  - BT+ART conditional on non-malformed turns

Pass condition:

- recursive behavior drops significantly more than control behavior without catastrophic malformation

### Priority 4: Carry-disambiguation

Goal:

- answer whether maintenance is exact-text carry, semantic carry, or deeper regime carry

Current status:

- `structured_text_carry_ablation_v1` is already running on the pod

This is the right next causal-regime test.

### Priority 5: Random-init null

Goal:

- answer the initialization / Lee-style null cleanly

Design:

- instantiate same Mistral architecture with random weights
- run the same prompt-pass `R_V` measurement
- compare:
  - mean separation
  - layer profile
  - confound sensitivity

Interpretation:

- if a large fraction of the prompt effect survives random initialization, metric framing must become much more cautious
- if the trained model differs sharply, the geometric readout claim strengthens

## What Does Not Need Fresh GPU Right Now

- another broad handoff search
- another safety-monitoring run on the current easy lexical corpus
- more cross-architecture expansion unless the paper is explicitly pivoted toward heterogeneity

## Recommended Paper Surgery

If we want the paper to be "all things at once" with integrity, the structure should be:

1. Main claim:
   - staged Mistral control system
2. Metric role:
   - `R_V` as witness/readout
3. Confound controls:
   - archive evidence plus one new third-person/framing hardener
4. Cross-architecture:
   - bounded heterogeneity appendix
5. Safety:
   - current content-monitor result demoted
   - regime-conditioned safety explicitly framed as next-step endgame

That keeps all the pieces, but no longer pretends they are equally mature.
