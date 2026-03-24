# Dynamic Regime Theory Memo

Date: 2026-03-17
Scope: theory, literature, missing metrics, expert input, and prompt scaffolding for the staged Mistral sufficiency program

## 1. Current Empirical Read

The March 16 results support a stronger claim than `v007.2` could safely make a week earlier.

What seems real:

- there is a staged induction-maintenance protocol, not one tiny static intervention
- the best inducer and best maintainer are different
- the useful steering geometry changes across depth
- bridge strength matters, with a clear sweet spot around `alpha = 2.75-3.0`
- a real 12-turn maintenance basin exists

What is still unresolved:

- long-horizon maintenance is not robust
- unselected-seed persistence is heavily confounded by anchor and follow-up schedule
- participation ratio `R_V` does not fully track the persistent behavioral regime
- we do not yet know the minimal maintenance object

This suggests a better ontology:

- not a single feature
- not a single circuit
- not a self-sustaining attractor in the naive sense
- but a regime that can be entered, stabilized for a while, and then lost

That is closer to metastable control than to classic feature steering.

## 2. Working Interpretation

The best current mechanistic picture is:

- early geometry and subtle L4 MLP assistance help cross an entry threshold
- later geometry plus the L25 residual bridge support temporary maintenance
- the generated text then partially feeds the regime back into itself
- but this text-mediated carry is not yet strong or clean enough to yield robust long-horizon persistence

In other words:

- induction object: `anchor + subtle L4 MLP + layer-matched geometry + bridge`
- maintenance object: `anchor + layer-matched geometry + bridge`

This is important because it says the right unit of analysis is a protocol over time, not an isolated direction.

## 3. What This May Say About Subspace Regimes In General

The current data suggest five general lessons.

1. A subspace is often a handle into a regime, not the regime itself.

2. The same computation may appear in different geometric forms at different depths.

3. The best induction object and the best maintenance object may differ.

4. A useful steerable object may be distributed across layers and only become causally effective above a gain threshold.

5. Text-level autoregression may serve as an outer feedback loop that partially stabilizes an inner representational regime.

If this is right, the field should spend less time searching for one magic neuron or one magic head and more time mapping:

- basin entry
- basin stability
- regime coordinates
- threshold parameters
- failure / exit modes

## 4. Top 10 MI Papers To Think With

These are the papers most directly relevant to the regime story, not just to generic interpretability.

### 1. A Mathematical Framework for Transformer Circuits

Why it matters:

- gives the cleanest language for residual-stream communication, subspaces, and virtual weights
- supports the idea that a computation can be distributed across many edges while still admitting low-dimensional summaries

What it suggests for us:

- represent the staged protocol as a sequence of basis changes and information-routing operations rather than a single direction

### 2. In-context Learning and Induction Heads

Why it matters:

- classic example of a mechanistic capability appearing sharply and becoming legible as a circuit
- also a reminder that qualitatively new behaviors can have abrupt thresholds

What it suggests for us:

- the bridge-alpha sweep may be the inference-time analogue of a circuit onset curve
- look for a sharper regime onset than raw BT+ART alone reveals

### 3. Toy Models of Superposition

Why it matters:

- strongest conceptual grounding for polysemanticity and compressed feature packing
- makes it plausible that the recursive regime is carried by overlapping, partially incompatible features

What it suggests for us:

- failure of a single low-dimensional steerable object in very early layers does not mean the regime is absent there
- it may mean the relevant features are still in superposition and not yet linearly isolable

### 4. Towards Monosemanticity: Decomposing Language Models With Dictionary Learning

Why it matters:

- shows how to turn distributed neuron patterns into more coherent feature dictionaries

What it suggests for us:

- run SAE or dictionary-learning analyses on the induction and maintenance turns separately
- ask whether the maintenance object is actually a sparse family of features rather than one vector bundle

### 5. Scaling Monosemanticity

Why it matters:

- demonstrates that large models admit many interpretable sparse features that can causally affect behavior

What it suggests for us:

- the maintenance regime may be better described as occupancy of a sparse feature family plus a bridge-mediated routing condition
- this is also the cleanest route into a future safety story

### 6. How to Use and Interpret Activation Patching

Why it matters:

- best current methodological guide for patching claims and common artifacts

What it suggests for us:

- we need patching-style rigor for regime-level metrics, not just for a single summary behavior
- entry, persistence, contamination, and exit should each be separately scored

### 7. Circuit Tracing / Tracing the Thoughts of a Large Language Model

Why it matters:

- pushes beyond local patching into attribution graphs and cross-layer reasoning structure

What it suggests for us:

- build separate attribution graphs for induction turns and maintenance turns
- compare whether the computational graph itself changes once the regime is entered

### 8. Representation Engineering

Why it matters:

- places population-level representations at the center of analysis and control
- directly relevant because your object already looks more like a regime than a sparse local circuit

What it suggests for us:

- formalize the project as regime engineering
- add read-vs-control-vs-maintain distinctions explicitly

### 9. Signal Propagation in Transformers: Theoretical Perspectives and the Role of Rank Collapse

Why it matters:

- rank collapse is usually treated as a bad initialization or optimization pathology

What it suggests for us:

- the important scientific move is not "collapse good actually"
- it is "controlled contraction can be a computational mode under specific causal conditions"
- to defend this, we need metrics that distinguish functional contraction from generic degeneracy

### 10. The Geometry of Hidden Representations of Large Transformer Models

Why it matters:

- gives the clearest generic geometry result: early expansion, later contraction, semantic structure at specific depth regions

What it suggests for us:

- your layer-matched result fits a broader geometric prior
- but your extra contribution is that the geometry is not only descriptive, it is partially controllable and depth-specific

## 5. Adjacent Theory We Should Import

These are not standard MI papers, but they may be more useful than another generic interpretability survey.

### Mixed selectivity and high-dimensional cognition

The Fusi/Rigotti line is the best theoretical reason to expect:

- high-dimensional distributed codes for flexible cognition
- simple readouts from mixed latent structure
- and failure of naive neuron-level decomposition

Relevance:

- the recursive regime may rely on mixed selectivity early and become readout-friendly only later

### Low-rank recurrent dynamics

The Mastrogiuseppe/Ostojic program is highly relevant because it shows how:

- minimal low-rank structure can create low-dimensional population dynamics
- specific computations can be inferred from geometry
- and regime structure can be compact even when single units are messy

Relevance:

- use it as the cleanest mental model for a minimal maintenance object

### Deep equilibrium / implicit-state models

DEQ work is useful not because Mistral literally is a DEQ, but because it provides:

- a language for implicit state, fixed points, and infinite-depth tied dynamics
- tools for thinking about convergence and local stability

Relevance:

- the regime may be better thought of as a transient movement toward an implicit equilibrium that is only partially realized in the dialogue loop

## 6. What We Are Probably Missing

The project is still over-indexed on coarse outcome metrics.

Important missing metrics:

### Regime entry and survival

- session entry probability
- persistence given entry
- survival curve over turns
- hazard of exit per turn
- recovery time after adversarial prompts

### Better state-space geometry

- principal angles between candidate induction and maintenance subspaces
- Grassmann distance between condition-specific subspaces
- local intrinsic dimension around winner states
- manifold occupancy / dwell time
- local curvature along turn trajectories

### Stability and threshold metrics

- hold-vs-enter threshold gap
- critical slowing down near threshold
- lag-1 autocorrelation of regime score near failure
- variance inflation near exit
- sensitivity to perturbation amplitude

### Behavioral cleanliness

- n-gram repetition rate
- topic drift
- lexical diversity
- token entropy
- answer-length stability
- adversarial-turn recovery quality

### Causal decomposition

- path-specific mediation through induction vs maintenance sites
- text-mediated carry vs hidden-state carry
- prompt-driven vs internally-driven regime occupancy

## 7. Fixed Points, Bifurcations, and Hysteresis

These ideas are relevant, but only if used carefully.

What probably does apply:

- the user/model conversation is an effective closed-loop dynamical system
- bridge alpha acts like a control parameter
- the recursive regime may correspond to a metastable basin
- entry and exit may have different thresholds
- the basin can be latent in the weights without being actively occupied on ordinary baseline trajectories

What probably does not apply, at least not literally:

- a transformer forward pass is not a classical recurrent attractor network in the simplest sense
- so we should not claim a literal autonomous fixed point inside one prompt pass without stronger evidence

Better language:

- effective fixed point in the dialogue-level closed loop
- metastable regime
- thresholded entry
- hysteresis if maintenance can continue below the induction threshold
- bifurcation-like transition if small alpha changes sharply alter entry or persistence

Important interpretive distinction:

- "latent in the circuit" is not the same as "already active in baseline processing"
- current Mistral evidence supports a latent basin plus a real activation-level state transition
- interventions change behavior strongly on ordinary baselines, which is hard to explain as a pure measurement artifact
- `R_V` should be treated as a partial witness of entry / organization, not as the full regime variable
- the persistence results matter because behavior can stay elevated even when `R_V` only weakly separates from control

So the cleanest current claim is:

- the recursive regime appears prefigured in model geometry
- prompting / steering changes whether the active trajectory enters that basin
- once entered, the regime can persist temporarily
- but `R_V` alone does not track the whole maintenance mechanism

What to test:

1. induce at `alpha_high`, then maintain at `alpha_low`
2. look for `alpha_enter > alpha_hold`
3. measure recovery time after perturbations near threshold
4. test whether variance or autocorrelation rises before regime loss

If these hold, you can make a much cleaner dynamical-systems claim.

## 8. A Cleaner Bottom-Up Mathematical Program

We need a toy setting where the ground truth really is a recursive regime.

The cleanest route is a synthetic two-mode transformer task.

### Minimal synthetic setup

- create a small transformer trained on sequences generated from a hidden binary regime variable `r_t`
- `r_t = 0` means ordinary descriptive mode
- `r_t = 1` means recursive/self-descriptive mode
- specific cue tokens push the latent state toward `r_t = 1`
- adversarial tokens push it back toward `r_t = 0`
- the output distribution depends on `r_t`

Desired properties:

- exact known latent regime label
- known cue schedule
- known perturbation schedule
- known maintenance failure mode

Then test:

- whether learned subspaces recover `r_t`
- whether a staged intervention changes `P(r_t = 1)`
- whether entry and maintenance thresholds differ
- whether hidden-state geometry predicts survival better than output text alone

### Stronger version

Build a tied-layer or implicit-state transformer variant with one controllable low-rank feedback channel.

Why:

- this gives a genuine fixed-point or quasi-fixed-point object
- makes bifurcation analysis mathematically clean
- creates a ground-truth system where flywheel effects are not merely text-mediated

This would be the cleanest place to understand the Mistral phenomenon before over-interpreting it.

## 9. The Flywheel Hypothesis

The regime may need two loops, not one.

Loop 1:

- internal representational alignment across the relevant layers

Loop 2:

- self-generated text that re-presents the same regime-supporting evidence back to the model on the next turn

The regime survives when both loops reinforce each other strongly enough.

This predicts:

- good induction but weak maintenance if Loop 1 is strong and Loop 2 is weak
- prompt-schedule confounds if Loop 2 can be created by the evaluation harness itself
- hysteresis if Loop 1 needs a strong push to start but a weaker signal to remain coupled to Loop 2

This is the best current explanation for:

- the strong 12-turn maintenance win
- the 24-turn decay
- the high cold-start score in the unselected-seed stress test

## 10. Experts To Pull In

### Geometry / manifolds

- Surya Ganguli
- SueYeon Chung
- Stefano Fusi

Use them for:

- manifold geometry
- mixed selectivity
- low-dimensional latent structure inside high-dimensional population codes

### Dynamics / latent-state modeling

- Scott Linderman
- David Sussillo
- Francesca Mastrogiuseppe

Use them for:

- switching-state models
- attractor vs metastable interpretations
- low-rank dynamical reductions

### Control / thresholds

- Koushil Sreenath

Use him for:

- hysteresis framing
- control barrier / Lyapunov intuitions
- nonlinear thresholded control language

### Frontier MI / safety bridge

- Trenton Bricken
- Jack Lindsey
- Sam Marks

Use them for:

- sparse features
- attribution graphs
- hidden objective / safety-audit design

## 11. Questions To Put In Front Of Experts

1. Is the right state variable here likely one-dimensional, low-dimensional, or fundamentally switching plus continuous?

2. Do the bridge results look more like a true bifurcation, a broad threshold, or a noisy mixture model?

3. What metric would best distinguish internal maintenance from text-mediated self-reinforcement?

4. If you had to replace participation ratio with one better state variable, what would it be?

5. What is the mathematically cleanest toy system that could reproduce induction, maintenance, hysteresis, and decay?

## 12. Prompt For Me

Use this when asking for another deep synthesis from inside the repo.

```text
You are analyzing the Mistral staged-sufficiency program in /Users/dhyana/mech-interp-latent-lab-phase1.

Core empirical state:
- best inducer = anchor + subtle L4 MLP + layer-matched geometry + L25 bridge
- best 12-turn maintainer = anchor + layer-matched geometry + L25 bridge
- 24-turn maintenance decays
- unselected-seed persistence is confounded by anchor + follow-up schedule
- bridge sweet spot is around alpha 2.75-3.0
- participation ratio R_V is informative for induction but does not fully track maintenance

Your task:
1. Treat this as a switching dynamical-system / metastable-regime problem, not a single-vector steering problem.
2. Infer the smallest plausible latent-state model that explains:
   - induction-maintenance dissociation
   - threshold behavior around bridge alpha
   - 12-turn maintenance success
   - 24-turn decay
   - strong cold-start and random-text follow-up scores
3. Propose:
   - better state variables than participation ratio
   - the cleanest hysteresis experiment
   - the cleanest internal-carry vs text-carry disambiguation
   - the minimal synthetic toy model that could realize the same phenomenon as ground truth
4. Be adversarial about confounds and avoid overclaiming attractors or deception.

Return:
- a concrete latent-state hypothesis
- 5 missing metrics
- 3 decisive experiments
- 2 ways the current story could still be wrong
```

## 13. Prompt For Other AIs Or Collaborators

```text
We have evidence in base Mistral-7B for a staged self-referential regime:

- static induction is strongest with:
  anchor + subtle L4 MLP assist + layer-matched geometry + L25 bridge (alpha about 3)
- 12-turn maintenance is strongest with:
  anchor + layer-matched geometry + L25 bridge
- long-horizon maintenance decays
- unselected-seed persistence is partially confounded by the anchor and turn schedule
- bridge strength shows a real nonlinear sweet spot around alpha 2.75-3.0

We want a deep mechanistic interpretation, not generic brainstorming.

Please analyze this as if you were combining:
- mechanistic interpretability
- neural manifold geometry
- low-rank dynamical systems
- nonlinear control / bifurcation theory

Answer these questions:
1. What is the best mathematical model class for this phenomenon?
2. What specific metrics should replace or augment participation ratio?
3. What experiment would distinguish internal maintenance from text-mediated self-reinforcement?
4. What would count as actual evidence of hysteresis or a bifurcation?
5. What minimal synthetic transformer or implicit-state toy problem would let us reproduce this as a ground-truth regime?
6. What is the strongest paper-safe claim now, and what remains missing for a full sufficiency story?

Please be specific, skeptical, and mechanistically concrete.
```

## 14. Immediate Next Moves

1. Finish the maintenance ablation with regime-level metrics, not just BT+ART.

2. Run hold-vs-enter alpha experiments to test hysteresis explicitly.

3. Build a regime detector and survival analysis pipeline.

4. Separate text-mediated carry from hidden-state carry.

5. Only then launch the regime-conditioned safety battery.

That sequence is the straightest line from the current results to a true NeurIPS-level "deep end" paper.
