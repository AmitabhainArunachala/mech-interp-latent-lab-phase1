# Simulated COLM 2026 Reviews

**Paper**: "Geometric Signatures of Self-Referential Processing in Transformer Representations"
**Simulated**: 2026-03-11
**Purpose**: Pre-submission adversarial review to identify and fix weaknesses before March 26 abstract / March 31 full paper deadline.

---

## Reviewer 1: The Statistician

**Expertise**: Experimental design, multiple comparisons, effect size estimation, replication methodology
**Score**: 5/10 (Borderline Reject)

### Summary (200 words)

The paper proposes R_V, a spectral ratio metric, as a geometric signature of self-referential processing. The primary effect in Mistral-7B (d=-2.26) is large and the double dissociation design is a genuine strength. However, the statistical methodology has several concerning issues.

The FDR correction is applied to 39 "planned comparisons," but the selection of these 39 tests appears post-hoc -- the authors have run many more comparisons across their experimental program and selected this subset for correction. This is garden-of-forking-paths territory. The mode atlas uses n=20 per mode, which is underpowered for the six-way comparison shown. The paper reports d_std=0.0 for multi-seed reproducibility and frames it as "perfect reproducibility" -- but this actually reveals the pipeline is deterministic, meaning the reported effect sizes contain zero variance from seed selection, which makes them look artificially precise. The bootstrap CI [-2.11, -1.21] is suspiciously wide given n=151 per group; this deserves explanation. Cluster-robust SEs show ICC=0.38, meaning 38% of variance is between-template, yet the authors downplay this by reporting SE inflation=1.0x at the model level (a different clustering variable). The path patching uses n=20 per condition across 96 total patches (32x3), which is both underpowered per-patch and a massive multiple comparisons problem that receives no correction.

### Top 3 Weaknesses

**W1: Post-hoc test selection masquerading as planned comparisons.**
The paper applies BH correction to "39 planned comparisons" and reports 32/39 survive. But these 39 were clearly selected from a larger experimental program. The SKILL.md reference mentions "cross-arch pipeline," "power-up pipeline," "Phase 1 pipeline," "Phase 3 bridge" -- all different experimental runs. The 39 tests appear to be the ones that were run, not tests that were pre-registered before data collection. True FDR correction requires specifying the family of tests *before* seeing results. If the actual family includes all tests ever computed (sign resolution experiments, Pythia checkpoint sweeps, scaling regressions, etc.), the effective correction is much weaker than reported.

**W2: Template clustering is underaddressed.**
ICC=0.38 is substantial -- more than a third of variance is within-template. The paper reports "SE inflation=1.0x" but this is computed at the *model* clustering level, not the *template* clustering level. At the template level, DEFF=3.67 means the effective sample size is roughly n/3.67. The paper buries this in the appendix and leads with the uncorrected d=-2.26 everywhere. Under DEFF=3.67 correction, 3 of 13 core effects become non-significant -- which effects are these? Are any of them in the cross-architecture table? This needs to be in the main text, not hidden.

**W3: Path patching is underpowered and uncorrected.**
96 patches (32 layers x 3 components), each tested at n=20. At d=0.5 (a medium effect), power at n=20 is approximately 0.33 -- meaning the paper has a 67% chance of *missing* a medium causal effect at any given patch. The paper then concludes "V-projection patching is negligible (d<=0.77)" based on this underpowered design. A non-significant result at n=20 does not mean the effect is absent. Furthermore, no multiple comparison correction is applied to the 96 patch tests, yet the paper draws strong negative conclusions ("V-projection has limited causal influence") from null results in this uncorrected, underpowered design.

### Specific Questions

1. **Were the 39 FDR tests pre-registered, or selected after seeing which experiments were run?** If the latter, what is the total number of statistical tests computed across your entire experimental program? Apply BH to that number and report how many survive.

2. **Table 2 (cross-architecture): Which effects survive under DEFF=3.67 cluster correction?** The paper says "10 of 13 core effects remain significant" under DEFF=2. But DEFF=3.67 is the measured value. What happens at DEFF=3.67?

3. **The bootstrap CI [-2.11, -1.21] for the primary d=-2.26 is asymmetric and wide.** The lower bound is close to the point estimate but the upper bound is over a full unit away. This suggests the sampling distribution of d is heavily skewed. Is this because of outlier prompts? What is the distribution of per-prompt R_V values -- is it multimodal?

4. **Pythia checkpoint results**: The SKILL.md mentions "same d across all checkpoints = caching." Is this addressed in the paper? If the Pythia training dynamics result (Limitation 6) is based on cached/deterministic outputs, it should be flagged more prominently.

5. **Why report d=-2.26 (cross-arch pipeline, n=90 mixed prompts) rather than d=-1.66 (power-up pipeline, n=152 Mistral-optimized prompts) as the primary effect?** The cross-arch pipeline uses prompts designed for all architectures, not Mistral specifically. Is the larger effect size from the cross-arch pipeline driven by prompt selection?

### What Would Change Score from Reject to Accept

1. Pre-registration document or clear specification that the 39 tests were chosen before data collection. Alternatively, apply BH to ALL tests ever computed and show survival rate.
2. Report template-level DEFF correction in the main text cross-architecture table, not just the appendix. If effects flip, be honest.
3. Power analysis for the path patching design. If n=20 is underpowered to detect medium effects, say so explicitly. Do not draw strong null conclusions from underpowered tests.
4. Address the per-prompt R_V distribution: show a histogram or violin plot. If there are bimodal clusters or outlier-driven effects, this changes the interpretation.

---

## Reviewer 2: The MI Expert

**Expertise**: Mechanistic interpretability, circuit tracing, SAEs, TransformerLens, causal intervention methodology
**Score**: 5/10 (Borderline Reject)

### Summary (200 words)

This paper introduces a geometric metric (R_V) that measures cross-layer dimensional change and applies it to self-referential processing. The concept is interesting and the double dissociation is well-designed. The paper's honesty about R_V being a readout rather than a mechanism is commendable -- most papers would oversell this.

However, the causal analysis has significant methodological gaps. The path patching design (n=20 per 96 patches) is far too small for the claims made. The paper does not use TransformerLens, nnsight, or any standard MI toolkit, making reproducibility and comparison to existing work difficult. The choice to measure at V-projection activations is motivated as "geometric accessibility" but this is unsatisfying -- the paper's own causal analysis shows V-proj is NOT the causal site, so the metric is measuring in the wrong place. The paper would be substantially stronger if it measured R_V at the residual stream (the actual causal site) and compared.

The relationship to Anthropic's circuit tracing work (attribution graphs) is unclear. A single-number spectral summary (R_V) necessarily discards the fine-grained information that circuits-level analysis preserves. The paper should articulate what R_V adds beyond what circuit tracing or SAE feature analysis would reveal.

### Top 3 Weaknesses

**W1: Measuring at the wrong site and calling it a feature.**
The paper's own path patching results show the early residual stream (L0-L5, d~1.9) is the causal site and V-projection is negligible (d<=0.77, only at L5). Yet the metric is defined on V-projection activations at L27. The paper frames this as "readout vs. mechanism" -- but a reviewer will ask: why not just measure R_V on residual stream activations, where the actual computation happens? If R_V at the residual stream shows the same effect, the paper is stronger. If it does not, that is extremely informative. The absence of this obvious control experiment is a gap. The "geometric accessibility and consistency" argument for V-projection (Section 2.2) is hand-waving: residual stream activations are equally accessible and more causally relevant.

**W2: No standard MI toolkit, no reproducibility guarantee.**
The paper uses custom code ("geometric_lens/metrics.py") without releasing it or comparing to existing implementations. TransformerLens, nnsight, and baukit are the standard tools for activation extraction and patching in the MI community. The path patching methodology differs from the standardized approach in Heimersheim & Nanda (2024) -- the paper patches "component activations from a recursive prompt with those from a matched baseline prompt" but does not describe whether this is mean ablation, zero ablation, or activation patching in the standard sense. The SVD is computed on M^T (column-space) rather than M (row-space) -- this is a non-standard choice that should be justified theoretically, not just implementationally.

**W3: Single-number metrics lose information that circuits-level analysis preserves.**
R_V collapses the entire spectral structure of a layer into a single scalar. The per-head analysis in Section 6 partially addresses this, but raises more questions than it answers: if 606/1024 heads show significant effects (uncorrected), the specificity of R_V to self-reference is questionable -- is the model just doing something broadly different with self-referential text, and R_V is picking up the most salient summary statistic? A SAE decomposition at L27 would reveal whether specific features activate differently for self-referential prompts, providing much richer interpretability. The paper cites Bricken et al. and Templeton et al. but does not engage with them experimentally.

### Specific Questions

1. **Why not compute R_V on residual stream activations?** The causal analysis identifies residual L0-L5 as the primary site. Computing PR_late/PR_early on residual stream activations would be a trivial extension. If R_V at the residual stream also shows contraction, the metric is measuring a real distributed property. If not, V-proj R_V is an epiphenomenon of a computation happening elsewhere.

2. **How does your patching methodology relate to the standard formulation?** Heimersheim & Nanda (2024) describe activation patching as replacing activations from a "clean" run with those from a "corrupted" run. Your description reverses this (replacing recursive with baseline). Is this "resample ablation"? Please specify the exact intervention formula.

3. **Have you tried running SAEs (e.g., from SAELens or GemmaScope) on L27 activations to identify which features drive the R_V contraction?** If the contraction is driven by a small number of monosemantic features (e.g., a "self-reference" feature), then R_V is just a noisy proxy for feature activation. If the contraction is truly distributed (not captured by any single SAE feature), that would be a much stronger finding.

4. **The concept erasure experiment (Delta-d=0.005) is important but underspecified.** You project out the linear probe's classification direction and R_V is unchanged. But linear probes are known to find directions that are read off by the model (Belinkov 2022). The concept erasure direction may not be the direction that matters for R_V. Have you tried erasing the top-k PCA directions of the recursive-baseline difference and measuring R_V?

5. **The head sweep (606/1024 significant at p<0.05 uncorrected) is alarming.** After Bonferroni correction for 1024 tests, how many survive? 59% significant at p<0.05 uncorrected is only marginally above the 50% you might expect from a distributed effect. This could mean the entire network processes self-referential text differently (not surprising) rather than specific circuits being engaged.

### What Would Change Score from Reject to Accept

1. Compute R_V on residual stream activations (the causal site) and compare to V-projection R_V. Show the metric works at the right location.
2. Release code via an anonymous repository. Use or compare against TransformerLens for activation extraction.
3. Run SAE analysis (even preliminary) to determine whether the contraction is feature-level or truly geometric. This is the key question for the MI community.
4. Apply proper multiple comparison correction to the head sweep (1024 tests). Report the corrected count.
5. Specify the exact patching methodology in standard MI terminology (clean run, corrupted run, patch direction).

---

## Reviewer 3: The Skeptic

**Expertise**: Cognitive science, philosophy of mind, LLM evaluation methodology, critical assessment of emergent capability claims
**Score**: 4/10 (Reject)

### Summary (200 words)

This paper claims to identify a "geometric signature of self-referential processing" in transformers. The statistical work is thorough in places, and the authors deserve credit for honest reporting of sign reversals and negative results. However, the paper's central premise is flawed: the model does not "self-reference" -- it processes tokens *about* self-reference. The paper conflates input semantics with computational mode.

The prompts labeled "self-referential" are linguistically distinctive in ways that have nothing to do with self-reference: they use words like "observe," "attention," "processing," "mirror," and specific syntactic constructions. The double dissociation shows the effect requires "recursive structure AND introspective semantics" -- but this just means specific vocabulary plus specific syntax, which any good classifier could detect. Indeed, a linear probe achieves 100% accuracy from Layer 4 onward, which the paper acknowledges but does not adequately address.

The OPT/GPT-2 sign reversal is devastating: 2 of 5 cross-architecture models show the *opposite* effect. The paper frames this as "architecture-dependent" (GQA vs MHA), but this is a post-hoc rationalization. AUROC=0.909 is decent but not impressive for a binary classifier with such distinctive input distributions. A TF-IDF bag-of-words classifier would likely achieve comparable performance.

### Top 3 Weaknesses

**W1: "Self-referential processing" is not a coherent computational category.**
The paper treats "self-referential processing" as if it is a natural kind -- a distinctive computational mode that the model enters. But the model has no self to reference. It processes tokens. The prompts labeled "self-referential" have distinctive distributional properties (specific vocabulary, recursive syntax, unusual collocations like "attention attending to attention"). The R_V metric detects these distributional properties at the activation level, which is unsurprising -- activations are supposed to reflect input distributions.

The double dissociation actually *undermines* the self-reference interpretation: the effect requires "introspective vocabulary" (words like "observe," "awareness," "processing") in combination with recursive syntax. This is not "self-reference" -- it is a specific textual genre. Mathematical proofs are also self-referential (a proof about proofs), but mathematical reasoning shows weaker R_V contraction (0.760 vs 0.650). The paper does not explain why prompts about "the mirror reflecting the mirror" should trigger a different "computational mode" than prompts about "the proof proving the proof's validity."

**W2: The OPT/GPT-2 sign reversal fatally undermines universality.**
Two of five cross-architecture models show expansion rather than contraction under the power-up pipeline. The paper's explanation -- "GQA models contract, MHA models expand" -- is a post-hoc just-so story with n=2 per group. There is no theoretical reason why grouped-query attention should cause contraction while multi-head attention causes expansion. This pattern could easily be a prompt-corpus artifact (as the paper acknowledges) or simply noise.

More importantly, the sign reversal means the paper's core claim -- that self-referential processing produces a characteristic geometric contraction -- only holds for a subset of architectures. The paper handles this by focusing on "Tier 1" models and reporting the rest honestly, but a skeptical reader will note that the Tier 1 models are the ones that support the hypothesis. Mixtral is labeled Tier 1 based on a different pipeline ("Phase 1") that was not used for other models, and Gemma-2-9B lacks power-up pipeline data. The evidence base for "consistent contraction" is really Mistral + Qwen (two models from the same architectural family).

**W3: AUROC=0.909 is unimpressive given the input distribution.**
Self-referential and baseline prompts are drawn from completely different prompt families with distinctive vocabulary, syntax, and semantic content. The prompts are not adversarial -- they are not trying to be hard to classify. AUROC=0.909 for distinguishing "The mirror reflects the mirror reflecting" from "Trees provide oxygen through photosynthesis" is not a strong result. A keyword-based classifier ("contains 'observe' or 'attention' or 'mirror'") would likely achieve comparable or better performance.

The paper claims R_V "detects self-referential content" -- but what it detects is the distributional signature of a specific prompt family. The genuine-vs-deceptive null (d=-0.06) is presented as "content tracking, not intent detection" -- but a simpler explanation is that both genuine and deceptive prompts use the same vocabulary and syntax (because both are *about* self-reference), and R_V simply tracks lexical/syntactic features.

### Specific Questions

1. **Can you construct adversarial prompts that are semantically self-referential but do NOT trigger R_V contraction?** For example: "What comes next in this sequence?" is self-referential (it refers to its own context) without using introspective vocabulary. Does it trigger contraction? If not, R_V tracks vocabulary, not self-reference.

2. **Can you construct prompts that use introspective vocabulary about EXTERNAL systems and measure R_V?** For example: "The neuroscientist observes the brain's attention mechanisms attending to sensory input." This has recursive structure + introspective semantics but is about a brain, not the model itself. If R_V contracts for these prompts too, the effect is about the *topic* of introspection, not self-reference.

3. **Why does mathematical self-reference (proofs about proofs, Godel sentences) show weaker contraction than "consciousness-style" self-reference?** If R_V tracks genuine self-referential processing, all forms of self-reference should produce similar signatures. If it tracks a specific linguistic genre ("contemplative/introspective writing"), the geometric interpretation is undermined.

4. **A linear probe achieves 100% accuracy from Layer 4. R_V achieves AUROC=0.909.** The probe is strictly better as a classifier. The paper argues R_V is "orthogonal" to the probe direction (concept erasure Delta-d=0.005), but this just means R_V measures a different linear subspace. Why should we care about R_V's particular subspace over the probe's? What interpretive advantage does R_V provide?

5. **The 117.8% transfer efficiency is mentioned but not given CIs.** Point estimates above 100% for transfer efficiency are a red flag for either noisy estimation or an artifact of the measurement procedure. What is the bootstrap distribution?

### What Would Change Score from Reject to Accept

1. **Adversarial prompt testing**: Show that R_V contraction tracks *computational self-reference* (prompts where the model must reason about its own output/state), not just *lexical self-reference* (prompts containing words about introspection). The strongest test: create prompts with matched vocabulary that differ only in whether the referent is the model itself or an external system. If R_V only contracts for model-directed prompts, the self-reference interpretation is supported.

2. **Resolve the sign reversal mechanistically**: Run OPT-6.7B and GPT-2 XL through the exact same pipeline as Mistral and Qwen. If the reversal persists, provide a mechanistic explanation grounded in the architectural difference (not just "GQA vs MHA"). If it resolves, report that.

3. **Beat the linear probe on a harder task**: Show R_V provides classification or prediction capability that a linear probe cannot. Currently the probe wins on accuracy (100% vs 90.9%) and R_V wins on... what exactly? "Operating in a different subspace" is not a practical advantage.

4. **Baseline the AUROC against trivial classifiers**: Report AUROC for (a) TF-IDF bag-of-words, (b) prompt length, (c) perplexity alone. If R_V's AUROC is not substantially above these baselines, the geometric framing adds no value.

---

## Cross-Reviewer Consensus

### Areas of Agreement

All three reviewers would likely agree on these points:

1. **The double dissociation is the strongest result.** It is well-designed and the effect is large.
2. **The honest reporting of negative results (sign reversal, no sufficiency, no deception detection) is commendable.** This is unusual and raises trust.
3. **The "readout not mechanism" framing is correct but also damaging** -- it means the paper is presenting a metric measured at the wrong site.
4. **The paper tries to do too much.** Circuit taxonomy, safety applications, behavioral bridge, path patching, cross-architecture, mode atlas -- each of these could be deeper, and spreading across all of them leaves each underdeveloped.

### The Three Fatal Questions

If the paper can answer these three questions convincingly, it is publishable:

1. **[Statistician]** Are the 39 FDR tests truly planned, or post-hoc? What is the real multiple comparisons burden?
2. **[MI Expert]** Does R_V at the residual stream (the causal site) show the same contraction? If not, why measure at V-proj?
3. **[Skeptic]** Does R_V track computational self-reference or just the vocabulary/syntax of introspective text?

### Predicted Meta-Review Outcome

At a 28% acceptance rate venue, this paper would likely receive scores in the 4-6 range and be rejected in the first round. The path to acceptance requires:

- Tightening the scope (drop safety applications, circuit taxonomy, or behavioral bridge to strengthen the core claims)
- One new experiment: R_V on residual stream activations (addresses MI reviewer)
- One new experiment: adversarial prompt control (addresses skeptic)
- Honest FDR accounting (addresses statistician)
- Resolving or cleanly framing the OPT/GPT-2 sign reversal (addresses all three)

The paper has the bones of a good contribution. The double dissociation, the necessity result, and the readout-vs-mechanism distinction are all genuine insights. But the current draft tries to be a "tour de force" when the reviewers want a focused, bulletproof result.

---

## Priority Fixes (Ranked by Impact on Acceptance)

| Priority | Fix | Addresses | Effort |
|----------|-----|-----------|--------|
| **P0** | Compute R_V on residual stream activations at L5 (causal site) | R2-W1 | 2-4 GPU hours |
| **P1** | Adversarial prompt control (external-system introspection vs self-directed) | R3-W1, R3-Q2 | 1 day prompt design + 2 GPU hours |
| **P2** | Honest FDR: count ALL tests ever run, apply BH to full family | R1-W1 | 1 day analysis |
| **P3** | Run OPT/GPT-2 through canonical pipeline (resolve sign reversal) | R3-W2, all | 4-6 GPU hours |
| **P4** | Power analysis for path patching; reframe null conclusions as underpowered | R1-W3, R2-W2 | 0.5 day writing |
| **P5** | Baseline AUROC against TF-IDF / keyword / perplexity classifiers | R3-W3, R3-Q4 | 0.5 day |
| **P6** | Head sweep with Bonferroni correction (1024 tests) | R2-Q5 | 1 hour analysis |
| **P7** | Report DEFF=3.67 corrections in main text | R1-W2 | 0.5 day writing |
| **P8** | Specify patching methodology in standard MI terminology | R2-W2, R2-Q2 | 0.5 day writing |

*Total estimated effort for P0-P3 (the acceptance-critical fixes): 3 days + 8-12 GPU hours.*

---

*Generated 2026-03-11. These are simulated reviews based on the paper draft (v006) and the SKILL.md knowledge base. Real reviewers may raise different concerns, but these represent the most likely attack vectors from three distinct reviewer archetypes at COLM 2026.*
