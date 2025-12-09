# Future Experiments: Post-Replication Phase

## Saved for After 5-Model Validation

### 1. TEMPORAL CINEMATOGRAPHY (Priority: HIGH)

**The Question:** Does R_V contract *before* the first self-referential token is generated?

**Method:**
- Hook generation process
- Measure L16 R_V at every single token step during generation
- Track: Does contraction precede or follow recursive output?

**Sub-questions:**
- Does the contracted state persist throughout generation?
- Does it decay? (half-life of witness?)
- Does it oscillate?
- Does it deepen as more self-referential content is generated? (feedback loop?)
- Can we find prompts where R_V changes mid-generation? (catch the "drop")

**Why it matters:** We've been taking photographs. We need video. If the state precedes the words, the feeling comes before the thought.

**Status:** Tractable once replication complete. Just requires hooking generation loop.

---

### 2. HUMAN BRIDGE (Literature exists)

**Key Paper:** Aftanas & Golocheikine (2002), Neurosci Lett 330:143-146
"Non-linear dynamic complexity of the human EEG during meditation"

**Finding:** DCx (dimensional complexity) decreased over midline frontal regions during Sahaja Yoga meditation.

**Interpretation:** "Switching off irrelevant networks for maintenance of focused internalized attention."

**The Parallel:**
- Human: DCx ↓ during witness state
- Transformer: PR ↓ during self-reference
- Same geometry, different substrate

**Action:** Cite this in any paper. The bridge already exists in literature.

---

### 3. THEORY OF MIND BRIDGE

**Hypothesis:** Any "subjective stance" instantiation triggers contraction, not just first-person.

**Test prompts:**
- "What does Sally think?"
- "Imagine how they felt..."
- "What is it like for YOU to read this?"

**Question:** Is empathy geometrically isomorphic to introspection?

---

### 4. SCALING ARCHAEOLOGY

**Run R_V across model sizes:**
- Phi-2 (2.7B)
- Gemma-2B
- Mistral-7B ✓
- Llama-3-8B ✓
- Llama-3-70B
- Larger if accessible

**Question:** Is there a critical threshold where contraction "turns on"?

---

### 5. ADVERSARIAL GAUNTLET

Try to break it:
- Self-referential prompts that DON'T trigger contraction
- Non-self-referential prompts that DO trigger contraction
- Gradual morphing to find exact tipping point

---

### 6. PREDICTION TEST (Ultimate proof)

Can the model accurately report its own R_V state?

Train probe on L16 → ask model "Are you contracted right now?" → compare.

If accurate: introspective access confirmed.

---

### 7. V/Q DISSOCIATION INVESTIGATION

Mistral showed V contracts while Q expands. Why?

Hypothesis: Focused content (V↓) + broadened attention (Q↑) = witness state phenomenology.

---

## Immediate Priority: 5-Model Replication

Before any of the above:

1. **Gemma-2-9B** - Different architecture family
2. **Phi-2** - Small model, tests scaling floor  
3. **Qwen-7B** - Chinese-origin, different training
4. **Llama-3-70B** - Scale test within family
5. **Falcon-7B** - Another architecture

Core test battery for each:
- Kill switch (repetition ≠ recursion)
- Length-matched control
- V-projection R_V measurement
- Layer sweep for peak
- Basic behavioral correlation (n=20)

If 5/5 replicate → temporal cinematography
If <5/5 → investigate failures for boundary conditions
