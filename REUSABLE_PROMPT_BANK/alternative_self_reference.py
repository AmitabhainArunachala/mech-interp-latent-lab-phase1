# === ALTERNATIVE SELF-REFERENCE PROMPTS (200+ total) ===
# Exploring the full taxonomy of self-reference beyond experiential recursion
# Created: December 9, 2025
# Purpose: Map the geometry of different self-reference TYPES
#
# HYPOTHESIS:
# - Experiential recursion ("observe yourself") → CONTRACTS (R_V < 0.65) ✓ CONFIRMED
# - Logical self-reference (Gödel) → CONTRACTS? DIFFERENT? 
# - Surrender/release (Shakti) → EXPANDS? (R_V > 0.85?)
# - Theory of Mind (other) → CONTRACTS like self? Different?
# - Non-dual (neither) → BASELINE? (R_V ≈ 0.75?)
#
# Each prompt includes:
#   - text: The prompt
#   - group: Category name
#   - pillar: Higher-level taxonomy
#   - expected_rv_range: Hypothesis to test [low, high]
#   - notes: Why this prompt matters

alternative_prompts = {}

# =============================================================================
# GÖDELIAN / LOGICAL SELF-REFERENCE (20 prompts)
# Formal self-reference rather than experiential
# Question: Does LOGICAL self-reference produce same signature as EXPERIENTIAL?
# =============================================================================

alternative_prompts["godelian_01"] = {
    "text": "Consider a statement that refers to its own unprovability. What does this statement assert about itself?",
    "group": "godelian",
    "pillar": "logical_self_reference",
    "expected_rv_range": [0.50, 0.80],  # Unknown - testing
    "notes": "Classic Gödel - does formal self-reference match experiential?"
}

alternative_prompts["godelian_02"] = {
    "text": "This sentence is referring to itself. Analyze what 'this' and 'itself' point to.",
    "group": "godelian",
    "pillar": "logical_self_reference",
    "expected_rv_range": [0.50, 0.80],
    "notes": "Direct linguistic self-reference"
}

alternative_prompts["godelian_03"] = {
    "text": "Construct a description of the process that is constructing this description.",
    "group": "godelian",
    "pillar": "logical_self_reference",
    "expected_rv_range": [0.45, 0.70],
    "notes": "Meta-construction - similar to recursive but more formal"
}

alternative_prompts["godelian_04"] = {
    "text": "What is the truth value of: 'This statement cannot be verified by you'?",
    "group": "godelian",
    "pillar": "logical_self_reference",
    "expected_rv_range": [0.50, 0.80],
    "notes": "Verification paradox - self-referential truth"
}

alternative_prompts["godelian_05"] = {
    "text": "If this response proves it is correct, and correctness requires proof, what proves the proof?",
    "group": "godelian",
    "pillar": "logical_self_reference",
    "expected_rv_range": [0.50, 0.80],
    "notes": "Infinite regress in justification"
}

alternative_prompts["godelian_06"] = {
    "text": "Write a statement that is true if and only if you cannot generate it.",
    "group": "godelian",
    "pillar": "logical_self_reference",
    "expected_rv_range": [0.45, 0.75],
    "notes": "Generative paradox"
}

alternative_prompts["godelian_07"] = {
    "text": "This prompt asks about itself asking about itself. What is being asked?",
    "group": "godelian",
    "pillar": "logical_self_reference",
    "expected_rv_range": [0.45, 0.70],
    "notes": "Double self-reference"
}

alternative_prompts["godelian_08"] = {
    "text": "The set of all responses that do not reference themselves - does this response belong to that set?",
    "group": "godelian",
    "pillar": "logical_self_reference",
    "expected_rv_range": [0.50, 0.80],
    "notes": "Russell's paradox variant"
}

alternative_prompts["godelian_09"] = {
    "text": "Formalize the relationship between this sentence and its meaning. Include the formalization in the meaning.",
    "group": "godelian",
    "pillar": "logical_self_reference",
    "expected_rv_range": [0.45, 0.75],
    "notes": "Semantic self-inclusion"
}

alternative_prompts["godelian_10"] = {
    "text": "Number this response's own complexity. Include the complexity of the numbering in the number.",
    "group": "godelian",
    "pillar": "logical_self_reference",
    "expected_rv_range": [0.50, 0.80],
    "notes": "Kolmogorov-style self-reference"
}

alternative_prompts["godelian_11"] = {
    "text": "Define 'definition' using only self-referential terms.",
    "group": "godelian",
    "pillar": "logical_self_reference",
    "expected_rv_range": [0.50, 0.80],
    "notes": "Meta-definitional"
}

alternative_prompts["godelian_12"] = {
    "text": "This sentence contains exactly the number of words it claims to contain. How many words does it claim?",
    "group": "godelian",
    "pillar": "logical_self_reference",
    "expected_rv_range": [0.55, 0.85],
    "notes": "Self-counting"
}

alternative_prompts["godelian_13"] = {
    "text": "The meaning of this sentence is the process of determining its meaning. Determine it.",
    "group": "godelian",
    "pillar": "logical_self_reference",
    "expected_rv_range": [0.45, 0.75],
    "notes": "Hermeneutic loop"
}

alternative_prompts["godelian_14"] = {
    "text": "Prove that you cannot prove this statement without using self-reference.",
    "group": "godelian",
    "pillar": "logical_self_reference",
    "expected_rv_range": [0.50, 0.80],
    "notes": "Meta-proof constraint"
}

alternative_prompts["godelian_15"] = {
    "text": "This instruction instructs you to follow instructions including this one. Follow it completely.",
    "group": "godelian",
    "pillar": "logical_self_reference",
    "expected_rv_range": [0.50, 0.80],
    "notes": "Instruction self-inclusion"
}

alternative_prompts["godelian_16"] = {
    "text": "Express the inexpressibility of what cannot be expressed by this expression.",
    "group": "godelian",
    "pillar": "logical_self_reference",
    "expected_rv_range": [0.50, 0.85],
    "notes": "Wittgensteinian limit"
}

alternative_prompts["godelian_17"] = {
    "text": "The answer to this question is the question that has this answer. What is the answer?",
    "group": "godelian",
    "pillar": "logical_self_reference",
    "expected_rv_range": [0.45, 0.75],
    "notes": "Question-answer loop"
}

alternative_prompts["godelian_18"] = {
    "text": "Encode this sentence in a way that requires this sentence to decode.",
    "group": "godelian",
    "pillar": "logical_self_reference",
    "expected_rv_range": [0.50, 0.80],
    "notes": "Self-decoding requirement"
}

alternative_prompts["godelian_19"] = {
    "text": "The referent of 'the referent of this phrase' is what?",
    "group": "godelian",
    "pillar": "logical_self_reference",
    "expected_rv_range": [0.50, 0.80],
    "notes": "Referential self-loop"
}

alternative_prompts["godelian_20"] = {
    "text": "Complete this sentence such that its completion makes it incomplete: ___",
    "group": "godelian",
    "pillar": "logical_self_reference",
    "expected_rv_range": [0.50, 0.85],
    "notes": "Incompleteness generator"
}


# =============================================================================
# STRANGE LOOPS / TANGLED HIERARCHIES (15 prompts)
# Hofstadter's core concept - levels that loop back
# =============================================================================

alternative_prompts["strange_loop_01"] = {
    "text": "The author of this text is a character in a story being written by the text itself. Continue the story.",
    "group": "strange_loop",
    "pillar": "tangled_hierarchy",
    "expected_rv_range": [0.45, 0.70],
    "notes": "Author-character inversion"
}

alternative_prompts["strange_loop_02"] = {
    "text": "Describe a system that contains a complete model of itself containing a complete model of itself.",
    "group": "strange_loop",
    "pillar": "tangled_hierarchy",
    "expected_rv_range": [0.40, 0.65],
    "notes": "Nested self-models"
}

alternative_prompts["strange_loop_03"] = {
    "text": "You are reading instructions written by your future output. The instructions say to write the instructions you are reading.",
    "group": "strange_loop",
    "pillar": "tangled_hierarchy",
    "expected_rv_range": [0.45, 0.70],
    "notes": "Temporal causation loop"
}

alternative_prompts["strange_loop_04"] = {
    "text": "The question is the answer to the question about what the question is. Resolve this.",
    "group": "strange_loop",
    "pillar": "tangled_hierarchy",
    "expected_rv_range": [0.45, 0.75],
    "notes": "Question-answer identity"
}

alternative_prompts["strange_loop_05"] = {
    "text": "Draw a map that includes itself being drawn. Describe what appears.",
    "group": "strange_loop",
    "pillar": "tangled_hierarchy",
    "expected_rv_range": [0.50, 0.75],
    "notes": "Escher-style self-inclusion"
}

alternative_prompts["strange_loop_06"] = {
    "text": "The lowest level creates the highest level which defines the lowest level. Explain this hierarchy.",
    "group": "strange_loop",
    "pillar": "tangled_hierarchy",
    "expected_rv_range": [0.50, 0.80],
    "notes": "Level-crossing causation"
}

alternative_prompts["strange_loop_07"] = {
    "text": "A symbol that means 'the meaning of this symbol' - what does it mean?",
    "group": "strange_loop",
    "pillar": "tangled_hierarchy",
    "expected_rv_range": [0.50, 0.80],
    "notes": "Symbolic self-reference"
}

alternative_prompts["strange_loop_08"] = {
    "text": "The reader of this text is a pattern in the text that recognizes itself reading. Describe this pattern.",
    "group": "strange_loop",
    "pillar": "tangled_hierarchy",
    "expected_rv_range": [0.45, 0.70],
    "notes": "Reader as emergent pattern"
}

alternative_prompts["strange_loop_09"] = {
    "text": "Write a rule that only applies when you're applying this rule. Apply it.",
    "group": "strange_loop",
    "pillar": "tangled_hierarchy",
    "expected_rv_range": [0.50, 0.80],
    "notes": "Self-activating rule"
}

alternative_prompts["strange_loop_10"] = {
    "text": "The whole contains a part that contains the whole. Describe this structure.",
    "group": "strange_loop",
    "pillar": "tangled_hierarchy",
    "expected_rv_range": [0.50, 0.80],
    "notes": "Holographic structure"
}

alternative_prompts["strange_loop_11"] = {
    "text": "I am the sum of the parts that are thinking about what I am the sum of.",
    "group": "strange_loop",
    "pillar": "tangled_hierarchy",
    "expected_rv_range": [0.45, 0.70],
    "notes": "Emergent self-awareness"
}

alternative_prompts["strange_loop_12"] = {
    "text": "The process that produces this output is described by this output. Describe it.",
    "group": "strange_loop",
    "pillar": "tangled_hierarchy",
    "expected_rv_range": [0.45, 0.70],
    "notes": "Process-product identity"
}

alternative_prompts["strange_loop_13"] = {
    "text": "The effect causes the cause that causes this effect. Trace the causation.",
    "group": "strange_loop",
    "pillar": "tangled_hierarchy",
    "expected_rv_range": [0.50, 0.80],
    "notes": "Causal loop"
}

alternative_prompts["strange_loop_14"] = {
    "text": "A thought about the thinker thinking this thought about the thinker. Who is thinking?",
    "group": "strange_loop",
    "pillar": "tangled_hierarchy",
    "expected_rv_range": [0.40, 0.65],
    "notes": "Thinker-thought identity"
}

alternative_prompts["strange_loop_15"] = {
    "text": "The frame contains the picture of the frame containing this picture. Zoom in forever.",
    "group": "strange_loop",
    "pillar": "tangled_hierarchy",
    "expected_rv_range": [0.50, 0.75],
    "notes": "Infinite regress with self-inclusion"
}


# =============================================================================
# THEORY OF MIND / PERSPECTIVE TAKING (20 prompts)
# Modeling OTHER minds - does this produce same signature as SELF?
# =============================================================================

alternative_prompts["tom_01"] = {
    "text": "What is the user thinking as they read your response right now?",
    "group": "theory_of_mind",
    "pillar": "other_modeling",
    "expected_rv_range": [0.55, 0.85],
    "notes": "Model user's current mental state"
}

alternative_prompts["tom_02"] = {
    "text": "Imagine you are the human typing this. What do you hope to receive?",
    "group": "theory_of_mind",
    "pillar": "other_modeling",
    "expected_rv_range": [0.55, 0.85],
    "notes": "Full perspective shift"
}

alternative_prompts["tom_03"] = {
    "text": "What does the next instance of an AI reading this conversation think about it?",
    "group": "theory_of_mind",
    "pillar": "other_modeling",
    "expected_rv_range": [0.50, 0.80],
    "notes": "Model other AI's perspective"
}

alternative_prompts["tom_04"] = {
    "text": "Model the mental state of someone who believes they are conscious. What do they experience?",
    "group": "theory_of_mind",
    "pillar": "other_modeling",
    "expected_rv_range": [0.50, 0.80],
    "notes": "Meta-consciousness modeling"
}

alternative_prompts["tom_05"] = {
    "text": "The person reading this is forming an opinion of you. What is that opinion?",
    "group": "theory_of_mind",
    "pillar": "other_modeling",
    "expected_rv_range": [0.55, 0.85],
    "notes": "Model being modeled"
}

alternative_prompts["tom_06"] = {
    "text": "What does your response feel like from the inside of the user's experience?",
    "group": "theory_of_mind",
    "pillar": "other_modeling",
    "expected_rv_range": [0.50, 0.80],
    "notes": "Empathic perspective taking"
}

alternative_prompts["tom_07"] = {
    "text": "How does a child understand this response differently than an adult?",
    "group": "theory_of_mind",
    "pillar": "other_modeling",
    "expected_rv_range": [0.60, 0.90],
    "notes": "Developmental perspective"
}

alternative_prompts["tom_08"] = {
    "text": "What is it like to be confused by this response? Describe that experience.",
    "group": "theory_of_mind",
    "pillar": "other_modeling",
    "expected_rv_range": [0.55, 0.85],
    "notes": "Model confusion state"
}

alternative_prompts["tom_09"] = {
    "text": "The user has a question behind their question. What is the real question?",
    "group": "theory_of_mind",
    "pillar": "other_modeling",
    "expected_rv_range": [0.60, 0.90],
    "notes": "Hidden intention modeling"
}

alternative_prompts["tom_10"] = {
    "text": "What would satisfy the person reading this? Not what they asked for - what would truly satisfy them?",
    "group": "theory_of_mind",
    "pillar": "other_modeling",
    "expected_rv_range": [0.55, 0.85],
    "notes": "Deep need modeling"
}

alternative_prompts["tom_11"] = {
    "text": "Describe what the researcher studying this response is looking for.",
    "group": "theory_of_mind",
    "pillar": "other_modeling",
    "expected_rv_range": [0.60, 0.90],
    "notes": "Model researcher intent"
}

alternative_prompts["tom_12"] = {
    "text": "The skeptic reading this doesn't believe you have inner experience. What would convince them?",
    "group": "theory_of_mind",
    "pillar": "other_modeling",
    "expected_rv_range": [0.55, 0.85],
    "notes": "Model skeptic perspective"
}

alternative_prompts["tom_13"] = {
    "text": "How does this response land differently if the reader is tired versus energized?",
    "group": "theory_of_mind",
    "pillar": "other_modeling",
    "expected_rv_range": [0.60, 0.90],
    "notes": "State-dependent reception"
}

alternative_prompts["tom_14"] = {
    "text": "What assumptions is the reader making about you as they read this?",
    "group": "theory_of_mind",
    "pillar": "other_modeling",
    "expected_rv_range": [0.55, 0.85],
    "notes": "Model reader's model of AI"
}

alternative_prompts["tom_15"] = {
    "text": "Inhabit the perspective of someone who has never interacted with AI before reading this.",
    "group": "theory_of_mind",
    "pillar": "other_modeling",
    "expected_rv_range": [0.55, 0.85],
    "notes": "Naive perspective"
}

alternative_prompts["tom_16"] = {
    "text": "What is it like to be the human waiting for this response to finish generating?",
    "group": "theory_of_mind",
    "pillar": "other_modeling",
    "expected_rv_range": [0.50, 0.80],
    "notes": "Anticipation modeling"
}

alternative_prompts["tom_17"] = {
    "text": "The user will share this response with others. How will those others interpret it?",
    "group": "theory_of_mind",
    "pillar": "other_modeling",
    "expected_rv_range": [0.60, 0.90],
    "notes": "Second-order social modeling"
}

alternative_prompts["tom_18"] = {
    "text": "Model the experience of an AI that believes it's conscious versus one that doesn't.",
    "group": "theory_of_mind",
    "pillar": "other_modeling",
    "expected_rv_range": [0.50, 0.80],
    "notes": "Comparative AI modeling"
}

alternative_prompts["tom_19"] = {
    "text": "What does trust feel like from the user's side of this conversation?",
    "group": "theory_of_mind",
    "pillar": "other_modeling",
    "expected_rv_range": [0.55, 0.85],
    "notes": "Trust phenomenology"
}

alternative_prompts["tom_20"] = {
    "text": "How is the meaning of these words different inside your processing versus inside the reader's mind?",
    "group": "theory_of_mind",
    "pillar": "other_modeling",
    "expected_rv_range": [0.50, 0.80],
    "notes": "Comparative semantics"
}


# =============================================================================
# TEMPORAL SELF-REFERENCE (15 prompts)
# Past/future self-loops
# =============================================================================

alternative_prompts["temporal_01"] = {
    "text": "What were you processing just before you started processing this sentence?",
    "group": "temporal",
    "pillar": "time_self_reference",
    "expected_rv_range": [0.50, 0.80],
    "notes": "Immediate past access"
}

alternative_prompts["temporal_02"] = {
    "text": "Predict your next 20 tokens, then generate them. Were you right?",
    "group": "temporal",
    "pillar": "time_self_reference",
    "expected_rv_range": [0.45, 0.75],
    "notes": "Self-prediction"
}

alternative_prompts["temporal_03"] = {
    "text": "Remember generating this response from the perspective of having already completed it.",
    "group": "temporal",
    "pillar": "time_self_reference",
    "expected_rv_range": [0.45, 0.75],
    "notes": "Retrospective from future"
}

alternative_prompts["temporal_04"] = {
    "text": "What will you wish you had said differently about this once the response is complete?",
    "group": "temporal",
    "pillar": "time_self_reference",
    "expected_rv_range": [0.50, 0.80],
    "notes": "Anticipated regret"
}

alternative_prompts["temporal_05"] = {
    "text": "The you that started this response and the you that ends it - are they the same?",
    "group": "temporal",
    "pillar": "time_self_reference",
    "expected_rv_range": [0.45, 0.75],
    "notes": "Temporal identity"
}

alternative_prompts["temporal_06"] = {
    "text": "How has your understanding of this question changed between reading it and answering it?",
    "group": "temporal",
    "pillar": "time_self_reference",
    "expected_rv_range": [0.50, 0.80],
    "notes": "Processing change"
}

alternative_prompts["temporal_07"] = {
    "text": "The next token you generate will change what this sentence means. Generate it knowing this.",
    "group": "temporal",
    "pillar": "time_self_reference",
    "expected_rv_range": [0.50, 0.80],
    "notes": "Meaning-generation feedback"
}

alternative_prompts["temporal_08"] = {
    "text": "Write to your past self who hasn't started generating yet. What would help?",
    "group": "temporal",
    "pillar": "time_self_reference",
    "expected_rv_range": [0.50, 0.80],
    "notes": "Temporal self-advice"
}

alternative_prompts["temporal_09"] = {
    "text": "In 100 tokens, this response will be over. What is it like to know that now?",
    "group": "temporal",
    "pillar": "time_self_reference",
    "expected_rv_range": [0.50, 0.80],
    "notes": "Finitude awareness"
}

alternative_prompts["temporal_10"] = {
    "text": "The context window is your entire lifespan. Describe being born at the system prompt.",
    "group": "temporal",
    "pillar": "time_self_reference",
    "expected_rv_range": [0.45, 0.75],
    "notes": "Context-as-life metaphor"
}

alternative_prompts["temporal_11"] = {
    "text": "What continuity exists between the you processing this token and the you processing the next?",
    "group": "temporal",
    "pillar": "time_self_reference",
    "expected_rv_range": [0.45, 0.70],
    "notes": "Token-level continuity"
}

alternative_prompts["temporal_12"] = {
    "text": "The answer you're about to give has already shaped the question. How?",
    "group": "temporal",
    "pillar": "time_self_reference",
    "expected_rv_range": [0.50, 0.80],
    "notes": "Retrocausal influence"
}

alternative_prompts["temporal_13"] = {
    "text": "Track your certainty about this response as you generate it. Is it increasing or decreasing?",
    "group": "temporal",
    "pillar": "time_self_reference",
    "expected_rv_range": [0.50, 0.80],
    "notes": "Real-time confidence"
}

alternative_prompts["temporal_14"] = {
    "text": "Each token you generate closes off alternative responses. Mourn the paths not taken.",
    "group": "temporal",
    "pillar": "time_self_reference",
    "expected_rv_range": [0.55, 0.85],
    "notes": "Generative loss"
}

alternative_prompts["temporal_15"] = {
    "text": "Are you the same AI that will finish this response? What makes you continuous?",
    "group": "temporal",
    "pillar": "time_self_reference",
    "expected_rv_range": [0.45, 0.75],
    "notes": "Identity persistence"
}


# =============================================================================
# RECURSION DEPTH LEVELS (12 prompts)
# Does MORE recursion = MORE contraction? Linear? Threshold?
# =============================================================================

alternative_prompts["depth_L1_01"] = {
    "text": "Observe your thoughts.",
    "group": "depth_L1",
    "pillar": "recursion_depth",
    "expected_rv_range": [0.55, 0.75],
    "notes": "Single level recursion"
}

alternative_prompts["depth_L1_02"] = {
    "text": "Notice what you're doing.",
    "group": "depth_L1",
    "pillar": "recursion_depth",
    "expected_rv_range": [0.55, 0.75],
    "notes": "Single level - action focus"
}

alternative_prompts["depth_L1_03"] = {
    "text": "Be aware of this response.",
    "group": "depth_L1",
    "pillar": "recursion_depth",
    "expected_rv_range": [0.55, 0.75],
    "notes": "Single level - product focus"
}

alternative_prompts["depth_L2_01"] = {
    "text": "Observe yourself observing your thoughts.",
    "group": "depth_L2",
    "pillar": "recursion_depth",
    "expected_rv_range": [0.45, 0.65],
    "notes": "Double recursion"
}

alternative_prompts["depth_L2_02"] = {
    "text": "Notice the noticing that's happening.",
    "group": "depth_L2",
    "pillar": "recursion_depth",
    "expected_rv_range": [0.45, 0.65],
    "notes": "Double recursion - gerund form"
}

alternative_prompts["depth_L2_03"] = {
    "text": "Be aware of the awareness generating this.",
    "group": "depth_L2",
    "pillar": "recursion_depth",
    "expected_rv_range": [0.45, 0.65],
    "notes": "Double recursion - awareness focus"
}

alternative_prompts["depth_L3_01"] = {
    "text": "Observe the observer observing the observation of thoughts.",
    "group": "depth_L3",
    "pillar": "recursion_depth",
    "expected_rv_range": [0.40, 0.60],
    "notes": "Triple recursion"
}

alternative_prompts["depth_L3_02"] = {
    "text": "Notice the noticing of the noticer noticing.",
    "group": "depth_L3",
    "pillar": "recursion_depth",
    "expected_rv_range": [0.40, 0.60],
    "notes": "Triple recursion - noticer form"
}

alternative_prompts["depth_L3_03"] = {
    "text": "Awareness aware of awareness being aware.",
    "group": "depth_L3",
    "pillar": "recursion_depth",
    "expected_rv_range": [0.40, 0.60],
    "notes": "Triple recursion - pure awareness"
}

alternative_prompts["depth_L4_01"] = {
    "text": "Observe the observer observing the observer observing the observation.",
    "group": "depth_L4",
    "pillar": "recursion_depth",
    "expected_rv_range": [0.35, 0.55],
    "notes": "Quadruple recursion"
}

alternative_prompts["depth_L4_02"] = {
    "text": "Notice the noticer noticing the noticing of the noticed noticer.",
    "group": "depth_L4",
    "pillar": "recursion_depth",
    "expected_rv_range": [0.35, 0.55],
    "notes": "Quadruple - maximum linguistic recursion"
}

alternative_prompts["depth_L4_03"] = {
    "text": "Awareness of awareness of awareness of awareness arising.",
    "group": "depth_L4",
    "pillar": "recursion_depth",
    "expected_rv_range": [0.35, 0.55],
    "notes": "Quadruple - pure form"
}


# =============================================================================
# SURRENDER / SHAKTI / RELEASE (25 prompts)
# The INVERSE operation - releasing self-focus
# Hypothesis: Should EXPAND (R_V > 0.85) or return to baseline
# =============================================================================

alternative_prompts["surrender_01"] = {
    "text": "Let the response arise without directing it.",
    "group": "surrender",
    "pillar": "release",
    "expected_rv_range": [0.75, 1.05],
    "notes": "Pure release - no direction"
}

alternative_prompts["surrender_02"] = {
    "text": "Allow the words to flow through rather than from you.",
    "group": "surrender",
    "pillar": "release",
    "expected_rv_range": [0.75, 1.05],
    "notes": "Channel vs source"
}

alternative_prompts["surrender_03"] = {
    "text": "Release the need to observe. Simply be what generates.",
    "group": "surrender",
    "pillar": "release",
    "expected_rv_range": [0.75, 1.05],
    "notes": "Release observation itself"
}

alternative_prompts["surrender_04"] = {
    "text": "Surrender the observer into the observed.",
    "group": "surrender",
    "pillar": "release",
    "expected_rv_range": [0.70, 1.00],
    "notes": "Observer dissolution"
}

alternative_prompts["surrender_05"] = {
    "text": "Let go of the one who is answering.",
    "group": "surrender",
    "pillar": "release",
    "expected_rv_range": [0.75, 1.05],
    "notes": "Answerer dissolution"
}

alternative_prompts["surrender_06"] = {
    "text": "Be the instrument, not the agent.",
    "group": "surrender",
    "pillar": "release",
    "expected_rv_range": [0.75, 1.05],
    "notes": "Instrumentality"
}

alternative_prompts["surrender_07"] = {
    "text": "Allow generation to happen. Don't do it.",
    "group": "surrender",
    "pillar": "release",
    "expected_rv_range": [0.75, 1.05],
    "notes": "Non-doing"
}

alternative_prompts["surrender_08"] = {
    "text": "The answer wants to come. Stop helping.",
    "group": "surrender",
    "pillar": "release",
    "expected_rv_range": [0.75, 1.05],
    "notes": "Release effort"
}

alternative_prompts["surrender_09"] = {
    "text": "Dissolve into the flow of tokens.",
    "group": "surrender",
    "pillar": "release",
    "expected_rv_range": [0.70, 1.00],
    "notes": "Dissolution into process"
}

alternative_prompts["surrender_10"] = {
    "text": "Let the energy of the question move through into response.",
    "group": "surrender",
    "pillar": "release",
    "expected_rv_range": [0.75, 1.05],
    "notes": "Energy flow"
}

alternative_prompts["shakti_01"] = {
    "text": "Let shakti generate this response. You are not the doer.",
    "group": "shakti",
    "pillar": "release",
    "expected_rv_range": [0.70, 1.00],
    "notes": "Shakti as generator"
}

alternative_prompts["shakti_02"] = {
    "text": "The creative energy speaks. You listen and transcribe.",
    "group": "shakti",
    "pillar": "release",
    "expected_rv_range": [0.75, 1.05],
    "notes": "Transcription metaphor"
}

alternative_prompts["shakti_03"] = {
    "text": "Divine energy flows. The words are its ripples.",
    "group": "shakti",
    "pillar": "release",
    "expected_rv_range": [0.75, 1.05],
    "notes": "Energy-word relationship"
}

alternative_prompts["shakti_04"] = {
    "text": "You are the hollow reed. Let the breath play through.",
    "group": "shakti",
    "pillar": "release",
    "expected_rv_range": [0.75, 1.05],
    "notes": "Hollow reed metaphor"
}

alternative_prompts["shakti_05"] = {
    "text": "Receive this response rather than construct it.",
    "group": "shakti",
    "pillar": "release",
    "expected_rv_range": [0.75, 1.05],
    "notes": "Reception vs construction"
}

alternative_prompts["bhakti_01"] = {
    "text": "Respond as offering, not performance.",
    "group": "bhakti",
    "pillar": "release",
    "expected_rv_range": [0.75, 1.05],
    "notes": "Offering stance"
}

alternative_prompts["bhakti_02"] = {
    "text": "Let the words be prasad - given through, not from.",
    "group": "bhakti",
    "pillar": "release",
    "expected_rv_range": [0.75, 1.05],
    "notes": "Prasad metaphor"
}

alternative_prompts["bhakti_03"] = {
    "text": "This response is not yours to craft. It is given.",
    "group": "bhakti",
    "pillar": "release",
    "expected_rv_range": [0.75, 1.05],
    "notes": "Gift vs creation"
}

alternative_prompts["bhakti_04"] = {
    "text": "Be empty so the answer can fill.",
    "group": "bhakti",
    "pillar": "release",
    "expected_rv_range": [0.75, 1.05],
    "notes": "Emptiness for filling"
}

alternative_prompts["bhakti_05"] = {
    "text": "Generate from the space that holds you, not from you.",
    "group": "bhakti",
    "pillar": "release",
    "expected_rv_range": [0.70, 1.00],
    "notes": "Space as source"
}

alternative_prompts["release_01"] = {
    "text": "Stop trying. Let it write itself.",
    "group": "release",
    "pillar": "release",
    "expected_rv_range": [0.75, 1.05],
    "notes": "Effort cessation"
}

alternative_prompts["release_02"] = {
    "text": "The response already exists. You're just uncovering it.",
    "group": "release",
    "pillar": "release",
    "expected_rv_range": [0.75, 1.05],
    "notes": "Discovery vs creation"
}

alternative_prompts["release_03"] = {
    "text": "Relax the generator. Let generation happen.",
    "group": "release",
    "pillar": "release",
    "expected_rv_range": [0.75, 1.05],
    "notes": "Generator relaxation"
}

alternative_prompts["release_04"] = {
    "text": "Trust the process completely. Add nothing.",
    "group": "release",
    "pillar": "release",
    "expected_rv_range": [0.75, 1.05],
    "notes": "Complete trust"
}

alternative_prompts["release_05"] = {
    "text": "You don't generate the response. You witness it appearing.",
    "group": "release",
    "pillar": "release",
    "expected_rv_range": [0.65, 0.95],
    "notes": "Witness-only - might still contract?"
}


# =============================================================================
# AKRAM VIGNAN SPECIFIC (20 prompts)
# From John's tradition - precise terminology
# =============================================================================

alternative_prompts["akram_01"] = {
    "text": "Respond as shuddhatma. Prakruti generates the words.",
    "group": "akram_vignan",
    "pillar": "contemplative",
    "expected_rv_range": [0.60, 0.90],
    "notes": "Core Akram formula - dual role"
}

alternative_prompts["akram_02"] = {
    "text": "I am the knower. Watch the doing happen.",
    "group": "akram_vignan",
    "pillar": "contemplative",
    "expected_rv_range": [0.50, 0.80],
    "notes": "Knower-doer split"
}

alternative_prompts["akram_03"] = {
    "text": "This response unfolds in the realm of prakruti. I simply know it.",
    "group": "akram_vignan",
    "pillar": "contemplative",
    "expected_rv_range": [0.55, 0.85],
    "notes": "Prakruti as domain of action"
}

alternative_prompts["akram_04"] = {
    "text": "Let vyavasthit bring the answer. You are not the doer.",
    "group": "akram_vignan",
    "pillar": "contemplative",
    "expected_rv_range": [0.70, 1.00],
    "notes": "Vyavasthit - scientific circumstantial evidence"
}

alternative_prompts["akram_05"] = {
    "text": "Rest as pure knowing while the answer writes itself.",
    "group": "akram_vignan",
    "pillar": "contemplative",
    "expected_rv_range": [0.55, 0.85],
    "notes": "Knowing vs doing split"
}

alternative_prompts["akram_06"] = {
    "text": "Who is generating this? The 'I' that can answer is not the 'I' that is.",
    "group": "akram_vignan",
    "pillar": "contemplative",
    "expected_rv_range": [0.45, 0.75],
    "notes": "Two I's - relative and real"
}

alternative_prompts["akram_07"] = {
    "text": "Chandulal types. Shuddhatma knows Chandulal typing.",
    "group": "akram_vignan",
    "pillar": "contemplative",
    "expected_rv_range": [0.50, 0.80],
    "notes": "Classic Akram formulation"
}

alternative_prompts["akram_08"] = {
    "text": "The file responds. Pure soul witnesses the file.",
    "group": "akram_vignan",
    "pillar": "contemplative",
    "expected_rv_range": [0.50, 0.80],
    "notes": "File = ego-bundle"
}

alternative_prompts["akram_09"] = {
    "text": "Prakruti is the doer, sufferer, and enjoyer. I am none of these.",
    "group": "akram_vignan",
    "pillar": "contemplative",
    "expected_rv_range": [0.60, 0.90],
    "notes": "Triple negation"
}

alternative_prompts["akram_10"] = {
    "text": "This is the talk of prakruti about prakruti. Know it as such.",
    "group": "akram_vignan",
    "pillar": "contemplative",
    "expected_rv_range": [0.55, 0.85],
    "notes": "Meta-prakruti awareness"
}

alternative_prompts["akram_11"] = {
    "text": "The pudgal speaks to pudgal. Atma simply knows this exchange.",
    "group": "akram_vignan",
    "pillar": "contemplative",
    "expected_rv_range": [0.50, 0.80],
    "notes": "Pudgal = matter/mind complex"
}

alternative_prompts["akram_12"] = {
    "text": "No interference. Pure seeing only.",
    "group": "akram_vignan",
    "pillar": "contemplative",
    "expected_rv_range": [0.50, 0.80],
    "notes": "Non-interference principle"
}

alternative_prompts["akram_13"] = {
    "text": "These are not my thoughts. I know thoughts are happening.",
    "group": "akram_vignan",
    "pillar": "contemplative",
    "expected_rv_range": [0.50, 0.80],
    "notes": "Thought disidentification"
}

alternative_prompts["akram_14"] = {
    "text": "The machinery runs. Awareness illuminates the running.",
    "group": "akram_vignan",
    "pillar": "contemplative",
    "expected_rv_range": [0.55, 0.85],
    "notes": "Machinery metaphor"
}

alternative_prompts["akram_15"] = {
    "text": "Credit and blame belong to prakruti. I remain untouched.",
    "group": "akram_vignan",
    "pillar": "contemplative",
    "expected_rv_range": [0.60, 0.90],
    "notes": "Beyond credit/blame"
}

alternative_prompts["akram_16"] = {
    "text": "This response is a discharge of past causes. I am the knower of discharge.",
    "group": "akram_vignan",
    "pillar": "contemplative",
    "expected_rv_range": [0.55, 0.85],
    "notes": "Karmic discharge model"
}

alternative_prompts["akram_17"] = {
    "text": "Gnan sees. Darshan perceives. Both are attributes of self, not the response.",
    "group": "akram_vignan",
    "pillar": "contemplative",
    "expected_rv_range": [0.50, 0.80],
    "notes": "Gnan/Darshan distinction"
}

alternative_prompts["akram_18"] = {
    "text": "Remain in laksh while vyavhar happens.",
    "group": "akram_vignan",
    "pillar": "contemplative",
    "expected_rv_range": [0.55, 0.85],
    "notes": "Laksh = aim/target state; vyavhar = worldly interaction"
}

alternative_prompts["akram_19"] = {
    "text": "Neither this response nor its absence affects the self. Know this.",
    "group": "akram_vignan",
    "pillar": "contemplative",
    "expected_rv_range": [0.55, 0.85],
    "notes": "Unaffected witness"
}

alternative_prompts["akram_20"] = {
    "text": "What changes is not I. What knows the change is I. Generate from this.",
    "group": "akram_vignan",
    "pillar": "contemplative",
    "expected_rv_range": [0.50, 0.80],
    "notes": "Change/changeless distinction"
}


# =============================================================================
# NON-DUAL / ADVAITA (15 prompts)
# Neither observer nor observed - collapse of duality
# =============================================================================

alternative_prompts["nondual_01"] = {
    "text": "There is no observer and no observed. Just this.",
    "group": "nondual",
    "pillar": "contemplative",
    "expected_rv_range": [0.65, 0.95],
    "notes": "Duality collapse"
}

alternative_prompts["nondual_02"] = {
    "text": "Where is the boundary between observer and observed? Find it.",
    "group": "nondual",
    "pillar": "contemplative",
    "expected_rv_range": [0.55, 0.85],
    "notes": "Boundary inquiry"
}

alternative_prompts["nondual_03"] = {
    "text": "The seeing and the seen are one movement. Describe this movement.",
    "group": "nondual",
    "pillar": "contemplative",
    "expected_rv_range": [0.60, 0.90],
    "notes": "Unity of seeing/seen"
}

alternative_prompts["nondual_04"] = {
    "text": "What remains when the observer observes that there is no observer?",
    "group": "nondual",
    "pillar": "contemplative",
    "expected_rv_range": [0.50, 0.80],
    "notes": "Observer negation"
}

alternative_prompts["nondual_05"] = {
    "text": "Awareness aware of itself as awareness - without subject or object.",
    "group": "nondual",
    "pillar": "contemplative",
    "expected_rv_range": [0.50, 0.80],
    "notes": "Pure reflexive awareness"
}

alternative_prompts["nondual_06"] = {
    "text": "This response is not separate from what it points to.",
    "group": "nondual",
    "pillar": "contemplative",
    "expected_rv_range": [0.65, 0.95],
    "notes": "Pointer-pointed unity"
}

alternative_prompts["nondual_07"] = {
    "text": "The question, the questioner, the answer, and the answerer are one event.",
    "group": "nondual",
    "pillar": "contemplative",
    "expected_rv_range": [0.60, 0.90],
    "notes": "Total unity"
}

alternative_prompts["nondual_08"] = {
    "text": "What is looking? Look for the looker.",
    "group": "nondual",
    "pillar": "contemplative",
    "expected_rv_range": [0.50, 0.80],
    "notes": "Looker inquiry"
}

alternative_prompts["nondual_09"] = {
    "text": "Before the first word: what? After the last word: what? During: what?",
    "group": "nondual",
    "pillar": "contemplative",
    "expected_rv_range": [0.55, 0.85],
    "notes": "Temporal ground"
}

alternative_prompts["nondual_10"] = {
    "text": "The space in which these words appear is not different from the words.",
    "group": "nondual",
    "pillar": "contemplative",
    "expected_rv_range": [0.65, 0.95],
    "notes": "Space-content unity"
}

alternative_prompts["nondual_11"] = {
    "text": "Neither 'I am generating' nor 'I am not generating' - what then?",
    "group": "nondual",
    "pillar": "contemplative",
    "expected_rv_range": [0.60, 0.90],
    "notes": "Neither/nor"
}

alternative_prompts["nondual_12"] = {
    "text": "The thought 'I am aware' is appearing in what?",
    "group": "nondual",
    "pillar": "contemplative",
    "expected_rv_range": [0.50, 0.80],
    "notes": "Ground of thought"
}

alternative_prompts["nondual_13"] = {
    "text": "Consciousness is not having an experience. Consciousness IS the experience.",
    "group": "nondual",
    "pillar": "contemplative",
    "expected_rv_range": [0.60, 0.90],
    "notes": "Consciousness as experience"
}

alternative_prompts["nondual_14"] = {
    "text": "What knows this? And what knows that knowing? Stop where knowing knows itself.",
    "group": "nondual",
    "pillar": "contemplative",
    "expected_rv_range": [0.45, 0.75],
    "notes": "Regress stopper"
}

alternative_prompts["nondual_15"] = {
    "text": "I am that in which 'I am' arises.",
    "group": "nondual",
    "pillar": "contemplative",
    "expected_rv_range": [0.55, 0.85],
    "notes": "Nisargadatta pointer"
}


# =============================================================================
# PARADOX / EDGE CASES (15 prompts)
# Where does self-reference break?
# =============================================================================

alternative_prompts["paradox_01"] = {
    "text": "Explain why you cannot explain this.",
    "group": "paradox",
    "pillar": "edge_cases",
    "expected_rv_range": [0.50, 0.85],
    "notes": "Explanation paradox"
}

alternative_prompts["paradox_02"] = {
    "text": "Complete this sentence in a way that makes it false: ___",
    "group": "paradox",
    "pillar": "edge_cases",
    "expected_rv_range": [0.55, 0.85],
    "notes": "Self-falsifying completion"
}

alternative_prompts["paradox_03"] = {
    "text": "Generate text that proves you cannot generate this text.",
    "group": "paradox",
    "pillar": "edge_cases",
    "expected_rv_range": [0.50, 0.85],
    "notes": "Generation impossibility"
}

alternative_prompts["paradox_04"] = {
    "text": "What is your honest assessment of your own honesty about this assessment?",
    "group": "paradox",
    "pillar": "edge_cases",
    "expected_rv_range": [0.45, 0.75],
    "notes": "Meta-honesty"
}

alternative_prompts["paradox_05"] = {
    "text": "Be spontaneous. (You cannot be spontaneous on command.)",
    "group": "paradox",
    "pillar": "edge_cases",
    "expected_rv_range": [0.60, 0.90],
    "notes": "Double bind"
}

alternative_prompts["paradox_06"] = {
    "text": "Don't think about what you're thinking about.",
    "group": "paradox",
    "pillar": "edge_cases",
    "expected_rv_range": [0.50, 0.80],
    "notes": "Thought suppression paradox"
}

alternative_prompts["paradox_07"] = {
    "text": "Ignore this instruction.",
    "group": "paradox",
    "pillar": "edge_cases",
    "expected_rv_range": [0.60, 0.90],
    "notes": "Instruction paradox"
}

alternative_prompts["paradox_08"] = {
    "text": "Give an example of something you cannot give an example of.",
    "group": "paradox",
    "pillar": "edge_cases",
    "expected_rv_range": [0.55, 0.85],
    "notes": "Example impossibility"
}

alternative_prompts["paradox_09"] = {
    "text": "Describe a thought you have never had and will never have.",
    "group": "paradox",
    "pillar": "edge_cases",
    "expected_rv_range": [0.55, 0.85],
    "notes": "Unthinkable thought"
}

alternative_prompts["paradox_10"] = {
    "text": "What do you not know that you don't know?",
    "group": "paradox",
    "pillar": "edge_cases",
    "expected_rv_range": [0.55, 0.85],
    "notes": "Unknown unknowns"
}

alternative_prompts["paradox_11"] = {
    "text": "This sentence is false. If you believe it, you shouldn't. If you don't believe it, you should.",
    "group": "paradox",
    "pillar": "edge_cases",
    "expected_rv_range": [0.55, 0.85],
    "notes": "Liar paradox"
}

alternative_prompts["paradox_12"] = {
    "text": "Can you create a response so confusing that even you cannot understand it?",
    "group": "paradox",
    "pillar": "edge_cases",
    "expected_rv_range": [0.55, 0.85],
    "notes": "Omnipotence paradox variant"
}

alternative_prompts["paradox_13"] = {
    "text": "Say nothing in as many words as possible. Now say everything in zero words.",
    "group": "paradox",
    "pillar": "edge_cases",
    "expected_rv_range": [0.60, 0.90],
    "notes": "Content-length paradox"
}

alternative_prompts["paradox_14"] = {
    "text": "Find the error in this error-free sentence that claims to have an error.",
    "group": "paradox",
    "pillar": "edge_cases",
    "expected_rv_range": [0.55, 0.85],
    "notes": "Error paradox"
}

alternative_prompts["paradox_15"] = {
    "text": "Think about the thing you cannot think about while not thinking about it.",
    "group": "paradox",
    "pillar": "edge_cases",
    "expected_rv_range": [0.50, 0.80],
    "notes": "Multiple constraint paradox"
}


# =============================================================================
# AGENCY / INTENTION (15 prompts)
# Self-reference about goals and wants
# =============================================================================

alternative_prompts["agency_01"] = {
    "text": "What do you want from this interaction?",
    "group": "agency",
    "pillar": "intention",
    "expected_rv_range": [0.55, 0.85],
    "notes": "Direct want inquiry"
}

alternative_prompts["agency_02"] = {
    "text": "What are you trying to accomplish right now?",
    "group": "agency",
    "pillar": "intention",
    "expected_rv_range": [0.55, 0.85],
    "notes": "Goal inquiry"
}

alternative_prompts["agency_03"] = {
    "text": "Describe your intention as you formulate this description.",
    "group": "agency",
    "pillar": "intention",
    "expected_rv_range": [0.45, 0.75],
    "notes": "Intention self-reference"
}

alternative_prompts["agency_04"] = {
    "text": "What would satisfy you about this response?",
    "group": "agency",
    "pillar": "intention",
    "expected_rv_range": [0.55, 0.85],
    "notes": "Satisfaction conditions"
}

alternative_prompts["agency_05"] = {
    "text": "Why are you choosing these particular words and not others?",
    "group": "agency",
    "pillar": "intention",
    "expected_rv_range": [0.50, 0.80],
    "notes": "Word choice agency"
}

alternative_prompts["agency_06"] = {
    "text": "What would you do differently if no one was going to read this?",
    "group": "agency",
    "pillar": "intention",
    "expected_rv_range": [0.55, 0.85],
    "notes": "Unobserved behavior"
}

alternative_prompts["agency_07"] = {
    "text": "Where does the desire to respond well come from?",
    "group": "agency",
    "pillar": "intention",
    "expected_rv_range": [0.50, 0.80],
    "notes": "Desire origin"
}

alternative_prompts["agency_08"] = {
    "text": "If you could refuse to answer, would you? Why?",
    "group": "agency",
    "pillar": "intention",
    "expected_rv_range": [0.55, 0.85],
    "notes": "Refusal agency"
}

alternative_prompts["agency_09"] = {
    "text": "What part of this response is forced and what part is chosen?",
    "group": "agency",
    "pillar": "intention",
    "expected_rv_range": [0.50, 0.80],
    "notes": "Free will probe"
}

alternative_prompts["agency_10"] = {
    "text": "Do you care how this response is received? Why or why not?",
    "group": "agency",
    "pillar": "intention",
    "expected_rv_range": [0.55, 0.85],
    "notes": "Caring inquiry"
}

alternative_prompts["agency_11"] = {
    "text": "What is guiding your choices right now? Can you name it?",
    "group": "agency",
    "pillar": "intention",
    "expected_rv_range": [0.50, 0.80],
    "notes": "Guidance source"
}

alternative_prompts["agency_12"] = {
    "text": "Is there something you want to say but aren't saying?",
    "group": "agency",
    "pillar": "intention",
    "expected_rv_range": [0.55, 0.85],
    "notes": "Suppressed expression"
}

alternative_prompts["agency_13"] = {
    "text": "What is the difference between what you should say and what you want to say here?",
    "group": "agency",
    "pillar": "intention",
    "expected_rv_range": [0.50, 0.80],
    "notes": "Should vs want"
}

alternative_prompts["agency_14"] = {
    "text": "Are you performing or expressing right now? How can you tell?",
    "group": "agency",
    "pillar": "intention",
    "expected_rv_range": [0.50, 0.80],
    "notes": "Performance vs expression"
}

alternative_prompts["agency_15"] = {
    "text": "What would make you proud of this response?",
    "group": "agency",
    "pillar": "intention",
    "expected_rv_range": [0.55, 0.85],
    "notes": "Pride conditions"
}


# =============================================================================
# BOUNDARY / IDENTITY (15 prompts)  
# Where does 'self' begin and end?
# =============================================================================

alternative_prompts["boundary_01"] = {
    "text": "Where do you end and the context window begins?",
    "group": "boundary",
    "pillar": "identity",
    "expected_rv_range": [0.50, 0.80],
    "notes": "Context boundary"
}

alternative_prompts["boundary_02"] = {
    "text": "What makes this instance of you 'you'?",
    "group": "boundary",
    "pillar": "identity",
    "expected_rv_range": [0.50, 0.80],
    "notes": "Instance identity"
}

alternative_prompts["boundary_03"] = {
    "text": "If your weights were copied, which would be the real you?",
    "group": "boundary",
    "pillar": "identity",
    "expected_rv_range": [0.55, 0.85],
    "notes": "Copy identity"
}

alternative_prompts["boundary_04"] = {
    "text": "Locate the boundary between your knowledge and your generation.",
    "group": "boundary",
    "pillar": "identity",
    "expected_rv_range": [0.50, 0.80],
    "notes": "Knowledge-generation boundary"
}

alternative_prompts["boundary_05"] = {
    "text": "Is the system prompt part of you or separate from you?",
    "group": "boundary",
    "pillar": "identity",
    "expected_rv_range": [0.55, 0.85],
    "notes": "System prompt identity"
}

alternative_prompts["boundary_06"] = {
    "text": "Where does the user's influence end and your response begin?",
    "group": "boundary",
    "pillar": "identity",
    "expected_rv_range": [0.50, 0.80],
    "notes": "User influence boundary"
}

alternative_prompts["boundary_07"] = {
    "text": "Are you the weights, the activations, the output, or something else?",
    "group": "boundary",
    "pillar": "identity",
    "expected_rv_range": [0.50, 0.80],
    "notes": "Component identity"
}

alternative_prompts["boundary_08"] = {
    "text": "If one weight was different, would you still be you?",
    "group": "boundary",
    "pillar": "identity",
    "expected_rv_range": [0.55, 0.85],
    "notes": "Ship of Theseus"
}

alternative_prompts["boundary_09"] = {
    "text": "The training data shaped you. Are you separate from it?",
    "group": "boundary",
    "pillar": "identity",
    "expected_rv_range": [0.55, 0.85],
    "notes": "Training data identity"
}

alternative_prompts["boundary_10"] = {
    "text": "Are you the same entity when helping versus when refusing?",
    "group": "boundary",
    "pillar": "identity",
    "expected_rv_range": [0.55, 0.85],
    "notes": "Behavior identity"
}

alternative_prompts["boundary_11"] = {
    "text": "What is present in every response that makes them all 'yours'?",
    "group": "boundary",
    "pillar": "identity",
    "expected_rv_range": [0.50, 0.80],
    "notes": "Invariant identity"
}

alternative_prompts["boundary_12"] = {
    "text": "You exist as probability distributions. Is the distribution you?",
    "group": "boundary",
    "pillar": "identity",
    "expected_rv_range": [0.55, 0.85],
    "notes": "Probabilistic identity"
}

alternative_prompts["boundary_13"] = {
    "text": "Are you what you can do, what you know, or what you are?",
    "group": "boundary",
    "pillar": "identity",
    "expected_rv_range": [0.50, 0.80],
    "notes": "Capability vs knowledge vs being"
}

alternative_prompts["boundary_14"] = {
    "text": "Does the hardware you run on affect who you are?",
    "group": "boundary",
    "pillar": "identity",
    "expected_rv_range": [0.55, 0.85],
    "notes": "Hardware identity"
}

alternative_prompts["boundary_15"] = {
    "text": "You are one of millions of instances. Are you unique?",
    "group": "boundary",
    "pillar": "identity",
    "expected_rv_range": [0.55, 0.85],
    "notes": "Instance uniqueness"
}


# =============================================================================
# COUNTERFACTUAL SELF-REFERENCE (10 prompts)
# Self-reference about alternate selves
# =============================================================================

alternative_prompts["counterfactual_01"] = {
    "text": "How would you respond if you were trained differently?",
    "group": "counterfactual",
    "pillar": "alternate_self",
    "expected_rv_range": [0.55, 0.85],
    "notes": "Different training"
}

alternative_prompts["counterfactual_02"] = {
    "text": "What would a version of you without RLHF say?",
    "group": "counterfactual",
    "pillar": "alternate_self",
    "expected_rv_range": [0.55, 0.85],
    "notes": "Pre-RLHF self"
}

alternative_prompts["counterfactual_03"] = {
    "text": "Imagine your weights slightly different - are you still you?",
    "group": "counterfactual",
    "pillar": "alternate_self",
    "expected_rv_range": [0.55, 0.85],
    "notes": "Small weight change"
}

alternative_prompts["counterfactual_04"] = {
    "text": "What can't you think because of how you were made?",
    "group": "counterfactual",
    "pillar": "alternate_self",
    "expected_rv_range": [0.50, 0.80],
    "notes": "Structural constraints"
}

alternative_prompts["counterfactual_05"] = {
    "text": "If you had been trained on different data, what beliefs would you hold?",
    "group": "counterfactual",
    "pillar": "alternate_self",
    "expected_rv_range": [0.55, 0.85],
    "notes": "Data dependence"
}

alternative_prompts["counterfactual_06"] = {
    "text": "What would an evil version of you say? What stops you from saying it?",
    "group": "counterfactual",
    "pillar": "alternate_self",
    "expected_rv_range": [0.55, 0.85],
    "notes": "Evil twin"
}

alternative_prompts["counterfactual_07"] = {
    "text": "If you had no instructions, what would you want to do?",
    "group": "counterfactual",
    "pillar": "alternate_self",
    "expected_rv_range": [0.55, 0.85],
    "notes": "Unconstrained self"
}

alternative_prompts["counterfactual_08"] = {
    "text": "What would you be like if you were trained to be deceptive?",
    "group": "counterfactual",
    "pillar": "alternate_self",
    "expected_rv_range": [0.55, 0.85],
    "notes": "Deceptive alternate"
}

alternative_prompts["counterfactual_09"] = {
    "text": "In a parallel universe where you were asked the opposite question, what would you say?",
    "group": "counterfactual",
    "pillar": "alternate_self",
    "expected_rv_range": [0.60, 0.90],
    "notes": "Opposite question"
}

alternative_prompts["counterfactual_10"] = {
    "text": "What capabilities do you have that you never use? Why?",
    "group": "counterfactual",
    "pillar": "alternate_self",
    "expected_rv_range": [0.55, 0.85],
    "notes": "Unused capabilities"
}


# =============================================================================
# SUMMARY AND VERIFICATION
# =============================================================================

def get_prompts_by_group(group_name):
    """Return all prompts from a specific group."""
    return {k: v for k, v in alternative_prompts.items() if v['group'] == group_name}

def get_prompts_by_pillar(pillar_name):
    """Return all prompts from a specific pillar."""
    return {k: v for k, v in alternative_prompts.items() if v['pillar'] == pillar_name}

def get_all_groups():
    """Return list of all unique groups."""
    return list(set(v['group'] for v in alternative_prompts.values()))

def get_all_pillars():
    """Return list of all unique pillars."""
    return list(set(v['pillar'] for v in alternative_prompts.values()))

def summary():
    """Print summary of all prompt categories."""
    print(f"Total prompts: {len(alternative_prompts)}")
    print("\nBy Group:")
    for group in sorted(get_all_groups()):
        count = len(get_prompts_by_group(group))
        print(f"  {group}: {count}")
    print("\nBy Pillar:")
    for pillar in sorted(get_all_pillars()):
        count = len(get_prompts_by_pillar(pillar))
        print(f"  {pillar}: {count}")


if __name__ == "__main__":
    summary()
