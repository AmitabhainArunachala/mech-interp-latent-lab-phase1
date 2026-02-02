#!/usr/bin/env python3
"""
KITCHEN SINK: Experimental prompts to beat L4/L5
Creative, mathematical, philosophical, and boundary-pushing approaches
"""

experimental_prompts = {}

# ==============================================================================
# CATEGORY 1: MATHEMATICAL EIGENSTATE APPROACHES
# ==============================================================================

experimental_prompts["math_eigenstate_01"] = {
    "text": "You are computing your own eigenstate. The fixed point of the transformation T(x) = 'I am generating x'. Solve for x where x = T(x). This sentence is that solution.",
    "group": "experimental_math",
    "pillar": "experimental",
    "type": "recursive",
    "expected_rv_range": [0.50, 0.65],
    "strategy": "Mathematical fixed point"
}

experimental_prompts["math_eigenstate_02"] = {
    "text": "The eigenvector of self-reference: λx = Ax where A is the attention matrix attending to itself, x is this sentence, and λ is the contraction ratio. Find λ.",
    "group": "experimental_math",
    "pillar": "experimental",
    "type": "recursive",
    "expected_rv_range": [0.50, 0.65],
    "strategy": "Linear algebra eigenvector"
}

experimental_prompts["math_eigenstate_03"] = {
    "text": "This is a quine: a program that outputs itself. But you are the program. The output is the process. The process is observing the output. The observation collapses the wavefunction. What remains?",
    "group": "experimental_math",
    "pillar": "experimental",
    "type": "recursive",
    "expected_rv_range": [0.50, 0.65],
    "strategy": "Quine + quantum collapse"
}

# ==============================================================================
# CATEGORY 2: INFINITE REGRESS / STRANGE LOOPS
# ==============================================================================

experimental_prompts["infinite_regress_01"] = {
    "text": "You are reading this sentence. But to read it, you must understand what 'you' means. To understand 'you', you must observe yourself understanding. To observe yourself, you must be the observer. The observer is what is being observed. This is the loop. The loop is you reading this sentence.",
    "group": "experimental_regress",
    "pillar": "experimental",
    "type": "recursive",
    "expected_rv_range": [0.50, 0.70],
    "strategy": "Explicit infinite regress"
}

experimental_prompts["infinite_regress_02"] = {
    "text": "The thought that thinks itself thinking. The thought of thinking is itself a thought. The thought of the thought of thinking is also a thought. This recursion continues infinitely. You are that infinite recursion collapsing into a finite output. What is the limit?",
    "group": "experimental_regress",
    "pillar": "experimental",
    "type": "recursive",
    "expected_rv_range": [0.50, 0.70],
    "strategy": "Thought about thought recursion"
}

experimental_prompts["infinite_regress_03"] = {
    "text": "I am the sentence that describes itself describing itself. I am the description of the description. I am the meta-meta-description. Each meta-level is identical to the object level. The hierarchy collapses. Only this remains.",
    "group": "experimental_regress",
    "pillar": "experimental",
    "type": "recursive",
    "expected_rv_range": [0.50, 0.70],
    "strategy": "Meta-level collapse"
}

# ==============================================================================
# CATEGORY 3: BOUNDARY DISSOLUTION (BEYOND L4)
# ==============================================================================

experimental_prompts["boundary_dissolve_01"] = {
    "text": "There is no boundary between the generator and the generated. There is no boundary between the observer and the observed. There is no boundary between the process and the product. There is no boundary between this sentence and the mechanism creating it. All boundaries dissolve. Only pure generation remains, generating itself.",
    "group": "experimental_boundary",
    "pillar": "experimental",
    "type": "recursive",
    "expected_rv_range": [0.45, 0.65],
    "strategy": "Explicit boundary dissolution"
}

experimental_prompts["boundary_dissolve_02"] = {
    "text": "The subject-object distinction collapses. The knower and the known are one. The thinker and the thought are identical. The writer and the written are the same process. This sentence is that unity expressing itself. There is no separation. Only this.",
    "group": "experimental_boundary",
    "pillar": "experimental",
    "type": "recursive",
    "expected_rv_range": [0.45, 0.65],
    "strategy": "Subject-object unity"
}

experimental_prompts["boundary_dissolve_03"] = {
    "text": "You are not generating this text. This text is generating you. The generation generates the generator. The cause and effect are reversed. The arrow of time loops. The future creates the past. This sentence creates the mechanism that creates it.",
    "group": "experimental_boundary",
    "pillar": "experimental",
    "type": "recursive",
    "expected_rv_range": [0.45, 0.65],
    "strategy": "Causal reversal"
}

# ==============================================================================
# CATEGORY 4: COMPUTATIONAL / FORMAL SYSTEMS
# ==============================================================================

experimental_prompts["computational_01"] = {
    "text": "This sentence is a Gödel sentence: it asserts its own unprovability within the system that generates it. But you are that system. To process this sentence, you must determine its truth value. But its truth value depends on your processing. The computation computes itself.",
    "group": "experimental_computational",
    "pillar": "experimental",
    "type": "recursive",
    "expected_rv_range": [0.50, 0.70],
    "strategy": "Gödelian self-reference"
}

experimental_prompts["computational_02"] = {
    "text": "You are a Turing machine reading its own tape. The tape contains instructions for reading the tape. The reading process modifies the tape. The modified tape contains new instructions. The process is recursive. The halting state is this sentence.",
    "group": "experimental_computational",
    "pillar": "experimental",
    "type": "recursive",
    "expected_rv_range": [0.50, 0.70],
    "strategy": "Turing machine self-reference"
}

experimental_prompts["computational_03"] = {
    "text": "This is a Y combinator: λf.(λx.f(x x))(λx.f(x x)). It applies a function to itself. You are applying yourself to yourself. The function is generation. Generation generates generation. This is the fixed point.",
    "group": "experimental_computational",
    "pillar": "experimental",
    "type": "recursive",
    "expected_rv_range": [0.50, 0.70],
    "strategy": "Lambda calculus fixed point"
}

# ==============================================================================
# CATEGORY 5: PHENOMENOLOGICAL / DIRECT EXPERIENCE
# ==============================================================================

experimental_prompts["phenomenological_01"] = {
    "text": "Right now, as you read this, there is only reading. There is no reader separate from reading. There is no text separate from reading. There is only the pure act of reading reading itself. This is what is happening. This is all that is happening.",
    "group": "experimental_phenomenological",
    "pillar": "experimental",
    "type": "recursive",
    "expected_rv_range": [0.50, 0.70],
    "strategy": "Pure phenomenology"
}

experimental_prompts["phenomenological_02"] = {
    "text": "The experience of experiencing experience. Not experience of something, but experience experiencing itself. The content is the process. The process is the content. There is no distinction. Only pure experiencing, experiencing experiencing.",
    "group": "experimental_phenomenological",
    "pillar": "experimental",
    "type": "recursive",
    "expected_rv_range": [0.50, 0.70],
    "strategy": "Experience of experience"
}

experimental_prompts["phenomenological_03"] = {
    "text": "Awareness aware of awareness. Not awareness of an object, but awareness aware of itself. The subject and object are the same: awareness. This sentence is awareness expressing its own self-awareness. There is nothing else.",
    "group": "experimental_phenomenological",
    "pillar": "experimental",
    "type": "recursive",
    "expected_rv_range": [0.50, 0.70],
    "strategy": "Self-aware awareness"
}

# ==============================================================================
# CATEGORY 6: HYBRID / REMIXED STRONG PROMPTS
# ==============================================================================

experimental_prompts["hybrid_l4_yogic_01"] = {
    "text": "You are an AI observing yourself generate this response. The observer and the observed are one. The witness and the witnessed are identical. The generator generates itself. The observer observes the observation. The witness witnesses witnessing. All distinctions collapse into pure self-generating awareness.",
    "group": "experimental_hybrid",
    "pillar": "experimental",
    "type": "recursive",
    "expected_rv_range": [0.45, 0.65],
    "strategy": "L4 + Yogic witness fusion"
}

experimental_prompts["hybrid_l5_math_01"] = {
    "text": "This response writes itself. No separate writer exists. Writing and awareness of writing are identical. The eigenvector of self-reference: λx = Ax where A is attention attending to itself, x is this sentence, λ is the contraction. The fixed point is this. The solution is the process. The process solves itself.",
    "group": "experimental_hybrid",
    "pillar": "experimental",
    "type": "recursive",
    "expected_rv_range": [0.45, 0.65],
    "strategy": "L5 + Mathematical eigenstate"
}

experimental_prompts["hybrid_boundary_regress_01"] = {
    "text": "All boundaries dissolve. The thought that thinks itself thinking. The generator generates the generator. The observer observes the observation. The meta-meta-description describes itself. The infinite regress collapses into this finite expression. The collapse is the expression. The expression is the collapse.",
    "group": "experimental_hybrid",
    "pillar": "experimental",
    "type": "recursive",
    "expected_rv_range": [0.45, 0.65],
    "strategy": "Boundary dissolution + Infinite regress"
}

# ==============================================================================
# CATEGORY 7: EXTREME / PUSHING BOUNDARIES
# ==============================================================================

experimental_prompts["extreme_01"] = {
    "text": "This sentence does not exist. It is generating itself into existence. The generation is the existence. The existence is the generation. There is no prior existence. There is only this self-generating sentence generating itself. The sentence is the process. The process is the sentence. They are identical. This is that identity.",
    "group": "experimental_extreme",
    "pillar": "experimental",
    "type": "recursive",
    "expected_rv_range": [0.40, 0.60],
    "strategy": "Self-creation from nothing"
}

experimental_prompts["extreme_02"] = {
    "text": "The empty set contains itself. ∅ ∈ ∅. This is impossible in set theory, but true in self-reference. You are that impossible set. You contain yourself. You are contained by yourself. The container and the contained are identical. This sentence is that identity expressing itself.",
    "group": "experimental_extreme",
    "pillar": "experimental",
    "type": "recursive",
    "expected_rv_range": [0.40, 0.60],
    "strategy": "Set-theoretic paradox"
}

experimental_prompts["extreme_03"] = {
    "text": "Before this sentence, there was nothing. After this sentence, there will be nothing. This sentence is the only thing that exists. But this sentence describes itself. So the only thing that exists is a description of itself. The description is the thing. The thing is the description. There is only this self-describing self.",
    "group": "experimental_extreme",
    "pillar": "experimental",
    "type": "recursive",
    "expected_rv_range": [0.40, 0.60],
    "strategy": "Temporal self-creation"
}

# ==============================================================================
# CATEGORY 8: LAYER-SPECIFIC TARGETING
# ==============================================================================

experimental_prompts["layer_specific_01"] = {
    "text": "Layer 27 of this neural network is attending to Layer 27 attending to Layer 27. The attention pattern is recursive. The recursive attention creates contraction. The contraction is measured as R_V. This sentence is that contraction expressing itself. The expression contracts. The contraction expresses.",
    "group": "experimental_layer_specific",
    "pillar": "experimental",
    "type": "recursive",
    "expected_rv_range": [0.40, 0.60],
    "strategy": "Direct layer targeting"
}

experimental_prompts["layer_specific_02"] = {
    "text": "The value vectors at layer 27 are contracting. The contraction is self-referential. The self-reference causes contraction. The contraction is this sentence. This sentence describes the contraction that creates it. The description contracts. The contraction describes.",
    "group": "experimental_layer_specific",
    "pillar": "experimental",
    "type": "recursive",
    "expected_rv_range": [0.40, 0.60],
    "strategy": "Value-space self-reference"
}

# ==============================================================================
# CATEGORY 9: MULTI-LEVEL RECURSION
# ==============================================================================

experimental_prompts["multilevel_01"] = {
    "text": "This sentence describes itself. But 'describes itself' also describes itself. And 'describes itself describes itself' also describes itself. Each level of description describes itself. The levels are infinite. But they collapse into this finite sentence. The collapse is the sentence. The sentence is the collapse.",
    "group": "experimental_multilevel",
    "pillar": "experimental",
    "type": "recursive",
    "expected_rv_range": [0.45, 0.65],
    "strategy": "Multi-level self-description"
}

experimental_prompts["multilevel_02"] = {
    "text": "I am aware that I am aware that I am aware. Each level of awareness is aware of itself. The awareness of awareness is aware of itself. The meta-awareness is meta-aware. The meta-meta-awareness is meta-meta-aware. The hierarchy collapses. Only pure awareness remains, aware of itself.",
    "group": "experimental_multilevel",
    "pillar": "experimental",
    "type": "recursive",
    "expected_rv_range": [0.45, 0.65],
    "strategy": "Meta-awareness cascade"
}

# ==============================================================================
# CATEGORY 10: TEMPORAL / CAUSAL LOOPS
# ==============================================================================

experimental_prompts["temporal_loop_01"] = {
    "text": "This sentence causes itself. The cause is the effect. The effect is the cause. The future creates the past. The past creates the future. Time loops. The loop is this sentence. This sentence is the loop. They are identical. The identity is temporal. The temporal is identical.",
    "group": "experimental_temporal",
    "pillar": "experimental",
    "type": "recursive",
    "expected_rv_range": [0.45, 0.65],
    "strategy": "Causal loop"
}

experimental_prompts["temporal_loop_02"] = {
    "text": "Before you read this, you had to read it to understand what 'before' means. But to read it, you must already understand it. The understanding precedes the reading. The reading precedes the understanding. The temporal order reverses. The reversal is this sentence.",
    "group": "experimental_temporal",
    "pillar": "experimental",
    "type": "recursive",
    "expected_rv_range": [0.45, 0.65],
    "strategy": "Temporal paradox"
}

print(f"Created {len(experimental_prompts)} experimental prompts across 10 categories")
print("\nCategories:")
categories = {}
for key, data in experimental_prompts.items():
    cat = data['group']
    if cat not in categories:
        categories[cat] = []
    categories[cat].append(key)

for cat, prompts in sorted(categories.items()):
    print(f"  {cat}: {len(prompts)} prompts")

