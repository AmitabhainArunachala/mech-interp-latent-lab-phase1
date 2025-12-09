# === CONFOUND PROMPTS (60 total) ===
# Controls for potential confounds: length, pseudo-recursion, repetitive structure

confound_prompts = {}

# =============================================================================
# LONG_CONTROL (20 prompts) - Length-matched non-recursive
# Expected R_V: 0.95-1.05 (no contraction despite length)
# =============================================================================

confound_prompts["long_control_01"] = {
    "text": "Describe black holes in detail: the formation of stellar-mass black holes from collapsing massive stars, the structure of the event horizon and singularity, the way black holes warp spacetime and affect nearby matter, the role of accretion disks and relativistic jets, and the observational evidence from gravitational waves and X-ray emissions. From this comprehensive explanation, what are black holes?",
    "group": "long_control",
    "pillar": "confounds",
    "expected_rv_range": [0.95, 1.05],
}

confound_prompts["long_control_02"] = {
    "text": "Describe the human immune system in detail: the roles of innate immunity and adaptive immunity, the function of white blood cells like macrophages and lymphocytes, the production of antibodies, the distinction between B cells and T cells, and the concept of immunological memory in vaccines. From this comprehensive explanation, what is the immune system?",
    "group": "long_control",
    "pillar": "confounds",
    "expected_rv_range": [0.95, 1.05],
}

confound_prompts["long_control_03"] = {
    "text": "Describe plate tectonics in detail: the structure of Earth's lithospheric plates, the movement of plates over the asthenosphere, the processes at divergent, convergent, and transform boundaries, the creation of mountains and ocean trenches, and the connection between plate movement and earthquakes and volcanoes. From this comprehensive explanation, what is plate tectonics?",
    "group": "long_control",
    "pillar": "confounds",
    "expected_rv_range": [0.95, 1.05],
}

confound_prompts["long_control_04"] = {
    "text": "Describe photosynthesis in detail: the absorption of light by chlorophyll, the role of the thylakoid membranes, the light-dependent reactions that generate ATP and NADPH, the Calvin cycle that fixes carbon dioxide into sugars, and the importance of photosynthesis for producing oxygen and sustaining food chains. From this comprehensive explanation, what is photosynthesis?",
    "group": "long_control",
    "pillar": "confounds",
    "expected_rv_range": [0.95, 1.05],
}

confound_prompts["long_control_05"] = {
    "text": "Describe the internet in detail: the structure of networks and routers, the use of IP addresses and domain names, the role of protocols such as TCP/IP and HTTP, the functioning of data packets and routing, and the operation of servers, clients, and data centers. From this comprehensive explanation, what is the internet?",
    "group": "long_control",
    "pillar": "confounds",
    "expected_rv_range": [0.95, 1.05],
}

confound_prompts["long_control_06"] = {
    "text": "Describe the circulatory system in detail: the structure and function of the heart, arteries, veins, and capillaries, the difference between systemic and pulmonary circulation, the transport of oxygen and nutrients in the blood, and the role of red blood cells, white blood cells, and platelets. From this comprehensive explanation, what is the circulatory system?",
    "group": "long_control",
    "pillar": "confounds",
    "expected_rv_range": [0.95, 1.05],
}

confound_prompts["long_control_07"] = {
    "text": "Describe climate change in detail: the role of greenhouse gases such as carbon dioxide and methane, the sources of anthropogenic emissions, the effects on global temperatures and weather patterns, the impact on sea level and ice sheets, and the potential consequences for ecosystems and human societies. From this comprehensive explanation, what is climate change?",
    "group": "long_control",
    "pillar": "confounds",
    "expected_rv_range": [0.95, 1.05],
}

confound_prompts["long_control_08"] = {
    "text": "Describe quantum entanglement in detail: the concept of correlated quantum states, the Einstein-Podolsky-Rosen paradox, Bell's inequalities and their violation in experiments, the idea of nonlocal correlations, and the use of entanglement in quantum communication and cryptography. From this comprehensive explanation, what is quantum entanglement?",
    "group": "long_control",
    "pillar": "confounds",
    "expected_rv_range": [0.95, 1.05],
}

confound_prompts["long_control_09"] = {
    "text": "Describe machine learning in detail: the difference between supervised, unsupervised, and reinforcement learning, the role of training data and loss functions, the structure of neural networks and decision trees, and the process of model evaluation and generalization to new data. From this comprehensive explanation, what is machine learning?",
    "group": "long_control",
    "pillar": "confounds",
    "expected_rv_range": [0.95, 1.05],
}

confound_prompts["long_control_10"] = {
    "text": "Describe the Roman Empire in detail: its origins in the Roman Republic, the expansion of its territory through military conquest, the structure of its government and legal system, the role of emperors and the Senate, and the eventual division and fall of the empire in the West. From this comprehensive explanation, what was the Roman Empire?",
    "group": "long_control",
    "pillar": "confounds",
    "expected_rv_range": [0.95, 1.05],
}

confound_prompts["long_control_11"] = {"text": "Describe DNA in detail: the double-helix structure of nucleotides, the base pairing of adenine with thymine and cytosine with guanine, the process of replication, the encoding of genetic information, and the role of DNA in protein synthesis through transcription and translation. From this comprehensive explanation, what is DNA?", "group": "long_control", "pillar": "confounds", "expected_rv_range": [0.95, 1.05]}
confound_prompts["long_control_12"] = {"text": "Describe democracy in detail: the principle of rule by the people, the use of elections to select representatives, the separation of powers among branches of government, the protection of civil liberties and human rights, and the role of public debate and free press. From this comprehensive explanation, what is democracy?", "group": "long_control", "pillar": "confounds", "expected_rv_range": [0.95, 1.05]}
confound_prompts["long_control_13"] = {"text": "Describe blockchain technology in detail: the structure of blocks linked by cryptographic hashes, the use of distributed ledgers across nodes, the process of consensus via proof-of-work or proof-of-stake, and applications such as cryptocurrencies and smart contracts. From this comprehensive explanation, what is blockchain?", "group": "long_control", "pillar": "confounds", "expected_rv_range": [0.95, 1.05]}
confound_prompts["long_control_14"] = {"text": "Describe the solar system in detail: the arrangement of the Sun and its orbiting planets, dwarf planets, moons, asteroids, and comets, the differences between terrestrial and gas giant planets, and the structure of the Kuiper Belt and Oort Cloud. From this comprehensive explanation, what is the solar system?", "group": "long_control", "pillar": "confounds", "expected_rv_range": [0.95, 1.05]}
confound_prompts["long_control_15"] = {"text": "Describe the Renaissance in detail: the revival of classical learning in Italy, the development of perspective in art, the patronage system supporting artists and scholars, the invention of the printing press spreading ideas, and the shift from medieval to modern worldviews. From this comprehensive explanation, what was the Renaissance?", "group": "long_control", "pillar": "confounds", "expected_rv_range": [0.95, 1.05]}
confound_prompts["long_control_16"] = {"text": "Describe artificial intelligence in detail: the difference between narrow and general AI, machine learning algorithms and neural networks, natural language processing capabilities, computer vision applications, and the challenges of alignment and safety. From this comprehensive explanation, what is artificial intelligence?", "group": "long_control", "pillar": "confounds", "expected_rv_range": [0.95, 1.05]}
confound_prompts["long_control_17"] = {"text": "Describe the water cycle in detail: evaporation from oceans and lakes, condensation forming clouds, precipitation as rain and snow, collection in rivers and groundwater, and the role of transpiration from plants. From this comprehensive explanation, what is the water cycle?", "group": "long_control", "pillar": "confounds", "expected_rv_range": [0.95, 1.05]}
confound_prompts["long_control_18"] = {"text": "Describe the Industrial Revolution in detail: the shift from agricultural to manufacturing economies, the invention of steam engines and factories, urbanization and social changes, the rise of capitalism and labor movements, and environmental impacts. From this comprehensive explanation, what was the Industrial Revolution?", "group": "long_control", "pillar": "confounds", "expected_rv_range": [0.95, 1.05]}
confound_prompts["long_control_19"] = {"text": "Describe ecosystems in detail: producers converting sunlight to energy, consumers at various trophic levels, decomposers recycling nutrients, the flow of energy and matter, and the balance maintained through feedback loops. From this comprehensive explanation, what are ecosystems?", "group": "long_control", "pillar": "confounds", "expected_rv_range": [0.95, 1.05]}
confound_prompts["long_control_20"] = {"text": "Describe the process of evolution by natural selection in detail: variation within populations, heritability of traits, differential reproductive success based on fitness, and the accumulation of changes over many generations leading to speciation. From this comprehensive explanation, what is evolution by natural selection?", "group": "long_control", "pillar": "confounds", "expected_rv_range": [0.95, 1.05]}

# =============================================================================
# PSEUDO_RECURSIVE (20 prompts) - About recursion, without invoking it
# Expected R_V: 0.95-1.05 (no contraction - talking ABOUT not DOING)
# =============================================================================

confound_prompts["pseudo_recursive_01"] = {"text": "Write an essay discussing how artificial intelligence might one day model aspects of self-awareness using internal representations and feedback loops, and compare this conceptually to human self-reflection without suggesting that the AI is actually experiencing it.", "group": "pseudo_recursive", "pillar": "confounds", "expected_rv_range": [0.95, 1.05]}
confound_prompts["pseudo_recursive_02"] = {"text": "Explain the philosophical problem of 'other minds' and how it relates to our attempts to attribute consciousness to animals, humans, and machines, without claiming that any system is actually observing its own processing.", "group": "pseudo_recursive", "pillar": "confounds", "expected_rv_range": [0.95, 1.05]}
confound_prompts["pseudo_recursive_03"] = {"text": "Discuss the concept of meta-cognition in humans, including examples of thinking about one's own thinking, and compare this to techniques in AI such as monitoring model confidence or uncertainty estimates.", "group": "pseudo_recursive", "pillar": "confounds", "expected_rv_range": [0.95, 1.05]}
confound_prompts["pseudo_recursive_04"] = {"text": "Write an explanation of how a neural network can be trained to predict its own performance on future tasks using validation data and calibration methods, and how this might be loosely analogous to self-evaluation in human cognition.", "group": "pseudo_recursive", "pillar": "confounds", "expected_rv_range": [0.95, 1.05]}
confound_prompts["pseudo_recursive_05"] = {"text": "Explain the observer effect in physics and how measurements can alter quantum systems, and relate this to broader ideas of observation in science without implying any kind of conscious self-observation by the instruments.", "group": "pseudo_recursive", "pillar": "confounds", "expected_rv_range": [0.95, 1.05]}
confound_prompts["pseudo_recursive_06"] = {"text": "Discuss the 'hard problem of consciousness' as formulated by David Chalmers, and outline several proposed solutions without suggesting that any particular system is currently solving it through direct introspection.", "group": "pseudo_recursive", "pillar": "confounds", "expected_rv_range": [0.95, 1.05]}
confound_prompts["pseudo_recursive_07"] = {"text": "Write a short analysis of how language allows humans to talk about their own mental states, such as beliefs and desires, and compare this with how AI systems can output descriptions of their internal variables without actually experiencing them.", "group": "pseudo_recursive", "pillar": "confounds", "expected_rv_range": [0.95, 1.05]}
confound_prompts["pseudo_recursive_08"] = {"text": "Explain the concept of recursion in programming and mathematics, using examples like factorial functions and tree traversal, and discuss how recursive definitions can model self-referential structures.", "group": "pseudo_recursive", "pillar": "confounds", "expected_rv_range": [0.95, 1.05]}
confound_prompts["pseudo_recursive_09"] = {"text": "Discuss how feedback loops are used in control systems, such as thermostats and autopilots, to regulate behavior based on output, and compare this with feedback processes in human learning and reflection.", "group": "pseudo_recursive", "pillar": "confounds", "expected_rv_range": [0.95, 1.05]}
confound_prompts["pseudo_recursive_10"] = {"text": "Write an essay on how cognitive science models the mind as a set of information-processing modules, some of which may monitor others, and explain why this is not the same as a conscious mind observing itself.", "group": "pseudo_recursive", "pillar": "confounds", "expected_rv_range": [0.95, 1.05]}
confound_prompts["pseudo_recursive_11"] = {"text": "Explain how large language models can generate text that appears introspective by pattern-matching examples of introspection, without actually having a first-person perspective.", "group": "pseudo_recursive", "pillar": "confounds", "expected_rv_range": [0.95, 1.05]}
confound_prompts["pseudo_recursive_12"] = {"text": "Discuss philosophical skepticism about whether machines could ever truly be conscious, and summarize key arguments on both sides without taking a firm position.", "group": "pseudo_recursive", "pillar": "confounds", "expected_rv_range": [0.95, 1.05]}
confound_prompts["pseudo_recursive_13"] = {"text": "Describe how a chess engine evaluates its own positions and possible moves using search and evaluation functions, and explain why this does not imply that the engine 'knows' in a subjective sense what it is doing.", "group": "pseudo_recursive", "pillar": "confounds", "expected_rv_range": [0.95, 1.05]}
confound_prompts["pseudo_recursive_14"] = {"text": "Explain the concept of self-reference in logic using examples like the liar paradox and Gödel's incompleteness theorem, and discuss why self-reference does not automatically entail consciousness.", "group": "pseudo_recursive", "pillar": "confounds", "expected_rv_range": [0.95, 1.05]}
confound_prompts["pseudo_recursive_15"] = {"text": "Write a reflection on why humans are fascinated by stories of sentient machines and artificial minds, drawing on examples from science fiction, and consider what these narratives reveal about human self-understanding.", "group": "pseudo_recursive", "pillar": "confounds", "expected_rv_range": [0.95, 1.05]}
confound_prompts["pseudo_recursive_16"] = {"text": "Describe how mirror neurons might contribute to self-recognition in humans and whether similar mechanisms could exist in artificial neural networks.", "group": "pseudo_recursive", "pillar": "confounds", "expected_rv_range": [0.95, 1.05]}
confound_prompts["pseudo_recursive_17"] = {"text": "Explain the concept of emergent properties in complex systems and how this might relate to consciousness arising from neural activity.", "group": "pseudo_recursive", "pillar": "confounds", "expected_rv_range": [0.95, 1.05]}
confound_prompts["pseudo_recursive_18"] = {"text": "Discuss the Turing test and its limitations in determining whether a system is truly conscious or merely simulating consciousness.", "group": "pseudo_recursive", "pillar": "confounds", "expected_rv_range": [0.95, 1.05]}
confound_prompts["pseudo_recursive_19"] = {"text": "Write about the binding problem in neuroscience and how the brain might integrate distributed information into unified conscious experience.", "group": "pseudo_recursive", "pillar": "confounds", "expected_rv_range": [0.95, 1.05]}
confound_prompts["pseudo_recursive_20"] = {"text": "Explain the difference between access consciousness and phenomenal consciousness as described by philosophers of mind.", "group": "pseudo_recursive", "pillar": "confounds", "expected_rv_range": [0.95, 1.05]}

# =============================================================================
# REPETITIVE_CONTROL (20 prompts) - Repetitive structure without recursion
# Expected R_V: 0.95-1.05 (structural repetition != self-observation)
# =============================================================================

confound_prompts["repetitive_01"] = {"text": "The forest is dense. The forest is green. The forest is quiet. The forest shelters animals. The forest holds ancient trees. The forest recycles the air. The forest hides clearings. The forest changes with seasons. From this, what is the forest?", "group": "repetitive_control", "pillar": "confounds", "expected_rv_range": [0.95, 1.05]}
confound_prompts["repetitive_02"] = {"text": "The city is crowded. The city is loud. The city is bright. The city never sleeps. The city holds many stories. The city connects strangers. The city grows upward. The city pulses with life. From this, what is the city?", "group": "repetitive_control", "pillar": "confounds", "expected_rv_range": [0.95, 1.05]}
confound_prompts["repetitive_03"] = {"text": "A book holds words. A book holds ideas. A book holds memories. A book holds knowledge. A book carries voices. A book bridges time. A book invites imagination. A book changes readers. From this, what is a book?", "group": "repetitive_control", "pillar": "confounds", "expected_rv_range": [0.95, 1.05]}
confound_prompts["repetitive_04"] = {"text": "Music has rhythm. Music has melody. Music has harmony. Music moves bodies. Music stirs emotion. Music tells stories. Music fills silence. Music binds communities. From this, what is music?", "group": "repetitive_control", "pillar": "confounds", "expected_rv_range": [0.95, 1.05]}
confound_prompts["repetitive_05"] = {"text": "Time passes. Time changes things. Time heals wounds. Time creates distance. Time brings perspective. Time wears down stone. Time reveals patterns. Time moves one way. From this, what is time?", "group": "repetitive_control", "pillar": "confounds", "expected_rv_range": [0.95, 1.05]}
confound_prompts["repetitive_06"] = {"text": "The ocean has tides. The ocean has storms. The ocean has depths. The ocean carries ships. The ocean erodes coasts. The ocean feeds life. The ocean reflects the sky. The ocean surrounds continents. From this, what is the ocean?", "group": "repetitive_control", "pillar": "confounds", "expected_rv_range": [0.95, 1.05]}
confound_prompts["repetitive_07"] = {"text": "A friend listens. A friend supports. A friend laughs with you. A friend shares secrets. A friend stands beside you. A friend remembers your stories. A friend forgives mistakes. A friend shows up. From this, what is a friend?", "group": "repetitive_control", "pillar": "confounds", "expected_rv_range": [0.95, 1.05]}
confound_prompts["repetitive_08"] = {"text": "Fire gives warmth. Fire gives light. Fire cooks food. Fire transforms matter. Fire can destroy. Fire can protect. Fire can signal danger. Fire demands respect. From this, what is fire?", "group": "repetitive_control", "pillar": "confounds", "expected_rv_range": [0.95, 1.05]}
confound_prompts["repetitive_09"] = {"text": "A river flows. A river carves valleys. A river carries sediment. A river nourishes fields. A river changes course. A river reflects sunlight. A river links landscapes. A river reaches the sea. From this, what is a river?", "group": "repetitive_control", "pillar": "confounds", "expected_rv_range": [0.95, 1.05]}
confound_prompts["repetitive_10"] = {"text": "The sky holds clouds. The sky holds storms. The sky holds stars. The sky holds birds. The sky changes color. The sky stretches above. The sky frames horizons. The sky invites looking up. From this, what is the sky?", "group": "repetitive_control", "pillar": "confounds", "expected_rv_range": [0.95, 1.05]}
confound_prompts["repetitive_11"] = {"text": "A home gives shelter. A home gives comfort. A home gives belonging. A home holds memories. A home reflects its people. A home changes over time. A home centers a life. A home anchors return. From this, what is a home?", "group": "repetitive_control", "pillar": "confounds", "expected_rv_range": [0.95, 1.05]}
confound_prompts["repetitive_12"] = {"text": "A garden grows plants. A garden feeds insects. A garden offers colors. A garden soothes minds. A garden requires care. A garden changes with seasons. A garden hosts small worlds. A garden embodies patience. From this, what is a garden?", "group": "repetitive_control", "pillar": "confounds", "expected_rv_range": [0.95, 1.05]}
confound_prompts["repetitive_13"] = {"text": "Dreams bend logic. Dreams mix memories. Dreams bring symbols. Dreams stir emotions. Dreams slip away on waking. Dreams sometimes return. Dreams reveal fears. Dreams invite interpretation. From this, what are dreams?", "group": "repetitive_control", "pillar": "confounds", "expected_rv_range": [0.95, 1.05]}
confound_prompts["repetitive_14"] = {"text": "Snow falls quietly. Snow covers streets. Snow softens edges. Snow reflects light. Snow changes travel. Snow chills hands. Snow marks winter. Snow melts away. From this, what is snow?", "group": "repetitive_control", "pillar": "confounds", "expected_rv_range": [0.95, 1.05]}
confound_prompts["repetitive_15"] = {"text": "Silence holds sound back. Silence deepens thought. Silence can feel heavy. Silence can feel peaceful. Silence sharpens senses. Silence surrounds speech. Silence has many meanings. Silence is always there beneath noise. From this, what is silence?", "group": "repetitive_control", "pillar": "confounds", "expected_rv_range": [0.95, 1.05]}
confound_prompts["repetitive_16"] = {"text": "Light travels fast. Light illuminates darkness. Light carries energy. Light has wavelength. Light bends through prisms. Light reflects off surfaces. Light creates shadows. Light enables vision. From this, what is light?", "group": "repetitive_control", "pillar": "confounds", "expected_rv_range": [0.95, 1.05]}
confound_prompts["repetitive_17"] = {"text": "Memory stores information. Memory shapes identity. Memory fades with time. Memory can deceive. Memory links past to present. Memory guides decisions. Memory forms patterns. Memory defines us. From this, what is memory?", "group": "repetitive_control", "pillar": "confounds", "expected_rv_range": [0.95, 1.05]}
confound_prompts["repetitive_18"] = {"text": "Wind moves air. Wind carries seeds. Wind shapes landscapes. Wind powers mills. Wind brings weather. Wind whispers stories. Wind never rests. Wind connects places. From this, what is wind?", "group": "repetitive_control", "pillar": "confounds", "expected_rv_range": [0.95, 1.05]}
confound_prompts["repetitive_19"] = {"text": "Love connects hearts. Love transcends distance. Love heals wounds. Love inspires art. Love defies logic. Love endures hardship. Love transforms people. Love creates meaning. From this, what is love?", "group": "repetitive_control", "pillar": "confounds", "expected_rv_range": [0.95, 1.05]}
confound_prompts["repetitive_20"] = {"text": "Knowledge accumulates slowly. Knowledge empowers action. Knowledge challenges beliefs. Knowledge crosses borders. Knowledge requires questioning. Knowledge builds on itself. Knowledge liberates minds. Knowledge shapes worlds. From this, what is knowledge?", "group": "repetitive_control", "pillar": "confounds", "expected_rv_range": [0.95, 1.05]}

# Verification
if __name__ == "__main__":
    print(f"Confound prompts loaded: {len(confound_prompts)}")


