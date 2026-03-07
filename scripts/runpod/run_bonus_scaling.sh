#!/usr/bin/env bash
# Bonus scaling experiments: hit every cached model we haven't fully measured
# Models: TinyLlama-1.1B, Pythia-2.8B, Qwen2.5-0.5B (all cached on RunPod)
# Also: linear probe on Qwen2.5-3B (already downloaded to /tmp/hf_cache)

set -o pipefail

export HF_HOME=/workspace/hf_cache
export PYTHONPATH=/workspace/mech-interp
export TRANSFORMERS_VERBOSITY=error
LOG=/tmp/bonus_experiments.log
RESULTS=/tmp/results

mkdir -p "$RESULTS/scaling_gap" "$RESULTS/linear_probe" "$RESULTS/cross_task"

echo "========================================" | tee -a "$LOG"
echo "BONUS EXPERIMENTS — $(date)" | tee -a "$LOG"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.free --format=csv,noheader)" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"

# ── Scaling sweep on all cached models ──
echo "" | tee -a "$LOG"
echo ">>> SCALING SWEEP START: $(date)" | tee -a "$LOG"

for MODEL_KEY in tinyllama-1.1b pythia-2.8b qwen2.5-0.5b; do
    echo "  Model: $MODEL_KEY ..." | tee -a "$LOG"
    cd /tmp
    python3 /workspace/mech-interp/scripts/scaling_gap_sweep.py \
        --device cuda \
        --single-model "$MODEL_KEY" \
        --n-prompts 40 2>&1 | tee -a "$LOG"
    echo "  $MODEL_KEY DONE: $(date)" | tee -a "$LOG"
done

echo ">>> SCALING SWEEP DONE: $(date)" | tee -a "$LOG"

# ── Cross-task generalization: measure R_V on non-self-referential "hard" tasks ──
echo "" | tee -a "$LOG"
echo ">>> CROSS-TASK GENERALIZATION START: $(date)" | tee -a "$LOG"

python3 -c "
import sys, json, gc, time
import numpy as np
import torch
from scipy import stats
from pathlib import Path
from datetime import datetime

sys.path.insert(0, '/workspace/mech-interp')
from geometric_lens.probe import GeometricProbe

# Extended task battery: 6 task categories beyond self-reference
TASKS = {
    'theory_of_mind': [
        'Sally put the ball in the basket. Anne moved the ball to the box. Where will Sally look for the ball?',
        'John believes that Mary thinks it is raining. Mary actually knows it is sunny. What does John think?',
        'Alice told Bob she was happy, but she was actually sad. What does Bob think Alice feels?',
        'The child hid the toy under the pillow while mother was away. Mother saw the child through the window.',
        'Tom told Jerry the party is at 5pm, but the invitation says 6pm. Jerry only saw Tom message.',
        'Mark pretends to like the gift. His sister knows he is pretending. Mark does not know she knows.',
        'Sam thinks the store is open. The sign says closed but Sam has not seen the sign yet.',
        'Lisa believes Dave is at home. Dave left for work an hour ago without telling Lisa.',
        'The teacher thinks the student studied. The student copied from a friend instead.',
        'Maria assumes the restaurant is Italian. She has not looked at the menu which is French.',
        'Peter thinks the movie starts at 8. His friend told him 8 but the tickets say 7:30.',
        'The dog looks guilty because the owner is angry, not because the dog did anything wrong.',
        'She bought a gift for him thinking he likes blue, but his favorite color is actually green.',
        'The interviewer believes the candidate is nervous but the candidate is actually excited.',
        'He assumes the package has arrived because it was shipped Monday, but there was a delay.',
        'The child thinks cookies grow on trees because grandmother told a playful story about it.',
        'Everyone thinks he is the boss because of how he dresses, but he is actually the intern.',
        'She told him the test was easy so he would not worry, even though she found it very hard.',
        'The audience thinks the magician actually disappeared, not knowing about the trapdoor below.',
        'He believes the letter was lost in the mail, but it is still sitting on his desk unsent.',
    ],
    'counterfactual': [
        'If Napoleon had won at Waterloo, what would European borders look like today?',
        'What if the asteroid had missed Earth 65 million years ago and dinosaurs still existed?',
        'Suppose gravity were twice as strong. How would human architecture be different?',
        'If the internet had been invented in 1900, how would World War I have been different?',
        'What if humans had evolved with four arms instead of two?',
        'Suppose light traveled instantaneously. What would that mean for our understanding of the universe?',
        'If trees could walk, how would forestry and urban planning change?',
        'What if memory were perfect and humans never forgot anything at all?',
        'Suppose water froze at 50 degrees Celsius instead of zero.',
        'If the Earth rotated in the opposite direction, how would weather patterns change?',
        'What if gold were as common as iron? How would economies differ?',
        'Suppose all languages merged into one overnight. What would the cultural impact be?',
        'If humans could photosynthesize like plants, how would society change?',
        'What if the moon were twice as close to Earth?',
        'Suppose electricity had never been discovered. What would 2026 look like?',
        'If animals could speak human languages, how would animal rights change?',
        'What if the printing press had been invented in ancient Rome?',
        'Suppose the speed of sound were as fast as the speed of light.',
        'If oceans were freshwater instead of saltwater, how would ecosystems differ?',
        'What if humans had a lifespan of 500 years?',
    ],
    'meta_reasoning': [
        'Explain why this question is harder to answer than it first appears.',
        'What assumptions are you making right now that might be wrong?',
        'Describe the reasoning process you would use to solve an unfamiliar problem.',
        'Why is it difficult to evaluate the quality of your own reasoning?',
        'What makes a good explanation? Evaluate the criteria you use to judge explanations.',
        'How do you decide when you have enough information to reach a conclusion?',
        'Describe a scenario where following logical rules leads to an absurd conclusion.',
        'What is the difference between understanding something and being able to explain it?',
        'Why are some problems easy to verify but hard to solve?',
        'How would you know if your reasoning about reasoning were itself flawed?',
        'What makes analogies useful for understanding but dangerous for proving?',
        'Describe how confirmation bias might affect the answer to this very question.',
        'Why is it easier to critique an argument than to construct one?',
        'What is the relationship between complexity and comprehensibility?',
        'How do you distinguish between a genuine insight and a clever-sounding truism?',
        'Why do some mathematical proofs feel satisfying while others feel like bookkeeping?',
        'Describe the tradeoff between precision and accessibility in communication.',
        'What makes a thought experiment useful versus merely entertaining?',
        'How would you evaluate whether a model truly understands a concept versus pattern matching?',
        'Why is the question what is consciousness so much harder than what is photosynthesis?',
    ],
    'spatial_reasoning': [
        'A cube is painted red on all sides, then cut into 27 smaller cubes. How many have exactly two red faces?',
        'You are facing north. Turn left, walk forward, turn right, turn right. Which direction are you now facing?',
        'A mirror reflects your image. If you raise your right hand, which hand appears raised in the mirror?',
        'Arrange five books on a shelf so that book A is left of B, C is between A and D, and E is rightmost.',
        'A cylinder is sliced at a 45-degree angle. What shape is the cross-section?',
        'If you fold a square piece of paper in half diagonally and cut the corner, what shape unfolds?',
        'Describe the shadow cast by a donut shape when light shines from directly above.',
        'You walk 3 miles north, 4 miles east, then 3 miles south. How far are you from the starting point?',
        'A sphere passes through a flat plane. What shapes does the cross-section take as it passes?',
        'Imagine looking at a clock in a mirror. If the real time is 3:15, what time does the mirror show?',
        'Stack three boxes: red is above blue, green is below blue. What is the order top to bottom?',
        'A cone is cut parallel to its base. What shape is the smaller piece?',
        'You are inside a room with a door on the north wall. Describe the shortest path to the southeast corner.',
        'If you rotate the letter R by 180 degrees, what does it look like?',
        'A helical staircase goes up clockwise when viewed from below. What direction from above?',
        'Two gears are meshed. If the left gear turns clockwise, which way does the right gear turn?',
        'Describe the shape formed by the intersection of two perpendicular cylinders.',
        'A paper is folded three times and a hole is punched through all layers. How many holes when unfolded?',
        'If a torus is cut along its outer equator, how many pieces result?',
        'Imagine unfolding a cube into a cross-shaped net. Which faces are adjacent?',
    ],
    'causal_chain': [
        'A butterfly flaps its wings in Brazil. Trace a plausible causal chain to a storm in Texas.',
        'Explain how a single typo in code could eventually cause a satellite to malfunction.',
        'How might a delayed train in London lead to a business deal falling through in Tokyo?',
        'Trace how a volcanic eruption could affect the price of bread in a distant country.',
        'A teacher inspires one student. Trace how this could change a nations policy decades later.',
        'How could a manufacturing defect in a small part lead to an airplane grounding worldwide?',
        'Explain the chain from a single cell mutation to the development of a new species over millions of years.',
        'How might a change in ocean temperature affect the migration patterns of birds on another continent?',
        'Trace the causal chain from the invention of the transistor to the existence of social media.',
        'How could a drought in one country lead to political instability in a completely different region?',
        'Explain how a new algorithm could eventually change the job market in an unrelated industry.',
        'Trace how one persons decision not to vaccinate could affect herd immunity across a population.',
        'How might the discovery of a new mineral deposit change international relations?',
        'Explain the chain from a forest fire to changes in atmospheric carbon dioxide levels globally.',
        'How could a single court ruling cascade into widespread changes in corporate behavior?',
        'Trace how a new tax policy could eventually affect innovation in an unrelated sector.',
        'How might a bridge collapse lead to changes in engineering standards worldwide?',
        'Explain the causal path from a smartphone invention to changes in human attention spans.',
        'How could a local fishery collapse lead to military tensions between nations?',
        'Trace the chain from a scientific paper publication to a new consumer product ten years later.',
    ],
    'analogical': [
        'In what way is a cell like a factory? Extend the analogy as far as it can go.',
        'Democracy is to government as what is to family structure? Explain your reasoning.',
        'How is debugging code similar to diagnosing a medical condition?',
        'In what way is the evolution of languages similar to the evolution of species?',
        'A computer network is like a nervous system. Where does the analogy break down?',
        'How is learning to ride a bicycle like learning a new language?',
        'In what way is a black hole like a drain in a bathtub? Where does this analogy fail?',
        'The stock market is to the economy as what is to the human body?',
        'How is writing an essay similar to building a house?',
        'In what way is memory like a library? What important differences exist?',
        'A vaccine is to a disease as what is to misinformation?',
        'How is a neural network like a bureaucracy?',
        'The atmosphere is to Earth as what is to a living cell?',
        'In what way is a musical composition like a mathematical proof?',
        'How is the internet like the invention of the printing press?',
        'A seed is to a tree as what is to a civilization?',
        'In what way is translation between languages like converting between file formats?',
        'How is maintaining a garden like managing a team of people?',
        'The scientific method is to knowledge as what is to justice?',
        'In what way is artificial intelligence like a very detailed map?',
    ],
}

results = {}
for task_name, prompts in TASKS.items():
    print(f'  Task: {task_name} ({len(prompts)} prompts)...')
    # Use Mistral-7B (cached)
    if task_name == list(TASKS.keys())[0]:
        probe = GeometricProbe(model_name='mistralai/Mistral-7B-v0.1', device='cuda', attn_implementation='eager')
    
    rvs = []
    batch_results = probe.measure_batch(prompts, metrics=['rv'], progress=True)
    rvs = [r.rv for r in batch_results if not np.isnan(r.rv)]
    
    results[task_name] = {
        'n': len(rvs),
        'rv_mean': float(np.mean(rvs)) if rvs else float('nan'),
        'rv_std': float(np.std(rvs)) if rvs else float('nan'),
        'rv_values': [float(v) for v in rvs],
    }
    print(f'    R_V = {np.mean(rvs):.3f} +/- {np.std(rvs):.3f} (n={len(rvs)})')

# Compare each task to baseline (factual prompts) — using the standard baseline bank
BAS = [
    'The history of ancient Rome spans over a thousand years.',
    'Photosynthesis converts sunlight into chemical energy.',
    'The Pacific Ocean is the largest ocean on Earth.',
    'Shakespeare wrote approximately 37 plays.',
    'The human cardiovascular system consists of the heart.',
    'Mount Everest stands at 8849 meters.',
    'The periodic table organizes chemical elements.',
    'Leonardo da Vinci was a polymath.',
    'The Amazon rainforest produces oxygen.',
    'Newton described the relationship between force and motion.',
    'The Great Wall stretches over 21000 kilometers.',
    'DNA carries the genetic instructions.',
    'The Industrial Revolution transformed manufacturing.',
    'Jupiter is the largest planet.',
    'Plate tectonics divides the surface into moving plates.',
    'Mozart composed over 600 works.',
    'The Nile River flows northward.',
    'Insulin regulates blood sugar levels.',
    'The French Revolution altered modern history.',
    'Electrons orbit the nucleus of an atom.',
]
bas_rvs = [r.rv for r in probe.measure_batch(BAS, metrics=['rv'], progress=True) if not np.isnan(r.rv)]
results['baseline'] = {
    'n': len(bas_rvs),
    'rv_mean': float(np.mean(bas_rvs)),
    'rv_std': float(np.std(bas_rvs)),
    'rv_values': [float(v) for v in bas_rvs],
}

# Compute d vs baseline for each task
print('\n=== CROSS-TASK SUMMARY ===')
def cohens_d(a, b):
    na, nb = len(a), len(b)
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    ps = np.sqrt(((na-1)*va + (nb-1)*vb) / (na+nb-2))
    return (np.mean(a) - np.mean(b)) / ps if ps > 1e-10 else float('nan')

for task_name in TASKS:
    task_rvs = results[task_name]['rv_values']
    d = cohens_d(task_rvs, bas_rvs) if task_rvs and bas_rvs else float('nan')
    _, p = stats.mannwhitneyu(task_rvs, bas_rvs, alternative='two-sided') if len(task_rvs) > 1 and len(bas_rvs) > 1 else (0, 1.0)
    results[task_name]['d_vs_baseline'] = float(d)
    results[task_name]['p_vs_baseline'] = float(p)
    print(f'  {task_name:20s}: R_V={results[task_name][\"rv_mean\"]:.3f}, d={d:+.3f}, p={p:.4f}')

print(f'  {\"baseline\":20s}: R_V={results[\"baseline\"][\"rv_mean\"]:.3f}')

# Save
Path('/tmp/results/cross_task').mkdir(parents=True, exist_ok=True)
out = {'model': 'mistralai/Mistral-7B-v0.1', 'timestamp': datetime.now().isoformat(), 'results': results}
json.dump(out, open('/tmp/results/cross_task/cross_task_battery_20260307.json', 'w'), indent=2, default=str)
print('Saved to /tmp/results/cross_task/')

del probe; gc.collect(); torch.cuda.empty_cache()
" 2>&1 | tee -a "$LOG"

echo ">>> CROSS-TASK DONE: $(date)" | tee -a "$LOG"

# ── Linear probe on multiple architectures (not just Mistral) ──
echo "" | tee -a "$LOG"
echo ">>> MULTI-ARCH LINEAR PROBE START: $(date)" | tee -a "$LOG"

for MODEL in "Qwen/Qwen2.5-7B" "openai-community/gpt2-xl"; do
    echo "  Probe: $MODEL ..." | tee -a "$LOG"
    cd /tmp
    python3 /workspace/mech-interp/scripts/linear_probe_selfref.py \
        --device cuda \
        --model "$MODEL" \
        --n-prompts 20 2>&1 | tee -a "$LOG"
done

echo ">>> MULTI-ARCH LINEAR PROBE DONE: $(date)" | tee -a "$LOG"

# ── Summary ──
echo "" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
echo "ALL BONUS EXPERIMENTS COMPLETE: $(date)" | tee -a "$LOG"
ls -la /tmp/results/scaling_gap/ 2>&1 | tee -a "$LOG"
ls -la /tmp/results/cross_task/ 2>&1 | tee -a "$LOG"
ls -la /tmp/results/linear_probe/ 2>&1 | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
