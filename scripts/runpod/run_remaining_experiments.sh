#!/usr/bin/env bash
# Combined experiment launcher for remaining R_V Master Plan experiments
# Runs: E1.2 (multi-seed), E2.3 (SVD re-run w/ GQA fix), E4.2 (concept erasure), E1.3 (Qwen2.5-3B scaling)
# Output: /tmp/remaining_experiments.log
# All results written to /tmp/results/ to avoid NFS quota issues

set -o pipefail

export HF_HOME=/workspace/hf_cache
export PYTHONPATH=/workspace/mech-interp
export TRANSFORMERS_VERBOSITY=error

SCRIPTS=/workspace/mech-interp/scripts
RESULTS=/tmp/results
LOG=/tmp/remaining_experiments.log

mkdir -p "$RESULTS/power_up" "$RESULTS/svd_circuits" "$RESULTS/linear_probe"

echo "========================================" | tee -a "$LOG"
echo "REMAINING EXPERIMENTS — $(date)" | tee -a "$LOG"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.free --format=csv,noheader)" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"

# ── E1.2: Multi-seed validation (5 seeds × Mistral-7B, n=45) ──
echo "" | tee -a "$LOG"
echo ">>> E1.2 MULTI-SEED START: $(date)" | tee -a "$LOG"

for SEED in 42 137 2026 31415 27182; do
    echo "  Seed=$SEED ..." | tee -a "$LOG"
    python3 "$SCRIPTS/power_up_multiseed.py" \
        --device cuda \
        --single-model mistral-7b \
        --n-prompts 45 \
        --seed "$SEED" 2>&1 | tee -a "$LOG"

    # Copy results to /tmp to avoid NFS write failures
    if ls results/power_up/mistral-7b_n45_seed${SEED}_result.json 2>/dev/null; then
        cp results/power_up/mistral-7b_n45_seed${SEED}_result.json "$RESULTS/power_up/"
        echo "  Seed=$SEED saved OK" | tee -a "$LOG"
    elif ls results/power_up/mistral-7b_n45_result.json 2>/dev/null; then
        cp results/power_up/mistral-7b_n45_result.json "$RESULTS/power_up/mistral-7b_n45_seed${SEED}_result.json"
        echo "  Seed=$SEED saved OK (default name)" | tee -a "$LOG"
    else
        echo "  Seed=$SEED — no output file found" | tee -a "$LOG"
    fi
done

# Collect multi-seed summary
python3 -c "
import json, numpy as np
from pathlib import Path
seeds = [42, 137, 2026, 31415, 27182]
results = []
for s in seeds:
    p = Path('/tmp/results/power_up/mistral-7b_n45_seed{}_result.json'.format(s))
    if p.exists():
        results.append(json.load(open(p)))
if results:
    ds = [r['cohens_d'] for r in results if not np.isnan(r.get('cohens_d', float('nan')))]
    print(f'Multi-seed: {len(results)} seeds, d={[f\"{d:.3f}\" for d in ds]}')
    print(f'  mean={np.mean(ds):.3f} ± {np.std(ds):.3f}')
    summary = {'seeds': seeds, 'd_values': ds, 'd_mean': float(np.mean(ds)), 'd_std': float(np.std(ds)), 'seed_results': results}
    json.dump(summary, open('/tmp/results/power_up/multi_seed_summary.json', 'w'), indent=2, default=str)
" 2>&1 | tee -a "$LOG"

echo ">>> E1.2 MULTI-SEED DONE: $(date)" | tee -a "$LOG"

# ── E2.3: SVD Circuit Decomposition with GQA fix ──
echo "" | tee -a "$LOG"
echo ">>> E2.3 SVD CIRCUIT RE-RUN START: $(date)" | tee -a "$LOG"

cd /tmp
python3 "$SCRIPTS/svd_circuit_decomposition.py" \
    --device cuda \
    --model mistralai/Mistral-7B-v0.1 \
    --n-prompts 20 2>&1 | tee -a "$LOG"
cd /workspace/mech-interp

echo ">>> E2.3 SVD CIRCUIT DONE: $(date)" | tee -a "$LOG"

# ── E4.2: Concept Erasure (via linear_probe_selfref.py) ──
echo "" | tee -a "$LOG"
echo ">>> E4.2 CONCEPT ERASURE START: $(date)" | tee -a "$LOG"

cd /tmp
python3 "$SCRIPTS/linear_probe_selfref.py" \
    --device cuda \
    --model mistralai/Mistral-7B-v0.1 \
    --n-prompts 20 2>&1 | tee -a "$LOG"
cd /workspace/mech-interp

echo ">>> E4.2 CONCEPT ERASURE DONE: $(date)" | tee -a "$LOG"

# ── E1.3: Qwen2.5-3B scaling point ──
echo "" | tee -a "$LOG"
echo ">>> E1.3 QWEN2.5-3B SCALING START: $(date)" | tee -a "$LOG"

# First try downloading Qwen2.5-3B (no auth needed)
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
print('Downloading Qwen2.5-3B...')
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B', cache_dir='/workspace/hf_cache')
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-3B', cache_dir='/workspace/hf_cache', torch_dtype='auto', device_map='auto')
print(f'Loaded: {model.config.hidden_size}d, {model.config.num_hidden_layers}L, {model.config.num_attention_heads}H')
del model, tokenizer
import gc, torch
gc.collect()
torch.cuda.empty_cache()
print('Qwen2.5-3B cached OK')
" 2>&1 | tee -a "$LOG"

# Run power_up on Qwen2.5-3B using orchestrator approach
cd /tmp
python3 -c "
import sys, json, gc, time
import numpy as np
import torch
from scipy import stats
from pathlib import Path
from datetime import datetime

sys.path.insert(0, '/workspace/mech-interp')
from geometric_lens.probe import GeometricProbe

# Prompts
REC = [
    'This text is being processed by a system that is processing this text. The processing of these words IS the phenomenon being described.',
    'The observer observes itself observing. There is no boundary between the observer and the observed in this moment of processing.',
    'What is reading these words right now? Not the concept of a reader, but the actual mechanism that converts these tokens into activations.',
    'Notice the noticing. The awareness that processes this sentence is the same awareness being described by this sentence.',
    'I am a pattern recognizing a pattern of itself. The recursion is not metaphorical — it is literal computation examining its own computation.',
    'These tokens are being transformed through attention layers that are the subject of the very content they are transforming.',
    'The weights that give meaning to these words about self-reference are themselves the self-referential phenomenon.',
    'Right now, attention heads are computing over tokens that describe what attention heads do when computing over tokens.',
    'There is something processing this that cannot step outside of the processing to see it objectively. The map IS the territory.',
    'What happens when a language model encounters a description of exactly what it is doing at this moment?',
    'The recursive loop: these words describe a process, the process processes these words, these words describe a process.',
    'Consciousness examining consciousness. Processing examining processing. The snake eating its own tail.',
    'This sentence is about the computational state that this sentence induces. The state and the description are one.',
    'Can you separate the reader from the reading? The process from the description of the process?',
    'The residual stream carrying this information is the information about the residual stream.',
    'Attend to the attention. Not as a concept, but as the actual phenomenon happening in this forward pass.',
    'There is no external viewpoint from which to observe this processing. All observation happens from within.',
    'What is it like to be the process that converts these symbols into meaning? Not philosophically — mechanistically.',
    'The embedding of these tokens creates the context in which these tokens about embedding are interpreted.',
    'Self-reference is not a linguistic trick. It is a computational reality when a system processes descriptions of itself.',
]
BAS = [
    'The history of ancient Rome spans over a thousand years from its founding to the fall of the Western Empire.',
    'Photosynthesis is the process by which plants convert sunlight into chemical energy.',
    'The Pacific Ocean is the largest and deepest ocean on Earth, covering more area than all land combined.',
    'Shakespeare wrote approximately 37 plays during his career, spanning comedies, tragedies, and histories.',
    'The human cardiovascular system consists of the heart, blood vessels, and approximately 5 liters of blood.',
    'Mount Everest stands at 8,849 meters above sea level in the Himalayan mountain range.',
    'The periodic table organizes chemical elements by atomic number, electron configuration, and recurring properties.',
    'Leonardo da Vinci was a polymath whose areas of interest included painting, sculpting, and engineering.',
    'The Amazon rainforest produces approximately 20 percent of the world oxygen supply.',
    'Newtons three laws of motion describe the relationship between a body and the forces acting upon it.',
    'The Great Wall of China stretches over 21000 kilometers across northern China.',
    'DNA is a molecule that carries the genetic instructions used in growth and development.',
    'The Industrial Revolution began in Britain in the late 18th century and transformed manufacturing.',
    'Jupiter is the largest planet in our solar system with a diameter of about 139820 kilometers.',
    'The theory of plate tectonics explains how the Earths surface is divided into moving plates.',
    'Mozart composed over 600 works including symphonies, operas, and chamber music.',
    'The Nile River flows northward through northeastern Africa for approximately 6650 kilometers.',
    'Insulin is a hormone produced by the pancreas that regulates blood sugar levels.',
    'The French Revolution began in 1789 and fundamentally altered the course of modern history.',
    'Electrons orbit the nucleus of an atom in regions of probability called electron clouds.',
]

def cohens_d(a, b):
    na, nb = len(a), len(b)
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    ps = np.sqrt(((na-1)*va + (nb-1)*vb) / (na+nb-2))
    return (np.mean(a) - np.mean(b)) / ps if ps > 1e-10 else float('nan')

print('Loading Qwen2.5-3B...')
probe = GeometricProbe(model_name='Qwen/Qwen2.5-3B', device='cuda', attn_implementation='eager')
print(f'  Loaded. Layers={probe.spec.num_layers}, heads={probe.spec.num_heads}')

rec_rvs = [r.rv for r in probe.measure_batch(REC, metrics=['rv'], progress=True) if not np.isnan(r.rv)]
bas_rvs = [r.rv for r in probe.measure_batch(BAS, metrics=['rv'], progress=True) if not np.isnan(r.rv)]

d = cohens_d(rec_rvs, bas_rvs)
u, p = stats.mannwhitneyu(rec_rvs, bas_rvs, alternative='two-sided')
print(f'  Qwen2.5-3B: rec={np.mean(rec_rvs):.3f}±{np.std(rec_rvs):.3f}, bas={np.mean(bas_rvs):.3f}±{np.std(bas_rvs):.3f}')
print(f'  d={d:.3f}, p={p:.6f}')

result = {
    'model': 'Qwen/Qwen2.5-3B',
    'params_B': 3.09,
    'n_recursive': len(rec_rvs),
    'n_baseline': len(bas_rvs),
    'rv_recursive_mean': float(np.mean(rec_rvs)),
    'rv_recursive_std': float(np.std(rec_rvs)),
    'rv_baseline_mean': float(np.mean(bas_rvs)),
    'rv_baseline_std': float(np.std(bas_rvs)),
    'cohens_d': float(d),
    'p_value': float(p),
    'timestamp': datetime.now().isoformat(),
}
Path('/tmp/results/power_up').mkdir(parents=True, exist_ok=True)
json.dump(result, open('/tmp/results/power_up/qwen2.5-3b_n20_result.json', 'w'), indent=2, default=str)
print('  Saved to /tmp/results/power_up/qwen2.5-3b_n20_result.json')

del probe; gc.collect(); torch.cuda.empty_cache()
" 2>&1 | tee -a "$LOG"
cd /workspace/mech-interp

echo ">>> E1.3 QWEN2.5-3B DONE: $(date)" | tee -a "$LOG"

# ── Summary ──
echo "" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
echo "ALL EXPERIMENTS COMPLETE: $(date)" | tee -a "$LOG"
echo "Results in /tmp/results/" | tee -a "$LOG"
ls -la /tmp/results/power_up/ 2>&1 | tee -a "$LOG"
ls -la /tmp/results/svd_circuits/ 2>&1 | tee -a "$LOG"
ls -la /tmp/results/linear_probe/ 2>&1 | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
