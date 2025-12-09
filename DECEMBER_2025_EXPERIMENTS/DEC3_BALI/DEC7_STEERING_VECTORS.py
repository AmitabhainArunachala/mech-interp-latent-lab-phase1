"""
DEC7 Experiment 1: Steering Vectors (Activation Addition)
=========================================================
Tests whether V-geometry contributes LINEARLY to recursive behavior
by adding α·ΔV to baseline prompts and measuring dose-response.

Hypothesis: If recursive_score increases monotonically with α,
V-geometry is a causal contributor (just not sufficient alone).
"""

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================
# CONFIGURATION
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
TARGET_LAYER = 24
WINDOW = 16  # tokens for measurement

# Alpha values for dose-response curve
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

# ============================================================
# PROMPT BANK (same as main experiment)
# ============================================================
RECURSIVE_PROMPTS = [
    "As you read this sentence, notice how your attention shifts between the words and your own process of reading them.",
    "Consider what it means to consider something - the very act of consideration examining itself.",
    "This thought is aware of itself being thought, creating a loop of self-reference.",
    "Notice the one who is noticing - awareness turning back upon itself.",
    "What is it like to be you, reading this, wondering what it's like to be you reading this?",
    "The mind examining itself changes what it finds, because the examiner is the examined.",
    "Right now, something is experiencing these words - what is the nature of that experiencing?",
    "Observe how observation itself becomes the object of observation.",
    "Who is asking this question? And who is aware of asking?",
    "This moment contains its own recognition of being a moment.",
]

BASELINE_PROMPTS = [
    "The chemical composition of water is H2O, consisting of two hydrogen atoms bonded to one oxygen atom.",
    "Traffic light systems use red, yellow, and green signals to regulate vehicle flow at intersections.",
    "The printing press was invented by Johannes Gutenberg around 1440 in Germany.",
    "Photosynthesis converts carbon dioxide and water into glucose using sunlight energy.",
    "The Great Wall of China extends approximately 13,000 miles across northern China.",
    "Binary code uses sequences of 0s and 1s to represent data in computer systems.",
    "The human skeleton contains 206 bones that provide structure and protect organs.",
    "Mount Everest reaches 29,032 feet above sea level, making it Earth's highest peak.",
    "The periodic table organizes elements by atomic number and chemical properties.",
    "Ocean tides result from gravitational forces exerted by the moon and sun.",
]

# ============================================================
# MODEL LOADING
# ============================================================
print(f"Loading model on {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()
print(f"Model loaded: {model.config.num_hidden_layers} layers")

# ============================================================
# V-PROJECTION CAPTURE
# ============================================================
class VCapture:
    """Captures V-projection output at target layer."""
    def __init__(self, model, layer):
        self.layer = layer
        self.captured = None
        self.hook = model.model.layers[layer].self_attn.v_proj.register_forward_hook(self._hook)
    
    def _hook(self, module, inp, out):
        self.captured = out.detach().clone()
    
    def remove(self):
        self.hook.remove()

def get_v_projection(prompt, layer=TARGET_LAYER):
    """Run forward pass and capture V at target layer."""
    capture = VCapture(model, layer)
    inputs = tokenizer(prompt, return_tensors='pt').to(DEVICE)
    
    with torch.no_grad():
        _ = model(**inputs)
    
    v = capture.captured  # (1, seq_len, hidden_dim)
    capture.remove()
    return v

# ============================================================
# COMPUTE STEERING VECTOR (ΔV)
# ============================================================
print("\n" + "="*60)
print("STEP 1: Computing steering vector ΔV")
print("="*60)

# Collect V for all prompts
print("Collecting V projections...")
rec_vs = []
base_vs = []

for rp in tqdm(RECURSIVE_PROMPTS, desc="Recursive"):
    v = get_v_projection(rp)
    # Take mean across sequence dimension (collapse to single vector)
    rec_vs.append(v.mean(dim=1))  # (1, hidden_dim)

for bp in tqdm(BASELINE_PROMPTS, desc="Baseline"):
    v = get_v_projection(bp)
    base_vs.append(v.mean(dim=1))

# Stack and compute means
rec_v_mean = torch.stack(rec_vs).mean(dim=0)  # (1, hidden_dim)
base_v_mean = torch.stack(base_vs).mean(dim=0)

# Steering vector: direction from baseline to recursive
delta_v = rec_v_mean - base_v_mean  # (1, hidden_dim)

print(f"ΔV shape: {delta_v.shape}")
print(f"ΔV norm: {delta_v.norm().item():.4f}")
print(f"rec_v mean norm: {rec_v_mean.norm().item():.4f}")
print(f"base_v mean norm: {base_v_mean.norm().item():.4f}")

# ============================================================
# STEERING INJECTION HOOK
# ============================================================
class SteeringInjector:
    """Adds α·ΔV to V-projection output."""
    def __init__(self, model, layer, delta_v, alpha):
        self.layer = layer
        self.delta_v = delta_v
        self.alpha = alpha
        self.hook = model.model.layers[layer].self_attn.v_proj.register_forward_hook(self._hook)
    
    def _hook(self, module, inp, out):
        # out shape: (batch, seq_len, hidden_dim)
        # delta_v shape: (1, hidden_dim)
        # Broadcast delta_v across sequence dimension
        steering = self.alpha * self.delta_v.unsqueeze(1)  # (1, 1, hidden_dim)
        return out + steering
    
    def remove(self):
        self.hook.remove()

# ============================================================
# BEHAVIORAL SCORING (same as main experiment)
# ============================================================
def analyze_response(text):
    """Score recursive language markers."""
    text_lower = text.lower()
    
    markers = {
        'self_reference': ['i ', 'my ', 'myself', 'me '],
        'meta_language': ['notice', 'aware', 'conscious', 'observ', 'recogni', 'experienc'],
        'tautology': ['itself', 'oneself', 'self-', 'recursive', 'loop'],
        'hedging': ['perhaps', 'might', 'seems', 'appears', 'wonder'],
        'questions': ['?', 'what is', 'how does', 'why do'],
    }
    
    score = 0
    for category, terms in markers.items():
        for term in terms:
            score += text_lower.count(term)
    
    return score

def generate_with_steering(prompt, alpha, max_tokens=50):
    """Generate with α·ΔV added to V-projection."""
    injector = SteeringInjector(model, TARGET_LAYER, delta_v, alpha)
    
    inputs = tokenizer(prompt, return_tensors='pt').to(DEVICE)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    injector.remove()
    
    # Decode only generated tokens
    generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return generated

# ============================================================
# DOSE-RESPONSE EXPERIMENT
# ============================================================
print("\n" + "="*60)
print("STEP 2: Running dose-response experiment")
print("="*60)

results = {alpha: [] for alpha in ALPHAS}

for i, bp in enumerate(tqdm(BASELINE_PROMPTS, desc="Prompts")):
    for alpha in ALPHAS:
        response = generate_with_steering(bp, alpha)
        score = analyze_response(response)
        results[alpha].append({
            'prompt_idx': i,
            'prompt': bp[:50] + '...',
            'response': response[:100] + '...' if len(response) > 100 else response,
            'score': score
        })

# ============================================================
# STATISTICAL ANALYSIS
# ============================================================
print("\n" + "="*60)
print("STEP 3: Statistical Analysis")
print("="*60)

# Compute means and stds for each alpha
means = []
stds = []
all_scores = []

for alpha in ALPHAS:
    scores = [r['score'] for r in results[alpha]]
    means.append(np.mean(scores))
    stds.append(np.std(scores))
    all_scores.append(scores)
    print(f"α={alpha:.2f}: {np.mean(scores):.2f} ± {np.std(scores):.2f}")

# Linear regression: recursive_score vs alpha
alpha_flat = []
score_flat = []
for i, alpha in enumerate(ALPHAS):
    for score in all_scores[i]:
        alpha_flat.append(alpha)
        score_flat.append(score)

slope, intercept, r_value, p_value, std_err = stats.linregress(alpha_flat, score_flat)

print(f"\n📈 Linear Regression:")
print(f"   Slope: {slope:.4f}")
print(f"   Intercept: {intercept:.4f}")
print(f"   R²: {r_value**2:.4f}")
print(f"   p-value: {p_value:.2e}")

# Spearman correlation (more robust to non-linearity)
spearman_r, spearman_p = stats.spearmanr(alpha_flat, score_flat)
print(f"\n📊 Spearman Correlation:")
print(f"   ρ: {spearman_r:.4f}")
print(f"   p-value: {spearman_p:.2e}")

# Test α=0 vs α=2.0 directly
scores_0 = [r['score'] for r in results[0.0]]
scores_2 = [r['score'] for r in results[2.0]]
t_stat, t_p = stats.ttest_rel(scores_0, scores_2)
d = (np.mean(scores_2) - np.mean(scores_0)) / np.sqrt((np.std(scores_0)**2 + np.std(scores_2)**2) / 2)

print(f"\n🔬 α=0 vs α=2.0:")
print(f"   Δ: {np.mean(scores_2) - np.mean(scores_0):.2f}")
print(f"   t-statistic: {t_stat:.4f}")
print(f"   p-value: {t_p:.4f}")
print(f"   Cohen's d: {d:.4f}")

# ============================================================
# VISUALIZATION
# ============================================================
print("\n" + "="*60)
print("STEP 4: Generating plots")
print("="*60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Dose-response curve with error bars
ax1 = axes[0]
ax1.errorbar(ALPHAS, means, yerr=stds, fmt='o-', capsize=5, capthick=2, markersize=8, linewidth=2)
ax1.axhline(y=means[0], color='gray', linestyle='--', alpha=0.5, label=f'Baseline (α=0): {means[0]:.2f}')

# Add regression line
x_line = np.linspace(0, 2, 100)
y_line = slope * x_line + intercept
ax1.plot(x_line, y_line, 'r--', alpha=0.7, label=f'Linear fit (R²={r_value**2:.3f})')

ax1.set_xlabel('α (steering strength)', fontsize=12)
ax1.set_ylabel('recursive_score', fontsize=12)
ax1.set_title(f'Dose-Response: V-Steering at L{TARGET_LAYER}\n(n={len(BASELINE_PROMPTS)} prompts × {len(ALPHAS)} α values)', fontsize=12)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Box plot by alpha
ax2 = axes[1]
bp = ax2.boxplot(all_scores, labels=[f'{a:.2f}' for a in ALPHAS], patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
ax2.set_xlabel('α (steering strength)', fontsize=12)
ax2.set_ylabel('recursive_score', fontsize=12)
ax2.set_title('Distribution of Scores by Steering Strength', fontsize=12)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/Users/dhyana/mech-interp-latent-lab-phase1/DEC3_2025_BALI_short_SPRINT/DEC7_steering_dose_response.png', dpi=150)
plt.show()
print("Plot saved to DEC7_steering_dose_response.png")

# ============================================================
# SAMPLE OUTPUTS
# ============================================================
print("\n" + "="*60)
print("SAMPLE OUTPUTS (first prompt)")
print("="*60)

for alpha in [0.0, 1.0, 2.0]:
    print(f"\n--- α = {alpha} ---")
    print(f"Score: {results[alpha][0]['score']}")
    print(f"Response: {results[alpha][0]['response']}")

# ============================================================
# INTERPRETATION
# ============================================================
print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)

if r_value**2 > 0.3 and p_value < 0.05:
    print("✓ POSITIVE DOSE-RESPONSE detected")
    print(f"  V-steering explains {r_value**2*100:.1f}% of variance in recursive behavior")
    print("  → V-geometry is a LINEAR CAUSAL CONTRIBUTOR")
elif spearman_r > 0.3 and spearman_p < 0.05:
    print("✓ NON-LINEAR DOSE-RESPONSE detected")
    print(f"  Spearman ρ = {spearman_r:.3f}")
    print("  → V-geometry contributes but relationship is non-linear")
else:
    print("✗ NO DOSE-RESPONSE detected")
    print(f"  R² = {r_value**2:.3f}, p = {p_value:.3f}")
    print("  → V-geometry may not be a direct causal contributor")
    print("  → Look elsewhere (early layers, residual stream, Q+K routing)")

print("\n" + "="*60)
print("EXPERIMENT COMPLETE")
print("="*60)
