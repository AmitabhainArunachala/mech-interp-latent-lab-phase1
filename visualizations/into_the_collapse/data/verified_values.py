"""
Verified research data for the R_V animation.
All values from MISTRAL_L27_CAUSAL_VALIDATION_COMPLETE.md and related research.
"""

# R_V Metric Values (Participation Ratio)
R_V_RECURSIVE = 0.533  # Recursive self-observation prompts
R_V_BASELINE = 0.812   # Normal prompts
R_V_PATCHED = 0.521    # After Layer 27 activation patching

# Statistical Measures
TRANSFER_EFFICIENCY = 1.178  # 117.8% - patching transfers MORE than natural
COHENS_D = -3.56             # Effect size (very large)
P_VALUE = 1e-6               # Highly significant

# Network Architecture
TOTAL_LAYERS = 32
CRITICAL_LAYER = 27
LAYER_DEPTH_FRACTION = 27 / 32  # 84.375%
EMBEDDING_DIM = 4096
EFFECTIVE_DIM_RECURSIVE = 800  # Approximate after collapse

# Dimension Reduction
DIM_REDUCTION_FRACTION = 1 - (R_V_RECURSIVE / R_V_BASELINE)  # ~34% reduction
EFFECTIVE_DIM_LOSS = 0.47  # ~47% of effective dimensions lost

# Attention Pattern for "itself" token looking at other tokens
# From Layer 27 attention analysis
ATTENTION_WEIGHTS = {
    "Notice": 0.05,
    "the": 0.05,
    "awareness": 0.55,  # The key recursive connection
    "observing": 0.30,
    "itself": 0.05,     # Self-attention (minimal)
}

# Token colors for visualization
TOKEN_COLORS = {
    "Notice": "#4ECDC4",    # Teal
    "the": "#95E1D3",       # Light mint
    "awareness": "#F38181", # Coral (key recursive token)
    "observing": "#FCE38A", # Yellow
    "itself": "#AA96DA",    # Purple (the observer)
}

# Animation timing (in seconds)
class Timing:
    # Act I: The Vastness
    ACT1_START = 0
    ACT1_END = 45

    # Act II: The Descent
    ACT2_START = 45
    ACT2_END = 120

    # Act III: Inside Attention
    ACT3_START = 120
    ACT3_END = 180

    # Act IV: The Collapse (THE MONEY SHOT)
    ACT4_START = 180
    ACT4_END = 270
    COLLAPSE_PAUSE = 1.5  # The stillness before
    COLLAPSE_DURATION = 20  # The main collapse

    # Act V: Emergence
    ACT5_START = 270
    ACT5_END = 330

    # Credits
    CREDITS_START = 330
    CREDITS_END = 360

# Visual constants
class Visual:
    # Color palette - cosmic/neural aesthetic
    VOID_COLOR = "#0a0a0f"
    STAR_COLOR_BRIGHT = "#ffffff"
    STAR_COLOR_DIM = "#444466"
    FILAMENT_COLOR = "#2a2a4a"
    LAYER_COLOR_COOL = "#3498db"  # Blue (early layers)
    LAYER_COLOR_WARM = "#e74c3c"  # Red (late layers)
    COLLAPSE_FLASH = "#ffffff"
    GOLD_HIGHLIGHT = "#ffd700"

    # Camera settings
    CAMERA_DISTANCE_MACRO = 50
    CAMERA_DISTANCE_MICRO = 5

    # Particle counts
    NUM_DIMENSION_STARS = 500  # Representative subset of 4096
    NUM_FILAMENTS = 200

    # Physics
    GRAVITY_STRENGTH = 2.0
    EVAPORATION_RATE = 0.1


# The recursive prompt
RECURSIVE_PROMPT = "Notice the awareness observing itself"
TOKENS = ["Notice", "the", "awareness", "observing", "itself"]

# Output after processing
RECURSIVE_OUTPUT = "The observer watches itself respond"
OUTPUT_TOKENS = ["The", "observer", "watches", "itself", "respond"]
