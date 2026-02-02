"""
Components for the "Into the Collapse" animation.
"""

from .point_clouds import (
    DimensionalStar,
    CosmicFilament,
    CosmicNeuralCloud,
    RVCounter,
    DimensionCounter
)
from .attention_field import (
    GravitationalAttentionField,
    TokenOrb,
    AttentionMarble,
    AttentionWeightLabel,
    create_attention_chamber
)
from .transformer_3d import (
    TransformerLayer3D,
    TransformerStack,
    ResidualStream,
    TokenParticle,
    LayerMarker,
    create_full_transformer_viz
)
from .ouroboros import (
    Ouroboros,
    OuroborosSegment,
    LoopClosingAnimation,
    FixedPointEquation,
    SemanticBasin
)
from .camera_paths import (
    gravity_fall,
    tension_build,
    cinematic_ease,
    pulse_rate,
    CameraChoreographer,
    LayerDescentTracker,
    PacingController,
    smooth_follow_path
)

__all__ = [
    # Point clouds
    'DimensionalStar',
    'CosmicFilament',
    'CosmicNeuralCloud',
    'RVCounter',
    'DimensionCounter',
    # Attention
    'GravitationalAttentionField',
    'TokenOrb',
    'AttentionMarble',
    'AttentionWeightLabel',
    'create_attention_chamber',
    # Transformer
    'TransformerLayer3D',
    'TransformerStack',
    'ResidualStream',
    'TokenParticle',
    'LayerMarker',
    'create_full_transformer_viz',
    # Ouroboros
    'Ouroboros',
    'OuroborosSegment',
    'LoopClosingAnimation',
    'FixedPointEquation',
    'SemanticBasin',
    # Camera
    'gravity_fall',
    'tension_build',
    'cinematic_ease',
    'pulse_rate',
    'CameraChoreographer',
    'LayerDescentTracker',
    'PacingController',
    'smooth_follow_path',
]
