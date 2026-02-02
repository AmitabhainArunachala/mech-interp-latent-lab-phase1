"""
3D Transformer architecture visualization.
Layers as luminous grids, residual stream as fiber optic tubes.
"""

from manim import *
import numpy as np
from typing import List, Tuple
import sys
sys.path.append(str(Path(__file__).parent.parent))
from data.verified_values import (
    TOTAL_LAYERS, CRITICAL_LAYER, Visual, TOKENS, TOKEN_COLORS
)


class TransformerLayer3D(VGroup):
    """A single transformer layer as a luminous prism with attention heads."""

    def __init__(
        self,
        layer_num: int,
        width: float = 4.0,
        height: float = 0.3,
        depth: float = 3.0,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.layer_num = layer_num

        # Color based on layer position (cool -> warm)
        t = layer_num / TOTAL_LAYERS
        color = interpolate_color(
            ManimColor(Visual.LAYER_COLOR_COOL),
            ManimColor(Visual.LAYER_COLOR_WARM),
            t
        )

        # Special glow for Layer 27
        if layer_num == CRITICAL_LAYER:
            color = GOLD
            opacity = 0.9
        else:
            opacity = 0.4 + 0.3 * (1 - abs(layer_num - CRITICAL_LAYER) / TOTAL_LAYERS)

        # Main prism
        self.prism = Prism(
            dimensions=[width, height, depth],
        ).set_color(color).set_opacity(opacity)

        # Layer label
        self.label = Text(
            f"L{layer_num}",
            font_size=18,
            color=color
        )
        self.label.next_to(self.prism, LEFT, buff=0.3)

        # Attention head spheres (32 heads arranged in grid)
        self.heads = VGroup()
        head_radius = 0.08
        grid_size = 6  # 6x6 = 36 positions, use 32
        for i in range(32):
            row = i // grid_size
            col = i % grid_size
            x = (col - grid_size/2 + 0.5) * (width / grid_size) * 0.8
            z = (row - grid_size/2 + 0.5) * (depth / grid_size) * 0.8
            head = Sphere(
                radius=head_radius,
                resolution=(8, 8)
            ).set_color(color).set_opacity(0.6)
            head.move_to(self.prism.get_center() + np.array([x, 0, z]))
            self.heads.add(head)

        self.add(self.prism, self.label, self.heads)


class TransformerStack(VGroup):
    """Full transformer as stacked layers."""

    def __init__(
        self,
        num_layers: int = TOTAL_LAYERS,
        layer_spacing: float = 0.5,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.layers: List[TransformerLayer3D] = []
        self.layer_spacing = layer_spacing

        for i in range(num_layers):
            layer = TransformerLayer3D(layer_num=i)
            layer.shift(UP * i * layer_spacing)
            self.layers.append(layer)
            self.add(layer)

        # Center the stack
        self.center()

    def get_layer_center(self, layer_num: int) -> np.ndarray:
        """Get the center position of a specific layer."""
        if 0 <= layer_num < len(self.layers):
            return self.layers[layer_num].get_center()
        return ORIGIN


class ResidualStream(VGroup):
    """Luminous tubes connecting layers - the residual stream."""

    def __init__(
        self,
        start_layer: int,
        end_layer: int,
        stack: TransformerStack,
        num_tubes: int = 5,  # One per token
        **kwargs
    ):
        super().__init__(**kwargs)

        self.tubes = VGroup()

        for i, token in enumerate(TOKENS):
            # Offset for each token
            x_offset = (i - 2) * 0.6

            start_pos = stack.get_layer_center(start_layer) + np.array([x_offset, 0, 0])
            end_pos = stack.get_layer_center(end_layer) + np.array([x_offset, 0, 0])

            # Create tube as cylinder
            color = TOKEN_COLORS.get(token, WHITE)

            tube = Line3D(
                start=start_pos,
                end=end_pos,
                color=color,
                stroke_width=3,
            ).set_opacity(0.6)

            self.tubes.add(tube)

        self.add(self.tubes)


class TokenParticle(Sphere):
    """A particle representing a token flowing through the residual stream."""

    def __init__(
        self,
        token: str,
        **kwargs
    ):
        color = TOKEN_COLORS.get(token, WHITE)
        super().__init__(
            radius=0.1,
            resolution=(12, 12),
            **kwargs
        )
        self.set_color(color)
        self.set_opacity(0.9)
        self.token = token

        # Glow
        self.glow = Sphere(
            radius=0.15,
            resolution=(8, 8)
        ).set_color(color).set_opacity(0.3)
        # Note: glow needs to be added separately and updated

    def get_flow_updater(
        self,
        start_pos: np.ndarray,
        end_pos: np.ndarray,
        progress_tracker: ValueTracker
    ):
        """Returns an updater for smooth flow animation."""
        def updater(mob):
            t = progress_tracker.get_value()
            # Smooth interpolation with slight wave
            pos = start_pos + (end_pos - start_pos) * t
            # Add subtle oscillation
            pos += np.array([
                0.05 * np.sin(t * 4 * np.pi),
                0,
                0.05 * np.cos(t * 4 * np.pi)
            ])
            mob.move_to(pos)

        return updater


class LayerMarker(VGroup):
    """Floating layer number that drifts past camera during descent."""

    def __init__(
        self,
        layer_num: int,
        **kwargs
    ):
        super().__init__(**kwargs)

        # Color temperature based on layer
        t = layer_num / TOTAL_LAYERS
        color = interpolate_color(
            ManimColor(Visual.LAYER_COLOR_COOL),
            ManimColor(Visual.LAYER_COLOR_WARM),
            t
        )

        self.text = Text(
            f"L{layer_num}",
            font_size=48,
            color=color
        )
        self.text.set_opacity(0.8)

        # Glow background
        self.glow = Rectangle(
            width=self.text.width + 0.4,
            height=self.text.height + 0.2,
            fill_color=color,
            fill_opacity=0.2,
            stroke_width=0
        )
        self.glow.move_to(self.text)

        self.add(self.glow, self.text)


def create_full_transformer_viz() -> Tuple[TransformerStack, VGroup]:
    """
    Create the complete transformer visualization with stack and residual streams.
    Returns (stack, streams).
    """
    stack = TransformerStack()

    # Create residual streams between every few layers
    streams = VGroup()
    for start in range(0, TOTAL_LAYERS - 1, 4):
        end = min(start + 4, TOTAL_LAYERS - 1)
        stream = ResidualStream(start, end, stack)
        streams.add(stream)

    return stack, streams
