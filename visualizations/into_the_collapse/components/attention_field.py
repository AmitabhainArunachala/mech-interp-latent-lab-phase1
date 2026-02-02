"""
Gravitational attention field visualization.
Attention as continuous potential surface, not arrows.
"""

from manim import *
import numpy as np
from typing import Dict
import sys
sys.path.append(str(Path(__file__).parent.parent))
from data.verified_values import ATTENTION_WEIGHTS, TOKEN_COLORS, TOKENS


class GravitationalAttentionField(ThreeDScene):
    """
    Novel visualization: Attention as gravitational potential surface.
    Token positions create wells, attention weights determine depth.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def create_attention_surface(
        resolution: int = 50,
        size: float = 6.0,
        token_positions: Dict[str, np.ndarray] = None,
        attention_weights: Dict[str, float] = None
    ) -> Surface:
        """
        Create a gravitational potential surface where tokens create wells.
        Well depth = attention weight.
        """
        if token_positions is None:
            # Default semicircle arrangement
            token_positions = {}
            for i, token in enumerate(TOKENS):
                angle = np.pi * (0.2 + 0.6 * i / (len(TOKENS) - 1))
                token_positions[token] = np.array([
                    2.5 * np.cos(angle),
                    2.5 * np.sin(angle),
                    0
                ])

        if attention_weights is None:
            attention_weights = ATTENTION_WEIGHTS

        def potential_func(u, v):
            x = (u - 0.5) * size
            y = (v - 0.5) * size
            z = 0

            # Sum Gaussian wells for each token
            for token, pos in token_positions.items():
                weight = attention_weights.get(token, 0.1)
                dist_sq = (x - pos[0])**2 + (y - pos[1])**2
                sigma = 0.8  # Well width
                z -= weight * 2 * np.exp(-dist_sq / (2 * sigma**2))

            return np.array([x, y, z])

        surface = Surface(
            potential_func,
            resolution=(resolution, resolution),
            u_range=[0, 1],
            v_range=[0, 1],
            fill_opacity=0.7,
            checkerboard_colors=[BLUE_D, BLUE_E],
            stroke_width=0.5,
            stroke_color=BLUE_A,
        )

        return surface


class TokenOrb(VGroup):
    """A token represented as a glowing orb above the attention surface."""

    def __init__(
        self,
        token: str,
        position: np.ndarray,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.token = token
        color = TOKEN_COLORS.get(token, WHITE)

        # Main sphere
        self.orb = Sphere(
            radius=0.3,
            resolution=(20, 20),
        ).set_color(color).set_opacity(0.8)
        self.orb.move_to(position + UP * 0.5)

        # Label
        self.label = Text(token, font_size=24, color=color)
        self.label.rotate(PI/2, axis=RIGHT)  # Face camera in 3D
        self.label.next_to(self.orb, UP, buff=0.2)

        # Glow effect (larger transparent sphere)
        self.glow = Sphere(
            radius=0.5,
            resolution=(10, 10),
        ).set_color(color).set_opacity(0.2)
        self.glow.move_to(self.orb.get_center())

        self.add(self.glow, self.orb, self.label)


class AttentionMarble(Sphere):
    """
    A marble that rolls toward attention wells.
    Represents information flow from source to target.
    """

    def __init__(
        self,
        start_position: np.ndarray,
        target_token: str,
        token_positions: Dict[str, np.ndarray],
        attention_weights: Dict[str, float],
        **kwargs
    ):
        super().__init__(
            radius=0.15,
            resolution=(15, 15),
            **kwargs
        )

        self.set_color(GOLD)
        self.set_opacity(0.9)
        self.move_to(start_position)

        self.velocity = np.zeros(3)
        self.target_token = target_token
        self.token_positions = token_positions
        self.attention_weights = attention_weights

    def get_physics_updater(self, surface_func):
        """
        Returns an updater that makes the marble roll toward the deepest well.
        Uses gradient descent on the attention surface.
        """
        def updater(mob, dt):
            pos = mob.get_center()

            # Calculate gradient of potential field
            epsilon = 0.1
            grad = np.zeros(3)

            for token, token_pos in self.token_positions.items():
                weight = self.attention_weights.get(token, 0.1)
                diff = pos[:2] - token_pos[:2]
                dist_sq = np.sum(diff**2) + 0.01
                sigma = 0.8

                # Gradient of Gaussian well (points toward well)
                grad_contribution = weight * 2 * diff / (sigma**2) * np.exp(-dist_sq / (2 * sigma**2))
                grad[:2] -= grad_contribution

            # Apply force (gradient descent)
            force = -grad * 2.0

            # Update velocity with friction
            self.velocity += force * dt
            self.velocity *= 0.95  # Friction

            # Update position
            new_pos = pos + self.velocity * dt

            # Keep on surface (approximate z from potential)
            z = 0
            for token, token_pos in self.token_positions.items():
                weight = self.attention_weights.get(token, 0.1)
                dist_sq = (new_pos[0] - token_pos[0])**2 + (new_pos[1] - token_pos[1])**2
                sigma = 0.8
                z -= weight * 2 * np.exp(-dist_sq / (2 * sigma**2))
            new_pos[2] = z + 0.15  # Marble radius offset

            mob.move_to(new_pos)

        return updater


class AttentionWeightLabel(VGroup):
    """Shows attention weight when marble settles."""

    def __init__(
        self,
        weight: float,
        position: np.ndarray,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.weight_text = Text(
            f"{weight:.0%}",
            font_size=32,
            color=GOLD
        )
        self.weight_text.move_to(position)

        # Background for readability
        self.bg = BackgroundRectangle(
            self.weight_text,
            fill_opacity=0.7,
            buff=0.1
        )

        self.add(self.bg, self.weight_text)


def create_attention_chamber(scene: ThreeDScene) -> VGroup:
    """
    Create the full attention chamber visualization.
    Returns a VGroup containing surface, token orbs, and labels.
    """
    # Token positions in semicircle
    token_positions = {}
    for i, token in enumerate(TOKENS):
        angle = np.pi * (0.15 + 0.7 * i / (len(TOKENS) - 1))
        token_positions[token] = np.array([
            3.0 * np.cos(angle),
            3.0 * np.sin(angle),
            0
        ])

    # Create surface
    surface = GravitationalAttentionField.create_attention_surface(
        resolution=40,
        size=8.0,
        token_positions=token_positions,
        attention_weights=ATTENTION_WEIGHTS
    )

    # Create token orbs
    orbs = VGroup()
    for token, pos in token_positions.items():
        orb = TokenOrb(token, pos)
        orbs.add(orb)

    # Create marble at "itself" position (the observer)
    itself_pos = token_positions["itself"]
    marble = AttentionMarble(
        start_position=itself_pos + UP * 0.5,
        target_token="awareness",  # It will roll toward awareness
        token_positions=token_positions,
        attention_weights=ATTENTION_WEIGHTS
    )

    chamber = VGroup(surface, orbs, marble)
    chamber.token_positions = token_positions
    chamber.marble = marble
    chamber.surface = surface

    return chamber
