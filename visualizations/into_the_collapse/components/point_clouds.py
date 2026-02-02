"""
Particle system for dimensional collapse visualization.
Implements: gravitational implosion, dimensional evaporation, and harmonic decay.
"""

from manim import *
import numpy as np
from typing import List, Tuple, Optional
import sys
sys.path.append(str(Path(__file__).parent.parent))
from data.verified_values import Visual, R_V_RECURSIVE, R_V_BASELINE


class DimensionalStar(Dot3D):
    """A single dimension represented as a star in the cosmic neural network."""

    def __init__(
        self,
        position: np.ndarray,
        eigenvalue: float = 1.0,
        **kwargs
    ):
        self.eigenvalue = eigenvalue
        self.original_position = position.copy()
        self.velocity = np.zeros(3)

        # Brightness based on eigenvalue magnitude
        brightness = min(1.0, eigenvalue)
        color = interpolate_color(
            ManimColor(Visual.STAR_COLOR_DIM),
            ManimColor(Visual.STAR_COLOR_BRIGHT),
            brightness
        )

        # Size based on eigenvalue
        radius = 0.02 + 0.03 * brightness

        super().__init__(
            point=position,
            radius=radius,
            color=color,
            **kwargs
        )

    def should_evaporate(self, threshold: float = 0.3) -> bool:
        """Low eigenvalue stars evaporate first."""
        return self.eigenvalue < threshold


class CosmicFilament(Line3D):
    """Connection between dimensions - like synapses or cosmic web strands."""

    def __init__(
        self,
        start_star: DimensionalStar,
        end_star: DimensionalStar,
        connection_strength: float = 1.0,
        **kwargs
    ):
        self.start_star = start_star
        self.end_star = end_star
        self.connection_strength = connection_strength
        self.original_opacity = min(0.3, connection_strength * 0.5)

        super().__init__(
            start=start_star.get_center(),
            end=end_star.get_center(),
            color=Visual.FILAMENT_COLOR,
            stroke_opacity=self.original_opacity,
            stroke_width=1,
            **kwargs
        )


class CosmicNeuralCloud(VGroup):
    """
    The cosmic neural universe - 4096 dimensions as stars with filament connections.
    Implements all three collapse styles simultaneously.
    """

    def __init__(
        self,
        num_stars: int = Visual.NUM_DIMENSION_STARS,
        num_filaments: int = Visual.NUM_FILAMENTS,
        radius: float = 3.0,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.num_stars = num_stars
        self.radius = radius
        self.stars: List[DimensionalStar] = []
        self.filaments: List[CosmicFilament] = []
        self.collapse_progress = ValueTracker(0)

        # Generate eigenvalue distribution (power law - few large, many small)
        self.eigenvalues = self._generate_eigenvalue_distribution(num_stars)

        # Create stars in spherical distribution
        self._create_stars()

        # Create filament connections
        self._create_filaments(num_filaments)

        # Add all to group
        for filament in self.filaments:
            self.add(filament)
        for star in self.stars:
            self.add(star)

    def _generate_eigenvalue_distribution(self, n: int) -> np.ndarray:
        """Generate realistic eigenvalue distribution (power law decay)."""
        # Top eigenvalues are large, tail falls off
        indices = np.arange(1, n + 1)
        eigenvalues = 1.0 / (indices ** 0.5)  # Power law
        # Normalize so max is 1
        eigenvalues = eigenvalues / eigenvalues.max()
        # Shuffle so they're not spatially ordered
        np.random.shuffle(eigenvalues)
        return eigenvalues

    def _create_stars(self):
        """Create stars in spherical distribution with slight clustering."""
        for i in range(self.num_stars):
            # Spherical coordinates with some structure
            phi = np.random.uniform(0, 2 * np.pi)
            theta = np.arccos(np.random.uniform(-1, 1))
            r = self.radius * (0.3 + 0.7 * np.random.uniform(0, 1) ** 0.5)

            # Add some clustering (dimensions that work together)
            cluster_offset = 0.3 * np.array([
                np.sin(i * 0.1),
                np.cos(i * 0.1),
                np.sin(i * 0.05)
            ])

            position = np.array([
                r * np.sin(theta) * np.cos(phi),
                r * np.sin(theta) * np.sin(phi),
                r * np.cos(theta)
            ]) + cluster_offset

            star = DimensionalStar(
                position=position,
                eigenvalue=self.eigenvalues[i]
            )
            self.stars.append(star)

    def _create_filaments(self, num_filaments: int):
        """Create filament connections between nearby dimensions."""
        if len(self.stars) < 2:
            return

        # Connect stars that are close together
        for _ in range(num_filaments):
            i = np.random.randint(0, len(self.stars))
            star1 = self.stars[i]

            # Find a nearby star
            distances = [
                np.linalg.norm(star1.get_center() - s.get_center())
                for s in self.stars
            ]
            distances[i] = float('inf')  # Exclude self

            # Pick from closest 20%
            sorted_indices = np.argsort(distances)
            j = sorted_indices[np.random.randint(0, max(1, len(self.stars) // 5))]
            star2 = self.stars[j]

            # Connection strength based on both eigenvalues
            strength = (star1.eigenvalue + star2.eigenvalue) / 2

            filament = CosmicFilament(star1, star2, strength)
            self.filaments.append(filament)

    def get_collapse_updater(self, collapse_axis: np.ndarray = np.array([0, 0, 1])):
        """
        Returns an updater function that implements the triple collapse:
        1. Gravitational implosion toward axis
        2. Dimensional evaporation (weak eigenvalues fade)
        3. Harmonic decay (implied by the visual)
        """
        collapse_axis = collapse_axis / np.linalg.norm(collapse_axis)

        def updater(mob, dt):
            progress = self.collapse_progress.get_value()

            for star in self.stars:
                # Get current position
                pos = star.get_center()

                # 1. GRAVITATIONAL IMPLOSION
                # Project position onto collapse axis
                projection = np.dot(pos, collapse_axis) * collapse_axis
                perpendicular = pos - projection

                # Pull toward axis (stronger as progress increases)
                gravity_force = -perpendicular * Visual.GRAVITY_STRENGTH * progress
                star.velocity += gravity_force * dt

                # Apply velocity with damping
                new_pos = pos + star.velocity * dt
                star.velocity *= 0.95  # Damping

                # Move star
                star.move_to(new_pos)

                # 2. DIMENSIONAL EVAPORATION
                if star.should_evaporate(threshold=0.3 + 0.4 * progress):
                    # Fade out and drift upward
                    current_opacity = star.get_fill_opacity()
                    new_opacity = max(0, current_opacity - Visual.EVAPORATION_RATE * dt * progress)
                    star.set_fill(opacity=new_opacity)
                    star.set_stroke(opacity=new_opacity)

                    # Gentle upward drift
                    star.shift(UP * 0.5 * dt * progress)

            # Update filaments to follow their stars
            for filament in self.filaments:
                # Update positions
                new_start = filament.start_star.get_center()
                new_end = filament.end_star.get_center()

                # Check if either star has evaporated
                start_opacity = filament.start_star.get_fill_opacity()
                end_opacity = filament.end_star.get_fill_opacity()
                min_opacity = min(start_opacity, end_opacity)

                # Stretch and potentially snap
                distance = np.linalg.norm(new_end - new_start)
                original_distance = np.linalg.norm(
                    filament.start_star.original_position -
                    filament.end_star.original_position
                )

                stretch_factor = distance / (original_distance + 0.01)

                # Filaments snap if stretched too far or stars fade
                if stretch_factor > 2.0 or min_opacity < 0.1:
                    filament.set_stroke(opacity=0)
                else:
                    # Fade with stretch
                    new_opacity = filament.original_opacity * min_opacity / max(1, stretch_factor - 1)
                    filament.set_stroke(opacity=max(0, new_opacity))

                # Update geometry
                filament.put_start_and_end_on(new_start, new_end)

        return updater


class RVCounter(VGroup):
    """Animated R_V value counter that changes color as it drops."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.rv_value = ValueTracker(1.0)

        self.label = Text("R_V = ", font_size=36)
        self.number = always_redraw(
            lambda: DecimalNumber(
                self.rv_value.get_value(),
                num_decimal_places=2,
                font_size=48
            ).next_to(self.label, RIGHT, buff=0.1).set_color(
                self._get_rv_color(self.rv_value.get_value())
            )
        )

        self.add(self.label, self.number)

    def _get_rv_color(self, value: float) -> ManimColor:
        """Color interpolation: WHITE (1.0) -> RED (0.5)"""
        t = max(0, min(1, (1.0 - value) / 0.5))
        return interpolate_color(WHITE, RED, t)

    def get_value_tracker(self) -> ValueTracker:
        return self.rv_value


class DimensionCounter(VGroup):
    """Shows dimension reduction: 4096 -> ~800"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.dim_value = ValueTracker(4096)

        self.number = always_redraw(
            lambda: Integer(
                int(self.dim_value.get_value()),
                font_size=36
            )
        )
        self.label = Text(" effective dimensions", font_size=28)
        self.label.next_to(self.number, RIGHT, buff=0.1)

        self.add(self.number, self.label)

    def get_value_tracker(self) -> ValueTracker:
        return self.dim_value
