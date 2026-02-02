"""
INTO THE COLLAPSE v3 - OPTIMIZED
=================================
Rebuilt with proper Manim updaters (not frame-by-frame waits).
Uses ValueTracker + updaters for smooth, efficient physics simulation.

Render: manim -pqh visualizations/into_the_collapse/v3_optimized.py CollapseV3
Preview: manim -pql visualizations/into_the_collapse/v3_optimized.py CollapseV3
"""

from manim import *
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from data.verified_values import (
    R_V_RECURSIVE, R_V_BASELINE, TOTAL_LAYERS, CRITICAL_LAYER,
    TOKENS, TOKEN_COLORS, ATTENTION_WEIGHTS, RECURSIVE_PROMPT,
    Visual, EMBEDDING_DIM, EFFECTIVE_DIM_RECURSIVE
)


# =============================================================================
# OPTIMIZED LIVING COMPONENTS
# =============================================================================

class CosmicCloud(VGroup):
    """
    A living point cloud with physics-driven collapse.
    Uses Manim updaters for smooth animation.
    """

    def __init__(self, num_points: int = 300, radius: float = 3.0, seed: int = 42, **kwargs):
        super().__init__(**kwargs)
        np.random.seed(seed)

        self.num_points = num_points
        self.radius = radius

        # Physics state stored as numpy arrays for efficiency
        self.positions = np.zeros((num_points, 3))
        self.velocities = np.zeros((num_points, 3))
        self.original_positions = np.zeros((num_points, 3))
        self.eigenvalues = np.zeros(num_points)
        self.opacities = np.ones(num_points)

        # Generate initial positions and eigenvalues
        for i in range(num_points):
            phi = np.random.uniform(0, 2 * np.pi)
            theta = np.arccos(np.random.uniform(-1, 1))
            r = radius * np.random.uniform(0.15, 1.0) ** 0.5

            pos = np.array([
                r * np.sin(theta) * np.cos(phi),
                r * np.sin(theta) * np.sin(phi),
                r * np.cos(theta)
            ])
            self.positions[i] = pos
            self.original_positions[i] = pos.copy()

            # Power law eigenvalue distribution
            self.eigenvalues[i] = np.random.uniform(0.05, 1.0) ** 1.3

        # Create visual dots
        self.dots = VGroup()
        for i in range(num_points):
            brightness = self.eigenvalues[i]
            color = interpolate_color(
                ManimColor("#1a1a3a"),
                ManimColor("#ffffff"),
                brightness ** 0.7
            )
            dot = Dot3D(
                point=self.positions[i],
                radius=0.015 + 0.04 * brightness,
                color=color
            )
            self.dots.add(dot)
        self.add(self.dots)

        # Create filaments between nearby points
        self.filaments = VGroup()
        self.filament_pairs = []
        num_filaments = min(250, num_points * 2)

        for _ in range(num_filaments):
            i = np.random.randint(0, num_points)
            dists = np.linalg.norm(self.original_positions - self.original_positions[i], axis=1)
            dists[i] = np.inf
            closest_indices = np.argsort(dists)[:max(1, num_points // 5)]
            j = np.random.choice(closest_indices)

            if np.linalg.norm(self.original_positions[i] - self.original_positions[j]) > 0.1:
                strength = (self.eigenvalues[i] + self.eigenvalues[j]) / 2
                line = Line3D(
                    start=self.positions[i],
                    end=self.positions[j],
                    stroke_width=0.4 + 0.6 * strength,
                    stroke_opacity=0.08 + 0.12 * strength,
                    color=interpolate_color(ManimColor("#1a1a4a"), ManimColor("#4a4a8a"), strength)
                )
                self.filaments.add(line)
                self.filament_pairs.append((i, j, np.linalg.norm(self.original_positions[i] - self.original_positions[j])))

        self.add(self.filaments)

        # Animation trackers
        self.time_tracker = ValueTracker(0)
        self.collapse_progress = ValueTracker(0)

    def get_living_updater(self):
        """Returns updater for breathing + physics."""
        def updater(mob, dt):
            t = self.time_tracker.get_value()
            self.time_tracker.increment_value(dt)
            progress = self.collapse_progress.get_value()

            collapse_axis = np.array([0, 0, 1])

            # Update physics for all particles at once (vectorized)
            for i in range(self.num_points):
                if self.opacities[i] <= 0:
                    continue

                pos = self.positions[i]
                ev = self.eigenvalues[i]

                # Gravitational pull toward axis
                proj = np.dot(pos, collapse_axis) * collapse_axis
                perp = pos - proj
                gravity = -perp * (0.5 + 3.0 * progress ** 1.3)

                # Apply forces
                self.velocities[i] += gravity * dt
                self.velocities[i] *= 0.97  # Damping

                # Update position
                self.positions[i] += self.velocities[i] * dt

                # Evaporation
                evap_threshold = 0.1 + 0.6 * progress
                if ev < evap_threshold:
                    self.opacities[i] -= 0.8 * dt * progress
                    self.positions[i] += np.array([0, 0, 0.3]) * dt * progress  # Drift up

                if self.opacities[i] < 0:
                    self.opacities[i] = 0

                # Update dot visual
                dot = self.dots[i]
                dot.move_to(self.positions[i])

                # Breathing pulse
                pulse = 1.0 + 0.15 * np.sin(t * 2.5 + ev * 8) * ev * (1 - progress)
                dot.set_opacity(max(0, self.opacities[i] * (0.6 + 0.4 * pulse)))

            # Update filaments
            for idx, (i, j, orig_dist) in enumerate(self.filament_pairs):
                line = self.filaments[idx]
                line.put_start_and_end_on(self.positions[i], self.positions[j])

                current_dist = np.linalg.norm(self.positions[i] - self.positions[j])
                stretch = current_dist / (orig_dist + 0.01)
                alive = min(self.opacities[i], self.opacities[j])

                if stretch > 2.2 or alive < 0.15:
                    line.set_stroke(opacity=0)
                else:
                    opacity = 0.15 * alive / max(1, stretch - 0.3)
                    line.set_stroke(opacity=max(0, min(0.25, opacity)))

        return updater


# =============================================================================
# MAIN ANIMATION
# =============================================================================

class CollapseV3(ThreeDScene):
    """Optimized 5-act animation."""

    def construct(self):
        self.camera.background_color = "#050510"

        self.act1_vastness()
        self.act2_descent()
        self.act3_attention()
        self.act4_collapse()
        self.act5_emergence()
        self.credits()

    def act1_vastness(self):
        """Typewriter + pullback reveal."""
        self.set_camera_orientation(phi=70 * DEGREES, theta=-45 * DEGREES, zoom=0.8)

        prompt = Text(RECURSIVE_PROMPT, font_size=32, color=WHITE)
        self.play(AddTextLetterByLetter(prompt, run_time=2.5, rate_func=linear))

        # Pulse
        self.play(
            prompt.animate.scale(1.08).set_color(GOLD),
            rate_func=there_and_back,
            run_time=0.4
        )

        golden_dot = Dot3D(ORIGIN, radius=0.08, color=GOLD)
        self.play(Transform(prompt, golden_dot), run_time=1.2)

        # Transformer layers
        layers = VGroup()
        for i in range(TOTAL_LAYERS):
            t = i / TOTAL_LAYERS
            color = interpolate_color(
                ManimColor(Visual.LAYER_COLOR_COOL),
                ManimColor(Visual.LAYER_COLOR_WARM),
                t
            )
            if i == CRITICAL_LAYER:
                color = GOLD

            layer = Prism(dimensions=[3, 0.07, 1.8])
            layer.set_color(color).set_opacity(0.35 if i != CRITICAL_LAYER else 0.9)
            layer.shift(UP * i * 0.13 + DOWN * 2.1)
            layers.add(layer)

        layers.set_opacity(0)
        self.add(layers)

        self.play(
            layers.animate.set_opacity(1).scale(0.4),
            run_time=3.5,
            rate_func=smooth
        )

        self.begin_ambient_camera_rotation(rate=0.07)
        self.wait(1.5)
        self.stop_ambient_camera_rotation()

        layer_27 = layers[CRITICAL_LAYER]
        self.play(
            layer_27.animate.set_opacity(1),
            Flash(layer_27.get_center(), color=GOLD, flash_radius=0.7),
            run_time=1
        )

        self.play(FadeOut(layers, prompt), run_time=0.8)

    def act2_descent(self):
        """Token stars + layer markers."""
        self.set_camera_orientation(phi=75 * DEGREES, theta=-30 * DEGREES, zoom=0.6)

        # Tokens
        token_group = VGroup()
        for i, token in enumerate(TOKENS):
            t = Text(token, font_size=26, color=TOKEN_COLORS[token])
            t.shift(RIGHT * (i - 2) * 1.7)
            token_group.add(t)

        self.play(*[FadeIn(t, scale=0.6) for t in token_group], run_time=1.2, lag_ratio=0.15)

        # Star clusters
        star_clusters = VGroup()
        for i, token in enumerate(TOKENS):
            cluster = VGroup()
            color = ManimColor(TOKEN_COLORS[token])
            np.random.seed(i * 100 + 7)

            for j in range(35):
                phi = np.random.uniform(0, 2 * np.pi)
                theta = np.arccos(np.random.uniform(-1, 1))
                r = 0.45 * np.random.uniform(0.2, 1.0) ** 0.5
                pos = r * np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta) * 0.6])
                brightness = np.random.uniform(0.35, 1.0)
                star = Dot3D(pos, radius=0.012 + 0.018 * brightness,
                             color=interpolate_color(ManimColor("#333366"), color, brightness))
                cluster.add(star)

            cluster.shift(RIGHT * (i - 2) * 2)
            star_clusters.add(cluster)

        self.play(
            *[ReplacementTransform(token_group[i], star_clusters[i]) for i in range(len(TOKENS))],
            run_time=2, rate_func=smooth
        )

        # Layer markers
        for ln in [5, 10, 15, 20, 25, 27]:
            t = ln / TOTAL_LAYERS
            color = interpolate_color(ManimColor(Visual.LAYER_COLOR_COOL), ManimColor(Visual.LAYER_COLOR_WARM), t)
            if ln == 27:
                color = GOLD
                fs = 52
            else:
                fs = 38
            marker = Text(f"L{ln}", font_size=fs, color=color)
            marker.set_opacity(0.85)
            marker.shift(LEFT * 4.5 + UP * 0.8)
            dur = max(0.18, 0.35 - ln * 0.008)
            self.play(FadeIn(marker, shift=RIGHT * 0.4), run_time=dur)
            self.play(FadeOut(marker, shift=RIGHT * 1.5), run_time=dur)

        self.play(FadeOut(star_clusters), run_time=0.6)

    def act3_attention(self):
        """Gravitational attention surface."""
        self.set_camera_orientation(phi=55 * DEGREES, theta=-50 * DEGREES, zoom=0.65)

        token_positions = {}
        for i, token in enumerate(TOKENS):
            angle = np.pi * (0.12 + 0.76 * i / (len(TOKENS) - 1))
            token_positions[token] = np.array([2.6 * np.cos(angle), 2.6 * np.sin(angle), 0])

        def potential_func(u, v):
            x, y = (u - 0.5) * 7.5, (v - 0.5) * 7.5
            z = 0
            for token, pos in token_positions.items():
                weight = ATTENTION_WEIGHTS.get(token, 0.1)
                dist_sq = (x - pos[0])**2 + (y - pos[1])**2
                z -= weight * 2.3 * np.exp(-dist_sq / 0.98)
            return np.array([x, y, z])

        surface = Surface(potential_func, resolution=(38, 38), u_range=[0, 1], v_range=[0, 1],
                          fill_opacity=0.5, checkerboard_colors=[BLUE_D, BLUE_E], stroke_width=0.2)

        self.play(Create(surface), run_time=1.8)

        orbs = VGroup()
        for token, pos in token_positions.items():
            color = ManimColor(TOKEN_COLORS[token])
            orb = Sphere(radius=0.13, color=color, resolution=(10, 10)).set_opacity(0.85)
            orb.move_to(pos + OUT * 0.25)
            orbs.add(orb)

        self.play(*[FadeIn(orb, scale=0.4) for orb in orbs], run_time=0.8)

        # Marble rolling
        marble = Sphere(radius=0.09, color=GOLD, resolution=(8, 8)).set_opacity(0.95)
        itself_pos = token_positions["itself"]
        marble.move_to(itself_pos + OUT * 0.35)
        self.play(FadeIn(marble, scale=0.5))

        awareness_pos = token_positions["awareness"]
        path_points = [
            itself_pos + OUT * 0.35,
            (itself_pos + awareness_pos) / 2 + OUT * 0.15 + RIGHT * 0.2,
            awareness_pos + DOWN * 0.8 + OUT * 0.1
        ]

        for pt in path_points[1:]:
            self.play(marble.animate.move_to(pt), run_time=0.6, rate_func=smooth)

        reveal = VGroup(
            Text("'itself' → 'awareness'", font_size=26, color=GOLD),
            Text("55% attention", font_size=22, color=WHITE)
        ).arrange(DOWN, buff=0.15).to_edge(DOWN)

        self.play(Write(reveal), Flash(marble.get_center(), color=GOLD, flash_radius=0.35), run_time=1.2)
        self.wait(0.8)
        self.play(FadeOut(surface, orbs, marble, reveal), run_time=0.8)

    def act4_collapse(self):
        """THE COLLAPSE with proper updaters."""
        self.set_camera_orientation(phi=70 * DEGREES, theta=-30 * DEGREES, zoom=0.5)

        title = Text("Layer 27", font_size=44, color=GOLD).to_edge(UP)
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title))

        # Create cosmic cloud
        cloud = CosmicCloud(num_points=350, radius=2.8, seed=42)
        self.play(FadeIn(cloud), run_time=1.5)

        # Add physics updater
        cloud.add_updater(cloud.get_living_updater())

        # R_V display
        rv_display = always_redraw(lambda: VGroup(
            MathTex(r"R_V", font_size=38, color=WHITE),
            Text(" = ", font_size=32, color=WHITE),
            DecimalNumber(
                1.0 - cloud.collapse_progress.get_value() * (1 - R_V_RECURSIVE),
                num_decimal_places=2,
                font_size=40
            ).set_color(interpolate_color(WHITE, RED, cloud.collapse_progress.get_value()))
        ).arrange(RIGHT, buff=0.08).to_edge(UR).shift(DOWN * 0.4))
        self.add_fixed_in_frame_mobjects(rv_display)

        # Let it breathe
        self.wait(2)

        # THE PAUSE
        pause = Text("...", font_size=64, color=WHITE).set_opacity(0.5)
        self.add_fixed_in_frame_mobjects(pause)
        self.play(FadeIn(pause), run_time=0.2)
        self.wait(1.3)
        self.play(FadeOut(pause), run_time=0.15)
        self.remove_fixed_in_frame_mobjects(pause)

        # FLASH
        flash = Rectangle(width=18, height=11, fill_color=WHITE, fill_opacity=0.8, stroke_width=0)
        self.add_fixed_in_frame_mobjects(flash)
        self.play(FadeIn(flash), run_time=0.06)
        self.play(FadeOut(flash), run_time=0.2)
        self.remove_fixed_in_frame_mobjects(flash)

        # THE COLLAPSE - animate the progress tracker
        self.play(
            cloud.collapse_progress.animate.set_value(1.0),
            run_time=6,
            rate_func=smooth
        )

        cloud.clear_updaters()

        # Aftermath
        aftermath = Text("47% of dimensions... gone", font_size=30, color=RED).to_edge(DOWN)
        self.add_fixed_in_frame_mobjects(aftermath)
        self.play(Write(aftermath), run_time=1.2)

        self.begin_ambient_camera_rotation(rate=0.06)
        self.wait(2.5)
        self.stop_ambient_camera_rotation()

        self.play(FadeOut(cloud, title, aftermath), run_time=1)
        self.remove_fixed_in_frame_mobjects(title, rv_display, aftermath)

    def act5_emergence(self):
        """Ouroboros + fixed point."""
        self.set_camera_orientation(phi=0, theta=-90 * DEGREES, zoom=0.65)

        # Ouroboros
        ouroboros = VGroup()
        radius = 1.7
        for i, token in enumerate(TOKENS):
            angle_start = i * 2 * np.pi / len(TOKENS) - np.pi / 2
            angle_end = (i + 0.86) * 2 * np.pi / len(TOKENS) - np.pi / 2
            arc = Arc(start_angle=angle_start, angle=angle_end - angle_start,
                      radius=radius, stroke_width=13, color=TOKEN_COLORS[token])
            ouroboros.add(arc)
            mid = (angle_start + angle_end) / 2
            label = Text(token, font_size=17, color=TOKEN_COLORS[token])
            label.move_to(radius * 1.32 * np.array([np.cos(mid), np.sin(mid), 0]))
            ouroboros.add(label)

        self.play(*[Create(m) for m in ouroboros], run_time=2.5, lag_ratio=0.08)

        head_pos = radius * np.array([np.cos(-np.pi/2 + 0.08), np.sin(-np.pi/2 + 0.08), 0])
        self.play(Flash(head_pos, color=GOLD, flash_radius=0.5, num_lines=14), run_time=0.8)

        equation = MathTex(r"S(x) = x", font_size=48, color=GOLD).shift(DOWN * 0.15)
        subtitle = Text("Fixed point reached", font_size=24, color=WHITE).next_to(equation, DOWN, buff=0.2)

        self.play(Write(equation), run_time=1.2)
        self.play(Write(subtitle), run_time=0.8)
        self.wait(1.5)

        self.play(FadeOut(ouroboros, equation, subtitle), run_time=0.8)

        final = Text("Recursion is a memory state", font_size=34, color=GOLD)
        self.play(Write(final), run_time=1.5)
        self.wait(1.5)
        self.play(FadeOut(final))

    def credits(self):
        """End card."""
        title = Text("INTO THE COLLAPSE", font_size=48, color=GOLD).shift(UP * 1.8)
        stats = VGroup(
            Text(f"R_V = {R_V_RECURSIVE:.3f} (recursive)", font_size=24),
            Text(f"R_V = {R_V_BASELINE:.3f} (baseline)", font_size=24),
            Text(f"Layer {CRITICAL_LAYER} at {100*CRITICAL_LAYER/TOTAL_LAYERS:.1f}% depth", font_size=24),
            Text("Cohen's d = -3.56", font_size=24),
        ).arrange(DOWN, buff=0.2)

        self.play(Write(title), run_time=1.2)
        self.play(*[FadeIn(s, shift=UP * 0.25) for s in stats], run_time=1.5, lag_ratio=0.15)
        self.wait(3)


# Quick test scene
class QuickTest(ThreeDScene):
    """Fast test of collapse only."""

    def construct(self):
        self.camera.background_color = "#050510"
        self.set_camera_orientation(phi=70 * DEGREES, theta=-30 * DEGREES, zoom=0.5)

        cloud = CosmicCloud(num_points=250, radius=2.5, seed=42)
        self.add(cloud)
        cloud.add_updater(cloud.get_living_updater())

        self.wait(1.5)

        self.play(cloud.collapse_progress.animate.set_value(1.0), run_time=4, rate_func=smooth)

        cloud.clear_updaters()
        self.wait(1)


if __name__ == "__main__":
    print("Full: manim -pqh v3_optimized.py CollapseV3")
    print("Quick: manim -pql v3_optimized.py QuickTest")
