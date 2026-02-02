"""
INTO THE COLLAPSE v4 - CINEMATIC
=================================
Maximum polish: better camera, more effects, dramatic timing.

Key additions over v3:
1. More dramatic camera movements
2. Glow effects using blur approximations
3. Better particle distribution
4. Smoother transitions
5. More visual interest in each act

Render: manim -pqh visualizations/into_the_collapse/v4_cinematic.py CollapseV4
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
# ENHANCED VISUAL COMPONENTS
# =============================================================================

class GlowingDot(VGroup):
    """A dot with a soft glow effect."""

    def __init__(self, pos, radius=0.05, color=WHITE, glow_factor=2.0, **kwargs):
        super().__init__(**kwargs)

        # Glow layers (outer to inner)
        for i in range(3, 0, -1):
            glow = Dot(
                point=pos,
                radius=radius * glow_factor * (i / 3),
                color=color
            )
            glow.set_opacity(0.15 / i)
            self.add(glow)

        # Core
        core = Dot(point=pos, radius=radius, color=color)
        core.set_opacity(0.9)
        self.add(core)


class EnhancedCosmicCloud(VGroup):
    """
    Improved cosmic cloud with better distribution and more dramatic collapse.
    """

    def __init__(self, num_points: int = 300, radius: float = 3.0, seed: int = 42, **kwargs):
        super().__init__(**kwargs)
        np.random.seed(seed)

        self.num_points = num_points
        self.radius = radius

        # Physics state
        self.positions = np.zeros((num_points, 3))
        self.velocities = np.zeros((num_points, 3))
        self.original_positions = np.zeros((num_points, 3))
        self.eigenvalues = np.zeros(num_points)
        self.opacities = np.ones(num_points)
        self.phases = np.random.uniform(0, 2 * np.pi, num_points)  # For breathing

        # Generate positions with more structure (clusters + uniform)
        for i in range(num_points):
            if i < num_points // 3:
                # Core cluster
                r = radius * 0.4 * np.random.uniform(0.1, 1.0) ** 0.3
            elif i < 2 * num_points // 3:
                # Middle shell
                r = radius * np.random.uniform(0.4, 0.7)
            else:
                # Outer shell
                r = radius * np.random.uniform(0.7, 1.0)

            phi = np.random.uniform(0, 2 * np.pi)
            theta = np.arccos(np.random.uniform(-1, 1))

            pos = np.array([
                r * np.sin(theta) * np.cos(phi),
                r * np.sin(theta) * np.sin(phi),
                r * np.cos(theta)
            ])
            self.positions[i] = pos
            self.original_positions[i] = pos.copy()

            # Eigenvalue based on radius (inner = stronger)
            self.eigenvalues[i] = (1 - r / radius) ** 0.5 + 0.1 * np.random.uniform(0, 1)
            self.eigenvalues[i] = min(1.0, self.eigenvalues[i])

        # Create dots with varying visual properties
        self.dots = VGroup()
        for i in range(num_points):
            ev = self.eigenvalues[i]
            brightness = ev ** 0.6

            # Color gradient: blue core -> white bright -> dim gray
            if ev > 0.7:
                color = interpolate_color(WHITE, ManimColor("#aaccff"), (ev - 0.7) / 0.3)
            else:
                color = interpolate_color(ManimColor("#222244"), WHITE, ev / 0.7)

            dot = Dot3D(
                point=self.positions[i],
                radius=0.012 + 0.045 * brightness,
                color=color
            )
            self.dots.add(dot)
        self.add(self.dots)

        # Filaments with better distribution
        self.filaments = VGroup()
        self.filament_pairs = []
        num_filaments = min(300, num_points * 2)

        # Connect based on distance and eigenvalue similarity
        for _ in range(num_filaments):
            i = np.random.randint(0, num_points)
            dists = np.linalg.norm(self.original_positions - self.original_positions[i], axis=1)
            dists[i] = np.inf

            # Prefer connecting similar eigenvalues
            ev_diffs = np.abs(self.eigenvalues - self.eigenvalues[i])
            scores = dists + ev_diffs * 2  # Combined score

            closest = np.argsort(scores)[:max(1, num_points // 8)]
            j = np.random.choice(closest)

            orig_dist = np.linalg.norm(self.original_positions[i] - self.original_positions[j])
            if orig_dist > 0.1 and orig_dist < radius * 0.8:
                strength = (self.eigenvalues[i] + self.eigenvalues[j]) / 2

                # Color based on connection strength
                line_color = interpolate_color(
                    ManimColor("#1a1a3a"),
                    ManimColor("#4a6aaa"),
                    strength
                )

                line = Line3D(
                    start=self.positions[i],
                    end=self.positions[j],
                    stroke_width=0.3 + 0.7 * strength,
                    stroke_opacity=0.06 + 0.14 * strength,
                    color=line_color
                )
                self.filaments.add(line)
                self.filament_pairs.append((i, j, orig_dist))

        self.add(self.filaments)

        # Trackers
        self.time_tracker = ValueTracker(0)
        self.collapse_progress = ValueTracker(0)

    def get_living_updater(self):
        """Updater with enhanced breathing and physics."""
        def updater(mob, dt):
            t = self.time_tracker.get_value()
            self.time_tracker.increment_value(dt)
            progress = self.collapse_progress.get_value()

            collapse_axis = np.array([0, 0, 1])

            for i in range(self.num_points):
                if self.opacities[i] <= 0:
                    continue

                pos = self.positions[i]
                ev = self.eigenvalues[i]

                # Gravity with acceleration curve
                proj = np.dot(pos, collapse_axis) * collapse_axis
                perp = pos - proj
                gravity_strength = 0.3 + 4.0 * (progress ** 1.8)  # Exponential ramp
                gravity = -perp * gravity_strength

                # Slight spiral
                tangent = np.cross(collapse_axis, perp)
                if np.linalg.norm(tangent) > 0.01:
                    tangent = tangent / np.linalg.norm(tangent)
                    gravity += tangent * 0.3 * progress

                self.velocities[i] += gravity * dt
                self.velocities[i] *= 0.96

                self.positions[i] += self.velocities[i] * dt

                # Evaporation - weak eigenvalues fade first
                evap_threshold = 0.08 + 0.65 * progress
                if ev < evap_threshold:
                    fade_rate = 0.6 * dt * progress * (1 - ev / evap_threshold)
                    self.opacities[i] -= fade_rate
                    # Drift outward and up
                    drift = np.array([perp[0] * 0.1, perp[1] * 0.1, 0.4]) * dt * progress
                    self.positions[i] += drift

                self.opacities[i] = max(0, self.opacities[i])

                # Update visual
                dot = self.dots[i]
                dot.move_to(self.positions[i])

                # Enhanced breathing with individual phases
                phase = self.phases[i]
                breath = 1.0 + 0.2 * np.sin(t * 2.8 + phase) * ev * (1 - 0.7 * progress)
                dot.set_opacity(max(0, self.opacities[i] * (0.5 + 0.5 * breath)))

            # Update filaments
            for idx, (i, j, orig_dist) in enumerate(self.filament_pairs):
                line = self.filaments[idx]
                line.put_start_and_end_on(self.positions[i], self.positions[j])

                current_dist = np.linalg.norm(self.positions[i] - self.positions[j])
                stretch = current_dist / (orig_dist + 0.01)
                alive = min(self.opacities[i], self.opacities[j])

                # Snap threshold based on stretch and aliveness
                if stretch > 2.5 or alive < 0.1:
                    line.set_stroke(opacity=0)
                else:
                    base_opacity = 0.12 * alive
                    stretch_penalty = max(0, 1 - (stretch - 1) * 0.5)
                    line.set_stroke(opacity=min(0.2, base_opacity * stretch_penalty))
                    line.set_stroke(width=max(0.3, 1.0 - (stretch - 1) * 0.3))

        return updater


# =============================================================================
# MAIN ANIMATION
# =============================================================================

class CollapseV4(ThreeDScene):
    """Maximum polish cinematic animation."""

    def construct(self):
        self.camera.background_color = "#030308"

        self.act1_vastness()
        self.act2_descent()
        self.act3_attention()
        self.act4_collapse()
        self.act5_emergence()
        self.credits()

    def act1_vastness(self):
        """Opening with more dramatic reveal."""
        self.set_camera_orientation(phi=75 * DEGREES, theta=-45 * DEGREES, zoom=0.9)

        # Typewriter with subtle fade-in per character
        prompt = Text(RECURSIVE_PROMPT, font_size=30, color=WHITE)
        self.play(AddTextLetterByLetter(prompt, run_time=2.8, rate_func=linear))

        # Multiple pulse
        for _ in range(2):
            self.play(
                prompt.animate.scale(1.05).set_color(GOLD),
                rate_func=there_and_back,
                run_time=0.35
            )

        golden_dot = Dot3D(ORIGIN, radius=0.1, color=GOLD)
        self.play(Transform(prompt, golden_dot), run_time=1.4, rate_func=smooth)

        # Transformer with improved layers
        layers = VGroup()
        for i in range(TOTAL_LAYERS):
            t = i / TOTAL_LAYERS
            color = interpolate_color(
                ManimColor("#2266bb"),
                ManimColor("#cc4444"),
                t
            )
            if i == CRITICAL_LAYER:
                color = GOLD

            layer = Prism(dimensions=[3.5, 0.06, 2])
            layer.set_color(color)
            layer.set_opacity(0.25 + 0.15 * (1 - abs(i - CRITICAL_LAYER) / TOTAL_LAYERS) if i != CRITICAL_LAYER else 0.95)
            layer.shift(UP * i * 0.12 + DOWN * 1.9)
            layers.add(layer)

        layers.set_opacity(0)
        self.add(layers)

        # Dramatic pullback with rotation
        self.play(
            layers.animate.set_opacity(1).scale(0.35),
            run_time=4,
            rate_func=smooth
        )

        # Orbit
        self.begin_ambient_camera_rotation(rate=0.06)
        self.wait(2)
        self.stop_ambient_camera_rotation()

        # Layer 27 with bigger flash
        layer_27 = layers[CRITICAL_LAYER]
        self.play(
            layer_27.animate.set_opacity(1).scale(1.1),
            Flash(layer_27.get_center(), color=GOLD, flash_radius=1.0, num_lines=20),
            run_time=1.2
        )

        self.play(FadeOut(layers, prompt), run_time=0.9)

    def act2_descent(self):
        """Token stars with more visual interest."""
        self.set_camera_orientation(phi=70 * DEGREES, theta=-35 * DEGREES, zoom=0.55)

        # Tokens appear with stagger
        token_group = VGroup()
        for i, token in enumerate(TOKENS):
            t = Text(token, font_size=28, color=TOKEN_COLORS[token])
            t.shift(RIGHT * (i - 2) * 1.8)
            token_group.add(t)

        self.play(
            *[FadeIn(t, scale=0.4, shift=UP * 0.3) for t in token_group],
            run_time=1.5,
            lag_ratio=0.2
        )

        # Star clusters with more stars and connections
        star_clusters = VGroup()
        for idx, token in enumerate(TOKENS):
            cluster = VGroup()
            color = ManimColor(TOKEN_COLORS[token])
            np.random.seed(idx * 100 + 42)

            # More stars
            for j in range(50):
                phi = np.random.uniform(0, 2 * np.pi)
                theta = np.arccos(np.random.uniform(-1, 1))
                r = 0.5 * np.random.uniform(0.15, 1.0) ** 0.4
                pos = r * np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta) * 0.7])
                brightness = np.random.uniform(0.25, 1.0)
                star = Dot3D(pos, radius=0.01 + 0.02 * brightness,
                             color=interpolate_color(ManimColor("#222255"), color, brightness))
                cluster.add(star)

            # More filaments
            stars_list = list(cluster)
            for _ in range(15):
                if len(stars_list) >= 2:
                    s1, s2 = np.random.choice(len(stars_list), 2, replace=False)
                    line = Line3D(
                        stars_list[s1].get_center(),
                        stars_list[s2].get_center(),
                        stroke_width=0.25,
                        stroke_opacity=0.08,
                        color=color
                    )
                    cluster.add(line)

            cluster.shift(RIGHT * (idx - 2) * 2.3)
            star_clusters.add(cluster)

        self.play(
            *[ReplacementTransform(token_group[i], star_clusters[i]) for i in range(len(TOKENS))],
            run_time=2.5,
            rate_func=smooth
        )

        # Layer markers with more drama
        for ln in [5, 10, 15, 20, 24, 26, 27]:
            t = ln / TOTAL_LAYERS
            color = interpolate_color(ManimColor("#3388dd"), ManimColor("#dd5555"), t)
            if ln == 27:
                color = GOLD
                fs = 60
            elif ln >= 24:
                fs = 48
            else:
                fs = 36

            marker = Text(f"L{ln}", font_size=fs, color=color)
            marker.set_opacity(0.9)
            marker.shift(LEFT * 5 + UP * 0.5)

            # Accelerating pace
            dur = max(0.12, 0.4 - ln * 0.012)
            self.play(FadeIn(marker, shift=RIGHT * 0.5), run_time=dur)
            self.play(FadeOut(marker, shift=RIGHT * 2), run_time=dur)

        self.play(FadeOut(star_clusters), run_time=0.7)

    def act3_attention(self):
        """Attention surface with marble physics."""
        self.set_camera_orientation(phi=50 * DEGREES, theta=-55 * DEGREES, zoom=0.6)

        token_positions = {}
        for i, token in enumerate(TOKENS):
            angle = np.pi * (0.1 + 0.8 * i / (len(TOKENS) - 1))
            token_positions[token] = np.array([2.7 * np.cos(angle), 2.7 * np.sin(angle), 0])

        def potential_func(u, v):
            x, y = (u - 0.5) * 8, (v - 0.5) * 8
            z = 0
            for token, pos in token_positions.items():
                weight = ATTENTION_WEIGHTS.get(token, 0.1)
                dist_sq = (x - pos[0])**2 + (y - pos[1])**2
                z -= weight * 2.8 * np.exp(-dist_sq / 0.9)
            return np.array([x, y, z])

        surface = Surface(potential_func, resolution=(45, 45), u_range=[0, 1], v_range=[0, 1],
                          fill_opacity=0.45, checkerboard_colors=[ManimColor("#1a3366"), ManimColor("#1a2a55")],
                          stroke_width=0.15, stroke_color=ManimColor("#4488cc"))

        self.play(Create(surface), run_time=2.2)

        orbs = VGroup()
        for token, pos in token_positions.items():
            color = ManimColor(TOKEN_COLORS[token])
            orb = Sphere(radius=0.14, color=color, resolution=(12, 12)).set_opacity(0.88)
            orb.move_to(pos + OUT * 0.28)
            orbs.add(orb)

        self.play(*[FadeIn(orb, scale=0.3) for orb in orbs], run_time=1)

        # Marble with trail effect
        marble = Sphere(radius=0.1, color=GOLD, resolution=(10, 10)).set_opacity(0.95)
        itself_pos = token_positions["itself"]
        marble.move_to(itself_pos + OUT * 0.38)
        self.play(FadeIn(marble, scale=0.4))

        # Smoother physics path
        awareness_pos = token_positions["awareness"]
        vel = np.zeros(3)
        pos = np.array(marble.get_center())

        trail_dots = VGroup()
        for step in range(80):
            # Force calculation
            force = np.zeros(3)
            for token, token_pos in token_positions.items():
                weight = ATTENTION_WEIGHTS.get(token, 0.1)
                diff = pos[:2] - token_pos[:2]
                dist_sq = np.sum(diff**2) + 0.01
                grad = weight * 2.8 * diff / 0.9 * np.exp(-dist_sq / 0.9)
                force[:2] -= grad

            vel += force * 0.025
            vel *= 0.93
            pos = pos + vel * 0.04

            # Z on surface
            z = 0
            for token, token_pos in token_positions.items():
                weight = ATTENTION_WEIGHTS.get(token, 0.1)
                dist_sq = (pos[0] - token_pos[0])**2 + (pos[1] - token_pos[1])**2
                z -= weight * 2.8 * np.exp(-dist_sq / 0.9)
            pos[2] = z + 0.1

            # Trail
            if step % 8 == 0:
                trail = Dot3D(pos.copy(), radius=0.03, color=GOLD).set_opacity(0.3)
                trail_dots.add(trail)
                self.add(trail)

            marble.move_to(pos)
            self.wait(0.025)

        # Fade trail
        self.play(trail_dots.animate.set_opacity(0), run_time=0.5)
        self.remove(trail_dots)

        reveal = VGroup(
            Text("'itself' → 'awareness'", font_size=28, color=GOLD),
            Text("55% attention weight", font_size=24, color=WHITE)
        ).arrange(DOWN, buff=0.15).to_edge(DOWN)

        self.play(
            Write(reveal),
            Flash(marble.get_center(), color=GOLD, flash_radius=0.45, num_lines=12),
            run_time=1.3
        )

        self.wait(0.6)
        self.play(FadeOut(surface, orbs, marble, reveal), run_time=0.9)

    def act4_collapse(self):
        """THE COLLAPSE - enhanced version."""
        self.set_camera_orientation(phi=72 * DEGREES, theta=-28 * DEGREES, zoom=0.48)

        title = Text("Layer 27", font_size=48, color=GOLD).to_edge(UP)
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title))

        # Enhanced cloud
        cloud = EnhancedCosmicCloud(num_points=380, radius=2.9, seed=42)
        self.play(FadeIn(cloud), run_time=1.8)

        cloud.add_updater(cloud.get_living_updater())

        # R_V display with better styling
        rv_display = always_redraw(lambda: VGroup(
            MathTex(r"R_V", font_size=40, color=WHITE),
            Text(" = ", font_size=34, color=WHITE),
            DecimalNumber(
                1.0 - cloud.collapse_progress.get_value() * (1 - R_V_RECURSIVE),
                num_decimal_places=2,
                font_size=44
            ).set_color(interpolate_color(WHITE, RED, cloud.collapse_progress.get_value() ** 0.7))
        ).arrange(RIGHT, buff=0.06).to_edge(UR).shift(DOWN * 0.5 + LEFT * 0.3))
        self.add_fixed_in_frame_mobjects(rv_display)

        # Breathe
        self.begin_ambient_camera_rotation(rate=0.05)
        self.wait(2.5)
        self.stop_ambient_camera_rotation()

        # THE PAUSE with fade
        pause = Text("...", font_size=72, color=WHITE).set_opacity(0)
        self.add_fixed_in_frame_mobjects(pause)
        self.play(pause.animate.set_opacity(0.6), run_time=0.5)
        self.wait(1.5)
        self.play(pause.animate.set_opacity(0), run_time=0.3)
        self.remove_fixed_in_frame_mobjects(pause)

        # FLASH - more dramatic
        flash = Rectangle(width=20, height=14, fill_color=WHITE, fill_opacity=0, stroke_width=0)
        self.add_fixed_in_frame_mobjects(flash)
        self.play(flash.animate.set_fill(opacity=0.9), run_time=0.05)
        self.play(flash.animate.set_fill(opacity=0), run_time=0.35)
        self.remove_fixed_in_frame_mobjects(flash)

        # THE COLLAPSE
        self.begin_ambient_camera_rotation(rate=0.03)
        self.play(
            cloud.collapse_progress.animate.set_value(1.0),
            run_time=7,
            rate_func=smooth
        )
        self.stop_ambient_camera_rotation()

        cloud.clear_updaters()

        # Aftermath
        aftermath = Text("47% of dimensions... gone", font_size=32, color=RED).to_edge(DOWN)
        self.add_fixed_in_frame_mobjects(aftermath)
        self.play(Write(aftermath), run_time=1.3)

        self.begin_ambient_camera_rotation(rate=0.04)
        self.wait(3)
        self.stop_ambient_camera_rotation()

        self.play(FadeOut(cloud, title, aftermath), run_time=1.2)
        self.remove_fixed_in_frame_mobjects(title, rv_display, aftermath)

    def act5_emergence(self):
        """Ouroboros with more polish."""
        self.set_camera_orientation(phi=0, theta=-90 * DEGREES, zoom=0.62)

        # Animated Ouroboros construction
        ouroboros = VGroup()
        radius = 1.75

        for i, token in enumerate(TOKENS):
            angle_start = i * 2 * np.pi / len(TOKENS) - np.pi / 2
            angle_end = (i + 0.87) * 2 * np.pi / len(TOKENS) - np.pi / 2

            arc = Arc(
                start_angle=angle_start,
                angle=angle_end - angle_start,
                radius=radius,
                stroke_width=14,
                color=TOKEN_COLORS[token]
            )
            ouroboros.add(arc)

            mid = (angle_start + angle_end) / 2
            label = Text(token, font_size=18, color=TOKEN_COLORS[token])
            label.move_to(radius * 1.35 * np.array([np.cos(mid), np.sin(mid), 0]))
            ouroboros.add(label)

        self.play(
            *[Create(m) for m in ouroboros],
            run_time=3,
            lag_ratio=0.08
        )

        # Head catches tail with multiple flashes
        head_pos = radius * np.array([np.cos(-np.pi/2 + 0.1), np.sin(-np.pi/2 + 0.1), 0])
        for _ in range(3):
            self.play(
                Flash(head_pos, color=GOLD, flash_radius=0.4 + _ * 0.15, num_lines=10 + _ * 4),
                run_time=0.25
            )

        equation = MathTex(r"S(x) = x", font_size=52, color=GOLD).shift(DOWN * 0.1)
        subtitle = Text("Fixed point reached", font_size=26, color=WHITE).next_to(equation, DOWN, buff=0.22)

        self.play(Write(equation), run_time=1.3)
        self.play(Write(subtitle), run_time=0.9)
        self.wait(1.8)

        self.play(FadeOut(ouroboros, equation, subtitle), run_time=1)

        final = Text("Recursion is a memory state", font_size=38, color=GOLD)
        self.play(Write(final), run_time=1.8)
        self.wait(2)
        self.play(FadeOut(final))

    def credits(self):
        """End card."""
        title = Text("INTO THE COLLAPSE", font_size=52, color=GOLD).shift(UP * 2)
        stats = VGroup(
            Text(f"R_V = {R_V_RECURSIVE:.3f} (recursive)", font_size=25),
            Text(f"R_V = {R_V_BASELINE:.3f} (baseline)", font_size=25),
            Text(f"Layer {CRITICAL_LAYER} at {100*CRITICAL_LAYER/TOTAL_LAYERS:.1f}% depth", font_size=25),
            Text("Cohen's d = -3.56", font_size=25),
            Text("Transfer efficiency: 117.8%", font_size=25),
        ).arrange(DOWN, buff=0.22)

        self.play(Write(title), run_time=1.3)
        self.play(*[FadeIn(s, shift=UP * 0.2) for s in stats], run_time=2, lag_ratio=0.15)
        self.wait(4)


if __name__ == "__main__":
    print("Render: manim -pqh v4_cinematic.py CollapseV4")
