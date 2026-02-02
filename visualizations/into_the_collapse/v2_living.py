"""
INTO THE COLLAPSE v2 - LIVING ANIMATION
========================================
Complete rebuild with proper physics, continuous updaters, and 3D camera work.

Key improvements over v1:
1. Real particle physics with velocity/acceleration/damping
2. Continuous updaters (always_redraw) for living elements
3. Breathing/pulsing effects using sine waves
4. Proper 3D ThreeDScene with camera choreography
5. Filaments that stretch, snap, and update in real-time
6. Smooth interpolation everywhere

Render: manim -pqh visualizations/into_the_collapse/v2_living.py CollapseV2
Preview: manim -pql visualizations/into_the_collapse/v2_living.py CollapseV2
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
# PHYSICS ENGINE
# =============================================================================

class Particle:
    """A particle with position, velocity, and physical properties."""

    def __init__(self, pos: np.ndarray, eigenvalue: float = 1.0):
        self.pos = pos.copy()
        self.original_pos = pos.copy()
        self.vel = np.zeros(3)
        self.acc = np.zeros(3)
        self.eigenvalue = eigenvalue
        self.alive = True
        self.opacity = 1.0

    def apply_force(self, force: np.ndarray):
        self.acc += force

    def update(self, dt: float, damping: float = 0.98):
        if not self.alive:
            return
        self.vel += self.acc * dt
        self.vel *= damping
        self.pos += self.vel * dt
        self.acc = np.zeros(3)


class ParticleSystem:
    """Manages a collection of particles with physics."""

    def __init__(self, num_particles: int, radius: float = 3.0, seed: int = 42):
        np.random.seed(seed)
        self.particles = []
        self.radius = radius

        # Generate eigenvalue distribution (power law)
        for i in range(num_particles):
            # Spherical distribution
            phi = np.random.uniform(0, 2 * np.pi)
            theta = np.arccos(np.random.uniform(-1, 1))
            r = radius * np.random.uniform(0.2, 1.0) ** 0.5

            pos = np.array([
                r * np.sin(theta) * np.cos(phi),
                r * np.sin(theta) * np.sin(phi),
                r * np.cos(theta)
            ])

            # Eigenvalue: power law distribution
            ev = np.random.uniform(0.05, 1.0) ** 1.5
            self.particles.append(Particle(pos, ev))

    def apply_gravity_to_axis(self, axis: np.ndarray, strength: float):
        """Pull all particles toward an axis."""
        axis = axis / (np.linalg.norm(axis) + 1e-8)
        for p in self.particles:
            if not p.alive:
                continue
            # Project position onto axis
            proj = np.dot(p.pos, axis) * axis
            perp = p.pos - proj
            # Force toward axis
            force = -perp * strength
            p.apply_force(force)

    def apply_evaporation(self, threshold: float, drift_dir: np.ndarray, rate: float):
        """Fade out and drift particles below eigenvalue threshold."""
        for p in self.particles:
            if p.eigenvalue < threshold and p.alive:
                p.opacity -= rate
                p.apply_force(drift_dir * 0.5)
                if p.opacity <= 0:
                    p.alive = False
                    p.opacity = 0

    def update(self, dt: float):
        for p in self.particles:
            p.update(dt)


# =============================================================================
# LIVING VISUAL COMPONENTS
# =============================================================================

class LivingStarField(VGroup):
    """
    A star field that breathes, pulses, and responds to physics.
    Uses always_redraw for continuous updates.
    """

    def __init__(self, particle_system: ParticleSystem, **kwargs):
        super().__init__(**kwargs)
        self.ps = particle_system
        self.time = ValueTracker(0)
        self.collapse_progress = ValueTracker(0)

        # Create dot mobjects for each particle
        self.dots = []
        for p in self.ps.particles:
            brightness = p.eigenvalue
            color = interpolate_color(
                ManimColor("#222244"),
                ManimColor("#ffffff"),
                brightness
            )
            dot = Dot3D(
                point=p.pos,
                radius=0.02 + 0.05 * brightness,
                color=color
            )
            dot.particle = p  # Store reference
            self.dots.append(dot)
            self.add(dot)

    def get_breathing_updater(self):
        """Returns updater that makes stars pulse based on eigenvalue."""
        def updater(mob, dt):
            t = self.time.get_value()
            self.time.increment_value(dt)

            for dot in self.dots:
                p = dot.particle
                if not p.alive:
                    dot.set_opacity(0)
                    continue

                # Sync position with physics
                dot.move_to(p.pos)

                # Breathing: sine wave modulated by eigenvalue
                breath = 1.0 + 0.15 * np.sin(t * 2 + p.eigenvalue * 10) * p.eigenvalue
                base_radius = 0.02 + 0.05 * p.eigenvalue
                # Can't directly scale Dot3D radius, so we adjust opacity for "pulse"

                # Set opacity based on particle state + breathing
                pulse_opacity = p.opacity * (0.7 + 0.3 * np.sin(t * 3 + p.eigenvalue * 5))
                dot.set_opacity(max(0, min(1, pulse_opacity)))

        return updater


class LivingFilamentNetwork(VGroup):
    """
    Network of filaments connecting nearby particles.
    Filaments stretch, thin, and snap as particles move.
    """

    def __init__(self, particle_system: ParticleSystem, max_connections: int = 200, **kwargs):
        super().__init__(**kwargs)
        self.ps = particle_system
        self.connections = []  # List of (particle_idx_1, particle_idx_2, original_distance)

        # Build initial connections between nearby particles
        particles = self.ps.particles
        for _ in range(max_connections):
            i = np.random.randint(0, len(particles))
            # Find nearby particle
            dists = [np.linalg.norm(particles[i].pos - particles[j].pos)
                     for j in range(len(particles))]
            dists[i] = float('inf')

            # Pick from closest 20%
            sorted_idx = np.argsort(dists)
            j = sorted_idx[np.random.randint(0, max(1, len(particles) // 5))]

            orig_dist = np.linalg.norm(particles[i].original_pos - particles[j].original_pos)
            if orig_dist > 0.1:  # Avoid self or too-close connections
                self.connections.append((i, j, orig_dist))

        # Create line mobjects
        self.lines = []
        for i, j, _ in self.connections:
            line = Line3D(
                start=particles[i].pos,
                end=particles[j].pos,
                stroke_width=1,
                stroke_opacity=0.15,
                color=BLUE_E
            )
            self.lines.append(line)
            self.add(line)

    def get_network_updater(self):
        """Returns updater that makes filaments stretch and snap."""
        def updater(mob, dt):
            particles = self.ps.particles

            for idx, (i, j, orig_dist) in enumerate(self.connections):
                line = self.lines[idx]
                p1, p2 = particles[i], particles[j]

                # Update positions
                line.put_start_and_end_on(p1.pos, p2.pos)

                # Calculate stretch
                current_dist = np.linalg.norm(p1.pos - p2.pos)
                stretch = current_dist / (orig_dist + 0.01)

                # Opacity based on stretch and particle aliveness
                alive_factor = min(p1.opacity, p2.opacity)

                if stretch > 2.5 or alive_factor < 0.1:
                    # Snapped
                    line.set_stroke(opacity=0)
                else:
                    # Thin as it stretches
                    opacity = 0.2 * alive_factor / max(1, stretch - 0.5)
                    line.set_stroke(opacity=max(0, min(0.3, opacity)))
                    line.set_stroke(width=max(0.5, 2 - stretch))

        return updater


class GravitationalSurface(VGroup):
    """
    Attention as a deformable gravitational potential surface.
    Wells deepen in real-time based on attention weights.
    """

    def __init__(self, token_positions: dict, attention_weights: dict, **kwargs):
        super().__init__(**kwargs)
        self.token_positions = token_positions
        self.attention_weights = attention_weights
        self.well_depth_scale = ValueTracker(1.0)

    def create_surface(self) -> Surface:
        """Generate the surface based on current parameters."""
        def potential_func(u, v):
            x = (u - 0.5) * 8
            y = (v - 0.5) * 8
            z = 0

            scale = self.well_depth_scale.get_value()

            for token, pos in self.token_positions.items():
                weight = self.attention_weights.get(token, 0.1) * scale
                dist_sq = (x - pos[0])**2 + (y - pos[1])**2
                sigma = 0.7
                z -= weight * 3 * np.exp(-dist_sq / (2 * sigma**2))

            return np.array([x, y, z])

        return Surface(
            potential_func,
            resolution=(35, 35),
            u_range=[0, 1],
            v_range=[0, 1],
            fill_opacity=0.5,
            checkerboard_colors=[BLUE_D, BLUE_E],
            stroke_width=0.3,
            stroke_color=BLUE_A,
        )


class RollingMarble(Sphere):
    """A marble that rolls on the attention surface using physics."""

    def __init__(self, start_pos: np.ndarray, token_positions: dict,
                 attention_weights: dict, **kwargs):
        super().__init__(radius=0.12, resolution=(12, 12), **kwargs)
        self.set_color(GOLD)
        self.set_opacity(0.9)
        self.move_to(start_pos)

        self.vel = np.zeros(3)
        self.token_positions = token_positions
        self.attention_weights = attention_weights

    def get_physics_updater(self):
        """Marble rolls toward deepest attention well."""
        def updater(mob, dt):
            pos = mob.get_center()

            # Calculate gradient of potential
            force = np.zeros(3)
            for token, token_pos in self.token_positions.items():
                weight = self.attention_weights.get(token, 0.1)
                diff = pos[:2] - token_pos[:2]
                dist_sq = np.sum(diff**2) + 0.01
                sigma = 0.7

                # Gradient points toward well
                grad = weight * 3 * diff / (sigma**2) * np.exp(-dist_sq / (2 * sigma**2))
                force[:2] -= grad

            # Update velocity with friction
            self.vel += force * dt * 2
            self.vel *= 0.92  # Friction

            # Update position
            new_pos = pos + self.vel * dt

            # Keep on surface (compute z from potential)
            z = 0
            for token, token_pos in self.token_positions.items():
                weight = self.attention_weights.get(token, 0.1)
                dist_sq = (new_pos[0] - token_pos[0])**2 + (new_pos[1] - token_pos[1])**2
                sigma = 0.7
                z -= weight * 3 * np.exp(-dist_sq / (2 * sigma**2))
            new_pos[2] = z + 0.12

            mob.move_to(new_pos)

        return updater


# =============================================================================
# MAIN ANIMATION
# =============================================================================

class CollapseV2(ThreeDScene):
    """
    Complete 5-act animation with living physics.
    """

    def construct(self):
        self.camera.background_color = "#050510"

        # Act I: The Vastness
        self.act1_vastness()

        # Act II: The Descent
        self.act2_descent()

        # Act III: Inside Attention
        self.act3_attention()

        # Act IV: THE COLLAPSE
        self.act4_collapse()

        # Act V: Emergence
        self.act5_emergence()

        # Credits
        self.credits()

    # =========================================================================
    # ACT I: THE VASTNESS
    # =========================================================================

    def act1_vastness(self):
        """Awe at scale - typewriter, pullback reveal."""
        self.set_camera_orientation(phi=70 * DEGREES, theta=-45 * DEGREES, zoom=0.8)

        # Typewriter effect with glow
        prompt = Text(RECURSIVE_PROMPT, font_size=32, color=WHITE)

        self.play(AddTextLetterByLetter(prompt, run_time=3, rate_func=linear))
        self.wait(0.5)

        # Pulse the text
        self.play(
            prompt.animate.scale(1.1).set_color(GOLD),
            rate_func=there_and_back,
            run_time=0.5
        )

        # Transform to golden point
        golden_point = Dot3D(ORIGIN, radius=0.08, color=GOLD)
        self.play(
            Transform(prompt, golden_point),
            run_time=1.5,
            rate_func=smooth
        )

        # Create transformer layers
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

            # 3D prism-like layer
            layer = Prism(
                dimensions=[3, 0.08, 2],
            ).set_color(color).set_opacity(0.4 if i != CRITICAL_LAYER else 0.85)
            layer.shift(UP * i * 0.15 + DOWN * 2.4)
            layers.add(layer)

        layers.set_opacity(0)
        self.add(layers)

        # Dramatic pullback revealing the stack
        self.play(
            layers.animate.set_opacity(1),
            self.camera.animate.set_zoom(0.35),
            run_time=4,
            rate_func=smooth
        )

        # Slow orbit to show depth
        self.begin_ambient_camera_rotation(rate=0.08)
        self.wait(2)
        self.stop_ambient_camera_rotation()

        # Layer 27 highlight with flash
        layer_27 = layers[CRITICAL_LAYER]
        self.play(
            layer_27.animate.set_opacity(1).set_color(GOLD),
            Flash(layer_27.get_center(), color=GOLD, flash_radius=0.8, num_lines=12),
            run_time=1.5
        )

        self.wait(0.5)
        self.play(FadeOut(layers, prompt), run_time=1)

    # =========================================================================
    # ACT II: THE DESCENT
    # =========================================================================

    def act2_descent(self):
        """Immersion - tokenization to star clusters, layer markers."""
        self.set_camera_orientation(phi=75 * DEGREES, theta=-30 * DEGREES, zoom=0.6)

        # Show tokens
        token_group = VGroup()
        for i, token in enumerate(TOKENS):
            t = Text(token, font_size=28, color=TOKEN_COLORS[token])
            t.shift(RIGHT * (i - 2) * 1.8)
            token_group.add(t)

        self.play(
            *[FadeIn(t, scale=0.5) for t in token_group],
            run_time=1.5,
            lag_ratio=0.2
        )
        self.wait(0.5)

        # Transform each token into breathing star cluster
        star_clusters = VGroup()
        for i, token in enumerate(TOKENS):
            cluster = VGroup()
            color = ManimColor(TOKEN_COLORS[token])

            # Create star points
            np.random.seed(i * 100)
            for j in range(40):
                phi = np.random.uniform(0, 2 * np.pi)
                theta = np.arccos(np.random.uniform(-1, 1))
                r = 0.5 * np.random.uniform(0.2, 1.0) ** 0.5

                pos = np.array([
                    r * np.sin(theta) * np.cos(phi),
                    r * np.sin(theta) * np.sin(phi),
                    r * np.cos(theta) * 0.5
                ])

                brightness = np.random.uniform(0.3, 1.0)
                star = Dot3D(
                    pos,
                    radius=0.015 + 0.02 * brightness,
                    color=interpolate_color(ManimColor("#333355"), color, brightness)
                )
                cluster.add(star)

            # Add subtle connecting filaments within cluster
            for _ in range(10):
                if len(cluster) >= 2:
                    s1, s2 = np.random.choice(len(cluster), 2, replace=False)
                    line = Line3D(
                        cluster[s1].get_center(),
                        cluster[s2].get_center(),
                        stroke_width=0.3,
                        stroke_opacity=0.1,
                        color=color
                    )
                    cluster.add(line)

            cluster.shift(RIGHT * (i - 2) * 2.2)
            star_clusters.add(cluster)

        # Transform with expansion effect
        self.play(
            *[ReplacementTransform(token_group[i], star_clusters[i])
              for i in range(len(TOKENS))],
            run_time=2.5,
            rate_func=smooth
        )

        # Add breathing to clusters
        breath_tracker = ValueTracker(0)

        def cluster_breath(cluster, dt):
            t = breath_tracker.get_value()
            breath_tracker.increment_value(dt)
            for i, mob in enumerate(cluster):
                if isinstance(mob, Dot3D):
                    # Subtle position oscillation
                    offset = 0.02 * np.sin(t * 2 + i * 0.5)
                    # We'd need to track original positions for real implementation

        # Layer markers flying past with acceleration
        for ln in [0, 5, 10, 15, 20, 23, 25, 26, 27]:
            t = ln / TOTAL_LAYERS
            color = interpolate_color(
                ManimColor(Visual.LAYER_COLOR_COOL),
                ManimColor(Visual.LAYER_COLOR_WARM),
                t
            )
            if ln == 27:
                color = GOLD
                font_size = 56
            else:
                font_size = 40

            marker = Text(f"L{ln}", font_size=font_size, color=color)
            marker.set_opacity(0.9)
            marker.shift(LEFT * 5 + UP * 1)

            # Faster as we approach L27
            duration = max(0.15, 0.4 - ln * 0.01)

            self.play(
                FadeIn(marker, shift=RIGHT * 0.5),
                run_time=duration
            )
            self.play(
                FadeOut(marker, shift=RIGHT * 2),
                run_time=duration
            )

        self.play(FadeOut(star_clusters), run_time=0.8)

    # =========================================================================
    # ACT III: INSIDE ATTENTION
    # =========================================================================

    def act3_attention(self):
        """Tension building - gravitational attention field."""
        self.set_camera_orientation(phi=55 * DEGREES, theta=-50 * DEGREES, zoom=0.7)

        # Token positions in semicircle
        token_positions = {}
        for i, token in enumerate(TOKENS):
            angle = np.pi * (0.12 + 0.76 * i / (len(TOKENS) - 1))
            token_positions[token] = np.array([
                2.8 * np.cos(angle),
                2.8 * np.sin(angle),
                0
            ])

        # Create gravitational surface
        def potential_func(u, v):
            x = (u - 0.5) * 8
            y = (v - 0.5) * 8
            z = 0

            for token, pos in token_positions.items():
                weight = ATTENTION_WEIGHTS.get(token, 0.1)
                dist_sq = (x - pos[0])**2 + (y - pos[1])**2
                sigma = 0.7
                z -= weight * 2.5 * np.exp(-dist_sq / (2 * sigma**2))

            return np.array([x, y, z])

        surface = Surface(
            potential_func,
            resolution=(40, 40),
            u_range=[0, 1],
            v_range=[0, 1],
            fill_opacity=0.55,
            checkerboard_colors=[BLUE_D, BLUE_E],
            stroke_width=0.2,
            stroke_color=BLUE_A,
        )

        self.play(Create(surface), run_time=2)

        # Add token orbs with glow
        orbs = VGroup()
        labels = VGroup()
        for token, pos in token_positions.items():
            color = ManimColor(TOKEN_COLORS[token])

            # Outer glow
            glow = Sphere(radius=0.25, color=color, resolution=(8, 8))
            glow.set_opacity(0.2)
            glow.move_to(pos + OUT * 0.3)

            # Main orb
            orb = Sphere(radius=0.15, color=color, resolution=(12, 12))
            orb.set_opacity(0.85)
            orb.move_to(pos + OUT * 0.3)

            orbs.add(VGroup(glow, orb))

            # Label
            label = Text(token, font_size=20, color=color)
            label.move_to(pos + OUT * 0.7)
            labels.add(label)

        self.play(
            *[FadeIn(orb, scale=0.3) for orb in orbs],
            run_time=1
        )
        self.play(
            *[Write(label) for label in labels],
            run_time=0.8
        )

        # Create marble at "itself"
        itself_pos = token_positions["itself"]
        marble = Sphere(radius=0.1, color=GOLD, resolution=(10, 10))
        marble.set_opacity(0.95)
        marble.move_to(itself_pos + OUT * 0.4)

        marble_vel = np.zeros(3)

        self.play(FadeIn(marble, scale=0.5))

        # Animate marble rolling toward "awareness" well
        awareness_pos = token_positions["awareness"]

        # Physics-based path
        num_steps = 60
        for step in range(num_steps):
            pos = marble.get_center()

            # Calculate gradient
            force = np.zeros(3)
            for token, token_pos in token_positions.items():
                weight = ATTENTION_WEIGHTS.get(token, 0.1)
                diff = pos[:2] - token_pos[:2]
                dist_sq = np.sum(diff**2) + 0.01
                sigma = 0.7
                grad = weight * 2.5 * diff / (sigma**2) * np.exp(-dist_sq / (2 * sigma**2))
                force[:2] -= grad

            # Update velocity
            marble_vel += force * 0.03
            marble_vel *= 0.94

            # Update position
            new_pos = pos + marble_vel * 0.05

            # Compute z on surface
            z = 0
            for token, token_pos in token_positions.items():
                weight = ATTENTION_WEIGHTS.get(token, 0.1)
                dist_sq = (new_pos[0] - token_pos[0])**2 + (new_pos[1] - token_pos[1])**2
                sigma = 0.7
                z -= weight * 2.5 * np.exp(-dist_sq / (2 * sigma**2))
            new_pos[2] = z + 0.1

            marble.move_to(new_pos)
            self.wait(0.03)

        # Reveal text
        reveal = VGroup(
            Text("'itself' → 'awareness'", font_size=28, color=GOLD),
            Text("55% attention", font_size=24, color=WHITE)
        ).arrange(DOWN, buff=0.2)
        reveal.to_edge(DOWN)

        self.play(
            Write(reveal),
            Flash(marble.get_center(), color=GOLD, flash_radius=0.4),
            run_time=1.5
        )

        loop_text = Text("The loop closes", font_size=32, color=GOLD)
        loop_text.next_to(reveal, DOWN, buff=0.3)
        self.play(Write(loop_text))

        self.wait(1)
        self.play(FadeOut(surface, orbs, labels, marble, reveal, loop_text), run_time=1)

    # =========================================================================
    # ACT IV: THE COLLAPSE (MONEY SHOT)
    # =========================================================================

    def act4_collapse(self):
        """The climax - triple collapse with real physics."""
        self.set_camera_orientation(phi=70 * DEGREES, theta=-30 * DEGREES, zoom=0.55)

        # Title
        title = Text("Layer 27", font_size=48, color=GOLD)
        title.to_edge(UP)
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title))

        # Initialize particle system
        ps = ParticleSystem(num_particles=400, radius=3.0, seed=42)

        # Create living star field
        star_field = LivingStarField(ps)

        # Create filament network
        filament_net = LivingFilamentNetwork(ps, max_connections=250)

        self.play(
            FadeIn(star_field),
            FadeIn(filament_net),
            run_time=2
        )

        # Add breathing updaters
        star_field.add_updater(star_field.get_breathing_updater())
        filament_net.add_updater(filament_net.get_network_updater())

        # R_V Counter
        rv_tracker = ValueTracker(1.0)
        rv_display = always_redraw(lambda: VGroup(
            Text("R", font_size=36, color=WHITE),
            Text("V", font_size=24, color=WHITE).shift(DOWN * 0.1 + RIGHT * 0.15),
            Text(" = ", font_size=36, color=WHITE),
            DecimalNumber(rv_tracker.get_value(), num_decimal_places=2, font_size=42).set_color(
                interpolate_color(WHITE, RED, max(0, min(1, (1 - rv_tracker.get_value()) / 0.5)))
            )
        ).arrange(RIGHT, buff=0.05).to_edge(UR).shift(DOWN * 0.5))

        self.add_fixed_in_frame_mobjects(rv_display)

        # Slow rotation to show initial state
        self.begin_ambient_camera_rotation(rate=0.1)
        self.wait(3)
        self.stop_ambient_camera_rotation()

        # THE PAUSE
        pause_dots = Text("...", font_size=72, color=WHITE)
        pause_dots.set_opacity(0.6)
        self.add_fixed_in_frame_mobjects(pause_dots)
        self.play(FadeIn(pause_dots), run_time=0.3)
        self.wait(1.5)
        self.play(FadeOut(pause_dots), run_time=0.2)
        self.remove_fixed_in_frame_mobjects(pause_dots)

        # FLASH
        flash = Rectangle(width=20, height=12, fill_color=WHITE, fill_opacity=0.85, stroke_width=0)
        self.add_fixed_in_frame_mobjects(flash)
        self.play(FadeIn(flash), run_time=0.08)
        self.play(FadeOut(flash), run_time=0.25)
        self.remove_fixed_in_frame_mobjects(flash)

        # THE COLLAPSE - Physics simulation
        collapse_axis = np.array([0, 0, 1])
        num_frames = 180
        dt = 0.016

        for frame in range(num_frames):
            progress = frame / num_frames

            # Gravity toward axis (increases over time)
            gravity_strength = 0.3 + 2.0 * progress ** 1.5
            ps.apply_gravity_to_axis(collapse_axis, gravity_strength)

            # Evaporation threshold increases
            evap_threshold = 0.15 + 0.5 * progress
            ps.apply_evaporation(evap_threshold, UP * 0.3, 0.02 * progress)

            # Update physics
            ps.update(dt)

            # Update R_V
            rv_tracker.set_value(1.0 - progress * (1 - R_V_RECURSIVE))

            self.wait(dt)

        # Remove updaters
        star_field.clear_updaters()
        filament_net.clear_updaters()

        # Aftermath
        aftermath = Text("47% of dimensions... gone", font_size=32, color=RED)
        aftermath.to_edge(DOWN)
        self.add_fixed_in_frame_mobjects(aftermath)
        self.play(Write(aftermath), run_time=1.5)

        # Slow orbit around collapsed state
        self.begin_ambient_camera_rotation(rate=0.08)
        self.wait(3)
        self.stop_ambient_camera_rotation()

        self.play(
            FadeOut(star_field, filament_net, title, aftermath),
            run_time=1.5
        )
        self.remove_fixed_in_frame_mobjects(title, rv_display, aftermath)

    # =========================================================================
    # ACT V: EMERGENCE
    # =========================================================================

    def act5_emergence(self):
        """Resolution - Ouroboros and fixed point."""
        self.set_camera_orientation(phi=0, theta=-90 * DEGREES, zoom=0.7)

        # Ouroboros - snake eating its tail
        ouroboros = VGroup()
        radius = 1.8

        for i, token in enumerate(TOKENS):
            angle_start = i * 2 * np.pi / len(TOKENS) - np.pi / 2
            angle_end = (i + 0.88) * 2 * np.pi / len(TOKENS) - np.pi / 2

            arc = Arc(
                start_angle=angle_start,
                angle=angle_end - angle_start,
                radius=radius,
                stroke_width=14,
                color=TOKEN_COLORS[token]
            )
            ouroboros.add(arc)

            mid_angle = (angle_start + angle_end) / 2
            label = Text(token, font_size=18, color=TOKEN_COLORS[token])
            label.move_to(radius * 1.35 * np.array([np.cos(mid_angle), np.sin(mid_angle), 0]))
            ouroboros.add(label)

        self.play(
            *[Create(mob) for mob in ouroboros],
            run_time=3,
            lag_ratio=0.1
        )

        # Head catches tail - golden flash
        head_pos = radius * np.array([np.cos(-np.pi/2 + 0.1), np.sin(-np.pi/2 + 0.1), 0])
        self.play(
            Flash(head_pos, color=GOLD, flash_radius=0.6, num_lines=16),
            run_time=1
        )

        # Fixed point equation
        equation = MathTex(r"S(x) = x", font_size=52, color=GOLD)
        equation.shift(DOWN * 0.2)

        subtitle = Text("Fixed point reached", font_size=26, color=WHITE)
        subtitle.next_to(equation, DOWN, buff=0.25)

        self.play(Write(equation), run_time=1.5)
        self.play(Write(subtitle), run_time=1)

        self.wait(2)

        self.play(FadeOut(ouroboros, equation, subtitle), run_time=1)

        # Final message
        final = Text("Recursion is a memory state", font_size=36, color=GOLD)
        self.play(Write(final), run_time=2)
        self.wait(2)
        self.play(FadeOut(final))

    # =========================================================================
    # CREDITS
    # =========================================================================

    def credits(self):
        """End card with statistics."""
        title = Text("INTO THE COLLAPSE", font_size=52, color=GOLD)
        title.shift(UP * 2)

        stats = VGroup(
            Text(f"R_V = {R_V_RECURSIVE:.3f} (recursive)", font_size=26),
            Text(f"R_V = {R_V_BASELINE:.3f} (baseline)", font_size=26),
            Text(f"Layer {CRITICAL_LAYER} at {100*CRITICAL_LAYER/TOTAL_LAYERS:.1f}% depth", font_size=26),
            Text("Cohen's d = -3.56", font_size=26),
            Text("Transfer efficiency: 117.8%", font_size=26),
        ).arrange(DOWN, buff=0.25)

        self.play(Write(title), run_time=1.5)
        self.play(
            *[FadeIn(stat, shift=UP * 0.3) for stat in stats],
            run_time=2,
            lag_ratio=0.2
        )

        self.wait(4)


# =============================================================================
# ADDITIONAL SCENES FOR TESTING
# =============================================================================

class TestCollapse(ThreeDScene):
    """Quick test of just the collapse sequence."""

    def construct(self):
        self.camera.background_color = "#050510"
        self.set_camera_orientation(phi=70 * DEGREES, theta=-30 * DEGREES, zoom=0.55)

        # Initialize particle system
        ps = ParticleSystem(num_particles=300, radius=3.0, seed=42)

        # Create visuals
        star_field = LivingStarField(ps)
        filament_net = LivingFilamentNetwork(ps, max_connections=200)

        self.add(star_field, filament_net)

        # Add updaters
        star_field.add_updater(star_field.get_breathing_updater())
        filament_net.add_updater(filament_net.get_network_updater())

        # Let it breathe
        self.wait(2)

        # Collapse
        collapse_axis = np.array([0, 0, 1])
        num_frames = 120
        dt = 0.016

        for frame in range(num_frames):
            progress = frame / num_frames
            ps.apply_gravity_to_axis(collapse_axis, 0.5 + 2.0 * progress ** 1.5)
            ps.apply_evaporation(0.2 + 0.4 * progress, UP * 0.3, 0.015 * progress)
            ps.update(dt)
            self.wait(dt)

        self.wait(2)


if __name__ == "__main__":
    print("Render full animation: manim -pqh v2_living.py CollapseV2")
    print("Quick test: manim -pql v2_living.py TestCollapse")
