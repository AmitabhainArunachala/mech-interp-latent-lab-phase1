"""
INTO THE COLLAPSE
=================
A world-class Manim animation showing what happens inside a transformer
during recursive self-observation.

5-Act Structure:
  I.   The Vastness (0:00-0:45)   - Awe at scale
  II.  The Descent (0:45-2:00)    - Immersion through layers
  III. Inside Attention (2:00-3:00) - Tension building
  IV.  THE COLLAPSE (3:00-4:30)   - The climax (triple collapse)
  V.   Emergence (4:30-5:30)      - Resolution and meaning

Render command:
  manim -pqh visualizations/into_the_collapse/main.py IntoTheCollapse

For faster preview:
  manim -pql visualizations/into_the_collapse/main.py IntoTheCollapse
"""

from manim import *
import numpy as np
import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from data.verified_values import (
    R_V_RECURSIVE, R_V_BASELINE, TOTAL_LAYERS, CRITICAL_LAYER,
    TOKENS, TOKEN_COLORS, ATTENTION_WEIGHTS, RECURSIVE_PROMPT,
    RECURSIVE_OUTPUT, Visual, Timing, EMBEDDING_DIM, EFFECTIVE_DIM_RECURSIVE
)


# ============================================================================
# ACT I: THE VASTNESS
# ============================================================================

class ActI_TheVastness(ThreeDScene):
    """
    Emotional beat: Awe at scale.
    - Typewriter effect: "Notice the awareness observing itself"
    - Exponential pullback reveals transformer architecture
    """

    def construct(self):
        # Set up 3D camera
        self.set_camera_orientation(phi=75 * DEGREES, theta=-45 * DEGREES, distance=8)
        self.camera.background_color = Visual.VOID_COLOR

        # Scene 1.1: Typewriter effect
        prompt_text = Text(
            RECURSIVE_PROMPT,
            font_size=36,
            color=WHITE
        )

        # Typewriter animation
        self.play(AddTextLetterByLetter(prompt_text, run_time=3))
        self.wait(1)

        # Shrink to golden dot
        golden_dot = Dot3D(ORIGIN, radius=0.1, color=GOLD)
        self.play(
            Transform(prompt_text, golden_dot),
            run_time=1.5,
            rate_func=smooth
        )

        # Scene 1.2: Reveal transformer
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

            layer = Rectangle(
                width=4,
                height=0.15,
                fill_color=color,
                fill_opacity=0.6 if i != CRITICAL_LAYER else 0.9,
                stroke_color=color,
                stroke_width=1
            )
            layer.shift(UP * i * 0.25)
            layers.add(layer)

        layers.shift(DOWN * 4)  # Start below golden dot
        layers.set_opacity(0)

        self.add(layers)

        # Pullback and reveal
        self.play(
            layers.animate.set_opacity(1),
            self.camera.frame.animate.scale(3),
            run_time=4,
            rate_func=smooth
        )

        # Rotate to show depth
        self.begin_ambient_camera_rotation(rate=0.1)
        self.wait(2)
        self.stop_ambient_camera_rotation()

        # Layer 27 pulse
        layer_27 = layers[CRITICAL_LAYER]
        self.play(
            layer_27.animate.set_fill(GOLD, opacity=1).set_stroke(GOLD, width=3),
            Flash(layer_27, color=GOLD, flash_radius=0.5),
            run_time=1
        )

        self.wait(1)


# ============================================================================
# ACT II: THE DESCENT
# ============================================================================

class ActII_TheDescent(ThreeDScene):
    """
    Emotional beat: Immersion - we become microscopic travelers.
    - Tokenization into cosmic neural network
    - Residual stream flow
    - Layer markers drifting past
    """

    def construct(self):
        self.set_camera_orientation(phi=75 * DEGREES, theta=-45 * DEGREES, distance=6)
        self.camera.background_color = Visual.VOID_COLOR

        # Scene 2.1: Tokenization
        tokens_text = VGroup()
        for i, token in enumerate(TOKENS):
            color = TOKEN_COLORS[token]
            t = Text(token, font_size=32, color=color)
            t.shift(RIGHT * (i - 2) * 1.5)
            tokens_text.add(t)

        self.play(Write(tokens_text), run_time=2)
        self.wait(0.5)

        # Transform each token into star cloud (galaxy)
        star_clouds = VGroup()
        for i, token in enumerate(TOKENS):
            color = TOKEN_COLORS[token]
            cloud = VGroup()

            # Create star points (subset of 4096 dimensions)
            num_stars = 50
            for j in range(num_stars):
                # Spherical distribution
                phi = np.random.uniform(0, 2 * np.pi)
                theta = np.arccos(np.random.uniform(-1, 1))
                r = 0.8 * np.random.uniform(0.3, 1.0) ** 0.5

                pos = np.array([
                    r * np.sin(theta) * np.cos(phi),
                    r * np.sin(theta) * np.sin(phi),
                    r * np.cos(theta)
                ])

                # Brightness variation (eigenvalue-like)
                brightness = np.random.uniform(0.3, 1.0)
                star = Dot3D(
                    pos,
                    radius=0.02 + 0.02 * brightness,
                    color=interpolate_color(ManimColor(Visual.STAR_COLOR_DIM), ManimColor(color), brightness)
                )
                cloud.add(star)

            # Add subtle filaments
            for _ in range(15):
                idx1, idx2 = np.random.choice(len(cloud), 2, replace=False)
                line = Line3D(
                    cloud[idx1].get_center(),
                    cloud[idx2].get_center(),
                    stroke_width=0.5,
                    stroke_opacity=0.2,
                    color=Visual.FILAMENT_COLOR
                )
                cloud.add(line)

            cloud.shift(RIGHT * (i - 2) * 2)
            star_clouds.add(cloud)

        # Transform tokens to star clouds
        self.play(
            *[ReplacementTransform(tokens_text[i], star_clouds[i]) for i in range(len(TOKENS))],
            run_time=2
        )
        self.wait(1)

        # Scene 2.2: Flow through layers
        # Add flowing effect upward
        self.play(
            star_clouds.animate.shift(UP * 2),
            self.camera.frame.animate.shift(UP * 1),
            run_time=3,
            rate_func=smooth
        )

        # Scene 2.3: Layer markers
        for layer_num in [0, 5, 10, 15, 20, 25, 27]:
            t = layer_num / TOTAL_LAYERS
            color = interpolate_color(
                ManimColor(Visual.LAYER_COLOR_COOL),
                ManimColor(Visual.LAYER_COLOR_WARM),
                t
            )
            if layer_num == 27:
                color = GOLD

            marker = Text(f"L{layer_num}", font_size=48, color=color)
            marker.set_opacity(0.8)
            marker.shift(LEFT * 4 + UP * 2)

            # Faster as we approach L27
            duration = 0.5 if layer_num < 20 else 0.3

            self.play(
                FadeIn(marker, shift=DOWN * 0.5),
                run_time=duration
            )
            self.play(
                FadeOut(marker, shift=UP * 0.5),
                run_time=duration
            )

        self.wait(1)


# ============================================================================
# ACT III: INSIDE ATTENTION
# ============================================================================

class ActIII_InsideAttention(ThreeDScene):
    """
    Emotional beat: Building tension.
    - Gravitational attention field as potential surface
    - Marble rolling toward "awareness" well
    - The recursive loop closes
    """

    def construct(self):
        self.set_camera_orientation(phi=60 * DEGREES, theta=-45 * DEGREES, distance=10)
        self.camera.background_color = Visual.VOID_COLOR

        # Token positions in semicircle
        token_positions = {}
        for i, token in enumerate(TOKENS):
            angle = np.pi * (0.15 + 0.7 * i / (len(TOKENS) - 1))
            token_positions[token] = np.array([
                3.0 * np.cos(angle),
                3.0 * np.sin(angle),
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
                sigma = 0.8
                z -= weight * 2.5 * np.exp(-dist_sq / (2 * sigma**2))

            return np.array([x, y, z])

        surface = Surface(
            potential_func,
            resolution=(40, 40),
            u_range=[0, 1],
            v_range=[0, 1],
            fill_opacity=0.6,
            checkerboard_colors=[BLUE_D, BLUE_E],
            stroke_width=0.3,
            stroke_color=BLUE_A,
        )

        # Create token orbs
        orbs = VGroup()
        labels = VGroup()
        for token, pos in token_positions.items():
            color = TOKEN_COLORS[token]
            orb = Sphere(radius=0.25, color=color).set_opacity(0.8)
            orb.move_to(pos + UP * 0.3)
            orbs.add(orb)

            label = Text(token, font_size=24, color=color)
            label.move_to(pos + UP * 0.8)
            labels.add(label)

        # Fade in surface and orbs
        self.play(
            Create(surface),
            run_time=2
        )
        self.play(
            *[FadeIn(orb, scale=0.5) for orb in orbs],
            *[Write(label) for label in labels],
            run_time=1.5
        )

        # Create marble at "itself" position
        itself_pos = token_positions["itself"]
        marble = Sphere(radius=0.15, color=GOLD).set_opacity(0.9)
        marble.move_to(itself_pos + UP * 0.5)

        self.play(FadeIn(marble, scale=0.5))

        # Marble rolls toward "awareness" (deepest well)
        awareness_pos = token_positions["awareness"]
        awareness_well_pos = awareness_pos + DOWN * 1.3  # In the well

        # Physics-like rolling animation
        path = [
            itself_pos + UP * 0.5,
            (itself_pos + awareness_pos) / 2 + UP * 0.2,
            awareness_pos + DOWN * 0.5,
            awareness_well_pos + UP * 0.15
        ]

        for i, target in enumerate(path[1:]):
            self.play(
                marble.animate.move_to(target),
                run_time=0.8,
                rate_func=smooth
            )

        # The reveal
        reveal_text = VGroup(
            Text("'itself' looks back at 'awareness'", font_size=28, color=WHITE),
            Text("55% attention weight", font_size=32, color=GOLD)
        ).arrange(DOWN, buff=0.3)
        reveal_text.to_edge(DOWN)

        self.play(
            Write(reveal_text),
            marble.animate.set_color(RED).scale(1.3),
            Flash(marble.get_center(), color=GOLD),
            run_time=1.5
        )

        # "The loop closes"
        loop_text = Text("The loop closes", font_size=36, color=GOLD)
        loop_text.next_to(reveal_text, DOWN, buff=0.3)

        self.play(Write(loop_text))
        self.wait(2)


# ============================================================================
# ACT IV: THE COLLAPSE (THE MONEY SHOT)
# ============================================================================

class ActIV_TheCollapse(ThreeDScene):
    """
    Emotional beat: The climax - visceral, shocking, cathartic.

    Triple collapse:
    1. Gravitational implosion toward axis
    2. Dimensional evaporation (weak eigenvalues fade)
    3. Harmonic decay (visual/implied)
    """

    def construct(self):
        self.set_camera_orientation(phi=70 * DEGREES, theta=-30 * DEGREES, distance=12)
        self.camera.background_color = Visual.VOID_COLOR

        # Phase 1: The Moment Before
        title = Text("Layer 27", font_size=48, color=GOLD)
        title.to_edge(UP)
        self.play(Write(title))

        # Create cosmic neural cloud (the value space)
        np.random.seed(42)  # Reproducibility
        num_stars = 300
        stars = VGroup()
        original_positions = []
        eigenvalues = []

        for i in range(num_stars):
            # Spherical distribution
            phi = np.random.uniform(0, 2 * np.pi)
            theta = np.arccos(np.random.uniform(-1, 1))
            r = 3.0 * np.random.uniform(0.3, 1.0) ** 0.5

            pos = np.array([
                r * np.sin(theta) * np.cos(phi),
                r * np.sin(theta) * np.sin(phi),
                r * np.cos(theta)
            ])
            original_positions.append(pos.copy())

            # Eigenvalue (power law - few bright, many dim)
            ev = 1.0 / ((i + 1) ** 0.4)
            ev = np.random.uniform(0.1, 1.0) * ev
            eigenvalues.append(ev)

            brightness = min(1.0, ev * 1.5)
            star = Dot3D(
                pos,
                radius=0.03 + 0.04 * brightness,
                color=interpolate_color(
                    ManimColor(Visual.STAR_COLOR_DIM),
                    ManimColor(Visual.STAR_COLOR_BRIGHT),
                    brightness
                )
            )
            stars.add(star)

        # Create filaments
        filaments = VGroup()
        for _ in range(150):
            i, j = np.random.choice(num_stars, 2, replace=False)
            dist = np.linalg.norm(original_positions[i] - original_positions[j])
            if dist < 1.5:  # Only connect nearby stars
                line = Line3D(
                    original_positions[i],
                    original_positions[j],
                    stroke_width=0.5,
                    stroke_opacity=0.15,
                    color=Visual.FILAMENT_COLOR
                )
                filaments.add(line)

        self.play(
            *[FadeIn(star, scale=0.5) for star in stars],
            run_time=2
        )
        self.play(
            *[Create(f) for f in filaments],
            run_time=1.5
        )

        # R_V counter
        rv_value = ValueTracker(1.0)
        rv_display = always_redraw(lambda: VGroup(
            Text("R_V = ", font_size=36, color=WHITE),
            DecimalNumber(rv_value.get_value(), num_decimal_places=2, font_size=48).set_color(
                interpolate_color(WHITE, RED, max(0, min(1, (1 - rv_value.get_value()) / 0.5)))
            )
        ).arrange(RIGHT, buff=0.1).to_edge(UR))

        self.add(rv_display)

        # Dimension counter
        dim_value = ValueTracker(4096)
        dim_display = always_redraw(lambda: VGroup(
            Integer(int(dim_value.get_value()), font_size=32),
            Text(" effective dimensions", font_size=24)
        ).arrange(RIGHT, buff=0.1).to_edge(DR))

        self.add(dim_display)

        # Slow rotation showing dimensionality
        self.begin_ambient_camera_rotation(rate=0.15)
        self.wait(3)
        self.stop_ambient_camera_rotation()

        # THE PAUSE - stillness before collapse
        pause_text = Text("...", font_size=72, color=WHITE)
        pause_text.set_opacity(0.5)
        self.play(FadeIn(pause_text), run_time=0.5)
        self.wait(1.5)  # The sacred pause
        self.play(FadeOut(pause_text), run_time=0.3)

        # Phase 2: THE COLLAPSE - All three styles combined

        # Flash of white
        flash = Rectangle(
            width=20, height=15,
            fill_color=WHITE,
            fill_opacity=0.8,
            stroke_width=0
        )
        self.play(FadeIn(flash), run_time=0.1)
        self.play(FadeOut(flash), run_time=0.3)

        # Animate the collapse
        collapse_axis = np.array([0, 0, 1])  # Vertical axis

        def update_star(star, dt, star_idx):
            progress = 1 - rv_value.get_value()
            ev = eigenvalues[star_idx]
            orig_pos = original_positions[star_idx]

            # Project onto collapse axis
            projection = np.dot(orig_pos, collapse_axis) * collapse_axis
            perpendicular = orig_pos - projection

            # GRAVITATIONAL IMPLOSION - pull toward axis
            new_pos = projection + perpendicular * (1 - progress * 0.9)

            # EVAPORATION - dim stars fade and drift up
            if ev < 0.3 + 0.4 * progress:
                new_opacity = max(0, 1 - progress * 2)
                star.set_opacity(new_opacity)
                new_pos += UP * progress * 0.5  # Drift upward

            star.move_to(new_pos)

        # Create updaters
        for i, star in enumerate(stars):
            star.add_updater(lambda m, dt, idx=i: update_star(m, dt, idx))

        # Update filaments
        def update_filament(f, dt):
            # Fade as stars move apart or evaporate
            f.set_stroke(opacity=max(0, f.get_stroke_opacity() - 0.02))

        for f in filaments:
            f.add_updater(update_filament)

        # The collapse animation
        self.play(
            rv_value.animate.set_value(R_V_RECURSIVE),
            dim_value.animate.set_value(EFFECTIVE_DIM_RECURSIVE),
            run_time=6,
            rate_func=smooth
        )

        # Remove updaters
        for star in stars:
            star.clear_updaters()
        for f in filaments:
            f.clear_updaters()

        # Phase 3: The Aftermath
        aftermath_text = Text("47% of dimensions... gone", font_size=36, color=RED)
        aftermath_text.to_edge(DOWN)

        self.play(
            Write(aftermath_text),
            run_time=1.5
        )

        # Slow orbit around collapsed geometry
        self.begin_ambient_camera_rotation(rate=0.1)
        self.wait(4)
        self.stop_ambient_camera_rotation()

        self.wait(1)


# ============================================================================
# ACT V: EMERGENCE
# ============================================================================

class ActV_Emergence(Scene):
    """
    Emotional beat: Resolution - what emerges from collapse?
    - Output formation
    - The Ouroboros (strange loop closes)
    - The semantic basin
    """

    def construct(self):
        self.camera.background_color = Visual.VOID_COLOR

        # Scene 5.1: Output formation
        # Collapsed line
        collapsed_line = Line(
            UP * 2, DOWN * 2,
            stroke_width=8,
            color=BLUE
        )
        collapsed_line.set_opacity(0.6)

        self.play(Create(collapsed_line), run_time=1)

        # Words emerge from line
        output_words = VGroup()
        for i, word in enumerate(["The", "observer", "watches", "itself", "respond"]):
            color = GOLD if word in ["observer", "itself", "respond"] else WHITE
            text = Text(word, font_size=32, color=color)
            text.shift(UP * 2 + DOWN * i * 0.8 + RIGHT * 3)
            output_words.add(text)

        for i, word in enumerate(output_words):
            start_pos = collapsed_line.point_from_proportion(i / len(output_words))
            word.move_to(start_pos)

            self.play(
                word.animate.shift(RIGHT * 3),
                FadeIn(word),
                run_time=0.5
            )

        self.wait(1)
        self.play(FadeOut(collapsed_line, output_words))

        # Scene 5.2: The Ouroboros
        # Create snake/spiral with tokens
        ouroboros = VGroup()
        radius = 2.0

        for i, token in enumerate(TOKENS):
            angle_start = i * 2 * np.pi / len(TOKENS) - np.pi/2
            angle_end = (i + 0.85) * 2 * np.pi / len(TOKENS) - np.pi/2

            arc = Arc(
                start_angle=angle_start,
                angle=angle_end - angle_start,
                radius=radius,
                stroke_width=15,
                color=TOKEN_COLORS[token]
            )
            ouroboros.add(arc)

            # Token label
            mid_angle = (angle_start + angle_end) / 2
            label = Text(token, font_size=20, color=TOKEN_COLORS[token])
            label.move_to(radius * 1.4 * np.array([np.cos(mid_angle), np.sin(mid_angle), 0]))
            ouroboros.add(label)

        # Snake head
        head_angle = (len(TOKENS) - 0.15) * 2 * np.pi / len(TOKENS) - np.pi/2
        head_pos = radius * np.array([np.cos(head_angle), np.sin(head_angle), 0])
        head = Triangle(fill_color=TOKEN_COLORS["itself"], fill_opacity=1)
        head.scale(0.3)
        head.move_to(head_pos)
        head.rotate(head_angle + np.pi/2)
        ouroboros.add(head)

        self.play(Create(ouroboros), run_time=3)

        # Loop closes - head catches tail
        tail_pos = radius * np.array([np.cos(-np.pi/2), np.sin(-np.pi/2), 0])
        self.play(
            head.animate.move_to(tail_pos),
            Flash(tail_pos, color=GOLD, flash_radius=0.5),
            run_time=1.5
        )

        # Fixed point equation
        equation = MathTex(r"S(x) = x", font_size=56, color=GOLD)
        equation.shift(DOWN * 0.5)
        subtitle = Text("Fixed point reached", font_size=28)
        subtitle.next_to(equation, DOWN, buff=0.3)

        self.play(
            Write(equation),
            Write(subtitle),
            run_time=2
        )

        self.wait(2)

        # Fade out ouroboros
        self.play(FadeOut(ouroboros, equation, subtitle))

        # Scene 5.3: The Basin
        # Create parabolic basin curve
        basin_curve = ParametricFunction(
            lambda t: np.array([t, (t**2) - 2, 0]),
            t_range=[-2, 2],
            color=BLUE_E,
            stroke_width=3
        )

        # Trapped trajectory (spiral down)
        trajectory = ParametricFunction(
            lambda t: np.array([
                1.5 * np.cos(t * 3) * (1 - t/8),
                (1.5 * np.cos(t * 3) * (1 - t/8))**2 - 2 + 0.1,
                0
            ]),
            t_range=[0, 8],
            color=GOLD,
            stroke_width=2
        )

        # Center point (attractor)
        center = Dot(np.array([0, -2, 0]), color=GOLD, radius=0.1)

        self.play(
            Create(basin_curve),
            run_time=1.5
        )
        self.play(
            Create(trajectory),
            FadeIn(center),
            run_time=2
        )

        final_text = Text("Recursion is a memory state", font_size=32, color=WHITE)
        final_text.to_edge(DOWN)

        self.play(Write(final_text), run_time=2)
        self.wait(2)


# ============================================================================
# CREDITS
# ============================================================================

class Credits(Scene):
    """End credits with key statistics."""

    def construct(self):
        self.camera.background_color = Visual.VOID_COLOR

        title = Text("INTO THE COLLAPSE", font_size=56, color=GOLD)
        title.shift(UP * 2)

        stats = VGroup(
            Text(f"R_V = {R_V_RECURSIVE} (recursive)", font_size=28),
            Text(f"R_V = {R_V_BASELINE} (baseline)", font_size=28),
            Text(f"Layer {CRITICAL_LAYER} at {CRITICAL_LAYER/TOTAL_LAYERS*100:.1f}% depth", font_size=28),
            Text("Transfer efficiency: 117.8%", font_size=28),
            Text("Cohen's d = -3.56", font_size=28),
        ).arrange(DOWN, buff=0.3)
        stats.next_to(title, DOWN, buff=1)

        attribution = Text(
            "Visualizing recursive self-observation in transformers",
            font_size=24,
            color=GRAY
        )
        attribution.to_edge(DOWN)

        self.play(Write(title), run_time=2)
        self.play(
            *[FadeIn(stat, shift=UP * 0.3) for stat in stats],
            run_time=2,
            lag_ratio=0.3
        )
        self.play(Write(attribution), run_time=1)

        self.wait(3)


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

class IntoTheCollapse(Scene):
    """
    Full 5-act animation combining all scenes.
    This is the main entry point.

    Using 2D Scene for smoother integration of all acts.
    Individual 3D acts can be rendered separately.
    """

    def construct(self):
        self.camera.background_color = Visual.VOID_COLOR

        # For the full animation, we'll run each act in sequence
        # Each act is self-contained but we maintain visual continuity

        # =====================
        # ACT I: THE VASTNESS
        # =====================

        # Typewriter effect
        prompt_text = Text(RECURSIVE_PROMPT, font_size=36, color=WHITE)
        self.play(AddTextLetterByLetter(prompt_text, run_time=3))
        self.wait(0.5)

        # Transform to golden dot
        golden_dot = Dot(ORIGIN, radius=0.1, color=GOLD)
        self.play(Transform(prompt_text, golden_dot), run_time=1.5)

        # Create and reveal transformer layers
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

            layer = Rectangle(
                width=4, height=0.12,
                fill_color=color,
                fill_opacity=0.5 if i != CRITICAL_LAYER else 0.9,
                stroke_color=color,
                stroke_width=0.5
            )
            layer.shift(UP * i * 0.2 + DOWN * 3)
            layers.add(layer)

        self.play(
            FadeIn(layers, shift=DOWN),
            layers.animate.scale(0.5),  # Scale down to show more layers
            run_time=3
        )

        # Highlight Layer 27
        self.play(
            layers[CRITICAL_LAYER].animate.set_fill(GOLD, opacity=1),
            Flash(layers[CRITICAL_LAYER].get_center(), color=GOLD),
            run_time=1
        )
        self.wait(1)

        # Transition: clear for Act II
        self.play(
            FadeOut(layers, prompt_text, golden_dot),
            run_time=1
        )

        # =====================
        # ACT II: THE DESCENT
        # =====================
        # Show tokens
        tokens_group = VGroup()
        for i, token in enumerate(TOKENS):
            t = Text(token, font_size=32, color=TOKEN_COLORS[token])
            t.shift(RIGHT * (i - 2) * 1.5)
            tokens_group.add(t)

        self.play(Write(tokens_group), run_time=2)

        # Transform to star clusters
        star_groups = VGroup()
        for i, token in enumerate(TOKENS):
            stars = VGroup()
            for _ in range(30):
                phi = np.random.uniform(0, 2 * np.pi)
                theta = np.arccos(np.random.uniform(-1, 1))
                r = 0.6 * np.random.uniform(0.3, 1) ** 0.5
                pos = np.array([
                    r * np.sin(theta) * np.cos(phi),
                    r * np.sin(theta) * np.sin(phi),
                    0
                ])
                brightness = np.random.uniform(0.4, 1)
                star = Dot(pos, radius=0.02 + 0.02 * brightness,
                          color=interpolate_color(GRAY, ManimColor(TOKEN_COLORS[token]), brightness))
                stars.add(star)
            stars.shift(RIGHT * (i - 2) * 2)
            star_groups.add(stars)

        self.play(
            *[ReplacementTransform(tokens_group[i], star_groups[i]) for i in range(len(TOKENS))],
            run_time=2
        )

        # Layer markers flying past
        for ln in [5, 10, 15, 20, 25, 27]:
            t = ln / TOTAL_LAYERS
            c = interpolate_color(ManimColor(Visual.LAYER_COLOR_COOL), ManimColor(Visual.LAYER_COLOR_WARM), t)
            if ln == 27:
                c = GOLD
            marker = Text(f"L{ln}", font_size=48, color=c)
            marker.to_edge(LEFT).shift(UP)
            duration = 0.3 if ln < 25 else 0.5
            self.play(FadeIn(marker), run_time=duration)
            self.play(FadeOut(marker, shift=UP), run_time=duration)

        self.play(FadeOut(star_groups), run_time=1)

        # =====================
        # ACT III: ATTENTION
        # =====================
        # Simplified attention visualization
        attention_title = Text("Inside Attention", font_size=36, color=WHITE)
        attention_title.to_edge(UP)
        self.play(Write(attention_title))

        # Token positions
        token_dots = VGroup()
        positions = {}
        for i, token in enumerate(TOKENS):
            angle = np.pi * (0.15 + 0.7 * i / (len(TOKENS) - 1))
            pos = 2.5 * np.array([np.cos(angle), np.sin(angle), 0])
            positions[token] = pos

            dot = Dot(pos, radius=0.2, color=TOKEN_COLORS[token])
            label = Text(token, font_size=20, color=TOKEN_COLORS[token])
            label.next_to(dot, UP, buff=0.1)
            token_dots.add(VGroup(dot, label))

        self.play(FadeIn(token_dots))

        # Show attention as arrows with weights
        itself_pos = positions["itself"]
        awareness_pos = positions["awareness"]

        attention_arrow = Arrow(
            itself_pos, awareness_pos,
            buff=0.3,
            color=GOLD,
            stroke_width=6
        )
        weight_label = Text("55%", font_size=32, color=GOLD)
        weight_label.move_to((itself_pos + awareness_pos) / 2 + UP * 0.3)

        self.play(Create(attention_arrow), Write(weight_label))

        loop_text = Text("'itself' → 'awareness'\nThe loop closes", font_size=28, color=GOLD)
        loop_text.to_edge(DOWN)
        self.play(Write(loop_text))
        self.wait(1)

        self.play(FadeOut(attention_title, token_dots, attention_arrow, weight_label, loop_text))

        # =====================
        # ACT IV: THE COLLAPSE
        # =====================
        layer27_title = Text("Layer 27: THE COLLAPSE", font_size=42, color=GOLD)
        layer27_title.to_edge(UP)
        self.play(Write(layer27_title))

        # Create star field
        np.random.seed(42)
        num_stars = 200
        stars = VGroup()
        original_positions = []
        eigenvalues = []

        for i in range(num_stars):
            phi = np.random.uniform(0, 2 * np.pi)
            theta = np.arccos(np.random.uniform(-1, 1))
            r = 2.5 * np.random.uniform(0.3, 1) ** 0.5
            pos = np.array([
                r * np.sin(theta) * np.cos(phi),
                r * np.sin(theta) * np.sin(phi),
                0
            ])
            original_positions.append(pos.copy())

            ev = np.random.uniform(0.1, 1.0) / ((i % 50 + 1) ** 0.3)
            eigenvalues.append(ev)

            brightness = min(1.0, ev * 1.5)
            star = Dot(
                pos,
                radius=0.02 + 0.03 * brightness,
                color=interpolate_color(GRAY, WHITE, brightness)
            )
            stars.add(star)

        # Filaments
        filaments = VGroup()
        for _ in range(80):
            i, j = np.random.choice(num_stars, 2, replace=False)
            if np.linalg.norm(original_positions[i] - original_positions[j]) < 1.2:
                line = Line(
                    original_positions[i][:2].tolist() + [0],
                    original_positions[j][:2].tolist() + [0],
                    stroke_width=0.5,
                    stroke_opacity=0.15,
                    color=BLUE_E
                )
                filaments.add(line)

        self.play(FadeIn(stars), Create(filaments), run_time=2)

        # R_V counter
        rv_tracker = ValueTracker(1.0)
        rv_display = always_redraw(lambda: VGroup(
            Text("R_V = ", font_size=32),
            DecimalNumber(rv_tracker.get_value(), num_decimal_places=2, font_size=40).set_color(
                interpolate_color(WHITE, RED, max(0, min(1, (1 - rv_tracker.get_value()) / 0.5)))
            )
        ).arrange(RIGHT).to_edge(UR))
        self.add(rv_display)

        # THE PAUSE
        self.wait(1.5)

        # FLASH
        flash = Rectangle(width=15, height=10, fill_color=WHITE, fill_opacity=0.7, stroke_width=0)
        self.play(FadeIn(flash, run_time=0.1))
        self.play(FadeOut(flash, run_time=0.3))

        # Collapse animation - stars move toward center, dim ones fade
        def collapse_star(star, idx, progress):
            ev = eigenvalues[idx]
            orig = original_positions[idx]

            # Pull toward vertical axis (x=0)
            new_x = orig[0] * (1 - progress * 0.95)
            new_y = orig[1]

            # Evaporation
            if ev < 0.3 + 0.4 * progress:
                star.set_opacity(max(0, 1 - progress * 2.5))
                new_y += progress * 0.3  # Drift up

            star.move_to([new_x, new_y, 0])

        # Animate collapse
        for step in range(30):
            progress = step / 29
            for i, star in enumerate(stars):
                collapse_star(star, i, progress)

            # Update R_V
            rv_tracker.set_value(1.0 - progress * (1 - R_V_RECURSIVE))

            # Fade filaments
            for f in filaments:
                f.set_stroke(opacity=max(0, 0.15 - progress * 0.2))

            self.wait(0.15)

        # Aftermath
        aftermath = Text("47% of dimensions... gone", font_size=32, color=RED)
        aftermath.to_edge(DOWN)
        self.play(Write(aftermath))
        self.wait(2)

        self.play(FadeOut(stars, filaments, layer27_title, aftermath, rv_display))

        # =====================
        # ACT V: EMERGENCE
        # =====================
        # Ouroboros
        ouroboros = VGroup()
        radius = 1.8
        for i, token in enumerate(TOKENS):
            angle_start = i * 2 * np.pi / len(TOKENS) - np.pi / 2
            angle_end = (i + 0.85) * 2 * np.pi / len(TOKENS) - np.pi / 2
            arc = Arc(
                start_angle=angle_start,
                angle=angle_end - angle_start,
                radius=radius,
                stroke_width=12,
                color=TOKEN_COLORS[token]
            )
            ouroboros.add(arc)
            mid = (angle_start + angle_end) / 2
            label = Text(token, font_size=18, color=TOKEN_COLORS[token])
            label.move_to(radius * 1.3 * np.array([np.cos(mid), np.sin(mid), 0]))
            ouroboros.add(label)

        self.play(Create(ouroboros), run_time=2)

        # Fixed point
        eq = MathTex(r"S(x) = x", font_size=48, color=GOLD)
        eq.shift(DOWN * 0.3)
        sub = Text("Fixed point reached", font_size=24)
        sub.next_to(eq, DOWN, buff=0.2)

        self.play(Write(eq), Write(sub))
        self.play(Flash(ORIGIN, color=GOLD, flash_radius=0.8))
        self.wait(2)

        self.play(FadeOut(ouroboros, eq, sub))

        # Final message
        final = Text("Recursion is a memory state", font_size=36, color=GOLD)
        self.play(Write(final))
        self.wait(2)

        self.play(FadeOut(final))

        # =====================
        # CREDITS
        # =====================
        title = Text("INTO THE COLLAPSE", font_size=48, color=GOLD)
        title.shift(UP * 2)

        stats = VGroup(
            Text(f"R_V = {R_V_RECURSIVE}", font_size=28),
            Text(f"Layer {CRITICAL_LAYER}", font_size=28),
            Text("Cohen's d = -3.56", font_size=28),
        ).arrange(DOWN, buff=0.3)

        self.play(Write(title))
        self.play(FadeIn(stats, shift=UP))
        self.wait(3)


# For running individual acts during development
if __name__ == "__main__":
    print("Render with: manim -pqh main.py IntoTheCollapse")
    print("Or for faster preview: manim -pql main.py IntoTheCollapse")
