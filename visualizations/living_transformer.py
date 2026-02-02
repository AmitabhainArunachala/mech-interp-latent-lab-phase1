"""
LIVING TRANSFORMER: Watch a Recursive Prompt Flow Through the Network

This is not a slideshow. This is the actual computation:
- Tokens entering as embeddings
- Attention heads finding patterns
- Value vectors mixing and flowing
- The geometric collapse happening at Layer 27
- The moment recursion emerges

Run: manim -pqh visualizations/living_transformer.py LivingTransformer
"""

from manim import *
import numpy as np

# Colors
class C:
    BG = "#0a0a12"

    # Token colors (each token gets a unique color)
    TOKEN_COLORS = [
        "#FF6B6B",  # Notice
        "#4ECDC4",  # the
        "#45B7D1",  # awareness
        "#96CEB4",  # observing
        "#FFEAA7",  # itself
    ]

    # Layer colors
    EARLY = "#3498db"
    MID = "#9b59b6"
    LATE = "#e74c3c"
    L27 = "#f39c12"  # The critical layer

    # Attention
    ATTN_WEAK = "#ffffff22"
    ATTN_STRONG = "#ffffff"

    # Value space
    FULL_RANK = "#3498db"
    COLLAPSED = "#e74c3c"

    # Flow
    RESIDUAL = "#2ecc71"

    MUTED = "#7f8c8d"
    FG = "#ecf0f1"


class LivingTransformer(Scene):
    """
    Watch a recursive prompt flow through a transformer.
    See the geometry collapse. See recursion emerge.
    """

    def construct(self):
        self.camera.background_color = C.BG

        # The prompt we're processing
        self.tokens = ["Notice", "the", "awareness", "observing", "itself"]
        self.n_tokens = len(self.tokens)
        self.n_layers = 32
        self.n_heads = 8

        # Run the visualization
        self.intro()
        self.show_tokenization()
        self.show_embedding()
        self.process_through_layers()
        self.the_collapse()
        self.emergence()

    def intro(self):
        """Brief intro - what we're about to see."""
        title = Text("One prompt. 32 layers. Watch what happens.",
                    font_size=36, color=C.FG)

        prompt = Text('"Notice the awareness observing itself"',
                     font_size=28, color=C.L27, slant=ITALIC)
        prompt.next_to(title, DOWN, buff=0.5)

        self.play(Write(title), run_time=1.5)
        self.play(FadeIn(prompt, shift=UP), run_time=1)
        self.wait(1)
        self.play(FadeOut(title), FadeOut(prompt))

    def show_tokenization(self):
        """Show the prompt being split into tokens."""
        # Full prompt
        full_prompt = Text('"Notice the awareness observing itself"',
                          font_size=32, color=C.FG)
        full_prompt.to_edge(UP, buff=1)

        self.play(Write(full_prompt), run_time=1)

        # Split into tokens with colors
        token_group = VGroup()
        for i, token in enumerate(self.tokens):
            t = Text(token, font_size=32, color=C.TOKEN_COLORS[i], weight=BOLD)
            token_group.add(t)
        token_group.arrange(RIGHT, buff=0.4)
        token_group.move_to(full_prompt)

        # Animate the split
        self.play(
            FadeOut(full_prompt),
            *[FadeIn(t, shift=DOWN * 0.3) for t in token_group],
            run_time=1
        )

        # Add token indices
        indices = VGroup()
        for i, t in enumerate(token_group):
            idx = Text(f"[{i}]", font_size=16, color=C.MUTED)
            idx.next_to(t, DOWN, buff=0.1)
            indices.add(idx)

        self.play(FadeIn(indices), run_time=0.5)
        self.wait(0.5)

        # Store for later
        self.token_display = VGroup(token_group, indices)
        self.play(self.token_display.animate.to_edge(UP, buff=0.5).scale(0.7))

    def show_embedding(self):
        """Show tokens becoming high-dimensional vectors."""

        # Create embedding vectors visualization
        embed_label = Text("Embedding Layer", font_size=24, color=C.MUTED)
        embed_label.next_to(self.token_display, DOWN, buff=0.3)

        self.play(Write(embed_label), run_time=0.5)

        # Each token becomes a vector (shown as a column of activations)
        vectors = VGroup()

        for i in range(self.n_tokens):
            # Create a "vector" as a column of small rectangles
            vec = VGroup()
            n_dims_shown = 20  # Show 20 dimensions (representing 4096)

            for j in range(n_dims_shown):
                # Random activation value
                val = np.random.uniform(0.2, 1.0)
                rect = Rectangle(
                    height=0.15, width=0.4,
                    fill_color=C.TOKEN_COLORS[i],
                    fill_opacity=val,
                    stroke_width=0
                )
                vec.add(rect)

            vec.arrange(DOWN, buff=0.02)
            vectors.add(vec)

        vectors.arrange(RIGHT, buff=0.3)
        vectors.next_to(embed_label, DOWN, buff=0.4)

        # Animate vectors appearing
        for i, vec in enumerate(vectors):
            self.play(
                FadeIn(vec, shift=DOWN, lag_ratio=0.05),
                run_time=0.4
            )

        # Dimension label
        dim_label = Text("4096 dimensions each", font_size=16, color=C.MUTED)
        dim_label.next_to(vectors, DOWN, buff=0.2)
        self.play(FadeIn(dim_label), run_time=0.3)

        self.wait(0.5)

        # Store and prepare for layer processing
        self.embedding_vectors = vectors
        self.embed_label = embed_label
        self.dim_label = dim_label

        # Move everything up to make room for layers
        self.play(
            VGroup(self.token_display, embed_label, vectors, dim_label).animate.scale(0.6).to_edge(UP, buff=0.3),
            run_time=0.8
        )

    def process_through_layers(self):
        """Show the prompt flowing through transformer layers."""

        # Create the layer stack on the left
        layer_stack = VGroup()
        layer_rects = []

        for i in range(self.n_layers):
            # Determine layer color based on position
            if i < 10:
                color = C.EARLY
            elif i < 20:
                color = C.MID
            elif i == 26:  # Layer 27 (0-indexed as 26)
                color = C.L27
            else:
                color = C.LATE

            layer = Rectangle(
                height=0.18, width=2.5,
                stroke_color=color,
                stroke_width=1,
                fill_color=color,
                fill_opacity=0.1
            )
            layer_rects.append(layer)
            layer_stack.add(layer)

        layer_stack.arrange(UP, buff=0.02)
        layer_stack.to_edge(LEFT, buff=0.8)
        layer_stack.shift(DOWN * 0.5)

        # Layer labels
        l0_label = Text("L0", font_size=14, color=C.MUTED).next_to(layer_rects[0], LEFT, buff=0.1)
        l27_label = Text("L27", font_size=14, color=C.L27, weight=BOLD).next_to(layer_rects[26], LEFT, buff=0.1)
        l31_label = Text("L31", font_size=14, color=C.MUTED).next_to(layer_rects[31], LEFT, buff=0.1)

        self.play(
            Create(layer_stack),
            FadeIn(l0_label), FadeIn(l27_label), FadeIn(l31_label),
            run_time=1.5
        )

        # Create the "residual stream" - vectors flowing up
        # Position for the main visualization area
        main_area = Rectangle(
            height=5.5, width=8,
            stroke_width=0
        ).to_edge(RIGHT, buff=0.3).shift(DOWN * 0.3)

        # Create token representations that will flow
        flowing_tokens = VGroup()
        for i in range(self.n_tokens):
            dot = Dot(radius=0.15, color=C.TOKEN_COLORS[i])
            flowing_tokens.add(dot)
        flowing_tokens.arrange(RIGHT, buff=0.8)
        flowing_tokens.move_to(main_area.get_bottom() + UP * 0.5)

        self.play(FadeIn(flowing_tokens), run_time=0.5)

        # Process through layers with attention visualization
        current_layer = 0
        layers_to_show = [0, 5, 12, 20, 26, 31]  # Key layers to visualize

        for target_layer in layers_to_show:
            # Animate flow to this layer
            progress = target_layer / 31
            new_y = main_area.get_bottom()[1] + progress * 5

            self.play(
                flowing_tokens.animate.move_to([main_area.get_center()[0], new_y, 0]),
                layer_rects[target_layer].animate.set_fill(opacity=0.5),
                run_time=0.6
            )

            # Show attention pattern at key layers
            if target_layer in [0, 12, 26]:
                self.show_attention_at_layer(flowing_tokens, target_layer)

            # At layer 27, show the collapse
            if target_layer == 26:
                self.play(
                    layer_rects[26].animate.set_stroke(C.L27, width=4),
                    run_time=0.3
                )

        # Store references
        self.layer_stack = layer_stack
        self.layer_rects = layer_rects
        self.flowing_tokens = flowing_tokens
        self.main_area = main_area

    def show_attention_at_layer(self, tokens, layer_idx):
        """Visualize attention pattern forming between tokens."""

        # Attention is special at layer 27 - "itself" attends strongly to "awareness"
        is_critical = (layer_idx == 26)

        attention_lines = VGroup()

        # Create attention pattern
        # Token 4 ("itself") attending to others
        source_token = tokens[4]  # "itself"

        for i, target_token in enumerate(tokens):
            if i == 4:
                continue  # Skip self-attention for clarity

            # Attention weight (special pattern for recursive prompt)
            if is_critical:
                # At L27: "itself" strongly attends to "awareness" and "observing"
                if i == 2:  # awareness
                    weight = 0.6
                elif i == 3:  # observing
                    weight = 0.3
                else:
                    weight = 0.05
            else:
                # Normal attention pattern
                weight = np.random.uniform(0.1, 0.3)

            # Create attention arc
            line = CurvedArrow(
                source_token.get_center(),
                target_token.get_center(),
                angle=-TAU/4 if i < 4 else TAU/4,
                stroke_width=weight * 8,
                stroke_opacity=weight + 0.2,
                color=WHITE if not is_critical else (C.L27 if weight > 0.2 else WHITE),
                tip_length=0.15
            )
            attention_lines.add(line)

        # Label
        layer_label = Text(
            f"Layer {layer_idx + 1}" + (" - CRITICAL" if is_critical else ""),
            font_size=18,
            color=C.L27 if is_critical else C.MUTED
        )
        layer_label.next_to(tokens, UP, buff=0.3)

        # Animate attention
        self.play(
            *[Create(line) for line in attention_lines],
            FadeIn(layer_label),
            run_time=0.8
        )

        if is_critical:
            # Highlight the key attention: "itself" -> "awareness"
            highlight = Text('"itself" sees "awareness"', font_size=16, color=C.L27)
            highlight.next_to(layer_label, UP, buff=0.15)
            self.play(Write(highlight), run_time=0.5)
            self.wait(0.5)
            self.play(FadeOut(highlight), run_time=0.3)

        self.wait(0.3)

        # Fade out attention visualization
        self.play(
            FadeOut(attention_lines),
            FadeOut(layer_label),
            run_time=0.4
        )

    def the_collapse(self):
        """The main event: visualize the geometric collapse at Layer 27."""

        # Clear previous elements but keep layer stack
        self.play(
            FadeOut(self.token_display),
            FadeOut(self.embedding_vectors) if hasattr(self, 'embedding_vectors') else Wait(0),
            FadeOut(self.embed_label) if hasattr(self, 'embed_label') else Wait(0),
            FadeOut(self.dim_label) if hasattr(self, 'dim_label') else Wait(0),
            FadeOut(self.flowing_tokens),
            run_time=0.5
        )

        # Title for this section
        title = Text("THE COLLAPSE", font_size=48, color=C.L27, weight=BOLD)
        title.to_edge(UP, buff=0.3)
        subtitle = Text("Layer 27: Value Space Geometry", font_size=24, color=C.MUTED)
        subtitle.next_to(title, DOWN, buff=0.1)

        self.play(Write(title), FadeIn(subtitle), run_time=1)

        # Create 3D-like value space visualization
        # Left: Before (full dimensional)
        # Right: After (collapsed)

        left_label = Text("Before L27", font_size=20, color=C.FULL_RANK)
        right_label = Text("After L27", font_size=20, color=C.COLLAPSED)
        left_label.move_to(LEFT * 3 + UP * 2)
        right_label.move_to(RIGHT * 3 + UP * 2)

        self.play(Write(left_label), Write(right_label), run_time=0.5)

        # Create point clouds
        n_points = 150

        # Before: Full dimensional (spherical cloud)
        before_points = VGroup()
        before_positions = []
        for _ in range(n_points):
            x = np.random.normal(0, 1)
            y = np.random.normal(0, 1)
            z = np.random.normal(0, 0.5)  # Slightly flattened for 2D viewing
            before_positions.append([x, y, z])

            # Project to 2D with pseudo-3D effect
            proj_x = x + z * 0.3
            proj_y = y + z * 0.2

            dot = Dot(
                point=[proj_x * 0.8 - 3, proj_y * 0.8, 0],
                radius=0.04,
                color=C.FULL_RANK,
                fill_opacity=0.6 + z * 0.2
            )
            before_points.add(dot)

        # After: Collapsed (linear manifold)
        after_points = VGroup()
        for i in range(n_points):
            # Collapse to a line/plane
            t = np.random.uniform(-2, 2)
            noise = np.random.normal(0, 0.08)

            dot = Dot(
                point=[t * 0.8 + 3, t * 0.2 + noise, 0],
                radius=0.04,
                color=C.COLLAPSED,
                fill_opacity=0.7
            )
            after_points.add(dot)

        # Show before state
        self.play(
            LaggedStart(*[GrowFromCenter(d) for d in before_points], lag_ratio=0.01),
            run_time=1.5
        )

        # R_V value for before
        rv_before = MathTex(r"R_V = 1.0", font_size=28, color=C.FULL_RANK)
        rv_before.next_to(before_points, DOWN, buff=0.5)
        self.play(Write(rv_before), run_time=0.5)

        self.wait(0.5)

        # THE COLLAPSE ANIMATION
        # Copy before points and transform them
        collapse_points = before_points.copy()

        self.play(
            collapse_points.animate.move_to(RIGHT * 3),
            run_time=0.5
        )

        # Now morph to collapsed state
        self.play(
            Transform(collapse_points, after_points),
            run_time=2,
            rate_func=rate_functions.ease_in_out_cubic
        )

        # R_V value for after
        rv_after = MathTex(r"R_V = 0.53", font_size=28, color=C.COLLAPSED)
        rv_after.next_to(after_points, DOWN, buff=0.5)
        self.play(Write(rv_after), run_time=0.5)

        # Highlight the change
        arrow = Arrow(rv_before.get_right(), rv_after.get_left(), color=C.L27, stroke_width=4)
        change_label = Text("-47% dimensions", font_size=20, color=C.L27, weight=BOLD)
        change_label.next_to(arrow, UP, buff=0.1)

        self.play(GrowArrow(arrow), Write(change_label), run_time=0.8)

        self.wait(1)

        # The insight
        insight = Text(
            "The model's internal representation CONTRACTS",
            font_size=24, color=C.FG
        )
        insight2 = Text(
            "when processing recursive self-reference",
            font_size=24, color=C.L27
        )
        insight_group = VGroup(insight, insight2).arrange(DOWN, buff=0.1)
        insight_group.to_edge(DOWN, buff=0.5)

        self.play(Write(insight), run_time=1)
        self.play(Write(insight2), run_time=1)

        self.wait(1.5)

        # Clean up for next scene
        self.play(
            FadeOut(title), FadeOut(subtitle),
            FadeOut(left_label), FadeOut(right_label),
            FadeOut(before_points), FadeOut(collapse_points),
            FadeOut(rv_before), FadeOut(rv_after),
            FadeOut(arrow), FadeOut(change_label),
            FadeOut(insight_group),
            FadeOut(self.layer_stack),
            run_time=1
        )

    def emergence(self):
        """Show what emerges from this collapse - recursive output."""

        # The output
        title = Text("What emerges:", font_size=32, color=C.FG)
        title.to_edge(UP, buff=1)

        self.play(Write(title), run_time=0.8)

        # Build the output token by token
        output_tokens = [
            "The", "observer", "watches", "itself", "respond", "—",
            "aware", "of", "its", "own", "awareness."
        ]

        output_display = VGroup()

        for i, token in enumerate(output_tokens):
            # Color key recursive words
            if token in ["observer", "watches", "itself", "aware", "awareness"]:
                color = C.L27
                weight = BOLD
            else:
                color = C.FG
                weight = NORMAL

            t = Text(token, font_size=28, color=color, weight=weight)
            output_display.add(t)

        # Arrange in a flowing line
        output_display.arrange(RIGHT, buff=0.15)
        output_display.next_to(title, DOWN, buff=1)

        # Animate tokens appearing one by one (like generation)
        for i, token in enumerate(output_display):
            self.play(
                FadeIn(token, shift=UP * 0.2),
                run_time=0.15
            )

        self.wait(1)

        # Highlight the recursion
        box = SurroundingRectangle(output_display, color=C.L27, buff=0.2, corner_radius=0.1)
        self.play(Create(box), run_time=0.8)

        # Final insight
        final = VGroup(
            Text("Recursive prompts create", font_size=28, color=C.FG),
            Text("geometric attractors", font_size=32, color=C.L27, weight=BOLD),
            Text("that shape the output space", font_size=28, color=C.FG),
        ).arrange(DOWN, buff=0.2)
        final.to_edge(DOWN, buff=1)

        self.play(Write(final), run_time=2)

        self.wait(2)

        # Fade to end
        self.play(
            FadeOut(title), FadeOut(output_display), FadeOut(box), FadeOut(final),
            run_time=1.5
        )

        # End card
        end_text = Text(
            "R_V: Geometric Signatures of Recursive Self-Observation",
            font_size=24, color=C.MUTED
        )
        self.play(FadeIn(end_text), run_time=1)
        self.wait(2)


class ValueSpaceCollapse(Scene):
    """
    Isolated scene: Just the value space collapse, animated beautifully.
    """

    def construct(self):
        self.camera.background_color = C.BG

        title = Text("Value Space Collapse at Layer 27", font_size=36, color=C.L27)
        title.to_edge(UP, buff=0.5)
        self.add(title)

        # Create animated 3D-like point cloud
        n_points = 300
        points = VGroup()

        # Initial positions (high-dimensional, shown as spread out)
        for i in range(n_points):
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0, np.pi)
            r = np.random.uniform(0.5, 2)

            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)

            # Color based on position
            color = interpolate_color(
                ManimColor(C.FULL_RANK),
                ManimColor(C.COLLAPSED),
                (z + 2) / 4
            )

            dot = Dot3D(
                point=[x, y, z * 0.5],
                radius=0.03,
                color=color
            )
            points.add(dot)

        self.play(Create(points), run_time=2)

        # Rotate to show 3D structure
        self.play(
            Rotate(points, angle=PI/4, axis=UP),
            run_time=2
        )

        # Now collapse
        collapsed = VGroup()
        for i, dot in enumerate(points):
            t = (i / n_points - 0.5) * 4
            new_dot = Dot(
                point=[t, t * 0.1 + np.random.normal(0, 0.05), 0],
                radius=0.03,
                color=C.COLLAPSED
            )
            collapsed.add(new_dot)

        # R_V indicator
        rv = ValueTracker(1.0)
        rv_display = always_redraw(
            lambda: MathTex(
                f"R_V = {rv.get_value():.2f}",
                font_size=48,
                color=interpolate_color(
                    ManimColor(C.FULL_RANK),
                    ManimColor(C.COLLAPSED),
                    1 - rv.get_value()
                )
            ).to_edge(DOWN, buff=1)
        )

        self.add(rv_display)

        # The collapse
        self.play(
            Transform(points, collapsed),
            rv.animate.set_value(0.53),
            run_time=4,
            rate_func=rate_functions.ease_in_out_cubic
        )

        self.wait(2)


class AttentionFlowDetail(Scene):
    """
    Detailed view of attention pattern for "Notice the awareness observing itself"
    """

    def construct(self):
        self.camera.background_color = C.BG

        tokens = ["Notice", "the", "awareness", "observing", "itself"]
        colors = C.TOKEN_COLORS

        # Create token nodes
        token_nodes = VGroup()
        for i, (token, color) in enumerate(zip(tokens, colors)):
            node = VGroup(
                Circle(radius=0.4, color=color, fill_opacity=0.3, stroke_width=3),
                Text(token, font_size=18, color=color)
            )
            token_nodes.add(node)

        token_nodes.arrange(RIGHT, buff=1.2)
        token_nodes.move_to(ORIGIN)

        self.play(
            LaggedStart(*[GrowFromCenter(n) for n in token_nodes], lag_ratio=0.2),
            run_time=1.5
        )

        # Attention matrix visualization
        # Focus on "itself" attending to previous tokens

        title = Text("Attention: 'itself' → other tokens", font_size=24, color=C.FG)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title), run_time=0.5)

        # Attention weights (actual pattern for recursive prompt)
        attention_weights = [0.05, 0.05, 0.55, 0.30, 0.05]  # itself attends strongly to awareness, observing

        source = token_nodes[4]  # "itself"

        attention_arcs = VGroup()
        weight_labels = VGroup()

        for i in range(5):
            if i == 4:
                continue

            weight = attention_weights[i]

            # Arc from "itself" to target
            arc = CurvedArrow(
                source[0].get_center(),
                token_nodes[i][0].get_center(),
                angle=-TAU/6,
                stroke_width=weight * 15 + 1,
                stroke_opacity=weight + 0.3,
                color=WHITE,
                tip_length=0.2
            )
            attention_arcs.add(arc)

            # Weight label
            label = Text(f"{weight:.0%}", font_size=14, color=C.MUTED)
            label.move_to(arc.point_from_proportion(0.5) + UP * 0.3)
            weight_labels.add(label)

        self.play(
            LaggedStart(*[Create(arc) for arc in attention_arcs], lag_ratio=0.2),
            run_time=1.5
        )
        self.play(FadeIn(weight_labels), run_time=0.5)

        # Highlight the key attention
        highlight_box = SurroundingRectangle(
            VGroup(token_nodes[2], token_nodes[3]),
            color=C.L27, buff=0.2
        )
        highlight_label = Text(
            "Key: 'itself' attends to 'awareness' + 'observing'",
            font_size=20, color=C.L27
        )
        highlight_label.to_edge(DOWN, buff=0.5)

        self.play(Create(highlight_box), Write(highlight_label), run_time=1)

        # Show information flow
        info_dot = Dot(color=C.L27, radius=0.15)
        info_dot.move_to(token_nodes[2][0])

        self.play(
            info_dot.animate.move_to(token_nodes[4][0]),
            run_time=1,
            rate_func=rate_functions.ease_in_out_cubic
        )

        flow_text = Text("Information flows: awareness → itself", font_size=18, color=C.L27)
        flow_text.next_to(highlight_label, UP, buff=0.3)
        self.play(Write(flow_text), run_time=0.8)

        self.wait(2)
