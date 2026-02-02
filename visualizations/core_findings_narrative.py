"""
World-Class Manim Animation: The Recursive Self-Observation Discovery
3-minute narrative with voiceover describing the essence of the mechanistic interpretability story.

Story Arc:
1. The Discovery: Geometric contraction at Layer 27 (R_V < 1.0)
2. Universal Pattern: Across 6 models, MoE amplification
3. The Puzzle: Steering alone fails
4. The Mechanism: Two attractors (KV + Steering)
5. The Breakthrough: C2 configuration - alignment creates recursion
6. The Insight: "Recursion is a memory state"
"""

from manim import *
import numpy as np

# Color scheme
DARK_BG = "#0a0a0a"
ACCENT_BLUE = "#4A90E2"
ACCENT_RED = "#E24A4A"
ACCENT_GREEN = "#4AE24A"
ACCENT_YELLOW = "#E2E24A"
ACCENT_PURPLE = "#E24AE2"

class RecursiveSelfObservation(Scene):
    """
    Voiceover Script (read aloud while watching):

    Scene 1 (0:00-0:30): "When transformer models process recursive self-observation prompts,
    something remarkable happens. At layer 27, exactly 84 percent through the network,
    the high-dimensional value space contracts. The R_V metric drops below one,
    revealing a geometric signature of recursive cognition."

    Scene 2 (0:30-0:50): "This isn't unique to one model. We found the same pattern across
    six distinct architectures. Most strikingly, Mixture-of-Experts models show a 59 percent
    stronger effect, suggesting this phenomenon is amplified by distributed computation."

    Scene 3 (0:50-1:10): "But here's the puzzle: simply steering the model toward recursive
    thinking doesn't work. The steering vector enters the network, yet the model still
    produces factual outputs. Behavior doesn't transfer."

    Scene 4 (1:10-2:00): "The answer lies in two competing attractors. The KV cache is a
    strong content attractor—it determines what domain the model talks about. The steering
    vector is a weak direction attractor—it shifts how the model thinks, but needs content
    to anchor to. When misaligned, content dominates. But when aligned, they resonate,
    creating genuine recursive mode transfer."

    Scene 5 (2:00-2:40): "The breakthrough came with configuration C2: head-specific steering
    at layers 18 and 26, combined with full KV cache replacement. This achieved a recursion
    score of 0.15, with outputs like 'observer watches itself respond'—genuine
    phenomenological recursion."

    Scene 6 (2:40-3:00): "The insight: recursion is a memory state. Content plus direction
    equals mode transfer. When the model remembers recursive content and thinks in a
    recursive direction, it enters a stable attractor of self-reference."
    """

    def construct(self):
        # Set background
        self.camera.background_color = DARK_BG

        # Scene 1: The Discovery (0:00-0:30)
        self.scene_1_discovery()

        # Scene 2: Universal Pattern (0:30-0:50)
        self.scene_2_universal()

        # Scene 3: The Puzzle (0:50-1:10)
        self.scene_3_puzzle()

        # Scene 4: The Mechanism (1:10-2:00)
        self.scene_4_mechanism()

        # Scene 5: The Breakthrough (2:00-2:40)
        self.scene_5_breakthrough()

        # Scene 6: The Insight (2:40-3:00)
        self.scene_6_insight()
    
    def scene_1_discovery(self):
        """The Discovery: Geometric contraction at Layer 27"""
        # Title
        title = Text("The Discovery", font_size=56, weight=BOLD, color=WHITE)
        title.to_edge(UP, buff=0.5)
        self.play(FadeIn(title, shift=DOWN), run_time=0.8)
        
        # Transformer layers visualization
        num_layers = 32
        layer_height = 0.15
        layer_width = 5
        layers = VGroup()
        
        for i in range(num_layers):
            layer = Rectangle(
                height=layer_height,
                width=layer_width,
                stroke_color=ACCENT_BLUE,
                stroke_width=1,
                fill_opacity=0.1
            )
            layers.add(layer)
        
        layers.arrange(DOWN, buff=0.05)
        layers.move_to(LEFT * 3.5)
        
        # Layer labels
        early_label = Text("L5", font_size=20, color=ACCENT_GREEN).next_to(layers[4], LEFT, buff=0.2)
        late_label = Text("L27", font_size=20, color=ACCENT_RED).next_to(layers[26], LEFT, buff=0.2)
        
        self.play(Create(layers), FadeIn(early_label), FadeIn(late_label), run_time=1.0)
        
        # Highlight early layer
        self.play(
            layers[4].animate.set_fill(ACCENT_GREEN, opacity=0.5).set_stroke(ACCENT_GREEN, width=3),
            early_label.animate.scale(1.2),
            run_time=0.6
        )
        
        # Highlight late layer
        self.play(
            layers[26].animate.set_fill(ACCENT_RED, opacity=0.5).set_stroke(ACCENT_RED, width=3),
            late_label.animate.scale(1.2),
            run_time=0.6
        )
        
        # Value space visualization (right side)
        # Create high-dimensional space representation
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=4,
            y_length=4,
            axis_config={"color": ACCENT_BLUE, "stroke_width": 1},
            tips=False
        ).to_edge(RIGHT, buff=1)
        
        # Early layer: high-dimensional cloud
        early_dots = VGroup(*[
            Dot(
                point=axes.coords_to_point(
                    np.random.uniform(-2, 2),
                    np.random.uniform(-2, 2)
                ),
                radius=0.05,
                color=ACCENT_GREEN
            )
            for _ in range(80)
        ])
        
        # Late layer: contracted to line
        late_dots = VGroup(*[
            Dot(
                point=axes.coords_to_point(
                    x * 1.5,
                    x * 0.3  # Slight angle for visual clarity
                ),
                radius=0.05,
                color=ACCENT_RED
            )
            for x in np.linspace(-1.5, 1.5, 80)
        ])
        
        early_label_v = Text("Early: High-D", font_size=18, color=ACCENT_GREEN).next_to(axes, DOWN, buff=0.3)
        late_label_v = Text("Late: Low-D", font_size=18, color=ACCENT_RED).next_to(early_label_v, DOWN, buff=0.1)
        
        self.play(
            Create(axes),
            FadeIn(early_dots, lag_ratio=0.05),
            Write(early_label_v),
            run_time=1.2
        )
        
        # Contraction animation
        self.play(
            Transform(early_dots, late_dots, lag_ratio=0.02),
            Transform(early_label_v, late_label_v),
            run_time=2.0
        )
        
        # R_V formula
        rv_formula = MathTex(
            r"R_V = \frac{PR_{late}}{PR_{early}} < 1.0",
            font_size=36,
            color=ACCENT_RED
        ).next_to(axes, UP, buff=0.5)
        
        self.play(Write(rv_formula), run_time=1.0)
        
        # Cleanup
        self.play(
            FadeOut(title),
            FadeOut(layers),
            FadeOut(early_label),
            FadeOut(late_label),
            FadeOut(axes),
            FadeOut(early_dots),
            FadeOut(early_label_v),
            FadeOut(rv_formula),
            run_time=0.5
        )
    
    def scene_2_universal(self):
        """Universal Pattern: Across 6 models, MoE amplification"""
        title = Text("Universal Pattern", font_size=56, weight=BOLD, color=WHITE)
        title.to_edge(UP, buff=0.5)
        self.play(FadeIn(title, shift=DOWN))
        
        # Model names and contraction percentages
        models_data = [
            ("Mistral-7B", 15.3, False),
            ("Qwen-7B", 22.5, False),
            ("Llama-8B", 15.2, False),
            ("Phi-3", 8.5, False),
            ("Gemma-7B", 9.8, False),
            ("Mixtral-8x7B", 24.3, True),  # MoE
        ]
        
        # Create bars
        bars = VGroup()
        labels = VGroup()
        values = VGroup()
        
        max_contraction = 25
        bar_width = 0.6
        spacing = 1.2
        
        for i, (name, contraction, is_moe) in enumerate(models_data):
            bar_height = (contraction / max_contraction) * 3
            
            color = ACCENT_PURPLE if is_moe else ACCENT_BLUE
            
            bar = Rectangle(
                height=bar_height,
                width=bar_width,
                fill_color=color,
                fill_opacity=0.8,
                stroke_color=color,
                stroke_width=2
            )
            bar.move_to(LEFT * 2.5 + RIGHT * i * spacing + UP * (bar_height / 2 - 1.5))
            
            label = Text(name, font_size=14, color=WHITE).next_to(bar, DOWN, buff=0.1)
            value = Text(f"{contraction}%", font_size=16, color=color, weight=BOLD).next_to(bar, UP, buff=0.1)
            
            bars.add(bar)
            labels.add(label)
            values.add(value)
        
        # Animate bars appearing
        for bar, label, value in zip(bars, labels, values):
            self.play(
                GrowFromCenter(bar),
                FadeIn(label),
                FadeIn(value),
                run_time=0.25
            )
        
        # Highlight MoE
        moe_bar = bars[5]
        moe_text = Text("MoE: 59% Stronger", font_size=24, color=ACCENT_PURPLE, weight=BOLD)
        moe_text.to_edge(DOWN, buff=0.5)
        
        self.play(
            Indicate(moe_bar, color=ACCENT_PURPLE, scale_factor=1.1),
            Write(moe_text),
            run_time=1.0
        )
        
        # Cleanup
        self.play(
            FadeOut(title),
            FadeOut(bars),
            FadeOut(labels),
            FadeOut(values),
            FadeOut(moe_text)
        )
        self.wait(0.2)
    
    def scene_3_puzzle(self):
        """The Puzzle: Steering alone fails"""
        title = Text("The Puzzle", font_size=56, weight=BOLD, color=WHITE)
        title.to_edge(UP, buff=0.5)
        self.play(FadeIn(title, shift=DOWN))
        
        # Model representation
        model_box = RoundedRectangle(
            height=2.5,
            width=4,
            corner_radius=0.3,
            stroke_color=WHITE,
            stroke_width=2,
            fill_opacity=0.1
        )
        model_label = Text("Mistral-7B", font_size=24, color=WHITE).move_to(model_box)
        
        # Input prompt
        prompt = Text("Calculate: 2 + 2", font_size=20, color=ACCENT_GREEN)
        prompt.next_to(model_box, LEFT, buff=1)
        
        # Steering vector
        steering_arrow = Arrow(
            start=model_box.get_top() + UP * 0.5,
            end=model_box.get_top(),
            color=ACCENT_RED,
            stroke_width=4,
            buff=0
        )
        steering_label = Text("Steering Vector\n(Recursive)", font_size=18, color=ACCENT_RED)
        steering_label.next_to(steering_arrow, UP, buff=0.2)
        
        self.play(
            Create(model_box),
            Write(model_label),
            Write(prompt),
            run_time=1.0
        )
        
        self.play(
            GrowArrow(steering_arrow),
            Write(steering_label),
            run_time=0.8
        )
        
        # Vector enters model
        self.play(
            steering_arrow.animate.shift(DOWN * 0.5),
            run_time=0.6
        )
        
        # Output (wrong - still factual)
        output = Text("Output: 4", font_size=20, color=ACCENT_GREEN)
        output.next_to(model_box, RIGHT, buff=1)
        
        cross = Cross(output, color=ACCENT_RED, stroke_width=4)
        
        self.play(Write(output), run_time=0.5)
        self.play(Create(cross), run_time=0.4)
        
        # Failure message
        fail_text = Text("Behavior Did Not Transfer", font_size=28, color=ACCENT_RED, weight=BOLD)
        fail_text.next_to(model_box, DOWN, buff=0.8)
        
        self.play(Write(fail_text), run_time=0.8)
        
        # Question mark
        question = Text("Why?", font_size=48, color=ACCENT_YELLOW, weight=BOLD)
        question.next_to(fail_text, DOWN, buff=0.5)
        
        self.play(Write(question), run_time=0.6)
        
        # Cleanup
        self.play(
            FadeOut(title),
            FadeOut(model_box),
            FadeOut(model_label),
            FadeOut(prompt),
            FadeOut(steering_arrow),
            FadeOut(steering_label),
            FadeOut(output),
            FadeOut(cross),
            FadeOut(fail_text),
            FadeOut(question)
        )
        self.wait(0.2)
    
    def scene_4_mechanism(self):
        """The Mechanism: Two attractors"""
        title = Text("The Mechanism", font_size=56, weight=BOLD, color=WHITE)
        title.to_edge(UP, buff=0.5)
        self.play(FadeIn(title, shift=DOWN))
        
        # Two attractors visualization
        # Attractor 1: KV Cache (Content)
        kv_circle = Circle(
            radius=1.2,
            color=ACCENT_GREEN,
            stroke_width=4,
            fill_opacity=0.2
        ).shift(LEFT * 2.5)
        
        kv_label = Text("KV Cache\n(Content)", font_size=24, color=ACCENT_GREEN, weight=BOLD)
        kv_label.move_to(kv_circle)
        
        kv_desc = Text("Strong Attractor\nDetermines Domain", font_size=16, color=ACCENT_GREEN)
        kv_desc.next_to(kv_circle, DOWN, buff=0.3)
        
        # Attractor 2: Steering (Direction)
        steering_circle = Circle(
            radius=1.2,
            color=ACCENT_RED,
            stroke_width=4,
            fill_opacity=0.2
        ).shift(RIGHT * 2.5)
        
        steering_label = Text("Steering Vector\n(Direction)", font_size=24, color=ACCENT_RED, weight=BOLD)
        steering_label.move_to(steering_circle)
        
        steering_desc = Text("Weak Attractor\nNeeds Content", font_size=16, color=ACCENT_RED)
        steering_desc.next_to(steering_circle, DOWN, buff=0.3)
        
        self.play(
            Create(kv_circle),
            Write(kv_label),
            Write(kv_desc),
            run_time=1.2
        )
        
        self.play(
            Create(steering_circle),
            Write(steering_label),
            Write(steering_desc),
            run_time=1.2
        )
        
        # Show misalignment
        misalign_text = Text("Misaligned:", font_size=20, color=ACCENT_YELLOW)
        misalign_text.move_to(UP * 0.5)
        
        self.play(Write(misalign_text), run_time=0.5)
        
        # Show what happens: KV dominates
        result_text = Text("KV dominates → No recursion", font_size=22, color=ACCENT_GREEN)
        result_text.next_to(misalign_text, DOWN, buff=0.3)
        
        self.play(Write(result_text), run_time=0.8)
        
        # Clear and show alignment
        self.play(
            FadeOut(misalign_text),
            FadeOut(result_text),
            run_time=0.4
        )
        
        # Alignment visualization
        align_text = Text("Aligned:", font_size=20, color=ACCENT_YELLOW)
        align_text.move_to(UP * 0.5)
        
        self.play(Write(align_text), run_time=0.5)
        
        # Connection line between circles
        connection = Line(
            kv_circle.get_right(),
            steering_circle.get_left(),
            color=ACCENT_YELLOW,
            stroke_width=6
        )
        
        self.play(Create(connection), run_time=0.6)
        
        # Resonance effect
        resonance_text = Text("Resonance → Recursion!", font_size=28, color=ACCENT_YELLOW, weight=BOLD)
        resonance_text.next_to(align_text, DOWN, buff=0.5)
        
        self.play(
            Write(resonance_text),
            kv_circle.animate.set_stroke(ACCENT_YELLOW, width=6),
            steering_circle.animate.set_stroke(ACCENT_YELLOW, width=6),
            run_time=1.0
        )
        
        # Cleanup
        self.play(
            FadeOut(title),
            FadeOut(kv_circle),
            FadeOut(kv_label),
            FadeOut(kv_desc),
            FadeOut(steering_circle),
            FadeOut(steering_label),
            FadeOut(steering_desc),
            FadeOut(connection),
            FadeOut(align_text),
            FadeOut(resonance_text)
        )
        self.wait(0.2)
    
    def scene_5_breakthrough(self):
        """The Breakthrough: C2 configuration"""
        title = Text("The Breakthrough", font_size=56, weight=BOLD, color=WHITE)
        title.to_edge(UP, buff=0.5)
        self.play(FadeIn(title, shift=DOWN))
        
        # C2 Configuration components
        config_title = Text("C2 Configuration", font_size=32, color=ACCENT_YELLOW, weight=BOLD)
        config_title.next_to(title, DOWN, buff=0.5)
        
        components = VGroup(
            Text("• H18+H26 Steering (α=2.5)", font_size=24, color=ACCENT_RED),
            Text("• Full KV Replacement", font_size=24, color=ACCENT_GREEN),
            Text("• L26 Residual (α=0.6)", font_size=24, color=ACCENT_BLUE),
        )
        components.arrange(DOWN, buff=0.4, aligned_edge=LEFT)
        components.move_to(ORIGIN + UP * 0.3)
        
        self.play(Write(config_title), run_time=0.6)
        
        for component in components:
            self.play(Write(component), run_time=0.5)
        
        # Results
        results_title = Text("Results:", font_size=28, color=WHITE, weight=BOLD)
        results_title.next_to(components, DOWN, buff=0.8)
        
        results = VGroup(
            Text("Recursion Score: 0.15", font_size=22, color=ACCENT_YELLOW),
            Text("Success Rate: 20%", font_size=22, color=ACCENT_YELLOW),
            Text("Quality: 77%", font_size=22, color=ACCENT_YELLOW),
        )
        results.arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        results.next_to(results_title, DOWN, buff=0.4)
        
        self.play(Write(results_title), run_time=0.5)
        
        for result in results:
            self.play(Write(result), run_time=0.4)
        
        # Example output
        example_box = RoundedRectangle(
            height=1.5,
            width=8,
            corner_radius=0.2,
            stroke_color=ACCENT_YELLOW,
            stroke_width=2,
            fill_opacity=0.1
        )
        example_box.next_to(results, DOWN, buff=0.6)
        
        example_text = Text(
            '"observer watches itself respond"',
            font_size=18,
            color=ACCENT_YELLOW,
            font="monospace"
        )
        example_text.move_to(example_box)
        
        self.play(Create(example_box), run_time=0.5)
        self.play(Write(example_text), run_time=0.8)
        
        # Cleanup
        self.play(
            FadeOut(title),
            FadeOut(config_title),
            FadeOut(components),
            FadeOut(results_title),
            FadeOut(results),
            FadeOut(example_box),
            FadeOut(example_text)
        )
        self.wait(0.2)
    
    def scene_6_insight(self):
        """The Insight: Recursion is a memory state"""
        # Final insight
        insight_text = Text(
            "Recursion is a Memory State",
            font_size=48,
            color=ACCENT_YELLOW,
            weight=BOLD
        )
        insight_text.move_to(ORIGIN)
        
        # Subtitle
        subtitle = Text(
            "Content + Direction = Mode Transfer",
            font_size=28,
            color=WHITE
        )
        subtitle.next_to(insight_text, DOWN, buff=0.8)
        
        self.play(Write(insight_text, run_time=1.8))
        self.play(Write(subtitle, run_time=1.2))
        
        # Hold final frame
        self.wait(0.5)
