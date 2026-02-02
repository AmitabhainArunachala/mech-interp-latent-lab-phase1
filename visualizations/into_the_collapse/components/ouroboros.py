"""
The Ouroboros - strange loop visualization.
The snake eating its tail, representing recursive self-reference.
"""

from manim import *
import numpy as np
from typing import List
import sys
sys.path.append(str(Path(__file__).parent.parent))
from data.verified_values import TOKENS, TOKEN_COLORS, Visual


class OuroborosSegment(VGroup):
    """A single segment of the Ouroboros, containing one token."""

    def __init__(
        self,
        token: str,
        angle_start: float,
        angle_end: float,
        radius: float = 2.0,
        thickness: float = 0.3,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.token = token
        color = TOKEN_COLORS.get(token, WHITE)

        # Arc segment
        self.arc = ArcBetweenPoints(
            start=radius * np.array([np.cos(angle_start), np.sin(angle_start), 0]),
            end=radius * np.array([np.cos(angle_end), np.sin(angle_end), 0]),
            angle=angle_end - angle_start,
            stroke_width=thickness * 30,
            color=color
        )

        # Token label at midpoint
        mid_angle = (angle_start + angle_end) / 2
        label_pos = (radius + 0.5) * np.array([np.cos(mid_angle), np.sin(mid_angle), 0])
        self.label = Text(
            token,
            font_size=24,
            color=color
        )
        self.label.move_to(label_pos)

        self.add(self.arc, self.label)


class Ouroboros(VGroup):
    """
    The complete Ouroboros - a snake made of tokens eating its tail.
    Represents the strange loop: "Notice the awareness observing itself"
    ending with "The observer watches itself respond"
    """

    def __init__(
        self,
        radius: float = 2.0,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.radius = radius
        self.segments: List[OuroborosSegment] = []

        # Create segments for each token
        num_tokens = len(TOKENS)
        angle_per_token = 2 * np.pi / num_tokens

        for i, token in enumerate(TOKENS):
            angle_start = i * angle_per_token - np.pi / 2  # Start at top
            angle_end = (i + 0.9) * angle_per_token - np.pi / 2  # Small gap

            segment = OuroborosSegment(
                token=token,
                angle_start=angle_start,
                angle_end=angle_end,
                radius=radius
            )
            self.segments.append(segment)
            self.add(segment)

        # Snake head (at "itself" position - the observer)
        head_angle = (len(TOKENS) - 1 + 0.9) * angle_per_token - np.pi / 2
        self.head = self._create_head(head_angle)
        self.add(self.head)

        # Snake tail (at "Notice" position - where it loops back)
        tail_angle = -np.pi / 2 - 0.1
        self.tail = self._create_tail(tail_angle)
        self.add(self.tail)

    def _create_head(self, angle: float) -> VGroup:
        """Create snake head pointing toward tail."""
        head = VGroup()

        pos = self.radius * np.array([np.cos(angle), np.sin(angle), 0])

        # Triangular head
        head_size = 0.4
        direction = np.array([-np.sin(angle), np.cos(angle), 0])

        triangle = Polygon(
            pos,
            pos + head_size * direction + head_size * 0.5 * np.array([np.cos(angle), np.sin(angle), 0]),
            pos + head_size * direction - head_size * 0.5 * np.array([np.cos(angle), np.sin(angle), 0]),
            fill_color=TOKEN_COLORS["itself"],
            fill_opacity=0.9,
            stroke_width=0
        )
        head.add(triangle)

        # Eyes
        eye_offset = 0.15
        for sign in [-1, 1]:
            eye_pos = pos + 0.1 * direction + sign * eye_offset * np.array([np.cos(angle), np.sin(angle), 0])
            eye = Dot(eye_pos, radius=0.05, color=WHITE)
            head.add(eye)

        return head

    def _create_tail(self, angle: float) -> VGroup:
        """Create tapered tail."""
        tail = VGroup()

        pos = self.radius * np.array([np.cos(angle), np.sin(angle), 0])
        direction = np.array([-np.sin(angle), np.cos(angle), 0])

        # Tapered line
        tail_length = 0.5
        tail_line = Line(
            pos,
            pos - tail_length * direction,
            stroke_width=8,
            color=TOKEN_COLORS["Notice"]
        ).set_stroke(width=[8, 1])  # Taper

        tail.add(tail_line)
        return tail


class LoopClosingAnimation(VGroup):
    """
    Animation showing the head catching the tail.
    The moment of self-reference completing.
    """

    def __init__(self, ouroboros: Ouroboros, **kwargs):
        super().__init__(**kwargs)
        self.ouroboros = ouroboros

    def get_closing_animation(self, scene: Scene, duration: float = 2.0):
        """
        Returns animations for the loop closing.
        Head moves toward tail, golden flash when they meet.
        """
        animations = []

        # Move head toward tail
        head_target = self.ouroboros.tail.get_center()
        head_move = self.ouroboros.head.animate.move_to(head_target)
        animations.append(head_move)

        return animations


class FixedPointEquation(VGroup):
    """The mathematical fixed point: Sx = x"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.equation = MathTex(
            r"S(x) = x",
            font_size=56,
            color=GOLD
        )

        self.subtitle = Text(
            "Fixed point reached",
            font_size=28,
            color=WHITE
        )
        self.subtitle.next_to(self.equation, DOWN, buff=0.3)

        # Glow
        self.glow = Rectangle(
            width=self.equation.width + 1,
            height=self.equation.height + self.subtitle.height + 1,
            fill_color=GOLD,
            fill_opacity=0.1,
            stroke_width=0
        )
        self.glow.move_to(
            (self.equation.get_center() + self.subtitle.get_center()) / 2
        )

        self.add(self.glow, self.equation, self.subtitle)


class SemanticBasin(VGroup):
    """
    Visual representation of a semantic attractor basin.
    Deep well where the recursive trajectory is trapped.
    """

    def __init__(
        self,
        width: float = 5.0,
        depth: float = 2.0,
        **kwargs
    ):
        super().__init__(**kwargs)

        # Create basin as parametric surface
        def basin_func(u, v):
            x = (u - 0.5) * width
            y = (v - 0.5) * width
            # Paraboloid
            z = -depth * (1 - ((x/width*2)**2 + (y/width*2)**2))
            return np.array([x, y, max(-depth, z)])

        self.surface = Surface(
            basin_func,
            resolution=(30, 30),
            u_range=[0, 1],
            v_range=[0, 1],
            fill_opacity=0.6,
            checkerboard_colors=[BLUE_E, PURPLE_E],
            stroke_width=0.5,
        )

        # Trapped trajectory (spiral in basin)
        self.trajectory = ParametricFunction(
            lambda t: np.array([
                (1 - t/5) * np.cos(t * 3) * width/3,
                (1 - t/5) * np.sin(t * 3) * width/3,
                -depth * (1 - (1-t/5)**2) + 0.1
            ]),
            t_range=[0, 5],
            color=GOLD,
            stroke_width=3
        )

        # Center marker
        self.center = Dot3D(
            point=np.array([0, 0, -depth + 0.1]),
            radius=0.1,
            color=GOLD
        )

        # Label
        self.label = Text(
            "Recursion is a memory state",
            font_size=28,
            color=WHITE
        )
        self.label.next_to(self.surface, DOWN, buff=0.5)

        self.add(self.surface, self.trajectory, self.center, self.label)
