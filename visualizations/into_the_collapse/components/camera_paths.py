"""
Choreographed camera movements for cinematic effect.
"""

from manim import *
import numpy as np
from typing import Callable
import sys
sys.path.append(str(Path(__file__).parent.parent))
from data.verified_values import Visual, TOTAL_LAYERS, CRITICAL_LAYER


# Custom rate functions for emotional pacing

def gravity_fall(t: float) -> float:
    """
    Rate function for gravitational collapse.
    Slow start, accelerating fall, slight bounce at end.
    """
    if t < 0.3:
        # Slow start (building tension)
        return (t / 0.3) ** 0.5 * 0.1
    elif t < 0.9:
        # Accelerating fall
        normalized = (t - 0.3) / 0.6
        return 0.1 + 0.85 * (normalized ** 2)
    else:
        # Slight overshoot and settle
        normalized = (t - 0.9) / 0.1
        return 0.95 + 0.05 * np.sin(normalized * np.pi)


def tension_build(t: float) -> float:
    """
    Rate function for building tension.
    Slow exponential approach to 1.
    """
    return 1 - np.exp(-3 * t)


def cinematic_ease(t: float) -> float:
    """
    Smooth cinematic easing.
    Slow start, linear middle, slow end.
    """
    if t < 0.15:
        return (t / 0.15) ** 2 * 0.15
    elif t < 0.85:
        return 0.15 + (t - 0.15) / 0.7 * 0.7
    else:
        return 0.85 + (1 - ((1 - t) / 0.15) ** 2) * 0.15


def pulse_rate(t: float, frequency: float = 2.0) -> float:
    """
    Pulsing rate for breathing effects.
    """
    return 0.5 + 0.5 * np.sin(t * frequency * 2 * np.pi)


class CameraChoreographer:
    """
    Manages camera movements for ThreeDScene.
    Provides smooth, intentional camera choreography.
    """

    def __init__(self, scene: ThreeDScene):
        self.scene = scene
        self.saved_states = {}

    def save_state(self, name: str):
        """Save current camera state."""
        self.saved_states[name] = {
            'phi': self.scene.camera.get_phi(),
            'theta': self.scene.camera.get_theta(),
            'distance': self.scene.camera.get_distance(),
            'center': self.scene.camera.frame_center.copy()
        }

    def restore_state(self, name: str, run_time: float = 2.0):
        """Animate back to saved state."""
        if name not in self.saved_states:
            return None

        state = self.saved_states[name]
        return AnimationGroup(
            self.scene.camera.animate.set_phi(state['phi']),
            self.scene.camera.animate.set_theta(state['theta']),
            # self.scene.camera.animate.set_distance(state['distance']),
            run_time=run_time,
            rate_func=cinematic_ease
        )

    # Pre-defined camera movements

    def vastness_pullback(self, start_distance: float = 5.0, end_distance: float = 50.0):
        """
        Act I: Exponential pullback revealing transformer scale.
        """
        # This would be implemented as camera distance animation
        # In Manim, we scale the frame instead
        return self.scene.camera.frame.animate.scale(end_distance / start_distance)

    def descent_path(self, layer_centers: list, duration: float = 10.0):
        """
        Act II: Camera follows tokens down through layers.
        Returns updater function for continuous movement.
        """
        total_layers = len(layer_centers)

        def camera_updater(mob, dt):
            # This would track progress and interpolate positions
            pass

        return camera_updater

    def orbit_collapsed(self, center: np.ndarray, radius: float = 5.0, duration: float = 5.0):
        """
        Act IV: Slow orbit around collapsed geometry.
        """
        return Rotate(
            self.scene.camera.frame,
            angle=TAU / 2,
            about_point=center,
            run_time=duration,
            rate_func=linear
        )

    def final_pullback(self, target_distance: float = 100.0):
        """
        Credits: Final pullback showing entire network.
        """
        return self.scene.camera.frame.animate.scale(2.0)


class LayerDescentTracker:
    """
    Tracks progress through layers for coordinated animations.
    """

    def __init__(self, num_layers: int = TOTAL_LAYERS):
        self.num_layers = num_layers
        self.progress = ValueTracker(0)  # 0 = layer 0, 1 = layer num_layers

    def get_current_layer(self) -> int:
        """Get the current layer number."""
        return int(self.progress.get_value() * self.num_layers)

    def get_layer_fraction(self) -> float:
        """Get fraction through current layer."""
        full = self.progress.get_value() * self.num_layers
        return full - int(full)

    def is_at_critical_layer(self) -> bool:
        """Check if at Layer 27."""
        return self.get_current_layer() == CRITICAL_LAYER

    def get_color_temperature(self) -> ManimColor:
        """Get color based on current layer (cool -> warm)."""
        t = self.progress.get_value()
        return interpolate_color(
            ManimColor(Visual.LAYER_COLOR_COOL),
            ManimColor(Visual.LAYER_COLOR_WARM),
            t
        )


class PacingController:
    """
    Controls animation pacing for emotional beats.
    """

    def __init__(self):
        self.beat_times = {
            'awe': 0.0,          # Act I start
            'immersion': 45.0,   # Act II start
            'tension': 120.0,    # Act III start
            'climax': 180.0,     # Act IV start
            'resolution': 270.0, # Act V start
        }

    def get_beat_progress(self, current_time: float, beat_name: str) -> float:
        """Get progress within a specific emotional beat."""
        if beat_name not in self.beat_times:
            return 0.0

        beat_start = self.beat_times[beat_name]

        # Find next beat
        beats = sorted(self.beat_times.items(), key=lambda x: x[1])
        beat_end = 360.0  # Default to end
        for i, (name, time) in enumerate(beats):
            if name == beat_name and i < len(beats) - 1:
                beat_end = beats[i + 1][1]
                break

        if current_time < beat_start:
            return 0.0
        elif current_time >= beat_end:
            return 1.0
        else:
            return (current_time - beat_start) / (beat_end - beat_start)


# Utility functions for camera work

def smooth_follow_path(
    points: list,
    progress_tracker: ValueTracker,
    smoothing: float = 0.3
) -> Callable:
    """
    Returns an updater that smoothly follows a path of points.
    """
    def updater(mob):
        t = progress_tracker.get_value()
        n = len(points)

        if t >= 1.0:
            mob.move_to(points[-1])
            return

        # Find segment
        segment = t * (n - 1)
        i = int(segment)
        frac = segment - i

        if i >= n - 1:
            mob.move_to(points[-1])
            return

        # Interpolate with smoothing
        if i > 0 and i < n - 2:
            # Catmull-Rom spline for smoothness
            p0 = points[i - 1]
            p1 = points[i]
            p2 = points[i + 1]
            p3 = points[i + 2] if i + 2 < n else points[-1]

            t2 = frac * frac
            t3 = t2 * frac

            pos = 0.5 * (
                (2 * p1) +
                (-p0 + p2) * frac +
                (2*p0 - 5*p1 + 4*p2 - p3) * t2 +
                (-p0 + 3*p1 - 3*p2 + p3) * t3
            )
        else:
            # Linear interpolation at edges
            pos = points[i] + frac * (points[i + 1] - points[i])

        mob.move_to(pos)

    return updater
