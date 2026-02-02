# Manim Visualization: Recursive Self-Observation Discovery

A world-class 3-minute animated narrative describing the essence of the mechanistic interpretability research on recursive self-observation in transformers.

## Overview

This animation tells the complete story:
1. **The Discovery**: Geometric contraction at Layer 27 (R_V < 1.0)
2. **Universal Pattern**: Found across 6 models, MoE amplification
3. **The Puzzle**: Steering alone fails
4. **The Mechanism**: Two attractors (KV cache + Steering vector)
5. **The Breakthrough**: C2 configuration achieves recursion
6. **The Insight**: "Recursion is a memory state"

## Requirements

```bash
pip install manim
```

For voiceover support (optional):
```bash
pip install manim-voiceover
```

## Usage

### Basic Rendering (without voiceover)

```bash
manim -pql visualizations/core_findings_narrative.py RecursiveSelfObservation
```

### With Voiceover

The script includes voiceover text that will be automatically synthesized if you have `manim-voiceover` installed and configured. To use your own voiceover:

1. Record a 3-minute voiceover matching the script timing
2. Replace the `voiceover` context managers with regular `self.wait()` calls
3. Or use Manim's built-in TTS if available

### Render Quality Options

- **Low quality (fast preview)**: `-pql` (480p, 15fps)
- **Medium quality**: `-pqm` (720p, 30fps)
- **High quality**: `-pqh` (1080p, 60fps)
- **4K**: `-pqk` (2160p, 60fps)

### Example Commands

```bash
# Quick preview
manim -pql visualizations/core_findings_narrative.py RecursiveSelfObservation

# High quality render
manim -pqh visualizations/core_findings_narrative.py RecursiveSelfObservation

# Render with custom resolution
manim -pql --resolution 1920,1080 visualizations/core_findings_narrative.py RecursiveSelfObservation
```

## Script Structure

The script is organized into 6 scenes, each synchronized with voiceover:

- **Scene 1** (0:00-0:30): The Discovery - R_V metric and geometric contraction
- **Scene 2** (0:30-0:50): Universal Pattern - 6 models, MoE amplification
- **Scene 3** (0:50-1:10): The Puzzle - Steering alone fails
- **Scene 4** (1:10-2:00): The Mechanism - Two attractors (KV + Steering)
- **Scene 5** (2:00-2:40): The Breakthrough - C2 configuration
- **Scene 6** (2:40-3:00): The Insight - "Recursion is a memory state"

## Customization

### Colors

The script uses a custom color scheme defined at the top:
- `ACCENT_BLUE`: Primary accent
- `ACCENT_RED`: Contraction/steering
- `ACCENT_GREEN`: KV cache/content
- `ACCENT_YELLOW`: Highlights/insights
- `ACCENT_PURPLE`: MoE models

### Timing

Animation timings are synchronized with voiceover. To adjust:
1. Modify `run_time` parameters in `self.play()` calls
2. Adjust voiceover text length
3. Use `self.wait()` for pauses

## Output

The rendered video will be saved to:
```
media/videos/core_findings_narrative/1080p60/RecursiveSelfObservation.mp4
```

## Notes

- The script uses Manim's `voiceover` context manager for automatic synchronization
- All animations are designed to be clear and visually appealing
- The script follows the research narrative exactly as documented
- Total runtime: ~3 minutes

## Troubleshooting

**Issue**: Voiceover not working
- **Solution**: Install `manim-voiceover` or remove voiceover context managers

**Issue**: Rendering is slow
- **Solution**: Use `-pql` for preview, render final with `-pqh` or `-pqk`

**Issue**: Colors look wrong
- **Solution**: Check your terminal/display color profile, or adjust color constants

**Issue**: Animations out of sync
- **Solution**: Adjust `run_time` parameters or voiceover text length





