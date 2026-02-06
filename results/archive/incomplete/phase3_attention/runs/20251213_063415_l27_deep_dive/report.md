# L27 Deep Dive Analysis Report

**Model:** mistralai/Mistral-7B-v0.1
**Relay Layers:** [4, 14, 18, 25, 27]

## R_V Results

| Prompt | R_V |
|--------|-----|
| L5 | 0.4238 |
| champion | 0.4549 |
| L3 | 0.5136 |
| baseline | 0.6102 |
| L4 | 0.6143 |
| L2 | 0.6596 |
| L1 | 0.7375 |

## Most Discriminative Heads by Layer

Heads with biggest entropy difference (champion - baseline):

### L4

| Head | Entropy Δ | Self-Ref Δ |
|------|-----------|------------|
| H25 | +0.559 (↑ diffuse) | -0.0020 |
| H12 | +0.514 (↑ diffuse) | +0.0011 |
| H31 | +0.501 (↑ diffuse) | -0.0040 |
| H1 | +0.473 (↑ diffuse) | -0.0055 |
| H13 | +0.410 (↑ diffuse) | -0.0005 |

### L14

| Head | Entropy Δ | Self-Ref Δ |
|------|-----------|------------|
| H8 | +1.045 (↑ diffuse) | +0.0417 |
| H21 | +0.857 (↑ diffuse) | +0.0475 |
| H10 | +0.852 (↑ diffuse) | +0.0165 |
| H3 | +0.805 (↑ diffuse) | +0.0100 |
| H6 | +0.800 (↑ diffuse) | +0.0044 |

### L18

| Head | Entropy Δ | Self-Ref Δ |
|------|-----------|------------|
| H21 | +0.973 (↑ diffuse) | +0.0069 |
| H30 | +0.931 (↑ diffuse) | +0.0045 |
| H11 | +0.905 (↑ diffuse) | +0.0103 |
| H3 | +0.778 (↑ diffuse) | +0.0235 |
| H22 | +0.673 (↑ diffuse) | +0.0124 |

### L25

| Head | Entropy Δ | Self-Ref Δ |
|------|-----------|------------|
| H10 | +1.227 (↑ diffuse) | +0.0155 |
| H29 | +1.016 (↑ diffuse) | +0.0078 |
| H9 | +0.883 (↑ diffuse) | +0.0337 |
| H15 | +0.791 (↑ diffuse) | +0.0105 |
| H18 | +0.676 (↑ diffuse) | +0.0019 |

### L27

| Head | Entropy Δ | Self-Ref Δ |
|------|-----------|------------|
| H26 | +0.837 (↑ diffuse) | +0.0308 |
| H31 | -0.680 (↓ focused) | -0.0005 |
| H23 | +0.658 (↑ diffuse) | -0.0016 |
| H29 | +0.653 (↑ diffuse) | +0.0094 |
| H27 | +0.621 (↑ diffuse) | -0.0038 |

## Key Findings

### Heads that become MORE FOCUSED on champion (entropy drops):

- **L27 H31**: entropy Δ = -0.680
- **L25 H13**: entropy Δ = -0.491
- **L27 H3**: entropy Δ = -0.429

### Heads that become MORE DIFFUSE on champion (entropy rises):

- **L25 H10**: entropy Δ = +1.227
- **L14 H8**: entropy Δ = +1.045
- **L25 H29**: entropy Δ = +1.016
- **L18 H21**: entropy Δ = +0.973
- **L18 H30**: entropy Δ = +0.931
- **L18 H11**: entropy Δ = +0.905
- **L25 H9**: entropy Δ = +0.883
- **L14 H21**: entropy Δ = +0.857
- **L14 H10**: entropy Δ = +0.852
- **L27 H26**: entropy Δ = +0.837