# Getting Started

This guide will help you get up and running with cos-comparison in 5 minutes.

## Installation

### Basic Installation

```bash
pip install cos-comparison
```

This installs the pure Python version, which has zero dependencies and works out of the box.

### Verify Installation

```python
from cos_comparison import core as cc
print(cc.__version__)
```

## Quick Example: 1D Signal

```python
from cos_comparison import core as cc

# Create a simple signal with a step transition
data = [1.0, 1.0, 1.0, 5.0, 5.0, 5.0]

# Compute self-similarity
result = cc.cos_comparison_passive_1d(
    data,
    window_size=(2,),
    step=(1,),
    d=(1,)
)

print("Input:", data)
print("Output:", result)
```

**What's happening**:
- Two windows of size 2 slide across the data
- Windows are offset by 1 position (displacement d=1)
- At each position, similarity between windows is computed
- Low similarity = big difference = edge/transition

## Quick Example: 2D Image Edges

```python
from cos_comparison import core as cc

# Simple 6x6 test image with a vertical edge
image = [
    [1, 1, 1, 0, 0, 0],
    [1, 1, 1, 0, 0, 0],
    [1, 1, 1, 0, 0, 0],
    [1, 1, 1, 0, 0, 0],
    [1, 1, 1, 0, 0, 0],
    [1, 1, 1, 0, 0, 0],
]

# Detect vertical edges
vertical_edges = cc.cos_comparison_passive_2d(
    image,
    window_size=(3, 3),
    step=(1, 1),
    d=(0, 1)
)

for row in vertical_edges:
    print([f"{x:.2f}" for x in row])
```

## Quick Example: Template Matching (Active Mode)

```python
from cos_comparison import core as cc

# Data with a pattern embedded
data = [0, 0, 1, 2, 3, 2, 1, 0, 0]

# Pattern to find
kernel = [1, 2, 3, 2, 1]

# Find the pattern
result = cc.cos_comparison_active_1d(
    data,
    kernel=kernel,
    step=(1,)
)

print("Match scores:", [f"{x:.2f}" for x in result])
```

## Understanding the Basics

### Two Modes

| Mode | Use Case | Key Parameter |
|------|----------|---------------|
| Passive | Edge detection, feature extraction | Displacement `d` |
| Active | Template matching, pattern search | Kernel `kernel` |

### Three Similarity Algorithms

| Algorithm | Measures | Use When |
|-----------|----------|----------|
| `_cos` | Direction (pattern shape) | Intensity variations don't matter |
| `_mod` | Magnitude (intensity level) | Only care about brightness |
| `_cosmod` | Both (default) | General purpose |

### Key Parameters

- **`window_size`**: Size of comparison window
- **`step`**: Sliding step size
- **`d`** (passive): Window displacement
- **`kernel`** (active): Template to match
- **`algorithm`**: Similarity measure

## Common Use Cases

### 1. Edge Detection

```python
# Horizontal edges
horizontal = cc.cos_comparison_passive_2d(
    image, window_size=(3, 3), step=(1, 1), d=(1, 0)
)

# Vertical edges
vertical = cc.cos_comparison_passive_2d(
    image, window_size=(3, 3), step=(1, 1), d=(0, 1)
)
```

### 2. Local Statistics

```python
# Local mean (blur / average pooling)
blurred = cc.mean_local_2d(image, local_size=(5, 5), step=(1, 1))

# Local variance (texture strength)
variance = cc.local_variance_2d(image, local_size=(5, 5), step=(1, 1))
```

## Working with Different Dimensions

### 1D (Signals, Audio, Time Series)

```python
result = cc.cos_comparison_passive_1d(signal, window_size=(10,), step=(1,), d=(1,))
```

### 2D (Images, Grids, Matrices)

```python
result = cc.cos_comparison_passive_2d(image, window_size=(3, 3), step=(1, 1), d=(1, 0))
```

### 3D (Video, Volumetric Data)

```python
result = cc.cos_comparison_passive_3d(
    volume,
    window_size=(3, 3, 3),
    step=(1, 1, 1),
    d=(1, 0, 0)
)
```

### N-Dimensional (Generic)

```python
result = cc.cos_comparison_passive(
    data,
    window_size=(3, 3, 3, 3),
    step=(1, 1, 1, 1),
    d=(1, 0, 0, 0)
)
```

## Backend Management

### Check Available Backends

```python
print(cc.get_mode())
```

### Switch Backend

```python
# Force pure Python (for debugging)
cc.set_mode("cos_comparison")

# Try C backend, fall back to pure Python
cc.set_mode(["cos_comparison_c", "cos_comparison"])
```

## Next Steps

- **[Passive Mode API](api/passive-mode.md)** — Detailed reference for passive mode
- **[Active Mode API](api/active-mode.md)** — Detailed reference for active mode
- **[Similarity Measures](principles/similarity-measures.md)** — Understand the algorithms
- **[Architecture](architecture/seven-layer.md)** — Learn about the seven-layer design

## Troubleshooting

### ImportError: No module named 'cos_comparison'

```bash
pip install cos-comparison
```

### All results are 1.0

Your data might be uniform. Try different data or parameters.

### All results are 0.0

Windows might be completely different. Check your data and parameters.

---

**Need help?** [Open an issue on GitHub](https://github.com/LiJinxin/cos-comparison/issues)
