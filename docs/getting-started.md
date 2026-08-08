# Getting Started

This guide will help you get up and running with cos-comparison in 5 minutes.

## Installation

### Basic Installation

```bash
pip install cos-comparison
```

This installs the package. The installer will automatically attempt to compile C acceleration backends during installation; if no C compiler is available, it will fall back to the pure Python backend automatically, which works out of the box with zero dependencies.

### Verify Installation

```python
import cos_comparison
print(cos_comparison.__version__)

from cos_comparison import core as cc
print(cc.get_mode())          # enabled backends, e.g. ('.cos_comparison_pydll', ...)
print(cc.get_available_backends())  # all configured backends
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
print("Output:", list(result))
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

# Detect vertical edges (compare with right neighbor, d=(0, 1))
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

## Working with Tensors

The core uses a stride-based tensor view (`vector_map_as_tensor`) with NumPy-like indexing:

```python
from cos_comparison import core as cc

# Create a tensor directly
t = cc.create_void_list((3, 3))        # 3x3 tensor filled with 0.0
t[1, 1] = 123.0                        # direct tuple assignment

# Load nested data into a tensor
v = cc.load_as_default_data([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Slicing creates zero-copy views
view = v[1:3]
print(view.shape)          # (2, 3)
print(v.mean())            # 5.0
print(v.variance())        # ~6.667

# Step slices, negative indices, dimension collapse — NumPy style
print(v[:, ::-1].shape)    # (3, 3), reversed columns
print(v[0, 0])             # 1.0, scalar after full indexing
```

## Common Use Cases

### 1. Edge Detection

```python
# Horizontal edges (compare with bottom neighbor)
horizontal = cc.cos_comparison_passive_2d(
    image, window_size=(3, 3), step=(1, 1), d=(1, 0)
)

# Vertical edges (compare with right neighbor)
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

# Try C extension first, then ctypes, fall back to pure Python
cc.set_mode(["cos_comparison_pydll", "cos_comparison_c", "cos_comparison"])
```

Backend names may be written with or without the leading dot (e.g. `"cos_comparison"` or `".cos_comparison"`).

## Next Steps

- **[Passive Mode API](api/passive-mode.md)** — Detailed reference for passive mode
- **[Active Mode API](api/active-mode.md)** — Detailed reference for active mode
- **[Similarity Measures](principles/similarity-measures.md)** — Understand the algorithms
- **[Architecture](architecture/seven-layer.md)** — Learn about the seven-layer design
- **[Cognitive Layer APIs](api/cognitive-layers.md)** — Memory, logic, interface and more

## Troubleshooting

### ImportError: No module named 'cos_comparison'

```bash
pip install cos-comparison
```

### All results are 1.0

Your data might be uniform. Try different data or parameters.

### All results are 0.0

Windows might be completely different. Check your data and parameters.

### ValueError: effectless args.

The window (+ displacement `d` in passive mode) does not fit inside the data bounds. Increase the data size or shrink `window_size` / `d`.

### ImportError when importing interface / data / generate_layer

These non-core packages have known import issues pending fixes — in a fresh interpreter, `cos_comparison.interface`, `cos_comparison.data`, and `cos_comparison.generate_layer` all fail to import (see the [Cognitive Layer APIs](api/cognitive-layers.md) notes). They do not affect the core module, which is fully production-ready.

---

**Need help?** [Open an issue on GitHub](https://github.com/LiJinxin-gx/cos-comparison/issues)