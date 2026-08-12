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

## Quick Examples

### 1D Signal: Detect a Step Transition

```python
from cos_comparison import core as cc

data = [1.0, 1.0, 1.0, 5.0, 5.0, 5.0]
result = cc.cos_comparison_passive_1d(data, window_size=(2,), step=(1,), d=(1,))

print("Input:", data)
print("Output:", list(result))
```

**What's happening**: Two windows of size 2 slide across the data, offset by 1 position (displacement `d=1`). Similarity between windows is computed at each position — low similarity = big difference = edge/transition.

### 2D Image: Detect Vertical Edges

```python
image = [
    [1, 1, 1, 0, 0, 0],
    [1, 1, 1, 0, 0, 0],
    [1, 1, 1, 0, 0, 0],
    [1, 1, 1, 0, 0, 0],
    [1, 1, 1, 0, 0, 0],
    [1, 1, 1, 0, 0, 0],
]

vertical_edges = cc.cos_comparison_passive_2d(
    image, window_size=(3, 3), step=(1, 1), d=(0, 1)   # compare with right neighbor
)

for row in vertical_edges:
    print([f"{x:.2f}" for x in row])
```

### 1D Template Matching (Active Mode)

```python
data = [0, 0, 1, 2, 3, 2, 1, 0, 0]
kernel = [1, 2, 3, 2, 1]

result = cc.cos_comparison_active_1d(data, kernel=kernel, step=(1,))
print("Match scores:", [f"{x:.2f}" for x in result])
```

## Understanding the Basics

- **Two modes**: Passive (self-similarity, parameter `d`) and Active (template matching, parameter `kernel`) — see [Dual Working Modes](principles/dual-mode.md)
- **Three similarity algorithms**: `_cos` (direction), `_mod` (magnitude), `_cosmod` (both, default) — see [Similarity Measures](principles/similarity-measures.md)
- **Key parameters**: `window_size`, `step`, `d` (passive), `kernel` (active), `algorithm` — see [Core Module — Common Parameters](api/core.md#common-parameters)

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
horizontal = cc.cos_comparison_passive_2d(image, window_size=(3, 3), step=(1, 1), d=(1, 0))  # bottom neighbor
vertical   = cc.cos_comparison_passive_2d(image, window_size=(3, 3), step=(1, 1), d=(0, 1))  # right neighbor
```

### 2. Local Statistics

```python
blurred  = cc.mean_local_2d(image, local_size=(5, 5), step=(1, 1))   # local mean (average pooling)
variance = cc.local_variance_2d(image, local_size=(5, 5), step=(1, 1))  # texture strength
```

## Working with Different Dimensions

```python
# 1D (signals, audio, time series)
result = cc.cos_comparison_passive_1d(signal, window_size=(10,), step=(1,), d=(1,))

# 2D (images, grids, matrices)
result = cc.cos_comparison_passive_2d(image, window_size=(3, 3), step=(1, 1), d=(1, 0))

# 3D (video, volumetric data)
result = cc.cos_comparison_passive_3d(volume, window_size=(3, 3, 3), step=(1, 1, 1), d=(1, 0, 0))

# N-dimensional (generic)
result = cc.cos_comparison_passive(data, window_size=(3, 3, 3, 3), step=(1, 1, 1, 1), d=(1, 0, 0, 0))
```

## Backend Management

```python
print(cc.get_mode())                      # current backend priority
cc.set_mode("cos_comparison")             # force pure Python (debugging)
cc.set_mode(["cos_comparison_pydll", "cos_comparison_c", "cos_comparison"])  # custom order
```

Backend names may be written with or without the leading dot (e.g. `"cos_comparison"` or `".cos_comparison"`). See [Backend Management System](architecture/backend-system.md) for details.

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

### Importing non-core packages

`cos_comparison.interface`, `cos_comparison.data`, and `cos_comparison.generate_layer` import cleanly in a fresh interpreter (verified). The core module remains the production-ready center; the non-core layers (`sense / memory / brain / action / generate`) keep evolving — see the [Cognitive Layer APIs](api/cognitive-layers.md) for their current surface.

---

**Need help?** [Open an issue on GitHub](https://github.com/LiJinxin-gx/cos-comparison/issues)