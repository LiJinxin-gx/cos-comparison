# Getting Started

5-minute guide to cos-comparison.

## Contents

- [Installation](#installation)
- [Quick Examples](#quick-examples)
- [Basics](#basics)
- [Tensors](#tensors)
- [Common Use Cases](#common-use-cases)
- [Dimensions](#dimensions)
- [Backend Management](#backend-management)
- [Troubleshooting](#troubleshooting)

---

## Installation

```bash
pip install cos-comparison
```

C backends compile automatically if a compiler is available; otherwise the pure Python backend works with zero dependencies.

```python
import cos_comparison
print(cos_comparison.__version__)
from cos_comparison import core as cc
print(cc.get_mode())  # enabled backends
```

---

## Quick Examples

### 1D Step Detection

```python
from cos_comparison import core as cc

data = [1.0, 1.0, 1.0, 5.0, 5.0, 5.0]
result = cc.cos_comparison_passive_1d(data, window_size=(2,), step=(1,), d=(1,))
# Low similarity at the 1.0→5.0 transition = edge detected
```

Two windows of size 2 slide across the data, offset by `d=1`. Low similarity = transition/edge.

### 2D Vertical Edges

```python
vertical_edges = cc.cos_comparison_passive_2d(
    image, window_size=(3, 3), step=(1, 1), d=(0, 1))  # compare with right neighbor
```

### 1D Template Matching

```python
result = cc.cos_comparison_active_1d(
    [0, 0, 1, 2, 3, 2, 1, 0, 0],
    kernel=[1, 2, 3, 2, 1],
    step=(1,))
# Peak indicates where the pattern occurs
```

---

## Basics

| Concept | Parameter | Details |
|---------|-----------|---------|
| **Passive mode** | `d` | Self-similarity: two offset windows within the same data |
| **Active mode** | `kernel` | Template matching: sliding kernel vs data windows |
| **Algorithm** | `algorithm` | `_cos` (direction), `_mod` (magnitude), `_cosmod` (both, default) |
| **Window** | `window_size` | Tuple matching data dimension, e.g. `(3, 3)` for 2D |
| **Step** | `step` | Sliding stride; larger = fewer output points = faster |

→ [Dual Modes](principles/dual-mode.md) · [Similarity Measures](principles/similarity-measures.md) · [Common Parameters](api/core.md#common-parameters)

---

## Tensors

The core uses a stride-based tensor view (`vector_map_as_tensor`) with NumPy-like indexing:

```python
from cos_comparison import core as cc

# Create and fill
t = cc.create_void_list((3, 3))        # 3×3 filled with 0.0
t[1, 1] = 123.0                        # direct tuple assignment

# Load nested data
v = cc.load_as_default_data([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Slicing creates zero-copy views
view = v[1:3]                          # shape (2, 3)
v.mean()                               # 5.0
v.variance()                           # ~6.667

# Step slices, negative indices, dimension collapse
v[:, ::-1].shape                       # (3, 3) reversed columns
v[0, 0]                                # 1.0 (scalar after full indexing)
```

---

## Common Use Cases

### Edge Detection

```python
# Horizontal edges (compare with bottom neighbor)
h = cc.cos_comparison_passive_2d(image, window_size=(3, 3), d=(1, 0))
# Vertical edges (compare with right neighbor)
v = cc.cos_comparison_passive_2d(image, window_size=(3, 3), d=(0, 1))
```

### Local Statistics

```python
blurred  = cc.mean_local_2d(image, local_size=(5, 5))      # average pooling
variance = cc.local_variance_2d(image, local_size=(5, 5))  # texture strength
```

---

## Dimensions

```python
# 1D (signals, audio, time series)
cc.cos_comparison_passive_1d(signal, window_size=(10,), d=(1,))

# 2D (images, grids, matrices)
cc.cos_comparison_passive_2d(image, window_size=(3, 3), d=(1, 0))

# 3D (video, volumetric data)
cc.cos_comparison_passive_3d(volume, window_size=(3, 3, 3), d=(1, 0, 0))

# N-dimensional (generic)
cc.cos_comparison_passive(data, window_size=(3, 3, 3, 3), d=(1, 0, 0, 0))
```

---

## Backend Management

```python
cc.get_mode()                  # current backend priority
cc.set_mode("cos_comparison")  # force pure Python for debugging
```

→ [Backend Management System](architecture/backend-system.md)

---

## Troubleshooting

| Symptom | Cause / Fix |
|---------|-------------|
| All results `1.0` | Data is uniform; try different data or parameters |
| All results `0.0` | Windows are completely different; check data/parameters |
| `ValueError: effectless args.` | Window (+ `d` in passive) doesn't fit; enlarge data or shrink window/`d` |
| `ImportError` | No backend available; ensure `pip install cos-comparison` succeeded |

> Non-core packages (`interface`, `data`, `generate_layer`) import cleanly in a fresh interpreter. The core module remains production-ready; non-core layers keep evolving — see [Cognitive Layer APIs](api/cognitive-layers.md).

---

**Need help?** [Open an issue](https://github.com/LiJinxin-gx/cos-comparison/issues)
