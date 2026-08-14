# Statistics Functions

## Overview

Cos-comparison provides sliding window statistical functions (local mean, variance) built on top of the active mode infrastructure.

## mean_local

Computes the local mean (average) within a sliding window. Equivalent to average pooling.

### Function Signature

```python
mean_local(data,
           *arg,
           local_size=None,
           step=None,
           weight=None,
           output=None,
           output_start=None,
           output_step=None,
           **kwarg)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | nested list / tensor | required | Input data; 1D–ND |
| `local_size` | tuple / int / `default_contain` | `default_contain(1)` | Window size — tuple for per-dimension sizes (e.g. `(3, 3)`), int for all dimensions |
| `step` | tuple / int / `default_contain` | `default_contain(1)` | Sliding-window step |
| `weight` | nested list / `None` | `None` | Optional custom kernel replacing the uniform ones-kernel (should match `local_size`) |
| `output` | tensor / container | `None` | Pre-allocated output container |
| `output_start` | tuple of ints | all zeros | Start position in the output container |
| `output_step` | tuple of ints | all ones | Step size for writing to the output |
| `**kwarg` | — | — | Extra arguments forwarded to `cos_comparison_active` (e.g. `start`, `end`, callbacks) |

> The window is filled with ones (uniform kernel) unless `weight` is given.

### Returns

- **Type**: `vector_map_as_tensor`
- **Description**: Local mean values at each position
- **Output size per dimension**:
  ```
  output_size = ((data_size - local_size) // step) + 1
  ```
- **Raises**: `ValueError("effectless args.")` if the window does not fit

### Implementation

`mean_local` is implemented using active mode with an all-ones kernel and a custom algorithm:

```python
algorithm = lambda a, b, ab, name: ab / N
```

Where `N` is the number of elements in the window.

This works because, with an all-ones kernel, `a` = sum of squares of the data window, `b` = sum of squares of the kernel (= N), and `ab` = dot product = sum of the data window — so `ab / N` is the window mean.

### Examples

#### 1D Moving Average

```python
from cos_comparison import core as cc

data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
result = cc.mean_local_1d(
    data,
    local_size=(3,),
    step=(1,)
)
# Result: [2.0, 3.0, 4.0, 5.0]
# (1+2+3)/3=2, (2+3+4)/3=3, etc.
```

#### 2D Average Pooling

```python
image = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16],
]

# 2x2 average pooling with stride 2
pooled = cc.mean_local_2d(
    image,
    local_size=(2, 2),
    step=(2, 2)
)
# Result:
# [
#   [3.5, 5.5],   # (1+2+5+6)/4=3.5, (3+4+7+8)/4=5.5
#   [11.5, 13.5]  # (9+10+13+14)/4=11.5, (11+12+15+16)/4=13.5
# ]

# Larger step → downsampling: 3x3 windows with step 2
downsampled = cc.mean_local_2d(image, local_size=(3, 3), step=(2, 2))
```

---

## local_variance

Computes the local variance within a sliding window.

### Function Signature

```python
local_variance(data,
               *arg,
               local_size=None,
               step=None,
               output=None,
               output_start=None,
               output_step=None,
               **kwarg)
```

### Parameters

Same parameters as `mean_local`. See above for details.

### Returns

- **Type**: `vector_map_as_tensor`
- **Description**: Local variance values at each position
- **Formula**: `variance = E[X²] - E[X]²`
  - `E[X]` = mean
  - `E[X²]` = mean of squares

### Implementation

`local_variance` is implemented using active mode with an all-ones kernel and a custom variance algorithm:

```python
def var_alg(a, b, ab, name):
    mean = ab / N
    return a / N - mean * mean
```

Where `a` = sum of squares of the data window (so `a/N` = mean of squares), `ab` = sum of the data window (so `ab/N` = mean) — the standard computational formula `variance = E[X²] − E[X]²`.

### Examples

#### 1D Local Variance

```python
from cos_comparison import core as cc

data = [1.0, 2.0, 3.0, 4.0, 5.0]
result = cc.local_variance_1d(
    data,
    local_size=(3,),
    step=(1,)
)
# Window [1,2,3]: mean=2, mean_of_squares=(1+4+9)/3≈4.67, var≈0.67
# Windows [2,3,4] and [3,4,5] give the same result (uniform data → constant local variance)
```

#### 2D Texture Analysis and Edge Detection

```python
# High variance = textured region; low variance = uniform region
variance_map = cc.local_variance_2d(image, local_size=(5, 5), step=(1, 1))

# Edges also have high local variance
edges_via_variance = cc.local_variance_2d(image, local_size=(3, 3), step=(1, 1))
```

---

## Dimension Aliases

Both `mean_local` and `local_variance` have dimension-specific aliases — all forwarding to the same generic N-dimensional functions:

```python
mean_local_1d/2d/3d/4d(data, ...)        local_variance_1d/2d/3d/4d(data, ...)
```

---

## Additional Statistics

### Standard Deviation

Standard deviation is not directly provided, but can be computed from variance:

```python
import math

variance = cc.local_variance_2d(image, local_size=(3, 3), step=(1, 1))
stddev = [[math.sqrt(v) for v in row] for row in variance]
```

### Normalization

Local statistics can be used for local contrast normalization:

```python
# (std computed as in the Standard Deviation example above)
mean_map = cc.mean_local_2d(image, local_size=(5, 5), step=(1, 1))
std_map = [[math.sqrt(v) for v in row] for row in
           cc.local_variance_2d(image, local_size=(5, 5), step=(1, 1))]

# normalized = (image - mean) / std
# (requires proper alignment of output sizes)
```

### Tensor Statistics

The `vector_map_as_tensor` type itself provides global statistics on whole tensors or views:

```python
t = cc.load_as_default_data([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
t.mean()      # 5.0  (Welford's online algorithm)
t.variance()  # 6.666666666666667

# Statistics on a view
t[1:3].mean()  # 6.5
```

---

## Notes

### Performance

Since statistics functions are built on active mode, they inherit its performance characteristics, backend acceleration, and callback system.

### Window Types

Currently, only uniform (rectangular) windows are supported via `local_size`. For custom window shapes:

```python
# Use cos_comparison_active directly with a custom kernel
gaussian_kernel = [...]  # your custom kernel
result = cc.cos_comparison_active(
    data,
    kernel=gaussian_kernel,
    algorithm=lambda a, b, ab, name: ab / sum(sum(row) for row in gaussian_kernel)
)
```

Alternatively, pass a custom kernel through the `weight` parameter of `mean_local`.

### Relation to Pooling

- `mean_local` = average pooling; variance pooling is unique to cos-comparison
- No max pooling currently (would require a different algorithm)

### Numerical Considerations

- The `E[X²] - E[X]²` formula can suffer from catastrophic cancellation for small variances (acceptable for most purposes)
- The tensor-level `variance()` method uses Welford's online algorithm, which is numerically stable for large tensors

---

**See also**:
- [Active Mode](active-mode.md) — Underlying active mode implementation
- [Core Module](core.md) — Overview of all core functions