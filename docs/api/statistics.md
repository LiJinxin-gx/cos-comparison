# Statistics Functions

## Overview

Cos-comparison provides sliding window statistical functions built on top of the active mode infrastructure. These functions compute local statistics (mean, variance) across the data.

## mean_local

Computes the local mean (average) within a sliding window. Equivalent to average pooling.

### Function Signature

```python
mean_local(data,
           *arg,
           local_size=None,
           step=None,
           output=None,
           output_start=None,
           output_step=None,
           **kwarg)
```

### Parameters

#### `data`
- **Type**: Nested list (or compatible tensor type)
- **Required**: Yes
- **Description**: Input data to compute local mean on
- **Dimensions**: 1D, 2D, 3D, 4D, or arbitrary N-dimensional

#### `local_size`
- **Type**: Tuple of ints, int, or `default_contain`
- **Default**: `default_contain(1)`
- **Description**: Size of the local window
- **Tuple**: Different size per dimension, e.g., `(3, 3)` for 2D
- **Int**: Same size for all dimensions (converted internally)
- **Note**: Window is filled with ones (uniform kernel)

#### `step`
- **Type**: Tuple of ints, int, or `default_contain`
- **Default**: `default_contain(1)`
- **Description**: Step size for sliding window
- **Tuple**: Different step per dimension
- **Int**: Same step for all dimensions

#### `output`
- **Type**: Nested list or compatible container
- **Default**: `None` (creates new output)
- **Description**: Pre-allocated output container

#### `output_start`
- **Type**: Tuple of ints
- **Default**: All zeros
- **Description**: Start position in output container

#### `output_step`
- **Type**: Tuple of ints
- **Default**: All ones
- **Description**: Step size for writing to output

#### `**kwarg`
- Additional keyword arguments passed to `cos_comparison_active`
- Supports all active mode parameters: `start`, `end`, `start_callback`, `end_callback`, etc.

### Returns

- **Type**: Nested list
- **Description**: Local mean values at each position
- **Output size per dimension**:
  ```
  output_size = ((data_size - local_size) // step) + 1
  ```

### Implementation

`mean_local` is implemented using active mode with an all-ones kernel and a custom algorithm:

```python
algorithm = lambda a, b, ab, name: ab / N
```

Where `N` is the number of elements in the window.

This works because:
- `a` = sum of squares of data window
- `b` = sum of squares of kernel (which equals N for all-ones kernel)
- `ab` = dot product of data window and kernel (which equals sum of data window)
- `ab / N` = mean of data window

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
```

#### With Stride

```python
# Stride larger than 1 (downsampling)
result = cc.mean_local_2d(
    image,
    local_size=(3, 3),
    step=(2, 2)
)
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

- **Type**: Nested list
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

Where:
- `a` = sum of squares of data window → `a/N` = mean of squares
- `ab` = sum of data window → `ab/N` = mean
- `variance = mean_of_squares - mean²`

This is the standard computational formula for variance.

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
# Window [1,2,3]: mean=2, mean_of_squares=(1+4+9)/3=14/3≈4.67, var=4.67-4=0.67
# Window [2,3,4]: mean=3, mean_of_squares=(4+9+16)/3=29/3≈9.67, var=9.67-9=0.67
# Window [3,4,5]: mean=4, mean_of_squares=(9+16+25)/3=50/3≈16.67, var=16.67-16=0.67
```

#### 2D Texture Analysis

```python
# High variance = textured region
# Low variance = uniform region
variance_map = cc.local_variance_2d(
    image,
    local_size=(5, 5),
    step=(1, 1)
)
```

#### Edge Detection via Variance

```python
# Edges have high local variance
edges_via_variance = cc.local_variance_2d(
    image,
    local_size=(3, 3),
    step=(1, 1)
)
```

---

## Dimension Aliases

Both `mean_local` and `local_variance` have dimension-specific aliases:

### mean_local
```python
mean_local_1d(data, ...)  # 1D
mean_local_2d(data, ...)  # 2D
mean_local_3d(data, ...)  # 3D
mean_local_4d(data, ...)  # 4D
```

### local_variance
```python
local_variance_1d(data, ...)  # 1D
local_variance_2d(data, ...)  # 2D
local_variance_3d(data, ...)  # 3D
local_variance_4d(data, ...)  # 4D
```

These are all aliases for the same generic N-dimensional functions.

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

Local statistics can be used for normalization:

```python
# Local contrast normalization
mean_map = cc.mean_local_2d(image, local_size=(5, 5), step=(1, 1))
std_map = [[math.sqrt(v) for v in row] for row in 
           cc.local_variance_2d(image, local_size=(5, 5), step=(1, 1))]

# normalized = (image - mean) / std
# (requires proper alignment of output sizes)
```

---

## Notes

### Performance

Since statistics functions are built on active mode:
- They inherit all performance characteristics of active mode
- Same backend acceleration applies
- Same callback system is available

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

### Relation to Pooling

- `mean_local` = average pooling
- No max pooling currently (would require different algorithm)
- Variance pooling is unique to cos-comparison

### Numerical Considerations

- Variance computation uses the `E[X²] - E[X]²` formula
- Can suffer from catastrophic cancellation for small variances
- For most practical purposes, this is acceptable
- Welford's algorithm could be used for better numerical stability (future enhancement)

---

**See also**:
- [Active Mode](active-mode.md) — Underlying active mode implementation
- [Core Module](core.md) — Overview of all core functions
