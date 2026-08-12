# Active Mode API

## Overview

Active mode performs template matching by sliding an external kernel across the data, computing similarity between each local window and the kernel. Use this mode to find occurrences of a specific pattern. See [Dual Working Modes](../principles/dual-mode.md) for the conceptual background.

## Function Signature

```python
cos_comparison_active(data,
                      *arg,
                      kernel=None,
                      w1=1, w2=1,
                      b1=0, b2=0,
                      start=None, end=None,
                      step=None,
                      algorithm=_default_algorithm,
                      output=None,
                      output_start=None, output_step=None,
                      start_callback=None,
                      end_callback=None,
                      global_error_callback=None,
                      local_error_callback=None,
                      return_callback=lambda output, name: output,
                      **kwargs)
```

## Parameters

Most parameters (`data`, `w1`/`w2`, `b1`/`b2`, `start`/`end`/`step`, `algorithm`, `output`/`output_start`/`output_step`, and all callbacks) are shared with `cos_comparison_passive`; their full reference is in [Core Module — Common Parameters](core.md#common-parameters).

**Mode-specific notes**:

- `data`: Duck typing — if `data` exposes `__cos_comparison_active__`, the call is delegated to it (allowing custom tensors to reload the implementation).

#### `kernel` (active only)

- **Type**: Nested list (or compatible tensor type)
- **Required**: Yes (raises `ValueError` if `None`)
- **Description**: Template/pattern to search for
- **Dimensions**: Must have same number of dimensions as data
- **Determines**: Window size is automatically set from kernel shape
- **Constraints**: Must be regular

## Returns

- **Type**: `vector_map_as_tensor` (or whatever `return_callback` returns)
- **Shape**: Determined by input shape, kernel size, and step
- **Output size per dimension**:
  ```
  num[i] = ((end[i] - start[i] - kernel_size[i]) // step[i]) + 1
  ```
  The output shape is `((num[i] - 1) * output_step[i] + 1, ...)` — equal to `num` when `output_step` is all ones.
- **Values**: Similarity scores between kernel and data window at each position — high values = good match, low values = poor match
- **Raises**: `ValueError("effectless args.")` if the kernel does not fit (`end - start - kernel_size < 0` in any dimension)

## Dimension Aliases

```python
cos_comparison_active_1d(data, kernel, ...)  # 1D
cos_comparison_active_2d(data, kernel, ...)  # 2D
cos_comparison_active_3d(data, kernel, ...)  # 3D
cos_comparison_active_4d(data, kernel, ...)  # 4D
```

All are aliases for the same generic function. Use whichever makes your code clearer.

## Examples

### 1D Pattern Matching

```python
from cos_comparison import core as cc

data = [1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0]
kernel = [3.0, 4.0, 5.0]  # pattern to find

result = cc.cos_comparison_active_1d(data, kernel=kernel, step=(1,))
# Peak in result indicates where the pattern occurs
```

### 2D Template Matching

```python
image = [
    [0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 1, 0],
    [0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0],
]
kernel = [[1, 1], [1, 1]]  # 2x2 square pattern

result = cc.cos_comparison_active_2d(image, kernel=kernel, step=(1, 1))
# High values at positions where the 2x2 square pattern is found
```

### Edge Detection with Directional Kernels

```python
horizontal_kernel = [
    [-1, -1, -1],
    [ 0,  0,  0],
    [ 1,  1,  1]
]
horizontal_edges = cc.cos_comparison_active_2d(
    image, kernel=horizontal_kernel, step=(1, 1), algorithm=cc._cos
)
```

### Shape-Only Matching

```python
# Use _cos when you care about pattern shape but not intensity
result = cc.cos_comparison_active_2d(
    image, kernel=kernel, step=(1, 1), algorithm=cc._cos
)
```

### Multi-Scale Template Matching

```python
for kernel in [small_kernel, medium_kernel, large_kernel]:
    matches = cc.cos_comparison_active_2d(image, kernel=kernel, step=(1, 1))
    # Find peaks in matches...
```

### Thresholding for Detection

```python
threshold = 0.8
matches = [(i, j, result[i][j]) for i in range(len(result))
           for j in range(len(result[0])) if result[i][j] > threshold]
```

### Pre-allocated Output

```python
out_h, out_w = len(image) - len(kernel) + 1, len(image[0]) - len(kernel[0]) + 1
output = cc.create_void_list((out_h, out_w))
cc.cos_comparison_active_2d(image, kernel=kernel, step=(1, 1), output=output)
```

## Kernel Design Tips

- **Normalization**: For best results with cosine similarity, normalize the kernel to unit length:
  ```python
  import math
  total = sum(sum(x*x for x in row) for row in kernel)
  scale = 1.0 / math.sqrt(total) if total > 0 else 1.0
  normalized_kernel = [[x * scale for x in row] for row in kernel]
  ```
- **Size**: Small kernels are fast with fine-grained detection; large kernels are slower but more robust to noise
- **Aspect ratio**: Should match the target pattern's aspect ratio
- **Multi-channel data** (e.g. RGB): process each channel separately and combine, flatten channels into one dimension, or use a 3D kernel (height, width, channels)

## Notes

### Performance Tips

1. **Kernel size**: Larger kernels = more computation per position
2. **Step size**: Larger step = fewer positions = faster
3. **Local error callback**: Avoid using `local_error_callback` for performance-critical code
4. **Backend choice**: Use C or C-extension backend for large data
5. **Kernel reuse**: Matching the same kernel multiple times does not require recreating it

### Edge Handling

- Kernel must fit entirely within the data; output is smaller than input by `kernel_size - 1` in each dimension
- No padding is applied by default
- Invalid parameter combinations raise `ValueError("effectless args.")` instead of silently truncating

### Relation to Convolution

Active mode is similar to convolution/correlation but replaces the dot product with a similarity measure (e.g. cosine, a normalized dot product), making it more robust to intensity variations.

### Numerical Stability

- All similarity algorithms handle zero vectors gracefully; division by zero is avoided with appropriate checks (see [Similarity Measures](../principles/similarity-measures.md#edge-case-handling))
- Results are always in a valid range

---

**See also**:
- [Passive Mode](passive-mode.md) — Self-similarity mode
- [Similarity Measures](../principles/similarity-measures.md) — Algorithm details
- [Statistics Functions](statistics.md) — Mean, variance, etc.
- [Core Module](core.md) — Common parameters and types