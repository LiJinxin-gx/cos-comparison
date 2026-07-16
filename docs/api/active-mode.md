# Active Mode API

## Overview

Active mode performs template matching by sliding an external kernel across the data, computing similarity between each local window and the kernel. Use this mode when you want to find occurrences of a specific pattern.

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

### Core Parameters

#### `data`
- **Type**: Nested list (or compatible tensor type)
- **Required**: Yes
- **Description**: Input data to search in
- **Dimensions**: 1D, 2D, 3D, 4D, or arbitrary N-dimensional
- **Constraints**: Must be regular (all sublists at same level have same length)

#### `kernel`
- **Type**: Nested list (or compatible tensor type)
- **Required**: Yes (raises ValueError if None)
- **Description**: Template/pattern to search for
- **Dimensions**: Must have same number of dimensions as data
- **Determines**: Window size is automatically set from kernel shape
- **Constraints**: Must be regular

### Linear Transform Parameters

#### `w1`, `w2`
- **Type**: float
- **Default**: `w1=1`, `w2=1`
- **Description**: Linear weights for data window and kernel
- **Formula**: `w1 * data_value + b1` and `w2 * kernel_value + b2`

#### `b1`, `b2`
- **Type**: float
- **Default**: `b1=0`, `b2=0`
- **Description**: Linear biases for data window and kernel

### Position Parameters

#### `start`
- **Type**: Tuple of ints
- **Default**: All zeros
- **Description**: Start position for the sliding window
- **Constraints**: Length must match data dimension

#### `end`
- **Type**: Tuple of ints
- **Default**: Full data shape
- **Description**: End position (exclusive) for the sliding window
- **Constraints**: Length must match data dimension

#### `step`
- **Type**: Tuple of ints
- **Default**: All ones
- **Description**: Step size for sliding the kernel
- **Constraints**: Length must match data dimension
- **Performance note**: Larger step = fewer output points = faster computation

### Algorithm Parameter

#### `algorithm`
- **Type**: Callable
- **Default**: `_cosmod`
- **Description**: Similarity algorithm to use
- **Built-in options**:
  - `_cos`: Cosine similarity (pattern shape, not intensity)
  - `_mod`: Magnitude similarity (intensity level, not pattern)
  - `_cosmod`: Combined similarity (both shape and intensity, default)
- **Custom function signature**:
  ```python
  def custom_algo(a, b, ab, name):
      # a = sum of squares of data window
      # b = sum of squares of kernel
      # ab = dot product of data window and kernel
      # name = func_name_space object
      return similarity_value
  ```

### Output Parameters

#### `output`
- **Type**: Nested list or compatible container
- **Default**: `None` (creates new output)
- **Description**: Pre-allocated output container to write results into
- **Use case**: Avoid memory allocation, write into existing structure
- **Constraints**: Must have correct shape for the output

#### `output_start`
- **Type**: Tuple of ints
- **Default**: All zeros
- **Description**: Start position in the output container
- **Use case**: Write results at an offset within a larger output

#### `output_step`
- **Type**: Tuple of ints
- **Default**: All ones
- **Description**: Step size for writing to output
- **Use case**: Strided writing into output

### Callback Parameters

#### `start_callback`
- **Type**: Callable or None
- **Default**: `None`
- **Signature**: `def callback(name_space): ...`
- **Description**: Called once before computation starts
- **Use case**: Initialization, timing, logging

#### `end_callback`
- **Type**: Callable or None
- **Default**: `None`
- **Signature**: `def callback(name_space): ...`
- **Description**: Called once after computation finishes
- **Use case**: Cleanup, timing, logging

#### `global_error_callback`
- **Type**: Callable or None
- **Default**: `None`
- **Signature**: `def callback(error, name_space): ...`
- **Description**: Called on errors in the outer loop
- **Use case**: Error handling, logging

#### `local_error_callback`
- **Type**: Callable or None
- **Default**: `None`
- **Signature**: `def callback(error, name_space): ...`
- **Description**: Called on errors in the inner window loop
- **Performance warning**: Called inside the innermost loop, can significantly impact performance
- **Use case**: Debugging, error recovery

#### `return_callback`
- **Type**: Callable
- **Default**: `lambda output, name: output`
- **Signature**: `def callback(output, name_space): return wrapped_output`
- **Description**: Called to wrap/transform the output before returning
- **Use case**: Custom output format, post-processing

## Returns

- **Type**: Nested list (or whatever `return_callback` returns)
- **Shape**: Determined by input shape, kernel size, and step
- **Output size per dimension**:
  ```
  output_size = ((end - start - kernel_size) // step) + 1
  ```
- **Values**: Similarity scores between kernel and data window at each position
  - High values = good match
  - Low values = poor match

## Dimension Aliases

For convenience, dimension-specific aliases are provided:

```python
cos_comparison_active_1d(data, kernel, ...)  # 1D
cos_comparison_active_2d(data, kernel, ...)  # 2D
cos_comparison_active_3d(data, kernel, ...)  # 3D
cos_comparison_active_4d(data, kernel, ...)  # 4D
```

These are all aliases for the same generic function. Use whichever makes your code clearer.

## Examples

### 1D Pattern Matching

```python
from cos_comparison import core as cc

data = [1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0]
kernel = [3.0, 4.0, 5.0]  # pattern to find

result = cc.cos_comparison_active_1d(
    data,
    kernel=kernel,
    step=(1,)
)
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

# 2x2 square pattern to find
kernel = [
    [1, 1],
    [1, 1]
]

result = cc.cos_comparison_active_2d(
    image,
    kernel=kernel,
    step=(1, 1)
)
# High values at positions where the 2x2 square pattern is found
```

### Edge Detection with Directional Kernels

```python
# Horizontal edge kernel (Sobel-like)
horizontal_kernel = [
    [-1, -1, -1],
    [ 0,  0,  0],
    [ 1,  1,  1]
]

horizontal_edges = cc.cos_comparison_active_2d(
    image,
    kernel=horizontal_kernel,
    step=(1, 1),
    algorithm=cc._cos
)
```

### Multi-Scale Template Matching

```python
# Search for pattern at multiple scales
kernels = [
    small_kernel,
    medium_kernel,
    large_kernel
]

for kernel in kernels:
    matches = cc.cos_comparison_active_2d(
        image,
        kernel=kernel,
        step=(1, 1)
    )
    # Find peaks in matches...
```

### Using Cosine Similarity for Shape-Only Matching

```python
# Use _cos when you care about pattern shape but not intensity
result = cc.cos_comparison_active_2d(
    image,
    kernel=kernel,
    step=(1, 1),
    algorithm=cc._cos  # direction only, magnitude doesn't matter
)
```

### Thresholding for Detection

```python
# Find positions where similarity exceeds threshold
threshold = 0.8
matches = []

for i in range(len(result)):
    for j in range(len(result[0])):
        if result[i][j] > threshold:
            matches.append((i, j, result[i][j]))

# matches contains (position_x, position_y, similarity_score) tuples
```

### Pre-allocated Output

```python
# Calculate output shape
data_h, data_w = len(image), len(image[0])
kernel_h, kernel_w = len(kernel), len(kernel[0])
out_h = data_h - kernel_h + 1
out_w = data_w - kernel_w + 1

# Create output container
output = cc.create_void_list((out_h, out_w))

# Write results into pre-allocated output
cc.cos_comparison_active_2d(
    image,
    kernel=kernel,
    step=(1, 1),
    output=output
)
```

## Kernel Design Tips

### Normalization

For best matching results, consider normalizing your kernel:

```python
# Normalize kernel to unit length (for cosine similarity)
import math
total = sum(sum(x*x for x in row) for row in kernel)
scale = 1.0 / math.sqrt(total) if total > 0 else 1.0
normalized_kernel = [[x * scale for x in row] for row in kernel]
```

### Size Considerations

- **Small kernels**: Fast, fine-grained detection
- **Large kernels**: Slower, more robust to noise
- **Aspect ratio**: Should match the target pattern's aspect ratio

### Multi-Channel Data

For multi-channel data (e.g., RGB images), you can:
1. Process each channel separately and combine results
2. Flatten channels into a single dimension
3. Use a 3D kernel (height, width, channels)

## Notes

### Performance Tips

1. **Kernel size**: Larger kernels = more computation per position
2. **Step size**: Larger step = fewer positions = faster
3. **Local error callback**: Avoid using `local_error_callback` for performance-critical code
4. **Backend choice**: Use C or C-extension backend for large data
5. **Kernel reuse**: If matching same kernel multiple times, no need to recreate it

### Edge Handling

- Kernel must fit entirely within the data
- Output is smaller than input by `kernel_size - 1` in each dimension
- No padding is applied by default

### Relation to Convolution

Active mode is similar to convolution/correlation but with important differences:
- **Convolution**: Dot product, sum of products
- **Active mode**: Similarity measure (cosine, etc.)
- **Cosine similarity** = normalized dot product
- Active mode is more robust to intensity variations

### Numerical Stability

- All similarity algorithms handle zero vectors gracefully
- Division by zero is avoided with appropriate checks
- Results are always in a valid range

---

**See also**:
- [Passive Mode](passive-mode.md) — Self-similarity mode
- [Similarity Measures](../principles/similarity-measures.md) — Algorithm details
- [Statistics Functions](statistics.md) — Mean, variance, etc.
