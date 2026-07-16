# Passive Mode API

## Overview

Passive mode computes self-similarity within data by comparing two offset sliding windows. It detects edges, boundaries, and texture variations without requiring any external template.

## Function Signature

```python
cos_comparison_passive(data,
                       *arg,
                       window_size=None,
                       w1=1, w2=1,
                       b1=0, b2=0,
                       start=None, end=None,
                       step=None, d=None,
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
- **Description**: Input data to process
- **Dimensions**: 1D, 2D, 3D, 4D, or arbitrary N-dimensional
- **Constraints**: Must be regular (all sublists at same level have same length)

#### `window_size`
- **Type**: Tuple of ints
- **Default**: `(1, 1, ...)` (all ones)
- **Description**: Size of the comparison window in each dimension
- **Constraints**: Length must match data dimension
- **Example**: `(3, 3)` for 2D, `(5,)` for 1D

#### `d`
- **Type**: Tuple of ints
- **Default**: `(1, 0, 0, ...)` (1 in first dimension, 0 elsewhere)
- **Description**: Displacement vector between the two comparison windows
- **Controls**: Direction and distance of comparison
- **Constraints**: Length must match data dimension
- **Example**:
  - `(1, 0)` → Compare with right neighbor (horizontal edges in 2D)
  - `(0, 1)` → Compare with bottom neighbor (vertical edges in 2D)
  - `(1, 1)` → Compare with diagonal neighbor

### Linear Transform Parameters

#### `w1`, `w2`
- **Type**: float
- **Default**: `w1=1`, `w2=1`
- **Description**: Linear weights for the two windows
- **Formula**: `w1 * value + b1` and `w2 * value + b2`

#### `b1`, `b2`
- **Type**: float
- **Default**: `b1=0`, `b2=0`
- **Description**: Linear biases for the two windows

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
- **Description**: Step size for sliding the window
- **Constraints**: Length must match data dimension
- **Performance note**: Larger step = fewer output points = faster computation

### Algorithm Parameter

#### `algorithm`
- **Type**: Callable
- **Default**: `_cosmod`
- **Description**: Similarity algorithm to use
- **Built-in options**:
  - `_cos`: Cosine similarity (direction only)
  - `_mod`: Magnitude similarity (amplitude only)
  - `_cosmod`: Combined similarity (direction + magnitude, default)
- **Custom function signature**:
  ```python
  def custom_algo(a, b, ab, name):
      # a = sum of squares of window 1
      # b = sum of squares of window 2
      # ab = dot product of windows 1 and 2
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
- **Shape**: Determined by input shape, window size, step, and displacement
- **Output size per dimension**:
  ```
  output_size = ((end - start - window_size - d) // step) + 1
  ```

## Dimension Aliases

For convenience, dimension-specific aliases are provided:

```python
cos_comparison_passive_1d(data, ...)  # 1D
cos_comparison_passive_2d(data, ...)  # 2D
cos_comparison_passive_3d(data, ...)  # 3D
cos_comparison_passive_4d(data, ...)  # 4D
```

These are all aliases for the same generic function. Use whichever makes your code clearer.

## Examples

### 1D Edge Detection

```python
from cos_comparison import core as cc

data = [1.0, 1.0, 1.0, 5.0, 5.0, 5.0]
result = cc.cos_comparison_passive_1d(
    data,
    window_size=(2,),
    step=(1,),
    d=(1,)
)
# Result shows low similarity at the transition from 1.0 to 5.0
```

### 2D Horizontal Edge Detection

```python
image = [
    [1, 1, 1, 0, 0, 0],
    [1, 1, 1, 0, 0, 0],
    [1, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 1],
    [0, 0, 0, 1, 1, 1],
    [0, 0, 0, 1, 1, 1],
]

# Horizontal edges (compare with right neighbor)
horizontal_edges = cc.cos_comparison_passive_2d(
    image,
    window_size=(3, 3),
    step=(1, 1),
    d=(0, 1)  # displacement in y-direction
)
```

### 2D Vertical Edge Detection

```python
# Vertical edges (compare with bottom neighbor)
vertical_edges = cc.cos_comparison_passive_2d(
    image,
    window_size=(3, 3),
    step=(1, 1),
    d=(1, 0)  # displacement in x-direction
)
```

### Multi-Scale Features

```python
# Extract features at multiple scales
for win_size in [(3, 3), (5, 5), (7, 7)]:
    features = cc.cos_comparison_passive_2d(
        image,
        window_size=win_size,
        step=(1, 1),
        d=(1, 0)
    )
    # Process features at this scale...
```

### Using Custom Algorithm

```python
def custom_similarity(a, b, ab, name):
    # Example: absolute difference normalized
    import math
    mag_a = math.sqrt(a)
    mag_b = math.sqrt(b)
    return 1.0 - abs(mag_a - mag_b) / max(mag_a, mag_b) if max(mag_a, mag_b) > 0 else 1.0

result = cc.cos_comparison_passive_2d(
    image,
    window_size=(3, 3),
    step=(1, 1),
    d=(1, 0),
    algorithm=custom_similarity
)
```

### Pre-allocated Output

```python
# Create output container
output_shape = (4, 4)  # depends on input size, window, step
output = cc.create_void_list(output_shape)

# Write results into pre-allocated output
cc.cos_comparison_passive_2d(
    image,
    window_size=(3, 3),
    step=(1, 1),
    d=(1, 0),
    output=output
)
```

## Notes

### Performance Tips

1. **Window size**: Larger windows = more computation per position
2. **Step size**: Larger step = fewer positions = faster
3. **Local error callback**: Avoid using `local_error_callback` for performance-critical code
4. **Backend choice**: Use C or C-extension backend for large data

### Edge Handling

- Windows must fit entirely within the data
- Output is smaller than input by `window_size + d - 1` in each dimension
- No padding is applied by default

### Numerical Stability

- All similarity algorithms handle zero vectors gracefully
- Division by zero is avoided with appropriate checks
- Results are always in a valid range

---

**See also**:
- [Active Mode](active-mode.md) — Template matching mode
- [Similarity Measures](../principles/similarity-measures.md) — Algorithm details
