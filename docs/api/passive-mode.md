# Passive Mode API

## Overview

Passive mode computes self-similarity within data by comparing two offset sliding windows, detecting edges, boundaries, and texture variations without any external template. See [Dual Working Modes](../principles/dual-mode.md) for the conceptual background.

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

Most parameters (`data`, `window_size`, `w1`/`w2`, `b1`/`b2`, `start`/`end`/`step`, `algorithm`, `output`/`output_start`/`output_step`, and all callbacks) are shared with `cos_comparison_active`; their full reference is in [Core Module — Common Parameters](core.md#common-parameters).

**Mode-specific notes**:

- `data`: Duck typing — if `data` exposes `__cos_comparison_passive__`, the call is delegated to it (allowing custom tensors to reload the implementation).

#### `d` (passive only)

- **Type**: Tuple of ints
- **Default**: `(1, 0, 0, ...)` (1 in first dimension, 0 elsewhere)
- **Description**: Displacement vector between the two comparison windows
- **Controls**: Direction and distance of comparison
- **Constraints**: Length must match data dimension
- **Example**:
  - `(1, 0)` → Compare with right neighbor (vertical edges in 2D)
  - `(0, 1)` → Compare with bottom neighbor (horizontal edges in 2D)
  - `(1, 1)` → Compare with diagonal neighbor

## Returns

- **Type**: `vector_map_as_tensor` (or whatever `return_callback` returns)
- **Shape**: Determined by input shape, window size, step, and displacement
- **Output size per dimension**:
  ```
  num[i] = ((end[i] - start[i] - window_size[i] - d[i]) // step[i]) + 1
  ```
  The output shape is `((num[i] - 1) * output_step[i] + 1, ...)` — equal to `num` when `output_step` is all ones.
- **Raises**: `ValueError("effectless args.")` if the window plus displacement does not fit (`end - start - window_size - d < 0` in any dimension)

## Dimension Aliases

```python
cos_comparison_passive_1d(data, ...)  # 1D
cos_comparison_passive_2d(data, ...)  # 2D
cos_comparison_passive_3d(data, ...)  # 3D
cos_comparison_passive_4d(data, ...)  # 4D
```

All are aliases for the same generic function. Use whichever makes your code clearer.

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

### 2D Edge Detection (Horizontal and Vertical)

```python
image = [
    [1, 1, 1, 0, 0, 0],
    [1, 1, 1, 0, 0, 0],
    [1, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 1],
    [0, 0, 0, 1, 1, 1],
    [0, 0, 0, 1, 1, 1],
]

# Horizontal edges (compare with bottom neighbor)
horizontal_edges = cc.cos_comparison_passive_2d(
    image, window_size=(3, 3), step=(1, 1), d=(1, 0)
)

# Vertical edges (compare with right neighbor)
vertical_edges = cc.cos_comparison_passive_2d(
    image, window_size=(3, 3), step=(1, 1), d=(0, 1)
)
```

### Multi-Scale Features

```python
for win_size in [(3, 3), (5, 5), (7, 7)]:
    features = cc.cos_comparison_passive_2d(
        image, window_size=win_size, step=(1, 1), d=(1, 0)
    )
    # Process features at this scale...
```

### Using a Custom Algorithm

```python
def custom_similarity(a, b, ab, name):
    import math
    mag_a, mag_b = math.sqrt(a), math.sqrt(b)
    return 1.0 - abs(mag_a - mag_b) / max(mag_a, mag_b) if max(mag_a, mag_b) > 0 else 1.0

result = cc.cos_comparison_passive_2d(
    image, window_size=(3, 3), step=(1, 1), d=(1, 0),
    algorithm=custom_similarity
)
```

### Pre-allocated Output

```python
output = cc.create_void_list((4, 4))  # shape depends on input, window, step
cc.cos_comparison_passive_2d(
    image, window_size=(3, 3), step=(1, 1), d=(1, 0), output=output
)
```

### Capturing Callback State

```python
seen = {}
def on_start(name):
    seen['window_size'] = name.window_size
    seen['linear'] = name.linear   # (w1, w2, b1, b2)
    seen['d'] = name.d

result = cc.cos_comparison_passive_2d(
    image, window_size=(3, 3), step=(1, 1), d=(1, 0),
    start_callback=on_start
)
```

## Notes

### Performance Tips

1. **Window size**: Larger windows = more computation per position
2. **Step size**: Larger step = fewer positions = faster
3. **Local error callback**: Avoid using `local_error_callback` for performance-critical code
4. **Backend choice**: Use C or C-extension backend for large data

### Edge Handling

- Both windows must fit entirely within the data; output is smaller than input by `window_size + d - 1` in each dimension
- No padding is applied by default
- Invalid parameter combinations raise `ValueError("effectless args.")` instead of silently truncating

### Numerical Stability

- All similarity algorithms handle zero vectors gracefully; division by zero is avoided with appropriate checks (see [Similarity Measures](../principles/similarity-measures.md#edge-case-handling))
- Results are always in a valid range

---

**See also**:
- [Active Mode](active-mode.md) — Template matching mode
- [Similarity Measures](../principles/similarity-measures.md) — Algorithm details
- [Core Module](core.md) — Common parameters and types