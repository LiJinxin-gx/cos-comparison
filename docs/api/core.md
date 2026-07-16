# Core Module

## Overview

The `cos_comparison.core` module provides the core functionality of the cos-comparison library. All functions are accessible through this module, regardless of which backend is loaded.

## Import

```python
from cos_comparison import core as cc
```

## Function Categories

### 1. Passive Mode Functions

Self-similarity comparison within data.

| Function | Description |
|----------|-------------|
| `cos_comparison_passive` | Generic N-dimensional passive mode |
| `cos_comparison_passive_1d` | 1D passive mode (alias) |
| `cos_comparison_passive_2d` | 2D passive mode (alias) |
| `cos_comparison_passive_3d` | 3D passive mode (alias) |
| `cos_comparison_passive_4d` | 4D passive mode (alias) |

**See also**: [Passive Mode API](passive-mode.md)

### 2. Active Mode Functions

Template matching with external kernel.

| Function | Description |
|----------|-------------|
| `cos_comparison_active` | Generic N-dimensional active mode |
| `cos_comparison_active_1d` | 1D active mode (alias) |
| `cos_comparison_active_2d` | 2D active mode (alias) |
| `cos_comparison_active_3d` | 3D active mode (alias) |
| `cos_comparison_active_4d` | 4D active mode (alias) |

**See also**: [Active Mode API](active-mode.md)

### 3. Full Tensor Similarity

Compute similarity between two complete tensors.

| Function | Description |
|----------|-------------|
| `cos` | Generic N-dimensional full tensor similarity |
| `cos_1d` | 1D cosine similarity |
| `cos_2d` | 2D cosine similarity |
| `cos_3d` | 3D cosine similarity |
| `cos_4d` | 4D cosine similarity |

### 4. Statistics Functions

Sliding window statistical operations.

| Function | Description |
|----------|-------------|
| `mean_local` | Local mean (average pooling) |
| `mean_local_1d` | 1D local mean (alias) |
| `mean_local_2d` | 2D local mean (alias) |
| `mean_local_3d` | 3D local mean (alias) |
| `mean_local_4d` | 4D local mean (alias) |
| `local_variance` | Local variance |
| `local_variance_1d` | 1D local variance (alias) |
| `local_variance_2d` | 2D local variance (alias) |
| `local_variance_3d` | 3D local variance (alias) |
| `local_variance_4d` | 4D local variance (alias) |

**See also**: [Statistics API](statistics.md)

### 5. Similarity Algorithms

| Function | Description |
|----------|-------------|
| `_cos` | Cosine similarity (direction only) |
| `_mod` | Magnitude similarity (amplitude only) |
| `_cosmod` | Combined similarity (default) |
| `_default_algorithm` | Default algorithm (`_cosmod`) |

**See also**: [Similarity Measures](../principles/similarity-measures.md)

### 6. Utility Functions

| Function | Description |
|----------|-------------|
| `multiple_chain` | Product of elements in iterable |
| `add_chain` | Sum of elements in iterable |
| `create_void_list` | Create nested empty list |
| `get_item` | Multi-dimensional indexing |

### 7. Backend Management

| Function | Description |
|----------|-------------|
| `get_mode` | Get current backend priority list |
| `set_mode` | Set backend(s) to use |

**See also**: [Backend Management System](../architecture/backend-system.md)

## Common Parameters

These parameters appear across multiple functions.

### Data Parameters

- **`data`**: Input data (nested list or compatible tensor type)
  - Can be 1D, 2D, 3D, 4D, or arbitrary N-dimensional
  - Must be regular (all sublists at same level have same length)

### Window / Kernel Parameters

- **`window_size`**: Size of comparison window (tuple of ints)
  - Length must match data dimension
  - Example: `(3, 3)` for 2D, `(5,)` for 1D

- **`kernel`**: Template kernel for active mode
  - Same dimensionality as data
  - Determines the window size automatically

### Position and Step Parameters

- **`start`**: Start position for computation (tuple of ints)
  - Default: all zeros
  - Must have same length as data dimension

- **`end`**: End position for computation (tuple of ints)
  - Default: full data shape
  - Exclusive boundary

- **`step`**: Step size for sliding window (tuple of ints)
  - Default: all ones
  - Larger step = fewer output points = faster computation

### Algorithm Parameter

- **`algorithm`**: Similarity algorithm function
  - Default: `_cosmod`
  - Options: `_cos`, `_mod`, `_cosmod`, or custom function
  - Custom function signature: `def algo(a, b, ab, name): return similarity`

### Linear Transform Parameters

- **`w1`, `w2`**: Linear weights for two comparison regions
  - Default: `w1=1`, `w2=1`
  - Applied as: `w1 * value + b1` and `w2 * value + b2`

- **`b1`, `b2`**: Linear biases for two comparison regions
  - Default: `b1=0`, `b2=0`

### Output Parameters

- **`output`**: Pre-allocated output container
  - Default: `None` (creates new output)
  - Allows writing into existing data structures
  - Must have correct shape

- **`output_start`**: Start position in output (tuple of ints)
  - Default: all zeros

- **`output_step`**: Step size in output (tuple of ints)
  - Default: all ones

### Callback Parameters

- **`start_callback`**: Called before computation starts
  - Signature: `def callback(name_space): ...`

- **`end_callback`**: Called after computation finishes
  - Signature: `def callback(name_space): ...`

- **`global_error_callback`**: Called on outer-loop errors
  - Signature: `def callback(error, name_space): ...`

- **`local_error_callback`**: Called on inner-loop errors
  - Signature: `def callback(error, name_space): ...`
  - Note: May impact performance significantly

- **`return_callback`**: Called to wrap return value
  - Signature: `def callback(output, name_space): return wrapped_output`
  - Default: returns output directly

## Data Types

### vector_map_as_tensor

A view class that maps a flat vector as a multi-dimensional tensor.

**Key features**:
- Zero-copy slicing (views share underlying data)
- Supports `__getitem__`, `__setitem__`, `__len__`
- Supports arithmetic operations: `+`, `-`, `*`
- Has `mean()` method

**Usage**:
```python
vec = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
tensor = cc.vector_map_as_tensor(vec, (2, 3))
# tensor[0] → first row view
# tensor[0, 1] → 2.0
```

### func_name_space

A simple namespace container for function parameters.

**Slots**:
`output`, `output_start`, `output_step`, `window_size`, `kernel`, `linear`, `start`, `end`, `d`, `step`, `algorithm`, `num`

### default_contain

A container that returns a default value for missing keys.

**Usage**:
```python
container = cc.default_contain(default_value=1.0)
container[5]  # returns 1.0 (default)
container[5] = 2.0
container[5]  # returns 2.0
```

## Version

```python
cc.__version__
# "0.2.0"
```

---

**Next**: Choose a specific API reference:
- [Passive Mode](passive-mode.md)
- [Active Mode](active-mode.md)
- [Statistics Functions](statistics.md)
