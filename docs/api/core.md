# Core Module

## Overview

The `cos_comparison.core` module provides the core functionality of the cos-comparison library. All functions are accessible through this module, regardless of which backend is loaded.

Current version: **0.4.2**

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
| `cos` | Generic N-dimensional full tensor similarity (default algorithm `_cos`) |
| `cos_1d` | 1D full tensor similarity |
| `cos_2d` | 2D full tensor similarity |
| `cos_3d` | 3D full tensor similarity |
| `cos_4d` | 4D full tensor similarity |

All functions flatten both tensors element-wise, accumulate `Σa²`, `Σb²`, `Σab`, then apply the algorithm. Shape mismatch raises `ValueError`.

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
| `private_dict` | Dict mapping algorithm names to functions |

All algorithms share the signature `func(a, b, ab, name)`, where `a`/`b` are the sums of squares of the two windows and `ab` is their dot product.

**See also**: [Similarity Measures](../principles/similarity-measures.md)

### 6. Utility Functions

| Function | Description |
|----------|-------------|
| `multiple_chain(iterable, base=1)` | Product of elements in iterable |
| `add_chain(iterable, base=0)` | Sum of elements in iterable |
| `create_void_list(length_list=(1,), default=0.0)` | Create a tensor filled with a default value |
| `load_as_default_data(data, start=None, shape=None, step=None)` | Load data (or a sub-region with optional sub-sampling) into the default tensor type |
| `load_data(source, target, *, source_start=None, source_step=None, shape=None, target_start=None, target_step=None)` | Bulk-copy a region between two data containers, each side with its own start/step; returns the number of elements copied |
| `infer_shape(data)` | Infer the shape of multi-dimensional data |
| `get_item(object, index)` | Multi-dimensional indexing |
| `set_item(object, index, value)` | Multi-dimensional assignment |
| `vector_chain_compute(A)` | Chained dot-product computation → `(compute, fix, get)` |
| `no_done(*arg, **kwarg)` | No-op placeholder |

**Utility details**:

- `create_void_list` now returns a `vector_map_as_tensor` (not a nested list), with shape `length_list` filled with `default`.
- `load_as_default_data` infers the input shape, validates `start`/`shape` (raises `ValueError` on mismatch or out-of-bounds), and flattens the requested region into a new tensor. Used internally to convert arbitrary nested data.
- `load_data` copies a sub-region from `source` into `target`. `shape` is the upper bound per dimension; the effective size is clipped to whatever the source and target can actually hold, so out-of-bounds requests truncate silently. Each side attempts the PyBuffer protocol first (read and write are probed independently, with a re-export check that the write actually persists); failures fall back to the generic `get_item` / `set_item` loop. Returns the total number of elements actually copied (`0` for an empty effective region). Iterative only, never recursive.
- `infer_shape` resolves shape with priority: PyBuffer protocol → `__shape__` attribute → recursive `len()` detection. Returns `None` if it cannot be inferred.
- `vector_chain_compute(A)` returns three closures: `compute(vector)` returns a tuple of dot products against the rows of `A`; `fix(new)` replaces `A`; `get()` returns the current `A`.
- `get_item` / `set_item` honor the `__get_item__` / `__set_item__` protocol when present, otherwise fall back to normal indexing — this allows custom tensors to hook into the core loops.

### 7. Backend Management

| Function | Description |
|----------|-------------|
| `get_mode` | Get enabled backends in priority order (immutable tuple) |
| `get_available_backends` | Get all configured backends, including disabled ones |
| `set_mode` | Force specific backend(s) to use |

**See also**: [Backend Management System](../architecture/backend-system.md)

### 8. Constants

| Symbol | Description |
|--------|-------------|
| `NaN` | `float("nan")`, used as a sentinel value |
| `sqrt` | `math.sqrt` |

## Common Parameters

These parameters appear across multiple functions.

### Data Parameters

- **`data`**: Input data (nested list or compatible tensor type)
  - Can be 1D, 2D, 3D, 4D, or arbitrary N-dimensional
  - Must be regular (all sublists at same level have same length)
  - Duck-typed tensor objects with `__get_item__` are supported directly

### Window / Kernel Parameters

- **`window_size`**: Size of comparison window (tuple of ints)
  - Length must match data dimension
  - Default: all ones
  - Example: `(3, 3)` for 2D, `(5,)` for 1D

- **`kernel`**: Template kernel for active mode
  - Same dimensionality as data
  - Determines the window size automatically
  - Required in active mode (`ValueError` if `None`)

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

- **`d`** (passive only): Displacement vector between the two comparison windows
  - Default: `(1, 0, 0, ...)` — 1 in the first dimension, 0 elsewhere
  - Example: `(0, 1)` compares with the bottom neighbor in 2D

### Algorithm Parameter

- **`algorithm`**: Similarity algorithm function
  - Default: `_default_algorithm` (= `_cosmod`)
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
  - Default: `None` (creates a new tensor via `create_void_list`)
  - Allows writing into existing data structures
  - Must have correct shape

- **`output_start`**: Start position in output (tuple of ints)
  - Default: all zeros

- **`output_step`**: Step size in output (tuple of ints)
  - Default: all ones

### Callback Parameters

- **`start_callback`**: Called before computation starts — `def callback(name_space): ...`
- **`end_callback`**: Called after computation finishes — `def callback(name_space): ...`
- **`global_error_callback`**: Called on outer-loop errors — `def callback(error, name_space): ...`
- **`local_error_callback`**: Called on inner-loop errors — `def callback(error, name_space): ...`; note: may impact performance significantly
- **`return_callback`**: Called to wrap the return value — `def callback(output, name_space): return wrapped_output`; default returns output directly

## Data Types

### vector_map_as_tensor

A view class that maps a flat vector as a multi-dimensional tensor using a **stride + offset** architecture.

**Flat index formula**:

```
flat_idx = start + offset + Σ strides[k] * (start_offset[k] + i_k * step_offset[k])
```

**Key features**:
- Zero-copy slicing: every slice returns a view sharing the underlying vector
- NumPy-like fancy indexing: arbitrary int/slice mixes, negative indices, arbitrary step sizes, automatic dimension collapse on integer indexing
- Iterative carry-based traversal — no recursion, safe for high-dimensional tensors
- `__shape__` protocol for zero-overhead shape inference
- Arithmetic: `+`, `-`, `*`, `/`, `**`, in-place variants, unary `+`/`-`, `abs` (Frobenius norm).  In-place variants accept scalars on every backend (including the C extension).  All arithmetic is computed in double precision: results are always float-valued tensors — integer-type promotion is intentionally NOT implemented to keep the C hot paths single-typed and SIMD-vectorizable
- Statistics: `mean()` and `variance()` using Welford's online algorithm (numerically stable)
- Buffer-protocol assignment in `__setitem__` (array.array, memoryview, bytes, ...)
- Buffer-protocol export (`memoryview(tensor)`): read-only, C-order, format `d`/`B`; the C extension exports zero-copy views (including strided sub-views), the Python backends export a materialized contiguous snapshot — all backends are read-only and reject writes with TypeError (matching numpy's `frombuffer` on immutable input)
- Buffer-backed construction write-through: tensors built from a writable buffer (`array.array`, `bytearray`, `memoryview`) share storage on every backend — writing to the tensor is visible in the original buffer and vice versa (zero-copy on the C extension)
- Subclass-friendly: operations and slicing return the actual instance type

**Constructor**:
```python
vector_map_as_tensor(*, vector=(1,), shape=(1,), start=0, strides=None,
                     offset=0, start_offset=None, step_offset=None)
```
- `vector=None` (or omitted) with `shape=` **auto-creates the default zero-filled flat vector** sized to the shape (a list of zeros on the Python backends, the native zero-filled array on the C backends).  Omitted `vector` (not passed) keeps the historical default `(1,)` with value `1.0`.
- `vector` may be a sequence, a number, a `vector_map_as_tensor` (creates a view), or a buffer object (`array.array`, `bytearray`, `memoryview`, numpy arrays).

**Usage**:
```python
vec = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
tensor = cc.vector_map_as_tensor(vector=vec, shape=(2, 3))
# tensor[0]       → first row view (shape (3,))
# tensor[0, 1]    → 2.0
# tensor[:, ::-1] → reversed-columns view, still zero-copy
# tensor[1, 2]    → 6.0

# auto-created zero tensor
zeros = cc.vector_map_as_tensor(vector=None, shape=(2, 3))   # all 0.0

# buffer-backed write-through
import array
buf = array.array('d', [1.0, 2.0, 3.0, 4.0])
t = cc.vector_map_as_tensor(vector=buf, shape=(4,))
t[0] = 99.0      # buf[0] becomes 99.0 too (shared storage)
```

**Properties**: `vector`, `shape`, `strides`, `start`, `offset`, `start_offset`, `step_offset`, plus `__shape__`, `dimension`, `tensor_size` (backward-compatible alias of `shape`).

### func_name_space

A simple namespace container for function parameters passed to callbacks and custom algorithms.

**Slots**:
`output`, `output_start`, `output_step`, `window_size`, `kernel`, `linear`, `start`, `end`, `d`, `step`, `algorithm`, `num`

### default_contain

A container that returns a default value for missing keys; used for uniform (per-dimension) parameters such as `local_size` and `step` in the statistics functions.

**Usage**:
```python
container = cc.default_contain(default_value=1.0)
container[5]  # returns 1.0 (default)
container[5] = 2.0  # via default_dict
container[5]  # returns 2.0
```

## Version

```python
import cos_comparison
cos_comparison.__version__
# "0.4.2"
```

> Note: `cos_comparison.core` itself does not define `__version__`; read the version from the top-level package instead.

---

**Next**: Choose a specific API reference:
- [Passive Mode](passive-mode.md)
- [Active Mode](active-mode.md)
- [Statistics Functions](statistics.md)
- [Cognitive Layer APIs](cognitive-layers.md)