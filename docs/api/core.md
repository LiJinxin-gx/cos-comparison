# Core Module

`cos_comparison.core` provides all core functionality regardless of backend. Current version: **0.4.3**

```python
from cos_comparison import core as cc
```

---

## Function Categories

### Passive / Active Modes

| Function | Description |
|----------|-------------|
| `cos_comparison_passive` | Generic N-dimensional passive self-similarity |
| `cos_comparison_passive_1d/2d/3d/4d` | Dimension aliases |
| `cos_comparison_active` | Generic N-dimensional active template matching |
| `cos_comparison_active_1d/2d/3d/4d` | Dimension aliases |

→ [Passive Mode](passive-mode.md) · [Active Mode](active-mode.md)

### Full Tensor Similarity

| Function | Description |
|----------|-------------|
| `cos` | Full tensor similarity (default `_cos`); `cos_1d/2d/3d/4d` aliases |

Flattens both tensors, accumulates `Σa²`, `Σb²`, `Σab`, applies algorithm. Shape mismatch raises `ValueError`.

### Statistics

| Function | Description |
|----------|-------------|
| `mean_local` | Local mean (average pooling); `mean_local_1d/2d/3d/4d` |
| `local_variance` | Local variance; `local_variance_1d/2d/3d/4d` |

→ [Statistics API](statistics.md)

### Similarity Algorithms

| Function | Description |
|----------|-------------|
| `_cos` | Cosine similarity (direction only) |
| `_mod` | Magnitude similarity (amplitude only) |
| `_cosmod` | Combined similarity (default) |
| `_convolution` | Linear correlation: returns `ab` (window–kernel dot product) |
| `_default_algorithm` | Default (`_cosmod`) |
| `private_dict` | Dict mapping algorithm names to functions (same on all backends) |

All share signature `func(a, b, ab, name)` where `a`/`b` are sums of squares, `ab` is dot product.

→ [Similarity Measures](../principles/similarity-measures.md)

### Utility Functions

| Function | Description |
|----------|-------------|
| `multiple_chain(iterable, base=1)` | Product of elements |
| `add_chain(iterable, base=0)` | Sum of elements |
| `create_void_list(length_list=(1,), default=0.0)` | Create tensor filled with default value |
| `load_as_default_data(data, start=None, shape=None, step=None)` | Load data/sub-region into default tensor type |
| `load_data(source, target, *, source_start=None, source_step=None, shape=None, target_start=None, target_step=None)` | Bulk-copy region between containers; returns elements copied |
| `infer_shape(data)` | Infer shape: PyBuffer → `__shape__` → iterative `len()`; `None` if uninferable |
| `get_item(obj, index)` / `set_item(obj, index, value)` | Multi-dimensional indexing (protocol-aware) |
| `data_filter(data, callback, *, start=None, shape=None, step=None, origin=None, basis=None)` | Yield positions whose `callback(value)` is truthy |
| `data_mapping(data, callback, *, start=None, shape=None, step=None, out=None, out_start=None, out_step=None)` | Map sampled elements through callback into output |
| `threshold_filter(data, low=None, high=None, *, inclusive=(True,True), **region)` | `data_filter` over interval `[low, high]` |
| `threshold_map(data, pairs, *, default_value=0.0, **region)` | First truthy `(func, value)` pair selects value; `default_value` fallback |
| `threshold_judge(low=None, high=None, *, inclusive=(True,True))` | Judge factory: `1` in range / `0` out; pairs with `threshold_map` |
| `vector_chain_compute(A)` | Chained dot products → `(compute, fix, get)` closures |
| `no_done(*arg, **kwarg)` | No-op placeholder |

**Utility details:**

- **`create_void_list`** returns a `vector_map_as_tensor` with shape `length_list` filled with `default`.
- **`load_as_default_data`** infers shape, validates `start`/`shape` (raises `ValueError` on mismatch or out-of-bounds), flattens the requested region into a new tensor.
- **`load_data`** copies a sub-region from `source` into `target`. `shape` is the upper bound per dimension; effective size clipped to what both containers can hold (out-of-bounds silently truncated). Each side attempts PyBuffer first (read/write probed independently with persistence re-export check); falls back to `get_item`/`set_item` loop. Returns total elements actually copied. Iterative only.
- **`infer_shape`** priority: PyBuffer protocol → `__shape__` attribute → iterative `len()` detection.
- **`vector_chain_compute(A)`** returns `compute(vector)` (tuple of dot products against rows of `A`), `fix(new)` (replaces `A`), `get()` (returns current `A`).
- **`get_item`/`set_item`** honor `__get_item__`/`__set_item__` protocol when present, otherwise normal indexing — allows custom tensors to hook into core loops.
- **`data_filter`** walks a sampled read region and yields the multi-dimensional position of every element whose `callback(value)` is truthy. Reported position = `origin + basis * local` (defaults: global read position). Callbacks stateless (value only); errors silently skipped. Iterative odometer walk.
- **`data_mapping`** applies `callback(value)` to every sampled element, writes result to output at `out_start + out_step * local` (out-of-bounds clipped). `out` pre-allocated or default-allocated (fresh tensor shaped like read region); returns output. Errors silently skipped.
- **`threshold_filter`/`threshold_map`** instantiate the above. `threshold_map` iterates `(func, value)` pairs per element — first truthy selects its value, else `default_value`. `threshold_judge` returns a judge function (`1` in range / `0` out) designed to pair with `threshold_map`, e.g. `threshold_map(data, [(threshold_judge(low=3, high=7), 2.0), (lambda v: True, 3.0)])`.

### Backend Management

| Function | Description |
|----------|-------------|
| `get_mode()` | Enabled backends in priority order (immutable tuple) |
| `get_available_backends()` | All configured backends including disabled |
| `set_mode(backend)` | Force specific backend(s) |

> `cos_comparison.core` hot-injects the full public API of the active
> backend into the module namespace (zero-overhead access); `__all__`
> covers module APIs plus every backend export, including private but
> exported names such as `_cos`/`_mod`/`_cosmod`.

→ [Backend Management System](../architecture/backend-system.md)

### Constants

| Symbol | Value |
|--------|-------|
| `NaN` | `float("nan")` sentinel |
| `sqrt` | `math.sqrt` |

---

## Common Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | nested list / tensor | required | Input; 1D–ND; must be regular; duck-typed `__get_item__` supported |
| `window_size` | tuple of ints | all ones | Comparison window size; length matches data dimension |
| `kernel` | nested list / tensor | required (active) | Template; same dimensionality as data; determines window size; `ValueError` if `None` |
| `start` / `end` | tuple of ints | 0 / full shape | Computation region; `end` is exclusive |
| `step` | tuple of ints | all ones | Sliding-window step; larger = fewer points = faster |
| `d` | tuple of ints | `(1,0,0,...)` | Passive only: displacement between comparison windows |
| `algorithm` | function | `_cosmod` | Similarity function; custom signature `def algo(a, b, ab, name)` |
| `w1`/`w2`/`b1`/`b2` | number | 1/1/0/0 | Linear transform: `w*value + b` for each comparison region |
| `output` | tensor | `None` | Pre-allocated output container (creates new if `None`) |
| `output_start` / `output_step` | tuple of ints | 0 / 1 | Output region placement |
| `start_callback` | function | `None` | Called before computation: `callback(name_space)` |
| `end_callback` | function | `None` | Called after computation: `callback(name_space)` |
| `global_error_callback` | function | `None` | Called on outer-loop errors: `callback(error, name_space)` |
| `local_error_callback` | function | `None` | Called on inner-loop errors: `callback(error, name_space)`; may impact performance |
| `return_callback` | function | identity | Wraps return value: `callback(output, name_space) → wrapped` |

---

## Data Types

### vector_map_as_tensor

View class mapping a flat vector as a multi-dimensional tensor via **stride + offset** architecture.

**Flat index:** `start + offset + Σ strides[k] * (start_offset[k] + i_k * step_offset[k])`

**Key features:**

- **Zero-copy slicing** — every slice returns a view sharing the underlying vector
- **NumPy-like fancy indexing** — arbitrary int/slice mixes, negative indices, arbitrary steps, dimension collapse on integer indexing
- **Iterative carry-based traversal** — no recursion, safe for high-dimensional tensors
- **`__shape__` protocol** — zero-overhead shape inference
- **Arithmetic** — `+`, `-`, `*`, `/`, `**`, in-place variants, unary `+`/`-`, `abs` (Frobenius norm). All in double precision; integer promotion intentionally not implemented (keeps C hot paths single-typed and SIMD-vectorizable)
- **Statistics** — `mean()` / `variance()` via Welford's online algorithm (numerically stable)
- **Buffer protocol** — `memoryview(tensor)` read-only, C-order, format `d`/`B`; C extension exports zero-copy strided views, Python backends export materialized snapshots
- **Buffer-backed write-through** — tensors from writable buffers (`array.array`, `bytearray`, `memoryview`) share storage on all backends
- **Subclass-friendly** — operations and slicing return the actual instance type

**Constructor:**

```python
vector_map_as_tensor(*, vector=(1,), shape=(1,), start=0, strides=None,
                     offset=0, start_offset=None, step_offset=None)
```

- `vector=None` with `shape=` auto-creates a zero-filled flat vector sized to the shape
- Omitted `vector` keeps historical default `(1,)` (value `1.0`)
- `vector` may be sequence, number, tensor (creates view), or buffer object

**Usage:**

```python
vec = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
tensor = cc.vector_map_as_tensor(vector=vec, shape=(2, 3))
tensor[0, 1]       # 2.0
tensor[:, ::-1]    # reversed-columns view (zero-copy)

zeros = cc.vector_map_as_tensor(vector=None, shape=(2, 3))  # all 0.0

# buffer-backed write-through
import array
buf = array.array('d', [1.0, 2.0, 3.0, 4.0])
t = cc.vector_map_as_tensor(vector=buf, shape=(4,))
t[0] = 99.0       # buf[0] becomes 99.0 (shared storage)
```

**Properties:** `vector`, `shape`, `strides`, `start`, `offset`, `start_offset`, `step_offset`, `__shape__`, `dimension`, `tensor_size`

### func_name_space

Namespace container for function parameters passed to callbacks and custom algorithms.

**Slots:** `output`, `output_start`, `output_step`, `window_size`, `kernel`, `linear`, `start`, `end`, `d`, `step`, `algorithm`, `num`

### default_contain

Container returning a default value for missing keys; used for uniform per-dimension parameters (e.g. `local_size`, `step` in statistics).

```python
c = cc.default_contain(default_value=1.0)
c[5]        # 1.0 (default)
c[5] = 2.0  # via default_dict
c[5]        # 2.0
```

---

## Version

```python
import cos_comparison
cos_comparison.__version__  # "0.4.3"
```

> `cos_comparison.core` does not define `__version__`; read from the top-level package.

---

**Next:** [Passive Mode](passive-mode.md) · [Active Mode](active-mode.md) · [Statistics](statistics.md) · [Cognitive Layers](cognitive-layers.md)
