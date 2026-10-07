# Backend Management System

Transparent switching between implementation backends with a unified API.

## Contents

- [Backend Priority](#backend-priority-default)
- [Key Features](#key-features)
- [Loading Mechanism](#loading-mechanism)
- [Configuration](#configuration)
- [Public API](#public-api)
- [API Contract](#api-contract)
- [External Engines (iterate injection, v0.5.0)](#external-engines-iterate-injection-v050)
- [Custom Backends](#custom-backends)
- [Benchmark](#benchmark-v050)

---

## Backend Priority (Default)

| Priority | Call name | Implementation | Speed | Memory | Free-thread |
|----------|-----------|----------------|-------|--------|-------------|
| 1 | `c` (`.cos_comparison_pydll`) | Python C Extension | 28–150× (see Benchmark) | ~5–8 MB | ✅ Full (no GIL) |
| 2 | `py` (`.cos_comparison`) | Pure Python | 1× (reference) | ~22 MB | ✅ Full |

> Speed range from the v0.5.0 benchmark: window aggregation is ~28× the
> reference, callback-bound element-wise paths narrow to ~2× (the Python
> callback dominates on both backends).  Memory based on a 322×424×3 image
> with a 3×3 window.  The ctypes backend was removed in v0.5.0: the compiled
> C extension IS the C backend; the retired `.cos_comparison_c` name stays
> available as a call-name alias.

### When to Use Which

| Scenario | Recommendation |
|----------|----------------|
| Small data | Pure Python (switching overhead may not be worth it) |
| Large data / multi-threaded | C extension (free-threaded no-GIL parallelism) |
| Many small calls | C extension (minimizes Python overhead) |
| External engines (GPU / parallel) | Keep the default backend and inject an iterator (`iterate=`) |

---

## Key Features

- **Automatic fallback** — if a higher-priority backend is unavailable, try the next; pure Python always appended as final fallback
- **Runtime switching** — `set_mode` at any time
- **Unified API** — all backends expose the same functions via `__all__`
- **Hot API injection** — high-frequency functions injected directly into module namespace for zero-overhead access; `__getattr__` fallback for others
- **LSP compliance** — full subclass operation support
- **Enhanced PyBuffer** — zero-copy creation and slice assignment for `array.array`, `bytes`, `memoryview` (double and unsigned char types)
- **SIMD auto-vectorization** — cross-compiler hints for element-wise loops (50–100% improvement)
- **Free-threaded support** — GIL released on compute-heavy operations (Python 3.13+)
- **Robust error handling** — zero division protection; original exceptions preserved

### Memory Efficiency

1. **C Extension** is most memory-efficient — direct C allocation avoids Python object overhead (~1/3 to 1/5 of pure Python)
2. **Window size inversely correlates** with memory — larger windows ⇒ smaller output ⇒ less memory
3. **Algorithm choice** has negligible memory impact — cos/mod/cosmod share the same framework

---

## Loading Mechanism

```
import core → load config.json → build priority list
→ try each backend in order → first success wins
→ all fail: restore previous state, raise ImportError
```

### Hot API + Dynamic Attribute Proxy

High-frequency APIs are injected directly into the module namespace:

```python
from cos_comparison import core as cc
result = cc.cos_comparison_passive(data, ...)  # zero-overhead direct access
```

**Injected APIs:** `create_void_list`, `load_as_default_data`, `infer_shape`, `vector_map_as_tensor`, `vector_chain_compute`, `set_item`, `get_item`, `_cos`, `_mod`, `_cosmod`, `_default_algorithm`, `NaN`.

All other attributes resolve through `__getattr__` (forwards to loaded backend); `__dir__` merges backend attributes for autocompletion.

---

## Configuration

### config.json

Location: `cos_comparison/core/config.json`

The configuration is an **ordered list**: the list order is the import
(priority) order; every entry maps a call name to its import module, and
several entries may point at the same module (that is how legacy call
names stay compatible).

```json
[
    {"name": "c", "module": ".cos_comparison_pydll"},
    {"name": ".cos_comparison_pydll", "module": ".cos_comparison_pydll"},
    {"name": ".cos_comparison_c", "module": ".cos_comparison_pydll"},
    {"name": "py", "module": ".cos_comparison"},
    {"name": ".cos_comparison", "module": ".cos_comparison"}
]
```

| Field | Description |
|-------|-------------|
| `name` | Call name (anything `set_mode` accepts; dot prefix optional) |
| `module` | Import module (package-relative, dot prefix optional) |

**Notes:**
- Imports try the list in order - each module once (the first entry naming
  a module represents it)
- Missing/invalid config silently falls back to built-in defaults
- Pure Python `.cos_comparison` is always appended as final fallback, even
  if not listed

### Built-in Default

If no config file exists: `c` (`.cos_comparison_pydll`) → `py`
(`.cos_comparison`).

---

## Public API

### `get_mode()`

Returns the configured backend call names in priority order (immutable
tuple), one entry per module; the first one that loaded is active.

```python
cc.get_mode()
# ('c', 'py')
```

### `get_available_backends()`

Returns all configured backend call names including aliases and disabled
ones (immutable tuple).

```python
cc.get_available_backends()
# ('c', '.cos_comparison_pydll', '.cos_comparison_c', 'py', '.cos_comparison')
```

### `set_mode(backends)`

Manually specify backend(s), overriding automatic mode. Names may include or omit leading dot.

```python
cc.set_mode("py")                               # single backend
cc.set_mode(["c", "py"])                        # tried in order
```

- `backends`: `str` or `list`/`tuple` of `str`
- **Raises** `TypeError` for bad arguments; `ImportError` if no backend available (previous state restored)

---

## API Contract

All backends must expose via `__all__`:

| Category | Functions |
|----------|-----------|
| Core | `cos_comparison_passive`, `cos_comparison_active`, `cos` (+ `*_1d/2d/3d/4d`), `mean_local`, `local_variance` |
| Utilities | `multiple_chain`, `add_chain`, `create_void_list`, `load_as_default_data`, `load_data`, `infer_shape`, `get_item`/`set_item`, `vector_chain_compute`, `no_done`, `data_filter`, `data_mapping`, `threshold_*` |
| Similarity | `_cos`, `_mod`, `_cosmod`, `_default_algorithm`, `private_dict` |
| Types | `vector_map_as_tensor`, `func_name_space`, `default_contain` |
| Constants | `NaN`, `sqrt` |

---

## External Engines (iterate injection, v0.5.0)

The element-wise functions (`data_mapping`, `data_filter`, `elementwise`,
`position_map`, `elementwise_position`, `pool`, `cos_comparison_passive`,
`cos_comparison_active` and the `threshold_*` / `mean_local` /
`local_variance` wrappers) accept an optional `iterate=` slot - a callable
`engine(kernel, **index_info)` that resolves the position parameters and
hands the index (positional) plus the kernel parameters to
`kernel(index, **params)`.  With `iterate=None` (default) the original
inline skeleton runs (zero serial regression).

`pool` follows the element-wise family model on the C backend: its
`iterate=` path hands a mode-selected C kernel (one window aggregation per
output position) to the engine instead of delegating to the pure Python
reference.  Engines may recognise the canonical reducers - the built-in
`max`/`min`/`sum` callables and `None` (mean) - for device execution and
fall back to per-window host dispatch otherwise.

This is the injection point for external engines (GPU / parallel
frameworks): the engine lives **outside** the project (`core` and
`interface` stay stdlib-only), so no third-party dependency enters the
package.  An OpenCL engine maps the iteration space to work-items and the
shards to work-groups (measured on Intel Arc Graphics, 112 CUs - see
[Benchmark](#benchmark-v050): element-wise mapping 35-605x the C fast path,
passive window aggregation 7.8-257x the C fast path and up to ~7300x the
pure Python reference).  The C extension accepts the
same keyword and delegates that path to the pure Python reference
implementation; its plain fast path is kept otherwise.

The extension point is **backend-agnostic**: an external OpenCL engine
(experiment area, `core_iter_exp/gpu_engine_module.py`) was driven through
`iterate=` on both backends - `{py, c} x {data_mapping, passive, active}`
with and without `transform1`/`transform2` - with identical results and
identical device timing on both (the C delegation costs are negligible).
The transform-aware variant shows how an engine may deepen the protocol
itself: a per-value map commutes with per-read application, so the engine
folds `transform1`/`transform2` into the device buffers on the host.

## Custom Backends

**Requirements:**
1. Implement all API-contract functions with identical signatures
2. Return identical data structures
3. Handle edge cases identically (zero vectors, empty inputs)
4. Define `__all__`

**Naming:** `cos_comparison_<backend_name>` (e.g. `cos_comparison_cuda`, `cos_comparison_opencl`)

**Registration:** add to `config.json` with appropriate priority:

```json
[
    {"name": "cos_comparison_cuda", "module": ".cos_comparison_cuda"},
    {"name": "c", "module": ".cos_comparison_pydll"},
    {"name": "py", "module": ".cos_comparison"}
]
```

## Benchmark (v0.5.0)

**Test environment:** Intel Core Ultra 5 125H (14C/18T, 3.6 GHz) + Intel
Arc Graphics (112 CUs, driver 31.0.101.5382), Windows 11 x64, Python 3.14.6,
numpy 2.5.2, pyopencl 2026.1.4.  Data: float32 square grids, 3×3 window,
d=(1,1); best-of-3 up to 512², single run above.  GPU end-to-end includes
host↔device transfers and buffer setup.  The reference engine (numpy + pyopencl) lives in the experiment area and
is not shipped with the package; it is driven through `iterate=` on both
backends.

### passive window aggregation (ms)

| size | pure Python | C extension | GPU (via py) | GPU (via c) | C/GPU | Python/GPU |
|------|-------------|-------------|--------------|-------------|-------|------------|
| 64² | 62.5 | 2.13 | 0.52 | 0.27 | 7.8× | 121× |
| 256² | 1108 | 39.3 | 0.35 | 0.36 | 108× | 3197× |
| 512² | 4554 | 161 | 0.62 | 0.66 | 244× | 7303× |
| 1024² | 18384 | 647 | 2.82 | 4.23 | 153× | 6521× |
| 2048² | — | 2608 | 9.5 | 10.2 | 257× | — |
| 4096² | — | 10519 | 40.7 | 40.9 | 257× | — |

### data_mapping (element-wise, v*2+1) (ms)

| size | pure Python | C extension | GPU (via py) | GPU (via c) | C/GPU | Python/GPU |
|------|-------------|-------------|--------------|-------------|-------|------------|
| 64² | 7.9 | 3.8 | 0.16 | 0.11 | 35× | 50× |
| 256² | 135 | 64.7 | 0.21 | 0.20 | 329× | 659× |
| 512² | 563 | 272 | 0.63 | 0.56 | 490× | 900× |
| 1024² | 2269 | 1060 | 2.64 | 3.24 | 327× | 858× |
| 2048² | — | 4316 | 9.9 | 9.2 | 470× | — |
| 4096² | — | 17721 | 33.8 | 29.3 | 605× | — |

**Findings:**
- the external engine is backend-agnostic: same results (parity with the CPU
  reference, rtol 1e-5) and same device timing on `py` and `c` — the C
  delegation costs are below measurement noise
- the GPU path wins at every size, including 64² (no crossover: the fixed
  overhead ~0.1-0.3 ms is below the C path's small-data overhead)
- element-wise mapping is callback-bound on the CPU (the C path still calls
  the Python callback per element, hence only ~2× vs the reference); the GPU
  engine moves the whole loop to the device
- with `iterate=None` (default, no GPU) nothing changes: the C fast path and
  the pure Python reference are exactly the pre-v0.5.0 paths

### Measuring your own backend

```python
import time
from cos_comparison import core as cc

data = cc.load_as_default_data([[1.0, 2.0, 3.0] * 100] * 100)

for backend in ["c", "py"]:
    cc.set_mode(backend)
    start = time.perf_counter()
    for _ in range(10):
        cc.cos_comparison_passive_2d(data, window_size=(3, 3), step=(1, 1), d=(1, 0))
    print(f"{backend}: {time.perf_counter() - start:.3f}s")
```

---

**Related:** [Core Module API](../api/core.md) · [Seven-Layer Architecture](seven-layer.md)
