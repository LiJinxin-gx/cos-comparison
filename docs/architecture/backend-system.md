# Backend Management System

Transparent switching between implementation backends with a unified API.

## Contents

- [Backend Priority](#backend-priority-default)
- [Key Features](#key-features)
- [Loading Mechanism](#loading-mechanism)
- [Configuration](#configuration)
- [Public API](#public-api)
- [API Contract](#api-contract)
- [Custom Backends](#custom-backends)

---

## Backend Priority (Default)

| Priority | Backend | Implementation | Speed | Memory | Free-thread |
|----------|---------|----------------|-------|--------|-------------|
| 1 | `.cos_comparison_pydll` | Python C Extension | 100–150× | ~5–8 MB | ✅ Full (no GIL) |
| 2 | `.cos_comparison_c` | C via ctypes | 60–80× | ~8–12 MB | ✅ Full |
| 3 | `.cos_comparison` | Pure Python | 1× (reference) | ~22 MB | ✅ Full |

> Memory based on 322×424×3 image with 3×3 window. C backend memory estimated from working set (`tracemalloc` cannot track C allocations).

### When to Use Which

| Scenario | Recommendation |
|----------|----------------|
| Small data | Pure Python (switching overhead may not be worth it) |
| Medium data | ctypes C (good speedup without compilation) |
| Large data / multi-threaded | C extension (free-threaded no-GIL parallelism) |
| Many small calls | C extension (minimizes Python overhead) |
| Memory-constrained | C extension (lowest memory usage) |

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
2. **ctypes C** has low overhead — minimal Python objects, zero-copy for compatible inputs
3. **Window size inversely correlates** with memory — larger windows ⇒ smaller output ⇒ less memory
4. **Algorithm choice** has negligible memory impact — cos/mod/cosmod share the same framework

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

```json
{
    "backends": [
        {"name": ".cos_comparison_pydll", "enabled": true},
        {"name": ".cos_comparison_c", "enabled": true},
        {"name": ".cos_comparison", "enabled": true}
    ]
}
```

| Field | Description |
|-------|-------------|
| `name` | Backend module (leading dot optional, auto-normalized to relative import) |
| `enabled` | Whether to consider this backend |

**Notes:**
- Missing/invalid config silently falls back to built-in defaults
- Pure Python `.cos_comparison` is always appended as final fallback, even if not listed
- Set ctypes backend `"enabled": false` if prebuilt binaries are unavailable on target platform

### Built-in Default

If no config file exists: `.cos_comparison_pydll` → `.cos_comparison_c` → `.cos_comparison`.

---

## Public API

### `get_mode()`

Returns enabled backends in priority order (immutable tuple). Names include leading dot.

```python
cc.get_mode()
# ('.cos_comparison_pydll', '.cos_comparison_c', '.cos_comparison')
```

### `get_available_backends()`

Returns all configured backends including disabled ones (immutable tuple).

### `set_mode(backends)`

Manually specify backend(s), overriding automatic mode. Names may include or omit leading dot.

```python
cc.set_mode("cos_comparison")                               # single backend
cc.set_mode(["pydll", "c", "cos_comparison"])               # tried in order
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

## Custom Backends

**Requirements:**
1. Implement all API-contract functions with identical signatures
2. Return identical data structures
3. Handle edge cases identically (zero vectors, empty inputs)
4. Define `__all__`

**Naming:** `cos_comparison_<backend_name>` (e.g. `cos_comparison_cuda`, `cos_comparison_opencl`)

**Registration:** add to `config.json` with appropriate priority:

```json
{"backends": [
    {"name": "cos_comparison_cuda", "enabled": true},
    {"name": ".cos_comparison_pydll", "enabled": true},
    {"name": ".cos_comparison", "enabled": true}
]}
```

### Benchmarking Backends

```python
import time
from cos_comparison import core as cc

data = cc.load_as_default_data([[1.0, 2.0, 3.0] * 100] * 100)

for backend in ["cos_comparison", "cos_comparison_c", "cos_comparison_pydll"]:
    try:
        cc.set_mode(backend)
        start = time.time()
        for _ in range(10):
            cc.cos_comparison_passive_2d(data, window_size=(3,3), step=(1,1), d=(1,0))
        print(f"{backend}: {time.time() - start:.3f}s")
    except ImportError:
        print(f"{backend}: not available")
```

---

**Related:** [Core Module API](../api/core.md) · [Seven-Layer Architecture](seven-layer.md)
