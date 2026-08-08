# Backend Management System

Cos-comparison features a sophisticated **multi-backend management system** that allows transparent switching between different implementation backends while maintaining a unified API.

## Overview

### Backend Priority (Default)

| Priority | Backend | Implementation | Performance | Status | Free-thread Support |
|----------|---------|----------------|-------------|--------|---------------------|
| 1 | `.cos_comparison_pydll` | Python C Extension | 100-150x | ✅ Stable | ✅ Full (no GIL) |
| 2 | `.cos_comparison_c` | C via ctypes | 60-80x | ✅ Stable (core functions) | ✅ Full |
| 3 | `.cos_comparison` | Pure Python | 1x (reference) | ✅ Stable | ✅ Full |

### Key Features

- **Automatic fallback**: If a higher-priority backend is unavailable, automatically try the next one
- **Runtime switching**: Manually switch backends at any time
- **Unified API**: All backends expose the same interface via `__all__`
- **Hot API injection**: High-frequency core functions are injected directly into the module namespace for zero-overhead access; `__getattr__` remains as fallback for all other attributes
- **Zero-dependency guarantee**: Pure Python backend always works and is unconditionally appended as the final fallback
- **LSP compliance**: Full support for subclass operations, follows Liskov Substitution Principle
- **Enhanced PyBuffer support**: Zero-copy creation and slice assignment for buffer protocol objects (array.array, bytes, memoryview), supports both double and unsigned char types
- **SIMD auto-vectorization**: Cross-compiler hints for all element-wise loops, automatic CPU vectorization for 50-100% performance improvement
- **Free-threaded support**: Automatically releases GIL during compute-heavy operations, supports true multi-threaded parallelism on Python 3.13+ free-threaded builds
- **Robust error handling**: Zero division protection for all core algorithms, preserves original exceptions instead of masking them with generic errors

## Architecture

### Loading Mechanism

```
User imports cos_comparison.core
          ↓
    Load config.json
          ↓
  Build backend priority list
          ↓
Try first backend → Success? → Yes → Use it
        ↓ No
Try next backend → ...
        ↓
All failed? → Restore previous state, raise ImportError
```

### Hot API + Dynamic Attribute Proxy

Frequently called APIs are injected into the module namespace directly, so user code never pays the proxy lookup cost:

```python
# User code
from cos_comparison import core as cc
result = cc.cos_comparison_passive(data, ...)
```

High-frequency hot APIs (injected): `create_void_list`, `load_as_default_data`, `infer_shape`, `vector_map_as_tensor`, `vector_chain_compute`, `set_item`, `get_item`, `_cos`, `_mod`, `_cosmod`, `_default_algorithm`, `NaN`.

All other attributes are resolved through `__getattr__`, which forwards the lookup to the loaded backend's attribute dict. `__dir__` merges backend attributes so autocompletion works. Users never need to know which backend is actually running.

## Configuration

### 1. config.json (Highest Priority)

Location: `cos_comparison/core/config.json`

Supports the new key-based format:

```json
{
    "backends": [
        {"name": ".cos_comparison_pydll", "enabled": true},
        {"name": ".cos_comparison_c", "enabled": true},
        {"name": ".cos_comparison", "enabled": true}
    ]
}
```

Also accepts the legacy list format, and lists of plain strings:

```json
[
    {"name": "cos_comparison_pydll", "enabled": true},
    {"name": "cos_comparison_c", "enabled": true},
    {"name": "cos_comparison", "enabled": true}
]
```

Fields:
- `name`: Backend module name (leading dot optional; it is normalized to a relative import automatically)
- `enabled`: Whether to consider this backend

Notes:
- A missing or invalid config file silently falls back to the built-in defaults
- The pure Python backend `.cos_comparison` is always appended as the final fallback, even if not listed
- The ctypes backend (`.cos_comparison_c`) is enabled both in the built-in default and the shipped config.json; revert it to `"enabled": false` if prebuilt binaries are unavailable on a target platform

### 2. Built-in Default (Lowest Priority)

If no config file exists, the built-in priority is used: `.cos_comparison_pydll` → `.cos_comparison_c` → `.cos_comparison`.

## Public API

### `get_mode()`

Returns the enabled backends in priority order (immutable tuple). Note that names include the leading dot (relative import form).

```python
from cos_comparison import core as cc

backends = cc.get_mode()
# Returns: ('.cos_comparison_pydll', '.cos_comparison_c', '.cos_comparison')
```

### `get_available_backends()`

Returns all configured backends, including disabled ones (immutable tuple).

```python
cc.get_available_backends()
# Returns: ('.cos_comparison_pydll', '.cos_comparison_c', '.cos_comparison')
```

### `set_mode(backends)`

Manually specify which backend(s) to use, overriding automatic mode. Names may be passed with or without the leading dot.

```python
# Single backend
cc.set_mode("cos_comparison")

# Multiple backends (tried in order)
cc.set_mode(["cos_comparison_pydll", "cos_comparison_c", "cos_comparison"])
```

**Parameters**:
- `backends`: `str` or `list`/`tuple` of `str` — backend name(s) to try, in priority order

**Raises**:
- `TypeError`: If `backends` is not a string or list/tuple, or any name is not a string
- `ImportError`: If none of the specified backends are available (previous backend state is restored)

## Backend Compatibility

### API Contract

All backends must define an `__all__` list (or expose public names) covering the following:

**Core Functions**:
- `cos_comparison_passive` — Passive mode (N-dimensional)
- `cos_comparison_active` — Active mode (N-dimensional)
- `cos` / `cos_1d`...`cos_4d` — Full tensor similarity
- `mean_local` — Local mean
- `local_variance` — Local variance

**Dimension Aliases**:
- `*_1d`, `*_2d`, `*_3d`, `*_4d` for each core function

**Utility Functions**:
- `multiple_chain` — Product of iterable
- `add_chain` — Sum of iterable
- `create_void_list` — Create tensor filled with a default value
- `load_as_default_data` — Load data / sub-region into the default tensor type
- `infer_shape` — Multi-priority shape inference
- `get_item` / `set_item` — Multi-dimensional indexing
- `vector_chain_compute` — Chained dot-product computation
- `no_done` — No-op placeholder

**Similarity Functions**:
- `_cos` — Cosine similarity
- `_mod` — Magnitude similarity
- `_cosmod` — Combined similarity
- `_default_algorithm` — Default algorithm (`_cosmod`)
- `private_dict` — Name → algorithm mapping

**Types/Classes**:
- `vector_map_as_tensor` — Stride-based tensor view over flat vector
- `func_name_space` — Function namespace container
- `default_contain` — Default value container

**Constants**:
- `NaN` — `float("nan")`
- `sqrt` — math.sqrt

## Performance Characteristics

### Backend Performance Estimates

| Backend | Relative Speed | Memory Usage | Best For |
|---------|---------------|--------------|----------|
| Pure Python | 1x | ~22 MB (baseline) | Reference, debugging, no dependencies |
| C (ctypes) | 50-100x | ~8-12 MB (estimated) | High performance, no Python C API dependency |
| Python C Extension | 100-200x | ~5-8 MB (estimated) | Maximum performance, tight integration, free-threaded parallelism |

> **Note**: Memory measurements are based on a 424×322×3 test image with 3×3 window. C backend memory is estimated from process working set, as `tracemalloc` cannot track C-level allocations.

### Memory Efficiency Analysis

#### Key Findings

1. **C Extension (pydll) is most memory-efficient**
   - Direct C-level memory allocation avoids Python object overhead
   - Estimated memory usage: 1/3 to 1/5 of pure Python

2. **C (ctypes) has low memory overhead**
   - ~8-12 MB estimated working set, significantly lower than pure Python
   - Minimal Python object overhead, most memory allocated in C
   - Supports zero-copy for compatible input types

3. **Window size inversely correlates with memory usage**
   - Larger windows → smaller output → less memory
   - Reason: output dimensions shrink as window grows

4. **Algorithm choice has negligible memory impact**
   - cos / mod / cosmod share the same computational framework

### When Performance Matters

- **Small data**: Pure Python is often fast enough; overhead of switching backends may not be worth it
- **Medium data**: ctypes C backend provides good speedup without compilation requirements
- **Large data / multi-threaded workloads**: Python C extension is recommended, supports free-threaded no-GIL parallelism
- **Many small calls**: C extension minimizes Python overhead
- **Memory-constrained environments**: C extension is preferred for lowest memory usage

## Developing Custom Backends

### Requirements

1. Implement all functions in the API contract
2. Maintain identical function signatures
3. Return identical data structures
4. Handle edge cases identically (zero vectors, empty inputs, etc.)
5. Define `__all__` to declare the public API

### Naming Convention

Backend module names should follow the pattern:
- `cos_comparison_<backend_name>`

Example: `cos_comparison_cuda`, `cos_comparison_opencl`, `cos_comparison_jax`

### Registration

Add your backend to `config.json` with appropriate priority:

```json
{
    "backends": [
        {"name": "cos_comparison_cuda", "enabled": true},
        {"name": "cos_comparison_pydll", "enabled": true},
        {"name": "cos_comparison", "enabled": true}
    ]
}
```

## Advanced Usage

### Forcing Pure Python

Useful for debugging or ensuring reproducibility:

```python
cc.set_mode("cos_comparison")
```

### Benchmarking Different Backends

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
        elapsed = time.time() - start
        print(f"{backend}: {elapsed:.3f}s")
    except ImportError:
        print(f"{backend}: not available")
```

---

**Related**:
- [Core Module API](../api/core.md) — Core function reference
- [Seven-Layer Architecture](seven-layer.md) — Overall architecture