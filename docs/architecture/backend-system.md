# Backend Management System

Cos-comparison features a sophisticated **multi-backend management system** that allows transparent switching between different implementation backends while maintaining a unified API.

## Overview

### Backend Priority (Default)

| Priority | Backend | Implementation | Performance | Status | Free-thread Support |
|----------|---------|----------------|-------------|--------|---------------------|
| 1 | `cos_comparison_pydll` | Python C Extension | 100-200x | ✅ Stable | ✅ Full (no GIL) |
| 2 | `cos_comparison_c` | C via ctypes | 50-100x | 🟡 Experimental | ✅ Full |
| 3 | `cos_comparison` | Pure Python | 1x (reference) | ✅ Stable | ✅ Full |

### Key Features

- **Automatic fallback**: If a higher-priority backend is unavailable, automatically try the next one
- **Runtime switching**: Manually switch backends at any time
- **Unified API**: All backends expose the same interface
- **Configuration flexibility**: Control backend priority via config file or environment variable
- **Zero-dependency guarantee**: Pure Python backend always works

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
All failed? → Raise ImportError
```

### Dynamic Attribute Proxy

The `__getattr__` mechanism provides transparent backend switching:

```python
# User code
from cos_comparison import core as cc
result = cc.cos_comparison_passive(data, ...)

# Internally
# 1. __getattr__ is called with "cos_comparison_passive"
# 2. Look up the attribute in the loaded backend's attribute dict
# 3. Return the backend's implementation
```

Users never need to know which backend is actually running.

## Configuration

### 1. config.json (Highest Priority)

Location: `cos_comparison/core/config.json`

```json
[
    {"name": "cos_comparison_pydll", "enabled": true},
    {"name": "cos_comparison_c", "enabled": true},
    {"name": "cos_comparison", "enabled": true}
]
```

Fields:
- `name`: Backend module name
- `enabled`: Whether to consider this backend

### 2. Environment Variable

```bash
# Unix
export COS_BACKEND=cos_comparison_pydll,cos_comparison

# Windows
set COS_BACKEND=cos_comparison_pydll,cos_comparison
```

Comma-separated list of backend names in priority order.

### 3. Built-in Default (Lowest Priority)

If neither config file nor environment variable is set, the built-in default priority is used.

## Public API

### `get_mode()`

Returns the list of enabled backends in priority order.

```python
from cos_comparison import core as cc

backends = cc.get_mode()
# Returns: ["cos_comparison_pydll", "cos_comparison_c", "cos_comparison"]
```

### `set_mode(backends)`

Manually specify which backend(s) to use, overriding automatic mode.

```python
# Single backend
cc.set_mode("cos_comparison")

# Multiple backends (tried in order)
cc.set_mode(["cos_comparison_c", "cos_comparison"])
```

**Parameters**:
- `backends`: `str` or `list of str` — Backend name(s) to try, in priority order

**Raises**:
- `TypeError`: If backends is not a string or list
- `ValueError`: If any backend name is not in config
- `ImportError`: If none of the specified backends are available

## Backend Compatibility

### API Contract

All backends must implement the following functions:

**Core Functions**:
- `cos_comparison_passive` — Passive mode (N-dimensional)
- `cos_comparison_active` — Active mode (N-dimensional)
- `cos` — Full tensor similarity
- `mean_local` — Local mean
- `local_variance` — Local variance

**Dimension Aliases**:
- `*_1d`, `*_2d`, `*_3d`, `*_4d` for each core function

**Utility Functions**:
- `multiple_chain` — Product of iterable
- `add_chain` — Sum of iterable
- `create_void_list` — Create nested empty list
- `get_item` — Multi-dimensional indexing

**Similarity Functions**:
- `_cos` — Cosine similarity
- `_mod` — Magnitude similarity
- `_cosmod` — Combined similarity
- `_default_algorithm` — Default algorithm

**Types/Classes**:
- `vector_map_as_tensor` — Tensor view over flat vector
- `func_name_space` — Function namespace container
- `default_contain` — Default value container

### Private Attribute Exposure

All attributes (including those starting with `_`) are exposed through the proxy, not just public ones. This allows advanced users to access internal functions regardless of backend.

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
   - Note: `tracemalloc` cannot track C-level allocations; C extension memory is estimated

2. **C (ctypes) has low memory overhead**
   - ~8-12 MB estimated working set, significantly lower than pure Python
   - Minimal Python object overhead, most memory allocated in C
   - Supports zero-copy for compatible input types

3. **Window size inversely correlates with memory usage**
   - Larger windows → smaller output → less memory
   - 3×3 window: highest memory
   - 7×7 window: lowest memory
   - Reason: output dimensions = input dimensions - window dimensions + 1

4. **Algorithm choice has negligible memory impact**
   - cos / mod / cosmod: memory difference < 0.03 MB
   - All three algorithms share the same computational framework

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

### Naming Convention

Backend module names should follow the pattern:
- `cos_comparison_<backend_name>`

Example: `cos_comparison_cuda`, `cos_comparison_opencl`, `cos_comparison_jax`

### Registration

Add your backend to `config.json` with appropriate priority:

```json
[
    {"name": "cos_comparison_cuda", "enabled": true},
    {"name": "cos_comparison_pydll", "enabled": true},
    {"name": "cos_comparison", "enabled": true}
]
```

## Advanced Usage

### Checking Current Backend

```python
# There's no direct "get_current_backend()" function,
# but you can infer from which functions are loaded
# or by checking module attributes
```

### Forcing Pure Python

Useful for debugging or ensuring reproducibility:

```python
cc.set_mode("cos_comparison")
```

### Benchmarking Different Backends

```python
import time
from cos_comparison import core as cc

data = [[1.0, 2.0, 3.0] * 100] * 100

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
