# Cos Comparison

[![PyPI version](https://badge.fury.io/py/cos-comparison.svg)](https://pypi.org/project/cos-comparison/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**An AGI-oriented project based on local similarity comparison for feature extraction – biologically inspired, zero-training edge / pattern detection.**

---

## Core Idea

Information is produced by **local comparison** in raw data.  
This module implements the **centre-surround antagonism** mechanism from neuroscience, extracting edges, textures, and keypoints using only sliding-window similarity.

The core formula (cosine-modulated similarity, recommended default):

$$
\text{cosmod} = \frac{2(A \cdot B)}{|A|^2 + |B|^2}
$$

* **A step toward biologically plausible AGI**
* **No training, no labels, no backpropagation**
* Works on **1D, 2D, 3D, 4D** data (audio, images, video, volumetric data)
* Supports **passive** (reflexive boundary detection) and **active** (template matching) modes
* Three high-performance backends with automatic fallback: Python C extension, ctypes pure C, pure Python
* Cross-platform support for Windows, Linux, and macOS

---

## Seven-Layer Cognitive Architecture

This project follows a biologically inspired seven-layer cognitive architecture, mimicking the structure of the mammalian brain. Currently only the core brainstem/cerebellum layer is production-ready; all other layers are in early evolutionary stage with skeleton implementations, and will be gradually improved in future versions:

| Layer | Directory | Corresponding Brain Structure | Maturity | Core Function |
|-------|-----------|--------------------------------|----------|---------------|
| 1 | `core` | Brainstem / Cerebellum | ✅ Production | Low-level local comparison calculation, three-backend acceleration, free-thread support |
| 2 | `sense_layer` | Sensory Cortex | 🔴 Skeleton | Receive external stimuli, extract raw features |
| 3 | `memory_layer` | Hippocampus / Cerebral Cortex | 🔴 Skeleton | Short-term and long-term memory storage |
| 4 | `brain_layer` | Prefrontal Cortex | 🟡 Early Development | High-level cognition, logical reasoning (symbolic logic system implemented) |
| 5 | `action_layer` | Motor Cortex | 🔴 Skeleton | Control action output, interact with environment |
| 6 | `generate_layer` | Broca's / Wernicke's Area | 🔴 Skeleton | Generate language, images and other high-level outputs |
| 7 | `expend_layer` | Association Cortex | 🔴 Skeleton | Extended functions and special capabilities |

> **Note**: Non-core layers are currently in early development and do not affect the stability of the core feature extraction API. The core `cos_comparison.core` module is fully production-ready and follows semantic versioning guarantees.

---

## Core Principles

1. **Information emerges from local comparison** – edges, textures, and patterns arise by comparing neighboring regions.
2. **Centre-surround antagonism** – two sliding windows are compared with a fixed displacement vector `d`, mimicking the response of retinal ganglion cells.
3. **Three complementary similarity measures** – `_cos` (angular similarity), `_mod` (magnitude similarity), `_cosmod` (recommended balanced combination).
4. **Dual operational modes** – **passive** (boundary detection without templates) and **active** (template matching with a user-supplied kernel).
5. **Multi-scale and multi-directionality** – vary window size for scale selection, change `d` for orientation selectivity (vertical, horizontal, diagonal).
6. **Dimension-agnostic** – the same algorithm runs natively on 1D, 2D, 3D, 4D, and higher-dimensional data.
7. **Zero training, zero labels** – fully deterministic, ready to use out of the box.
8. **Full determinism and interpretability** – every output has a clear geometric meaning.
9. **Modular, pluggable architecture** – pure Python reference implementation with two optional high-performance C backends and automatic fallback.

---

## Design Principles

The library adheres to two key design principles:

* **Zero internal dependencies** – the core is self-contained and does not rely on any third-party modules. It runs everywhere Python runs.
* **Open integration** – a generic hook/interface mechanism is provided, allowing users to seamlessly plug in any external module that conforms to the specified protocols (e.g., GPU/NPU accelerators, custom tensor types, or specialized hardware backends). This keeps the core lightweight while enabling unlimited extensibility.

---

## 🚀 What's New in Version 0.3.5

Version 0.3.5 is an emergency bugfix release addressing critical stability issues:
- **Fixed C extension stack overflow crash**: Resolved a critical stack overflow bug in the C extension that caused crashes with large datasets (500x500+). The bug was introduced during the recursive-to-iterative refactoring, where `alloca()` was incorrectly placed inside a loop, causing unbounded stack growth.
- **Removed ctypes numpy implicit dependency**: The ctypes backend no longer has any implicit numpy dependency, achieving true zero external dependencies for the core package.
- **Memory leak fixes**: Fixed multiple memory leaks in the C extension where temporary Data structures were not properly freed after function calls.
- **Improved stability**: All backends now pass extreme data stress tests (1000x1000+ elements) without crashes.

## 🚀 What's New in Version 0.3.4

Version 0.3.4 brings full free-threaded Python support, complete C API alignment, and packaging improvements.

### Highlights

* **Free-threaded Python (CPython 3.13+) Support** – C extension is now marked as free-thread safe, supports running without the GIL (`-Xgil=0`) for true multi-threaded parallelism
* **Complete C API Alignment** – `vector_map_as_tensor` class in C extension now has 100% method parity with the pure Python implementation:
  * Added `variance()` method matching pure Python behavior
  * Added unary operators: `__neg__` (-x), `__pos__` (+x), `__abs__` (abs(x), returns L2 norm)
  * Fixed division operators to support tensor/tensor division with proper zero-division checks
  * Fixed in-place division to support both scalar and tensor operands
  * Added uint8 image dtype support for zero-copy image processing
* **Packaging Fixes** – sdist source distribution now correctly includes all C source files, automatically compiles C backends on installation
* **ctypes Backend Improvements** – fixed compilation errors on Windows, added variance function to C layer, fixed callback structure type mismatches
* **Critical Bug Fixes** – fixed void pointer arithmetic error in C extension (MSVC compatibility), fixed function signature linkage errors in ctypes backend

### Backend Performance Comparison

| Backend | Relative Speed | Peak Memory | Status | Free-thread Support |
|---------|----------------|-------------|--------|---------------------|
| Python C Extension | 100–200× | ~5–8 MB | ✅ Stable | ✅ Full (no GIL) |
| ctypes C Backend | 50–100× | ~8–12 MB | 🟡 Experimental | ✅ Full |
| Pure Python | 1× | ~22 MB | ✅ Stable | ✅ Full |

> **Note**: Memory measurements are based on a 424×322×3 test image with a 3×3 window. C backend memory is estimated from process working set measurements, as `tracemalloc` cannot track C-level allocations. All C backends automatically fall back to pure Python if compilation or loading fails.

### ⚠️ **Breaking Changes**

**1. NumPy backend removed**  
Introducing a dedicated NumPy backend added unnecessary complexity and potential supply-chain risk. The core now natively supports NumPy arrays (and any type implementing `__index__` and `__len__`, such as PyTorch tensors) via the duck-typing protocol, without requiring a separate backend.

**2. Spelling errors corrected**  
All historical misspellings (e.g., *genrate* → *generate*) have been fixed throughout the codebase and documentation.

---

## ⚠️ Migration Guide from 0.2.x

Starting from version **0.2.0**, the package has been restructured.  
All core functions now reside in the `core` submodule.

**Correct import statement:**

```python
from cos_comparison import core
```

Legacy code using `import cos_comparison as cc` will **not** work with 0.3.0. Please update your imports accordingly.

---

## Installation

```bash
pip install cos-comparison
```

The installer will automatically attempt to compile both C backends during installation. If a C compiler is not available on your system, installation will complete successfully with the pure Python backend only.

To install with optional test dependencies:

```bash
pip install cos-comparison[test]
```

### Manual Compilation

If you need to recompile the C backends after installation (e.g. after changing compiler settings):

```bash
# From the project root directory
python setup.py build_ext --inplace
```

This will build both the Python C extension and the ctypes shared library.

---

## Quick Start

### 1D – detect discontinuities (passive mode)

```python
from cos_comparison import core

data = [1.0, 2.0, 3.0, 4.0, 5.0]
result = core.cos_comparison_passive_1d(
    data,
    window_size=(2,),
    step=(1,),
    d=(1,)
)
print(result)
```

### 2D – edge detection (passive mode)

```python
from cos_comparison import core

image = [
    [1, 1, 0, 0],
    [1, 1, 0, 0],
    [0, 0, 1, 1],
    [0, 0, 1, 1]
]
edges = core.cos_comparison_passive_2d(
    image,
    window_size=(2, 2),
    step=(1, 1),
    d=(1, 0)
)
print(edges)
```

### Active mode – template matching

```python
from cos_comparison import core

data = [1.0, 2.0, 3.0, 4.0, 5.0]
kernel = [1.0, 0.0]
result = core.cos_comparison_active_1d(
    data,
    kernel=kernel,
    step=(1,)
)
print(result)
```

### Whole-tensor cosine similarity

```python
from cos_comparison import core

a = [1, 2, 3]
b = [2, 3, 4]
sim = core.cos_1d(a, b)
print(sim)
```

### Using the recommended `_cosmod` similarity measure

```python
from cos_comparison import core

result = core.cos_comparison_passive_1d(
    data,
    window_size=(2,),
    step=(1,),
    d=(1,),
    algorithm=core._cosmod
)
```

---

## Multi-Backend System

The package provides a sophisticated multi-backend manager that automatically selects the fastest available backend while maintaining a unified API.

### Default Backend Priority

| Priority | Backend | Implementation | Performance | Status |
|----------|---------|----------------|-------------|--------|
| 1 | `cos_comparison_pydll` | Python C Extension | 100–200× | ✅ Stable |
| 2 | `cos_comparison_c` | ctypes Pure C Backend | 50–100× | ✅ Stable |
| 3 | `cos_comparison` | Pure Python | 1× | ✅ Stable |

### Key Features

* **Automatic fallback** – if a higher-priority backend is unavailable, the system automatically falls back to the next option.
* **Runtime switching** – manually switch backends at any time during execution.
* **Unified API** – all backends expose exactly the same interface.
* **Configuration flexibility** – control backend priority via a config file or environment variable.
* **Zero-dependency guarantee** – the pure Python backend always works, no compilation required.

### Switching Backends

```python
from cos_comparison import core

# Check current backend configuration
backends = core.get_available_backends()
print(backends)

# Force pure Python mode (useful for debugging)
core.set_mode("cos_comparison")

# Multiple backends (tried in specified order)
core.set_mode(["cos_comparison_pydll", "cos_comparison_c", "cos_comparison"])

# Enable ctypes backend only
core.set_mode(["cos_comparison_c", "cos_comparison"])
```

### Environment Variable

```bash
# Unix
export COS_BACKEND=cos_comparison_pydll,cos_comparison_c,cos_comparison

# Windows
set COS_BACKEND=cos_comparison_pydll,cos_comparison_c,cos_comparison
```

---

## API Overview

### Passive mode (self-similarity / edge detection)

* `cos_comparison_passive_1d` / `_2d` / `_3d` / `_4d` + generic N-dimensional `cos_comparison_passive`

### Active mode (template matching)

* `cos_comparison_active_1d` / `_2d` / `_3d` / `_4d` + generic N-dimensional `cos_comparison_active`

### Whole-tensor similarity measures

* `cos_1d` / `_2d` / `_3d` / `_4d` (cosine similarity)
* `mod_1d` / `_2d` / `_3d` / `_4d` (magnitude similarity)
* `cosmod_1d` / `_2d` / `_3d` / `_4d` (cosine-modulated similarity, recommended)

### Local statistics (sliding window)

* `mean_local_1d` / `_2d` / `_3d` / `_4d`
* `local_variance_1d` / `_2d` / `_3d` / `_4d`

### Generic helpers

* `cos(A, B, algorithm=core._cos)` – works on arbitrarily nested lists and array-like objects.
* `execute_many(func, arg_iter, kwarg_iter)` – batch execution (returns a list).
* `execute_many_iter(func, arg_iter, kwarg_iter)` – batch execution (lazy generator).

---

## Performance & Memory Benchmarks

Benchmark results for all three backends using a **322×424×3 RGB test image**.  
All tests were run with the same parameter grid: 3 window sizes, 3 offsets, and 3 similarity algorithms (27 combinations total).  
Test environment: Windows 11 x64, 18-thread CPU, Python 3.14.6, JIT enabled, MSVC -O2 optimization.

> **Note**: Memory figures for C backends are **estimated** because `tracemalloc` cannot track C-level allocations. Figures are derived from process-level working set measurements.

---

### ⏱️ Execution Time (27-run average)

| Backend | 3×3 window (s) | 5×5 window (s) | 7×7 window (s) | Total (27 runs) | Speedup vs Python |
|---------|----------------|----------------|----------------|-----------------|-------------------|
| **Pure Python** | 7.31 | 18.80 | 28.00 | 486.8 s | 1× |
| **ctypes C Backend** | 0.142 | 0.358 | 0.682 | 10.9 s | ~45× |
| **Python C Extension** | **0.088** | **0.213** | **0.407** | **6.35 s** | **~77×** |

* Algorithm choice (cos / mod / cosmod) has **negligible impact** (<5%) on execution time.
* Window size is the dominant factor – larger windows increase execution time proportionally.
* The C extension provides approximately **77× speedup** over pure Python for typical workloads.

---

### 🧠 Memory Usage (peak working set)

| Backend | Average peak memory | Notes |
|---------|---------------------|-------|
| **Pure Python** | ~21.5 MB | Stable across all parameter combinations |
| **ctypes C Backend** | ~8–12 MB (estimated) | C-allocated memory, minimal Python overhead |
| **Python C Extension** | ~5–8 MB (estimated) | Tightest integration, lowest memory usage |

* Memory usage is **largely independent** of window size, offset, or algorithm choice.
* The pure Python backend uses a fixed ~21–22 MB for the standard test image.
* The C extension is extremely memory-efficient, allocating only the minimal required output tensor.

---

### ✅ Output Consistency

All three backends produce **numerically identical** output values (within floating-point precision) for every parameter combination – verifying that the core algorithm is correctly implemented across all backends.

---

### 📊 Full Raw Data

Complete CSV results (time and memory for all 27 combinations) are available in the `tests/` directory:

* `test_python.txt`, `test_pydll.txt`, `test_ctypes.txt` – performance timing data
* `memory_python.txt`, `memory_pydll.txt`, `memory_ctypes.txt` – memory usage data
* `out_*.txt` – execution logs with saved heatmap output paths

---

### 🧪 How to Reproduce

```bash
# Performance test (all available backends)
python tests/test_performance.py --image tests/testdata/image_20260408.png --repeat 3 --output results.csv

# Memory test
python tests/test_memory.py --image tests/testdata/image_20260408.png --output memory.csv
```

*Benchmarks run on a standard x86-64 CPU. Your results may vary depending on hardware, compiler optimizations, and operating system.*

---

## Author

I was born on May 31, 2008, and I feel fortunate to grow up in an era of rapid progress in artificial intelligence.

I have run extensive tests and observed many surprising emergent properties in the outputs. Earlier versions of this project contained numerous issues, as examination-oriented education left me limited time for thorough testing. I now have the opportunity to properly test, refine, and polish this work.

There remains a long road ahead to achieve true general artificial intelligence. I may be forced to set aside this research due to personal circumstances, but I do not want these ideas to fade away unnoticed. The purpose of open-sourcing this project is to share my thoughts, in the hope that others may build upon them and continue this line of inquiry.

---

## Next Steps

1. **Introduce a formal extension interface in the core**, allowing users to plug in GPU, NPU, or other custom accelerators.
2. **Complete the remaining cognitive submodules**, working toward a simple but functional embodied AI agent.

---

## License

MIT © 2026 Li Jinxin. See [LICENSE](LICENES.txt) for details.
