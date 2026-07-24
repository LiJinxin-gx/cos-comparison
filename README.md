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
* **Zero external dependencies** for core functionality

---

## Seven-Layer Cognitive Architecture

This project follows a biologically inspired seven-layer cognitive architecture, mimicking the structure of the mammalian brain. Currently only the core brainstem/cerebellum layer is production-ready; all other layers are in early evolutionary stage with skeleton implementations, and will be gradually improved in future versions:

| Layer | Directory | Corresponding Brain Structure | Maturity | Core Function |
|-------|-----------|--------------------------------|----------|---------------|
| 1 | `core` | Brainstem / Cerebellum | ✅ Production | Low-level local comparison calculation, three-backend acceleration, free-thread support |
| 2 | `sense_layer` | Sensory Cortex | 🟡 Early Development | Receive external stimuli, extract raw features (Data/Auto_Data classes available) |
| 3 | `memory_layer` | Hippocampus / Cerebral Cortex | 🔴 Skeleton | Short-term and long-term memory storage |
| 4 | `brain_layer` | Prefrontal Cortex | 🟡 Early Development | High-level cognition, logical reasoning (symbolic logic system implemented) |
| 5 | `action_layer` | Motor Cortex | 🔴 Skeleton | Control action output, interact with environment |
| 6 | `generate_layer` | Broca's / Wernicke's Area | 🔴 Skeleton | Generate language, images and other high-level outputs |
| 7 | `extension_layer` | Association Cortex | 🔴 Skeleton | Extended functions and special capabilities |

> **Note**: Non-core layers are currently in early development and do not affect the stability of the core feature extraction API. The core `cos_comparison.core` module is fully production-ready and follows semantic versioning guarantees. All non-core modules now import without fatal errors as of v0.3.6.

---

## 🚀 What's New in Version 0.3.6

Version 0.3.6 is a stability and API alignment release:
- **Constructor API fully aligned with pure Python**: Fixed critical C extension constructor bug, now supports standard `vector_map_as_tensor(flat_data, shape_tuple)` initialization for N-dimensional tensors
- **New `__set_item__` interface aligned across all backends**: Standardized multi-index assignment API (`v.__set_item__((i,j,k), value)`) with 2-3x faster output writing via fast path, eliminating intermediate subview creation overhead
- **Fixed subclass inheritance crash**: Python subclasses inheriting from C extension `vector_map_as_tensor` (e.g. `sense_layer.Data`) now work correctly without memory access violations
- **All non-core modules importable**: Fixed fatal import errors in sense_layer, brain_layer, test_tool and other modules; all seven layers can now be imported without errors
- **All implementations fully iterative**: Eliminated all recursive code paths to prevent stack overflow on large/high-dimensional data
- **Dual Python 3.14 support**: Precompiled binaries for both standard GIL and free-threaded (no-GIL) Python 3.14
- **100% API parity**: All three backends (C extension, ctypes, pure Python) have identical external behavior
- **Zero compiler warnings**: All C code compiles cleanly with no warnings on MSVC, GCC and Clang

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

If you need to recompile the C backends after modifying source code:

```bash
# From the project root directory
python setup.py build_ext --inplace
```

This will automatically build both the Python C extension and the ctypes shared library.

---

## Performance Benchmarks

Benchmark results for all three backends using a **322×424×3 RGB test image** with 3×3 window.  
Test environment: Windows 11 x64, 18-thread CPU, Python 3.14.6, JIT enabled, MSVC -O2 optimization.

| Backend | Execution Time | Speedup vs Pure Python | Peak Memory | Status | Free-thread Support |
|---------|----------------|------------------------|-------------|--------|---------------------|
| Python C Extension | 0.004s | ~130× | ~5–8 MB | ✅ Stable | ✅ Full (no GIL) |
| ctypes C Backend | 0.007s | ~70× | ~8–12 MB | ✅ Stable | ✅ Full |
| Pure Python | 0.52s | 1× | ~22 MB | ✅ Stable | ✅ Full |

> All three backends produce numerically identical output values within floating-point precision. C backends automatically fall back to pure Python if compilation or loading fails.

---

## Quick Start

### 2D edge detection (passive mode)
```python
from cos_comparison import core

# Create test data
data = core.create_void_list((5,5))
for i in range(5):
    for j in range(5):
        data[i,j] = 1.0 if i < 2 and j < 2 else 0.0

# Detect vertical edges
edges = core.cos_comparison_passive(data, window_size=3, d=(0,1))
print(edges[1,1])
```

### Multi-index assignment
```python
# Fast tuple assignment (new in 0.3.6)
v = core.create_void_list((3,3))
v[1,1] = 123.0  # Direct flat-offset assignment, no intermediate objects
assert v[1][1] == 123.0
```

### Switching backends
```python
# Check current backend
print(core.get_mode())

# Force pure Python mode for debugging
core.set_mode("cos_comparison")
```

---

## Author

I was born on May 31, 2008, and I feel fortunate to grow up in an era of rapid progress in artificial intelligence.

I have run extensive tests and observed many surprising emergent properties in the outputs. Earlier versions of this project contained numerous issues, as examination-oriented education left me limited time for thorough testing. I now have the opportunity to properly test, refine, and polish this work.

There remains a long road ahead to achieve true general artificial intelligence. I may be forced to set aside this research due to personal circumstances, but I do not want these ideas to fade away unnoticed. The purpose of open-sourcing this project is to share my thoughts, in the hope that others may build upon them and continue this line of inquiry.

---

## Contact & Feedback

- **Bug Reports & Issues**: Please submit issues on [GitHub Issues](https://github.com/LiJinxin-gx/cos-comparison/issues)
- **Email Contact**: lijinxin_gx@sina.cn

---

## License

MIT © 2026 Li Jinxin. See [LICENSE](LICENSE.txt) for details.
