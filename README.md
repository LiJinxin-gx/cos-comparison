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

## 🚀 What's New in Version 0.4.0

Version 0.4.0 is a major architecture upgrade release, modernizing the indexing system and adding powerful new features:

* **Breaking change**: Complete removal of legacy indexing parameters (`p`, `end`, `cache`) from `vector_map_as_tensor`, final migration to stride+offset architecture
* **New `infer_shape` function**: Multi-priority shape inference for all backends - PyBuffer protocol > `__shape__()` method > iterative length detection, fast path for internal tensors
* **`__shape__` protocol method**: All tensor types now expose `__shape__()` method for zero-overhead shape inference, overridable by subclasses
* **Enhanced `load_as_default_data`**: Added `step` parameter support for sub-sampling during data loading, tensor fast path using native slicing
* **Keyword-only initialization**: All `vector_map_as_tensor` constructors now use keyword arguments for optional parameters, cleaner and safer API
* **Code simplification**: Removed all backward compatibility shims, simplified indexing logic, more general and maintainable codebase
* **Enhanced PyBuffer protocol**: Improved zero-copy support for `array.array`, `memoryview`, `bytes`, and other buffer-like objects, automatic type conversion for common numeric formats (double/float/int/short/long/long long/unsigned char)
* **Further SIMD optimizations**: More loops annotated with cross-compiler ivdep hints, better auto-vectorization on all compilers, portable optimization macros (unroll, alignment, branch prediction)
* **Optimized `[::,::]` slice performance**: Optimized non-contiguous view access patterns, faster read and assignment for stepped slices
* **All three backends updated**: Pure Python (reference), C extension, and ctypes backends all implement v0.4.0 features with 100% behavioral parity
* **ARM / piwheels compatibility fixes**: Removed all x86-specific assumptions, fully portable C11 code, fixes for ARM platform compilation
* **Fixed ctypes backend call errors**: Corrected function signatures and type mappings, all ctypes operations work correctly
* **Updated core module loader**: `infer_shape` and `load_data` added to hot API list for zero-overhead access
* **default_contain API consistency**: C extension `default_contain` now matches pure Python exactly - same constructor signature (`default`, `default_dict`), same behavior across all backends
* **Strict keyword-only constructors**: `vector_map_as_tensor` enforces keyword-only arguments across all backends, matching pure Python `def __init__(self, *, ...)` signature
* **Free-threaded Python 3.14 verified**: Full functionality verified on free-threaded Python 3.14t, GIL correctly released in compute-heavy paths
* **Dual version support**: Precompiled binaries for both standard GIL and free-threaded (no-GIL) Python 3.14, zero compiler warnings
* **Zero external dependencies**: Core package remains 100% dependency-free, no numpy or other third-party requirements
* **Comprehensively tested**: All functionality tested in clean virtual environment, 100% behavioral parity across all three backends, no recursion guaranteed
* **ctypes backend stability hotfix**: Fixed no-callback crash, invalid `Data_free` heap corruption, and callback lookup crash on Python 3.14 - all fixes are pure-Python side (id-based callback registry), no API changes and no performance regression (see History.txt v0.4.0)
* **Upper-layer (non-core) stability fixes**: `brain_layer.logic`/`brain_layer.map` constructor and attribute fixes (NameError, `__eq__` typo, `arg_names`), `sense_layer.Receptor.point` accepts scalar and tuple indices, `TensorReceptor` passive comparison works again, `DatabaseMemory.refer` default hook fixed, `interface` ctypes `argtypes` / rollback / `Process` shadowing fixed, `data.tensor` tuple `start` mapping fixed
* **Verified**: full unittest suite (core + upper layers) green on the installed wheel

---

## Seven-Layer Cognitive Architecture

This project follows a biologically inspired seven-layer cognitive architecture, mimicking the structure of the mammalian brain. Currently only the core brainstem/cerebellum layer is production-ready; all other layers are in early evolutionary stage with skeleton implementations, and will be gradually improved in future versions:

|Layer|Directory|Corresponding Brain Structure|Maturity|Core Function|
|-|-|-|-|-|
|1|`core`|Brainstem / Cerebellum|✅ Production|Low-level local comparison calculation, three-backend acceleration, free-thread support, stride-based indexing|
|2|`sense_layer`|Sensory Cortex|🟡 Early Development|Receive external stimuli, extract raw features (Data/Auto_Data classes available)|
|3|`memory_layer`|Hippocampus / Cerebral Cortex|🟡 Early Development|Short-term and long-term memory storage|
|4|`brain_layer`|Prefrontal Cortex|🟡 Early Development|High-level cognition, logical reasoning (symbolic logic system implemented)|
|5|`action_layer`|Motor Cortex|🔵 Exploratory Development|Control action output, interact with environment|
|6|`generate_layer`|Broca's / Wernicke's Area|🔵 Exploratory Development|Generate language, images and other high-level outputs|
|7|`extension_layer`|Association Cortex|🔴 Skeleton|Extended functions and special capabilities|

> **Note**: Non-core layers are currently in early development and do not affect the stability of the core feature extraction API. The core `cos_comparison.core` module is fully production-ready and follows semantic versioning guarantees. All non-core modules import without fatal errors.

---

## Module Architecture (v0.4.0)

This project follows a **modular architecture** defined since v0.4.0. Module responsibility boundaries, dependency rules, and the attach-and-take data flow are specified in detail in [Modular Architecture](docs/architecture/modular-architecture.md). Below is the summary:

### Absolute-independent foundation

| Module | Responsibility | Dependencies |
|--------|----------------|--------------|
| `core` | Internal low-level algorithms **and** exposure of the internal data format (`vector_map_as_tensor`, three-backend engine, stride indexing) | Standard library only |
| `interface` | All **external** interaction and interface abstraction (system commands / processes, concurrent locks, shared arrays & thread launching, async periodic event hosts, IO / socket / pipe / file communication, dynamic-library & module invocation, context tools) | Standard library only |

### Built on top of the foundation

| Module | Responsibility |
|--------|----------------|
| `data` | Dedicated **data carrying and generic abstraction**: `DataWrap`, `Tensor` / `SafeTensor` / `ParallelTensor`, shape/stride/type normalization |
| `sense_layer` | Sensory layer — attaches to incoming data and invokes core algorithms |
| `memory_layer` | Memory layer — hierarchical memory carriers, transactions, rollback |
| `brain_layer` | Cognition layer — symbolic logic, probabilistic logic, reflex monitoring |
| `action_layer` | Action layer — execution scheduling |
| `generate_layer` | Generation layer — attaches an external callee to produce output data |
| `test_tool` / `app` | Testing utilities / future business aggregation entry |

### Dependency rules (red lines)

- `core` and `interface` import **nothing internal** — they are fully independent;
- `data` may depend on `core` (data format) and `interface` (shared-memory / Locks);
- Each functional layer may depend on `core`, `interface`, and `data` — but layers **must not** depend on each other;
- Layers **do not own or transform data**: data is constructed by the caller using core-exposed formats / data providers and taken over ("attach-and-take") by the layer entry point (e.g. `Receptor(data)`, `Generator(data)`, `Memory.process(caller, ...)`);
- Everything that crosses the model boundary goes through `interface` (no direct sockets / processes / databases inside layers);
- Async / thread **mechanisms** also belong to `interface` — layers use `EventLoop`, `run_in_thread`, integrated locks from `interface.api`, never raw `asyncio` / `threading` in their own logic (e.g. `reflex.Monitor` is pure logic).

---

## Performance Benchmarks

cos-comparison is designed for maximum performance while maintaining portability and zero dependencies. Performance varies by backend and operation type:

### Backend Performance Comparison (relative to pure Python)

|Backend|Element-wise Arithmetic|Mean/Variance|Core Comparison|Memory Overhead|
|-|-|-|-|-|
|Pure Python|1.0x (baseline)|1.0x (baseline)|1.0x (baseline)|Lowest|
|ctypes (pure C)|5-10x|8-15x|10-20x|Low|
|C Extension|10-20x|15-30x|20-40x|Medium|

### Key Performance Features

* **Zero-copy PyBuffer support**: Direct memory access for array.array, memoryview, numpy arrays, etc.
* **SIMD auto-vectorization**: Cross-compiler hints enable automatic SIMD instruction generation (SSE/AVX on x86, NEON on ARM)
* **View-based slicing**: All slice operations create views with zero data copying, even for non-contiguous steps
* **Stride-based indexing**: Efficient multi-dimensional access with precomputed strides
* **Carry-mechanism iteration**: 100% recursion-free iteration, no stack overflow even for high-dimensional tensors
* **Free-threaded support**: Compute-heavy functions release GIL for true multi-threaded parallelism on Python 3.13+

> **Note**: Performance numbers are approximate and depend on hardware, compiler, data size, and specific operation. All backends produce bit-identical results.

### Real-World Benchmark (322×424×3 RGB Test Image)

Benchmark results for all three backends using a **322×424×3 RGB test image** with 3×3 window.  
Test environment: Windows 11 x64, 18-thread CPU, Python 3.14.6, JIT enabled, MSVC -O2 optimization.

|Backend|Execution Time|Speedup vs Pure Python|Peak Memory|Status|Free-thread Support|
|-|-|-|-|-|-|
|Python C Extension|0.004s|~130×|~5–8 MB|✅ Stable|✅ Full (no GIL)|
|ctypes C Backend|0.007s|~70×|~8–12 MB|✅ Stable|✅ Full|
|Pure Python|0.52s|1×|~22 MB|✅ Stable|✅ Full|

> All three backends produce numerically identical output values within floating-point precision. C backends automatically fall back to pure Python if compilation or loading fails.

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

## Quick Start

### 2D edge detection (passive mode)

```python
from cos_comparison import core

# Create test data of default type
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
# Note : suggest to add the character "." to show as a relative import
core.set_mode(".cos_comparison")
```

---

## Author

I was born on May 31, 2008, and I feel fortunate to grow up in an era of rapid progress in artificial intelligence.

I have run extensive tests and observed many surprising emergent properties in the outputs. Earlier versions of this project contained numerous issues, as examination-oriented education left me limited time for thorough testing. I now have the opportunity to properly test, refine, and polish this work.

There remains a long road ahead to achieve true general artificial intelligence. I may be forced to set aside this research due to personal circumstances, but I do not want these ideas to fade away unnoticed. The purpose of open-sourcing this project is to share my thoughts, in the hope that others may build upon them and continue this line of inquiry. 

---

## Contact & Feedback

* **Bug Reports & Issues**: Please submit issues on [GitHub Issues](https://github.com/LiJinxin-gx/cos-comparison/issues)
* **Email Contact**: lijinxin_gx@sina.cn

---

## License

MIT © 2026 Li Jinxin. See [LICENSE](LICENSE.txt) for details.
