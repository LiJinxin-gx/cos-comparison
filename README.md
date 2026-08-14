# Cos Comparison

[![PyPI version](https://badge.fury.io/py/cos-comparison.svg)](https://pypi.org/project/cos-comparison/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**An AGI-oriented project based on local similarity comparison for feature extraction – biologically inspired, zero-training edge / pattern detection.**

---

## 💡 Core Idea

Information is produced by **local comparison** in raw data.  
This module implements the **centre-surround antagonism** mechanism from neuroscience, extracting edges, textures, and keypoints using only sliding-window similarity.

The core formula (cosine-modulated similarity, recommended default):

$$
\text{cosmod} = \frac{2(A \cdot B)}{|A|^2 + |B|^2}
$$

Three similarity measures are provided, selected via the `algorithm` parameter:

| Measure | Formula | Use Case |
|---------|---------|----------|
| **cos** | `(A·B) / (|A|·|B|)` | Directional similarity, edge orientation |
| **mod** | `2|A|·|B| / (|A|²+|B|²)` | Magnitude similarity, blob/region detection |
| **cosmod** | `2(A·B) / (|A|²+|B|²)` | Combined (default), edge/keypoint detection |

Key features:

- **No training, no labels, no backpropagation** — a step toward biologically plausible AGI
- Works on **1D, 2D, 3D, 4D** data (audio, images, video, volumetric data)
- **Passive** (reflexive boundary detection) and **active** (template matching) modes
- Three high-performance backends with automatic fallback: Python C extension, ctypes pure C, pure Python
- Cross-platform (Windows, Linux, macOS) and **zero external dependencies**
- **Callback system**: `start_callback`, `iter_a_callback`, `iter_b_callback`, `end_callback`, `return_callback`, `local_error_callback`, `global_error_callback` for progress tracking and custom I/O
- **Flexible output**: write results into pre-allocated tensors with `output`, `output_start`, `output_step`
- **Duck-typed indices**: index parameters accept any `__index__`-capable object, not just `int` subclasses

---

## 🚀 What's New in Version 0.4.2

Version 0.4.2 is a **portability & robustness release**. No API change but with less API append.

- **ARM / piwheels support**: fixed C99 label-after-declaration errors and non-static inline declarations that blocked compilation on ARM Linux (Raspberry Pi)
- **Empty-input robustness**: all backends now raise consistent `IndexError`/`ValueError` for empty tensors instead of crashing (the C extension previously segfaulted on `vector_map_as_tensor(vector=[])`)
- **Memory safety**: exhaustive `malloc` NULL checks on every allocation path; fixed a missing-brace bug that caused unconditional `return NULL`; fixed 12 reference/memory leaks in arithmetic and statistics paths
- **Duck typing**: index parameters now accept any `__index__`-capable object (PyNumber_Index), not just `int` subclasses
- **Free-threaded Python**: verified on Python 3.14t; GIL released on compute-heavy paths
- **`vector_map_as_tensor(vector=None, shape=)` auto-creation**: `vector=None` now auto-creates the default zero-filled vector (list on Python backends, native array on C backends); the C extension previously crashed on `vector=None`, and an omitted `vector` keeps the historical `(1,)` → `1.0` default
- **Buffer protocol**: `memoryview(tensor)` is read-only and write-rejecting on all backends (numpy `frombuffer` contract); buffer-backed tensors (`array.array`, `bytearray`, `memoryview`) share storage and write through on every backend — the C extension now exports memoryview inputs zero-copy (PyBUF_ND/STRIDES instead of the copy-producing SIMPLE request)
- **C extension in-place scalar operators**: `t += scalar` / `t -= scalar` now work on pydll, matching the ctypes/pure backends
- **Type-promotion decision**: integer-typed arithmetic results were evaluated and deliberately not implemented — double remains the single universal element type to keep C hot loops SIMD-vectorizable

See [History.txt](History.txt) for the full changelog.

---

## 🚀 What's New in Version 0.4.1

Version 0.4.1 is a major architecture upgrade release, modernizing the indexing system and adding powerful new features:

- **Breaking change**: Complete removal of legacy indexing parameters (`p`, `end`, `cache`) from `vector_map_as_tensor`, final migration to stride+offset architecture
- **New `infer_shape` function**: Multi-priority shape inference for all backends - PyBuffer protocol > `__shape__()` method > iterative length detection, fast path for internal tensors
- **`__shape__` protocol method**: All tensor types now expose `__shape__()` method for zero-overhead shape inference, overridable by subclasses
- **Enhanced `load_as_default_data`**: Added `step` parameter support for sub-sampling during data loading, tensor fast path using native slicing
- **New `load_data` function**: Bulk-copy a sub-region between two data containers with independent start/step for source and target, PyBuffer fast path with write-persistence probing, automatic bounds clamping
- **Keyword-only initialization**: All `vector_map_as_tensor` constructors now use keyword arguments for optional parameters, cleaner and safer API
- **Code simplification**: Removed all backward compatibility shims, simplified indexing logic, more general and maintainable codebase
- **Enhanced PyBuffer protocol**: Improved zero-copy support for `array.array`, `memoryview`, `bytes`, and other buffer-like objects, automatic type conversion for common numeric formats (double/float/int/short/long/long long/unsigned char)
- **Further SIMD optimizations**: More loops annotated with cross-compiler ivdep hints, better auto-vectorization on all compilers, portable optimization macros (unroll, alignment, branch prediction)
- **Optimized `[::,::]` slice performance**: Optimized non-contiguous view access patterns, faster read and assignment for stepped slices
- **All three backends updated**: Pure Python (reference), C extension, and ctypes backends all implement v0.4.1 features with 100% behavioral parity
- **ARM / piwheels compatibility fixes**: Removed all x86-specific assumptions, fully portable C11 code, fixes for ARM platform compilation
- **Fixed ctypes backend call errors**: Corrected function signatures and type mappings, all ctypes operations work correctly
- **Updated core module loader**: `infer_shape` and `load_data` added to hot API list for zero-overhead access
- **default_contain API consistency**: C extension `default_contain` now matches pure Python exactly - same constructor signature (`default`, `default_dict`), same behavior across all backends
- **Strict keyword-only constructors**: `vector_map_as_tensor` enforces keyword-only arguments across all backends, matching pure Python `def __init__(self, *, ...)` signature
- **Free-threaded Python 3.14 verified**: Full functionality verified on free-threaded Python 3.14t, GIL correctly released in compute-heavy paths
- **Dual version support**: Precompiled binaries for both standard GIL and free-threaded (no-GIL) Python 3.14, zero compiler warnings
- **Zero external dependencies**: Core package remains 100% dependency-free, no numpy or other third-party requirements
- **Comprehensively tested**: All functionality tested in clean virtual environment, 100% behavioral parity across all three backends, no recursion guaranteed
- **ctypes backend stability hotfix**: Fixed no-callback crash, invalid `Data_free` heap corruption, and callback lookup crash on Python 3.14 - all fixes are pure-Python side (id-based callback registry), no API changes and no performance regression (see History.txt v0.4.1)
- **Upper-layer (non-core) stability fixes**: `brain_layer.logic`/`brain_layer.map` constructor and attribute fixes (NameError, `__eq__` typo, `arg_names`), `sense_layer.Receptor.point` accepts scalar and tuple indices, `TensorReceptor` passive comparison works again, `DatabaseMemory.refer` default hook fixed, `interface` ctypes `argtypes` / rollback / `Process` shadowing fixed, `data.tensor` tuple `start` mapping fixed
- **Verified**: full unittest suite (core + upper layers) green on the installed wheel

---

## 🧬 Seven-Layer Cognitive Architecture

This project follows a biologically inspired seven-layer cognitive architecture, mimicking the structure of the mammalian brain. Currently only the core brainstem/cerebellum layer is production-ready; all other layers are in early evolutionary stage with skeleton implementations, and will be gradually improved in future versions:

| # | Layer | Directory | Corresponding Brain Structure | Maturity | Core Function |
|---|-------|-----------|-------------------------------|----------|---------------|
| 1 | Core | `core` | Brainstem / Cerebellum | ✅ Production | Low-level local comparison calculation, three-backend acceleration, free-thread support, stride-based indexing |
| 2 | Sense | `sense_layer` | Sensory Cortex | 🟡 Early Development | Receive external stimuli, extract raw features (Data/Auto_Data classes available) |
| 3 | Memory | `memory_layer` | Hippocampus / Cerebral Cortex | 🟡 Early Development | Short-term and long-term memory storage |
| 4 | Brain | `brain_layer` | Prefrontal Cortex | 🟡 Early Development | High-level cognition, logical reasoning (symbolic logic system implemented) |
| 5 | Action | `action_layer` | Motor Cortex | 🔵 Exploratory Development | Control action output, interact with environment |
| 6 | Generate | `generate_layer` | Broca's / Wernicke's Area | 🔵 Exploratory Development | Generate language, images and other high-level outputs |
| 7 | Extension | `extension_layer` | Association Cortex | 🔴 Skeleton | Extended functions and special capabilities |

> **Note**: Non-core layers are currently in early development and do not affect the stability of the core feature extraction API. The core `cos_comparison.core` module is fully production-ready and follows semantic versioning guarantees. All non-core modules import without fatal errors.

---

## 🏗️ Module Architecture (v0.4.1)

This project follows a **modular architecture** defined since v0.4.1. Module responsibility boundaries, dependency rules, and the attach-and-take data flow are specified in detail in the modular architecture documentation. Below is the summary:

### Three-flow logical decoupling: data flow, operation flow, control flow

The architecture logically decouples three orthogonal flows. Each flow has a clear owner, so that changing one flow never leaks into the others:

| Flow | Responsibility | Owner | Concrete manifestation |
|------|----------------|-------|------------------------|
| 🧬 **Data flow** | What data is carried and how it travels through the model | `core` (data format) + `data` (carrying abstraction) | Caller constructs data using core-exposed formats / `data` wrappers; layers **attach-and-take** it at entry points (`Receptor(data)`, `Generator(data)`, `Memory.process(caller, ...)`) and never own or transform it |
| ⚙️ **Operation flow** | What the model does — algorithms and capability operations | `core` (algorithm family) + each functional layer (its capability) | Operations consume the attached data only through core-exposed algorithms; no format conversion, no duplicate logic inside layers |
| 🎛️ **Control flow** | When and how the model coordinates — concurrency, timing, processes, IO | `interface` | `EventLoop`, `run_in_thread`, integrated locks, process / shared-memory primitives; layers never use raw `asyncio` / `threading` — `reflex.Monitor` stays pure logic, with all mechanisms delegated to `interface` |

Benefits of the decoupling:

- Data flow can be extended independently — any data expressible in core-exposed formats participates without touching operations or control;
- Operation flow can evolve independently — layers are interchangeable as long as they obey attach-and-take;
- Control flow can be swapped independently — e.g. replacing a thread-based mechanism with a process-based one requires no change in data or operation flows.

### Absolute-independent foundation

| Module | Responsibility | Dependencies |
|--------|----------------|--------------|
| `core` | Internal low-level algorithms **and** exposure of the internal data format (`vector_map_as_tensor`, three-backend engine, stride indexing) | Standard library only |
| `interface` | All **external** interaction and interface abstraction (system commands / processes, concurrent locks, shared arrays & thread launching, async periodic event hosts, IO / socket / pipe / file communication, dynamic-library & module invocation, context tools) | Standard library only |

### Built on top of the foundation

The five functional layers (`sense_layer`, `memory_layer`, `brain_layer`, `action_layer`, `generate_layer`) are detailed in the Seven-Layer table above; each attaches to incoming data, consumes core algorithms, and delegates all mechanisms to `interface`. Two additional cross-cutting modules sit on the foundation:

| Module | Responsibility |
|--------|----------------|
| `data` | Dedicated **data carrying and generic abstraction**: `DataWrap`, `Tensor` / `SafeTensor` / `ParallelTensor`, shape/stride/type normalization |
| `test_tool` / `app` | Testing utilities / future business aggregation entry |

### Dependency rules (red lines)

- `core` and `interface` import **nothing internal** — they are fully independent;
- `data` may depend on `core` (data format) and `interface` (shared-memory / Locks);
- Each functional layer may depend on `core`, `interface`, and `data` — but layers **must not** depend on each other;
- Layers **do not own or transform data**: data is constructed by the caller using core-exposed formats / data providers and taken over ("attach-and-take") by the layer entry point (e.g. `Receptor(data)`, `Generator(data)`, `Memory.process(caller, ...)`);
- Everything that crosses the model boundary goes through `interface` (no direct sockets / processes / databases inside layers);
- Async / thread **mechanisms** also belong to `interface` — layers use `EventLoop`, `run_in_thread`, integrated locks from `interface.api`, never raw `asyncio` / `threading` in their own logic (e.g. `reflex.Monitor` is pure logic).

---

## ⚡ Performance Benchmarks

Performance varies by backend and operation type:

### Backend Performance Comparison (relative to pure Python)

| Backend | Element-wise Arithmetic | Mean/Variance | Core Comparison | Memory Overhead |
|---------|------------------------|---------------|-----------------|-----------------|
| Pure Python | 1.0x (baseline) | 1.0x (baseline) | 1.0x (baseline) | Lowest |
| ctypes (pure C) | 5-10x | 8-15x | 10-20x | Low |
| C Extension | 10-20x | 15-30x | 20-40x | Medium |

### Key Performance Features

- **Zero-copy PyBuffer support**: Direct memory access for array.array, memoryview, numpy arrays, etc.
- **SIMD auto-vectorization**: Cross-compiler hints enable automatic SIMD instruction generation (SSE/AVX on x86, NEON on ARM)
- **View-based slicing**: All slice operations create views with zero data copying, even for non-contiguous steps
- **Stride-based indexing**: Efficient multi-dimensional access with precomputed strides
- **Carry-mechanism iteration**: 100% recursion-free iteration, no stack overflow even for high-dimensional tensors
- **Free-threaded support**: Compute-heavy functions release GIL for true multi-threaded parallelism on Python 3.13+

### Real-World Benchmark (322×424×3 RGB Test Image)

Benchmark results for all three backends using a **322×424×3 RGB test image** with 3×3 window.  
Test environment: Windows 11 x64, 18-thread CPU, Python 3.14.6, JIT enabled, MSVC -O2 optimization.

| Backend | Execution Time | Speedup vs Pure Python | Peak Memory | Status | Free-thread Support |
|---------|---------------|------------------------|-------------|--------|---------------------|
| Python C Extension | 0.004s | ~130× | ~5–8 MB | ✅ Stable | ✅ Full (no GIL) |
| ctypes C Backend | 0.007s | ~70× | ~8–12 MB | ✅ Stable | ✅ Full |
| Pure Python | 0.52s | 1× | ~22 MB | ✅ Stable | ✅ Full |

### Synthetic Benchmark (1000×1000, Intel N150)

A 1000×1000 float64 array, `cos_comparison_passive` with `window_size=(3,3)`, measured on an Intel N150 (4 cores, 1.3 GHz):

| Backend | Time | Relative |
|---------|------|----------|
| Pure Python 3.14.6 | 126 s | 1× (baseline) |
| ctypes + DLL | 21 s | **6× faster** |
| C extension (3.14) | 14 s | **9× faster** |
| C extension (3.14t, 4 threads) | 4.6 s | **27× faster** |

> **Note**: Performance numbers are approximate and depend on hardware, compiler, data size, and specific operation. All three backends produce numerically identical output values within floating-point precision, and C backends automatically fall back to pure Python if compilation or loading fails.

---

## 📦 Installation

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

### Requirements

- Python **3.8+** (3.13+ for free-threaded builds)
- C compiler (optional, for acceleration backends)
- No runtime dependencies

---

## 🚀 Quick Start

### 2D edge detection (passive & active modes)

```python
from cos_comparison import core

# 64x64 checkerboard test data
data = core.create_void_list((64, 64))
for i in range(64):
    for j in range(64):
        data[i, j] = 1.0 if (i + j) % 2 == 0 else 0.0

# Passive mode: sliding-window self-similarity (cosmod, default)
edges = core.cos_comparison_passive(data, window_size=(3, 3))
print(edges.shape)  # (61, 62)

# The displacement parameter d shifts the second window (edge orientation)
edges2 = core.cos_comparison_passive(data, window_size=(3, 3), d=(0, 1))
print(edges2.shape)  # (62, 61)

# Active mode: template matching with an external kernel
kernel = core.create_void_list((3, 3), default=1.0)
matches = core.cos_comparison_active(data, kernel=kernel)
print(matches.shape)  # (62, 62)
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
# Note: suggest to add the character "." to show as a relative import
core.set_mode(".cos_comparison")
```

---

## 🧪 Testing

```bash
# From the project root, using the venv_test environment:
python -m unittest tests.test_core_algorithms tests.test_core_tensor
python tests/test_empty_edge.py          # subprocess-isolated edge cases
python tests/test_normal_consistency.py  # three-backend parity
```

The test suite covers core algorithms, tensor operations, backend parity, empty/edge cases, upper layers, and import hygiene. See `tests/README.md` for details.

---

## 👤 Author

I was born on May 31, 2008, and I feel fortunate to grow up in an era of rapid progress in artificial intelligence.

I have run extensive tests and observed many surprising emergent properties in the outputs. Earlier versions of this project contained numerous issues, as examination-oriented education left me limited time for thorough testing. I now have the opportunity to properly test, refine, and polish this work.

There remains a long road ahead to achieve true general artificial intelligence. I may be forced to set aside this research due to personal circumstances, but I do not want these ideas to fade away unnoticed. The purpose of open-sourcing this project is to share my thoughts, in the hope that others may build upon them and continue this line of inquiry.

Thanks to the assistance of artificial intelligence, I was able to quickly check and fix the errors in the project code. I have realized the importance of using tools.

---

## 📚 Documentation

- [Documentation Hub](docs/README.md) — navigation index for all docs
- [Getting Started](docs/getting-started.md) — 5-minute tutorial
- [API Reference](docs/api/README.md) — core, passive/active modes, statistics, upper layers
- [Architecture](docs/architecture/README.md) — seven-layer design, modular architecture, backends
- [Principles](docs/principles/README.md) — first principle, similarity measures, dual modes

---

## 📫 Contact & Feedback

- **Bug Reports & Issues**: Please submit issues on [GitHub Issues](https://github.com/LiJinxin-gx/cos-comparison/issues)
- **Email Contact**: lijinxin_gx@sina.cn
- **GitHub**: [LiJinxin-gx/cos-comparison](https://github.com/LiJinxin-gx/cos-comparison)
- **PyPI**: [cos-comparison](https://pypi.org/project/cos-comparison/)

---

## 📄 License

MIT © 2026 Li Jinxin. See [LICENSE](LICENSE.txt) for details.
