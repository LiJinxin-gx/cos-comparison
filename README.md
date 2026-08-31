# Cos Comparison

[![PyPI version](https://badge.fury.io/py/cos-comparison.svg)](https://pypi.org/project/cos-comparison/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**An AGI-oriented project based on local similarity comparison for feature extraction — biologically inspired, zero-training edge / pattern detection.**

---

## Table of Contents

- [Core Idea](#-core-idea)
- [Key Features](#-key-features)
- [What's New](#-whats-new)
- [Seven-Layer Architecture](#-seven-layer-cognitive-architecture)
- [Performance](#-performance)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Testing](#-testing)
- [Documentation](#-documentation)
- [Version History](#-version-history)
- [Author & Contact](#-author--contact)

---

## 💡 Core Idea

Information is produced by **local comparison** in raw data. This module implements the **centre-surround antagonism** mechanism from neuroscience, extracting edges, textures, and keypoints using only sliding-window similarity.

The core formula (cosine-modulated similarity, recommended default):

$$
\text{cosmod} = \frac{2(A \cdot B)}{|A|^2 + |B|^2}
$$

Three similarity measures, selected via the `algorithm` parameter:

| Measure | Formula | Use Case |
|---------|---------|----------|
| **cos** | `(A·B) / (\|A\|·\|B\|)` | Directional similarity, edge orientation |
| **mod** | `2\|A\|·\|B\| / (\|A\|²+\|B\|²)` | Magnitude similarity, blob/region detection |
| **cosmod** | `2(A·B) / (\|A\|²+\|B\|²)` | Combined (default), edge/keypoint detection |

---

## 🔑 Key Features

- **No training, no labels, no backpropagation** — a step toward biologically plausible AGI
- Works on **1D–4D** data (audio, images, video, volumetric data)
- **Passive** (reflexive boundary detection) and **active** (template matching) modes
- Three high-performance backends with automatic fallback: C extension, ctypes C, pure Python
- Cross-platform (Windows, Linux, macOS) and **zero external dependencies**
- Callback system for progress tracking and custom I/O; flexible output into pre-allocated tensors
- Duck-typed indices: any `__index__`-capable object accepted
- **Unified command line**: shell / batch / app share one instruction file protocol — see [Command Line](docs/command-line.md)
- **Dimension-generic linear algebra** (`math_tool.linear_algebra`): duck typed, `output` keyword pass-out, identical Python / C behaviour

> **Beyond the core engine**, the project explores general learning mechanisms (hierarchical isolation, multi-template clustering, reflective adjustment, von Neumann behavior agents) across multiple domains. See [docs/exploration/](docs/exploration/) for details.

---

## 🆕 What's New

### v0.4.4 — Exploration Test Suite & Behavior-Driven Frameworks

**Exploration & Learning:**
- Exploration test suite (`tests/exploration_*`): delegated standard-library primitives, OpenAI-format local server, face/digits/captcha/image tasks, NLP clustering, data generation, autonomous captcha Agent, light-theme GUI
- Video abstraction & generation training: L0–L3 hierarchy (raw frames → cos features → scene-segment prototypes → video summary); prototypes and contrast points in SQLite; +20.2% generation improvement; 10 videos / 22545 frames
- Agent web-search knowledge base: selenium + Edge headless pipeline; 118/118 chemical elements and 188 ISO 639 languages stored with provenance
- Continuous-mapping hierarchical-isolation memory: layered abstract memory (L1 entries → L2 topic prototypes → L3 views) with two-way layer driving; demand-driven extraction; search and arrangement decoupled

**Behavior & Agent Frameworks:**
- Behavior-composition memory Agent: von Neumann structure — behaviors as data in DB, instruction cycle fetch→decode→reflect-execute→write-back; dynamic reconfiguration via DB only (zero code change)
- Generic executor: atomic instructions stored in DB, fixed `run_program` reads and executes; `$N` result references; `program:<sub>` composite embedding; any-module reflection (plugin-style calls)
- Von-Neumann instructionized web surf: BEHAVE/REFLECT/CORRECT instructions; collect → reflect → correct loop; composite accuracy 86/100
- Feedback-driven search: derive queries from stored data, targeted re-search, cluster groups 2→5, accuracy 90/100

**Core & Upper Layers:**
- Element-wise filter/mapping API (all three backends): `data_filter`, `data_mapping`, `threshold_filter` / `threshold_map`
- Logic layers (protocol-style): `event_context` delegated slots, relative-probability axiom system (A1–A5), `EventBinds` binds-as-engine protocol container
- Fourier module: generic `dft` / `idft` / `power_spectrum`, multi-dimensional, recursion-free, trig-variant float arithmetic
- Default dict-protocol implementations: `Map` three-slot defaults, truthiness substitution
- Nested control flow: flat `Sequence` / `Branch` / `Loop` containers; `ControlFlatten` expands nested flows iteratively (explicit stack, recursion-free)
- Action layer async execution: `ExecuterDriver.call_all` runs on delegated background worker; per-item captured `out`/`err`
- Core hot injection: full backend API hot-injected into `core` namespace with explicit `__all__`

### v0.4.3 — Robustness, Protocol-Style Delegation & Default-Algorithm Release

**C Core Hardening (18 fixes, API unchanged):**
- Memory-safety NULL checks, single-fire end/return callbacks, step=0 division protection
- Output write-bounds checks, short-parameter tolerance, integer-overflow guards
- `global_error` callback ABI alignment (pydll + ctypes backends)

**Element-wise API (all three backends):**
- `data_filter`: yield positions whose callback(value) is truthy, with origin/basis position reporting
- `data_mapping`: map a region through callback into new or pre-allocated output
- `threshold_filter` / `threshold_map`: interval `[low, high]` with inclusive endpoint control
- C implementation: C99-strict, no VLA/GNU extensions, all static, const-correct, shared `_resolve_read_region` core

**Upper-Layer Defect Fixes:**
- Database `executemany` partial-replay removal; `MapMemory.commit` atomicity
- `Process.stop(terminate=True)` child termination; monitor handle race fixed
- `Graph.__repr__` and `DirectedGraph.shortest_path` nested-lock deadlocks eliminated

**Logic & Protocol:**
- `event_context` delegated slots (`init_func`/`add_func`/`probability_func`)
- Relative-probability axiom system (A1–A5: relativism, reflexivity, chain rule, Bayes duality, union)
- Two-stage rigorous resolution: exact hit → relative Bayes → shortest-path chain fallback
- `EventBinds` protocol container (cached graph, stats, strict switch)

**Other:**
- `TensorGenerator.generate(func, args, kwargs)` unified delegation entry
- Sense layer `data_match`: integrated matching-position iterator (active comparison + threshold filter)
- `get_item` scalar-index parity: pure-Python and ctypes backends treat scalar index as 1-D like C extension
- Version handling: `VERSION.txt` restored, `__init__.py` reads with `.strip()` and `utf-8-sig`

### v0.4.2 — Portability & Robustness

- ARM/piwheels C99 fixes, empty-input consistency, exhaustive malloc NULL checks
- 12 reference/memory leaks fixed; duck typing (`PyNumber_Index`); free-threaded 3.14t verified
- Zero-copy buffer protocol; C extension `Vector_init` default shape `(1,)` matching pure Python
- `vector_map_as_tensor(vector=None, shape=)` auto-creation: zero-filled flat vector, fixes heap corruption on C extension
- Buffer-protocol write-through for memoryview inputs: `PyBUF_ND | PyBUF_FORMAT | PyBUF_STRIDES`
- C type-size/overflow hardening: `sizeof(double)` instead of hardcoded 8; `Py_ssize_t`→`int` raises OverflowError

### v0.4.1 — Architecture Upgrade

- Stride+offset indexing, `infer_shape` / `__shape__` protocol, `load_data` bulk-copy
- Keyword-only constructors, PyBuffer zero-copy, SIMD hints, slice performance, free-threaded dual binaries
- Complete removal of legacy indexing parameters (`p`, `end`, `cache`)
- Portable optimization macros (`COS_UNROLL_LOOP`, `COS_ASSUME_ALIGNED`, `COS_INLINE`, `COS_LIKELY/COS_UNLIKELY`)

> See [Version History](#-version-history) below for v0.3.x and earlier, and [History.txt](History.txt) for the full changelog.

---

## 🧬 Seven-Layer Cognitive Architecture

Biologically inspired architecture mimicking mammalian brain structure. Only the core layer is production-ready.

| # | Layer | Directory | Brain Structure | Maturity | Core Function |
|---|-------|-----------|-----------------|----------|---------------|
| 1 | Core | `core` | Brainstem / Cerebellum | ✅ Production | Local comparison, three-backend acceleration, free-thread support |
| 2 | Sense | `sense_layer` | Sensory Cortex | 🟡 Early | Stimulus reception, raw feature extraction |
| 3 | Memory | `memory_layer` | Hippocampus | 🟡 Early | Short/long-term storage, hierarchical isolation memory |
| 4 | Brain | `brain_layer` | Prefrontal Cortex | 🟡 Early | Cognition, logical reasoning, control flow |
| 5 | Action | `action_layer` | Motor Cortex | 🔵 Exploratory | Action output, async execution, environment interaction |
| 6 | Generate | `generate_layer` | Broca's / Wernicke's | 🔵 Exploratory | Language, image generation, template reverse-generation |
| 7 | Extension | `extension_layer` | Association Cortex | 🔴 Skeleton | Extended capabilities |

> Non-core layers do not affect core API stability. `cos_comparison.core` follows semantic versioning.

### Module Architecture

Three-flow logical decoupling (data / operation / control) with clear ownership boundaries:

| Foundation | Role | Independence |
|------------|------|:------------:|
| **`core`** | Low-level algorithms and data format | ✅ stdlib only |
| **`interface`** | All external interaction (processes, locks, shared memory, async, IO) | ✅ stdlib only |
| **`data`** | Data carrying and generic abstraction (`DataWrap`, `Tensor`/`SafeTensor`/`ParallelTensor`) | — |
| **Functional layers** | Attach to data, consume core algorithms, delegate mechanisms to `interface` | — |

**Dependency rules:** layers must not depend on each other; data is attach-and-take (never owned/transformed by layers); all boundary crossing goes through `interface`.

> See [docs/architecture/](docs/architecture/) for full details.

---

## ⚡ Performance

Benchmarked on a 322×424×3 RGB image with 3×3 window (Windows 11 x64, Python 3.14.6, MSVC -O2):

| Backend | Time | Speedup | Free-thread |
|---------|------|---------|-------------|
| C Extension | 0.004s | ~130× | ✅ Full (no GIL) |
| ctypes C | 0.007s | ~70× | ✅ Full |
| Pure Python | 0.52s | 1× | ✅ Full |

On Intel N150 (1000×1000, 3×3 passive): C extension 14s (9×), free-threaded 4 threads 4.6s (27×) vs pure Python 126s.

**Memory:** zero-copy PyBuffer protocol (no data duplication on read), view-based slicing (stride+offset, no copy). C extension static memory footprint < 64 KB.

**Key performance features:** zero-copy PyBuffer, SIMD auto-vectorization (SSE/AVX/NEON with safe fallback), view-based slicing, stride indexing, recursion-free carry iteration, GIL release on compute paths.

---

## 📦 Installation

```bash
pip install cos-comparison          # core package (C compilation attempted automatically)
pip install cos-comparison[test]    # with test dependencies
```

If no C compiler is available, installation succeeds with the pure Python backend only.

| Requirement | Details |
|-------------|---------|
| Python | 3.8+ (3.13+ for free-threaded builds) |
| C compiler | Optional (auto-fallback to pure Python) |
| Runtime deps | None |

To recompile after source changes: `python setup.py build_ext --inplace`

---

## 🚀 Quick Start

```python
from cos_comparison.core import cos_comparison_passive, cos_comparison_active, cos

# Full-tensor similarity
sim = cos([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])  # -> 1.0

# Passive mode: edge detection via sliding window self-similarity
edges = cos_comparison_passive(
    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
    window_size=(2, 2), step=(1, 1), d=(1, 0),
)

# Active mode: template matching
response = cos_comparison_active(
    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
    kernel=[[1.0, 0.0], [0.0, 1.0]], step=(1, 1),
)
```

> For more examples and API reference, see [docs/getting-started.md](docs/getting-started.md) and [docs/api/](docs/api/).

---

## 🧪 Testing

```bash
python -m pytest tests/                           # full suite
python -m pytest tests/test_core_algorithms.py -v # core algorithms only
```

The suite covers core algorithms, tensor ops, backend parity, empty/edge cases, upper layers, and import hygiene. Tests run on both traditional and free-threaded interpreters.

---

## 📚 Documentation

| Section | Contents |
|---------|----------|
| [Getting Started](docs/getting-started.md) | 5-minute tutorial with code examples |
| [Building](docs/build.md) | C backends, dual-compiler strict-mode checks, free-threaded builds |
| [Command Line](docs/command-line.md) | Unified instruction files: shell / batch / app syntax |
| [API Reference](docs/api/README.md) | Core, passive/active modes, statistics, upper layers |
| [Architecture](docs/architecture/README.md) | Seven-layer design, modular architecture, backends |
| [Exploration Tests](docs/exploration/README.md) | Cross-task exploratory tests, learning mechanisms |
| [Principles](docs/principles/README.md) | First principle, similarity measures, dual modes |

---

## 📋 Version History

### v0.3.x — Multi-Backend & Indexing Maturity

| Version | Theme | Key Highlights |
|---------|-------|----------------|
| **v0.3.10** | Bug fixes & enhancements | Stability and feature enhancements |
| **v0.3.9** | Indexing architecture | Stride+offset fancy indexing (NumPy-like), zero-copy views, 100% recursion-free, backward compatibility for `p`/`end`/`cache` |
| **v0.3.8** | Stability & portability | alloca-free, LSP compliance (isinstance), PyBuffer format detection, zero-division protection, free-threaded GIL release |
| **v0.3.7** | Performance & stability | Python GC (tp_traverse/tp_clear), Welford's online algorithm, SIMD hints, `**`/`pow` operators, subclass return types |
| **v0.3.6** | API alignment | C extension constructor fix, subclass inheritance fix, non-core module import fixes |
| **v0.3.5** | Interface alignment | `__set_item__` standardization, tuple assignment fast path, PyBuffer output writing |
| **v0.3.0** | Multi-backend release | C extension + ctypes backends, three-backend automatic fallback, PyBuffer zero-copy, operator overloading, mean/variance |

### v0.2.x — Tensor System

- **v0.2.0**: `vector_map_as_tensor` N-dimensional tensor view system, sliding window local comparison, cos/mod/cosmod metrics, passive/active modes, output parameter support, multi-dimensional indexing and slicing

### v0.1.x — Initial

- **v0.1.0**: Core cosine similarity comparison algorithm, basic 1D/2D data processing, centre-surround antagonism mechanism, pure Python implementation

> See [History.txt](History.txt) for the full changelog.

---

## 👤 Author

I was born on May 31, 2008, and feel fortunate to grow up in an era of rapid progress in artificial intelligence. I have run extensive tests and observed many surprising emergent properties. Earlier versions had numerous issues, as examination-oriented education left me limited time for thorough testing. I now have the opportunity to properly test and refine this work.

There remains a long road to true AGI. I may be forced to set aside this research due to personal circumstances, but I do not want these ideas to fade unnoticed. The purpose of open-sourcing is to share my thoughts, in the hope that others may build upon them.

---

## 📫 Contact

| Channel | Link |
|---------|------|
| GitHub | [LiJinxin-gx/cos-comparison](https://github.com/LiJinxin-gx/cos-comparison) |
| Issues | [GitHub Issues](https://github.com/LiJinxin-gx/cos-comparison/issues) |
| Email | lijinxin_gx@sina.cn |
| PyPI | [cos-comparison](https://pypi.org/project/cos-comparison/) |

---

## 📄 License

MIT © 2026 Li Jinxin. See [LICENSE](LICENSE.txt) for details.
