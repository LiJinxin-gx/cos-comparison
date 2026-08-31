# Cos Comparison

[![PyPI version](https://badge.fury.io/py/cos-comparison.svg)](https://pypi.org/project/cos-comparison/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**An AGI-oriented project based on local similarity comparison for feature extraction — biologically inspired, zero-training edge / pattern detection.**

---

## Table of Contents

- [Core Idea](#-core-idea)
- [Key Features](#-key-features)
- [Seven-Layer Architecture](#-seven-layer-cognitive-architecture)
- [Performance](#-performance)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
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
| **`data`** | Data carrying and generic abstraction | — |
| **Functional layers** | Attach to data, consume core algorithms, delegate to `interface` | — |

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

### v0.4.x — Architecture & Exploration

| Version | Theme | Key Highlights |
|---------|-------|----------------|
| **v0.4.4** | Exploration & frameworks | Element-wise API, protocol-style Docker, batch tool, cross-task exploration (face/captcha/agent/text/video), behavior-composition Agent, generic executor, hierarchical isolation memory |
| **v0.4.3** | Robustness & protocols | C core hardening (18 fixes), filter/mapping API (all backends), logic layers (event_context, EventBinds), Fourier module, default dict-protocol implementations |
| **v0.4.2** | Portability | ARM/piwheels C99 fixes, empty-input consistency, 12 memory leaks fixed, duck typing (`PyNumber_Index`), free-threaded 3.14t verified |
| **v0.4.1** | Architecture upgrade | Stride+offset indexing, `infer_shape` / `__shape__` protocol, `load_data` bulk-copy, keyword-only constructors, PyBuffer zero-copy, SIMD hints |

### v0.3.x — Multi-Backend & Indexing Maturity

| Version | Theme | Key Highlights |
|---------|-------|----------------|
| **v0.3.9** | Indexing architecture | Stride+offset fancy indexing (NumPy-like), zero-copy views, 100% recursion-free, backward compatibility for `p`/`end`/`cache` |
| **v0.3.8** | Stability & portability | alloca-free, LSP compliance (isinstance), PyBuffer format detection, zero-division protection, free-threaded GIL release |
| **v0.3.7** | Performance & stability | Python GC (tp_traverse/tp_clear), Welford's online algorithm, SIMD hints, `**`/`pow` operators, subclass return types |
| **v0.3.6** | API alignment | C extension constructor fix, subclass inheritance fix, non-core module import fixes |
| **v0.3.5** | Interface alignment | `__set_item__` standardization, tuple assignment fast path, PyBuffer output writing |
| **v0.3.0** | Multi-backend release | C extension + ctypes backends, three-backend automatic fallback, PyBuffer zero-copy, operator overloading, mean/variance |

### v0.2.x — Tensor System

- **v0.2.0**: `vector_map_as_tensor` N-dimensional tensor view, sliding window comparison, cos/mod/cosmod metrics, passive/active modes, multi-dimensional indexing

### v0.1.x — Initial

- **v0.1.0**: Core cosine similarity, 1D/2D processing, centre-surround antagonism, pure Python

> See [History.txt](History.txt) for the full changelog.

---

## 👤 Author

I was born on May 31, 2008, and feel fortunate to grow up in an era of rapid progress in artificial intelligence. I have run extensive tests and observed many surprising emergent properties. The purpose of open-sourcing is to share my thoughts, in the hope that others may build upon them.

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
