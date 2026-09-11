# Cos Comparison

[![PyPI version](https://badge.fury.io/py/cos-comparison.svg)](https://pypi.org/project/cos-comparison/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**An AGI-oriented project based on local similarity comparison for feature extraction — biologically inspired, zero-training edge / pattern detection.**

---

## Core Idea

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

### Key Features

- **No training, no labels, no backpropagation** — biologically plausible AGI direction
- Works on **1D–4D** data (audio, images, video, volumetric data)
- **Passive** (reflexive boundary detection) and **active** (template matching) modes
- Two high-performance backends with automatic fallback: C extension, pure Python
- Free-threaded (no-GIL) C extensions on Python 3.13+ (core backend and math_tool), with a reliable fallback on older interpreters
- Cross-platform (Windows, Linux, macOS) and **zero external dependencies**
- Callback system for progress tracking and custom I/O; flexible output into pre-allocated tensors
- Duck-typed indices: any `__index__`-capable object accepted
- **Unified instruction protocol**: shell / batch / app share one instruction file format
- **External command execution**: `sh` plugin (system shell, interactive command line)
- **Dimension-generic linear algebra**: duck typed, `output` keyword pass-out, identical Python / C behaviour
- **Proactive plugin hosting** (`extension_layer.plugin`): `PluginPool` batch aggregation

---

## What's New

### v0.5.0 — Backend Consolidation, Iterate / Transform Extension Points

**Backend consolidation (maintainability):**
- The ctypes backend is retired to keep a **single compiled C path** — one C source to maintain, one build stage, one warning-clean target; the compiled C extension IS the C backend, and the retired call names keep working through a **compatibility mapping** in `config.json` (`.cos_comparison_c` → `.cos_comparison_pydll`), so existing code needs no changes; `config.json` becomes an ordered list of call-name → module mappings and the build system drops the shared-library stage

**Free-threaded support:**
- All C extensions (the core backend and the math_tool extensions) declare `Py_MOD_GIL_NOT_USED` under `#if PY_VERSION_HEX >= 0x030D0000`: free-threaded builds (cp314t) run without the GIL, older interpreters simply skip the slot (reliable fallback); strict-mode portability verified with MSVC `/Wall /WX`, MinGW-w64 GCC and Linux GCC

**Core extensibility (the core module was restructured around it):**
- **`iterate` slot** — element-wise (A-class) and B-class functions accept an external engine (`iterate=engine`); with `None` (default) the original inline skeleton runs unchanged; the C extension delegates the iterator path to the pure Python reference, so external GPU / parallel engines plug in without touching the package
- **`transform1` / `transform2`** — extensible per-value maps replace the retired linear transform (`w1`/`w2`/`b1`/`b2`); `None` = identity (bit-identical default), any callable works (nonlinear, thresholds, lookups)
- **B-class interface cleanup** — `iter_a_callback`/`iter_b_callback` and the `linear` name-space field removed; legacy keyword arguments are still accepted silently

**Tooling:**
- Operator tools / unified value grammar (shell / batch / app): Python-style literals, infix expressions, nested functional calls, zero recursion

**Quality:**
- **math_tool protocol-style refactor** — duck-typed sequence / mapping / set / index / numeric protocols replace the hard-coded container and numeric types across `fourier` / `topology` / `unit_map` (Python reference and the C extensions): numpy scalars, custom containers and subclasses now work through both implementations, with protocol behaviour verified against both
- **Recursion removal** — a package-wide scan (AST call graph + C brace matching) reports zero real recursion: the value-grammar prefix compiler and the shell / app control-flow executors (IF/WHILE) run on explicit frame stacks with the original condition / interrupt / result semantics; the scanner reports zero real recursion across the package

**Benchmark (v0.5.0):**
- An external OpenCL engine on Intel Arc: passive window aggregation up to ~7300× the pure Python reference and 257× the C fast path (end-to-end); both backends drive the same engine with identical results (see [Backend Management](docs/architecture/backend-system.md#benchmark-v050))

> Full changelog available in `History.txt`.

---

## Architecture

### Seven-Layer Cognitive Architecture

Biologically inspired design mimicking mammalian brain structure. Only the core layer is production-ready.

| # | Layer | Directory | Brain Structure | Maturity | Core Function |
|---|-------|-----------|-----------------|----------|---------------|
| 1 | Core | `core` | Brainstem / Cerebellum | ✅ Production | Local comparison, two-backend acceleration, free-thread support, element-wise filter/mapping |
| 2 | Sense | `sense_layer` | Sensory Cortex | 🟡 Early | Stimulus reception, raw feature extraction, data matching |
| 3 | Memory | `memory_layer` | Hippocampus | 🟡 Early | Short/long-term storage, IO-stream memory, database-backed persistence |
| 4 | Brain | `brain_layer` | Prefrontal Cortex | 🟡 Early | Relative-probability logic (A1–A5 axioms), symbol logic, nested control flow, reflex feedback/monitor/trigger, context mapping |
| 5 | Action | `action_layer` | Motor Cortex | 🟡 Early | Async execution driver, action result wrapping, delegated background workers |
| 6 | Generate | `generate_layer` | Broca's / Wernicke's | 🔵 Exploratory | Template-based reverse generation, multi-modal output (validated in exploration) |
| 7 | Extension | `extension_layer` | Association Cortex | 🔴 Skeleton | Extended capabilities (placeholder) |

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

---

## Installation

```bash
pip install cos-comparison          # core package (C compilation attempted automatically)
```

If no C compiler is available, installation succeeds with the pure Python backend only.

| Requirement | Details |
|-------------|---------|
| Python | 3.8+ (3.13+ for free-threaded builds) |
| C compiler | Optional (auto-fallback to pure Python) |
| Runtime deps | None |

The test suite is stdlib-only (`unittest`) — no test dependencies to install.

To recompile after source changes: `python setup.py build_ext --inplace`

### Quick Start

The core module exposes three primary APIs: full-tensor similarity (`cos`), passive self-similarity (`cos_comparison_passive`), and active template matching (`cos_comparison_active`). All three operate on 1D–4D tensors with identical signatures across backends.

→ See [Getting Started](docs/getting-started.md) for step-by-step usage, and the [API reference](docs/api/README.md) for full signatures.

---

## Exploration & Performance

### Exploration Repository

Extended experiments and algorithm exploration are maintained in a dedicated repository:

🔬 **[cos-comparison-experiment](https://github.com/LiJinxin-gx/cos-comparison-experiment)** — zero-dependency algorithm sharing, tensor-input design, cross-domain exploration.

### Exploration Results (Overview)

The core engine has been validated across multiple domains through exploratory experiments:

- **Computer vision**: face recognition (67 ID-photo training → life-photo validation), captcha solving, multi-scale feature extraction
- **Audio**: music genre classification, spectral feature extraction, unsupervised speech recognition (atomic contrast point matching, 47.1% on 30 classes)
- **Text**: token frequency analysis, hierarchical clustering, demand-driven memory
- **Video**: frame abstraction, scene segmentation, cross-video generation (+20.2% improvement)
- **Agent systems**: web-search knowledge base (118 elements, 188 languages), behavior-composition Agent (von Neumann stored-program), generic executor (DB-stored instructions)
- **Learning mechanisms**: continuous-mapping hierarchical isolation, multi-template clustering, reflective self-correction, feedback-driven search

All exploration uses the same core local-comparison engine — no deep learning, no backpropagation, no third-party ML dependencies.

### Performance

**Test platform:** Intel Core Ultra 5 125H (14C/18T, 3.6 GHz) + Intel Arc
Graphics (112 CUs, driver 31.0.101.5382); Windows 11 x64; Python 3.14.6
(CPython); numpy 2.5.2; pyopencl 2026.1.4.

**Test suite:** the benchmark harness (numpy + pyopencl) is kept in the
experiment area and is not shipped with the package; it measures the
default backends against an external OpenCL engine injected through
`iterate=` over float32 square grids, sizes 64²–4096², 3×3 window,
d=(1,1); element-wise mapping applies `v*2+1`.  Timing is the minimum of
repeated runs (best-of-3 up to 512², single run above); the GPU figures
are end-to-end and include host↔device transfers and buffer setup.

passive window aggregation (ms):

| Size | Pure Python | C Extension | GPU engine | C/GPU |
|------|-------------|-------------|------------|-------|
| 64² | 62.5 | 2.13 | 0.27 | 7.8× |
| 512² | 4554 | 161 | 0.66 | 244× |
| 4096² | — | 10519 | 40.9 | 257× |

data_mapping (element-wise, v*2+1, ms): 512² — pure Python 563, C extension
272, GPU 0.56 (490× over C); 4096² — C extension 17721, GPU 29.3 (605×).
The C path is callback-bound on the element-wise ops (it still calls the
Python callback per element); the GPU engine moves the whole loop to the
device.

The GPU engine is an **external module** injected via `iterate=` (see
[Backend Management](docs/architecture/backend-system.md#benchmark-v050)):
the package stays stdlib-only, and both backends drive the same engine with
identical results and identical device timing (the C extension delegates the
iterator path to the pure Python reference).

**Memory:** zero-copy PyBuffer protocol (no data duplication on read), view-based slicing (stride+offset, no copy). C extension static memory footprint < 64 KB.

**Key performance features:** zero-copy PyBuffer, SIMD auto-vectorization (SSE/AVX/NEON with safe fallback), view-based slicing, stride indexing, recursion-free carry iteration, GIL release on compute paths.

---

## Testing

```bash
python -m unittest discover -s tests              # full suite (stdlib only)
python -m unittest tests.test_core_algorithms -v  # core algorithms only
```

The suite covers core algorithms, tensor ops, backend parity, empty/edge cases, upper layers, and import hygiene; it uses only the standard library (`unittest`; pytest works too but is not required). Tests run on both traditional and free-threaded interpreters.

---

## Author

I was born on May 31, 2008, and feel fortunate to grow up in an era of rapid progress in artificial intelligence. I have run extensive tests and observed many surprising emergent properties. The purpose of open-sourcing is to share my thoughts, in the hope that others may build upon them.

---

## Contact

| Channel | Link |
|---------|------|
| GitHub (main) | [LiJinxin-gx/cos-comparison](https://github.com/LiJinxin-gx/cos-comparison) |
| GitHub (exploration) | [LiJinxin-gx/cos-comparison-experiment](https://github.com/LiJinxin-gx/cos-comparison-experiment) |
| Issues | [GitHub Issues](https://github.com/LiJinxin-gx/cos-comparison/issues) |
| Email | lijinxin_gx@sina.cn |
| PyPI | [cos-comparison](https://pypi.org/project/cos-comparison/) |

---

## License

MIT © 2026 Li Jinxin. See `LICENSE.txt` for details.
