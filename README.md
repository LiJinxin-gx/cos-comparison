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
- Three high-performance backends with automatic fallback: C extension, ctypes C, pure Python
- Cross-platform (Windows, Linux, macOS) and **zero external dependencies**
- Callback system for progress tracking and custom I/O; flexible output into pre-allocated tensors
- Duck-typed indices: any `__index__`-capable object accepted
- **Unified instruction protocol**: shell / batch / app share one instruction file format
- **External command execution**: `sh` plugin (system shell, interactive command line)
- **Dimension-generic linear algebra**: duck typed, `output` keyword pass-out, identical Python / C behaviour
- **Proactive plugin hosting** (`extension_layer.plugin`): `PluginPool` batch aggregation

---

## What's New

### v0.4.5 — Unified Instruction Protocol, Memory & Extension Primitives

**Instruction Protocol & Execution:**
- Unified instruction file protocol (shell / batch / app): `data[index]` references, `%` / `%%` unpacking, `let`, IF / WHILE control-flow blocks, region parameters, pre-imported built-ins, `import_all_module`, interrupt handling
- External command execution `sh` plugin: interactive / captured modes; `subdocker` synchronous child-Docker delegation
- Docker suspend/resume: cooperative step-boundary `suspend()` returns a `SuspendSnapshot`; `resume()` continues from the recorded cursor (each step runs exactly once)

**Sense / Generate:**
- Sense primitives: `threshold_map` / `threshold_match` position iterators; `elementwise_extract` read→map→write regions
- Generate: `transform_self` self-modifying element-wise transform

**Core:**
- Position callbacks: `position_map` / `elementwise_position` — region walk reporting logical coordinates `logical = real × scale − origin` (scale=0 legal; integral values → int; py/ctypes/pydll identical)

**Memory & Tools:**
- IO-stream memory: `IOStreamMemory` — `IOFile` / `IOMemory` as the memory carrier; delegating slots with working defaults provide MapMemory-style key/value storage (transactional commit, lazy restore, arbitrary hashable literal keys round-trip)
- Linear algebra module (duck-typed ops, Python/C parity); time tools; `PluginPool` proactive plugin aggregation; Database abstraction (`DatabaseMemory`)
- Reflex feedback hub: `Feedback` — Trigger-style paired (feedback object, function) registrations with a non-blocking receive trigger and delegating hub functions

**Parallel & GPU:**
- Super-parallel framework: `SuperParallel` (grid-style layered parallel — `sp[1,2,3]` scale setup / `sp.grid(dim)` coordinates / `sp(*args)` invocation / `getIdx()` layer hierarchy) with `DefaultParallel` (process-thread model); external GPU engines plug in via the executor protocol; verified on Intel Arc (OpenCL — ~6–9× pure-compute speedup)
- Unit mapper: `UnitMap` — variable-length symbol runs fold to fixed-length real flags (tensor-ready); atomic-scale `probe_dim`, fully iterative, recurrence-style appending, count statistics (`total`/`count`/`runs`/`most_common`/`count_vector`)

### v0.4.4 — Exploration Test Suite & Behavior-Driven Frameworks

- Exploration test suite: face/digits/captcha/image tasks, NLP clustering, data generation, autonomous Agent, GUI viewers
- Video abstraction & generation: L0–L3 hierarchy, prototypes and contrast points in SQLite; +20.2% generation improvement
- Agent web-search knowledge base: 118 chemical elements and 188 ISO 639 languages stored with provenance
- Behavior-composition memory Agent: von Neumann structure — behaviors as data in DB, fetch→decode→reflect-execute→write-back
- Generic executor: atomic instructions stored in DB, fixed code runs multi-logic; `$N` result references; plugin-style reflection
- Element-wise filter/mapping API (all three backends): `data_filter`, `data_mapping`, `threshold_filter` / `threshold_map`
- Logic layers: `event_context` delegated slots, relative-probability axiom system (A1–A5), `EventBinds` protocol container
- Fourier module: generic `dft` / `idft` / `power_spectrum`, multi-dimensional, recursion-free
- Nested control flow: flat `Sequence` / `Branch` / `Loop`; `ControlFlatten` iterative expansion
- Action layer async execution: `ExecuterDriver.call_all` on delegated background worker

### Earlier Versions

| Version | Theme | Key Highlights |
|---------|-------|----------------|
| **v0.4.3** | Robustness, Protocol-Style Delegation | C core hardening (18 fixes), element-wise API C99-strict, upper-layer fixes, relative-probability axioms, sense layer `data_match` |
| **v0.4.2** | Portability & Robustness | ARM/piwheels C99 fixes, empty-input consistency, 12 memory leaks fixed, duck typing, free-threaded 3.14t verified, zero-copy buffer protocol |
| **v0.4.1** | Architecture Upgrade | Stride+offset indexing, `infer_shape` / `__shape__` protocol, `load_data` bulk-copy, PyBuffer zero-copy, SIMD hints, free-threaded dual binaries |
| **v0.3.9** | Indexing architecture | Stride+offset fancy indexing, zero-copy views, 100% recursion-free |
| **v0.3.0** | Multi-backend release | C extension + ctypes, three-backend fallback, operator overloading |
| **v0.2.0** | Tensor system | N-dimensional tensor view, sliding window, cos/mod/cosmod metrics |
| **v0.1.0** | Initial release | Core cosine similarity, centre-surround antagonism, pure Python |

> Full changelog available in `History.txt`.

---

## Architecture

### Seven-Layer Cognitive Architecture

Biologically inspired design mimicking mammalian brain structure. Only the core layer is production-ready.

| # | Layer | Directory | Brain Structure | Maturity | Core Function |
|---|-------|-----------|-----------------|----------|---------------|
| 1 | Core | `core` | Brainstem / Cerebellum | ✅ Production | Local comparison, three-backend acceleration, free-thread support, element-wise filter/mapping |
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
pip install cos-comparison[test]    # with test dependencies
```

If no C compiler is available, installation succeeds with the pure Python backend only.

| Requirement | Details |
|-------------|---------|
| Python | 3.8+ (3.13+ for free-threaded builds) |
| C compiler | Optional (auto-fallback to pure Python) |
| Runtime deps | None |

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

Benchmarked on a 322×424×3 RGB image with 3×3 window (WSL Ubuntu x64, Python 3.14.4, gcc -O2):

| Backend | Time | Speedup | Free-thread |
|---------|------|---------|-------------|
| C Extension | 0.0001s | ~35× | ✅ Full (no GIL) |
| ctypes C | 0.0004s | ~9× | ✅ Full |
| Pure Python | 0.0035s | 1× | ✅ Full |

On 1000×1000 passive 3×3: C extension 0.08s (~112×) vs pure Python 8.98s.

**Memory:** zero-copy PyBuffer protocol (no data duplication on read), view-based slicing (stride+offset, no copy). C extension static memory footprint < 64 KB.

**Key performance features:** zero-copy PyBuffer, SIMD auto-vectorization (SSE/AVX/NEON with safe fallback), view-based slicing, stride indexing, recursion-free carry iteration, GIL release on compute paths.

---

## Testing

```bash
python -m pytest tests/                           # full suite
python -m pytest tests/test_core_algorithms.py -v # core algorithms only
```

The suite covers core algorithms, tensor ops, backend parity, empty/edge cases, upper layers, and import hygiene. Tests run on both traditional and free-threaded interpreters.

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
