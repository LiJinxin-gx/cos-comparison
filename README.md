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

### v0.5.1 — Brain-Layer Logic Hardening

- **Rule availability & validation** — rules without `Logic.TRUE` status excluded; `Atomic_proposition` rejects mismatched args; NaN-valued Variables are reflexive
- **Statement judge** — default judge decides correctness of full statements `(a, b[, limit[, is_true]])`; truth claims must be derivable within `limit`
- **Event domain & bindings** — strict-chain event domain (container events expand, scalars stay whole); duck-typed bind engines for any Mapping container
- **Value domain** — Decimal / Fraction probability chains work; `strict=` on probability functions
- **math_tool hardening** — C topology crash scenarios fixed; `_fourier.idft` accepts complex inputs
- **C99 & tri-compiler hardening** — five C units clean under MSVC `/Wall /WX` + GCC strict C99; GIL-safety crash cluster fixed
- **Tests** — protocol/robustness suite (39 cases) + statement-semantics (14 cases); full suite 836 OK

### v0.5.0 — Backend Consolidation & Extension Points

- **Single compiled C path** — ctypes backend retired, compatibility mapping in `config.json` keeps old call names working
- **Free-threaded support** — all C extensions declare `Py_MOD_GIL_NOT_USED` on Python 3.13+, reliable fallback on older interpreters
- **Extensibility** — `iterate=` slot for external engines (GPU/parallel); `transform1`/`transform2` per-value maps replace linear transforms
- **math_tool refactor** — duck-typed protocols replace hard-coded types across fourier/topology/unit_map
- **Recursion removal** — package-wide scan confirms zero real recursion
- **Benchmark** — OpenCL engine on Intel Arc: up to ~7300× pure Python, 257× C fast path

> Full changelog available in `History.txt`.

---

## Architecture

### Seven-Layer Cognitive Architecture

Biologically inspired design mimicking mammalian brain structure. Only the core layer is production-ready.

| # | Layer | Directory | Brain Structure | Maturity | Core Function |
|---|-------|-----------|-----------------|----------|---------------|
| 1 | Core | `core` | Brainstem / Cerebellum | ✅ Production | Local comparison, two-backend acceleration, free-thread support, element-wise filter/mapping |
| 2 | Sense | `sense_layer` | Sensory Cortex | 🟡 Early | Stimulus reception, raw feature extraction, data matching |
| 3 | Memory | `memory_layer` | Hippocampus | 🟡 Early | Short/long-term storage, IO-stream memory, database-backed persistence, protocol-based pluggable backends |
| 4 | Brain | `brain_layer` | Prefrontal Cortex | 🟠 Beta | Relative-probability logic (A1–A5 axioms), symbol logic, nested control flow, reflex feedback/monitor/trigger, context mapping, statement judge |
| 5 | Action | `action_layer` | Motor Cortex | 🟡 Early | Async execution driver, action result wrapping, delegated background workers |
| 6 | Generate | `generate_layer` | Broca's / Wernicke's | 🔵 Exploratory | Template-based reverse generation, multi-modal output (validated in exploration) |
| 7 | Extension | `extension_layer` | Association Cortex | 🔵 Exploratory | Plugin hosting, `PluginPool` batch aggregation, external engine integration |

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

- **Computer vision**: face recognition, captcha solving, multi-scale feature extraction, template-based generation
- **Audio**: music genre classification, spectral feature extraction, unsupervised speech recognition
- **Text**: token frequency analysis, hierarchical clustering, demand-driven memory
- **Video**: frame abstraction, scene segmentation, cross-video generation
- **Agent systems**: web-search knowledge base, behavior-composition Agent, generic executor
- **Learning mechanisms**: continuous-mapping hierarchical isolation, multi-template clustering, reflective self-correction, feedback-driven search
- **Reasoning**: top-down hypothesis testing, multi-level evidence verification, bidirectional A↔B mapping

All exploration uses the same core local-comparison engine — no deep learning, no backpropagation, no third-party ML dependencies.

### Performance

Real measured data on a single reference machine — full audit with
methodology, per-module tables, GPU comparisons and resource figures:
**[Performance & Resource Audit](docs/performance.md)**.

**Test platform:** Intel Core Ultra 5 125H (14C/18T) + Intel Arc Graphics
(112 CUs, driver 31.0.101.5382); Windows 11 x64; CPython 3.14.6. Numbers are
medians from the in-house harness (kept in the experiment area, not shipped);
data are nested Python float grids (deterministic LCG); the GPU path is
end-to-end (host↔device transfers included).

**Core — passive 3×3 window aggregation (cosmod, step 1, d (1,1)):**

| Size | Pure Python | C Extension | GPU engine (OpenCL) |
|------|------------:|------------:|--------------------:|
| 512² | 3 838 ms | 26.7 ms | 0.51 ms |
| 2048² | 55 573 ms | 418 ms | 6.96 ms |

The GPU engine (an external module injected through `iterate=`) runs 52–60×
the C extension and up to ~7 500× the pure Python reference; against a
single-thread numpy baseline on the same computation it is 15–41×.

**Other highlights** (512² unless noted, pure Python → C extension):

- `mean_local` 10 640 ms → 41.9 ms (254×), `local_variance` 4 687 ms → 24.0 ms (195×)
- `data_mapping` (per-element callback) 681 ms → 208 ms; `data_filter` 247 ms → 114 ms
- `cos` over 1e6 floats: 1 372 ms → 42.6 ms (32×)
- math_tool over 1e6 elements: C advantage 1.3–4.1× (status/output-passing API); `dft` 15.3×; unit-map fold 6.4×
- cold `import cos_comparison`: no measurable overhead over the interpreter baseline; core C backend +21 ms; four math_tool C extensions +117 ms
- compiled artifacts: core backend 316 KB; math_tool extensions 20–38 KB each

**Known issues found by the audit** (details in the audit doc): intermittent
access violation in the C topology graph types under repeated large builds
(Python reference stable; fix pending); C `_fourier.idft` rejects complex
inputs that the Python reference accepts; single-element `get_item`/`set_item`
have negative C ROI; large pure-Python rows vary up to ±2.5× from laptop
thermal throttling.

**Memory:** zero-copy PyBuffer protocol (no data duplication on read), view-based slicing (stride+offset, no copy); package import costs ~0.03 MB RSS (core C backend +1.2 MB, the four math_tool extensions +10 MB); a nested 2048² Python grid costs ~175 MB where a float32 buffer costs ~8 MB — use the buffer paths at scale; full resource figures in the audit.

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

> **Note:** Due to academic commitments, development speed may be slower and updates may be irregular. Issues and PRs are always welcome.

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
