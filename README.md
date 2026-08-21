# Cos Comparison

[![PyPI version](https://badge.fury.io/py/cos-comparison.svg)](https://pypi.org/project/cos-comparison/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**An AGI-oriented project based on local similarity comparison for feature extraction — biologically inspired, zero-training edge / pattern detection.**

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

**Key features:**

- **No training, no labels, no backpropagation** — a step toward biologically plausible AGI
- Works on **1D–4D** data (audio, images, video, volumetric data)
- **Passive** (reflexive boundary detection) and **active** (template matching) modes
- Three high-performance backends with automatic fallback: C extension, ctypes C, pure Python
- Cross-platform (Windows, Linux, macOS) and **zero external dependencies**
- Callback system for progress tracking and custom I/O; flexible output into pre-allocated tensors
- Duck-typed indices: any `__index__`-capable object accepted

---

## 🚀 What's New

**v0.4.3** — robustness, protocol-style delegation, default-algorithm release:

- **C core hardening (18 fixes, API unchanged)**: memory-safety NULL checks, single-fire end/return callbacks, `step=0` division protection, output write-bounds checks, short-parameter tolerance, integer-overflow guards, `global_error` callback ABI alignment (pydll + ctypes backends)
- **Element-wise filter/mapping API (all three backends)**: `data_filter`, `data_mapping`, `threshold_filter` / `threshold_map` (interval `[low, high]` with inclusive endpoint control). C implementation is C99-strict (no VLA/GNU extensions, all static, const-correct, shared `_resolve_read_region` core, long-long totals, explicit free on every path) and backend-switch safe
- **Upper-layer defect fixes**: database `executemany` partial-replay removal, `MapMemory.commit` atomicity, `Process.stop(terminate=True)` child termination + synchronous reader threads, monitor handle race via `parallel_lock`, topology nested-lock deadlocks, `Receptor.receptor` kwargs=None handling
- **Logic layers (protocol-style)**: `event_context` delegated slots (`init_func`/`add_func`/`probability_func`), relative-probability axiom system (A1–A5), two-stage rigorous resolution, `EventBinds` binds-as-engine protocol container (cached graph, stats, strict switch), default graph-reachability judge (`logic_judge`)
- **Default dict-protocol implementations**: `Map` three-slot defaults (mapper), truthiness substitution → `is not None` default construction (call_api / executer / symbol_logic / logic contexts)
- **Generate layer**: `TensorGenerator.generate(func, args, kwargs)` unified delegation entry, `set_point` via the core `set_item` protocol, module-level `copy_region` region fill (core `load_data` wrapper, target-first)
- **Sense layer**: `data_match` — integrated matching-position iterator (active comparison + threshold filter; algorithm default resolved inside the backend, pydll-safe; None region parameters omitted so backends treat them uniformly); `Receptor.receptor` delegation entry
- **Fourier module** (`interface.tools.math_tool.fourier`): generic `dft` / `idft` / `power_spectrum` from the cos/sin integral formula — multi-dimensional (axis-wise, odometer), recursion-free, trig-variant float arithmetic (no complex-object overhead), iterative loops only
- **get_item scalar-index parity**: the pure-Python and ctypes backends now treat a scalar index as a 1-D index like the C extension
- **Version handling**: VERSION.txt restored at the package directory, `__init__.py` reads it with `.strip()` and `utf-8-sig` (clean `version_tuple`, BOM/newline tolerant); `setup.py` writes the pure version string and reads pyproject.toml BOM-tolerantly; version set to 0.4.3
- **Nested control flow (`brain_layer.control`)**: flat `Sequence` / `Branch` / `Loop` containers yield FUNCTIONS (caller executes, containers never invoke); `ControlFlatten` expands nested flows iteratively (explicit stack, recursion-free, custom flattening via subclassing); `ControlFlowDriver` composes flows dynamically through the sequence protocol, mapping keys and multi-index access; `Control` is a pure placeholder marker
- **Function helper tools (`interface.tools.func_tool`)**: `ComposalFunction` composes multiple functions into one callable over a shared stack (first/last direct, middle steps read configured slots, maintenance delegated to `ComposalFunctionManage`); `FuncHelper` wraps stdlib helpers (partial / reduce / compose / wraps / itemgetter / attrgetter) with injectable slots; `FuncWrap` discard-first-arg wrapper
- **Test tools**: attach-style debug probes — `ResultManager` output collection (injectable `output` callback, no standard-stream writes), `MemoryProbe` (tracemalloc-backed, injectable slots), `ErrorWatcher` (run-space protection: errors recorded then swallowed), `TraceProbe` (call-level tracing); decorators `@memory_report` / `@error_watch` / `@trace_report`
- **Action layer async execution**: `ExecuterDriver.call_all` runs the call list strictly in order on a delegated background worker — non-blocking, unlimited/limited `wait`, per-item captured `out`/`err` via `ActionResult`; thread launching stays inside `interface`
- **Core hot injection**: the full backend API is hot-injected into the `core` module namespace (zero-overhead access) with explicit `__all__` (including private-but-exported `_cos` / `_mod` / `_cosmod`); `__all__` added across every `import *` source module to prevent namespace pollution
- **Docs**: README, cognitive-layers / modular-architecture / seven-layer / core / backend-system updated for v0.4.3

**v0.4.2** — portability & robustness:

- ARM/piwheels C99 fixes, empty-input consistency, exhaustive malloc NULL checks
- 12 reference/memory leaks fixed; duck typing (`PyNumber_Index`); free-threaded 3.14t verified
- Zero-copy buffer protocol

**v0.4.1** — architecture upgrade:

- Stride+offset indexing, `infer_shape` / `__shape__` protocol, `load_data` bulk-copy
- Keyword-only constructors, PyBuffer zero-copy, SIMD hints, slice performance, free-threaded dual binaries

> See [History.txt](History.txt) for the full changelog.

---

## 🧬 Seven-Layer Cognitive Architecture

Biologically inspired architecture mimicking mammalian brain structure. Only the core layer is production-ready.

| # | Layer | Directory | Brain Structure | Maturity | Core Function |
|---|-------|-----------|-----------------|----------|---------------|
| 1 | Core | `core` | Brainstem / Cerebellum | ✅ Production | Local comparison, three-backend acceleration, free-thread support |
| 2 | Sense | `sense_layer` | Sensory Cortex | 🟡 Early | Stimulus reception, raw feature extraction |
| 3 | Memory | `memory_layer` | Hippocampus | 🟡 Early | Short/long-term storage |
| 4 | Brain | `brain_layer` | Prefrontal Cortex | 🟡 Early | Cognition, logical reasoning |
| 5 | Action | `action_layer` | Motor Cortex | 🔵 Exploratory | Action output, environment interaction |
| 6 | Generate | `generate_layer` | Broca's / Wernicke's | 🔵 Exploratory | Language, image generation |
| 7 | Extension | `extension_layer` | Association Cortex | 🔴 Skeleton | Extended capabilities |

> Non-core layers do not affect core API stability. `cos_comparison.core` follows semantic versioning.

---

## 🏗️ Module Architecture

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

**Key performance features:** zero-copy PyBuffer, SIMD auto-vectorization (SSE/AVX/NEON), view-based slicing, stride indexing, recursion-free carry iteration, GIL release on compute paths.

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
from cos_comparison import core

# --- Create a 64x64 checkerboard ---
data = core.create_void_list((64, 64))
for i in range(64):
    for j in range(64):
        data[i, j] = 1.0 if (i + j) % 2 == 0 else 0.0

# --- Passive mode: sliding-window self-similarity ---
edges = core.cos_comparison_passive(data, window_size=(3, 3))
print(edges.shape)  # (61, 62)

# Displacement vector d shifts the second window (edge orientation)
edges2 = core.cos_comparison_passive(data, window_size=(3, 3), d=(0, 1))

# --- Active mode: template matching ---
kernel = core.create_void_list((3, 3), default=1.0)
matches = core.cos_comparison_active(data, kernel=kernel)

# --- Switch backends ---
core.set_mode(".cos_comparison")  # force pure Python for debugging
```

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
| [Getting Started](docs/getting-started.md) | 5-minute tutorial |
| [API Reference](docs/api/README.md) | Core, passive/active modes, statistics, upper layers |
| [Architecture](docs/architecture/README.md) | Seven-layer design, modular architecture, backends |
| [Principles](docs/principles/README.md) | First principle, similarity measures, dual modes |

---

## 👤 Author

I was born on May 31, 2008, and feel fortunate to grow up in an era of rapid progress in artificial intelligence.

I have run extensive tests and observed many surprising emergent properties. Earlier versions had numerous issues, as examination-oriented education left me limited time for thorough testing. I now have the opportunity to properly test and refine this work.

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
