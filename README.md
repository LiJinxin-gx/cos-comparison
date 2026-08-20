# Cos Comparison

[![PyPI version](https://badge.fury.io/py/cos-comparison.svg)](https://pypi.org/project/cos-comparison/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**An AGI-oriented project based on local similarity comparison for feature extraction – biologically inspired, zero-training edge / pattern detection.**

---

## 💡 Core Idea

Information is produced by **local comparison** in raw data. This module implements the **centre-surround antagonism** mechanism from neuroscience, extracting edges, textures, and keypoints using only sliding-window similarity.

The core formula (cosine-modulated similarity, recommended default):

$$
\text{cosmod} = \frac{2(A \cdot B)}{|A|^2 + |B|^2}
$$

Three similarity measures are provided, selected via the `algorithm` parameter:

| Measure | Formula | Use Case |
|---------|---------|----------|
| **cos** | `(A·B) / (\|A\|·\|B\|)` | Directional similarity, edge orientation |
| **mod** | `2\|A\|·\|B\| / (\|A\|²+\|B\|²)` | Magnitude similarity, blob/region detection |
| **cosmod** | `2(A·B) / (\|A\|²+\|B\|²)` | Combined (default), edge/keypoint detection |

Key features:

- **No training, no labels, no backpropagation** — a step toward biologically plausible AGI
- Works on **1D–4D** data (audio, images, video, volumetric data)
- **Passive** (reflexive boundary detection) and **active** (template matching) modes
- Three high-performance backends with automatic fallback: C extension, ctypes C, pure Python
- Cross-platform (Windows, Linux, macOS) and **zero external dependencies**
- Callback system for progress tracking and custom I/O; flexible output into pre-allocated tensors
- Duck-typed indices: any `__index__`-capable object accepted

---

## 🚀 What's New

**v0.4.3** — robustness, protocol-style delegation and default-algorithm release:
- Element-wise filter/mapping API (`data_filter`, `data_mapping`, `threshold_filter`/`threshold_map`/`threshold_judge`) across all backends
- C core hardening: memory-safety NULL checks, single-fire callbacks, `step=0` protection, output bounds checks, overflow guards
- Protocol-style logic layers with relative-probability axiom system and `EventBinds` graph engine
- Fourier transforms (`interface.tools.math_tool.fourier`): generic multi-dimensional DFT/IDFT, recursion-free
- Default dict-protocol implementations; `TensorGenerator.generate` unified delegation; `data_match` template matching
- Upper-layer defect fixes (database atomicity, process management, topology locks, scalar-index parity)

**v0.4.2** — portability & robustness:
- ARM/piwheels C99 fixes, empty-input consistency
- Exhaustive malloc NULL checks; 12 reference/memory leaks fixed
- Duck typing (`PyNumber_Index`); free-threaded Python 3.14t verified
- Zero-copy buffer protocol

**v0.4.1** — architecture upgrade:
- Stride+offset indexing migration, `infer_shape` / `__shape__` protocol
- `load_data` bulk-copy, keyword-only constructors, PyBuffer zero-copy
- SIMD hints, slice performance, free-threaded dual binaries

See [History.txt](History.txt) for the full changelog.

---

## 🧬 Seven-Layer Cognitive Architecture

Biologically inspired seven-layer architecture mimicking mammalian brain structure. Only the core layer is production-ready; others are in early development.

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

- **`core`** — low-level algorithms and data format; standard library only, fully independent
- **`interface`** — all external interaction (processes, locks, shared memory, async, IO); standard library only
- **`data`** — data carrying and generic abstraction (`DataWrap`, `Tensor`/`SafeTensor`/`ParallelTensor`)
- **Five functional layers** — attach to incoming data, consume core algorithms, delegate mechanisms to `interface`

Dependency rules: layers must not depend on each other; data is attach-and-take (never owned/transformed by layers); all boundary crossing goes through `interface`.

See [docs/architecture/](docs/architecture/) for full details.

---

## ⚡ Performance

Benchmarked on a 322×424×3 RGB image with 3×3 window (Windows 11 x64, Python 3.14.6, MSVC -O2):

| Backend | Time | Speedup | Free-thread |
|---------|------|---------|-------------|
| C Extension | 0.004s | ~130× | ✅ Full (no GIL) |
| ctypes C | 0.007s | ~70× | ✅ Full |
| Pure Python | 0.52s | 1× | ✅ Full |

On Intel N150 (1000×1000, 3×3 passive): C extension 14s (9×), free-threaded 4 threads 4.6s (27×) vs pure Python 126s.

Key performance features: zero-copy PyBuffer, SIMD auto-vectorization (SSE/AVX/NEON), view-based slicing, stride indexing, recursion-free carry iteration, GIL release on compute paths.

---

## 📦 Installation

```bash
pip install cos-comparison          # core package (attempts C compilation automatically)
pip install cos-comparison[test]    # with test dependencies
```

If no C compiler is available, installation succeeds with the pure Python backend only.

**Requirements**: Python 3.8+ (3.13+ for free-threaded builds), optional C compiler, no runtime dependencies.

To recompile after source changes: `python setup.py build_ext --inplace`

---

## 🚀 Quick Start

```python
from cos_comparison import core

# 64x64 checkerboard
data = core.create_void_list((64, 64))
for i in range(64):
    for j in range(64):
        data[i, j] = 1.0 if (i + j) % 2 == 0 else 0.0

# Passive mode: sliding-window self-similarity
edges = core.cos_comparison_passive(data, window_size=(3, 3))
print(edges.shape)  # (61, 62)

# Displacement vector d shifts the second window (edge orientation)
edges2 = core.cos_comparison_passive(data, window_size=(3, 3), d=(0, 1))

# Active mode: template matching
kernel = core.create_void_list((3, 3), default=1.0)
matches = core.cos_comparison_active(data, kernel=kernel)

# Switch backends
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

## 👤 Author

I was born on May 31, 2008, and feel fortunate to grow up in an era of rapid progress in artificial intelligence.

I have run extensive tests and observed many surprising emergent properties. Earlier versions had numerous issues, as examination-oriented education left me limited time for thorough testing. I now have the opportunity to properly test and refine this work.

There remains a long road to true AGI. I may be forced to set aside this research due to personal circumstances, but I do not want these ideas to fade unnoticed. The purpose of open-sourcing is to share my thoughts, in the hope that others may build upon them.

---

## 📚 Documentation

- [Getting Started](docs/getting-started.md) — 5-minute tutorial
- [API Reference](docs/api/README.md) — core, passive/active modes, statistics, upper layers
- [Architecture](docs/architecture/README.md) — seven-layer design, modular architecture, backends
- [Principles](docs/principles/README.md) — first principle, similarity measures, dual modes

## 📫 Contact

- **GitHub**: [LiJinxin-gx/cos-comparison](https://github.com/LiJinxin-gx/cos-comparison)
- **Issues**: [GitHub Issues](https://github.com/LiJinxin-gx/cos-comparison/issues)
- **Email**: lijinxin_gx@sina.cn
- **PyPI**: [cos-comparison](https://pypi.org/project/cos-comparison/)

## 📄 License

MIT © 2026 Li Jinxin. See [LICENSE](LICENSE.txt) for details.
