# cos-comparison Documentation

> Complete documentation for the cos-comparison project. Start with [Getting Started](getting-started.md), then dive into the section you need.

---

## 📑 Documentation Map

### Getting Started Guides

| Document | Purpose | Key Contents | Reading Time |
|----------|---------|--------------|:------------:|
| 🚀 [Getting Started](getting-started.md) | First-time setup and basic usage | Installation, 1D/2D/3D examples, tensor creation, slicing, backend switching, troubleshooting | 5 min |
| 🔨 [Building](build.md) | Compile C backends from source | In-place builds, dual-compiler strict-mode (MSVC + Clang), free-threaded (cp314t) builds, test instructions | 10 min |
| 💻 [Command Line](command-line.md) | Unified instruction file protocol | Shell / batch / app syntax, namespace management, control flow blocks, linear algebra integration | 8 min |

### API Reference

| Document | Purpose | Key Contents | Reading Time |
|----------|---------|--------------|:------------:|
| 📖 [Core Module](api/core.md) | Core API surface | Functions, tensor types (`vector_map_as_tensor`), common parameters (`window_size`, `step`, `d`, `algorithm`), utility functions (`infer_shape`, `load_data`, `create_void_list`) | 15 min |
| 📖 [Passive Mode](api/passive-mode.md) | Edge / boundary detection | `cos_comparison_passive` — self-similarity sliding window, parameter guide, output shapes, 1D–4D examples | 8 min |
| 📖 [Active Mode](api/active-mode.md) | Template matching | `cos_comparison_active` — kernel-based matching, `kernel` parameter, output parameter, algorithm selection | 8 min |
| 📖 [Statistics](api/statistics.md) | Local statistics | `mean_local`, `local_variance` — windowed statistical measures, parameter reference | 5 min |
| 📖 [Cognitive Layer APIs](api/cognitive-layers.md) | Upper-layer APIs | Sense, memory, brain, action, generate, interface, data, test tools | 20 min |

### Architecture & Design

| Document | Purpose | Key Contents | Reading Time |
|----------|---------|--------------|:------------:|
| 🏗️ [Seven-Layer Architecture](architecture/seven-layer.md) | Brain-inspired layered design | Layer details, maturity levels, current implementation status, roadmap | 10 min |
| 🏗️ [Modular Architecture](architecture/modular-architecture.md) | Module responsibility boundaries | Three-flow decoupling (data/operation/control), dependency rules, attach-and-take data flow | 8 min |
| 🏗️ [Backend Management](architecture/backend-system.md) | Multi-backend system | Dynamic loading, configuration (`config.json`), API contract, fallback mechanism | 8 min |
| 📊 [Performance & Resources](performance.md) | Measured performance and resource audit | Environment, methodology, per-module Python/C/GPU benchmarks, memory and import figures, findings | 15 min |

### Theory & Principles

| Document | Purpose | Key Contents | Reading Time |
|----------|---------|--------------|:------------:|
| 🧠 [First Principle](principles/first-principle.md) | Theoretical foundation | Why local comparison produces information, centre-surround antagonism, biological inspiration | 8 min |
| 🧠 [Similarity Measures](principles/similarity-measures.md) | Three algorithms | cos (directional), mod (magnitude), cosmod (combined) — formulas, properties, use cases | 6 min |
| 🧠 [Dual Working Modes](principles/dual-mode.md) | Passive vs active | Passive (reflexive boundary detection), active (template matching), when to use which | 6 min |

### Exploration & Experiments

| Document | Purpose | Key Contents | Reading Time |
|----------|---------|--------------|:------------:|
| 🧪 [Exploration Tests](exploration/README.md) | Cross-task experiments | Face recognition, captcha solving, music genre, text clustering, video abstraction, Agent frameworks, hierarchical isolation memory, von Neumann behavior agents | 20 min |
| 🧩 [Core Identity: Extraction–Generation Inverse](exploration/core_identity_closure.md) | v0.5.3 latest result | One local comparison relation, forward extract / reverse generate, closure criterion, open-module platform, formal-backend digit closure 4e-6, faces 88.8%, video 95.8%, text 73.5%, corrected data ceiling | 12 min |
| 🧪 [Atomic Contrast Matching](exploration/atomic_contrast_matching.md) | Unsupervised structure-first matching | Binary contrast point sets, Jaccard similarity, hierarchical coarse-to-fine, speech recognition 47.1%, scaling/threshold/K sweeps | 10 min |

---

## 🗺️ Recommended Reading Path

```
Getting Started (run your first edge detection)
        ↓
Principles → First Principle (understand why local comparison works)
        ↓
API → Core Module (master the core API surface and common parameters)
        ↓
API → Passive Mode / Active Mode (choose the right mode for your task)
        ↓
Architecture → Seven-Layer (see the big picture beyond core)
        ↓
Exploration Tests (see what the project can do across domains)
```

---

## ⚡ Quick Reference

### Most Used APIs

| Need | Function | Doc |
|------|----------|-----|
| Full-tensor similarity | `cos(a, b)` | [Core](api/core.md) |
| Edge / boundary detection | `cos_comparison_passive(data, ...)` | [Passive Mode](api/passive-mode.md) |
| Template matching | `cos_comparison_active(data, kernel=..., ...)` | [Active Mode](api/active-mode.md) |
| Local mean / variance | `mean_local(data, ...)` / `local_variance(data, ...)` | [Statistics](api/statistics.md) |
| Element-wise filter | `data_filter(data, callback, ...)` | [Core](api/core.md) |
| Element-wise mapping | `data_mapping(data, callback, output=..., ...)` | [Core](api/core.md) |
| Threshold operations | `threshold_filter` / `threshold_map` / `threshold_judge` | [Core](api/core.md) |
| Tensor creation | `vector_map_as_tensor(vector=..., shape=...)` | [Core](api/core.md) |
| Shape inference | `infer_shape(data)` | [Core](api/core.md) |
| Bulk data loading | `load_data(src, dst, ...)` | [Core](api/core.md) |
| Backend switching | `set_mode(['c', 'py'])` | [Backend System](architecture/backend-system.md) |

### Common Patterns

```python
# 1. Edge detection on a 2D image (horizontal edges)
from cos_comparison.core import cos_comparison_passive
edges = cos_comparison_passive(image, window_size=(3, 3), step=(1, 1), d=(1, 0))

# 2. Template matching with output pre-allocation
from cos_comparison.core import cos_comparison_active, create_void_list
output = create_void_list((out_h, out_w))
cos_comparison_active(image, kernel=template, step=(1, 1), output=output)

# 3. Switch to fastest available backend
from cos_comparison.core import set_mode
set_mode(['c', 'py'])

# 4. Threshold filter: find positions where value > 0.5
from cos_comparison.core import data_filter
positions = list(data_filter(tensor, lambda v: v > 0.5))

# 5. Tensor with custom strides (view, no copy)
from cos_comparison.core import vector_map_as_tensor
t = vector_map_as_tensor(vector=flat_data, shape=(3, 4), strides=(4, 1))
```

### Backend Selection

| Backend | Call name | Import module | Speed | Requires |
|---------|-----------|---------------|-------|----------|
| C Extension | `c` | `cos_comparison_pydll` | 28–150× | C compiler at install time |
| Pure Python | `py` | `cos_comparison` | 1× | Nothing (always available) |

> Speed range from the v0.5.0 benchmark (see
> [Backend Management](architecture/backend-system.md#benchmark-v050)):
> window aggregation is ~28× the reference; callback-bound element-wise
> paths narrow to ~2×.

```python
from cos_comparison import core
print(core.get_mode())  # shows currently active backends
core.set_mode(['c', 'py'])  # prefer the C extension, fall back to pure Python
```

---

## 📌 Conventions

- **Code blocks** are tested patterns, not pseudocode.
- **Backend parity**: both backends (C extension / pure Python) expose the same API and produce bit-identical results unless a known divergence is documented.
- **Versioning**: non-core layers are under active development; the core module follows semantic versioning.
- **Zero dependencies**: the core package requires no third-party libraries at runtime.
- **Dimension aliases**: `cos_comparison_passive_1d` / `_2d` / `_3d` / `_4d` are aliases for the dimension-generic `cos_comparison_passive` — use whichever reads clearer.
- **Keyword-only**: `vector_map_as_tensor` constructors use keyword arguments for optional parameters (`vector_map_as_tensor(vector=data, shape=(2,3))`).

---

## 🔗 Related Resources

| Resource | Link |
|----------|------|
| Project Overview | [Root README](../README.md) |
| Full Changelog | [History.txt](../History.txt) |
| PyPI Package | [cos-comparison](https://pypi.org/project/cos-comparison/) |
| Source Code | [GitHub](https://github.com/LiJinxin-gx/cos-comparison) |
| Issue Tracker | [GitHub Issues](https://github.com/LiJinxin-gx/cos-comparison/issues) |
| Exploration Code | `explore/` directory (packaged with distribution) |
