# cos-comparison Documentation

> Complete documentation for the cos-comparison project. Start with [Getting Started](getting-started.md), then dive into the section you need.

---

## 📑 Sections

| Section | Purpose | Reading Time |
|---------|---------|:------------:|
| 🚀 [Getting Started](getting-started.md) | 5-minute tutorial: install, quick examples, tensors, backends, troubleshooting | 5 min |
| 🔨 [Building](build.md) | Build C backends, dual-compiler strict-mode checks, free-threaded (cp314t) builds, tests | 10 min |
| 💻 [Command Line](command-line.md) | Unified instruction files: shell / batch / app syntax, namespace, linear algebra | 8 min |
| 📖 [API Reference](api/README.md) | Core module, passive/active modes, statistics, cognitive layer APIs | 15 min |
| 🏗️ [Architecture](architecture/README.md) | Seven-layer design, modular architecture, backend management | 10 min |
| 🧪 [Exploration Tests](exploration/README.md) | Cross-task exploratory tests, learning mechanisms, Agent & generation experiments | 20 min |
| 🧠 [Principles](principles/README.md) | Theoretical foundations: first principle, similarity measures, dual modes | 8 min |

---

## 🗺️ Recommended Reading Path

```
Getting Started (run your first edge detection)
        ↓
Principles → First Principle (understand why local comparison works)
        ↓
API → Core Module (master the core API surface and common parameters)
        ↓
Architecture → Seven-Layer (see the big picture)
        ↓
Exploration Tests (see what the project can do beyond the core)
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
| Threshold operations | `threshold_filter` / `threshold_map` | [Core](api/core.md) |
| Tensor creation | `vector_map_as_tensor(vector=..., shape=...)` | [Core](api/core.md) |
| Backend switching | `set_mode(['cos_comparison_c', 'cos_comparison'])` | [Backend System](architecture/backend-system.md) |

### Common Patterns

```python
# 1. Edge detection on a 2D image
from cos_comparison.core import cos_comparison_passive
edges = cos_comparison_passive(image, window_size=(3, 3), step=(1, 1), d=(1, 0))

# 2. Template matching
from cos_comparison.core import cos_comparison_active
response = cos_comparison_active(image, kernel=template, step=(1, 1))

# 3. Switch to fastest available backend
from cos_comparison.core import set_mode
set_mode(['cos_comparison_pydll', 'cos_comparison_c', 'cos_comparison'])
```

---

## 📌 Conventions

- **Code blocks** are tested patterns, not pseudocode.
- **Backend parity**: all three backends (C extension / ctypes / pure Python) expose the same API and produce bit-identical results unless a known divergence is documented.
- **Versioning**: non-core layers are under active development; the core module follows semantic versioning.
- **Zero dependencies**: the core package requires no third-party libraries at runtime.

---

## 🔗 Related Resources

| Resource | Link |
|----------|------|
| Project Overview | [Root README](../README.md) |
| Full Changelog | [History.txt](../History.txt) |
| PyPI Package | [cos-comparison](https://pypi.org/project/cos-comparison/) |
| Source Code | [GitHub](https://github.com/LiJinxin-gx/cos-comparison) |
| Issue Tracker | [GitHub Issues](https://github.com/LiJinxin-gx/cos-comparison/issues) |
