# Cos Comparison Documentation

**Local similarity comparison for feature extraction — biologically inspired, zero-training edge/pattern detection.**

---

## Navigation

### Principles
- [First Principle](principles/first-principle.md) — Information arises from local comparison
- [Similarity Measures](principles/similarity-measures.md) — cos, mod, cosmod algorithms
- [Dual Working Modes](principles/dual-mode.md) — Passive and Active modes

### Architecture
- [Seven-Layer Cognitive Architecture](architecture/seven-layer.md) — Brain-inspired layered design
- [Backend Management System](architecture/backend-system.md) — Multi-backend dynamic loading

### API Reference
- [Core Module](api/core.md) — Core functions and common parameters
- [Passive Mode](api/passive-mode.md) — cos_comparison_passive
- [Active Mode](api/active-mode.md) — cos_comparison_active
- [Statistics Functions](api/statistics.md) — mean_local, local_variance

### Getting Started
- [Quick Start](getting-started.md) — 5-minute guide

---

## About

**cos-comparison** implements the **centre-surround antagonism** mechanism from neuroscience, extracting edges, textures, and keypoints using only sliding window similarity.

### Key Features

- **Zero training, zero labels** — Deterministic computation, ready to use
- **Dimension-agnostic** — Works on 1D, 2D, 3D, 4D and beyond
- **Dual modes** — Passive (self-similarity) + Active (template matching)
- **Multi-backend acceleration** — Pure Python / NumPy / C / Python C extension
- **Fully interpretable** — Every output has clear geometric meaning
- **Zero-dependency core** — Pure Python core works out of the box

### Use Cases

- Edge detection and feature extraction
- Template matching and object detection
- On-device intelligence with limited resources
- AI applications requiring interpretability
- Zero-shot / few-shot learning scenarios

---

## Related Links

- [GitHub Repository](https://github.com/LiJinxin/cos-comparison)
- [PyPI Project Page](https://pypi.org/project/cos-comparison/)
- [Issue Tracker](https://github.com/LiJinxin/cos-comparison/issues)
