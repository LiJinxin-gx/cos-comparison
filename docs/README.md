# cos-comparison Documentation

Welcome to the cos-comparison documentation. The docs are organized into four sections:

| Section | Purpose |
|---------|---------|
| [Getting Started](getting-started.md) | 5-minute tutorial: install, quick examples, tensors, backends, troubleshooting |
| [API Reference](api/README.md) | Core module, passive/active modes, statistics, cognitive layer APIs |
| [Architecture](architecture/README.md) | Seven-layer design, modular architecture, backend management |
| [Principles](principles/README.md) | Theoretical foundations: first principle, similarity measures, dual modes |

## Recommended Reading Path

1. **[Getting Started](getting-started.md)** — run your first edge detection
2. **[Principles → First Principle](principles/first-principle.md)** — why local comparison
3. **[API → Core Module](api/core.md)** — the core API surface and common parameters
4. **[Architecture → Seven-Layer](architecture/seven-layer.md)** — the big picture

## Quick Links

- **Core API**: [core.md](api/core.md) — functions, tensor types, common parameters
- **Passive mode**: [passive-mode.md](api/passive-mode.md)
- **Active mode**: [active-mode.md](api/active-mode.md)
- **Statistics**: [statistics.md](api/statistics.md)
- **Upper layers**: [cognitive-layers.md](api/cognitive-layers.md)
- **Backend switching**: [backend-system.md](architecture/backend-system.md)

## Conventions

- Code blocks are tested patterns, not pseudocode.
- All backends (C extension / ctypes / pure Python) expose the same API and produce bit-identical results unless a known divergence is documented.
- Non-core layers are under active development; the core module follows semantic versioning.

> See the root [README](../README.md) for the project overview and the [History](../History.txt) for the changelog.
