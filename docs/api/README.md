# API Reference

Detailed API documentation for cos-comparison.

## Documents

| Doc | Contents |
|-----|----------|
| [Core Module](core.md) | Functions, tensor types, common parameters, utility details |
| [Passive Mode](passive-mode.md) | `cos_comparison_passive` — self-similarity |
| [Active Mode](active-mode.md) | `cos_comparison_active` — template matching |
| [Statistics Functions](statistics.md) | `mean_local`, `local_variance` |
| [Cognitive Layer APIs](cognitive-layers.md) | Sense, memory, brain, action, generate, interface, extension, data, test tools |

## Additional Interfaces

| Module | Contents |
|--------|----------|
| `interface.api.time_api` | `timestamp`, `iso_time`, `format_time`, `parse_time`, `sleep`, `elapsed`; types `TimeStamp`, `Stopwatch`, `Deadline` |
| `extension_layer.plugin` | `PluginPool` (resources / plugins / func_pool keyed pools, proactive hosting) |

---

**Related:** [Architecture](../architecture/README.md) · [Principles](../principles/README.md) · [Getting Started](../getting-started.md)
