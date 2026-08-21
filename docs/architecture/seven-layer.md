# Seven-Layer Cognitive Architecture

A **seven-layer brain-inspired cognitive architecture**, where each layer corresponds to a level of biological cognitive processing. This design enables modular development and clear separation of concerns.

## Contents

- [Overview](#overview)
- [Design Principles](#design-principles)
- [Layer Details](#layer-details)
- [Cross-Cutting Modules](#cross-cutting-modules)
- [Current State](#current-state)

---

## Overview

| Layer | Directory | Brain Analogue | Maturity | Core Function |
|-------|-----------|----------------|----------|---------------|
| 1 | `core` | Brainstem / Cerebellum | ✅ Production | Local comparison, stride indexing, multi-backend acceleration |
| 2 | `sense_layer` | Sensory Cortex | 🟡 Early | Stimulus reception (Receptor / TensorReceptor) |
| 3 | `memory_layer` | Hippocampus / Cortex | 🟡 Early | Map/table/database memory, transactions |
| 4 | `brain_layer` | Prefrontal Cortex | 🟡 Partial | Symbolic/probabilistic logic, reflex system |
| 5 | `action_layer` | Motor Cortex | 🔵 Exploratory | Action output (ExecuterDriver) |
| 6 | `generate_layer` | Broca's / Wernicke's | 🔵 Exploratory | Data generation (Generator / TensorGenerator) |
| 7 | `extension_layer` | Association Areas | 🔴 Skeleton | Extended capabilities |

**Cross-cutting modules:**

| Module | Role |
|--------|------|
| `interface` | External abstraction: system/process, call containers, communication, parallel tools, context tools |
| `data` | Data carriers and tensor containers |
| `test_tool` | Debug probes and benchmarking (Timer, MemoryProbe, ErrorWatcher, TraceProbe) |
| `app` | Aggregation of cross-layer functionality (planned) |

---

## Design Principles

### 1. Bottom-Up Dependency

Each layer depends only on layers below it, never above:

```
Layer 7 (Extension)
    ↑ depends on
Layer 6 (Generation)      Layer 5 (Action)
    ↑                                 ↑
Layer 4 (Brain)           Layer 3 (Memory)
    ↑                                 ↑
Layer 2 (Sense)           Layer 1 (Core)
```

This ensures independently testable layers, optimizable lower layers, and a system that remains functional even if higher layers are incomplete.

### 2. Neuroscience Homology

Each layer maps to a biological brain structure:

| Layer | Structure | Role |
|-------|-----------|------|
| Core | Brainstem / Cerebellum | Low-level, fast, automatic processing |
| Sense | Sensory Cortex | Initial feature extraction |
| Memory | Hippocampus | Storage and retrieval |
| Brain | Prefrontal Cortex | Reasoning and planning |
| Action | Motor Cortex | Execution of actions |
| Generate | Language areas | Complex output generation |
| Extension | Association areas | Integration |

### 3. Dual-Mode Consistency

The passive/active philosophy extends vertically:

| Layer | Passive | Active |
|-------|---------|--------|
| Core | Self-similarity | Template matching |
| Memory | Content-based retrieval | Pattern-based search |
| Brain | Bottom-up reasoning | Top-down planning |

---

## Layer Details

### Layer 1: Core ✅ Production

**Responsibilities:**
- Local comparison (passive self-similarity, active template matching) and statistics (mean, variance)
- Dimension-agnostic stride-based `vector_map_as_tensor` with fancy indexing, zero-copy views, negative indices, dimension collapse
- `__shape__` protocol / `infer_shape`, `load_as_default_data`, `load_data`, `vector_chain_compute`
- Similarity algorithms (cos, mod, cosmod) and callback system
- Three backends (C extension / ctypes / pure Python) with automatic fallback; free-threaded 3.13+ support; duck-typing for NumPy/PyTorch arrays

→ [Core Module API](../api/core.md)

### Layer 2: Sense 🟡 Early

**Responsibilities:** Input normalization, basic feature extraction, sensory adaptation.

**Current:** `Receptor` (wraps data, `point(index)` access), `TensorReceptor` (core `get_item` + `comparison_passive()`/`comparison_active()` shortcuts).

→ [Cognitive Layer APIs — Sense](../api/cognitive-layers.md#sense-layer)

### Layer 3: Memory 🟡 Early

**Responsibilities:** Short/long-term storage, consolidation, retrieval, dual-mode database.

**Current:** `Memory` (lifecycle with pluggable rules), `Status` flags, `MapMemory` (dict-backed, atomic transactions), `TableMemory` (nested-key), `DatabaseMemory` (SQLite), wrappers `MemoryWrap`/`MemoryWrapPool`/`MemoryWrapMap`.

→ [Cognitive Layer APIs — Memory](../api/cognitive-layers.md#memory-layer)

### Layer 4: Brain 🟡 Partial

**Responsibilities:** Symbolic/probabilistic logic, decision making, planning, metacognition.

**Current:**
- **Symbolic logic**: `Logic`, `Variable`, `Atomic_proposition`, `Logic_bind`, `Logic_context` (graph-reachability judge)
- **Probabilistic logic**: `event_bind`/`event_context` with relative-probability axioms, `EventBinds` cached-graph engine
- **Reflex**: `Trigger` (trigger→callback via shared stack), `Monitor` (pure logic, all mechanisms delegated to `interface`)

→ [Cognitive Layer APIs — Brain](../api/cognitive-layers.md#brain-layer)

### Layer 5: Action 🔵 Exploratory

**Responsibilities:** Action selection/execution, motor control, environment interaction.

**Current:** `ExecuterDriver` — ordered callable list; `call_all` runs strictly in order on a delegated background worker (non-blocking, `wait`/timeout, per-item `out`/`err` via `ActionResult`).

→ [Cognitive Layer APIs — Action](../api/cognitive-layers.md#action-layer)

### Layer 6: Generate 🔵 Exploratory

**Responsibilities:** Language/image/signal generation, output formatting.

**Current:** `Generator` (applies callable), `TensorGenerator` (`generate(func)` unified entry + `set_point`), `copy_region` region fill.

→ [Cognitive Layer APIs — Generate](../api/cognitive-layers.md#generate-layer)

### Layer 7: Extension 🔴 Skeleton

**Responsibilities:** Specialized capabilities, cross-modal integration, plugin system.

**Current:** Empty package.

---

## Cross-Cutting Modules

| Module | Details |
|--------|---------|
| `interface` | System/process control, call containers, IO/socket/pipe/file communication, locks/shared arrays/threads, `AsyncRunner`/`EventLoop`, context tools. Stdlib only. Imports cleanly. |
| `data` | `DataWrap` carrier; `Tensor`/`SafeTensor`/`ParallelTensor` over core tensor. Imports cleanly. |
| `test_tool` | `Timer`, `perf_count`, `ResultManager`, `MemoryProbe`, `ErrorWatcher`, `TraceProbe`, decorators. Stable. |
| `app` | Skeleton aggregation (planned). |

→ [Cognitive Layer APIs](../api/cognitive-layers.md)

---

## Current State

### Strengths

- Clear architectural vision with strong neuroscience inspiration
- Production-ready core with solid, well-tested implementation
- Clean separation of concerns and dependency direction (layers never depend on each other)
- Dual-mode philosophy provides conceptual coherence
- Memory layer has functional backends (map/table/database) with transaction support
- Brain layer has symbolic + probabilistic logic and a mechanism-free reflex system
- Three backends fully API-aligned with zero-warning C code; v0.4.3 adds memory-safety fixes, protocol-style logic layers, and default dict-protocol implementations

### Gaps

- Action and generation layers have minimal primitives
- Extension layer is an empty skeleton
- No clear roadmap for layer-by-layer development

### Recommendations

1. **Prioritize memory layer** — bridge from perception to cognition
2. **Flesh out brain layer** — more reasoning capabilities
3. **Define clear interfaces** between layers before extensive implementation
4. **Build vertical demos** showing multiple layers working together

---

**Next:** [Backend Management System](backend-system.md) · [Modular Architecture](modular-architecture.md)
