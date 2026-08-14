# Seven-Layer Cognitive Architecture

Cos-comparison adopts a **seven-layer brain-inspired cognitive architecture**, where each layer corresponds to different levels of biological cognitive processing. This layered design enables modular development and clear separation of concerns.

## Overview

| Layer | Directory | Brain Analogue | Maturity | Core Function |
|-------|-----------|----------------|----------|---------------|
| 1 | `core` | Brainstem / Cerebellum | ✅ Production | Low-level local comparison computation, stride-based tensor indexing, multi-backend acceleration |
| 2 | `sense_layer` | Sensory Cortex | 🟡 Early Development | Receive external stimuli (Receptor / TensorReceptor) |
| 3 | `memory_layer` | Hippocampus / Cortex | 🟡 Early Development | Short-term and long-term memory (map, table, database backends) |
| 4 | `brain_layer` | Prefrontal Cortex | 🟡 Early Development | Symbolic logic, probabilistic logic, reflex system |
| 5 | `action_layer` | Motor Cortex | 🔵 Exploratory Development | Action output control (ExecuterDriver) |
| 6 | `generate_layer` | Broca's / Wernicke's Area | 🔵 Exploratory Development | Generate data by matching data structures (Generator) |
| 7 | `extension_layer` | Association Areas | 🔴 Skeleton | Extended functions and special capabilities |

Cross-cutting modules that serve every layer:

| Module | Role |
|--------|------|
| `interface` | External interface abstraction: system/process control, call containers, communication, parallel tools, context tools |
| `data` | Unified data carrier interfaces and tensor containers |
| `test_tool` | Testing and debugging utilities (Timer, perf_count) |
| `app` | Aggregation of functionality across layers |

## Design Principles

### 1. Bottom-Up Dependency

Each layer only depends on layers below it, never on layers above.

```
Layer 7 (Extension)
    ↑ depends on
Layer 6 (Generation)      Layer 5 (Action)
    ↑                                 ↑
Layer 4 (Brain)           Layer 3 (Memory)
    ↑                                 ↑
Layer 2 (Sense)           Layer 1 (Core)
```

This ensures clean separation of concerns, independently testable layers, optimizable lower layers without affecting higher ones, and a system that remains functional even if higher layers are incomplete.

### 2. Neuroscience Homology

Each layer corresponds to a biological brain structure — low-level, fast, automatic processing (core → brainstem/cerebellum), initial feature extraction (sense → sensory cortex), storage and retrieval (memory → hippocampus), reasoning and planning (brain → prefrontal cortex), execution of actions (action → motor cortex), generation of complex outputs (generate → language areas), and integration (extension → association areas).

### 3. Dual-Mode Consistency

The passive/active dual-mode philosophy extends vertically through all layers — e.g. core: self-similarity vs template matching; memory: content-based retrieval vs pattern-based search; brain: bottom-up reasoning vs top-down planning.

## Layer Details

### Layer 1: Core

**Status**: ✅ Production (v0.4.2)

**Responsibilities**:
- Fundamental local comparison computation (passive self-similarity, active template matching) and statistics (mean, variance)
- Dimension-agnostic (1D, 2D, 3D, 4D, ...); stride-based `vector_map_as_tensor` with NumPy-like fancy indexing, zero-copy views, arbitrary slices/steps, negative indices, dimension collapse
- `__shape__` protocol / `infer_shape` (PyBuffer > `__shape__` > length detection), `load_as_default_data`, `load_data`, `vector_chain_compute`
- Similarity algorithms (cos, mod, cosmod) and callback system
- Zero-dependency pure Python reference plus three acceleration backends with automatic fallback (C extension / ctypes C / pure Python); full free-threaded Python (3.13+) support; duck-typing support for NumPy/PyTorch arrays

**Full API**: [Core Module API](../api/core.md)

### Layer 2: Sense Layer

**Status**: 🟡 Early Development

**Responsibilities**: Input normalization and preprocessing, basic feature extraction pipelines, sensory adaptation.

**Current implementation**: `Receptor` (wraps any data, `point(index)` element access) and `TensorReceptor` (memoryview-based, `comparison_passive()`/`comparison_active()` shortcuts to core modes). **Details**: [Cognitive Layer APIs — sense layer](../api/cognitive-layers.md#sense-layer)

**Design ideas**: standardized input interfaces for different data types, automatic dimension detection, sensory attention mechanisms (spatial/temporal), adaptation to input statistics.

### Layer 3: Memory Layer

**Status**: 🟡 Early Development (functional memory backends implemented)

**Responsibilities**: Short/long-term memory storage, memory consolidation, retrieval, dual-mode database (passive + active).

**Current implementation**: Lifecycle memory (`BaseMemory`/`Memory` with pluggable rules), `Status` flag (READ/WRITE/EXECUTE), `MapMemory` (dict-backed, transactions, atomic commit), `TableMemory` (nested-key convenience), `DatabaseMemory` (SQLite-backed), wrappers `MemoryWrap`/`MemoryWrapPool`/`MemoryWrapMap` with hierarchical tagging, `Transaction`. **Details**: [Cognitive Layer APIs — memory layer](../api/cognitive-layers.md#memory-layer)

**Design ideas**: key-value store abstraction, vector similarity search, memory decay and reinforcement, episodic vs semantic memory distinction, database backend pluggability.

### Layer 4: Brain Layer

**Status**: 🟡 Partial (logic + reflex implemented)

**Responsibilities**: Symbolic/probabilistic logic and reasoning, decision making, planning, metacognition.

**Current implementation**: Three-valued style symbolic logic (`Logic`, `Variable`, `Atomic_proposition`, `Logic_bind`, `Logic_context`, `No_limit`), probabilistic logic (`UnionEvent`/`IntersectionEvent`/`GlobalEvent`, `event_bind`, `event_context`), reflex (`Trigger` binds trigger→callback via shared result stack; `Monitor` is pure monitoring logic — all mechanisms delegated to `interface`). **Details**: [Cognitive Layer APIs — brain layer](../api/cognitive-layers.md#brain-layer)

**Design ideas**: three-valued logic (addressing hallucination), rule-based reasoning engine, goal hierarchy management, planning with subgoals, confidence estimation.

### Layer 5: Action Layer

**Status**: 🔵 Exploratory Development

**Responsibilities**: Action selection/execution, motor control, environment interaction, outcome evaluation, reflex arcs.

**Current implementation**: `ExecuterDriver` — ordered callable list with indexed invocation. **Details**: [Cognitive Layer APIs — action layer](../api/cognitive-layers.md)

**Design ideas**: action primitive library, action sequencing, feedback-based adjustment, exploration vs exploitation balance, innate vs learned reflexes.

### Layer 6: Generate Layer

**Status**: 🔵 Exploratory Development

**Responsibilities**: Language/image/signal generation, output formatting, expression of internal states.

**Current implementation**: `Generator` (applies a callable to wrapped data), `TensorGenerator` (memoryview-based point writing). *Most stub-like layer — generation primitives only.* **Details**: [Cognitive Layer APIs — generate layer](../api/cognitive-layers.md)

**Design ideas**: template-based generation, compositional generation, style control, multi-modal output generation.

### Layer 7: Extension Layer

**Status**: 🔴 Skeleton (empty package)

**Responsibilities**: Specialized capabilities, cross-modal integration, plugin system for extensions.

**Design ideas**: plugin architecture, cross-modal association, meta-learning capabilities, social cognition modules.

## Cross-Cutting Modules

- **`interface`** — system/process control, call containers (Python modules & native libraries), IO/socket/pipe/file communication, thread/process locks & shared arrays & `run_in_thread`, blocking async event host `AsyncRunner` and periodic task host `EventLoop`, context tools. Imports cleanly.
- **`data`** — `DataWrap` generic carrier; `Tensor`/`SafeTensor`/`ParallelTensor` over `core.vector_map_as_tensor`. Imports cleanly.
- **`test_tool`** — `Timer` (`mark`/`get_time`/`reset`), `perf_count = time.perf_counter`. Stable.
- **`app`** — skeleton aggregation of cross-layer functionality (planned).

Full class-level details: [Cognitive Layer APIs](../api/cognitive-layers.md)

## Current State Assessment (v0.4.2)

### Strengths
- Clear architectural vision with strong neuroscience inspiration
- Production-ready core layer with solid, well-tested implementation
- Clean separation of concerns and dependency direction (bottom-up, layers never depend on each other)
- Dual-mode philosophy provides conceptual coherence
- Memory layer has functional backends (map/table/database) with transaction support
- Brain layer has symbolic + probabilistic logic and a mechanism-free reflex system (Monitor delegated to `interface` mechanisms)
- Three backends fully API-aligned with zero-warning C code; v0.4.2 adds empty-input hardening, exhaustive malloc checks, and ARM/piwheels C99 compatibility

### Gaps
- Action and generation layers only have minimal primitives
- Extension layer is an empty skeleton
- No clear roadmap for layer-by-layer development

### Recommendations
1. **Prioritize memory layer** as the next milestone — it's the bridge from perception to cognition
2. **Flesh out brain layer** with more reasoning capabilities
3. **Define clear interfaces** between layers before extensive implementation
4. **Build vertical demos** showing multiple layers working together

---

**Next**: [Backend Management System](backend-system.md) — Multi-backend dynamic loading