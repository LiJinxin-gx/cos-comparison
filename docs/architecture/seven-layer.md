# Seven-Layer Cognitive Architecture

Cos-comparison adopts a **seven-layer brain-inspired cognitive architecture**, where each layer corresponds to different levels of biological cognitive processing. This layered design enables modular development and clear separation of concerns.

## Overview

| Layer | Directory | Brain Analogue | Maturity | Core Function |
|-------|-----------|----------------|----------|---------------|
| 1 | `core` | Brainstem / Cerebellum | ✅ Production | Low-level local comparison computation, stride-based tensor indexing, multi-backend acceleration |
| 2 | `sense_layer` | Sensory Cortex | 🟡 Early Development | Receive external stimuli (Receptor / TensorReceptor) |
| 3 | `memory_layer` | Hippocampus / Cortex | 🟡 Early Development | Short-term and long-term memory (map, table, database backends) |
| 4 | `brain_layer` | Prefrontal Cortex | 🟡 Early Development | Symbolic logic, probabilistic logic, reflex system |
| 5 | `action_layer` | Motor Cortex | 🟡 Early Development | Action output control (ExecuterDriver) |
| 6 | `generate_layer` | Broca's / Wernicke's Area | 🟡 Early Development | Generate data by matching data structures (Generator) |
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
Layer 6 (Generation)
    ↑ depends on
Layer 5 (Action)
    ↑ depends on
Layer 4 (Brain)
    ↑ depends on
Layer 3 (Memory)
    ↑ depends on
Layer 2 (Sense)
    ↑ depends on
Layer 1 (Core)
```

This ensures:
- Clean separation of concerns
- Each layer can be tested independently
- Lower layers can be optimized without affecting higher layers
- The system remains functional even if higher layers are incomplete

### 2. Neuroscience Homology

Each layer has a clear correspondence with biological brain structures:

- **Core (Layer 1)**: Analogous to brainstem and cerebellum — low-level, fast, automatic processing
- **Sense (Layer 2)**: Analogous to sensory cortex — initial feature extraction from raw input
- **Memory (Layer 3)**: Analogous to hippocampus and association cortex — storage and retrieval of information
- **Brain (Layer 4)**: Analogous to prefrontal cortex — reasoning, planning, decision making
- **Action (Layer 5)**: Analogous to motor cortex — execution of actions
- **Generate (Layer 6)**: Analogous to language areas — generation of complex outputs
- **Extension (Layer 7)**: Analogous to association areas — integration and special functions

### 3. Dual-Mode Consistency

The passive/active dual-mode philosophy extends vertically through all layers:

| Layer | Passive Mode | Active Mode |
|-------|-------------|-------------|
| Core | Self-similarity comparison | Template matching |
| Memory | Content-based retrieval | Pattern-based search |
| Brain | Bottom-up reasoning | Top-down planning |
| ... | ... | ... |

This consistent design philosophy creates architectural coherence.

## Layer Details

### Layer 1: Core

**Status**: ✅ Production (v0.3.10)

**Responsibilities**:
- Fundamental local comparison computation
- Passive mode: sliding window self-similarity
- Active mode: template matching
- Statistical operations: mean, variance
- Multiple backend implementations

**Key Features**:
- Dimension-agnostic (1D, 2D, 3D, 4D, ...)
- Stride-based `vector_map_as_tensor`: NumPy-like fancy indexing, zero-copy views, arbitrary slices and steps, negative indices, dimension collapse
- `__shape__` protocol for zero-overhead shape inference; `infer_shape` with PyBuffer > `__shape__` > recursive detection
- `load_as_default_data` for sub-region loading; `vector_chain_compute` for chained dot-product computation
- Multiple similarity algorithms (cos, mod, cosmod)
- Zero-dependency pure Python reference
- Three acceleration backends with automatic fallback: Python C extension, ctypes pure C, pure Python
- Full free-threaded Python (CPython 3.13+) support, runs without GIL for true parallelism
- Duck-typing support for NumPy/PyTorch arrays without dedicated backend
- Callback system for extensibility

### Layer 2: Sense Layer

**Status**: 🟡 Early Development

**Responsibilities**:
- Input data normalization and preprocessing
- Basic feature extraction pipelines
- Sensory adaptation mechanisms

**Current Implementation** (`sense_layer.receptor`):
- `Receptor`: wraps arbitrary data; `initialize(caller, *args, **kwargs)` runs a caller over the data; `point(index)` accesses a single element
- `TensorReceptor(Receptor)`: wraps data as a `memoryview`; provides `compaison_passive()` / `compaison_active()` shortcuts to the core modes
- All classes work on both pure Python and C extension backends

**Design Ideas**:
- Standardized input interfaces for different data types
- Automatic dimension detection and handling
- Sensory attention mechanisms (spatial/temporal)
- Adaptation to input statistics

### Layer 3: Memory Layer

**Status**: 🟡 Early Development (functional memory backends implemented)

**Responsibilities**:
- Short-term memory (working memory)
- Long-term memory storage
- Memory consolidation
- Pattern-based memory retrieval
- Dual-mode database (passive + active)

**Current Implementation** (`memory_layer.memory`):
- `BaseMemory` / `Memory`: lifecycle (initialize/save/commit/rollback/refer/close) with pluggable function rules
- `Status` flag enum: READ / WRITE / EXECUTE
- `MapMemory`: dict-backed memory with transaction cache, nested key support and atomic commit
- `TableMemory`: nested-map convenience subclass of MapMemory
- `DatabaseMemory`: SQLite-backed memory with execute/executemany, context manager support
- `MemoryWrap` / `MemoryWrapPool` / `MemoryWrapMap`: unified wrappers with hierarchical tagging (name, level, id) supporting short/long-term storage
- `Transaction`: atomic write record (key, value, nesting, create)

**Design Ideas**:
- Key-value store abstraction
- Vector similarity search
- Memory decay and reinforcement mechanisms
- Episodic vs semantic memory distinction
- Database backend pluggability

### Layer 4: Brain Layer

**Status**: 🟡 Partial (logic + reflex implemented)

**Responsibilities**:
- Symbolic logic and reasoning
- Decision making
- Goal management
- Planning and problem solving
- Metacognition

**Current Implementation**:

*Symbolic logic* (`brain_layer.logic.symbol_logic`):
- `Logic` flag enum: TRUE / SURE (three-valued style logic with confidence)
- `Variable`, `Atomic_proposition` (subject-verb-object structure with modifiers and limits)
- `Logic_bind`: implication/binding between propositions
- `Logic_context`: knowledge context management with add/pop operations
- `No_limit`: unlimited constraint container; `LogicError` exception

*Probabilistic logic* (`brain_layer.logic.probability_logic`):
- `UnionEvent`, `IntersectionEvent`, `GlobalEvent` (relative probability benchmark)
- `event_bind`: conditional probability storage `P(A|B)`
- `event_context`: Bayes-formula-based probability context

*Reflex* (`brain_layer.reflex.reflex`):
- `Trigger`: bind a trigger function to a callback with a shared result stack
- `Monitor`: asyncio-based background monitoring of object attributes with callbacks

**Design Ideas**:
- Three-valued logic system (addressing hallucination problem)
- Rule-based reasoning engine
- Goal hierarchy management
- Planning with subgoals
- Confidence estimation

### Layer 5: Action Layer

**Status**: 🟡 Early Development

**Responsibilities**:
- Action selection and execution
- Motor control
- Environment interaction
- Action outcome evaluation
- Reflex arcs

**Current Implementation** (`action_layer.executer`):
- `ExecuterDriver`: ordered callable list with indexed invocation

**Design Ideas**:
- Action primitive library
- Action sequencing
- Feedback-based action adjustment
- Exploration vs exploitation balance
- Innate vs learned reflexes

### Layer 6: Generate Layer

**Status**: 🟡 Early Development

**Responsibilities**:
- Language generation
- Image/signal generation
- Output formatting
- Creative generation
- Expression of internal states

**Current Implementation** (`generate_layer.generator`):
- `Generator`: wraps data; `fix(call, ...)` applies a callable to the data
- `TensorGenerator(Generator)`: memoryview-based generator with `set_point(index, data)` point writing

> **Note**: The entire `cos_comparison.generate_layer` package is currently unimportable — its `__init__.py` references a missing `basedata` module, and importing any submodule triggers the parent package `__init__` first. Expected to be fixed in a future release.

**Design Ideas**:
- Template-based generation
- Compositional generation
- Style control
- Multi-modal output generation

### Layer 7: Extension Layer

**Status**: 🔴 Skeleton (empty package)

**Responsibilities**:
- Specialized capabilities
- Cross-modal integration
- Advanced cognitive functions
- Plugin system for extensions

**Design Ideas**:
- Plugin architecture
- Cross-modal association
- Meta-learning capabilities
- Social cognition modules

## Cross-Cutting Modules

### interface

**Status**: 🟡 Early Development

- `interface.api.system_api`: `command()` (subprocess capture), `getpid`, `getppid`, `kill`, `home_executable`, `BaseProcess`, and `Process` — a subprocess wrapper with buffered stdin/stdout/stderr, background reader threads, `execute()`, `get_stdout()`, `get_stderr()`, `get_stdin()`, `stop()`
- `interface.api.call_api`: `BaseCallContainer`, `Module_CallContain`, `C_CallContainer`, `CDLL_CallContainer`, `WinDLL_CallContainer`, `CallDict` — unified call abstraction over Python modules and native libraries
- `interface.api.communicate_api`: `Communicate` chain — `IOCommunicate`, `FdCommunicate`, `PIPECommunicate`, `SocketCommunicate`, `FileCommunicate`
- `interface.api.parallel_api`: thread/process locks and RLocks, `IntegrateContext` composition, `Thread` / `Process` aliases
- `interface.tools.context_tool`: `VoidContext`, `IntegrateContext`, `AsyncIntegrateContext`

> **Note**: The entire `cos_comparison.interface` package (including `interface.api`, `interface.tools` and all their submodules) is currently unimportable in a fresh interpreter because `parallel_api` uses a broken absolute import (`from context_tool import *`). Individual modules can no longer be reached since parent package `__init__` files always run first. Expected to be fixed in a future release.

### data

**Status**: 🟡 Early Development

- `data.DataWrap`: unified data carrier with `process` / `call` / `getattr` / `setattr` delegation
- `data.tensor`: `BaseTensor`, `Tensor` (inherits `core.vector_map_as_tensor`, loads via `load_as_default_data`), `SafeTensor` (lock-protected writes), `ParallelTensor` (shared memory), `Task`

> **Note**: `cos_comparison.data` currently fails to import because `DataWrap` references an undefined `BaseData`; this is expected to be fixed in the next version.

### test_tool

**Status**: ✅ Stable

- `Timer`: high-resolution performance benchmarking (`mark`, `get_time`, `reset`), `perf_count = time.perf_counter`

### app

**Status**: 🔴 Skeleton — aggregation of cross-layer functionality (planned).

## Current State Assessment (v0.3.10)

### Strengths
- Clear architectural vision with strong neuroscience inspiration
- Production-ready core layer with solid, well-tested implementation
- Clean separation of concerns and dependency direction
- Dual-mode philosophy provides conceptual coherence
- Memory layer has functional backends (map/table/database) with transaction support
- Brain layer has symbolic + probabilistic logic and an asyncio reflex system
- Three backends fully API-aligned with zero-warning C code

### Gaps
- `generate_layer`, `interface`, and `data` packages currently have import errors pending fixes
- Action and generation layers only have minimal primitives
- Extension layer is an empty skeleton
- No clear roadmap for layer-by-layer development

### Recommendations
1. **Fix the pending import errors** in `generate_layer` / `interface` / `data` first — they block the "all layers importable" guarantee
2. **Prioritize memory layer** as the next milestone — it's the bridge from perception to cognition
3. **Flesh out brain layer** with more reasoning capabilities
4. **Define clear interfaces** between layers before extensive implementation
5. **Build vertical demos** showing multiple layers working together

---

**Next**: [Backend Management System](backend-system.md) — Multi-backend dynamic loading