# Modular Architecture (v0.4.3)

This section documents the **overall module architecture** of cos-comparison: module responsibility boundaries, dependency rules, and data-flow conventions.

Frozen as of **v0.4.1** (maintained through v0.4.3) — all internal algorithms and data formats are exposed by `core`; all interaction between the model and the outside world is abstracted by `interface`; every other module operates on top of these two foundations and manipulates data exclusively through the **attach-and-take** pattern, without depending on each other.

---

## 1. Core Principles (Overview)

> **`core` and `interface` are two absolutely independent foundations.**
>
> - `core`: the model's **internal** low-level algorithms and exposed data formats (algorithm family, three-backend acceleration, the `vector_map_as_tensor` data shape);
> - `interface`: the model's **external** interactions and interface abstraction (system commands/processes, concurrency/shared primitives, communication, dynamic-library/module invocation, context tools);
> - `data`: dedicated **data carrying (holding) and generic abstraction**;
> - All remaining modules (`sense_layer` / `memory_layer` / `brain_layer` / `action_layer` / `generate_layer` / ...) represent different capability layers and work on top of `core`, `interface`, and `data`.

---

## 2. Module Responsibility Matrix

| Module | Role | Core Responsibility | Absolutely Independent |
|--------|------|---------------------|:----------------------:|
| `core` | Algorithm core | Local similarity algorithm family (cos / mod / cosmod), data formats (`vector_map_as_tensor`, ...), three-backend (pydll / c / pure) auto-loading, stride-based indexing | Yes — stdlib only |
| `interface` | External abstraction | System commands & processes (`system_api`), thread/process locks, shared arrays and thread launching (`parallel_api`), blocking async event host with thread-safe event injection plus periodic task host `EventLoop` (`async_api`), IO/Socket/Pipe/File communication (`communicate_api`), dynamic library & module calls (`call_api`), database access (`database_api`), context tools (`context_tool`) | Yes — stdlib only |
| `data` | Data carrying | `DataWrap` generic container, `Tensor` / `SafeTensor` / `ParallelTensor` family, shape/strides/type normalization | — |
| `sense_layer` | Sensory layer | Receives external data and attaches it to core algorithms | — |
| `memory_layer` | Memory layer | Tiered memory carriers (Map / Table / Database), transaction & rollback support | — |
| `brain_layer` | Cognition layer | Symbolic logic, probability logic, reflex monitoring (Trigger / Monitor — pure logic, mechanisms delegated to `interface`) | — |
| `action_layer` | Action layer | Execution scheduling (`ExecuterDriver`) | — |
| `generate_layer` | Generation layer | Attaches an external callee to produce generated data | — |
| `test_tool` / `app` | Utilities / aggregation | Benchmarking (`Timer`); future business aggregation entry | — |

---

## 3. Dependency Rules (Red Lines)

```
┌───────────── core ─────────────┐      ┌────────── interface ──────────┐
│ Algorithms + data format        │      │ System / process / IO /       │
│ dependency: stdlib only         │      │ shared-memory / call abstr.   │
└───────────────┬─────────────────┘      │ dependency: stdlib only       │
                │   `data` ──depends on──▶ core (creates/converts data)  │
                │   `data` ──depends on──▶ interface (shared / locks)    │
                └────────────────────────┬───────────────────────────────┘
                                         │
       Every layer (sense / memory / brain / action / generate) ─
                      depends ONLY on core + interface + data
                                         │
                   Layers [must never] depend on one another
```

### Rules

1. `core` must not import any other package of this project (stdlib only);
2. `interface` must not import `core`, `data`, or any layer module (stdlib only);
3. `data` may depend on `core` (data format) and `interface` (parallel/shared primitives), but must not depend on any layer module;
4. Every functional layer may depend on `core`, `interface`, `data`; it **must not** depend on any other layer module;
5. `app` (aggregation layer) may depend on all layers (future business entry);
6. No module may redefine or re-implement data-format semantics already exposed by `core`.

---

## 4. Data Flow Convention: Attach-and-Take

Layered modules **do not own data and do not convert data formats**. Data is constructed by the caller using core-exposed formats (or `data` wrappers); the layer module "attaches and takes over" the data through its unified entry point:

| Pattern | Existing implementation | When the attachment happens |
|---------|------------------------|-----------------------------|
| Take over at construction | `Receptor(data)`, `Generator(data)`, `Memory(memory_body)` | Constructor arguments already carry the data |
| Inject at call time | `Generator.fix(call, ...)`, `Memory.process(caller, ...)`, `ExecuterDriver.call(index, args)` | The operator (caller) is passed in by the caller |
| Shared-slot transfer | `reflex.Trigger`'s `stack` + index references | Results/arguments exchanged through stack slots; layer never owns a data structure |

### Conventions

- A layer module **must not transform data on its own** (e.g. wrapping data with `memoryview()` internally) — shape/format normalization is the responsibility of `core` (data format) and `data` (carrying abstraction);
- Input data should be forwarded verbatim from the caller into `core` algorithms or `data` wrappers, without creating copies inside the layer;
- All model-to-world data crossings must go through `interface` (layers must not open sockets / processes / databases directly).

---

## 5. Architecture Conformance Checklist

| Check | Status (v0.4.3) |
|-------|:----------------:|
| `core` depends on stdlib only | Pass |
| `interface` depends on stdlib only | Pass |
| Layers do not depend on other layers | Pass |
| `data` dependency direction is correct (core / interface) | Pass |
| Layers never own non-core data formats | Partial — see legacy items R2/R3 |

---

## 6. Legacy Items and Evolution (flagged for v0.4.1)

Before this architecture was finalized (v0.4.1), code retained design remnants from unclear boundaries. The following legacy items are being cleaned up progressively:

| ID | Location | Legacy | Status |
|----|----------|--------|--------|
| R1 | `memory_layer/memory/database_memory.py` | Connected to external sqlite3 directly, bypassing `interface` | ✅ Resolved — `interface/api/database_api.py` now owns the driver; `memory_layer` only consumes `DatabaseToolWrap` / `DATABASE_DRIVER` |
| R2 | `sense_layer/receptor.py`, `generate_layer/generator.py` | Wrap data with `memoryview(data)` inside the layer (format conversion in the layer) | **Resolved (v0.4.3)** — neither layer wraps data in `memoryview`; receptor reads via `core.get_item`, generator writes via `core.set_item` / `core.load_data` |
| R3 | `cos_comparison/__init__.py` | `VERSION` read depended on the current working directory | ✅ Resolved — `VERSION.txt` is located via `__file__` with `import os.path as _osp`; the earlier `del os.path` variant (which removed the `path` attribute from the global `os` module) was replaced |
| R4 | `brain_layer/map`, `reflex/feedback`, `interface/tools/math_tool`, `extension_layer`, `app` | Empty placeholder directories | Keep as evolution placeholders or remove in later versions |
| R6 | `memory_layer/memory/basememory.py`, `interface/api/communicate_api.py` | `no_done` redefined in three places (core / basememory / communicate_api) | ✅ Resolved — single source in `interface/tools/func_tools.py`; layers import the same object (core keeps its own for dependency independence) |
| R8 | `data/tensor/parallel_tensor.py` | Dead `try/except ImportError` fallbacks assuming `interface` may be missing | ✅ Removed — direct imports from `interface.api` |
