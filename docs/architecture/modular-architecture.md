# Modular Architecture

Module responsibility boundaries, dependency rules, and data-flow conventions. Frozen as of v0.4.1 (maintained through v0.4.3).

## Contents

- [Core Principles](#core-principles)
- [Responsibility Matrix](#responsibility-matrix)
- [Dependency Rules](#dependency-rules)
- [Attach-and-Take Data Flow](#attach-and-take-data-flow)
- [Conformance Checklist](#conformance-checklist)
- [Legacy Items](#legacy-items)

---

## Core Principles

> **`core` and `interface` are two absolutely independent foundations.**

| Foundation | Role | Independence |
|------------|------|:------------:|
| **`core`** | Internal low-level algorithms and data formats (algorithm family, three-backend acceleration, `vector_map_as_tensor`) | ✅ stdlib only |
| **`interface`** | External interactions (system/processes, concurrency, communication, dynamic-library/module calls, context tools) | ✅ stdlib only |
| **`data`** | Data carrying and generic abstraction (`DataWrap`, tensor family) | — |
| **Functional layers** | Operate on top of `core`, `interface`, `data` via attach-and-take | — |

All internal algorithms and data formats are exposed by `core`; all model-to-world interaction is abstracted by `interface`; every other module manipulates data exclusively through attach-and-take, without depending on each other.

---

## Responsibility Matrix

| Module | Role | Core Responsibility |
|--------|------|---------------------|
| `core` | Algorithm core | Local similarity (cos/mod/cosmod), data formats, three-backend auto-loading, stride indexing |
| `interface` | External abstraction | System/process (`system_api`), locks/shared arrays/threads (`parallel_api`), async hosts (`async_api`), IO/socket/pipe/file (`communicate_api`), dynamic calls (`call_api`), database (`database_api`), context tools |
| `data` | Data carrying | `DataWrap`, `Tensor`/`SafeTensor`/`ParallelTensor`, shape/stride/type normalization |
| `sense_layer` | Sensory | Receives external data, attaches to core algorithms |
| `memory_layer` | Memory | Map/Table/Database carriers, transaction & rollback |
| `brain_layer` | Cognition | Symbolic/probabilistic logic, reflex monitoring (pure logic, mechanisms in `interface`) |
| `action_layer` | Action | Execution scheduling (`ExecuterDriver`, delegated async execution) |
| `generate_layer` | Generation | Attaches external callees to produce data |
| `test_tool` / `app` | Utilities / aggregation | Debug probes & benchmarking; future business aggregation |

---

## Dependency Rules

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

**Rules:**

1. `core` imports no other project package (stdlib only)
2. `interface` imports no `core`/`data`/layer module (stdlib only)
3. `data` may depend on `core` and `interface`, but no layer module
4. Each functional layer may depend on `core`, `interface`, `data` — **must not** depend on other layers
5. `app` (aggregation) may depend on all layers
6. No module may redefine data-format semantics already exposed by `core`

---

## Attach-and-Take Data Flow

Layers **do not own or transform data**. Data is constructed by the caller; layers take over at entry points:

| Pattern | Examples | When |
|---------|----------|------|
| At construction | `Receptor(data)`, `Generator(data)`, `Memory(memory_body)` | Constructor carries the data |
| At call time | `Generator.fix(call)`, `Memory.process(caller)`, `ExecuterDriver.call(index)` | Operator passed by caller |
| Shared-slot transfer | `reflex.Trigger` stack + index references | Results exchanged through stack slots |

**Conventions:**

- No internal `memoryview()` wrapping or format conversion (that is `core`/`data`'s job)
- No copies inside layers; input forwarded verbatim into `core` algorithms or `data` wrappers
- All model-to-world crossings go through `interface` (layers must not open sockets/processes/databases directly)

---

## Conformance Checklist

| Check | Status (v0.4.3) |
|-------|:----------------:|
| `core` depends on stdlib only | ✅ Pass |
| `interface` depends on stdlib only | ✅ Pass |
| Layers do not depend on other layers | ✅ Pass |
| `data` dependency direction correct (core / interface) | ✅ Pass |
| Layers never own non-core data formats | ✅ Pass |

---

## Legacy Items

| ID | Item | Status |
|----|------|--------|
| R1 | `database_memory` connected sqlite3 directly | ✅ Resolved — `interface.database_api` owns driver |
| R2 | Receptor/generator wrapped data in `memoryview` | ✅ Resolved (v0.4.3) — uses `core.get_item`/`set_item`/`load_data` |
| R3 | `VERSION` read depended on CWD | ✅ Resolved — located via `__file__` |
| R4 | Empty placeholder dirs (`map`, `feedback`, `extension_layer`, `app`) | Kept as evolution placeholders |
| R6 | `no_done` redefined in three places | ✅ Resolved — single source in the `core` module |
| R8 | `parallel_tensor` dead `try/except ImportError` | ✅ Removed — direct imports |

---

**Related:** [Seven-Layer Architecture](seven-layer.md) · [Backend Management System](backend-system.md)
