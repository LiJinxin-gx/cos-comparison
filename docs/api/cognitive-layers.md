# Cognitive Layer APIs

This document covers the API surface of the non-core layers: `sense_layer`, `memory_layer`, `brain_layer`, `action_layer`, `generate_layer`, plus the cross-cutting modules `interface`, `data`, and `test_tool`.

> **Compatibility note**: These layers are under active development. The packages `cos_comparison.interface`, `cos_comparison.data`, and `cos_comparison.generate_layer` **currently fail to import in a fresh interpreter** (see the notes below). The core module is unaffected and fully production-ready.

## Sense Layer (`cos_comparison.sense_layer`)

Receives external stimuli and exposes them to the core computation. The package imports cleanly.

### Receptor

Wraps arbitrary data and provides access to it.

```python
from cos_comparison.sense_layer import Receptor

r = Receptor(data)
r.initialize(caller, *args, **kwargs)  # runs caller(data, *args, **kwargs)
```

- `initialize(caller, *args, **kwargs)`: applies `caller` to the wrapped data
- `point(index)`: intended to return `data[*index]`, but **currently broken** — `Receptor` has no `__getitem__`, so `point()` raises `TypeError`

### TensorReceptor

Intended to wrap data as a `memoryview` and provide `compaision_passive()` / `compaision_active()` shortcuts to the core modes, but **currently broken** — its `__init__` references an undefined local `data`, so construction always raises `NameError`.

## Memory Layer (`cos_comparison.memory_layer`)

Provides memory backends (map / table / database) and unified wrappers with hierarchical tagging. The package imports cleanly and the memory backends work.

### Status

A `Flag` enum for access permissions: `Status.READ`, `Status.WRITE`, `Status.EXECUTE` (aliases `ST_READ`, `ST_WRITE`, `ST_EXECUTE`).

### Memory (basememory)

Lifecycle-based memory with pluggable function rules:

```python
from cos_comparison.memory_layer.memory import Memory

m = Memory(memory_obj,
           init_func=None, save_func=None, commit_func=None,
           rollback_func=None, refer_func=None, close_func=None)
m.initialize(*args, **kwargs)
m.save(*args, **kwargs)
m.commit(*args, **kwargs)
m.rollback(*arg, **kwarg)
m.refer(*args, **kwargs)
m.close(*args, **kwargs)
m.process(caller, *args, **kwargs)   # apply caller to the memory object
m.call(name, *args, **kwargs)        # call a method of the memory object
```

### MapMemory

Dict-backed memory with transaction cache and atomic commit:

```python
from cos_comparison.memory_layer.memory import MapMemory

m = MapMemory(map_obj, close_commit=False, closer=None)
m.save(key, value, nesting=False, create=True)  # record a transaction
m.commit()                                      # apply all cached transactions atomically
m.rollback()                                    # discard cached transactions
m.refer(key, nesting=False)                     # read value (nested key traversal optional)
m.close()                                       # commit if close_commit, release memory
```

- `Transaction(key, value, nesting=True, create=True)`: the atomic write record; hashable, iterable over its fields

### TableMemory

`MapMemory` subclass with nested-key convenience: `save(keys, value)` and `refer(keys)` always use nested traversal with auto-creation.

### DatabaseMemory

SQLite-backed memory with context manager support:

```python
import sqlite3
from cos_comparison.memory_layer.memory import DatabaseMemory

with DatabaseMemory(database_tool=sqlite3, database=":memory:") as db:
    db.execute("CREATE TABLE t (k TEXT, v REAL)")
    db.executemany("INSERT INTO t VALUES (?,?)", [("a", 1.0), ("b", 2.0)])
    db.commit()
    db.execute("SELECT v FROM t ORDER BY v")
    print(db.cursor.fetchall())   # [(1.0,), (2.0,)]
```

- Provides `execute(command, arg=())`, `executemany(command, args=())`, `commit()`, `rollback()`, `close()`
- **Note**: Pass the database tool explicitly (`database_tool=sqlite3`). The default-tool fallback path (omitting `database_tool`) currently fails — a known limitation to be fixed in a future release

### Wrappers (`memory_layer.memory`)

- `MemoryWrap(memory_body=None, memory_type=None, args=(), kwargs=None, name="", level=0)`: unified interface over any memory; `process(caller, ...)`, `call(name, ...)`, `get(name)`, `set(name, value)`
- `MemoryWrapPool(pool=None, name="", level=0)`: list-like pool of `MemoryWrap`; `add()`, `set()`, `operate()`, `get_memory_attr()`
- `MemoryWrapMap(map_pool=None, name="", level=0)`: dict-keyed pool; `add(index, wrap)`, `get_by_name(name)`, membership via value equality

```python
from cos_comparison.memory_layer.memory import (
    MapMemory, MemoryWrap, MemoryWrapPool, MemoryWrapMap,
)

m = MapMemory({}); m.save("k", 42); m.commit()
wrap = MemoryWrap(memory_body=m, name="test", level=1)
wrap.process(lambda body, *a, **k: body.refer("k"))   # 42
wrap.call("refer", ("k",))                            # 42

pool = MemoryWrapPool(); pool.add(wrap)               # len(pool) == 1
mp = MemoryWrapMap(); mp.add("w1", wrap)
mp.get_by_name("w1").name                             # 'test'
```

## Brain Layer (`cos_comparison.brain_layer`)

### Symbolic Logic (`brain_layer.logic.symbol_logic`)

Three-valued style logic system for cognitive reasoning. All classes below work.

- `Logic(Flag)`: `TRUE`, `SURE`; constants `Logic_true`, `Logic_sure`, `sure_true = TRUE | SURE`
- `Variable(name, value=None)`: named variable
- `No_limit()`: constraint container that contains everything
- `LogicError(Exception)`: logical error
- `Atomic_proposition(subject, verb, objects, adv, limit, status)`: structured proposition; supports `__bool__` (based on `status & TRUE`), dict-style item access
- `Logic_bind(reason, result, limit, status)`: implication between propositions
- `Logic_context(name, binds)`: knowledge context; `add(logic_bind)` raises `ValueError` on duplicate reason/result pairs; `pop(logic_bind)` removes by reason/result pair

```python
from cos_comparison.brain_layer import logic

ctx = logic.Logic_context(name="world")
p1 = logic.Atomic_proposition(subject="sky", verb="is", objects="blue")
p2 = logic.Atomic_proposition(subject="sky", verb="is", objects="red")
ctx.add(logic.Logic_bind(p1, p2))
```

### Probabilistic Logic (`brain_layer.logic.probability_logic`)

Uncertain reasoning with conditional probabilities.

- `UnionEvent(*event)`, `IntersectionEvent(*event)`: frozenset-based event classes
- `GlobalEvent()` / `global_event`: relative probability benchmark (all probabilities are relative)
- `event_bind(name, event)`: conditional probability storage; `bind(event, p)` (validates `0 ≤ p ≤ 1`), `unbind(event)`, `get_bind(event)`, `bind_exist(event)`
- `event_context(name, binds)`: probability context based on the Bayes form `P(AB|C) = P(A|B)·P(B|C)`; `add_bind(binds)`, `bind_probability(A, B=global_event)`

### Reflex (`brain_layer.reflex.reflex`)

- `Trigger(trigger, callback, stack=None, a_res_index=DEVNULL, b_res_index=DEVNULL, args_index=(), kwargs_index=())`: intended to bind a trigger function to a callback through a shared result stack; **currently broken** — `kwargs_index` is missing from its `__slots__`, so construction always raises `AttributeError`
- `Monitor(callbacks=None, target=None)`: asyncio-based background monitor; `add_target(target, trigger, callback, times=-1)`; `run()` starts a daemon thread — **works**

```python
from cos_comparison.brain_layer import reflex

m = reflex.Monitor()
m.add_target(target_obj, lambda v: v > 100, on_high)
m.run()  # background monitoring in a daemon thread
```

## Action Layer (`cos_comparison.action_layer`)

- `ExecuterDriver(caller_list=())`: intended to be an ordered callable list with indexed invocation, but **currently broken** — its `__init__` references an undefined `call_list` (parameter is `caller_list`), so construction always raises `NameError`

## Generate Layer (`cos_comparison.generate_layer`)

Intended to provide generation tools:

- `Generator(data)`: `fix(call, args=(), kwargs=None)` applies a callable to the wrapped data
- `TensorGenerator(data)`: memoryview-based generator; `set_point(index, data)` writes a point

> **Note**: The **entire `generate_layer` package is currently unimportable** — its `__init__.py` runs `from .basedata import *`, and the `basedata` module does not exist. Even `from cos_comparison.generate_layer.generator import Generator` fails in a fresh interpreter because the parent package `__init__` runs first. Expected to be fixed in a future release.

## Interface (`cos_comparison.interface`)

Abstraction of the external interface. The following modules exist:

- **system_api**: `command(commands)` (returns `(out, err, returncode)`), `getpid`, `getppid`, `home_executable`, `kill`, `BaseProcess` (subprocess.Popen with piped stdio), `Process(executable, arg_list, **kwarg)` — subprocess wrapper with byte-buffered stdin/stdout/stderr, background reader threads, `execute(command)`, `get_stdout(clear=False)`, `get_stderr(clear=False)`, `get_stdin(clear=False)`, `stop(timeout=0.5)`
- **call_api**: `BaseCallContainer(obj)` with `call(name, args, kwargs, init_func)`, `get_call`, `get`, `set`; `Module_CallContain(module_name, *args, package=None)`; `C_CallContainer(library_path, loader=ctypes.CDLL)`; `CDLL_CallContainer` / `WinDLL_CallContainer`; `CallDict(init_dict)` with `add(tag, func)` / `call(tag, args)`
- **communicate_api**: class chain `BaseCommunicate → Communicate → IOCommunicate`, with concrete `FdCommunicate`, `PIPECommunicate(auto=False)`, `SocketCommunicate`, `FileCommunicate(file_path, mode="rb", buffering=-1)` — send/recv abstraction over OS fds, sockets and files
- **parallel_api**: `thread_lock`/`process_lock`, `thread_rlock`/`process_rlock`, composed `parallel_lock`/`parallel_rlock` (via `IntegrateContext`), `Thread`, `Process`

- **tools/context_tool**: `VoidContext()` (no-op context manager), `IntegrateContext(*contexts)` (composes context managers, exits in reverse order), `AsyncIntegrateContext(*a_context)` (async variant)

> **Note**: The **entire `cos_comparison.interface` package is currently unimportable in a fresh interpreter** — `interface/api/__init__.py` imports `parallel_api`, which uses a broken absolute import (`from context_tool import *`). Importing any submodule (e.g. `cos_comparison.interface.api.system_api`) triggers the parent package `__init__` first and fails with `ModuleNotFoundError`. `PIPECommunicate(auto=True)` additionally raises `TypeError` due to a signature mismatch. Expected to be fixed in a future release.

## Data (`cos_comparison.data`)

Unified data carrier interfaces.

- `DataWrap(data_body=None)`: intended to be a unified data carrier with `process(caller, ...)`, `call(name, ...)`, `getattr`/`setattr`, plus `__getitem__`/`__setitem__` and the `__get_item__`/`__set_item__` protocol

> **Note**: The **entire `cos_comparison.data` package is currently unimportable** — `DataWrap` inherits from an undefined `BaseData`, raising `NameError` at class creation. This also blocks `cos_comparison.data.tensor`. Expected to be fixed in a future release.

### Tensor Containers (`data.tensor`)

- `BaseTensor(ABC)`: abstract `__getitem__`/`__setitem__`
- `Tensor(BaseTensor, core.vector_map_as_tensor)`: tensor container inheriting the core stride-based tensor; loads data via `load_as_default_data`
- `SafeTensor(Tensor)`: adds a lock around `__setitem__`/`__set_item__` for parallel safety
- `ParallelTensor(Tensor)`: backed by a shared array for cross-process use
- `Task(caller, args=(), kwargs=None)`: callable task wrapper

## Test Tools (`cos_comparison.test_tool`)

Stable and fully working.

- `perf_count = time.perf_counter`
- `Timer(start=0.0, timer=perf_count)`: `mark()`, `get_time()`, `reset()` — high-resolution performance measurement

```python
from cos_comparison.test_tool import Timer

t = Timer()
...  # work
t.mark()
print(t.get_time())
```

---

**See also**:
- [Seven-Layer Architecture](../architecture/seven-layer.md) — Layer overview and maturity levels
- [Core Module](core.md) — The production-ready core API