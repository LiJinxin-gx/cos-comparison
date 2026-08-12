# Cognitive Layer APIs

This document covers the API surface of the non-core layers: `sense_layer`, `memory_layer`, `brain_layer`, `action_layer`, `generate_layer`, plus the cross-cutting modules `interface`, `data`, and `test_tool`.

> **Compatibility note**: These layers are under active development. All packages (`interface`, `data`, `generate_layer` included) import cleanly in a fresh interpreter; the `interface` surface is exercised by dedicated tests. The core module remains fully production-ready.

## Sense Layer

Receives external stimuli and exposes them to the core computation. The package imports cleanly.

### Receptor

Wraps arbitrary data and provides access to it.

```python
from cos_comparison.sense_layer import Receptor

r = Receptor(data)
r.initialize(caller, *args, **kwargs)  # runs caller(data, *args, **kwargs)
```

- `initialize(caller, *args, **kwargs)`: applies `caller` to the wrapped data
- `point(index)`: returns `data[*index]` — implemented as `self.data.__getitem__(*index)`, so it works with containers whose `__getitem__` accepts unpacked multi-element indices (e.g. tensors, nested lists with a single-element tuple); on plain `list` containers, multi-element indices raise `TypeError`

### TensorReceptor

Wraps data as a `memoryview` when possible (falls back to the raw object on failure) and provides `comparison_passive(output=None, **kwargs)` / `comparison_active(output=None, **kwargs)` shortcuts to the core modes, returning the core result (e.g. the output tensor).

## Memory Layer

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
m.commit()                                      # apply all cached transactions, then clear the cache (repeated commits are no-ops)
m.rollback()                                    # discard only the pending (uncommitted) transactions; the cache stays a valid list
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

## Brain Layer

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
- `event_context(name, binds)`: probability context based on the Bayes form `P(AB|C) = P(A|B)·P(B|C)`; `add_bind(binds)`, `bind_probability(A, B=global_event)` — returns the stored conditional `P(A|B)`; on a cache miss it delegates to `probability_func` (default: returns `0`), which can be supplied at construction (`event_context(probability_func=...)`) for custom inference

### Reflex (`brain_layer.reflex.reflex`)

- `Trigger(trigger, callback, stack=None, a_res_index=DEVNULL, b_res_index=DEVNULL, args_index=(), kwargs_index=())`: binds a trigger function to a callback through a shared result stack — `trigger(*args, kwargs)` result is stored at `stack[a_res_index]`, the callback result at `stack[b_res_index]`; returns `0` on success
- `Monitor(maintainer=None, poller=None, runner=None, scheduler=None, stopper=None, observer=None)`: **pure monitoring logic** with six delegation slots — every mechanism lives in `interface`, and `monitor.py` imports no `asyncio` / `threading` / `time`. By default the poller is an `EventLoop` (periodic host), the maintainer is `run_in_thread` (thread carrying), and bookkeeping (hits / errors) is guarded by the ready-made `parallel_lock`
  - `add_event(trigger, callback=None, *, interval=None, times=-1)`: probe until truthy → fire callback on every hit; auto-removed after `times` hits (`-1` = forever, `0` = never registered); returns the schedule handle
  - `remove(handle)`, `run(*a, **k)` (blocking drive), `maintrain(*a, **k)` (background thread, `run_in_thread`), `stop`, `wait(timeout=None)`, `is_running()`, `hits(handle=None)`, `errors(handle=None)`

```python
from cos_comparison.brain_layer.reflex import Monitor

m = Monitor()
m.add_event(lambda: obj.value > 100, on_high, interval=0.02, times=3)
m.maintrain()  # block-free carrying through interface.run_in_thread
```

The same delegation slots are public (`monitor.Monitor(...)`), so a layer can swap in its own scheduler/stopper without touching the mechanisms.

## Action Layer (`cos_comparison.action_layer`)

- `ExecuterDriver(caller_list=())`: an ordered callable list with indexed invocation — `call(index, args=(), kwargs=None)` invokes `call_list[index](*args, **kwargs)`

## Generate Layer (`cos_comparison.generate_layer`)

Intended to provide generation tools:

- `Generator(data)`: `fix(call, args=(), kwargs=None)` applies a callable to the wrapped data
- `TensorGenerator(data)`: memoryview-based generator; `set_point(index, data)` writes a point

> **Note**: `cos_comparison.generate_layer` imports cleanly in a fresh interpreter. It is still the most stub-like layer (generation primitives only).

## Interface (`cos_comparison.interface`)

Abstraction of the external interface. The following modules exist:

- **system_api**: `command(commands)` (returns `(out, err, returncode)`), `getpid`, `getppid`, `home_executable`, `kill`, `BaseProcess` (subprocess.Popen with piped stdio), `Process(executable, arg_list, **kwarg)` — subprocess wrapper with byte-buffered stdin/stdout/stderr, background reader threads, `execute(command)`, `get_stdout(clear=False)`, `get_stderr(clear=False)`, `get_stdin(clear=False)`, `stop(timeout=0.5)`
- **call_api**: `BaseCallContainer(obj)` with `call(name, args, kwargs, init_func)`, `get_call`, `get`, `set`; `Module_CallContain(module_name, *args, package=None)`; `C_CallContainer(library_path, loader=ctypes.CDLL)` — `get_call(name, argstypes=(), restype=None)` sets ctypes `argstypes`/`restype` on the wrapped function; `CDLL_CallContainer` / `WinDLL_CallContainer`; `CallDict(init_dict)` with `add(tag, func)` / `call(tag, args)`
- **communicate_api**: class chain `BaseCommunicate → Communicate → IOCommunicate`, with concrete `FdCommunicate`, `PIPECommunicate(auto=False)`, `SocketCommunicate`, `FileCommunicate(file_path, mode="rb", buffering=-1)` — send/recv abstraction over OS fds, sockets and files
- **parallel_api**: `thread_lock`/`process_lock`, `thread_rlock`/`process_rlock`, composed ready-to-use locks `parallel_lock`/`parallel_rlock` (via `IntegrateContext`), `Thread`, `Process`, shared arrays (`share_array` / `load_array`), and `run_in_thread(target, args=(), kwargs=None, is_join=False, **kws)` — thread launching & joining, owned by interface
- **async_api**: `AsyncRunner()` — blocking async event host with thread-safe event injection (`add_event`/`remove`/`result`/`done`/`active`, timeout exit, exception isolation, single-life); `EventLoop(interval=0.01)` — periodic task host on top of `AsyncRunner` (`schedule(task, interval=None)` → handle, `unschedule`, `run`, `stop`, `wait`, `active`)

- **tools/context_tool**: `VoidContext()` (no-op context manager), `IntegrateContext(*contexts)` (composes context managers, exits in reverse order), `AsyncIntegrateContext(*a_context)` (async variant)

> **Note**: `cos_comparison.interface` (including `interface.api` and `interface.tools`) imports cleanly in a fresh interpreter — `EventLoop`, `run_in_thread` and the integrated locks are exercised by `tests/test_async_api.py` / `tests/test_parallel_api.py`.

## Data (`cos_comparison.data`)

Unified data carrier interfaces.

- `DataWrap(data_body=None)`: intended to be a unified data carrier with `process(caller, ...)`, `call(name, ...)`, `getattr`/`setattr`, plus `__getitem__`/`__setitem__` and the `__get_item__`/`__set_item__` protocol

> **Note**: `cos_comparison.data` imports cleanly in a fresh interpreter — the `DataWrap` carrier and the tensor containers are exercised by `tests/test_tensor_comprehensive.py`.

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