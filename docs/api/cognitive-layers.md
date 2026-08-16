# Cognitive Layer APIs

This document covers the API surface of the non-core layers: `sense_layer`, `memory_layer`, `brain_layer`, `action_layer`, `generate_layer`, plus the cross-cutting modules `interface`, `data`, and `test_tool`.

> **Compatibility note**: These layers are under active development. All packages (`interface`, `data`, `generate_layer` included) import cleanly in a fresh interpreter; the `interface` surface is exercised by dedicated tests. The core module remains fully production-ready.

## Sense Layer

Receives external stimuli and exposes them to the core computation.

### Receptor

Wraps arbitrary data and provides access to it.

```python
from cos_comparison.sense_layer import Receptor

r = Receptor(data)
r.initialize(caller, *args, **kwargs)  # runs caller(data, *args, **kwargs)
r.receptor(caller, args=(), kwargs=None)  # runs caller(data, *args, **kwargs)
```

- `initialize(caller, *args, **kwargs)`: applies `caller` to the wrapped data
- `receptor(caller, args=(), kwargs=None)`: delegated application of `caller` to the wrapped data

### TensorReceptor

Stores the raw data object and provides `point(index)` (via the core `get_item` protocol — `__get_item__` authoritative, plain indexing fallback) plus `comparison_passive(output=None, **kwargs)` / `comparison_active(output=None, **kwargs)` shortcuts to the core modes, returning the core result (e.g. the output tensor).

## Memory Layer

Provides memory backends (map / table / database) and unified wrappers with hierarchical tagging. The memory backends work.

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
m.commit()                                      # apply all cached transactions; on failure the applied entries are dropped and the remaining ones stay queued (a retry never re-applies)
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
- **Note**: `database_tool` is optional — when omitted, the interface-owned default driver (`interface.api.DATABASE_DRIVER`, sqlite3) is used; a `RuntimeError` is raised only if no driver is available. `executemany` re-raises real execution errors (no silent fallback replay); when the cursor is unavailable the connection path is used directly

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
- `Logic_context(name="", binds=None, init_func=None, add_func=None, pop_func=None, judge_func=None)`: knowledge context with delegated slots; `add(logic_bind)` / `pop(logic_bind)` are delegated (default `no_done` — inject `add_func` / `pop_func` for storage or de-duplication); `logic_judge(a, b, **kwargs)` is delegated to `judge_func`, whose default `default_judge_func` answers by graph reachability over the binds (arc `reason → result`, shortest path via `interface.tools.math_tool.topology.shortest_path_between`; `return_path=True` returns the rule list, `[]` when `a == b`, `None` when unreachable); `extension` slot for callbacks

```python
from cos_comparison.brain_layer import logic

ctx = logic.Logic_context(name="world",
                          add_func=lambda binds, lb, **kw: binds.append(lb))
p1 = logic.Atomic_proposition(subject="sky", verb="is", objects="blue")
p2 = logic.Atomic_proposition(subject="sky", verb="is", objects="red")
ctx.add(logic.Logic_bind(p1, p2))
print(ctx.logic_judge(p1, p2))   # True (default graph-reachability judge)
```

### Probabilistic Logic (`brain_layer.logic.probability_logic`)

Uncertain reasoning with conditional probabilities.

- `UnionEvent(*event)`, `IntersectionEvent(*event)`: frozenset-based event classes
- `GlobalEvent()` / `global_event`: relative probability benchmark (all probabilities are relative)
- `event_bind(name, event)`: conditional probability storage; `bind(event, p)` (validates `0 ≤ p ≤ 1`), `unbind(event)`, `get_bind(event)`, `bind_exist(event)`
- `event_context(name="", binds=None, init_func=None, add_func=None, probability_func=None)`: protocol-style context — every operation is a delegated slot (`init_func` / `add_func` / `probability_func`); `binds=None` constructs an `EventBinds()` protocol container by default (a `dict` subclass carrying a cached dependency graph, resolution statistics and a `strict` switch; its class methods `add_bind` / `resolve` are injected into the slots), `extension` holds callbacks, and `initialize()` warms the graph cache (default `default_context_init`)
  - `add_bind(triples)` stores `(outcome, condition, p)` as `binds[(outcome, condition)] = p`; `bind_probability(A, B=global_event)` delegates **entirely** to the probability slot
  - All probabilities are **relative** — `global_event` is the benchmark (axioms A1–A5: relativism, reflexivity `P(X|X)=1`, chain rule `P(A|C)=∏P(vᵢ|vᵢ₋₁)`, Bayes duality `P(A|B)=P(B|A)·P(A|C)/P(B|C)`, union). Absolute forms degenerate from `C = global_event`
  - Default resolution (two-stage): exact hit → relative Bayes over direct references only (`strict` mode, division guarded) → shortest-path chain fallback; `0.0` only when unreachable
  - Related functions: `chain_probability`, `default_probability_func`, `strict_probability_func`, `chain_intersection`, `union_probability`, `consistency_diagnostic`, `EventBinds`, `EventContextProtocol` (deploy form)

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

## Mapper (`cos_comparison.brain_layer.mapper`)

- `BaseMap(ABC)`: abstract `__getitem__` / `__setitem__` / `__contains__` mapping operations
- `Map(map_obj=None, map_func=None, set_func=None, contain_judge_func=None)`: three-slot protocol container with default **dict-protocol** implementations (`default_map_func` = `map_obj[x]`, `default_set_func` = `map_obj[x]=v`, `default_contain_judge_func` = `x in map_obj`); injecting a function replaces the slot entirely; an unconfigured `map_obj` raises `TypeError` on use instead of silently returning `None`

```python
from cos_comparison.brain_layer.mapper import Map

m = Map(map_obj={})
m["a"] = 1            # default set_func
m["a"]                # 1 (default map_func)
"a" in m              # True (default contain_judge_func)
```

## Action Layer (`cos_comparison.action_layer`)

- `ExecuterDriver(caller_list=())`: an ordered callable list with indexed invocation — `call(index, args=(), kwargs=None)` invokes `call_list[index](*args, **kwargs)`

## Generate Layer (`cos_comparison.generate_layer`)

Provides generation and modification tools over wrapped data.

- `Generator(data)`: `fix(call, args=(), kwargs=None)` applies a callable to the wrapped data
- `TensorGenerator(data)`: `generate(func, args=(), kwargs=None)` is the **unified delegation entry** — it runs `func(self.data, *args, **kwargs)` (generation/modification logic lives in external functions); `set_point(index, value)` writes one element via the core `set_item` protocol (`__set_item__` authoritative, plain nested assignment fallback)
- Module-level `copy_region(target, source, *, shape=None, source_start=None, source_step=None, target_start=None, target_step=None)`: direct region fill — copies a sub-region of `source` into `target`'s corresponding positions (core `load_data` wrapper; target-first so `generate(copy_region, args=(template,))` fills the owned data); out-of-bounds silently clipped, returns the number of elements copied

```python
from cos_comparison.generate_layer import TensorGenerator, copy_region

tg = TensorGenerator([[0, 0], [0, 0]])
tg.generate(copy_region, args=([[1, 2], [3, 4]],))   # tg.data becomes the template
tg.set_point((0, 0), 9)                                # core set_item protocol write
```

> **Note**: Generation logic is written as external functions and executed through the unified `generate` entry.

## Interface (`cos_comparison.interface`)

Abstraction of the external interface. The following modules exist:

- **system_api**: `command(commands)` (returns `(out, err, returncode)`), `getpid`, `getppid`, `home_executable`, `kill`, `BaseProcess` (subprocess.Popen with piped stdio), `Process(executable, arg_list, **kwarg)` — subprocess wrapper with byte-buffered stdin/stdout/stderr, background reader threads (started synchronously in `__init__`), `execute(command)`, `get_stdout(clear=False)`, `get_stderr(clear=False)`, `get_stdin(clear=False)`, `stop(timeout=0.5, terminate=True)` — by default terminates and reaps the child and joins the reader threads; pass `terminate=False` to keep the child alive
- **call_api**: `BaseCallContainer(obj)` with `call(name, args, kwargs, init_func)`, `get_call`, `get`, `set`; `Module_CallContain(module_name, *args, package=None)`; `C_CallContainer(library_path, loader=ctypes.CDLL)` — `get_call(name, argstypes=(), restype=None)` sets ctypes `argstypes`/`restype` on the wrapped function; `CDLL_CallContainer` / `WinDLL_CallContainer`; `CallDict(init_dict)` with `add(tag, func)` / `call(tag, args)`
- **communicate_api**: class chain `BaseCommunicate → Communicate → IOCommunicate`, with concrete `FdCommunicate`, `PIPECommunicate(auto=False)`, `SocketCommunicate`, `FileCommunicate(file_path, mode="rb", buffering=-1)` — send/recv abstraction over OS fds, sockets and files
- **parallel_api**: `thread_lock`/`process_lock`, `thread_rlock`/`process_rlock`, composed ready-to-use locks `parallel_lock`/`parallel_rlock` (via `IntegrateContext`), `Thread`, `Process`, shared arrays (`share_array` / `load_array`), and `run_in_thread(target, args=(), kwargs=None, is_join=False, **kws)` — thread launching & joining, owned by interface
- **async_api**: `AsyncRunner()` — blocking async event host with thread-safe event injection (`add_event`/`remove`/`result`/`done`/`active`, timeout exit, exception isolation, single-life); `EventLoop(interval=0.01)` — periodic task host on top of `AsyncRunner` (`schedule(task, interval=None)` → handle, `unschedule`, `run`, `stop`, `wait`, `active`)

- **tools/context_tool**: `VoidContext()` (no-op context manager), `IntegrateContext(*contexts)` (composes context managers, exits in reverse order), `AsyncIntegrateContext(*a_context)` (async variant)
- **tools/math_tool**: `topology` — `Graph` / `DirectedGraph` (union-find components, cycle rank, Euler characteristic, iterative Tarjan strong components, Kahn topological sort, BFS reachability) plus the module-level `shortest_path_between(graph, src, dst)` (BFS over any `neighbors`-protocol graph; used by the logic-layer default algorithms); `DirectedGraph.shortest_path(src, dst)` is lock-safe (no nested locking)

> **Note**: `cos_comparison.interface` (including `interface.api` and `interface.tools`) imports cleanly in a fresh interpreter — `EventLoop`, `run_in_thread` and the integrated locks are exercised by `tests/test_layers.py` (`TestInterfaceApi`, `TestInterfaceTools`).

## Data (`cos_comparison.data`)

Unified data carrier interfaces.

- `DataWrap(data_body=None)`: intended to be a unified data carrier with `process(caller, ...)`, `call(name, ...)`, `getattr`/`setattr`, plus `__getitem__`/`__setitem__` and the `__get_item__`/`__set_item__` protocol

> **Note**: `cos_comparison.data` imports cleanly in a fresh interpreter — the `DataWrap` carrier and the tensor containers are exercised by `tests/test_layers.py` (`TestDataLayer`).

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