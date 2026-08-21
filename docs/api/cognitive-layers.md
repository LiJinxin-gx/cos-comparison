# Cognitive Layer APIs

API surface of the non-core layers: `sense_layer`, `memory_layer`, `brain_layer`, `action_layer`, `generate_layer`, plus cross-cutting `interface`, `data`, and `test_tool`.

> **Compatibility note**: These layers are under active development. All packages import cleanly; `interface` is exercised by dedicated tests. The core module remains production-ready.

## Contents

- [Sense Layer](#sense-layer)
- [Memory Layer](#memory-layer)
- [Brain Layer](#brain-layer)
- [Action Layer](#action-layer)
- [Generate Layer](#generate-layer)
- [Interface](#interface)
- [Data](#data)
- [Test Tools](#test-tools)

---

## Sense Layer

Receives external stimuli and exposes them to core computation.

### Receptor / TensorReceptor

```python
from cos_comparison.sense_layer import Receptor

r = Receptor(data)
r.initialize(caller, *args, **kwargs)  # caller(data, *args, **kwargs)
r.receptor(caller, args=(), kwargs=None)
```

`TensorReceptor` stores raw data and provides:
- `point(index)` — element access via core `get_item` protocol (`__get_item__` authoritative, plain indexing fallback)
- `comparison_passive(output=None, **kwargs)` / `comparison_active(output=None, **kwargs)` — shortcuts to core modes

### data_match

Template matching that iterates matching positions (core filter functions used internally):

```python
from cos_comparison.sense_layer import data_match
hits = list(data_match(data, template, low=0.3))
```

**Signature:** `data_match(data, template, start=None, end=None, step=None, algorithm=None, low=None, high=None, inclusive=(True, True))`

- Runs core active comparison and yields positions whose match value lies in `[low, high]` (omitted bounds match everything)
- `algorithm=None` resolves the core default inside the backend (valid on every backend)
- `inclusive=(lo_in, hi_in)` controls endpoint membership
- Lazy iterator; usable standalone or via `receptor()` delegation

---

## Memory Layer

Memory backends (map / table / database) with unified wrappers.

### Status

`Flag` enum: `READ`, `WRITE`, `EXECUTE` (aliases `ST_READ`, `ST_WRITE`, `ST_EXECUTE`).

### Memory (basememory)

Lifecycle-based memory with pluggable function rules:

```python
from cos_comparison.memory_layer.memory import Memory

m = Memory(memory_obj,
           init_func=None, save_func=None, commit_func=None,
           rollback_func=None, refer_func=None, close_func=None)
```

| Method | Description |
|--------|-------------|
| `initialize(*args, **kwargs)` | Initialize the memory object |
| `save(*args, **kwargs)` | Save data |
| `commit(*args, **kwargs)` | Commit pending changes |
| `rollback(*arg, **kwarg)` | Roll back pending changes |
| `refer(*args, **kwargs)` | Read data |
| `close(*args, **kwargs)` | Close and release |
| `process(caller, *args, **kwargs)` | Apply caller to memory object |
| `call(name, *args, **kwargs)` | Call a method of the memory object |

### MapMemory

Dict-backed memory with transaction cache and atomic commit:

```python
from cos_comparison.memory_layer.memory import MapMemory

m = MapMemory(map_obj, close_commit=False, closer=None)
m.save(key, value, nesting=False, create=True)  # record transaction
m.commit()      # apply all cached transactions atomically
m.rollback()    # discard pending transactions
m.refer(key)    # read value (nested traversal optional)
m.close()       # commit if close_commit, release memory
```

- `Transaction(key, value, nesting=True, create=True)` — atomic write record; hashable, iterable

### TableMemory

`MapMemory` subclass: `save(keys, value)` and `refer(keys)` always use nested traversal with auto-creation.

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
    print(db.cursor.fetchall())  # [(1.0,), (2.0,)]
```

| Method | Description |
|--------|-------------|
| `execute(command, arg=())` | Execute single SQL |
| `executemany(command, args=()))` | Execute parameterized SQL (re-raises real errors) |
| `commit()` / `rollback()` / `close()` | Transaction control |

- `database_tool` omitted → interface default driver (`interface.api.DATABASE_DRIVER`, sqlite3); `RuntimeError` if no driver available

### Wrappers

| Class | Description | Key methods |
|-------|-------------|-------------|
| `MemoryWrap(memory_body=None, ...)` | Unified interface over any memory | `process()`, `call()`, `get()`, `set()` |
| `MemoryWrapPool(pool=None)` | List-like pool of `MemoryWrap` | `add()`, `set()`, `operate()` |
| `MemoryWrapMap(map_pool=None)` | Dict-keyed pool | `add(index, wrap)`, `get_by_name(name)` |

```python
from cos_comparison.memory_layer.memory import MapMemory, MemoryWrap

m = MapMemory({}); m.save("k", 42); m.commit()
wrap = MemoryWrap(memory_body=m, name="test", level=1)
wrap.process(lambda body, *a, **k: body.refer("k"))  # 42
wrap.call("refer", ("k",))                             # 42
```

---

## Brain Layer

### Symbolic Logic

Three-valued style logic system for cognitive reasoning.

### Control Flow

Flat containers (`Sequence` / `Branch` / `Loop`) yield FUNCTIONS — the
caller executes them (`for func in ctrl: func()`); containers never invoke
them. Nested flows are expanded iteratively (explicit stack, no recursion)
by `ControlFlatten`; custom flattening via subclassing. `ControlFlowDriver`
composes flows dynamically through the sequence protocol
(`driver[i] = flow`), mapping keys (`driver[i]["if_func"]`) and multi-index
access (`driver[i, key] = value`). `Control` is a pure placeholder marker.


| Class | Description |
|-------|-------------|
| `Logic(Flag)` | `TRUE`, `SURE`; constants `Logic_true`, `Logic_sure`, `sure_true` |
| `Variable(name, value=None)` | Named variable |
| `No_limit()` | Constraint container containing everything |
| `LogicError(Exception)` | Logical error |
| `Atomic_proposition(subject, verb, objects, adv, limit, status)` | Structured proposition; `__bool__` based on `status & TRUE`; dict-style access |
| `Logic_bind(reason, result, limit, status)` | Implication between propositions |
| `Logic_context(name="", binds=None, ...)` | Knowledge context with delegated slots |

`Logic_context` slots: `init_func`, `add_func`, `pop_func`, `judge_func`. Default `judge_func` answers by graph reachability over binds (shortest path via topology; `return_path=True` returns rule list, `[]` when `a==b`, `None` when unreachable).

```python
from cos_comparison.brain_layer import logic

ctx = logic.Logic_context(name="world",
                          add_func=lambda binds, lb, **kw: binds.append(lb))
p1 = logic.Atomic_proposition(subject="sky", verb="is", objects="blue")
p2 = logic.Atomic_proposition(subject="sky", verb="is", objects="red")
ctx.add(logic.Logic_bind(p1, p2))
print(ctx.logic_judge(p1, p2))  # True (graph-reachability judge)
```

### Probabilistic Logic

Uncertain reasoning with conditional probabilities.

| Class/Function | Description |
|----------------|-------------|
| `UnionEvent(*event)` / `IntersectionEvent(*event)` | Frozenset-based event classes |
| `GlobalEvent()` / `global_event` | Relative probability benchmark |
| `event_bind(name, event)` | Conditional probability storage; `bind(event, p)` validates `0 ≤ p ≤ 1` |
| `event_context(name="", binds=None, ...)` | Protocol-style context; `binds=None` → `EventBinds()` (dict subclass with cached dependency graph, stats, `strict` switch) |

**Axioms** (all probabilities relative to `global_event`): relativism, reflexivity `P(X|X)=1`, chain rule, Bayes duality, union.

**Default resolution**: exact hit → relative Bayes over direct references (`strict`, division guarded) → shortest-path chain fallback; `0.0` only when unreachable.

Related: `chain_probability`, `default_probability_func`, `strict_probability_func`, `union_probability`, `consistency_diagnostic`, `EventBinds`, `EventContextProtocol`.

### Reflex

| Class | Description |
|-------|-------------|
| `Trigger(trigger, callback, stack=None, ...)` | Binds trigger to callback via shared result stack; returns `0` on success |
| `Monitor(maintainer=None, poller=None, ...)` | Pure monitoring logic — all mechanisms delegated to `interface` (no `asyncio`/`threading`/`time` imports) |

`Monitor` default: poller = `EventLoop`, maintainer = `run_in_thread`, bookkeeping guarded by `parallel_lock`.

| Method | Description |
|--------|-------------|
| `add_event(trigger, callback=None, *, interval=None, times=-1)` | Probe until truthy → fire callback; auto-removed after `times` hits; returns handle |
| `remove(handle)` | Remove event |
| `run(*a, **k)` | Blocking drive |
| `maintrain(*a, **k)` | Background thread via `run_in_thread` |
| `stop()`, `wait(timeout=None)`, `is_running()` | Lifecycle control |
| `hits(handle=None)`, `errors(handle=None)` | Statistics |

```python
from cos_comparison.brain_layer.reflex import Monitor

m = Monitor()
m.add_event(lambda: obj.value > 100, on_high, interval=0.02, times=3)
m.maintrain()  # block-free carrying through interface.run_in_thread
```

### Mapper

| Class | Description |
|-------|-------------|
| `BaseMap(ABC)` | Abstract `__getitem__`/`__setitem__`/`__contains__` |
| `Map(map_obj=None, map_func=None, set_func=None, contain_judge_func=None)` | Three-slot protocol container with default dict-protocol implementations; unconfigured `map_obj` raises `TypeError` |

```python
from cos_comparison.brain_layer.mapper import Map

m = Map(map_obj={})
m["a"] = 1       # default set_func
m["a"]           # 1 (default map_func)
"a" in m         # True (default contain_judge_func)
```

---

## Action Layer

| Class | Description |
|-------|-------------|
| `ActionResult(out=None, err=None)` | Per-call result: `out` = return value, `err` = captured exception |
| `ExecuterDriver(caller_list=(), worker_func=None)` | Ordered callable list with indexed invocation |

| Method | Description |
|--------|-------------|
| `call(index, args=(), kwargs=None)` | Synchronous invocation of `call_list[index]` |
| `call_all(args=(), kwargs=None)` | Submit all entries strictly in order on background worker; non-blocking, returns self |
| `wait(timeout=None)` | Wait for batch; `False` on timeout (running functions not interrupted) |
| `is_done()`, `out(index)`, `err(index)`, `results`, `clear()` | Batch status and per-entry access |

- `worker_func` (callable or object with `launch`/`submit`/`run`) replaces default background launcher; default uses `interface.api.parallel_api.Thread`

```python
from cos_comparison.action_layer import ExecuterDriver

d = ExecuterDriver([f1, f2, f3])
d.call_all()                 # starts batch in background
d.wait(timeout=5.0)          # False when still running
print(d.out(0), d.err(1))    # captured return / exception
```

---

## Generate Layer

Generation and modification tools over wrapped data.

| Class/Function | Description |
|----------------|-------------|
| `Generator(data)` | `fix(call, args=(), kwargs=None)` applies callable to wrapped data |
| `TensorGenerator(data)` | `generate(func, args=(), kwargs=None)` — unified delegation entry running `func(self.data, ...)`; `set_point(index, value)` via core `set_item` protocol |
| `copy_region(target, source, *, ...)` | Region fill (core `load_data` wrapper); out-of-bounds clipped; returns elements copied |

```python
from cos_comparison.generate_layer import TensorGenerator, copy_region

tg = TensorGenerator([[0, 0], [0, 0]])
tg.generate(copy_region, args=([[1, 2], [3, 4]],))  # tg.data becomes template
tg.set_point((0, 0), 9)                              # core set_item protocol write
```

> Generation logic is written as external functions and executed through the unified `generate` entry.

---

## Interface

External interface abstraction (standard library only).

### API Modules

| Module | Key components |
|--------|----------------|
| `system_api` | `command(commands)` → `(out, err, returncode)`; `Process(executable, arg_list)` — subprocess wrapper with byte-buffered stdio, background reader threads; `execute()`, `get_stdout()`, `get_stderr()`, `stop(timeout=0.5, terminate=True)` |
| `call_api` | `BaseCallContainer`, `Module_CallContain(module_name)`, `C_CallContainer(library_path)` (ctypes argtypes/restype), `CallDict(init_dict)` with `add(tag, func)`/`call(tag, args)` |
| `communicate_api` | `FdCommunicate`, `PIPECommunicate`, `SocketCommunicate`, `FileCommunicate` — send/recv over fds, sockets, files |
| `parallel_api` | `thread_lock`/`process_lock`, `parallel_lock`/`parallel_rlock`, `Thread`, `Process`, `share_array`/`load_array`, `run_in_thread(target, ...)` |
| `async_api` | `AsyncRunner()` — blocking async host with thread-safe event injection, timeout exit, exception isolation; `EventLoop(interval=0.01)` — periodic task host |

### Tools

| Module | Key components |
|--------|----------------|
| `tools/context_tool` | `VoidContext()`, `IntegrateContext(*contexts)`, `AsyncIntegrateContext(*a_context)` |
| `tools/func_tool` | `ComposalFunction` — callable composed from multiple functions sharing a stack (first/last direct, middle steps read slots); `ComposalFunctionManage` (delegated maintenance); `FuncHelper` (stdlib helper slots: partial/reduce/compose/wrap/itemgetter/attrgetter); `FuncWrap` (discard-first-arg wrapper) |
| `tools/math_tool` | `topology` — `Graph`/`DirectedGraph` (union-find, iterative Tarjan SCC, Kahn topological sort, BFS reachability, Eulerian paths); `shortest_path_between(graph, src, dst)` (BFS, lock-safe) |
| `tools/math_tool/fourier` | Generic multi-dimensional DFT/IDFT, recursion-free |

> `cos_comparison.interface` imports cleanly in a fresh interpreter; `EventLoop`, `run_in_thread` and integrated locks are exercised by `tests/test_layers.py`.

---

## Data

Unified data carrier interfaces.

| Class | Description |
|-------|-------------|
| `DataWrap(data_body=None)` | Unified carrier: `process()`, `call()`, `getattr`/`setattr`, `__getitem__`/`__setitem__`, `__get_item__`/`__set_item__` protocol |
| `Tensor(BaseTensor, core.vector_map_as_tensor)` | Tensor container inheriting core stride-based tensor; loads via `load_as_default_data` |
| `SafeTensor(Tensor)` | Adds lock around `__setitem__`/`__set_item__` for parallel safety |
| `ParallelTensor(Tensor)` | Backed by shared array for cross-process use |
| `Task(caller, args=(), kwargs=None)` | Callable task wrapper |

> `cos_comparison.data` imports cleanly; exercised by `tests/test_layers.py` (`TestDataLayer`).

---

## Test Tools

Stable and fully working.

| Tool | Description |
|------|-------------|
| `Timer(start=0.0, timer=perf_count)` | `mark()`, `get_time()`, `reset()` — high-resolution timing |
| `ResultManager(output=None)` | Output collection: `write()`, `lines()`, `content()`, `clear()`; `output` callback redirects lines anywhere; `default_result` is default sink |
| `format_bytes(n)` | Human-readable byte formatting |
| `MemoryProbe(...)` | Memory tracing (tracemalloc, lazy-imported): `start()/stop()/current()/peak()/snapshot()/diff(top_n)/report()` |
| `ErrorWatcher(...)` | Error recording; context form **swallows** block errors (run-space protection); `count()/last()/stats()/clear()/watch(fn)` |
| `TraceProbe(...)` | Call-level tracing (`sys.settrace`): `start()/stop()/count()/depth_peak()/elapsed()/report()` |
| `@memory_report`, `@error_watch`, `@trace_report` | Decorators attaching probes to functions |

```python
from cos_comparison.test_tool import Timer, error_watch

t = Timer()
...  # work
t.mark()
print(t.get_time())

@error_watch()
def risky():
    raise ValueError("recorded, not raised")
```

---

**See also:** [Seven-Layer Architecture](../architecture/seven-layer.md) · [Core Module](core.md)
