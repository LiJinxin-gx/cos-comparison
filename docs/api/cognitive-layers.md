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
- `threshold_map(pairs, default_value=0.0, output=None, start=None, shape=None, step=None, out_start=None, out_step=None)` — threshold sensing map: the read region is mapped through the `(func, value)` pair sequence into `output` (write region); returns a status code (`0` ok / `1` mismatch / `2` no output / `3` failure)
- `threshold_match(low=None, high=None, inclusive=(True, True), start=None, shape=None, step=None)` — threshold position iterator (`data_filter` + `threshold_judge`; lazily yields matching positions)

Module-level `elementwise_extract(*tensors, func=None, output=None, start=None, shape=None, step=None, out_start=None, out_step=None)` — unified element-wise extraction: the read region is extracted via core `get_item`, passed to `core.elementwise`; a write region tries an output slice view first and falls back to `set_item`; returns a status code (output=None is decided by the underlying elementwise).

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
| `executemany(command, args=())` | Execute parameterized SQL (re-raises real errors) |
| `commit()` / `rollback()` / `close()` | Transaction control |

- `database_tool` omitted → interface default driver (`interface.api.DATABASE_DRIVER`, sqlite3); `RuntimeError` if no driver available

### IOStreamMemory

An io stream as the memory carrier — a delegating `Memory` whose working
defaults store records in MapMemory key/value style on the stream:

```python
from cos_comparison.interface.api.io_api import IOFile, IOMemory
from cos_comparison.memory_layer.memory import IOStreamMemory

m = IOStreamMemory(IOMemory("", binary=False))       # in-memory carrier
m.save("name", "cos")        # queue (MapMemory-compatible key/value)
m.commit()                   # append records at the tail + flush
m.refer("name")              # recall "cos"
```

- Carrier: an io_api `IOStream` — `IOFile(path, mode)` (file; `"r"` /
  `"a"` / `"a+"` …) or `IOMemory` (BytesIO / StringIO); mode-adaptive —
  read-only carriers reject `commit`, write-only ones reject `refer`
- Memory-level slots (`save_func` / `commit_func` / `rollback_func` /
  `refer_func` / `init_func` / `close_func`) decide the operation
  signatures — working defaults align with `MapMemory`: `save(key, value)`
  (key must be hashable; re-saving overwrites, keeping the newest record),
  uncommitted records are invisible until `commit`, `refer(key)` recalls
  via a lazily scanned key → offset index (existing stream content is
  restored on first use)
- Stream-interaction slots transcribe the carrier methods by default
  (`open_func` / `read_func` / `write_func` / `seek_func` / `tell_func` /
  `flush_func` / `readline_func` / `stream_close_func` / `capability_func`);
  `encode_func` / `decode_func` are the record-format extension points —
  the working default stores text records as repr lines (resolved with
  `ast.literal_eval`, so literal values round-trip with their type) and
  binary records as repr key + length-prefixed raw content
- Mapping protocol (`__getitem__` / `keys()` / `len` / `in`) — wraps
  directly with `MemoryWrap`

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
| `Feedback(receive_func=None, dispatch_func=None, manage_func=None)` | Trigger-style reflex hub: paired (feedback object, feedback function) registrations; all hub functions are delegating slots |

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

`Feedback` (Trigger-style hub): `register(obj, func)` pairs a feedback
object with the feedback function that operates on it; `receive()` (the
hub-receive function — default **non-blocking**, returns the background
thread) triggers the hub, `dispatch()` (the hub-feedback function —
default synchronous) hands each feedback function its own paired object;
errors are caught (kept on `last_error`) and the dispatch continues.
The hub functions are delegating slots — inject a synchronous
`receive_func` or a custom `dispatch_func` / `manage_func` to replace the
delivery/management loop.

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
| `TensorGenerator(data)` | `generate(func, args=(), kwargs=None)` — unified delegation entry running `func(self.data, ...)`; `set_point(index, value)` via core `set_item` protocol; `transform_self(func, *others, start=None, shape=None, step=None, out_start=None, out_step=None)` — self-modifying element-wise transform (core `elementwise` with output = self or a write region); returns a status code |
| `copy_region(target, source, *, ...)` | Region fill (core `load_data` wrapper); out-of-bounds clipped; returns elements copied |
| `transform_self(tensor, func, *others, ...)` | Module-level self-modifying element-wise transform (same semantics as the method) |

```python
from cos_comparison.generate_layer import TensorGenerator, copy_region

tg = TensorGenerator([[0, 0], [0, 0]])
tg.generate(copy_region, args=([[1, 2], [3, 4]],))  # tg.data becomes template
tg.set_point((0, 0), 9)                              # core set_item protocol write
```

> Generation logic is written as external functions and executed through the unified `generate` entry.

---

## Extension Layer

Proactive plugin batch aggregation (`extension_layer.plugin`).

### PluginPool

`PluginPool(resources=None, plugins=None, func_pool=None)` — three keyed pools (each defaults to a list, key = index; duck-supports any keyed container):

| Pool | Contents |
|------|----------|
| `resources` | hosted resources (shared data / contexts) |
| `plugins` | hosted plugins (external objects, unchanged — no registration, no modification) |
| `func_pool` | directly callable objects (functions / callables) |

```python
from cos_comparison.extension_layer.plugin import PluginPool

p = PluginPool()
p.add_plugin(0, external_plugin)          # host unchanged external plugin
p.call_plugin(0, "method", arg)           # plugin.method(arg)
p.add_func(0, my_function)                # direct callable
p.call_func(0, value)                     # my_function(value)
p.add_resource(0, ctx)                    # shared resource
p.get_resource(0)
```

Methods: `add_resource/add_plugin/add_func(key, value)` (mappings take `c[key]=value`; sequences append); `call_func(name, *args, **kwargs)`; `call_plugin(name, method, *args, **kwargs)`; `get_plugin_attr(name, attr)`; `get_resource(name)`; `get_plugin(name)`.

---

## Interface

External interface abstraction (standard library only).

### API Modules

| Module | Key components |
|--------|----------------|
| `system_api` | `command(commands, input=None, timeout=None)` → `(out, err, returncode)`; `Process(executable, arg_list)` — subprocess wrapper with byte-buffered stdio, background reader threads; `execute()`, `get_stdout()`, `get_stderr()`, `stop(timeout=0.5, terminate=True)`; `getpid()`/`getppid()`/`kill(pid, signal)` |
| `io_api` | `BaseIO` (ABC); `IOStream` — delegated stream container; `IOFile(path, mode, encoding)` — lazy file stream; `IOMemory(data, binary)` — in-memory BytesIO/StringIO; `read_file(path=None, fd=None, encoding=None)` / `write_file(path=None, data=None, fd=None, encoding=None)` convenience functions |
| `database_api` | `DatabaseToolWrap(tool, connect_func, cursor_func)` — distributed DB driver abstraction; PEP 249 exception hierarchy; `DatabaseCursor`/`DatabaseConnection` wrappers |
| `time_api` | `timestamp` / `iso_time` / `format_time` / `parse_time` / `sleep` / `elapsed`; types `TimeStamp` / `Stopwatch` / `Deadline` |
| `call_api` | `BaseCallContainer`, `Module_CallContain(module_name)`, `C_CallContainer(library_path)` (ctypes argtypes/restype), `CDLL_CallContainer`/`WinDLL_CallContainer`, `CallDict(init_dict)` with `add(tag, func)`/`call(tag, args)` |
| `communicate_api` | `FdCommunicate`, `PIPECommunicate`, `SocketCommunicate`, `FileCommunicate`, `IOCommunicate` — send/recv over fds, sockets, files; `Communicate` base with delegating slots |
| `parallel_api` | `thread_lock`/`process_lock`, `parallel_lock`/`parallel_rlock`, `Thread`, `Process`, `share_array(dtype, length)`/`load_array(dtypes, sequence, length=None, start=0)`, `run_in_thread(target, args=(), kwargs=None, is_join=False, **kws)`; `thread_barrier`/`thread_bounded_semaphore`/`thread_event`/`thread_semaphore`; `SuperParallel` — grid-style layered parallel framework (scale setup `sp[1,2,3]` / `sp.grid(dim)` coordinates / `sp(*args)` invocation, `getIdx()` layer hierarchy, delegating slots); `DefaultParallel` — default process-thread two-layer super-concurrency model (1-D auto-expansion, 2-D explicit).  GPU path verified on Intel Arc (OpenCL — element-space coordinates map to work items; ~6-9x pure-compute speedup at n ≥ 10^6; benchmark data in `docs/exploration/README.md`) |
| `async_api` | `AsyncRunner()` — blocking async host with thread-safe event injection, timeout exit, exception isolation; `EventLoop(interval=0.01)` — periodic task host |
| `file_api` | `FileManager` with `copy`/`move`/`remove`/`read`/`write`/`list`/`info`/`hash`; `copy_path(src, dst)` / `move_path(src, dst)` / `remove_path(path)` / `make_dirs(path)`; `file_hash(path=None, fd=None, algo='sha256', chunk=65536)`; `file_info(path=None, fd=None)`; `file_match(path=None, pattern=None, fd=None, start=0, encoding=None)`; `find_files(root, pattern, recursive=True)`; `list_dir(path=None, fd=None, pattern=None, recursive=False, sort=False)` |

### Tools

| Module | Key components |
|--------|----------------|
| `tools/context_tool` | `VoidContext()`, `IntegrateContext(*contexts)`, `AsyncIntegrateContext(*a_context)` |
| `tools/func_tool` | `ComposalFunction` — callable composed from multiple functions sharing a stack (first/last direct, middle steps read slots); `ComposalFunctionManage` (delegated maintenance); `FuncHelper` (stdlib helper slots: partial/reduce/compose/wrap/itemgetter/attrgetter); `FuncWrap` (discard-first-arg wrapper) |
| `tools/math_tool/topology` | `Graph` — `add_edge`, `components_count`, `cycle_rank`, `edges_count`, `euler_characteristic`, `is_connected`, `vertices_count`; `DirectedGraph` — `add_edge`, `strong_components` (Tarjan, iterative), `strong_components_count`, `topological_sort` (Kahn), `reachable` (BFS), `shortest_path` (BFS), `is_dag`, `is_weakly_connected`, `weak_components_count`, `in_degree`/`out_degree`, `has_eulerian_path`/`has_eulerian_circuit`, `neighbors`; `shortest_path_between(graph, src, dst)` (BFS, lock-safe); `Euler_characteristic_compute_by_cell(cell_list)` |
| `tools/math_tool/fourier` | `dft(data, axis=None)` / `idft(data, axis=None)` — generic multi-dimensional DFT/IDFT, recursion-free; `power_spectrum(data, axis=None)`; `dft_kernel_real(shape, frequencies, scales=1.0, offsets=0.0, amplitudes=1.0, biases=0.0)` / `dft_kernel_imag(...)` — kernel generation for matched filtering |
| `tools/math_tool/linear_algebra` | Dimension-generic duck-typed ops: `dot` / `norm` / `normalize` / `scale` / `add` / `multiply` / `power` / `clip` / `flatten` / `tensor_sum` / `tensor_mean` — tensor out via the `output` keyword, integer status return; identical Python / C behaviour |
| `tools/math_tool/unit_map` | `UnitMap` — run-folding unit mapper: any objects as units (duck), consecutive-equal runs fold into single real flag elements (variable-length runs → fixed-length elements, ready for tensor mapping); instance-held cumulative content table with query (`flag_of`) / decode compatibility surface; stateful output (`put(output=None, buffering=None)` — file-pointer continuation), `add(*seq)` input, `get_state`/`set_state`/`clear`; frequency statistics (`total()` / `count(obj)` / `runs(obj)` / `most_common(k)` / `count_vector()`) — count-based (no normalization), dynamic (no shared global counter, safe under concurrent adds, live with the open pending run); `window_units`/`map_data` with a `probe_dim` dimension-probe limit (default 1 — atomic-scale guard: nested rows stay atomic units; explicit N-D needs `probe_dim=k`); fully iterative (windows, content-signature folding, deep equality — no recursion); appending is a recurrence (adding data after a completed add only processes the new part); identical Python / C behaviour (C99, zero warnings under MSVC /W4+ and WSL gcc 15 -Wall -Wextra -Wpedantic -Werror) |

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
