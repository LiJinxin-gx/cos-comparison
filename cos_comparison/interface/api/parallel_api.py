"""
parallel_api.py - abstraction of underlying parallel (concurrency) tools.

Synchronous primitives (threads / processes / locks / events / semaphores
/ barriers / shared memory); async orchestration lives in async_api.py.
``thread_*``/``process_*`` are factory types; ``parallel_*`` are
ready-to-use integrated instances; ``make_parallel_*()`` creates fresh ones.

On spawn platforms (Windows) module-level process locks guard threads only;
use the make_* factories inside the worker for cross-process sync.
"""

import itertools
import multiprocessing
import os
import threading
from concurrent.futures import ThreadPoolExecutor

from ..tools.context_tool import IntegrateContext

__all__ = (
    "DefaultParallel",
    "Layer",
    "Process",
    "SuperParallel",
    "Thread",
    "load_array",
    "make_parallel_lock",
    "make_parallel_rlock",
    "parallel_lock",
    "parallel_rlock",
    "process_barrier",
    "process_bounded_semaphore",
    "process_event",
    "process_lock",
    "process_rlock",
    "process_semaphore",
    "run_in_thread",
    "share_array",
    "thread_barrier",
    "thread_bounded_semaphore",
    "thread_event",
    "thread_lock",
    "thread_rlock",
    "thread_semaphore",
)

# -------- threads / processes --------
Thread = threading.Thread
Process = multiprocessing.Process

# -------- lock factories (types) --------
thread_lock = threading.Lock
process_lock = multiprocessing.Lock
thread_rlock = threading.RLock
process_rlock = multiprocessing.RLock

# -------- event / semaphore / barrier factories (types) --------
thread_event = threading.Event
process_event = multiprocessing.Event

thread_semaphore = threading.Semaphore
process_semaphore = multiprocessing.Semaphore

thread_bounded_semaphore = threading.BoundedSemaphore
process_bounded_semaphore = multiprocessing.BoundedSemaphore

thread_barrier = threading.Barrier
process_barrier = multiprocessing.Barrier

# -------- ready-to-use integrated instances --------
parallel_lock = IntegrateContext(process_lock(), thread_lock())
parallel_rlock = IntegrateContext(process_rlock(), thread_rlock())


def make_parallel_lock():
    """Create a fresh integrated lock (process + thread level)."""
    return IntegrateContext(process_lock(), thread_lock())


def make_parallel_rlock():
    """Create a fresh integrated reentrant lock (process + thread level)."""
    return IntegrateContext(process_rlock(), thread_rlock())


# -------- thread launching (thread management is an interface capability) --------
def run_in_thread(target, args=(), kwargs=None, is_join=False, **kws):
    """Run ``target`` in a background thread; ``is_join`` blocks the caller
    until the thread finishes."""
    thread = Thread(
        target=target,
        args=args,
        kwargs={} if kwargs is None else kwargs,
        **kws,
    )
    thread.start()
    if is_join:
        thread.join()


# -------- shared memory --------
def share_array(dtype, length):
    """Return a fresh array shareable across processes.

    dtype : multiprocessing.Array typecode (e.g. 'd'); length : elements.
    """
    return multiprocessing.Array(dtype, length)


def load_array(dtypes, sequence, length=None, start=0):
    """
    Fill a fresh shared array from a sequence, optionally from an offset.

    dtypes   : multiprocessing.Array typecode (e.g. 'd').
    sequence : indexable/iterable of values.
    length   : element count (None infers ``start + len(sequence)``).
    start    : offset of the first element (>= 0).

    If the sequence overflows the container, the tail is truncated and the
    remainder stays zero-filled; a whole-sequence slice copy is used when
    it fits exactly.
    """
    if isinstance(start, bool) or not isinstance(start, int) or start < 0:
        raise TypeError("start must be a non-negative integer, got %r" % (start,))
    if length is not None and (isinstance(length, bool) or
                               not isinstance(length, int) or length < 0):
        raise TypeError("length must be a non-negative integer, got %r"
                        % (length,))
    seq_len = len(sequence) if hasattr(sequence, "__len__") else None
    if length is None:
        if seq_len is None:
            # iterables without __len__ are materialized once to size the
            # container; pass length= to skip this for big streams
            sequence = list(sequence)
            seq_len = len(sequence)
        length = start + seq_len
    container = share_array(dtypes, length)
    if seq_len is None or seq_len == 0 or start + seq_len > length:
        # no-len/empty/overflow: element-wise copy clamped to the bounds
        for i, value in enumerate(sequence):
            idx = start + i
            if idx >= length:
                break
            container[idx] = value
    else:
        container[start:start + seq_len] = sequence
    return container

# -------- SuperParallel: grid-style layered parallel framework --------



class Layer:
    """One allocation layer: name, the assigned index (an integer - safe
    across parallel execution) and the layer size."""
    __slots__ = ("index", "name", "size")

    def __init__(self, name, index, size):
        self.name = name
        self.index = index
        self.size = size

    def __iter__(self):
        yield self.name
        yield self.index
        yield self.size

    def __repr__(self):
        return f"Layer({self.name!r}, {self.index}, {self.size})"


class SuperParallel:
    """Grid-style layered parallel framework (delegating slots).

    Scale setup is decoupled from actual invocation:

      sp[1, 2, 3]   scale setup (subscript protocol) - each element is
                    one allocation layer size (arbitrary depth)
      sp.grid(dim)  convenience coordinates in unified sequence form:
                    dim=1 yields (i,), dim=3 yields (i, j, k), ... over
                    the whole configured space
      sp(*args)     invocation protocol - the delegated ``batch`` runs
                    the kernel (set via set_kernel) in parallel; every
                    argument is passed straight to the kernel function

    ``getIdx()`` (callable as SuperParallel.getIdx() from kernels, also
    bound as an instance method) returns the hierarchy of ``Layer``
    allocators of the current execution unit; () outside any allocation.
    The allocation context is a CLASS-level thread-local, so kernels run
    in worker processes can query it without shipping the object.

    The implementation slots (allocate / batch) can be replaced by direct
    assignment - external engines (CUDA / Intel GPU wrappers etc.) plug
    in through the same protocol.  Output is not captured (the kernel
    decides what to do with its results)."""
    __slots__ = ("_shape", "allocate", "batch", "kernel")

    _ctx = threading.local()  # class-level allocation context

    def __init__(self, allocate=None, batch=None):
        self._shape = ()
        self.kernel = None
        self.allocate = (allocate if allocate is not None
                         else self.default_allocate)
        self.batch = batch if batch is not None else self.default_batch

    # -- scale setup (subscript protocol) ------------------------------

    def __getitem__(self, key):
        self._shape = (key,) if not isinstance(key, tuple) else key
        return self

    # -- convenience coordinates ----------------------------------------

    def grid(self, dim):
        """Unified coordinate sequences over the configured space:
        dim=1 yields ((0,), (1,), ...); dim>=2 yields odometer tuples
        (requires len(shape) == dim for dim >= 2)."""
        if not isinstance(dim, int) or dim < 1:
            raise ValueError("grid(dim): dim must be a positive int")
        shape = self._shape or (1,)
        if dim == 1:
            total = 1
            for s in shape:
                total *= s
            return ((i,) for i in range(total))
        if len(shape) != dim:
            raise ValueError(
                "grid(dim): dim >= 2 requires the scale setup to have "
                "the same dimension (or use grid(1))")
        return itertools.product(*[range(s) for s in shape])

    # -- invocation ------------------------------------------------------

    def __call__(self, *args, **kwargs):
        if self.kernel is None:
            raise RuntimeError("SuperParallel: no kernel set "
                               "(set_kernel first)")
        return self.batch(self, args, kwargs)

    def set_kernel(self, fn):
        """Bind the atomic (kernel) function executed per unit."""
        self.kernel = fn
        return self

    @classmethod
    def getIdx(cls):
        """The allocation hierarchy of the current unit: a tuple of
        Layer allocators (outermost first); () outside any allocation."""
        return getattr(cls._ctx, "idx", None) or ()

    @classmethod
    def _enter(cls, layers):
        cls._ctx.idx = layers

    @classmethod
    def _leave(cls):
        cls._ctx.idx = None

    # -- default (delegated) implementations ----------------------------

    def default_allocate(self, shape):
        """Abstract allocation: the configured shape as-is (layers)."""
        return tuple(shape)

    def default_batch(self, sp, args, kwargs):
        """Default batch: sequential execution over grid(1) coordinates
        (each coordinate one kernel call; getIdx per call)."""
        count = 0
        for coord in sp.grid(1):
            SuperParallel._enter((Layer("unit", coord[0], 1),))
            try:
                sp.kernel(*args, **kwargs)
            finally:
                SuperParallel._leave()
            count += 1
        return count


class DefaultParallel(SuperParallel):
    """Default process-thread two-layer super-concurrency model.

    One-dimensional setups (``sp[N]``) expand into
    ``P = min(cpu_count, N)`` processes with ``ceil(N / P)`` thread units
    each; two-dimensional setups (``sp[P, T]``) map P processes x T
    threads explicitly.  Every inner unit executes the kernel once; its
    layer hierarchy (process / thread) is available through getIdx().
    The kernel and its arguments must be pickleable for the process
    layer (use top-level functions)."""

    def _layers(self, sp):
        shape = sp._shape
        if not shape:
            return (), 0
        if len(shape) == 1:
            n = shape[0]
            procs = min((os.cpu_count() or 1), n)
            per = -(-n // procs)  # ceil
            return (procs, per), n
        if len(shape) == 2:
            return (shape[0], shape[1]), shape[0] * shape[1]
        raise ValueError("DefaultParallel supports one- or two-"
                         "dimensional scale setups")

    def _call_kernel(self, sp, args, kwargs, layers):
        SuperParallel._enter(layers)
        try:
            sp.kernel(*args, **kwargs)
        finally:
            SuperParallel._leave()

    def default_batch(self, sp, args, kwargs):
        """Execute the kernel over the configured space through the
        process-thread hierarchy (thread-only path when processes=1)."""
        (procs, per), total = self._layers(sp)
        if procs <= 1:
            self._thread_run(sp, args, kwargs, per)
        else:
            self._process_run(sp, args, kwargs, procs, per)
        return total

    def _thread_run(self, sp, args, kwargs, per):
        with ThreadPoolExecutor(max_workers=per) as pool:
            futures = []
            for t in range(per):
                def call(t=t):
                    self._call_kernel(
                        sp, args, kwargs,
                        (Layer("process", 0, 1),
                         Layer("thread", t, per)))
                futures.append(pool.submit(call))
            for f in futures:
                f.result()

    def _process_run(self, sp, args, kwargs, procs, per):
        from multiprocessing import Pool
        payload = [(p, sp.kernel, args, kwargs, procs, per)
                   for p in range(procs)]
        with Pool(procs) as pool:
            pool.map(_default_worker_run, payload)


def _default_worker_run(payload):
    """Top-level process worker of the default model: runs the thread
    layer inside the process (pickle boundary - receives only the
    kernel/args and its process index)."""
    p, kernel, args, kwargs, procs, per = payload
    with ThreadPoolExecutor(max_workers=per) as pool:
        futures = []
        for t in range(per):
            def call(t=t):
                SuperParallel._enter(
                    (Layer("process", p, procs),
                     Layer("thread", t, per)))
                try:
                    kernel(*args, **kwargs)
                finally:
                    SuperParallel._leave()
            futures.append(pool.submit(call))
        for f in futures:
            f.result()
