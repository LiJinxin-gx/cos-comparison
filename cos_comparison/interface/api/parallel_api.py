"""
parallel_api.py - abstraction of underlying parallel (concurrency) tools.

Synchronous primitives (threads / processes / locks / events / semaphores
/ barriers / shared memory); async orchestration lives in async_api.py.
``thread_*``/``process_*`` are factory types; ``parallel_*`` are
ready-to-use integrated instances; ``make_parallel_*()`` creates fresh ones.

On spawn platforms (Windows) module-level process locks guard threads only;
use the make_* factories inside the worker for cross-process sync.
"""

import threading
import multiprocessing

from ..tools.context_tool import IntegrateContext

__all__ = (
    "Thread", "Process",
    "thread_lock", "process_lock", "parallel_lock",
    "thread_rlock", "process_rlock", "parallel_rlock",
    "thread_event", "process_event",
    "thread_semaphore", "process_semaphore",
    "thread_bounded_semaphore", "process_bounded_semaphore",
    "thread_barrier", "process_barrier",
    "make_parallel_lock", "make_parallel_rlock",
    "share_array", "load_array",
    "run_in_thread",
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