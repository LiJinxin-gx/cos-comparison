"""
parallel_api.py - abstraction of underlying parallel (concurrency) tools.

Synchronous concurrency primitives (threads / processes / locks / events /
semaphores / barriers / shared memory). Async orchestration lives in
async_api.py (AsyncRunner).

Convention:
  - `thread_*` and `process_*` names are factory types (classes), like the
    underlying stdlib types;
  - `parallel_*` names are ready-to-use integrated instances (process-level
    lock + thread-level lock combined);
  - `make_parallel_*()` factories create fresh integrated instances on demand.

Note on spawn platforms (e.g. Windows): a multiprocessing Lock/RLock created
at module import time exists in each process independently and therefore only
guards threads inside one process, not cross-process critical sections. Use
the make_* factories inside the worker context when cross-process
synchronization is required.
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
    """
    Run ``target`` in a background thread.

    args/kwargs are passed to the target; ``is_join`` optionally blocks the
    caller until the thread finishes. Thread creation and lifecycle stay
    inside interface.
    """
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
    """
    Provide an array shareable across processes.

    dtype : a typecode accepted by multiprocessing.Array (e.g. 'd').
    length : number of elements.
    """
    return multiprocessing.Array(dtype, length)


def load_array(dtypes, sequence, length=None):
    """
    Fill a fresh shared array from a sequence (fast path: slice copy).

    Returns the shared array. The sequence must be indexable/iterable of the
    given length; when length is None it is inferred from the sequence.
    """
    length = len(sequence) if length is None else length
    container = share_array(dtypes, length)
    if hasattr(sequence, "__len__"):
        container[:] = sequence
    else:
        for i, value in enumerate(sequence):
            if i >= length:
                break
            container[i] = value
    return container