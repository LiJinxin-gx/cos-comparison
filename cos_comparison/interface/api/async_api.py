"""
async_api.py - async (asyncio) orchestration abstraction.

Provides the AsyncRunner: a blocking asynchronous event host.

- ``run()`` drives the event loop in the *calling* thread - the class
  contains no thread management logic of its own; the caller decides which
  thread blocks inside ``run``.
- Events (coroutine objects or async factories) can be added thread-safely
  at any time, including while the loop is already running.
- ``remove`` / ``cancel_all`` provide thread-safe intervention on events.

Layered modules use this host instead of creating threads and event loops
directly (Modular Architecture: concurrency primitives supplied by
interface).
"""

import asyncio
import concurrent.futures
import inspect
import threading
import time

__all__ = ("AsyncRunner",)


class AsyncRunner:
    """
    Blocking asynchronous event host (single-life).

    - ``run(timeout=None)``: blocks in the calling thread, driving the event
      loop. Ends when ``stop()`` is requested from another thread or when
      ``timeout`` seconds elapse. The loop is closed afterwards; a new
      AsyncRunner instance is required for a new session.
    - ``add_event(awaitable, *, name=None, on_done=None)``: thread-safe
      submission. Coroutine objects and async factories are both accepted.
      Events added before ``run`` are queued and loaded at loop start.
    - ``remove`` / ``cancel_all``: thread-safe intervention.
    - ``is_running`` / ``active`` / ``done`` / ``result`` / ``exceptions``:
      thread-safe queries.
    """

    __slots__ = (
        "_queue", "_known", "_done", "_tasks", "_futures", "_exceptions",
        "_results", "_seq", "_lock", "_loop", "_alive", "_started",
        "_exit_event",
    )

    def __init__(self):
        self._queue = []            # pending event descriptions (pre-run)
        self._known = set()         # handles: active or queued
        self._done = set()          # handles: finished (ok / error / cancel)
        self._tasks = {}            # handle -> asyncio.Task (loop side only)
        self._futures = {}          # handle -> concurrent.futures.Future
        self._exceptions = {}       # handle -> BaseException (finished events)
        self._results = {}          # handle -> (value, exc|None) permanent
        self._seq = 0
        self._lock = threading.Lock()
        self._loop = None
        self._alive = False
        self._started = False       # run() was entered (single life)
        self._exit_event = threading.Event()

    # ------------------------------------------------------------------ run

    def run(self, timeout=None):
        """
        Block in the calling thread and drive the event loop.

        Ends when ``stop()`` is requested from another thread, or when
        ``timeout`` seconds elapse. After returning, the loop is closed and
        this runner cannot be started again.
        """
        with self._lock:
            if self._started:
                raise RuntimeError("AsyncRunner is single-life; use a new instance")
            self._started = True

        loop = asyncio.new_event_loop()
        with self._lock:
            self._loop = loop
            self._alive = True
        asyncio.set_event_loop(loop)

        with self._lock:
            pending, self._queue = self._queue, []
        for desc in pending:
            self._dispatch(desc)

        deadline = None if timeout is None else time.monotonic() + timeout
        try:
            while True:
                with self._lock:
                    if not self._alive:
                        break
                if deadline is not None and time.monotonic() >= deadline:
                    break
                loop.run_until_complete(asyncio.sleep(0.001))
        finally:
            with self._lock:
                handles = list(self._tasks)
            for handle in handles:
                self._cancel_task(handle)
            try:
                loop.run_until_complete(asyncio.sleep(0.005))
            except Exception:
                pass
            loop.close()
            with self._lock:
                self._loop = None
                self._alive = False
            self._exit_event.set()

    # ------------------------------------------------------------------ stop

    def stop(self, timeout=0.2):
        """
        Thread-safely request cancellation of all events and end of ``run``
        (waits up to ``timeout`` seconds for ``run`` to return).
        """
        with self._lock:
            loop = self._loop
        if loop is None:
            return
        loop.call_soon_threadsafe(self._request_stop)
        self._exit_event.wait(timeout=timeout)

    def _request_stop(self):
        with self._lock:
            self._alive = False
            handles = list(self._tasks)
        for handle in handles:
            self._cancel_task(handle)

    # ------------------------------------------------------------------ add

    def add_event(self, awaitable, *, name=None, on_done=None):
        """
        Thread-safely add an event.

        ``awaitable`` is a coroutine object or an async factory returning
        one. If the loop is not running yet the event is queued and loaded
        when ``run`` starts. Returns a handle string.
        """
        with self._lock:
            self._seq += 1
            handle = "ev%d" % self._seq
            self._known.add(handle)
            self._futures[handle] = concurrent.futures.Future()
            desc = (handle, awaitable, name, on_done)
            loop = self._loop
            if loop is None:
                self._queue.append(desc)
                return handle
        loop.call_soon_threadsafe(self._dispatch, desc)
        return handle

    # ------------------------------------------------------------------ intervention

    def remove(self, handle):
        """
        Thread-safely cancel a single event (running or queued).

        Returns False when the handle is unknown or already finished.
        """
        with self._lock:
            if handle not in self._known:
                return False
            self._known.discard(handle)
            loop = self._loop
        if loop is None:
            with self._lock:
                self._queue = [d for d in self._queue if d[0] != handle]
            fut = self._futures.get(handle)
            exc = asyncio.CancelledError()
            if fut is not None:
                fut.set_exception(exc)
            with self._lock:
                self._results[handle] = (None, exc)
                self._done.add(handle)
        else:
            loop.call_soon_threadsafe(self._cancel_task, handle)
        return True

    def cancel_all(self):
        """Thread-safely cancel every active and queued event."""
        with self._lock:
            handles = list(self._known)
        for handle in handles:
            self.remove(handle)

    # ------------------------------------------------------------------ queries

    def is_running(self):
        with self._lock:
            return self._alive

    def active(self):
        with self._lock:
            return sorted(self._known)

    def done(self, handle):
        with self._lock:
            return handle in self._done

    def result(self, handle, timeout=None):
        """Blocking fetch of an event result (thread-safe)."""
        with self._lock:
            res = self._results.get(handle)
            if res is not None:
                value, exc = res
                if exc is not None:
                    raise exc
                return value
            fut = self._futures.get(handle)
        if fut is None:
            raise KeyError(handle)
        return fut.result(timeout)

    def exceptions(self):
        with self._lock:
            return dict(self._exceptions)

    # ------------------------------------------------------------------ loop side

    def _dispatch(self, desc):
        handle, awaitable, name, on_done = desc
        with self._lock:
            if handle not in self._known:
                fut = self._futures.pop(handle, None)
                exc = asyncio.CancelledError()
                if fut is not None:
                    fut.set_exception(exc)
                self._results[handle] = (None, exc)
                self._done.add(handle)
                return
        try:
            coro = awaitable() if inspect.iscoroutinefunction(awaitable) else awaitable
            if not inspect.iscoroutine(coro):
                raise TypeError("event must be a coroutine or an async factory")
            task = self._loop.create_task(coro)
        except BaseException as exc:
            self._finish_early(handle, exc)
            return
        task.add_done_callback(lambda t, h=handle: self._finish(h, t))
        if on_done is not None:
            task.add_done_callback(
                lambda t, h=handle, cb=on_done: self._fire_done(h, t, cb)
            )
        with self._lock:
            self._tasks[handle] = task

    def _finish_early(self, handle, exc):
        with self._lock:
            fut = self._futures.pop(handle, None)
            self._known.discard(handle)
            self._done.add(handle)
            self._exceptions[handle] = exc
            self._results[handle] = (None, exc)
        if fut is not None:
            fut.set_exception(exc)

    def _finish(self, handle, task):
        exc = None
        try:
            value = task.result()
        except asyncio.CancelledError:
            exc = asyncio.CancelledError("cancelled")
        except BaseException as e:
            exc = e
        with self._lock:
            fut = self._futures.pop(handle, None)
            self._tasks.pop(handle, None)
            self._known.discard(handle)
            self._done.add(handle)
            if exc is not None:
                self._exceptions[handle] = exc
                self._results[handle] = (None, exc)
            else:
                self._results[handle] = (value, None)
        if fut is not None:
            if exc is not None:
                fut.set_exception(exc)
            else:
                fut.set_result(value)

    def _fire_done(self, handle, task, on_done):
        try:
            exc = task.exception()
        except asyncio.CancelledError:
            exc = asyncio.CancelledError("cancelled")
        except BaseException as e:
            exc = e
        try:
            on_done(handle, exc)
        except Exception:
            pass

    def _cancel_task(self, handle):
        task = self._tasks.get(handle)
        if task is not None:
            task.cancel()
