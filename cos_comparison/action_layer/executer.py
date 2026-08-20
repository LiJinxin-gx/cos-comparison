"""
action executor with delegated async execution.

Runs the call list strictly in order, in the background by default
(non-blocking call_all), capturing per-item out/err results.
Thread launching is delegated to interface (never imported here directly);
an injected worker replaces the default launcher entirely.
"""
#a executer tool set.

import time

from ..interface.api.parallel_api import Thread as _InterfaceThread

class ActionResult:
    """Result of one delegated call: out (return value) and err (exception)."""
    __slots__ = ("out", "err")
    def __init__(self, out=None, err=None):
        self.out = out
        self.err = err

def _resolve_func(source, names):
    """Resolve an injected callable or object into a worker; None if no match."""
    if source is None:
        return None
    if callable(source):
        return source
    for name in names:
        func = getattr(source, name, None)
        if callable(func):
            return func
    return None

def _default_worker(target, args, kwargs):
    """Default launcher: interface thread, started non-blocking."""
    thread = _InterfaceThread(target=target, args=args, kwargs=kwargs)
    thread.start()
    return thread

def _handle_done(handle):
    """Judge handle completion (Thread is_alive / Future done / None)."""
    if handle is None:
        return True
    alive = getattr(handle, "is_alive", None)
    if callable(alive):
        return not alive()
    done = getattr(handle, "done", None)
    if callable(done):
        return done()
    return False

def _wait_handle(handle, timeout):
    """Block until handle completes; False when the budget runs out."""
    join = getattr(handle, "join", None)
    if callable(join):
        join(timeout)
        return _handle_done(handle)
    deadline = None if timeout is None else time.monotonic() + timeout
    while not _handle_done(handle):
        if deadline is not None and time.monotonic() >= deadline:
            return False
        time.sleep(0.005)
    return True

class ExecuterDriver:
    """Execute the call list strictly in order via a delegated launcher."""
    __slots__ = ("call_list", "results", "_worker", "_handle")
    def __init__(self, caller_list=(), worker_func=None):
        self.call_list = list(caller_list)
        self.results = [ActionResult() for _ in self.call_list]
        self._worker = _resolve_func(worker_func, ("launch", "submit", "run")) or _default_worker
        self._handle = None
    def call(self, index, args=(), kwargs=None):
        """Synchronous call of one entry (original contract)."""
        kwargs = kwargs if kwargs is not None else {}
        return self.call_list[index](*args, **kwargs)
    def call_all(self, args=(), kwargs=None):
        """Submit all entries in order on a background worker; non-blocking."""
        self.wait()
        self.results = [ActionResult() for _ in self.call_list]
        self._handle = self._worker(self._run_serial, args, kwargs if kwargs is not None else {})
        return self
    def _run_serial(self, args=(), kwargs=None):
        kwargs = kwargs if kwargs is not None else {}
        for index, func in enumerate(self.call_list):
            try:
                out = func(*args, **kwargs)
                self.results[index] = ActionResult(out=out, err=None)
            except Exception as exc:
                self.results[index] = ActionResult(out=None, err=exc)
    def wait(self, timeout=None):
        """Wait for completion (default unlimited); False on timeout."""
        if self._handle is None or _handle_done(self._handle):
            return True
        return _wait_handle(self._handle, timeout)
    def is_done(self):
        """Whether the current batch has completed."""
        return _handle_done(self._handle)
    def out(self, index):
        """Captured return value of entry index (None if error or none)."""
        return self.results[index].out
    def err(self, index):
        """Captured exception of entry index (None when no error)."""
        return self.results[index].err
    def clear(self):
        """Reset results without touching the call list."""
        self.results = [ActionResult() for _ in self.call_list]
