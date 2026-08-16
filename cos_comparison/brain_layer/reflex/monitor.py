#It is a module to achieve monitors.
#
# Monitoring logic only: probe truthiness, callback triggering, hit/error
# bookkeeping, rule lifetime. Every mechanism (async periodic scheduling,
# threads, locks) is provided by interface (EventLoop / run_in_thread /
# parallel_lock) - this module imports no asyncio / threading / time.

import inspect

from ...interface.api import EventLoop, run_in_thread, parallel_lock

__all__ = (
    "Monitor",
    "default_maintainer", "default_runner", "default_poller",
    "default_scheduler", "default_stopper", "default_observer",
)

# ---------------------------------------------------------------------------
# monitoring logic (worker semantics; the periodic drive belongs to interface)

def _record_error(record, handle, exc):
    with parallel_lock:
        record.setdefault("errors", {}).setdefault(handle, []).append(exc)


def _make_worker(state, record):
    """One single probe watcher (monitoring semantics only)."""
    probe, callback = state["probe"], state["callback"]

    async def worker():
        # Read the handle under the same lock the scheduler holds while
        # binding it, so a worker that starts right after schedule() can
        # never observe the unbound (None) handle.
        with parallel_lock:
            handle = state["handle"]
        hit = False
        try:
            hit = bool(probe())
        except BaseException as exc:
            _record_error(record, handle, exc)
            return
        if not hit:
            return
        with parallel_lock:
            record.setdefault("hits", {}).setdefault(handle, 0)
            record["hits"][handle] += 1
            count = record["hits"][handle]
        if callback is not None:
            try:
                res = callback()
                if inspect.isawaitable(res):
                    await res
            except BaseException as exc:
                _record_error(record, handle, exc)
        if 0 < state["times"] and count >= state["times"]:
            state["poller"].unschedule(handle)

    return worker


# ---------------------------------------------------------------------------
# default delegations (semantics here, mechanisms imported from interface)

default_maintainer = run_in_thread                    # thread carrying (interface)
default_poller = EventLoop                            # async periodic host (interface)


def default_runner(poller, *args, **kwargs):
    """Drive the poller (blocking); the loop mechanism lives in interface."""
    return poller.run(*args, **kwargs)


def default_scheduler(poller, operation, record, *args, **kwargs):
    """
    op 'add'    : (trigger, callback, interval, times) -> handle|None
    op 'remove' : (handle) -> bool
    """
    if operation == "add":
        trigger, callback, interval, times = args
        if times == 0:
            return None
        state = {
            "handle": None,
            "probe": trigger,
            "callback": callback,
            "times": times,
        }
        # Bind the handle while holding parallel_lock: the worker reads it
        # under the same lock, so a worker already scheduled on a running
        # loop can never observe the unbound (None) handle.
        with parallel_lock:
            handle = poller.schedule(_make_worker(state, record), interval)
            state["handle"] = handle
        with parallel_lock:
            record.setdefault("hits", {})[handle] = 0
            record.setdefault("errors", {})[handle] = []
        return handle
    if operation == "remove":
        (handle,) = args
        return poller.unschedule(handle)
    raise ValueError("unknown scheduler operation: %r" % (operation,))


def default_stopper(poller, operation, *args, **kwargs):
    """op 'stop' | 'wait' | 'running' - all forwarded to the interface host."""
    if operation == "stop":
        return poller.stop(*args, **kwargs)
    if operation == "wait":
        return poller.wait(*args, **kwargs)
    if operation == "running":
        return poller.is_running()
    raise ValueError("unknown stopper operation: %r" % (operation,))


def default_observer(poller, operation, record, handle=None):
    """op 'hits' | 'errors' - bookkeeping snapshots (monitor logic side)."""
    if operation == "hits":
        with parallel_lock:
            if handle is not None:
                return record.get("hits", {}).get(handle)
            return dict(record.get("hits", {}))
    if operation == "errors":
        with parallel_lock:
            if handle is not None:
                return list(record.get("errors", {}).get(handle, []))
            return {h: list(v) for h, v in record.get("errors", {}).items()}
    raise ValueError("unknown observer operation: %r" % (operation,))


# ---------------------------------------------------------------------------
# Monitor: pure delegation container (monitoring logic, no mechanisms)

class Monitor:
    __slots__ = ("maintainer", "poller", "runner",
                 "scheduler", "stopper", "observer", "record")

    def __init__(self, maintainer=None, poller=None, runner=None,
                 scheduler=None, stopper=None, observer=None):
        self.maintainer = default_maintainer if maintainer is None else maintainer
        self.poller = default_poller() if poller is None else poller
        self.runner = default_runner if runner is None else runner
        self.scheduler = default_scheduler if scheduler is None else scheduler
        self.stopper = default_stopper if stopper is None else stopper
        self.observer = default_observer if observer is None else observer
        self.record = {"hits": {}, "errors": {}}

    # -- single-line delegations ---------------------------------------------
    def add_event(self, trigger, callback=None, *, interval=None, times=-1):
        return self.scheduler(self.poller, "add", self.record,
                              trigger, callback, interval, times)

    def remove(self, handle):
        return self.scheduler(self.poller, "remove", self.record, handle)

    def run(self, *args, **kwargs):
        return self.runner(self.poller, *args, **kwargs)

    def maintrain(self, *args, **kwargs):
        return self.maintainer(self.run, *args, **kwargs)

    def stop(self, *args, **kwargs):
        return self.stopper(self.poller, "stop", *args, **kwargs)

    def wait(self, *args, **kwargs):
        return self.stopper(self.poller, "wait", *args, **kwargs)

    def is_running(self):
        return self.stopper(self.poller, "running")

    def hits(self, handle=None):
        return self.observer(self.poller, "hits", self.record, handle)

    def errors(self, handle=None):
        return self.observer(self.poller, "errors", self.record, handle)