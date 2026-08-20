"""
Testing and debugging utilities.
Provides Timer for benchmarking and attach-style debug probes:
ResultManager (output collection), MemoryProbe, ErrorWatcher, TraceProbe.
All probes attach via decorator, context manager or explicit start/stop,
collect results into a ResultManager, and never write to standard streams.
"""
#It gives basic tools to debug.

#--------- import ----------
import sys
import time

#-------- time tools ---------
perf_count = time.perf_counter

class Timer:
    __slots__ = ("timer","total_time","last_time")
    def __init__( self , start = 0.0 , timer = perf_count ):
        self.total_time = start 
        self.timer = timer
        self.last_time = self.timer()
        
    def mark(self):
        a = self.last_time
        self.last_time = self.timer()
        self.total_time += ( self.last_time - a )
        
    def get_time(self):
        return self.total_time + self.timer() - self.last_time

    def reset(self):
        self.total_time = 0.0
        self.last_time = self.timer()

#-------- result manager ---------
class ResultManager:
    """Collect output lines; forward each line to an injected output callback."""
    __slots__ = ("output", "_lines")
    def __init__(self, output=None):
        self.output = output
        self._lines = []
    def write(self, text):
        if self.output is not None:
            self.output(text)
        else:
            self._lines.append(text)
    def lines(self):
        return list(self._lines)
    def content(self):
        return "\n".join(self._lines)
    def clear(self):
        self._lines = []

default_result = ResultManager()

def format_bytes(n):
    """Format a byte count into a short human readable string."""
    if n < 1024:
        return "%d B" % n
    for unit in ("KB", "MB", "GB", "TB"):
        n = n / 1024.0
        if n < 1024:
            return "%.1f %s" % (n, unit)
    return "%.1f PB" % n

def _resolve_func(source, names):
    """Resolve an injected callable or object into a function; None if no match."""
    if source is None:
        return None
    if callable(source):
        return source
    for name in names:
        func = getattr(source, name, None)
        if callable(func):
            return func
    return None

#-------- memory probe ---------
_tracemalloc = None

def _import_tracemalloc():
    global _tracemalloc
    if _tracemalloc is None:
        import tracemalloc
        _tracemalloc = tracemalloc
    return _tracemalloc

def _default_mem_start():
    _import_tracemalloc().start()

def _default_mem_stop():
    _import_tracemalloc().stop()

def _default_mem_current():
    current, _ = _import_tracemalloc().get_traced_memory()
    return current

def _default_mem_peak():
    _, peak = _import_tracemalloc().get_traced_memory()
    return peak

def _default_mem_snapshot():
    return _import_tracemalloc().take_snapshot()

def _default_mem_diff(base, current, top_n):
    if base is None or current is None:
        return []
    result = []
    for stat in current.compare_to(base, "lineno")[:top_n]:
        frame = stat.traceback[0]
        result.append((stat.size_diff, "%s:%d" % (frame.filename, frame.lineno)))
    return result

class MemoryProbe:
    """Attach memory tracing (tracemalloc by default); slots are injectable."""
    __slots__ = ("_current_func", "_peak_func", "_snapshot_func", "_diff_func",
                 "_start_func", "_stop_func", "_manager", "_baseline", "active",
                 "_last_current", "_last_peak")
    def __init__(self, output=None, current_func=None, peak_func=None,
                 snapshot_func=None, diff_func=None, start_func=None, stop_func=None):
        self._current_func = _resolve_func(current_func, ("current",)) or _default_mem_current
        self._peak_func = _resolve_func(peak_func, ("peak",)) or _default_mem_peak
        self._snapshot_func = _resolve_func(snapshot_func, ("snapshot",)) or _default_mem_snapshot
        self._diff_func = _resolve_func(diff_func, ("diff",)) or _default_mem_diff
        self._start_func = _resolve_func(start_func, ("start",)) or _default_mem_start
        self._stop_func = _resolve_func(stop_func, ("stop",)) or _default_mem_stop
        self._manager = default_result if output is None else ResultManager(output)
        self._baseline = None
        self._last_current = 0
        self._last_peak = 0
        self.active = False
    def start(self):
        if self.active:
            return
        self._last_current = 0
        self._last_peak = 0
        self._start_func()
        self.active = True
        try:
            self._baseline = self._snapshot_func()
        except Exception:
            self._baseline = None    # snapshot unavailable: diff disabled
    def stop(self):
        if not self.active:
            return
        self._last_current = self._current_func()
        self._last_peak = self._peak_func()
        self._stop_func()
        self.active = False
    def current(self):
        if not self.active:
            return self._last_current    # frozen at stop
        return self._current_func()
    def peak(self):
        if not self.active:
            return self._last_peak    # frozen at stop
        return self._peak_func()
    def snapshot(self):
        return self._snapshot_func()
    def diff(self, top_n=10):
        return self._diff_func(self._baseline, self._snapshot_func(), top_n)
    def report(self):
        text = "MemoryProbe: current=%s peak=%s" % (
            format_bytes(self.current()), format_bytes(self.peak()))
        parts = self.diff()
        if parts:
            text += " top diffs:"
            for size, where in parts:
                text += " %s@%s" % (format_bytes(size), where)
        self._manager.write(text)
        return text
    def __enter__(self):
        self.start()
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False

#-------- error watcher ---------
class ErrorWatcher:
    """Record raised errors; context form swallows errors (run-space protection)."""
    __slots__ = ("_report_func", "_manager", "_records", "active")
    def __init__(self, output=None, report_func=None):
        self._report_func = _resolve_func(report_func, ("report",))
        self._manager = default_result if output is None else ResultManager(output)
        self._records = []
        self.active = False
    def start(self):
        self.active = True
    def stop(self):
        self.active = False
    def record(self, exc_type, exc_val, exc_tb):
        if not self.active:
            return
        frame = exc_tb
        while frame is not None and frame.tb_next is not None:
            frame = frame.tb_next
        where = None
        if frame is not None:
            where = "%s:%d" % (frame.tb_frame.f_code.co_filename, frame.tb_lineno)
        self._records.append((exc_type, str(exc_val), where))
        if self._report_func is not None:
            self._report_func(exc_type, exc_val, exc_tb)
    def count(self, err_type=None):
        if err_type is None:
            return len(self._records)
        return sum(1 for item in self._records if item[0] is err_type)
    def last(self):
        return self._records[-1] if self._records else None
    def clear(self):
        self._records = []
    def stats(self):
        result = {}
        for item in self._records:
            result[item[0]] = result.get(item[0], 0) + 1
        return result
    def report(self):
        text = "ErrorWatcher: %d error(s)" % len(self._records)
        for err_type, message, where in self._records:
            text += "\n  %s: %s @ %s" % (err_type.__name__, message, where)
        self._manager.write(text)
        return text
    def __enter__(self):
        self.start()
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.record(exc_type, exc_val, exc_tb)
            return True    # swallow: protect the outer run space
        return False
    def watch(self, fn, *args, **kwargs):
        """Run fn under protection: record and swallow any error."""
        self.start()
        try:
            return fn(*args, **kwargs)
        except Exception:
            exc_type, exc_val, exc_tb = sys.exc_info()
            self.record(exc_type, exc_val, exc_tb)
            return None
        finally:
            self.stop()

#-------- trace probe ---------
class TraceProbe:
    """Attach call-level tracing (sys.settrace); counts calls and depth."""
    __slots__ = ("_trace_func", "_manager", "active", "_count", "_depth",
                 "_depth_peak", "_skip_self", "_started_at", "_elapsed")
    def __init__(self, output=None, trace_func=None, skip_self=True):
        self._trace_func = _resolve_func(trace_func, ("trace",)) or self._default_trace
        self._manager = default_result if output is None else ResultManager(output)
        self.active = False
        self._count = 0
        self._depth = 0
        self._depth_peak = 0
        self._skip_self = skip_self
        self._started_at = None
        self._elapsed = 0.0
    def _default_trace(self, frame, event, arg):
        if self._skip_self and frame.f_code.co_filename == __file__:
            return None
        if event == "call":
            self._count += 1
            self._depth += 1
            if self._depth > self._depth_peak:
                self._depth_peak = self._depth
            return self._default_trace
        if event == "return" and self._depth > 0:
            self._depth -= 1
        return None
    def start(self):
        if self.active:
            return
        self._count = 0
        self._depth = 0
        self._depth_peak = 0
        self._started_at = perf_count()
        sys.settrace(self._trace_func)
        self.active = True
    def stop(self):
        if not self.active:
            return
        sys.settrace(None)
        self._elapsed += perf_count() - self._started_at
        self.active = False
    def count(self):
        return self._count
    def depth_peak(self):
        return self._depth_peak
    def elapsed(self):
        base = self._elapsed
        if self.active:
            base += perf_count() - self._started_at
        return base
    def report(self):
        text = "TraceProbe: calls=%d depth_peak=%d elapsed=%.3f s" % (
            self._count, self._depth_peak, self.elapsed())
        self._manager.write(text)
        return text
    def __enter__(self):
        self.start()
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False

#-------- attach decorators ---------
def memory_report(fn=None, *, output=None):
    """Decorator: run fn once, report memory usage into the result manager."""
    def deco(func):
        def wrapped(*args, **kwargs):
            probe = MemoryProbe(output=output)
            probe.start()
            try:
                return func(*args, **kwargs)
            finally:
                probe.report()    # report while still tracing
                probe.stop()
        return wrapped
    return deco(fn) if fn is not None else deco

def error_watch(fn=None, *, output=None, report_func=None):
    """Decorator: run fn, record and swallow any error into the result manager."""
    def deco(func):
        def wrapped(*args, **kwargs):
            watcher = ErrorWatcher(output=output, report_func=report_func)
            watcher.start()
            try:
                return func(*args, **kwargs)
            except Exception:
                exc_type, exc_val, exc_tb = sys.exc_info()
                watcher.record(exc_type, exc_val, exc_tb)
                watcher.report()
                return None
            finally:
                watcher.stop()
        return wrapped
    return deco(fn) if fn is not None else deco

def trace_report(fn=None, *, output=None):
    """Decorator: run fn under call tracing, report counts into the result manager."""
    def deco(func):
        def wrapped(*args, **kwargs):
            probe = TraceProbe(output=output)
            probe.start()
            try:
                return func(*args, **kwargs)
            finally:
                probe.stop()
                probe.report()
        return wrapped
    return deco(fn) if fn is not None else deco
