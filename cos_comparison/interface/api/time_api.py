"""time_api.py - time tools and time-data wrapper types.

Basic time utilities (timestamp / formatting / parsing / sleep / elapsed)
and data-wrapper types (TimeStamp / Stopwatch / Deadline) built on the
standard ``time`` and ``datetime`` modules.
"""

import time as _time
from datetime import datetime as _datetime

__all__ = (
    "Deadline",
    "Stopwatch",
    "TimeStamp",
    "elapsed",
    "format_time",
    "iso_time",
    "parse_time",
    "sleep",
    "timestamp",
)


def timestamp():
    """Current time in seconds (float, UTC epoch)."""
    return _time.time()


def iso_time(value=None):
    """ISO 8601 string of a timestamp (default: now, local time)."""
    if value is None:
        value = timestamp()
    return _time.strftime("%Y-%m-%dT%H:%M:%S", _time.localtime(value))


def format_time(value=None, fmt="%Y-%m-%d %H:%M:%S"):
    """Format a timestamp (default: now, local time)."""
    if value is None:
        value = timestamp()
    return _time.strftime(fmt, _time.localtime(value))


def parse_time(text, fmt="%Y-%m-%d %H:%M:%S"):
    """Parse a formatted time string back into a timestamp (local time)."""
    return _time.mktime(_time.strptime(text, fmt))


def sleep(seconds):
    """Sleep for the given number of seconds."""
    _time.sleep(seconds)


def elapsed(start_time):
    """Seconds elapsed since start_time (monotonic clock)."""
    return _time.monotonic() - start_time


class TimeStamp:
    """Wrapper type around a time value (epoch seconds) with conversion
    tools (iso / formatted / datetime)."""

    __slots__ = ("value",)

    def __init__(self, value=None):
        self.value = timestamp() if value is None else value

    def iso(self):
        """ISO 8601 string of this timestamp."""
        return iso_time(self.value)

    def format(self, fmt="%Y-%m-%d %H:%M:%S"):
        """Formatted string of this timestamp."""
        return format_time(self.value, fmt)

    def datetime(self):
        """The corresponding datetime object (local time)."""
        return _datetime.fromtimestamp(self.value)  # noqa: DTZ006 - local by design

    def __float__(self):
        return float(self.value)

    def __repr__(self):
        return f"TimeStamp({self.value!r})"


class Stopwatch:
    """Monotonic stopwatch: start / elapsed / lap / reset."""

    __slots__ = ("_start",)

    def __init__(self, autostart=True):
        self._start = _time.monotonic() if autostart else None

    def start(self):
        """Start (or restart) the stopwatch."""
        self._start = _time.monotonic()
        return self

    def elapsed(self):
        """Seconds since start (0.0 when not started)."""
        if self._start is None:
            return 0.0
        return _time.monotonic() - self._start

    def lap(self):
        """Seconds since start, then restart the stopwatch."""
        now = _time.monotonic()
        lap = 0.0 if self._start is None else now - self._start
        self._start = now
        return lap

    def reset(self):
        """Stop and clear the stopwatch."""
        self._start = None

    def __repr__(self):
        return f"Stopwatch(elapsed={self.elapsed()!r})"


class Deadline:
    """Countdown deadline: remaining / expired / extend."""

    __slots__ = ("_end",)

    def __init__(self, seconds):
        self._end = _time.monotonic() + seconds

    def remaining(self):
        """Seconds left (negative once expired)."""
        return self._end - _time.monotonic()

    def expired(self):
        """True when the deadline has passed."""
        return self.remaining() <= 0

    def extend(self, seconds):
        """Extend the deadline by the given seconds."""
        self._end += seconds
        return self

    def __repr__(self):
        return f"Deadline(remaining={self.remaining()!r})"
