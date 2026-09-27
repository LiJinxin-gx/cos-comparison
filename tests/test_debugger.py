# -*- coding: utf-8 -*-
"""test_tool.debugger: Timer, ResultManager, MemoryProbe, ErrorWatcher, TraceProbe."""

import time
import unittest

from cos_comparison.test_tool.debugger import (
    Timer,
    ResultManager,
    MemoryProbe,
    ErrorWatcher,
    TraceProbe,
    format_bytes,
    memory_report,
    error_watch,
    trace_report,
)


class TestTimer(unittest.TestCase):
    """Timer benchmarking utility."""

    def test_timer_creation(self):
        t = Timer()
        self.assertGreaterEqual(t.total_time, 0.0)

    def test_mark_and_get_time(self):
        t = Timer()
        time.sleep(0.01)
        t.mark()
        elapsed = t.get_time()
        self.assertGreater(elapsed, 0.005)

    def test_reset(self):
        t = Timer()
        time.sleep(0.01)
        t.mark()
        t.reset()
        self.assertEqual(t.total_time, 0.0)


class TestResultManager(unittest.TestCase):
    """ResultManager collect and forward output."""

    def test_write_and_lines(self):
        rm = ResultManager()
        rm.write("line1\n")
        rm.write("line2\n")
        lines = rm.lines()
        self.assertEqual(len(lines), 2)

    def test_content(self):
        rm = ResultManager()
        rm.write("hello\n")
        rm.write("world\n")
        content = rm.content()
        self.assertIn("hello", content)
        self.assertIn("world", content)

    def test_clear(self):
        rm = ResultManager()
        rm.write("test\n")
        rm.clear()
        self.assertEqual(len(rm.lines()), 0)

    def test_output_callback(self):
        collected = []
        rm = ResultManager(output=lambda text: collected.append(text))
        rm.write("via callback\n")
        self.assertEqual(len(collected), 1)
        self.assertIn("callback", collected[0])


class TestFormatBytes(unittest.TestCase):
    """format_bytes human readable byte formatting."""

    def test_bytes(self):
        self.assertEqual(format_bytes(512), "512 B")

    def test_kilobytes(self):
        self.assertEqual(format_bytes(2048), "2.0 KB")

    def test_megabytes(self):
        self.assertEqual(format_bytes(5 * 1024 * 1024), "5.0 MB")

    def test_gigabytes(self):
        self.assertEqual(format_bytes(2 * 1024**3), "2.0 GB")


class TestMemoryProbe(unittest.TestCase):
    """MemoryProbe context manager and reporting."""

    def test_context_manager(self):
        with MemoryProbe() as probe:
            self.assertTrue(probe.active)
            # Allocate some memory
            _ = [0] * 10000
        self.assertFalse(probe.active)

    def test_report(self):
        with MemoryProbe() as probe:
            _ = [0] * 5000
            text = probe.report()
        self.assertIn("MemoryProbe", text)
        self.assertIn("current=", text)

    def test_injectable_functions(self):
        """Test with mock memory functions."""
        current_val = [1024]
        peak_val = [2048]

        def mock_current():
            return current_val[0]

        def mock_peak():
            return peak_val[0]

        probe = MemoryProbe(current_func=mock_current, peak_func=mock_peak)
        probe.start()
        self.assertEqual(probe.current(), 1024)
        self.assertEqual(probe.peak(), 2048)
        probe.stop()


class TestErrorWatcher(unittest.TestCase):
    """ErrorWatcher record and swallow errors."""

    def test_record_error(self):
        watcher = ErrorWatcher()
        watcher.start()
        try:
            raise ValueError("test error")
        except ValueError:
            import sys
            exc_type, exc_val, exc_tb = sys.exc_info()
            watcher.record(exc_type, exc_val, exc_tb)
        watcher.stop()
        self.assertEqual(watcher.count(), 1)
        self.assertEqual(watcher.count(ValueError), 1)

    def test_context_manager_swallow(self):
        watcher = ErrorWatcher()
        with watcher:
            raise RuntimeError("should be swallowed")
        self.assertEqual(watcher.count(RuntimeError), 1)

    def test_watch_method(self):
        watcher = ErrorWatcher()

        def failing_func():
            raise ValueError("boom")

        result = watcher.watch(failing_func)
        self.assertIsNone(result)
        self.assertEqual(watcher.count(), 1)

    def test_stats(self):
        watcher = ErrorWatcher()
        watcher.start()
        watcher.record(ValueError("e1"), None, None) if False else None
        # Manually add records
        watcher._records = [(ValueError, "e1", "loc1"),
                            (TypeError, "e2", "loc2"),
                            (ValueError, "e3", "loc3")]
        stats = watcher.stats()
        self.assertEqual(stats[ValueError], 2)
        self.assertEqual(stats[TypeError], 1)


class TestTraceProbe(unittest.TestCase):
    """TraceProbe call-level tracing."""

    def test_basic_tracing(self):
        probe = TraceProbe()
        probe.start()

        def sample_func(x):
            return x * 2

        result = sample_func(5)
        probe.stop()

        self.assertEqual(result, 10)
        self.assertGreater(probe.count(), 0)

    def test_depth_peak(self):
        probe = TraceProbe()
        probe.start()

        def func_a():
            func_b()

        def func_b():
            pass

        func_a()
        probe.stop()

        self.assertGreaterEqual(probe.depth_peak(), 1)


class TestDecorators(unittest.TestCase):
    """Decorator helpers: memory_report, error_watch, trace_report."""

    def test_memory_report_decorator(self):
        @memory_report()
        def test_func():
            return 42

        result = test_func()
        self.assertEqual(result, 42)

    def test_error_watch_decorator(self):
        @error_watch()
        def failing_func():
            raise ValueError("decorator test")

        result = failing_func()
        self.assertIsNone(result)

    def test_trace_report_decorator(self):
        @trace_report()
        def traced_func():
            return "traced"

        result = traced_func()
        self.assertEqual(result, "traced")


if __name__ == "__main__":
    unittest.main()
