"""time_api tests: time utilities and time-data wrapper types."""

import time as _time
import unittest

from cos_comparison.interface.api.time_api import (
    Deadline,
    Stopwatch,
    TimeStamp,
    elapsed,
    format_time,
    iso_time,
    parse_time,
    sleep,
    timestamp,
)


class TestTimeUtilities(unittest.TestCase):
    def test_timestamp(self):
        t = timestamp()
        self.assertIsInstance(t, float)
        self.assertAlmostEqual(t, _time.time(), delta=2.0)

    def test_iso_time(self):
        s = iso_time()
        self.assertIn("T", s)
        self.assertEqual(len(s), 19)  # YYYY-MM-DDTHH:MM:SS

    def test_format_and_parse(self):
        fmt = "%Y-%m-%d %H:%M:%S"
        t = timestamp()
        s = format_time(t, fmt)
        self.assertAlmostEqual(parse_time(s, fmt), t, delta=1.0)

    def test_sleep_and_elapsed(self):
        start = _time.monotonic()
        sleep(0.02)
        self.assertGreaterEqual(elapsed(start), 0.01)


class TestTimeStamp(unittest.TestCase):
    def test_default_now(self):
        ts = TimeStamp()
        self.assertAlmostEqual(ts.value, timestamp(), delta=2.0)

    def test_conversions(self):
        from datetime import datetime as _dt
        ts = TimeStamp(timestamp())
        self.assertIn("T", ts.iso())
        self.assertEqual(ts.format("%Y"), format_time(ts.value, "%Y"))
        self.assertEqual(ts.datetime().year,
                         _dt.now().year)  # noqa: DTZ005 - local by design
        self.assertAlmostEqual(float(ts), ts.value, delta=1.0)

    def test_repr(self):
        self.assertIn("TimeStamp", repr(TimeStamp(0.0)))


class TestStopwatch(unittest.TestCase):
    def test_elapsed(self):
        sw = Stopwatch()
        _time.sleep(0.02)
        self.assertGreaterEqual(sw.elapsed(), 0.01)

    def test_lap_resets(self):
        sw = Stopwatch()
        _time.sleep(0.02)
        first = sw.lap()
        self.assertGreaterEqual(first, 0.01)
        second = sw.lap()
        self.assertLess(second, first)

    def test_reset_and_not_started(self):
        sw = Stopwatch(autostart=False)
        self.assertEqual(sw.elapsed(), 0.0)
        sw.start()
        self.assertGreaterEqual(sw.elapsed(), 0.0)
        sw.reset()
        self.assertEqual(sw.elapsed(), 0.0)


class TestDeadline(unittest.TestCase):
    def test_remaining_and_expired(self):
        dl = Deadline(0.05)
        self.assertGreater(dl.remaining(), 0.0)
        self.assertFalse(dl.expired())
        _time.sleep(0.1)
        self.assertLessEqual(dl.remaining(), 0.0)
        self.assertTrue(dl.expired())

    def test_extend(self):
        dl = Deadline(0.01)
        dl.extend(0.1)
        self.assertGreater(dl.remaining(), 0.05)

    def test_repr(self):
        self.assertIn("Deadline", repr(Deadline(1.0)))


class TestExports(unittest.TestCase):
    def test_all_exports(self):
        import cos_comparison.interface.api.time_api as mod
        for name in ("TimeStamp", "Stopwatch", "Deadline", "timestamp",
                     "iso_time", "format_time", "parse_time", "sleep",
                     "elapsed"):
            self.assertTrue(hasattr(mod, name), name)


if __name__ == "__main__":
    unittest.main()
