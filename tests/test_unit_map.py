"""unit_map tests: run-folding unit mapper.  The C extension (_unit_map)
and the pure-Python reference (unit_map.py) are exercised with the same
shared cases; window/map_data helpers are tested per implementation."""

import importlib
import unittest

from cos_comparison.interface.tools import math_tool

try:
    from cos_comparison.interface.tools.math_tool import _unit_map as _c_mod
except ImportError:  # pragma: no cover - extension not built
    _c_mod = None

# the package attribute may alias the C extension (C-first __init__):
# import the submodule explicitly so the shared cases run against the
# pure-Python reference
_py_mod = importlib.import_module(
    "cos_comparison.interface.tools.math_tool.unit_map")
UnitMap = _py_mod.UnitMap


class UnitMapCase(unittest.TestCase):
    """Shared behavioural cases (run against both implementations)."""

    UNIT = None

    def _map(self, *args, **kwargs):
        if self.UNIT is None:
            self.skipTest("abstract shared-case base")
        return self.UNIT(*args, **kwargs)

    def test_run_folding(self):
        m = self._map()
        m.add("A", "A", "A", "B", "C", "C")
        self.assertEqual(m.put(), [0.0, 1.0, 2.0])

    def test_multi_add_and_flag_reuse(self):
        m = self._map()
        m.add([1, 1, 2])
        m.add([1, 3])
        self.assertEqual(m.put(), [0.0, 1.0, 0.0, 2.0])
        self.assertEqual(len(m), 3)

    def test_flag_of_and_contains(self):
        m = self._map()
        m.add([7, 7, 8])
        self.assertEqual(m.flag_of(7), 0.0)
        self.assertEqual(m.flag_of(8), 1.0)
        self.assertIsNone(m.flag_of(9))
        self.assertIn(7, m)
        self.assertNotIn(9, m)

    def test_decode_compatibility(self):
        m = self._map()
        m.add(["x", "y"])
        self.assertEqual(m.decode([0.0, 1.0]), ["x", "y"])
        self.assertIsNone(m.decode([99.0])[0])

    def test_state_round_trip(self):
        m = self._map()
        m.add([1, 1, 2])
        m.put()
        st = m.get_state()
        m2 = self._map()
        m2.set_state(st)
        self.assertEqual(len(m2), len(m))
        self.assertEqual(m2.put(), [])
        self.assertIn(1, m2)
        self.assertEqual(m2.flag_of(2), 1.0)

    def test_buffering_file_pointer(self):
        m = self._map()
        m.add([10, 10, 20, 30, 40])
        acc = []
        self.assertEqual(m.put(output=acc, buffering=2), 2)
        self.assertEqual(m.put(output=acc, buffering=2), 2)
        self.assertEqual(m.put(output=acc), 0)
        self.assertEqual(acc, [0.0, 1.0, 2.0, 3.0])

    def test_preallocated_sequence_output(self):
        m = self._map()
        m.add("X", "X", "Y")
        buf = [None] * 2
        self.assertEqual(m.put(output=buf), 2)
        self.assertEqual(buf, [0.0, 1.0])

    def test_unhashable_units(self):
        m = self._map()
        m.add([[1, 2], [1, 2], [1, 2], [3]])
        self.assertEqual(m.put(), [0.0, 1.0])
        self.assertEqual(len(m), 2)
        self.assertEqual(m.flag_of([1, 2]), 0.0)
        self.assertEqual(m.decode([1.0]), [[3]])

    def test_pending_closed_by_put(self):
        m = self._map()
        m.add("Z", "Z")
        self.assertEqual(m.put(), [0.0])

    def test_clear(self):
        m = self._map()
        m.add([1, 1, 2])
        m.clear()
        self.assertEqual(len(m), 0)
        self.assertEqual(m.put(), [])

    def test_generator_input(self):
        m = self._map()
        m.add(x for x in [1, 1, 2, 3])
        self.assertEqual(m.put(), [0.0, 1.0, 2.0])

    def test_custom_start_step(self):
        m = self._map(start=10.0, step=2.0)
        m.add([1, 1, 2])
        self.assertEqual(m.put(), [10.0, 12.0])

    def test_stats_counts_and_total(self):
        m = self._map()
        m.add([1, 1, 1, 2, 1, 3, 3, [4, 5], [4, 5]])
        m.add([1])  # pending 1 stays open: statistics are live
        self.assertEqual(m.total(), 10)
        self.assertEqual(m.count(1), 5)
        self.assertEqual(m.runs(1), 3)
        self.assertEqual(m.count(2), 1)
        self.assertEqual(m.runs(2), 1)
        self.assertEqual(m.count(3), 2)
        self.assertEqual(m.count([4, 5]), 2)  # unhashable unit
        self.assertEqual(m.count(9), 0)
        self.assertEqual(m.runs(9), 0)

    def test_stats_after_put_and_append(self):
        m = self._map()
        m.add([1, 1, 2])
        m.put()  # closes the pending run
        self.assertEqual(m.total(), 3)
        m.add([1, 1])
        self.assertEqual(m.total(), 5)  # incremental recurrence
        self.assertEqual(m.count(1), 4)
        self.assertEqual(m.runs(1), 2)

    def test_count_vector_flag_order(self):
        m = self._map()
        m.add([7, 7, 8, 9, 9, 9, [0]])
        self.assertEqual(m.flag_of(7), 0.0)
        self.assertEqual(m.count_vector(), [2, 1, 3, 1])

    def test_most_common_order(self):
        m = self._map()
        m.add([1, 1, 1, 2, 3, 3, 4])
        top = m.most_common(2)
        self.assertEqual(top[0][0], 1)
        self.assertEqual(top[0][1], 3)
        self.assertEqual(top[1][1], 2)  # only 3 has count 2; 2 and 4 have 1
        self.assertEqual(len(m.most_common()), 4)

    def test_stats_after_clear(self):
        m = self._map()
        m.add([1, 1, 2])
        m.clear()
        self.assertEqual(m.total(), 0)
        self.assertEqual(m.count(1), 0)
        self.assertEqual(m.count_vector(), [])
        self.assertEqual(m.most_common(), [])


    def test_threshold_filter(self):
        m = self._map(threshold=2)
        m.add(["a", "a", "b", "a"])
        flag = m.flag_of("a")
        self.assertEqual(m.put(), [flag, "b", flag])

    def test_length_segments(self):
        m = self._map(length=2)
        m.add([9] * 5)
        self.assertEqual(m.put(), [0.0, 0.0, 0.0])

    def test_length_buffering_carry(self):
        m = self._map(length=2)
        m.add([9] * 5)
        acc = []
        while m.put(output=acc, buffering=2):
            pass
        self.assertEqual(acc, [0.0, 0.0, 0.0])

    def test_merge_stream_order(self):
        a = self._map()
        a.add([1, 2, 2])
        b = self._map()
        b.add([2, 3])
        self.assertEqual((a + b).put(), [0.0, 1.0, 2.0])
        c = self._map()
        c.add([2, 3])
        d = self._map()
        d.add([1, 2, 2])
        self.assertNotEqual((c + d).put(), [0.0, 1.0, 2.0])
        self.assertEqual((a + b).flag_of(2), 1.0)

    def test_merge_associative_open(self):
        a = self._map()
        a.add([1, 1])
        b = self._map(threshold=4)
        b.add([2, 2])
        c = self._map()
        c.add([1, 1])
        merged = a + b + c
        self.assertEqual(merged.threshold, 1)
        self.assertEqual(merged.length, None)
        self.assertEqual(merged.put(), [0.0, 1.0, 0.0])

    def test_merge_errors(self):
        a = self._map()
        b = self._map(start=5.0)
        with self.assertRaises(ValueError):
            a + b
        used = self._map()
        used.add([1, 1])
        used.put()
        with self.assertRaises(ValueError):
            used + self._map()

    def test_streaming_visibility_eager(self):
        # eager registration: flags/len/count_vector are available before
        # put even with a threshold configured
        m = self._map(threshold=4)
        m.add([1, 1, 2])
        self.assertEqual(m.flag_of(1), 0.0)
        self.assertEqual(len(m), 2)
        self.assertEqual(m.count_vector(), [2, 1])

    def test_new_state_round_trip(self):
        m = self._map(threshold=2, length=2)
        m.add([1, 1, 1, 2])
        st = m.get_state()
        m2 = self._map()
        m2.set_state(st)
        self.assertEqual(m2.threshold, 2)
        self.assertEqual(m2.length, 2)
        self.assertEqual(m2.put(), m.put())


class TestPyUnitMap(UnitMapCase):
    UNIT = UnitMap

    def test_py_window_and_map(self):
        data = [[1, 1, 2], [1, 1, 2], [1, 1, 2]]
        units = list(_py_mod.window_units(data, (2, 2), step=(1, 1),
                                          probe_dim=2))
        self.assertEqual(len(units), 4)
        self.assertEqual(units[0], units[2])
        m = self._map()
        m.add(units)
        self.assertEqual(m.put(), [0.0, 1.0, 0.0, 1.0])
        self.assertEqual(
            _py_mod.map_data(data, (2, 2), step=(1, 1), probe_dim=2),
            [0.0, 1.0, 0.0, 1.0])

    def test_py_probe_dimension_default_1d(self):
        # atomic-scale guard: nested rows stay atomic units by default
        data = [[1, 2], [1, 2], [3]]
        units = list(_py_mod.window_units(data, local_size=1))
        self.assertEqual(units, [[[1, 2]], [[1, 2]], [[3]]])
        m = self._map()
        m.add([row for win in units for row in win])
        self.assertEqual(m.put(), [0.0, 1.0])

    def test_py_nd_length(self):
        m = self._map(shape=(2, 2), length=(1, 1))
        m.add([1, 1, 1, 1])
        self.assertEqual(m.put(), [0.0, 0.0, 0.0, 0.0])
        m = self._map(shape=(2, 2), length=(2, 2))
        m.add([1, 1, 1, 1])
        self.assertEqual(m.put(), [0.0])

    def test_py_map_data_threshold(self):
        data = [[1, 1, 2], [1, 1, 2]]
        out = _py_mod.map_data(data, (2, 2), step=(1, 1), probe_dim=2, threshold=2)
        self.assertEqual(out, [[[1, 1], [1, 1]], [[1, 2], [1, 2]]])

    def test_py_incremental_recurrence(self):
        # appending after a completed add equals a one-shot parse
        once = self._map()
        once.add([1, 1, 2, 3, 3])
        expected = once.put()
        inc = self._map()
        inc.add([1, 1, 2])
        inc.add([3, 3])  # appended after a "completed" segment
        self.assertEqual(inc.put(), expected)
        self.assertEqual(inc.flag_of(1), once.flag_of(1))


@unittest.skipIf(_c_mod is None, "C extension not built")
class TestCUnitMap(UnitMapCase):
    UNIT = _c_mod.UnitMap if _c_mod is not None else None

    def test_c_window_and_map(self):
        data = [[1, 1, 2], [1, 1, 2], [1, 1, 2]]
        units = _c_mod.window_units(data, (2, 2), step=(1, 1),
                                    probe_dim=2)
        self.assertEqual(len(units), 4)
        self.assertEqual(units[0], units[2])
        m = self._map()
        m.add(units)
        self.assertEqual(m.put(), [0.0, 1.0, 0.0, 1.0])
        self.assertEqual(
            _c_mod.map_data(data, (2, 2), step=(1, 1), probe_dim=2),
            [0.0, 1.0, 0.0, 1.0])


    def test_c_nd_length(self):
        m = self._map(shape=(2, 2), length=(1, 1))
        m.add([1, 1, 1, 1])
        self.assertEqual(m.put(), [0.0, 0.0, 0.0, 0.0])

    def test_c_map_data_threshold(self):
        data = [[1, 1, 2], [1, 1, 2]]
        out = _c_mod.map_data(data, (2, 2), step=(1, 1), probe_dim=2, threshold=2)
        self.assertEqual(out, [[[1, 1], [1, 1]], [[1, 2], [1, 2]]])


class TestModuleSelection(unittest.TestCase):
    def test_math_tool_exposes_unit_map(self):
        self.assertTrue(hasattr(math_tool, "unit_map"))
        self.assertTrue(hasattr(math_tool.unit_map, "UnitMap"))


if __name__ == "__main__":
    unittest.main()
