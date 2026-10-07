"""v0.5.3 S0 hardening: critical-condition checks for the touched C areas.

Covers: clip extreme / degenerate bounds and value parity with the pure
Python reference; GC cycle collection for UnitMap / Graph / DirectedGraph;
and a concurrent clip smoke test (real threads on free-threaded builds).
"""

import gc
import importlib
import math
import random
import threading
import unittest

from cos_comparison.interface.tools.math_tool import topology as topo
from cos_comparison.interface.tools.math_tool import unit_map as um

_py_la = importlib.import_module(
    "cos_comparison.interface.tools.math_tool.linear_algebra")
try:
    _c_la = importlib.import_module(
        "cos_comparison.interface.tools.math_tool._linear_algebra")
except ImportError:  # pragma: no cover - compiled extension not built
    _c_la = None


def _call_clip(mod, a, lo, hi):
    out = [[0.0] * len(row) for row in a]
    status = mod.clip(a, lo, hi, output=out)
    return status, out


CLIP_CASES = [
    ([[0.0, 1.0, -1.0]], 0.0, 1.0),
    ([[2.0, 3.0]], 2.0, 2.0),
    ([[1.0, 5.0]], 5.0, 1.0),
    ([[float("nan"), 1.0]], -1.0, 2.0),
    ([[float("inf"), -float("inf")]], -1.0, 1.0),
    ([[-0.0, 0.0]], 0.0, 0.0),
    ([[1e308, -1e308]], -1e307, 1e307),
    ([[0.5]], -1.0, 1.0),
]


@unittest.skipIf(_c_la is None, "C extension not built")
class TestClipBoundaries(unittest.TestCase):

    def test_py_c_parity(self):
        for a, lo, hi in CLIP_CASES:
            st_p, out_p = _call_clip(_py_la, a, lo, hi)
            st_c, out_c = _call_clip(_c_la, a, lo, hi)
            self.assertEqual(st_p, _py_la.OK)
            self.assertEqual(st_c, st_p)
            for row_p, row_c in zip(out_p, out_c):
                for x, y in zip(row_p, row_c):
                    if math.isnan(x) or math.isnan(y):
                        self.assertTrue(math.isnan(x) and math.isnan(y))
                    else:
                        self.assertEqual(x, y)

    def test_clip_without_output_status(self):
        self.assertEqual(_py_la.clip([[1.0]], 0.0, 1.0), _py_la.NO_OUTPUT)
        # the compiled module exposes no constants; the documented value is
        # shared with the pure reference
        self.assertEqual(_c_la.clip([[1.0]], 0.0, 1.0), _py_la.NO_OUTPUT)

    def test_clip_empty(self):
        st_c, out_c = _call_clip(_c_la, [], 0.0, 1.0)
        st_p, out_p = _call_clip(_py_la, [], 0.0, 1.0)
        self.assertEqual(st_p, _py_la.OK)
        self.assertEqual(st_c, st_p)
        self.assertEqual(out_c, out_p)


def _count(typ):
    return sum(1 for o in gc.get_objects() if isinstance(o, typ))


class TestCycleCollection(unittest.TestCase):

    def test_graph_cycle_collected(self):
        gc.collect()
        base = _count(topo.Graph)
        for _ in range(10):
            class V:
                pass
            v = V()
            g = topo.Graph()
            g.add_edge(v, v)
            v.g = g
            del v, g
        gc.collect()
        self.assertEqual(_count(topo.Graph), base)

    def test_directed_graph_cycle_collected(self):
        gc.collect()
        base = _count(topo.DirectedGraph)
        for _ in range(10):
            class V:
                pass
            v = V()
            g = topo.DirectedGraph()
            g.add_edge(v, v)
            v.g = g
            del v, g
        gc.collect()
        self.assertEqual(_count(topo.DirectedGraph), base)

    def test_unit_map_cycle_collected(self):
        gc.collect()
        base = _count(um.UnitMap)
        for _ in range(10):
            m = um.UnitMap()
            m.put(m)
            del m
        gc.collect()
        self.assertEqual(_count(um.UnitMap), base)


@unittest.skipIf(_c_la is None, "C extension not built")
class TestConcurrentClip(unittest.TestCase):

    def test_concurrent_clip_matches_serial(self):
        errors = []

        def worker(seed):
            try:
                rnd = random.Random(seed)
                for _ in range(50):
                    lo = -rnd.uniform(0.1, 3.0)
                    hi = rnd.uniform(0.1, 3.0)
                    a = [[rnd.uniform(-5.0, 5.0) for _ in range(16)]
                         for _ in range(4)]
                    out = [[0.0] * 16 for _ in range(4)]
                    status = _c_la.clip(a, lo, hi, output=out)
                    exp = [[lo if v < lo else (hi if v > hi else v)
                            for v in row] for row in a]
                    if status != 0 or out != exp:
                        errors.append((seed, lo, hi, status))
            except BaseException as exc:  # never let the thread die silently
                errors.append((seed, "exception", repr(exc)))

        threads = [threading.Thread(target=worker, args=(i,))
                   for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
