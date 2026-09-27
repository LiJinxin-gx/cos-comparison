"""Linear algebra tests (dimension-generic, duck typing, output keyword
pass-out, integer status return).  The pure-Python fallback and the C
extension must behave identically (external consistency)."""

import unittest

from cos_comparison.interface.tools.math_tool import linear_algebra as la

try:
    from cos_comparison.interface.tools.math_tool import _linear_algebra as _c
except ImportError:
    _c = None

try:
    from cos_comparison.interface.tools.math_tool import linear_algebra as _py
except ImportError:
    _py = None


class DuckList:
    """Non-list container built on the sequence protocol (duck typing:
    __iter__ / __getitem__ / __setitem__ / __len__)."""

    def __init__(self, items):
        self._items = list(items)

    def __iter__(self):
        return iter(self._items)

    def __getitem__(self, i):
        return self._items[i]

    def __len__(self):
        return len(self._items)

    def __setitem__(self, i, v):
        self._items[i] = v

    def __repr__(self):
        return repr(self._items)


class TestLinearAlgebra(unittest.TestCase):
    def test_dot(self):
        out = [0.0]
        self.assertEqual(la.dot([1.0, 2.0, 3.0], [1.0, 2.0, 3.0],
                                output=out), 0)
        self.assertAlmostEqual(out[0], 14.0)

    def test_norm(self):
        out = [0.0]
        self.assertEqual(la.norm([[1.0, 2.0], [3.0, 4.0]], output=out), 0)
        self.assertAlmostEqual(out[0], 5.477225575051661)

    def test_normalize(self):
        out = [0.0, 0.0]
        self.assertEqual(la.normalize([3.0, 4.0], output=out), 0)
        self.assertAlmostEqual(out[0], 0.6)
        self.assertAlmostEqual(out[1], 0.8)

    def test_scale(self):
        out = [[0.0, 0.0], [0.0, 0.0]]
        self.assertEqual(la.scale([[1.0, 2.0], [3.0, 4.0]], 3.0,
                                  output=out), 0)
        self.assertEqual(out, [[3.0, 6.0], [9.0, 12.0]])

    def test_add(self):
        out = [[0.0, 0.0], [0.0, 0.0]]
        self.assertEqual(la.add([[1.0, 2.0], [3.0, 4.0]],
                                [[5.0, 6.0], [7.0, 8.0]],
                                output=out), 0)
        self.assertEqual(out, [[6.0, 8.0], [10.0, 12.0]])

    def test_multiply(self):
        out = [[0.0, 0.0], [0.0, 0.0]]
        self.assertEqual(la.multiply([[1.0, 2.0], [3.0, 4.0]],
                                     [[2.0, 2.0], [2.0, 2.0]],
                                     output=out), 0)
        self.assertEqual(out, [[2.0, 4.0], [6.0, 8.0]])

    def test_power(self):
        out = [[0.0, 0.0], [0.0, 0.0]]
        self.assertEqual(la.power([[1.0, 2.0], [3.0, 4.0]], 2.0,
                                  output=out), 0)
        self.assertEqual(out, [[1.0, 4.0], [9.0, 16.0]])

    def test_clip(self):
        out = [[0.0, 0.0], [0.0, 0.0]]
        self.assertEqual(la.clip([[1.0, 5.0], [9.0, 2.0]], 2.0, 7.0,
                                 output=out), 0)
        self.assertEqual(out, [[2.0, 5.0], [7.0, 2.0]])

    def test_flatten(self):
        out = [0.0] * 6
        self.assertEqual(la.flatten([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                                    output=out), 0)
        self.assertEqual(out, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    def test_tensor_sum_mean(self):
        out = [0.0]
        self.assertEqual(la.tensor_sum([1.0, 2.0, 3.0], output=out), 0)
        self.assertAlmostEqual(out[0], 6.0)
        self.assertEqual(la.tensor_mean([1.0, 2.0, 3.0], output=out), 0)
        self.assertAlmostEqual(out[0], 2.0)

    def test_status_codes(self):
        self.assertEqual(la.dot([1.0], [1.0]), 2)          # no output
        self.assertEqual(la.add([1.0], [1.0, 2.0],
                                output=[0.0]), 1)          # mismatch
        self.assertEqual(la.dot(["x"], [1.0], output=[0.0]), 3)

    def test_duck_typing(self):
        out = DuckList([0.0, 0.0])
        self.assertEqual(la.add(DuckList([1.0, 2.0]),
                                DuckList([3.0, 4.0]), output=out), 0)
        self.assertEqual(out._items, [4.0, 6.0])
        sout = [0.0]
        self.assertEqual(la.norm(DuckList([3.0, 4.0]), output=sout), 0)
        self.assertAlmostEqual(sout[0], 5.0)

    def test_empty_tensor(self):
        out = [0.0]
        self.assertEqual(la.tensor_sum([], output=out), 0)
        self.assertEqual(out[0], 0.0)
        self.assertEqual(la.tensor_mean([], output=out), 0)
        self.assertEqual(out[0], 0.0)


class TestConsistency(unittest.TestCase):
    """Pure-Python and C-extension backends produce identical external
    behaviour (same input -> same result / status code).  Skipped when the
    C extension is not built (e.g. source-copy environments)."""

    @classmethod
    def setUpClass(cls):
        if _c is None:
            raise unittest.SkipTest("C extension not built")

    def _check(self, func, args, out_size, expect_values):
        results = []
        for mod in (_py, _c):
            if mod is None:
                continue
            if out_size == 1:
                out = [0.0]
            else:
                out = [[0.0] * out_size[1] for _ in range(out_size[0])]
            status = getattr(mod, func)(*args, output=out)
            results.append((status, out))
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[0][0], 0)
        self.assertEqual(results[0][1], expect_values)

    def test_consistent_basic(self):
        self._check("dot", ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]), 1, [14.0])
        self._check("norm", ([[1.0, 2.0], [3.0, 4.0]],), 1,
                    [5.477225575051661])
        self._check("add", ([[1.0, 2.0], [3.0, 4.0]],
                            [[5.0, 6.0], [7.0, 8.0]]), (2, 2),
                    [[6.0, 8.0], [10.0, 12.0]])
        self._check("multiply", ([[1.0, 2.0], [3.0, 4.0]],
                                 [[2.0, 2.0], [2.0, 2.0]]), (2, 2),
                    [[2.0, 4.0], [6.0, 8.0]])
        self._check("scale", ([[1.0, 2.0], [3.0, 4.0]], 3.0), (2, 2),
                    [[3.0, 6.0], [9.0, 12.0]])
        for mod in (_py, _c):
            if mod is None:
                continue
            out = [0.0, 0.0]
            self.assertEqual(mod.normalize([3.0, 4.0], output=out), 0)
            self.assertAlmostEqual(out[0], 0.6)
            self.assertAlmostEqual(out[1], 0.8)

    def test_consistent_status(self):
        for mod in (_py, _c):
            if mod is None:
                continue
            self.assertEqual(mod.dot([1.0], [1.0]), 2)
            self.assertEqual(mod.add([1.0], [1.0, 2.0], output=[0.0]), 1)
            self.assertEqual(mod.dot(["x"], [1.0], output=[0.0]), 3)


if __name__ == "__main__":
    unittest.main()
