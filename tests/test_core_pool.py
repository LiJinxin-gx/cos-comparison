# -*- coding: utf-8 -*-
"""Pooling tests (v0.5.1): pool window reduction, both backends.

Contract under test:
  * pool(data, window_size=..., func=..., step=..., start=..., end=...,
    output=..., iterate=...) reduces every valid window to one output
    element; func(*values) receives the window values in row-major order,
    func=None means the window mean; returns 0;
  * output is required with the window-count shape; errors raise
    TypeError/ValueError (elementwise style);
  * the iterate engine receives engine(kernel, data_shape=..., **params)
    and the kernel receives index + the remaining keywords;
  * pure Python and the C extension produce identical results.
"""
import unittest

from cos_comparison import core

import testutil


def _has_c_backend():
    try:
        import cos_comparison.core.cos_comparison_pydll  # noqa: F401
    except ImportError:  # pragma: no cover - compiled extension not built
        return False
    return True


_HAS_C = _has_c_backend()
MODES = ("py", "c") if _HAS_C else ("py",)
_ORIGINAL_BACKEND = None


def setUpModule():
    global _ORIGINAL_BACKEND
    _ORIGINAL_BACKEND = core.get_active_backend()


def tearDownModule():
    if _ORIGINAL_BACKEND is not None:
        core.set_mode(_ORIGINAL_BACKEND)


def shape_of(data):
    shape = []
    temp = data
    while isinstance(temp, (list, tuple)):
        shape.append(len(temp))
        if not temp:
            break
        temp = temp[0]
    return tuple(shape)


def odometer(dims):
    """Row-major index tuples (last dimension fastest)."""
    dims = list(dims)
    dimension = len(dims)
    index = [0] * dimension
    total = 1
    for n in dims:
        total *= n
    for _ in range(total):
        yield tuple(index)
        d = dimension - 1
        while d >= 0:
            index[d] += 1
            if index[d] < dims[d]:
                break
            index[d] = 0
            d -= 1


def nested(dims, fill=0.0):
    if len(dims) == 1:
        return [fill] * dims[0]
    return [nested(dims[1:], fill) for _ in range(dims[0])]


def flatten(data):
    flat = []
    stack = [data]
    while stack:
        item = stack.pop()
        if isinstance(item, (list, tuple)):
            for sub in reversed(item):
                stack.append(sub)
        else:
            flat.append(item)
    return flat


def ref_pool(data, window, start, step, func, end=None):
    """Reference pooling with explicit row-major window offsets."""
    dimension = len(window)
    shape = shape_of(data)
    if end is None:
        end = shape
    num = [(end[i] - start[i] - window[i]) // step[i] + 1
           for i in range(dimension)]
    values_out = []
    for index in odometer(num):
        values = []
        for offset in odometer(window):
            pos = tuple(start[i] + step[i] * index[i] + offset[i]
                        for i in range(dimension))
            temp = data
            for p in pos:
                temp = temp[p]
            values.append(temp)
        values_out.append(sum(values) / len(values) if func is None
                          else func(*values))
    return values_out


def reference_iterate(kernel, *, data_shape, **params):
    """Serial reference iterator (odometer over the window count)."""
    for index in odometer(list(data_shape)):
        kernel(index, **params)


def make_data(n=12):
    return [[float((i * 5 + j) % 11) * 0.25 for j in range(n)]
            for i in range(n)]


class TestPoolParity(unittest.TestCase):
    """pure Python vs C extension: identical results on both paths."""

    N = 12

    def setUp(self):
        self.data = make_data(self.N)

    def _run_all(self):
        data = self.data
        cases = [
            ((3, 3), (1, 1), (0, 0), (self.N, self.N), lambda *v: max(v)),
            ((3, 3), (1, 1), (0, 0), (self.N, self.N),
             lambda *v: sum(v) / len(v)),
            ((3, 3), (1, 1), (0, 0), (self.N, self.N), None),
            ((2, 3), (1, 2), (1, 0), (self.N - 1, self.N), None),
        ]
        n = self.N
        for window, step, start, end, func in cases:
            num = tuple((end[i] - start[i] - window[i]) // step[i] + 1
                        for i in range(2))
            out_a = nested(num)
            out_b = nested(num)
            r_a = core.pool(data, window_size=window, func=func, step=step,
                            start=start, end=end, output=out_a)
            r_b = core.pool(data, window_size=window, func=func, step=step,
                            start=start, end=end, output=out_b,
                            iterate=reference_iterate)
            want = ref_pool(data, window, start, step, func, end=end)
            self.assertEqual(r_a, 0)
            self.assertEqual(r_b, 0)
            flat_a = flatten(out_a)
            flat_b = flatten(out_b)
            self.assertEqual(len(flat_a), len(want))
            for got, expect in zip(flat_a, want):
                self.assertAlmostEqual(got, expect, places=9)
            for got, expect in zip(flat_b, want):
                self.assertAlmostEqual(got, expect, places=9)

    def test_pure_python(self):
        core.set_mode("py")
        self._run_all()

    @unittest.skipUnless(_HAS_C, "C extension not built")
    def test_c_backend(self):
        core.set_mode("c")
        self._run_all()

    def test_1d_and_3d(self):
        core.set_mode("py")
        data_1d = [float(i % 5) for i in range(10)]
        out_1d = nested([4])
        self.assertEqual(core.pool(data_1d, window_size=(3,), func=None,
                                   step=(2,), output=out_1d), 0)
        want = ref_pool(data_1d, (3,), (0,), (2,), None)
        for got, expect in zip(flatten(out_1d), want):
            self.assertAlmostEqual(got, expect, places=9)

        data_3d = [[[float(i + j + k) for k in range(4)]
                    for j in range(5)] for i in range(4)]
        window = (2, 2, 3)
        num = tuple(shape_of(data_3d)[i] - window[i] + 1
                    for i in range(3))
        out_a = nested(num)
        out_b = nested(num)
        core.pool(data_3d, window_size=window, output=out_a)
        if _HAS_C:
            core.set_mode("c")
            core.pool(data_3d, window_size=window, output=out_b)
        want = ref_pool(data_3d, window, (0, 0, 0), (1, 1, 1), None)
        for got, expect in zip(flatten(out_a), want):
            self.assertAlmostEqual(got, expect, places=9)
        if _HAS_C:
            self.assertEqual(flatten(out_a), flatten(out_b))

    def test_default_window_is_all_ones(self):
        core.set_mode("py")
        data = make_data(5)
        out = nested((5, 5))
        core.pool(data, output=out)
        self.assertEqual(flatten(out), flatten(data))


class TestPoolSlidingWindows(unittest.TestCase):
    """Sliding-window semantics: overlap, exact tiling and gaps."""

    DATA = [float(i) for i in range(10)]

    def test_1d_overlap_tile_gap(self):
        cases = [
            ((3,), (1,), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
            ((3,), (2,), [1.0, 3.0, 5.0, 7.0]),
            ((3,), (3,), [1.0, 4.0, 7.0]),          # exact tiling
            ((3,), (4,), [1.0, 5.0]),               # gapped
            ((3,), (5,), [1.0, 6.0]),               # gapped
            ((3,), (9,), [1.0]),
        ]
        for backend in MODES:
            core.set_mode(backend)
            for window, step, want in cases:
                out = [0.0] * len(want)
                status = core.pool(self.DATA, window_size=window,
                                   step=step, output=out)
                self.assertEqual(status, 0)
                for got, expect in zip(out, want):
                    self.assertAlmostEqual(got, expect, places=12)

    def test_trailing_window_dropped_with_end(self):
        core.set_mode("py")
        out = [0.0] * 2
        core.pool(self.DATA, window_size=(3,), step=(3,), end=(8,),
                  output=out)
        self.assertAlmostEqual(out[0], 1.0)
        self.assertAlmostEqual(out[1], 4.0)

    def test_2d_mixed_overlap_and_gap(self):
        grid = [[float(i * 6 + j) for j in range(6)] for i in range(7)]
        num = (2, 4)  # (7-2)//4+1, (6-3)//1+1
        want = []
        for i in range(num[0]):
            for j in range(num[1]):
                total = 0.0
                for a in range(2):
                    for b in range(3):
                        total += grid[i * 4 + a][j + b]
                want.append(total / 6.0)
        for backend in MODES:
            core.set_mode(backend)
            out = nested(num)
            core.pool(grid, window_size=(2, 3), step=(4, 1), output=out)
            for got, expect in zip(flatten(out), want):
                self.assertAlmostEqual(got, expect, places=12)

    def test_engine_path_gap(self):
        def reference_iterate(kernel, *, data_shape, **params):
            for index in odometer(list(data_shape)):
                kernel(index, **params)

        for backend in MODES:
            core.set_mode(backend)
            out = [0.0] * 2
            core.pool(self.DATA, window_size=(3,), step=(5,), output=out,
                      iterate=reference_iterate)
            self.assertAlmostEqual(out[0], 1.0)
            self.assertAlmostEqual(out[1], 6.0)


class TestPoolValidation(unittest.TestCase):
    def setUp(self):
        core.set_mode("py")
        self.data = make_data(6)
        self.out = nested((4, 4))

    def test_output_required(self):
        with self.assertRaises(ValueError):
            core.pool(self.data, window_size=(3, 3), func=None)

    def test_output_shape_mismatch(self):
        with self.assertRaises(ValueError):
            core.pool(self.data, window_size=(3, 3), func=None,
                      output=nested((5, 5)))

    def test_window_length_mismatch(self):
        with self.assertRaises(ValueError):
            core.pool(self.data, window_size=(3,), func=None,
                      output=self.out)

    def test_scalar_window_requires_1d(self):
        with self.assertRaises(ValueError):
            core.pool(self.data, window_size=3, func=None, output=self.out)
        out_1d = nested([2])
        self.assertEqual(core.pool([1.0, 2.0, 3.0], window_size=2,
                                   output=out_1d), 0)
        self.assertAlmostEqual(flatten(out_1d)[0], 1.5)
        self.assertAlmostEqual(flatten(out_1d)[1], 2.5)

    def test_effectless_args(self):
        with self.assertRaises(ValueError):
            core.pool(self.data, window_size=(9, 9), func=None,
                      output=self.out)

    def test_bad_step_and_start(self):
        with self.assertRaises(ValueError):
            core.pool(self.data, window_size=(3, 3), step=(1, 0),
                      func=None, output=self.out)
        with self.assertRaises(ValueError):
            core.pool(self.data, window_size=(3, 3), start=(-1, 0),
                      func=None, output=self.out)

    def test_func_must_be_callable(self):
        with self.assertRaises(TypeError):
            core.pool(self.data, window_size=(3, 3), func=42,
                      output=self.out)

    def test_func_must_return_number(self):
        with self.assertRaises(TypeError):
            core.pool(self.data, window_size=(3, 3), func=lambda *v: "x",
                      output=self.out)


class TestPoolContract(unittest.TestCase):
    def test_iterate_receives_kernel_and_params(self):
        seen = {}

        def spy_iterate(kernel, **index_info):
            seen.update(index_info)
            kernel((0, 0), **{k: v for k, v in index_info.items()
                              if k != "data_shape"})

        data = make_data(6)
        out = nested((4, 4))
        core.set_mode("py")
        core.pool(data, window_size=(3, 3), output=out,
                  iterate=spy_iterate)
        for key in ("data_shape", "data", "output", "window_start",
                    "window_step", "window_size", "func"):
            self.assertIn(key, seen)
        self.assertEqual(tuple(seen["data_shape"]), (4, 4))
        self.assertEqual(seen["window_start"], (0, 0))
        self.assertEqual(seen["window_step"], (1, 1))
        self.assertEqual(seen["window_size"], (3, 3))

    def test_c_backend_matches_py_reference(self):
        code, out, err = testutil.run_backend(".cos_comparison_pydll", (
            "from cos_comparison.core import cos_comparison as _py\n"
            "def odometer(dims):\n"
            "    dims = list(dims)\n"
            "    index = [0] * len(dims)\n"
            "    total = 1\n"
            "    for n in dims:\n"
            "        total *= n\n"
            "    result = []\n"
            "    for _ in range(total):\n"
            "        result.append(tuple(index))\n"
            "        d = len(dims) - 1\n"
            "        while d >= 0:\n"
            "            index[d] += 1\n"
            "            if index[d] < dims[d]:\n"
            "                break\n"
            "            index[d] = 0\n"
            "            d -= 1\n"
            "    return result\n"
            "def reference_iterate(kernel, *, data_shape, **params):\n"
            "    for index in odometer(data_shape):\n"
            "        kernel(index, **params)\n"
            "data = [[float((i * 5 + j) % 11) * 0.25 for j in range(9)]\n"
            "        for i in range(9)]\n"
            "funcs = [None, lambda *v: max(v), lambda *v: sum(v) / len(v)]\n"
            "same = True\n"
            "for func in funcs:\n"
            "    a = [[0.0] * 7 for _ in range(7)]\n"
            "    b = [[0.0] * 7 for _ in range(7)]\n"
            "    c = [[0.0] * 7 for _ in range(7)]\n"
            "    ra = core.pool(data, window_size=(3, 3), func=func, output=a)\n"
            "    core.pool(data, window_size=(3, 3), func=func, output=b,\n"
            "              iterate=reference_iterate)\n"
            "    _py.pool(data, window_size=(3, 3), func=func, output=c)\n"
            "    same = same and (a == b == c) and ra == 0\n"
            "print(json.dumps({'same': bool(same)}))\n"))
        self.assertEqual(code, 0, err[-500:])
        self.assertEqual(testutil.json_result(out), {"same": True})


if __name__ == "__main__":
    unittest.main()
