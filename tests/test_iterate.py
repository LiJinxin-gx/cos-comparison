"""Iterate-parameter tests: the optional custom-iterator path of the
element-wise A-class functions (data_mapping / data_filter / elementwise /
position_map / elementwise_position and the threshold_* combinations).

Contract under test:
  * ``iterate(kernel, **index_info)`` - the iterator resolves the position
    parameters (data_shape/start/shape/step) and hands the index (positional)
    plus the remaining keyword parameters to ``kernel(index, **params)``;
  * with iterate=None (default) the original inline skeleton runs - results
    must be bit-identical to the iterator path;
  * both backends (pure Python and the C extension) honour the contract.
"""
import unittest

from cos_comparison import core
from cos_comparison.core import cos_comparison as _py

import testutil


def reference_iterate(kernel, *, data_shape, start=None, shape=None,
                      step=None, **params):
    """Serial reference iterator: resolve the position parameters and
    dispatch index + keyword parameters to the kernel."""
    effective, _r_start, _r_step = _py._region_spec_by_shape(
        data_shape, start, shape, step)
    for index in _py._region_walk(effective):
        kernel(index, **params)


def make_data(n=24):
    return [[float((i * 7 + j) % 13) * 0.5 for j in range(n)]
            for i in range(n)]


class TestIterateParity(unittest.TestCase):
    """Default (inline skeleton) vs iterate path: identical results."""

    N = 24

    def setUp(self):
        self.data = make_data(self.N)

    def _fresh_out(self):
        return [[0.0] * self.N for _ in range(self.N)]

    def test_data_mapping(self):
        a = self._fresh_out()
        b = self._fresh_out()
        core.data_mapping(self.data, lambda v: v * 2.0 + 1.0, out=a)
        core.data_mapping(self.data, lambda v: v * 2.0 + 1.0, out=b,
                          iterate=reference_iterate)
        self.assertEqual(a, b)

    def test_data_filter(self):
        a = list(core.data_filter(self.data, lambda v: v > 3.0))
        b = list(core.data_filter(self.data, lambda v: v > 3.0,
                                  iterate=reference_iterate))
        self.assertEqual(a, b)
        self.assertGreater(len(a), 0)

    def test_elementwise(self):
        a = self._fresh_out()
        b = self._fresh_out()
        core.elementwise(self.data, self.data, func=lambda x, y: x + y,
                         output=a)
        core.elementwise(self.data, self.data, func=lambda x, y: x + y,
                         output=b, iterate=reference_iterate)
        self.assertEqual(a, b)

    def test_position_map(self):
        a = self._fresh_out()
        b = self._fresh_out()
        ra = core.position_map(a, lambda pos: float(pos[0] + pos[1]))
        rb = core.position_map(b, lambda pos: float(pos[0] + pos[1]),
                               iterate=reference_iterate)
        self.assertEqual(a, b)
        self.assertEqual(ra, rb)

    def test_elementwise_position(self):
        a = self._fresh_out()
        b = self._fresh_out()
        ra = core.elementwise_position(
            a, self.data, self.data,
            callback=lambda vals, pos: vals[0] + vals[1])
        rb = core.elementwise_position(
            b, self.data, self.data,
            callback=lambda vals, pos: vals[0] + vals[1],
            iterate=reference_iterate)
        self.assertEqual(a, b)
        self.assertEqual(ra, rb)

    def test_threshold_combinations(self):
        a = self._fresh_out()
        b = self._fresh_out()
        core.threshold_map(self.data, [(lambda v: v > 3.0, 1.0)],
                           default_value=0.0, out=a)
        core.threshold_map(self.data, [(lambda v: v > 3.0, 1.0)],
                           default_value=0.0, out=b,
                           iterate=reference_iterate)
        self.assertEqual(a, b)
        self.assertEqual(
            list(core.threshold_filter(self.data, 2.0, 4.0)),
            list(core.threshold_filter(self.data, 2.0, 4.0,
                                       iterate=reference_iterate)))


class TestIterateContract(unittest.TestCase):
    """The iterator receives the position parameters as keywords and the
    kernel receives index (positional) + the remaining keywords."""

    def test_index_info_keys(self):
        seen = {}

        def spy_iterate(kernel, **index_info):
            seen.update(index_info)

        data = make_data(6)
        core.data_mapping(data, lambda v: v, iterate=spy_iterate)
        self.assertIn("data_shape", seen)
        self.assertIn("start", seen)
        self.assertIn("shape", seen)
        self.assertIn("step", seen)
        self.assertIn("data", seen)
        self.assertIn("callback", seen)
        self.assertIn("out", seen)
        self.assertEqual(tuple(seen["data_shape"]), (6, 6))

    def test_kernel_call_shape(self):
        calls = []

        def spy_iterate(kernel, **index_info):
            # call the kernel exactly as the contract prescribes
            params = {k: v for k, v in index_info.items()
                      if k not in ("data_shape", "start", "shape", "step")}
            kernel((0, 0), **params)
            calls.append(1)

        data = make_data(4)
        out = [[0.0] * 4 for _ in range(4)]
        core.data_mapping(data, lambda v: v + 1.0, out=out,
                          iterate=spy_iterate)
        self.assertEqual(len(calls), 1)
        self.assertEqual(out[0][0], data[0][0] + 1.0)

    def test_elementwise_kernel_signature(self):
        seen = {}

        def spy_iterate(kernel, **index_info):
            seen.update(index_info)

        a = make_data(4)
        b = [[0.0] * 4 for _ in range(4)]
        core.elementwise(a, a, func=lambda x, y: x + y, output=b,
                         iterate=spy_iterate)
        self.assertIn("tensors", seen)
        self.assertIn("func", seen)
        self.assertIn("output", seen)
        self.assertIn("data_shape", seen)


class TestIterateBackends(unittest.TestCase):
    """The C extension honours the same iterate contract."""

    def test_c_backend_iterate(self):
        code, out, err = testutil.run_backend(".cos_comparison_pydll", (
            "from cos_comparison.core import cos_comparison as _py\n"
            "def reference_iterate(kernel, *, data_shape, start=None,\n"
            "                      shape=None, step=None, **params):\n"
            "    effective, _rs, _rt = _py._region_spec_by_shape(\n"
            "        data_shape, start, shape, step)\n"
            "    for index in _py._region_walk(effective):\n"
            "        kernel(index, **params)\n"
            "data = [[float((i * 7 + j) % 13) * 0.5 for j in range(8)]\n"
            "        for i in range(8)]\n"
            "out_a = [[0.0] * 8 for _ in range(8)]\n"
            "out_b = [[0.0] * 8 for _ in range(8)]\n"
            "core.data_mapping(data, lambda v: v * 2.0 + 1.0, out=out_a)\n"
            "core.data_mapping(data, lambda v: v * 2.0 + 1.0, out=out_b,\n"
            "                  iterate=reference_iterate)\n"
            "h_a = list(core.data_filter(data, lambda v: v > 3.0))\n"
            "h_b = list(core.data_filter(data, lambda v: v > 3.0,\n"
            "                            iterate=reference_iterate))\n"
            "print(json.dumps({'same': out_a == out_b, 'hits': h_a == h_b}))\n"))
        self.assertEqual(code, 0, err[-500:])
        self.assertEqual(testutil.json_result(out),
                         {"same": True, "hits": True})


class TestIterateBClass(unittest.TestCase):
    """B-class (passive / active / mean_local / local_variance) iterate
    path: default skeleton vs iterator injection, on both backends."""

    N = 16

    def setUp(self):
        self.data = make_data(self.N)
        self.kernel = [[1.0, 0.5], [0.5, 1.0]]

    @staticmethod
    def _flat(t):
        return [t[i][j] for i in range(len(t)) for j in range(len(t[i]))]

    def _run_all(self):
        data = self.data
        a = core.cos_comparison_passive(data, window_size=(2, 2), d=(1, 1))
        b = core.cos_comparison_passive(data, window_size=(2, 2), d=(1, 1),
                                        iterate=reference_iterate)
        self.assertEqual(self._flat(a), self._flat(b))

        a2 = core.cos_comparison_active(data, kernel=self.kernel)
        b2 = core.cos_comparison_active(data, kernel=self.kernel,
                                        iterate=reference_iterate)
        self.assertEqual(self._flat(a2), self._flat(b2))

        m1 = core.mean_local(data, local_size=(2, 2))
        m2 = core.mean_local(data, local_size=(2, 2),
                             iterate=reference_iterate)
        self.assertEqual(self._flat(m1), self._flat(m2))

        v1 = core.local_variance(data, local_size=(2, 2))
        v2 = core.local_variance(data, local_size=(2, 2),
                                 iterate=reference_iterate)
        self.assertEqual(self._flat(v1), self._flat(v2))

    def test_pure_python(self):
        core.set_mode("py")
        self._run_all()

    def test_c_backend(self):
        core.set_mode("c")
        self._run_all()

    def test_outer_callbacks_order(self):
        events = []
        core.set_mode("py")
        core.cos_comparison_passive(
            self.data, window_size=(2, 2), d=(1, 1),
            iterate=reference_iterate,
            start_callback=lambda n: events.append("start"),
            end_callback=lambda n: events.append("end"),
            return_callback=lambda o, n: (events.append("return"), o)[1])
        self.assertEqual(events, ["start", "end", "return"])

    def test_legacy_kwargs_accepted(self):
        # w1/w2/b1/b2 are retired but must not raise (absorbed silently)
        for backend in ("py", "c"):
            core.set_mode(backend)
            core.cos_comparison_passive(self.data, window_size=(2, 2),
                                        d=(1, 1), w1=2.0, w2=1.0,
                                        b1=0.5, b2=0.0)


class TestTransformBClass(unittest.TestCase):
    """transform1/transform2 replace the retired linear transform
    (w1/w2/b1/b2): callable per-value maps applied to the two comparison
    windows; default None = identity (bit-identical to no transform)."""

    N = 8

    def setUp(self):
        self.data = [[float((i * 7 + j * 3) % 11) * 0.5 - 2.0
                      for j in range(self.N)] for i in range(self.N)]
        self.kernel = [[0.25, -0.5], [1.0, 0.75]]
        self.algo = _py._default_algorithm
        self.t1 = lambda v: v * 2.0
        self.t2 = lambda v: v + 1.0

    @staticmethod
    def _flat(t):
        return [t[i][j] for i in range(len(t)) for j in range(len(t[i]))]

    def _ref_passive(self, t1, t2):
        data, ws, d = self.data, (2, 2), (1, 0)
        rows = self.N - ws[0] - d[0] + 1
        cols = self.N - ws[1] - d[1] + 1
        out = [[0.0] * cols for _ in range(rows)]
        for oi in range(rows):
            for oj in range(cols):
                main = other = mu = 0.0
                for i in range(ws[0]):
                    for j in range(ws[1]):
                        a = data[oi + i][oj + j]
                        b = data[oi + i + d[0]][oj + j + d[1]]
                        if t1 is not None:
                            a = t1(a)
                        if t2 is not None:
                            b = t2(b)
                        main += a * a
                        other += b * b
                        mu += a * b
                out[oi][oj] = self.algo(main, other, mu, None)
        return out

    def _ref_active(self, t1, t2):
        data, kernel = self.data, self.kernel
        ws = (len(kernel), len(kernel[0]))
        out = [[0.0] * (self.N - ws[1] + 1) for _ in range(self.N - ws[0] + 1)]
        for oi in range(len(out)):
            for oj in range(len(out[0])):
                main = other = mu = 0.0
                for i in range(ws[0]):
                    for j in range(ws[1]):
                        a = data[oi + i][oj + j]
                        b = kernel[i][j]
                        if t1 is not None:
                            a = t1(a)
                        if t2 is not None:
                            b = t2(b)
                        main += a * a
                        other += b * b
                        mu += a * b
                out[oi][oj] = self.algo(main, other, mu, None)
        return out

    def _assert_close(self, expected, got):
        pairs = zip(self._flat(expected), self._flat(got))
        self.assertTrue(max(abs(x - y) for x, y in pairs) < 1e-12)

    def _check_backend(self):
        got = core.cos_comparison_passive(self.data, window_size=(2, 2),
                                          transform1=self.t1, transform2=self.t2)
        self._assert_close(self._ref_passive(self.t1, self.t2), got)
        got = core.cos_comparison_active(self.data, kernel=self.kernel,
                                         transform1=self.t1, transform2=self.t2)
        self._assert_close(self._ref_active(self.t1, self.t2), got)

    def test_transform_reference_pure_python(self):
        core.set_mode("py")
        self._check_backend()

    def test_transform_reference_c_backend(self):
        core.set_mode("c")
        self._check_backend()

    def test_identity_is_default(self):
        ident = lambda v: v
        for backend in ("py", "c"):
            core.set_mode(backend)
            base = core.cos_comparison_passive(self.data, window_size=(2, 2))
            same = core.cos_comparison_passive(self.data, window_size=(2, 2),
                                               transform1=ident, transform2=ident)
            self.assertEqual(self._flat(base), self._flat(same))

    def test_inherited_by_wrappers(self):
        for backend in ("py", "c"):
            core.set_mode(backend)
            out = core.mean_local(self.data, local_size=(2, 2),
                                  transform1=lambda v: v * v)
            self.assertEqual(len(out), self.N - 1)
            var = core.local_variance(self.data, local_size=(2, 2),
                                      transform1=lambda v: v * v)
            self.assertEqual(len(var), self.N - 1)


if __name__ == "__main__":
    unittest.main()
