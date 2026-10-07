"""Core algorithms: cos_comparison_passive / active / cos / mean_local /
local_variance, including a hand-computed formula check of _cosmod."""

import unittest

from cos_comparison import core


def block(size, value=1.0):
    """2D tensor with the top-left square (value) on a zero background."""
    v = core.create_void_list(size)
    for i in range(min(2, size[0])):
        for j in range(min(2, size[1])):
            v[i, j] = value
    return v


def mod_2(A, B):
    """Window cosmod through the real kernel: the vector norms and dot
    product are hand-computed here, then combined by ``core._cosmod``."""
    dot = sum(a * b for a, b in zip(A, B))
    n2a = sum(a * a for a in A)
    n2b = sum(b * b for b in B)
    return core._cosmod(n2a, n2b, dot, "t")


class TestCosmodFormula(unittest.TestCase):
    """core._cosmod is the scalar combination kernel used inside the
    window loops: cosmod(a, b, ab, name) = 2*ab / (a+b), where a and b
    are the two window norms and ab their dot product; a == b == 0
    yields 1.0.  (This differs from the README's vector formula
    cosmod = 2(A.B)/(|A|^2+|B|^2).)"""

    def test_identical(self):
        self.assertEqual(mod_2([1.0, 1.0], [1.0, 1.0]), 1.0)

    def test_opposite(self):
        self.assertEqual(mod_2([1.0, 0.0], [0.0, 1.0]), 0.0)

    def test_half(self):
        self.assertAlmostEqual(mod_2([2.0, 0.0], [1.0, 1.0]), 2.0 / 3.0)

    def test_negative_agreement(self):
        self.assertEqual(mod_2([-1.0, -1.0], [-1.0, -1.0]), 1.0)

    def test_zero_vector(self):
        self.assertEqual(mod_2([0.0, 0.0], [1.0, 1.0]), 0.0)

    def test_private_cosmod_scalar(self):
        # 2*ab/(a+b) = 2*2/(2+1) = 4/3
        self.assertAlmostEqual(core._cosmod(2.0, 1.0, 2.0, "t"), 4.0 / 3.0)
        self.assertAlmostEqual(core._cosmod(0.0, 0.0, 0.0, "t"), 1.0)


class TestPassive(unittest.TestCase):
    def test_output_shape(self):
        v = block((5, 5))
        r = core.cos_comparison_passive(v, window_size=(3, 3), d=(0, 1))
        # (N - window - d + 1) per axis: (5-3-0+1, 5-3-1+1) -> (3, 2)
        self.assertEqual(r.shape, (3, 2))

    def test_edge_response(self):
        # horizontal edge: top half bright, bottom half dark; passive
        # emits a *similarity* map, so windows crossing the edge score
        # below the flat-region score of 1.0
        v = core.create_void_list((10, 10))
        for i in range(5):
            for j in range(10):
                v[i, j] = 1.0
        r = core.cos_comparison_passive(v, window_size=(3, 3), d=(1, 0))
        values = [float(r[i, j]) for i in range(r.shape[0])
                  for j in range(r.shape[1])]
        self.assertEqual(max(values), 1.0)          # flat region
        self.assertLess(min(values), 1.0)           # edge dip

    def test_flat_region_ones(self):
        # a fully uniform input is everywhere similar to itself -> 1.0
        v = core.create_void_list((6, 6), default=1.0)
        r = core.cos_comparison_passive(v, window_size=(3, 3), d=(1, 0))
        for i in range(r.shape[0]):
            for j in range(r.shape[1]):
                self.assertAlmostEqual(r[i, j], 1.0, places=6)

    def test_window_size_default(self):
        v = block((4, 4))
        r = core.cos_comparison_passive(v)
        self.assertEqual(r.shape, (3, 4))  # window (1,1), d (1,0)

    def test_1d(self):
        v = core.create_void_list((8,))
        v[0], v[1], v[2] = 1.0, 1.0, 1.0
        r = core.cos_comparison_passive(v, window_size=(3,))
        # (8 - 3 - 1 + 1) = 5 outputs along the default d=(1,0) axis
        self.assertEqual(r.shape, (5,))

    def test_return_callback(self):
        v = block((5, 5))
        calls = []

        def record(out, name):
            calls.append((out, name))
            return out

        result = core.cos_comparison_passive(
            v, window_size=(3, 3), return_callback=record)
        self.assertEqual(len(calls), 1)
        out, name = calls[0]
        self.assertTrue(hasattr(out, "shape"))
        self.assertIs(result, out)


class TestActive(unittest.TestCase):
    def test_kernel_required(self):
        # both backends reject a missing kernel, with different
        # exception types (pure Python: ValueError, pydll: TypeError)
        v = block((5, 5))
        with self.assertRaises((ValueError, TypeError)):
            core.cos_comparison_active(v)

    def test_output_shape(self):
        v = block((5, 5))
        k = core.create_void_list((2, 2), default=1.0)
        r = core.cos_comparison_active(v, kernel=k)
        self.assertEqual(r.shape, (4, 4))

    def test_match_response(self):
        v = core.create_void_list((5, 5))
        for i in range(2):
            for j in range(2):
                v[i, j] = 1.0
        k = core.create_void_list((2, 2), default=1.0)
        r = core.cos_comparison_active(v, kernel=k)
        best = max(r[i, j] for i in range(r.shape[0])
                   for j in range(r.shape[1]))
        self.assertGreater(best, 0.9)  # exact match -> cosmod 1.0


class TestCos(unittest.TestCase):
    def test_whole_tensor_similarity(self):
        a = core.create_void_list((2, 2), default=1.0)
        b = core.create_void_list((2, 2), default=1.0)
        self.assertAlmostEqual(core.cos(a, b), 1.0)

    def test_orthogonal(self):
        a = core.create_void_list((2, 2))
        b = core.create_void_list((2, 2))
        a[0, 0], a[1, 1] = 1.0, 1.0
        b[0, 1], b[1, 0] = 1.0, 1.0
        self.assertAlmostEqual(core.cos(a, b), 0.0, places=9)

    def test_cosine_similarity_value(self):
        # cos() is the plain cosine similarity, not cosmod:
        # A=[2,0], B=[1,1] -> 2/(2*sqrt(2)) = 1/sqrt(2)
        a = core.create_void_list((2, 1))
        b = core.create_void_list((2, 1))
        a[0, 0], a[1, 0] = 2.0, 0.0
        b[0, 0], b[1, 0] = 1.0, 1.0
        self.assertAlmostEqual(float(core.cos(a, b)), 1.0 / 2.0 ** 0.5,
                               places=9)


class TestMeanAndVariance(unittest.TestCase):
    def setUp(self):
        self.v = core.create_void_list((6, 6))
        for i in range(6):
            for j in range(6):
                self.v[i, j] = float(i)

    def test_mean_local(self):
        # window (2,2) slides over 6x6 -> (5,5); row mean at (3,0):
        # rows 3,4 in the window -> (3+4)/2 = 3.5
        r = core.mean_local(self.v, local_size=(2, 2))
        self.assertEqual(r.shape, (5, 5))
        self.assertAlmostEqual(r[3, 0], 3.5)

    def test_local_variance(self):
        # rows 3,4 with columns 0,1: values 3,3,4,4 -> variance 0.25
        r = core.local_variance(self.v, local_size=(2, 2))
        self.assertEqual(r.shape, (5, 5))
        self.assertAlmostEqual(r[3, 0], 0.25)

    def test_mean_flat(self):
        v = core.create_void_list((4,), default=2.0)
        r = core.mean_local(v, local_size=(2,))
        self.assertEqual(r.shape, (3,))
        self.assertAlmostEqual(r[0], 2.0)


class TestThresholdMap(unittest.TestCase):
    """threshold_map: iterate (func, value) pairs, first truthy func(value)
    selects the paired value; else default_value (default 0.0)."""

    def test_first_truthy_mapping(self):
        pairs = [(lambda v: v < 3, -1.0), (lambda v: v < 8, 0.0),
                 (lambda v: True, 1.0)]
        out = core.threshold_map([[1.0, 5.0], [9.0, 3.0]], pairs)
        self.assertEqual(core.get_item(out, (0, 0)), -1.0)
        self.assertEqual(core.get_item(out, (0, 1)), 0.0)
        self.assertEqual(core.get_item(out, (1, 0)), 1.0)
        self.assertEqual(core.get_item(out, (1, 1)), 0.0)

    def test_default_value(self):
        out = core.threshold_map([[1.0, 2.0]],
                                 [(lambda v: v > 100, 9.0)])
        self.assertEqual(core.get_item(out, (0, 0)), 0.0)
        out2 = core.threshold_map([[1.0, 2.0]],
                                  [(lambda v: v > 100, 9.0)],
                                  default_value=5.0)
        self.assertEqual(core.get_item(out2, (0, 1)), 5.0)

    def test_region_params_kept(self):
        data = [[1.0, 5.0, 9.0], [3.0, 7.0, 11.0]]
        out = core.threshold_map(data, [(lambda v: v >= 7, 1.0)],
                                 start=(1, 0), shape=(1, 3), step=(1, 1),
                                 default_value=-1.0)
        vals = [core.get_item(out, i) for i in [(0, 0), (0, 1), (0, 2)]]
        self.assertEqual(vals, [-1.0, 1.0, 1.0])

    def test_error_skipped(self):
        def bad(value):
            raise RuntimeError("boom")

        data = [[1.0, 5.0]]
        out = core.threshold_map(data, [(bad, 1.0), (lambda v: v > 4, 2.0)],
                                 default_value=-1.0)
        self.assertEqual(core.get_item(out, (0, 0)), -1.0)
        self.assertEqual(core.get_item(out, (0, 1)), 2.0)


class TestThresholdJudge(unittest.TestCase):
    """threshold_judge: judge-function factory returning 1 (in range) /
    0 (out of range); pairs with threshold_map's (func, value) iteration."""

    def test_judge_value(self):
        judge = core.threshold_judge(low=3, high=7)
        self.assertEqual(judge(5.0), 1)
        self.assertEqual(judge(1.0), 0)
        self.assertEqual(judge(3.0), 1)

    def test_single_bound_and_open(self):
        judge_lo = core.threshold_judge(low=5)
        self.assertEqual(judge_lo(6.0), 1)
        self.assertEqual(judge_lo(4.0), 0)
        judge_open = core.threshold_judge(low=3, high=7,
                                          inclusive=(False, True))
        self.assertEqual(judge_open(3.0), 0)

    def test_with_threshold_map(self):
        data = [[1.0, 5.0, 9.0], [3.0, 7.0, 11.0]]
        pairs = [(core.threshold_judge(low=3, high=7), 2.0),
                 (lambda v: True, 3.0)]
        out = core.threshold_map(data, pairs)
        vals = [core.get_item(out, i) for i in [(0, 0), (0, 1), (0, 2), (1, 0)]]
        self.assertEqual(vals, [3.0, 2.0, 3.0, 2.0])

    def test_no_bounds_raises(self):
        with self.assertRaises(ValueError):
            core.threshold_judge()


class TestElementwise(unittest.TestCase):
    """elementwise(*tensors, func=f, output=o) -> 0: strong protocol /
    duck typing, positional-only func call, integrated get_item/set_item,
    iterative, strict same shape, numeric result enforced."""

    def test_unary_returns_zero(self):
        a = core.vector_map_as_tensor(vector=[1.0, 2.0, 3.0], shape=(3,))
        out = core.vector_map_as_tensor(vector=[0.0, 0.0, 0.0], shape=(3,))
        rc = core.elementwise(a, func=lambda x: x * 2, output=out)
        self.assertEqual(rc, 0)
        self.assertEqual(list(out.vector), [2.0, 4.0, 6.0])

    def test_binary(self):
        a = core.vector_map_as_tensor(vector=[1.0, 2.0], shape=(2,))
        b = core.vector_map_as_tensor(vector=[10.0, 20.0], shape=(2,))
        out = core.vector_map_as_tensor(vector=[0.0, 0.0], shape=(2,))
        self.assertEqual(
            core.elementwise(a, b, func=lambda x, y: x + y, output=out), 0)
        self.assertEqual(list(out.vector), [11.0, 22.0])

    def test_ternary(self):
        a = core.vector_map_as_tensor(vector=[1.0, 2.0], shape=(2,))
        b = core.vector_map_as_tensor(vector=[10.0, 20.0], shape=(2,))
        c = core.vector_map_as_tensor(vector=[100.0, 200.0], shape=(2,))
        out = core.vector_map_as_tensor(vector=[0.0, 0.0], shape=(2,))
        self.assertEqual(core.elementwise(
            a, b, c, func=lambda x, y, z: x + y + z, output=out), 0)
        self.assertEqual(list(out.vector), [111.0, 222.0])

    def test_list_duck_input(self):
        out = [0.0, 0.0, 0.0]
        rc = core.elementwise([1.0, 2.0, 3.0],
                              func=lambda x: x + 1, output=out)
        self.assertEqual(rc, 0)
        self.assertEqual(out, [2.0, 3.0, 4.0])

    def test_2d(self):
        a = core.vector_map_as_tensor(
            vector=[1.0, 2.0, 3.0, 4.0], shape=(2, 2))
        out = core.vector_map_as_tensor(
            vector=[0.0, 0.0, 0.0, 0.0], shape=(2, 2))
        self.assertEqual(core.elementwise(
            a, func=lambda x: x * 10, output=out), 0)
        self.assertEqual(list(out.vector), [10.0, 20.0, 30.0, 40.0])

    def test_empty_tensor(self):
        a = core.create_void_list((0,))
        out = core.create_void_list((0,))
        self.assertEqual(core.elementwise(
            a, func=lambda x: x, output=out), 0)

    def test_shape_mismatch(self):
        a = core.vector_map_as_tensor(vector=[1.0, 2.0], shape=(2,))
        b = core.vector_map_as_tensor(vector=[1.0], shape=(1,))
        out = core.vector_map_as_tensor(vector=[0.0, 0.0], shape=(2,))
        with self.assertRaises(ValueError):
            core.elementwise(a, b, func=lambda x, y: x + y, output=out)

    def test_out_shape_mismatch(self):
        a = core.vector_map_as_tensor(vector=[1.0, 2.0], shape=(2,))
        out = core.vector_map_as_tensor(vector=[0.0], shape=(1,))
        with self.assertRaises(ValueError):
            core.elementwise(a, func=lambda x: x, output=out)

    def test_func_error_propagates(self):
        a = core.vector_map_as_tensor(vector=[1.0], shape=(1,))
        out = core.vector_map_as_tensor(vector=[0.0], shape=(1,))
        with self.assertRaises(ZeroDivisionError):
            core.elementwise(a, func=lambda x: 1 / 0, output=out)

    def test_non_numeric_result(self):
        a = core.vector_map_as_tensor(vector=[1.0], shape=(1,))
        out = core.vector_map_as_tensor(vector=[0.0], shape=(1,))
        with self.assertRaises(TypeError):
            core.elementwise(a, func=lambda x: "s", output=out)

    def test_missing_args(self):
        a = core.vector_map_as_tensor(vector=[1.0], shape=(1,))
        out = core.vector_map_as_tensor(vector=[0.0], shape=(1,))
        with self.assertRaises(TypeError):
            core.elementwise(a, output=out)
        with self.assertRaises(ValueError):
            core.elementwise(a, func=lambda x: x)
        with self.assertRaises(TypeError):
            core.elementwise(func=lambda x: x, output=out)


class TestPositionCallbacks(unittest.TestCase):
    """position_map / elementwise_position: position-driven element
    callbacks - real traversal, logical callback coordinates (origin /
    per-dimension scale), duck write via set_item."""

    def test_position_map_basic(self):
        from cos_comparison.core.cos_comparison import position_map
        out = [0.0] * 5
        self.assertEqual(position_map(out, lambda pos: pos[0] * 10), 0)
        self.assertEqual(out, [0.0, 10.0, 20.0, 30.0, 40.0])

    def test_position_map_2d(self):
        from cos_comparison.core.cos_comparison import position_map
        out = [[0.0] * 2, [0.0] * 2]
        self.assertEqual(position_map(
            out, lambda pos: pos[0] + pos[1] * 0.1), 0)
        self.assertEqual(out, [[0.0, 0.1], [1.0, 1.1]])

    def test_position_map_region(self):
        from cos_comparison.core.cos_comparison import position_map
        out = [0.0] * 4
        self.assertEqual(position_map(out, lambda pos: 1.0,
                                      start=(1,), shape=(2,), step=(1,)), 0)
        self.assertEqual(out, [0.0, 1.0, 1.0, 0.0])

    def test_position_map_logical_coordinates(self):
        from cos_comparison.core.cos_comparison import position_map
        got = []
        position_map([0.0] * 4, lambda pos: got.append(pos) or 0.0,
                     origin=(10,), scale=(2,))
        self.assertEqual(got, [(-10,), (-8,), (-6,), (-4,)])

    def test_position_map_callback_error_skipped(self):
        from cos_comparison.core.cos_comparison import position_map
        out = [0.0] * 3

        def bad(pos):
            if pos[0] == 1:
                raise ValueError("boom")
            return pos[0]

        self.assertEqual(position_map(out, bad), 0)
        self.assertEqual(out, [0.0, 0.0, 2.0])

    def test_elementwise_position_basic(self):
        from cos_comparison.core.cos_comparison import elementwise_position
        out = [0.0] * 3
        self.assertEqual(elementwise_position(
            out, [1, 2, 3], [10, 20, 30],
            callback=lambda els, pos: sum(els)), 0)
        self.assertEqual(out, [11, 22, 33])

    def test_elementwise_position_self_writeback(self):
        from cos_comparison.core.cos_comparison import elementwise_position
        out = [0.0] * 3
        self.assertEqual(elementwise_position(
            out, [1, 2, 3], callback=lambda els, pos: els[0] * 2), 0)
        self.assertEqual(out, [2, 4, 6])


    def test_position_map_integer_logical(self):
        from cos_comparison.core.cos_comparison import position_map
        got = []
        position_map([0.0] * 3, lambda pos: got.append(pos) or 0.0,
                     origin=(2,), scale=(1,))
        self.assertEqual(got, [(-2,), (-1,), (0,)])
        self.assertTrue(all(type(p[0]) is int for p in got))

    def test_fractional_scale(self):
        from cos_comparison.core.cos_comparison import position_map
        got = []
        position_map([0.0] * 3, lambda pos: got.append(pos) or 0.0,
                     origin=(10,), scale=(0.5,))
        self.assertEqual(got, [(-10,), (-9.5,), (-9,)])
        self.assertEqual([type(p[0]) for p in got],
                         [int, float, int])  # integral values stay int

    def test_zero_scale_valid(self):
        from cos_comparison.core.cos_comparison import position_map
        got = []
        out = [0.0] * 2
        self.assertEqual(position_map(
            out, lambda pos: got.append(pos) or 0.0,
            origin=(3,), scale=(0,)), 0)
        self.assertEqual(got, [(-3,), (-3,)])


class TestAcrossBackendsPosition(unittest.TestCase):
    """Position callbacks consistent across every available backend."""

    def test_position_consistent_backends(self):
        import cos_comparison.core.cos_comparison as _py
        from cos_comparison import core
        old = core.get_active_backend()
        if old is not None:
            self.addCleanup(core.set_mode, old)
        checked = 0
        for name in core.get_available_backends():
            try:
                core.set_mode([name])
            except ImportError:
                continue  # optional compiled backend not built
            checked += 1
            op = [0.0] * 4
            st = core.position_map(op, lambda pos: pos[0] * 3)
            self.assertEqual(st, 0)
            ep = [0.0] * 3
            est = core.elementwise_position(
                ep, [1, 2, 3], callback=lambda e, p: e[0] + 1)
            self.assertEqual(est, 0)
            pout = [0.0] * 4
            _py.position_map(pout, lambda pos: pos[0] * 3)
            self.assertEqual(op, pout)
        self.assertGreaterEqual(checked, 1)


class TestReloadHooks(unittest.TestCase):
    """The bound __cos_comparison_passive__ / __cos_comparison_active__
    reload hooks receive the call's extra arguments (data is the bound
    self) and may delegate back; the result must match the direct call."""

    class PassiveHook:
        def __init__(self, base, call=None):
            self.base = base
            self.call = call if call is not None else core.cos_comparison_passive

        def __cos_comparison_passive__(self, *args, **kwargs):
            return self.call(self.base, *args, **kwargs)

    class ActiveHook:
        def __init__(self, base, call=None):
            self.base = base
            self.call = call if call is not None else core.cos_comparison_active

        def __cos_comparison_active__(self, *args, **kwargs):
            return self.call(self.base, *args, **kwargs)

    def test_passive_hook_delegates(self):
        base = [1.0, 2.0, 3.0, 4.0]
        direct = core.cos_comparison_passive(base, window_size=(2,), d=(0,))
        via = core.cos_comparison_passive(self.PassiveHook(base),
                                          window_size=(2,), d=(0,))
        self.assertEqual(list(direct), list(via))

    def test_active_hook_delegates(self):
        base = [1.0, 2.0, 3.0, 4.0]
        kernel = [1.0, 0.5]
        direct = core.cos_comparison_active(base, kernel=kernel)
        via = core.cos_comparison_active(self.ActiveHook(base), kernel=kernel)
        self.assertEqual(list(direct), list(via))

    def test_passive_hook_python_reference(self):
        import cos_comparison.core.cos_comparison as _py

        base = [1.0, 2.0, 3.0, 4.0]
        direct = _py.cos_comparison_passive(base, window_size=(2,), d=(0,))
        via = _py.cos_comparison_passive(
            self.PassiveHook(base, call=_py.cos_comparison_passive),
            window_size=(2,), d=(0,))
        self.assertEqual(list(direct), list(via))

    def test_active_hook_python_reference(self):
        import cos_comparison.core.cos_comparison as _py

        base = [1.0, 2.0, 3.0, 4.0]
        kernel = [1.0, 0.5]
        direct = _py.cos_comparison_active(base, kernel=kernel)
        via = _py.cos_comparison_active(
            self.ActiveHook(base, call=_py.cos_comparison_active),
            kernel=kernel)
        self.assertEqual(list(direct), list(via))


if __name__ == "__main__":
    unittest.main()

