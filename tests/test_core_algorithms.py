# -*- coding: utf-8 -*-
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
    """cosine-modulated similarity from the README formula."""
    dot = sum(a * b for a, b in zip(A, B))
    n2a = sum(a * a for a in A)
    n2b = sum(b * b for b in B)
    if n2a + n2b == 0.0:
        return 0.0
    return 2.0 * dot / (n2a + n2b)


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
        self.assertEqual(r.shape, (3, 4))  # window (1,1), d (0,0)

    def test_1d(self):
        v = core.create_void_list((8,))
        v[0], v[1], v[2] = 1.0, 1.0, 1.0
        r = core.cos_comparison_passive(v, window_size=(3,))
        self.assertGreater(r.shape[0], 0)

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


if __name__ == "__main__":
    unittest.main()
