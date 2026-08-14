# -*- coding: utf-8 -*-
"""vector_map_as_tensor data structure: creation, indexing, slicing,
assignment, iteration, arithmetic, statistics."""

import unittest

from cos_comparison import core


def make(shape, default=0.0):
    return core.create_void_list(shape, default=default)


class TestTensorBasics(unittest.TestCase):
    def test_create_shape_and_value(self):
        v = make((3, 4), default=7.0)
        self.assertEqual(v.shape, (3, 4))
        self.assertEqual(v.dimension, 2)
        self.assertEqual(v[0, 0], 7.0)
        self.assertEqual(v[2, 3], 7.0)

    def test_scalar_create_known_crash(self):
        # scalar tensors (shape=()) are broken on the C backends:
        # ctypes returns a bare float, pydll corrupts the heap at
        # interpreter exit.  Covered (in a subprocess) by
        # TestKnownDivergences - skipped here on purpose.
        self.skipTest("shape=() tensor crashes the pydll backend; "
                      "see test_core_backends.TestKnownDivergences")

    def test_1d_create(self):
        v = make((10,), default=1.0)
        self.assertEqual(len(v), 10)
        self.assertEqual(v[9], 1.0)
        self.assertEqual(v[-1], 1.0)

    def test_3d_4d_create(self):
        v3 = make((2, 3, 4))
        self.assertEqual(v3.shape, (2, 3, 4))
        self.assertEqual(v3[1, 2, 3], 0.0)
        v4 = make((2, 2, 2, 2))
        self.assertEqual(v4.shape, (2, 2, 2, 2))
        self.assertEqual(v4[1, 1, 1, 1], 0.0)


class TestTensorAssignment(unittest.TestCase):
    def test_flat_assign(self):
        v = make((5, 5))
        v[1, 1] = 123.0
        self.assertEqual(v[1, 1], 123.0)
        self.assertEqual(v[1][1], 123.0)

    def test_negative_index(self):
        v = make((5, 5))
        v[-1, -1] = 9.0
        self.assertEqual(v[4, 4], 9.0)

    def test_1d_assign(self):
        v = make((5,))
        v[2] = 42.0
        self.assertEqual(v[2], 42.0)
        self.assertEqual(v[-3], 42.0)

    def test_bulk_assign_rows(self):
        v = make((3, 3))
        v[:, 0] = [1.0, 2.0, 3.0]
        self.assertEqual([v[i, 0] for i in range(3)], [1.0, 2.0, 3.0])

    def test_bulk_assign_2d(self):
        # nested lists are NOT auto-flattened; pass a flat list instead
        v = make((2, 2))
        v[:, :] = [1.0, 2.0, 3.0, 4.0]
        self.assertEqual(v[0, 0], 1.0)
        self.assertEqual(v[1, 1], 4.0)

    def test_bulk_assign_nested_rejected(self):
        v = make((2, 2))
        with self.assertRaises((ValueError, TypeError)):
            v[:, :] = [[1.0, 2.0], [3.0, 4.0]]


class TestTensorSlicing(unittest.TestCase):
    def setUp(self):
        self.v = make((6, 6))
        for i in range(6):
            for j in range(6):
                self.v[i, j] = float(i * 10 + j)

    def test_row_slice(self):
        row = self.v[2, :]
        self.assertEqual(row.shape, (6,))
        self.assertEqual(row[0], 20.0)
        self.assertEqual(row[5], 25.0)

    def test_column_slice(self):
        col = self.v[:, 3]
        self.assertEqual(col.shape, (6,))
        self.assertEqual(col[0], 3.0)

    def test_step_slice(self):
        sub = self.v[::2, ::2]
        self.assertEqual(sub.shape, (3, 3))
        self.assertEqual(sub[1, 1], 22.0)

    def test_view_write_through(self):
        sub = self.v[1:4, 1:4]
        sub[0, 0] = 99.0
        self.assertEqual(self.v[1, 1], 99.0)


class TestTensorIteration(unittest.TestCase):
    def test_iter_rows(self):
        v = make((3, 2))
        rows = [list(r) for r in v]
        self.assertEqual(rows, [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])

    def test_len_matches_shape(self):
        v = make((3, 4, 5))
        self.assertEqual(len(v), 3)
        self.assertEqual(len(v[0]), 4)
        self.assertEqual(len(v[0][0]), 5)


class TestTensorArithmetic(unittest.TestCase):
    def setUp(self):
        self.a = make((2, 2), default=2.0)

    def test_add(self):
        b = self.a + self.a
        self.assertEqual(b[0, 0], 4.0)
        self.assertEqual(b[1, 1], 4.0)

    def test_radd(self):
        b = 1.0 + self.a
        self.assertEqual(b[0, 0], 3.0)

    def test_sub_mul_div(self):
        self.assertEqual((self.a - 1.0)[0, 0], 1.0)
        self.assertEqual((self.a * 3.0)[0, 0], 6.0)
        self.assertEqual((self.a / 2.0)[0, 0], 1.0)

    def test_inplace(self):
        self.a *= 2.0
        self.assertEqual(self.a[0, 0], 4.0)
        self.a += self.a
        self.assertEqual(self.a[0, 0], 8.0)

    def test_unary(self):
        self.assertEqual((-self.a)[0, 0], -2.0)

    def test_type_check(self):
        with self.assertRaises((TypeError, ValueError)):
            _ = self.a + make((3, 3))


class TestTensorStatistics(unittest.TestCase):
    def test_mean(self):
        v = make((4,))
        v[0], v[1], v[2], v[3] = 1.0, 2.0, 3.0, 4.0
        self.assertAlmostEqual(v.mean(), 2.5)

    def test_variance(self):
        v = make((4,))
        v[0], v[1], v[2], v[3] = 1.0, 2.0, 3.0, 4.0
        self.assertAlmostEqual(v.variance(), 1.25)

    def test_negative_values(self):
        v = make((3,))
        v[0], v[1], v[2] = -1.0, 0.0, 1.0
        self.assertAlmostEqual(v.mean(), 0.0)
        self.assertAlmostEqual(v.variance(), 2.0 / 3.0)


class TestToolChain(unittest.TestCase):
    def test_multiple_chain(self):
        self.assertEqual(core.multiple_chain((2, 3, 4)), 24)
        self.assertEqual(core.multiple_chain([], 5), 5)

    def test_add_chain(self):
        self.assertEqual(core.add_chain((1, 2, 3)), 6)
        self.assertEqual(core.add_chain([], 5), 5)

    def test_vector_chain_compute(self):
        compute, fix, get = core.vector_chain_compute(((1.0, 2.0), (3.0, 4.0)))
        self.assertEqual(compute((1.0, 1.0)), (3.0, 7.0))
        fix(((1.0, 1.0), (1.0, 1.0)))
        self.assertEqual(compute((1.0, 1.0)), (2.0, 2.0))
        self.assertEqual(get(), ((1.0, 1.0), (1.0, 1.0)))

    def test_get_set_item(self):
        v = make((3, 3))
        core.set_item(v, (1, 1), 7.0)
        self.assertEqual(core.get_item(v, (1, 1)), 7.0)

    def test_default_contain_read(self):
        # item assignment is not supported on any backend (v0.4.1);
        # lookup falls back to the default value
        dc = core.default_contain(5.0)
        self.assertEqual(len(dc), 1)
        self.assertEqual(dc["anything"], 5.0)
        self.assertEqual(dc[None], 5.0)


if __name__ == "__main__":
    unittest.main()
