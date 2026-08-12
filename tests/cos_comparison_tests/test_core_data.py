# -*- coding: utf-8 -*-
"""infer_shape, create_void_list, load_as_default_data, load_data."""

import unittest

from cos_comparison import core


class TestInferShape(unittest.TestCase):
    def test_nested_lists(self):
        self.assertEqual(core.infer_shape([[1, 2], [3, 4]]), (2, 2))
        self.assertEqual(core.infer_shape([1, 2, 3]), (3,))
        self.assertEqual(core.infer_shape([]), (0,))

    def test_3d(self):
        data = [[[1]] * 2 for _ in range(3)]
        self.assertEqual(core.infer_shape(data), (3, 2, 1))

    def test_tensor(self):
        v = core.create_void_list((4, 5, 6))
        self.assertEqual(core.infer_shape(v), (4, 5, 6))

    def test_bytes_buffer(self):
        self.assertEqual(core.infer_shape(b"abcd"), (4,))

    def test_scalar_backend_dependent(self):
        # infer_shape(3.14): pure Python returns None, pydll returns ()
        # - backend divergence, see TestKnownDivergences
        self.assertIn(core.infer_shape(3.14), (None, ()))

    def test_ragged(self):
        # ragged rows: shape follows the first row's length
        self.assertEqual(core.infer_shape([[1], [2, 3]]), (2, 1))


class TestCreateVoidList(unittest.TestCase):
    def test_empty_shape(self):
        v = core.create_void_list()
        self.assertEqual(v.shape, (1,))

    def test_big_volume(self):
        v = core.create_void_list((100, 100), default=1.0)
        self.assertEqual(v[99, 99], 1.0)

    def test_default_float(self):
        self.assertEqual(core.create_void_list((2,)).shape, (2,))
        self.assertEqual(core.create_void_list((2,))[0], 0.0)


class TestLoadAsDefaultData(unittest.TestCase):
    def test_nested_list_full(self):
        v = core.load_as_default_data([[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(v.shape, (2, 2))
        self.assertEqual(v[1, 1], 4.0)

    def test_with_step(self):
        data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        v = core.load_as_default_data(data, step=(2, 2))
        self.assertEqual(v.shape, (2, 2))
        self.assertEqual(v[0, 0], 1.0)
        self.assertEqual(v[1, 1], 9.0)

    def test_with_start_and_shape(self):
        data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        v = core.load_as_default_data(data, start=(1, 1), shape=(2, 2))
        self.assertEqual(v.shape, (2, 2))
        self.assertEqual(v[0, 0], 5.0)
        self.assertEqual(v[1, 1], 9.0)

    def test_from_tensor(self):
        src = core.create_void_list((6, 6))
        for i in range(6):
            for j in range(6):
                src[i, j] = float(i * 10 + j)
        # start=(1,1), shape=(2,2) -> end=(3,3); step=(2,2) keeps only
        # index 1 in each axis -> result is (1,1) holding src[1,1] = 11
        v = core.load_as_default_data(src, start=(1, 1), shape=(2, 2),
                                      step=(2, 2))
        self.assertEqual(v.shape, (1, 1))
        self.assertEqual(v[0, 0], 11.0)

    def test_bad_step_length(self):
        with self.assertRaises(ValueError):
            core.load_as_default_data([[1.0, 2.0]], step=(1, 1, 1))

    def test_zero_step(self):
        with self.assertRaises(ValueError):
            core.load_as_default_data([[1.0, 2.0]], step=(0, 1))

    def test_no_shape_inferable(self):
        with self.assertRaises(ValueError):
            core.load_as_default_data(3.14)

    def test_1d(self):
        v = core.load_as_default_data([1.0, 2.0, 3.0, 4.0], step=(2,))
        self.assertEqual(v.shape, (2,))
        self.assertEqual(v[0], 1.0)
        self.assertEqual(v[1], 3.0)


class TestLoadData(unittest.TestCase):
    def test_copy_count(self):
        src = core.load_as_default_data([[1.0, 2.0], [3.0, 4.0]])
        dst = core.create_void_list((2, 2))
        n = core.load_data(src, dst)
        self.assertEqual(n, 4)
        self.assertEqual(dst[1, 1], 4.0)

    def test_subregion(self):
        src = core.create_void_list((6, 6))
        for i in range(6):
            for j in range(6):
                src[i, j] = float(i * 10 + j)
        dst = core.create_void_list((2, 2))
        n = core.load_data(src, dst, source_start=(2, 2), shape=(2, 2))
        self.assertEqual(n, 4)
        self.assertEqual(dst[0, 0], 22.0)
        self.assertEqual(dst[1, 1], 33.0)


if __name__ == "__main__":
    unittest.main()
