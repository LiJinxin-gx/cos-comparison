# -*- coding: utf-8 -*-
"""generate_layer.generator: TensorGenerator and data generation primitives."""

import unittest

from cos_comparison.generate_layer import generator


class TestStatusCodes(unittest.TestCase):
    """Status code constants."""

    def test_status_values(self):
        self.assertEqual(generator.OK, 0)
        self.assertEqual(generator.SHAPE_MISMATCH, 1)
        self.assertEqual(generator.NO_OUTPUT, 2)
        self.assertEqual(generator.CONVERSION_FAILURE, 3)


class TestCopyRegion(unittest.TestCase):
    """copy_region function."""

    def test_copy_1d(self):
        source = [1, 2, 3, 4, 5]
        target = [0, 0, 0]
        copied = generator.copy_region(target, source, shape=(3,),
                                       source_start=(1,))
        self.assertEqual(copied, 3)
        self.assertEqual(target, [2, 3, 4])

    def test_copy_2d(self):
        source = [[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]]
        target = [[0, 0], [0, 0]]
        copied = generator.copy_region(target, source, shape=(2, 2))
        self.assertEqual(copied, 4)
        self.assertEqual(target, [[1, 2], [4, 5]])


class TestTransformSelf(unittest.TestCase):
    """transform_self function."""

    def test_transform(self):
        data = [[1, 2], [3, 4]]
        status = generator.transform_self(data, func=lambda x: x * 2)
        self.assertEqual(status, 0)
        self.assertEqual(data, [[2, 4], [6, 8]])

    def test_transform_region_step(self):
        data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        status = generator.transform_self(data, func=lambda v: 1.0,
                                          start=(0,), shape=(3,), step=(2,))
        self.assertEqual(status, 0)
        self.assertEqual(data, [1.0, 0.0, 1.0, 0.0, 1.0, 0.0])


class TestTensorGenerator(unittest.TestCase):
    """TensorGenerator class."""

    def test_creation(self):
        gen = generator.TensorGenerator([1, 2, 3])
        self.assertEqual(gen.data, [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
