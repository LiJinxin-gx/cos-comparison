# -*- coding: utf-8 -*-
"""generate_layer.generator: TensorGenerator and data generation primitives."""

import unittest

from cos_comparison import core
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
        data = [1, 2, 3, 4, 5]
        output = [0, 0, 0]
        generator.copy_region(data, start=(1,), shape=(3,), output=output, out_start=(0,))
        self.assertEqual(output, [2, 3, 4])

    def test_copy_2d(self):
        data = [[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]]
        output = [[0, 0], [0, 0]]
        generator.copy_region(data, start=(0, 0), shape=(2, 2), output=output, out_start=(0, 0))
        self.assertEqual(output, [[1, 2], [4, 5]])


class TestTransformSelf(unittest.TestCase):
    """transform_self function."""

    def test_transform(self):
        data = [[1, 2], [3, 4]]
        result = generator.transform_self(data, func=lambda x: x * 2)
        # Should transform in place or return transformed
        self.assertIsNotNone(result)


class TestTensorGenerator(unittest.TestCase):
    """TensorGenerator class."""

    def test_creation(self):
        gen = generator.TensorGenerator()
        self.assertIsNotNone(gen)


if __name__ == "__main__":
    unittest.main()
