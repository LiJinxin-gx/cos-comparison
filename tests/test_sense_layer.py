# -*- coding: utf-8 -*-
"""sense_layer.receptor: TensorReceptor and data extraction primitives."""

import unittest

from cos_comparison import core
from cos_comparison.sense_layer import receptor


class TestStatusCodes(unittest.TestCase):
    """Status code constants."""

    def test_status_values(self):
        self.assertEqual(receptor.OK, 0)
        self.assertEqual(receptor.SHAPE_MISMATCH, 1)
        self.assertEqual(receptor.NO_OUTPUT, 2)
        self.assertEqual(receptor.CONVERSION_FAILURE, 3)


class TestRegionIndices(unittest.TestCase):
    """_region_indices iterator (no recursion)."""

    def test_1d_region(self):
        indices = list(receptor._region_indices(start=(0,), shape=(3,), step=(1,)))
        self.assertEqual(indices, [(0,), (1,), (2,)])

    def test_2d_region(self):
        indices = list(receptor._region_indices(start=(0, 0), shape=(2, 2), step=(1, 1)))
        self.assertEqual(indices, [(0, 0), (0, 1), (1, 0), (1, 1)])

    def test_step_size(self):
        indices = list(receptor._region_indices(start=(0,), shape=(5,), step=(2,)))
        self.assertEqual(indices, [(0,), (2,), (4,)])


class TestElementwiseExtract(unittest.TestCase):
    """elementwise_extract function."""

    def test_basic_extract(self):
        data = [[1, 2, 3],
                [4, 5, 6]]
        output = [[0, 0, 0], [0, 0, 0]]
        result = receptor.elementwise_extract(data,
                                               start=(0, 0), shape=(2, 3),
                                               output=output, out_start=(0, 0),
                                               func=lambda x: x * 2)
        self.assertEqual(output[0], [2, 4, 6])


class TestTensorReceptor(unittest.TestCase):
    """TensorReceptor class."""

    def test_creation(self):
        data = [[1, 2], [3, 4]]
        rec = receptor.TensorReceptor(data)
        self.assertIsNotNone(rec)


class TestDataMatch(unittest.TestCase):
    """data_match function."""

    def test_same_shape(self):
        data = [[1, 2], [3, 4]]
        template = [[1, 2], [3, 4]]
        result = list(receptor.data_match(data, template, low=0.9, high=1.0))
        self.assertGreater(len(result), 0)


if __name__ == "__main__":
    unittest.main()
