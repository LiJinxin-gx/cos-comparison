# -*- coding: utf-8 -*-
"""sense_layer.receptor: TensorReceptor and data extraction primitives."""

import unittest

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
        self.assertEqual(indices, [(0,), (2,), (4,), (6,), (8,)])


class TestElementwiseExtract(unittest.TestCase):
    """elementwise_extract function."""

    def test_basic_extract(self):
        data = [[1, 2, 3],
                [4, 5, 6]]
        output = [[0, 0, 0], [0, 0, 0]]
        receptor.elementwise_extract(data,
                                     start=(0, 0), shape=(2, 3),
                                     output=output, out_start=(0, 0),
                                     func=lambda x: x * 2)
        self.assertEqual(output[0], [2, 4, 6])

    def test_region_clipped_to_data(self):
        output = [0.0, 0.0]
        status = receptor.elementwise_extract([1.0, 2.0, 3.0, 4.0],
                                              func=lambda v: v * 2,
                                              output=output, start=(2,),
                                              shape=(5,))
        self.assertEqual(status, receptor.OK)
        self.assertEqual(output, [6.0, 8.0])

    def test_region_shape_defaults_to_full(self):
        output = [0.0, 0.0]
        status = receptor.elementwise_extract([1.0, 2.0, 3.0],
                                              func=lambda v: v + 1,
                                              output=output, start=(1,))
        self.assertEqual(status, receptor.OK)
        self.assertEqual(output, [3.0, 4.0])

    def test_region_step_selects_samples(self):
        output = [0.0, 0.0, 0.0]
        status = receptor.elementwise_extract(
            [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            func=lambda v: v, output=output,
            start=(0,), shape=(3,), step=(2,))
        self.assertEqual(status, receptor.OK)
        self.assertEqual(output, [10.0, 30.0, 50.0])

    def test_invalid_step_returns_mismatch(self):
        status = receptor.elementwise_extract([1.0, 2.0],
                                              func=lambda v: v,
                                              output=[0.0, 0.0],
                                              start=(0,), shape=(2,),
                                              step=(0,))
        self.assertEqual(status, receptor.SHAPE_MISMATCH)

    def test_nd_region_full_output(self):
        output = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        status = receptor.elementwise_extract(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            func=lambda v: v * 2, output=output,
            start=(0, 0), shape=(2, 3))
        self.assertEqual(status, receptor.OK)
        self.assertEqual(output, [[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]])


class TestTensorReceptor(unittest.TestCase):
    """TensorReceptor class."""

    def test_creation(self):
        data = [[1, 2], [3, 4]]
        rec = receptor.TensorReceptor(data)
        self.assertEqual(rec.data, data)

    def test_threshold_map_dimension_mismatch(self):
        tr = receptor.TensorReceptor([1.0, 2.0, 3.0])
        status = tr.threshold_map([], output=[[0.0], [0.0]])
        self.assertEqual(status, receptor.SHAPE_MISMATCH)


class TestDataMatch(unittest.TestCase):
    """data_match function."""

    def test_same_shape(self):
        data = [[1, 2], [3, 4]]
        template = [[1, 2], [3, 4]]
        result = list(receptor.data_match(data, template, low=0.9, high=1.0))
        # identical template: only the (0, 0) origin sits in [0.9, 1.0]
        self.assertEqual(result, [(0, 0)])


if __name__ == "__main__":
    unittest.main()
