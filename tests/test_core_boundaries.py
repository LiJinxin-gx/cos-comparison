"""core boundary conditions: formerly uncatchable failures (access
violations, heap corruption, a hanging shape walk) must surface as
Python exceptions instead.  The assertions accept the exception class of
either backend (C extension or pure-Python fallback)."""

import unittest

from cos_comparison import core


class TestVectorBounds(unittest.TestCase):
    def test_delete_scalar_is_type_error(self):
        v = core.create_void_list((3,))
        with self.assertRaises((TypeError, AttributeError)):
            del v[0]

    def test_delete_slice_is_type_error(self):
        v = core.create_void_list((3,))
        with self.assertRaises((TypeError, AttributeError)):
            del v[0:2]

    def test_start_override_out_of_bounds(self):
        with self.assertRaises((ValueError, IndexError, OverflowError)):
            v = core.vector_map_as_tensor(vector=[1.0, 2.0], start=10 ** 7)
            v[0]

    def test_strides_override_out_of_bounds(self):
        with self.assertRaises((ValueError, IndexError, OverflowError)):
            v = core.vector_map_as_tensor(vector=[1.0], shape=(4,),
                                          strides=(1000000,))
            v[3]

    def test_buffer_shape_beyond_buffer(self):
        with self.assertRaises((ValueError, IndexError, TypeError,
                                OverflowError)):
            v = core.vector_map_as_tensor(vector=memoryview(b"12345678"),
                                          shape=(1000,))
            v[999] = 1.0


class TestRegionBounds(unittest.TestCase):
    def test_negative_displacement(self):
        with self.assertRaises(ValueError):
            core.cos_comparison_passive_1d([1.0, 2.0, 3.0],
                                           window_size=(1,), d=(-1000000,))

    def test_end_beyond_data(self):
        with self.assertRaises(ValueError):
            core.cos_comparison_passive_1d([1.0, 2.0, 3.0],
                                           window_size=(1,), end=(100,))

    def test_negative_start_with_callback(self):
        with self.assertRaises(ValueError):
            core.cos_comparison_passive_1d(
                [1.0, 2.0], window_size=(1,), start=(-1000000,),
                start_callback=lambda ns: None)

    def test_output_start_without_output(self):
        with self.assertRaises(ValueError):
            core.cos_comparison_passive_1d(
                list(range(10)), window_size=(1,), output_start=(10 ** 9,))

    def test_negative_output_step(self):
        with self.assertRaises(ValueError):
            core.cos_comparison_passive_1d(
                list(range(10)), window_size=(1,), output_step=(-1,))

    def test_pool_end_beyond_data(self):
        with self.assertRaises(ValueError):
            core.pool([1.0], window_size=(1000000,), end=(1000000,),
                      output=[0.0])


class TestInferenceBoundaries(unittest.TestCase):
    def test_infer_shape_text_is_none(self):
        self.assertIsNone(core.infer_shape("abc"))

    def test_infer_shape_mapping_is_none(self):
        self.assertIsNone(core.infer_shape({1: 2}))

    def test_load_zero_shape_is_empty(self):
        result = core.load_as_default_data([1.0, 2.0], start=(0,),
                                           shape=(0,), step=(1,))
        self.assertEqual(result.shape, (0,))


if __name__ == "__main__":
    unittest.main()
