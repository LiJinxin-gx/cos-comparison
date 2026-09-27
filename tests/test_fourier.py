# -*- coding: utf-8 -*-
"""fourier tests: kernel customization
y[i] = amp[i]*func(2*pi*sum_d(scale[d]*freq[d]*x_d/N_d) + sum_d(offset[d]))
       + bias[i]."""
import math
import unittest

from cos_comparison.interface.tools.math_tool import fourier


class TestLegacyCompatibility(unittest.TestCase):
    """Default offset/scale/amp/bias must reproduce legacy kernels."""

    def test_legacy_1d_real(self):
        k = fourier.dft_kernel_real(8, 1)
        self.assertEqual(len(k), 8)
        for x in range(8):
            self.assertAlmostEqual(k[x], math.cos(2 * math.pi * x / 8))

    def test_legacy_1d_imag(self):
        k = fourier.dft_kernel_imag(8, 2)
        for x in range(8):
            self.assertAlmostEqual(k[x], math.sin(2 * math.pi * 2 * x / 8))

    def test_legacy_2d(self):
        k = fourier.dft_kernel_real((4, 4), (1, 0))
        for i in range(4):
            for j in range(4):
                self.assertAlmostEqual(
                    k[i][j], math.cos(2 * math.pi * 1 * i / 4))


class TestOffset(unittest.TestCase):
    """offset is a per-axis sequence; sum shifts the phase outside 2*pi."""

    def test_offset_1d_sequence(self):
        k = fourier.dft_kernel_real(8, 1, offsets=(0.25,))
        for x in range(8):
            self.assertAlmostEqual(
                k[x], math.cos(2 * math.pi * x / 8 + 0.25))

    def test_offset_2d_sequence(self):
        k = fourier.dft_kernel_real((4, 4), (1, 1), offsets=(0.25, 0.5))
        for i in range(4):
            for j in range(4):
                self.assertAlmostEqual(
                    k[i][j],
                    math.cos(2 * math.pi * (i / 4.0 + j / 4.0) + 0.75))

    def test_offset_scalar_broadcast(self):
        k = fourier.dft_kernel_real(8, 1, offsets=0.5)
        self.assertAlmostEqual(k[0], math.cos(0.5))


class TestScale(unittest.TestCase):
    """scale is a per-axis affine factor (scalar or sequence)."""

    def test_scalar_scale(self):
        k = fourier.dft_kernel_real(8, 1, scales=2.0)
        for x in range(8):
            self.assertAlmostEqual(
                k[x], math.cos(2 * math.pi * 2.0 * x / 8))

    def test_per_axis_scale_2d(self):
        k = fourier.dft_kernel_real((4, 4), (1, 1), scales=(2.0, 0.5))
        for i in range(4):
            for j in range(4):
                self.assertAlmostEqual(
                    k[i][j],
                    math.cos(2 * math.pi * (2.0 * i / 4 + 0.5 * j / 4)))


class TestAmpBias(unittest.TestCase):
    """amp/bias are per-element (scalar broadcast or element sequence)."""

    def test_scalar_amp_bias(self):
        k = fourier.dft_kernel_real(4, 1, amplitudes=2.0, biases=1.0)
        for x in range(4):
            self.assertAlmostEqual(
                k[x], 2.0 * math.cos(2 * math.pi * x / 4) + 1.0)

    def test_per_element_amp(self):
        amps = [0.0, 1.0, 2.0, 3.0]
        k = fourier.dft_kernel_real(4, 1, amplitudes=amps)
        for x in range(4):
            self.assertAlmostEqual(
                k[x], amps[x] * math.cos(2 * math.pi * x / 4))

    def test_per_element_bias_2d(self):
        shape = (2, 2)
        biases = [0.0, 1.0, 2.0, 3.0]
        k = fourier.dft_kernel_real(shape, (1, 0), biases=biases)
        flat = [k[0][0], k[0][1], k[1][0], k[1][1]]
        for t in range(4):
            self.assertAlmostEqual(
                flat[t],
                math.cos(2 * math.pi * (t // 2) / 2) + biases[t])


class TestCombined(unittest.TestCase):
    def test_full_formula(self):
        k = fourier.dft_kernel_imag(4, 1, scales=2.0, offsets=(0.125,),
                                    amplitudes=3.0, biases=0.5)
        for x in range(4):
            self.assertAlmostEqual(
                k[x],
                3.0 * math.sin(2 * math.pi * 2.0 * x / 4 + 0.125) + 0.5)


class TestValidation(unittest.TestCase):
    def test_scale_length_mismatch(self):
        with self.assertRaises(ValueError):
            fourier.dft_kernel_real((4, 4), (1, 1), scales=(1.0,))

    def test_offset_length_mismatch(self):
        with self.assertRaises(ValueError):
            fourier.dft_kernel_real((4, 4), (1, 1), offsets=(0.0,))

    def test_amp_length_mismatch(self):
        with self.assertRaises(ValueError):
            fourier.dft_kernel_real((2, 2), (1, 1), amplitudes=(1.0, 1.0))

    def test_freq_out_of_range(self):
        with self.assertRaises(ValueError):
            fourier.dft_kernel_real(8, 9)


if __name__ == "__main__":
    unittest.main()
