"""
test_explore_smoke.py — layered integration tests.

Data flow contract (decoupled):
    explore_data.read(path) -> tensor | str | bytes   (acquisition)
    analyze_* (tensor | str) -> result               (analysis)

The exploratory scripts live in the `explore/` directory at the
project root; every test exercises the full pipeline with synthetic
in-memory data — no files, no network, no third-party dependencies.
"""
import unittest

from cos_comparison.core import cos
from explore_data import (BinarySource, ImageSource, MnistSource, TextSource,
                          VideoSource, WavSource, block_pool, flat, resize_grid)
import explore_text as AT
import explore_vision as AV
import explore_audio as AA
import explore_generate as GD
import explore_memory as MH


class TestSources(unittest.TestCase):
    """Acquisition layer: uniform read() -> model-ready types."""

    def test_binary(self):
        data = BinarySource(read_func=lambda path: b"\x00\x01")
        self.assertEqual(data.read("x"), b"\x00\x01")

    def test_text(self):
        t = TextSource(read_func=lambda path: "hello 测试")
        self.assertEqual(t.read("x"), "hello 测试")

    def test_image_tensor(self):
        img = ImageSource(read_func=lambda path:
                          [[0.0, 1.0], [0.5, 0.25]])
        tensor = img.read("x")
        self.assertEqual(len(tensor), 2)          # indexable + len
        self.assertEqual(len(tensor[0]), 2)
        self.assertEqual(tensor[1][0], 0.5)

    def test_mnist(self):
        m = MnistSource(image_func=lambda path, gz=True: [[[0.0, 1.0]]],
                        label_func=lambda path, gz=True: [0])
        self.assertEqual(m.read_images("x")[0][0][1], 1.0)
        self.assertEqual(m.read_labels("x"), [0])

    def test_wav(self):
        import struct
        raw = b"".join(struct.pack("<h", int(v * 32767))
                       for v in (0.5, -0.5))
        w = WavSource(read_func=lambda path, seconds=4.0:
                      [struct.unpack("<h", raw[i:i + 2])[0] / 32768.0
                       for i in range(0, len(raw) - 1, 2)])
        samples = w.read("x")
        self.assertEqual(len(samples), 2)
        self.assertAlmostEqual(samples[0], 0.5, places=3)

    def test_video_unavailable(self):
        from explore_data import VideoSource
        with self.assertRaises(NotImplementedError):
            VideoSource().read("movie.mp4")

    def test_tensor_helpers(self):
        self.assertEqual(flat([[1, 2], [3, 4]]), [1, 2, 3, 4])
        self.assertEqual(resize_grid([[1, 2], [3, 4]], 1, 4),
                         [[1, 1, 2, 2]])
        self.assertEqual(block_pool([[1, 0], [0, 0]], 2), [1.0, 0, 0, 0])
        self.assertAlmostEqual(cos([1, 0], [1, 0]), 1.0)


class TestVisionAnalysis(unittest.TestCase):
    """Analysis layer: tensors in, classification out."""

    def _tensor(self, kind):
        if kind == "a":
            return [[1.0 if x % 2 == 0 else 0.0 for x in range(8)]
                    for _ in range(8)]
        return [[1.0 if x % 3 == 0 else 0.0 for x in range(8)]
                for _ in range(8)]

    def test_feature_and_classify(self):
        tensors = [self._tensor("a")] * 4 + [self._tensor("b")] * 4
        labels = ["a"] * 4 + ["b"] * 4
        protos = AV.build_prototypes(tensors, labels)
        self.assertEqual(AV.prototype_classify(self._tensor("a"), protos),
                         "a")
        self.assertEqual(AV.prototype_classify(self._tensor("b"), protos),
                         "b")

    def test_anchor_map(self):
        mapped = AV.anchor_map([[0.1, 0.9], [0.5, 0.0]])
        self.assertEqual(len(mapped), 2)
        self.assertEqual(len(mapped[0]), 2)


class TestTextAnalysis(unittest.TestCase):
    """Analysis layer: strings in, labels out."""

    def test_token_freq(self):
        freq = AT.token_freq("abc abc xyz")
        self.assertEqual(freq["abc"], 2)

    def test_cluster(self):
        groups, _vocab = AT.cluster(["aa bb cc", "aa bb dd",
                                    "zz yy xx", "zz yy ww"])
        self.assertEqual(len(groups), 2)

    def test_hierarchical_labels(self):
        labels = AT.hierarchical_labels(["aa bb cc dd", "aa bb cc ee"],
                                        levels=2)
        self.assertEqual(labels[0], labels[1])


class TestAudioAnalysis(unittest.TestCase):
    def test_spectral_feature(self):
        import math
        samples = [math.sin(i / 20.0) for i in range(2048)]
        feat = AA.spectral_feature(samples, frame_size=256, step=128)
        self.assertGreater(len(feat), 0)


class TestGeneration(unittest.TestCase):
    def test_templates(self):
        tpl = [[1.0, 0.0], [0.0, 1.0]]
        gen = GD.template_pool([tpl], 3, noise=0.0)
        self.assertEqual(len(gen), 3)
        self.assertEqual(len(gen[0]), 2)


class TestHierMemory(unittest.TestCase):
    def test_demand_driven(self):
        mem = MH.HierMemory()
        mem.absorb([{"text": "student pressure mental health", "url": "1"},
                    {"text": "exam competition involution", "url": "2"},
                    {"text": "unrelated shopping mall", "url": "3"}])
        mem.set_needs(["pressure", "health"])
        r = mem.map_l2()
        self.assertEqual(r["n_topics"], 1)
        self.assertEqual(r["isolated"], 1)      # unrelated stays in L1
        mem.map_l3({"health": ["health", "mental"],
                    "competition": ["competition", "involution"]})
        self.assertIn("competition", mem.needs)  # gap drives needs
        self.assertEqual(mem.report()["l2_sizes"], [2])


if __name__ == "__main__":
    unittest.main()
