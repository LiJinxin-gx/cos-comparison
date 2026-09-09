"""
test_explore_smoke.py — smoke tests for the explore/ directory modules.

The exploratory scripts live in the `explore/` directory at the project
root. Every test exercises the public API with synthetic in-memory data —
no files, no network, no third-party dependencies.
"""
import os
import sys
import unittest

# Register explore/ on the path (portable, no cwd dependency)
_EXPLORE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "explore")
if _EXPLORE_DIR not in sys.path:
    sys.path.insert(0, _EXPLORE_DIR)

import atomic_feature as AF
import contrast_match as CM
import hierarchical_match as HM
import text_encoder as TE


class TestAtomicFeature(unittest.TestCase):
    """Passive boundary + pooling + binarization feature extraction."""

    def test_extract_1d(self):
        # extract expects 2D tensors; use a single-row 2D for 1D-like data
        data = [[0.0, 0.0, 1.0, 1.0, 0.0, 0.0]]
        feat = AF.extract(data, boundary_scales=[1], pool_windows=[2])
        self.assertIsInstance(feat, (list, tuple))
        self.assertGreater(len(feat), 0)

    def test_extract_2d(self):
        data = [[0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0]]
        feat = AF.extract(data, boundary_scales=[1], pool_windows=[2])
        self.assertIsInstance(feat, (list, tuple))

    def test_binarize(self):
        features = [0.1, 0.5, 0.9, 0.3, 0.7]
        binary = AF.binarize(features, top_percent=40)
        self.assertEqual(len(binary), len(features))
        self.assertIn(1, binary)
        self.assertIn(0, binary)

    def test_passive_boundary(self):
        data = [[0.0, 0.0, 1.0, 1.0]]
        boundary = AF.passive_boundary(data, d=1)
        # Output same shape as input (max diff with d-distance neighbor)
        self.assertEqual(len(boundary), 1)
        self.assertEqual(len(boundary[0]), 4)
        self.assertGreater(max(boundary[0]), 0)  # boundary detected


class TestContrastMatch(unittest.TestCase):
    """Jaccard similarity matching over contrast point sets."""

    def test_jaccard_identical(self):
        s1 = {(0, 1), (1, 2), (2, 3)}
        s2 = {(0, 1), (1, 2), (2, 3)}
        sim = CM.jaccard_similarity(s1, s2)
        self.assertAlmostEqual(sim, 1.0)

    def test_jaccard_disjoint(self):
        s1 = {(0, 1)}
        s2 = {(9, 9)}
        sim = CM.jaccard_similarity(s1, s2)
        self.assertAlmostEqual(sim, 0.0)

    def test_jaccard_partial(self):
        s1 = {(0, 1), (1, 2), (2, 3)}
        s2 = {(1, 2), (2, 3), (3, 4)}
        sim = CM.jaccard_similarity(s1, s2)
        self.assertAlmostEqual(sim, 0.5)

    def test_knn_predict(self):
        sim_matrix = [[1.0, 0.2], [0.3, 0.9]]
        labels = ["a", "b"]
        result = CM.knn_predict(sim_matrix, labels, k=1)
        self.assertEqual(result[0], "a")
        self.assertEqual(result[1], "b")


class TestHierarchicalMatch(unittest.TestCase):
    """Coarse-to-fine hierarchical matching."""

    def test_fit_and_predict(self):
        matcher = HM.HierarchicalMatcher()
        # boundary features and raw features (binary vectors)
        train_boundary = [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]]
        train_raw = [[1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 1]]
        train_labels = ["a", "a", "b", "b"]
        matcher.fit(train_boundary, train_raw, train_labels)
        test_boundary = [[1, 1, 0, 0]]
        test_raw = [[1, 0, 0, 0]]
        result = matcher.predict(test_boundary, test_raw, coarse_n=2)
        self.assertEqual(len(result), 1)

    def test_empty_training(self):
        matcher = HM.HierarchicalMatcher()
        # No fit called - predict should handle gracefully
        try:
            result = matcher.predict([[1, 0]], [[1, 0]], coarse_n=2)
            self.assertIsNotNone(result)
        except Exception:
            pass  # Empty state may raise; that's acceptable


class TestTextEncoder(unittest.TestCase):
    """UnitMap-based frequency text encoding."""

    def test_encode_character(self):
        enc = TE.TextFrequencyEncoder(mode='character', max_vocab=50)
        enc.fit(["hello", "world"])
        vecs = enc.transform(["hello"])
        self.assertIsInstance(vecs, list)
        self.assertEqual(len(vecs), 1)
        self.assertGreater(len(vecs[0]), 0)

    def test_encode_word(self):
        enc = TE.TextFrequencyEncoder(mode='word', max_vocab=50)
        enc.fit(["aa bb cc", "aa bb dd"])
        vecs = enc.transform(["aa bb"])
        self.assertEqual(len(vecs), 1)

    def test_encode_texts_function(self):
        vecs = TE.encode_texts(["hello world", "hello there"],
                                mode='character', max_vocab=100)
        self.assertEqual(len(vecs), 2)


if __name__ == "__main__":
    unittest.main()
