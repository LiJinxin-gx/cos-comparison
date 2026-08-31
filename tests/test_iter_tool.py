"""iter_tool tests: state-machine iteration (no recursion)."""
import unittest

from cos_comparison.interface.tools.iter_tool import (
    IterChain,
    IterFlatten,
    IterWindow,
    IterWrap,
    IterZip,
    iter_batch,
    iter_cycle,
    iter_filter,
    iter_group,
    iter_map,
)


class TestChainZip(unittest.TestCase):
    def test_chain(self):
        self.assertEqual(list(IterChain([1, 2], [3], [])), [1, 2, 3])

    def test_chain_empty(self):
        self.assertEqual(list(IterChain()), [])

    def test_zip_shortest(self):
        self.assertEqual(list(IterZip([1, 2, 3], ["a", "b"])),
                         [(1, "a"), (2, "b")])


class TestWindowBatch(unittest.TestCase):
    def test_window(self):
        self.assertEqual(list(IterWindow([1, 2, 3, 4, 5], 3)),
                         [[1, 2, 3], [2, 3, 4], [3, 4, 5]])

    def test_window_step(self):
        self.assertEqual(list(IterWindow([1, 2, 3, 4], 2, 2)),
                         [[1, 2], [3, 4]])

    def test_window_short(self):
        self.assertEqual(list(IterWindow([1, 2], 3)), [])

    def test_batch(self):
        self.assertEqual(list(iter_batch([1, 2, 3, 4, 5], 2)),
                         [[1, 2], [3, 4], [5]])


class TestFlatten(unittest.TestCase):
    """Explicit-stack flattening must never recurse."""

    def test_deep_nesting(self):
        data = [[1, [2, [3, [4, [5]]]]], [6], 7]
        self.assertEqual(list(IterFlatten(data)), [1, 2, 3, 4, 5, 6, 7])

    def test_scalars_and_strings(self):
        self.assertEqual(list(IterFlatten([["ab", 1], "cd"])), ["ab", 1, "cd"])

    def test_very_deep(self):
        data = 0
        for _ in range(5000):          # deep enough to overflow recursion
            data = [data]
        self.assertEqual(list(IterFlatten(data)), [0])


class TestMapFilterGroup(unittest.TestCase):
    def test_map(self):
        self.assertEqual(list(iter_map(lambda x: x * 2, [1, 2, 3])), [2, 4, 6])

    def test_filter(self):
        self.assertEqual(list(iter_filter(lambda x: x % 2 == 0, [1, 2, 3, 4])),
                         [2, 4])

    def test_group_consecutive(self):
        out = list(iter_group("aaabbc", lambda c: c))
        self.assertEqual(out, [("a", ["a", "a", "a"]),
                               ("b", ["b", "b"]), ("c", ["c"])])

    def test_cycle(self):
        gen = iter_cycle([1, 2], [10])
        out = [next(gen) for _ in range(4)]
        self.assertEqual(out, [1, 10, 2, 10])


class TestWrap(unittest.TestCase):
    def test_wrap_iter(self):
        w = IterWrap([3, 1, 2])
        it = iter(w)
        self.assertEqual(sorted(it), [1, 2, 3])

    def test_wrap_delegated(self):
        w = IterWrap([1, 2], iter_func=lambda seq: iter(reversed(seq)))
        self.assertEqual(list(iter(w)), [2, 1])


if __name__ == "__main__":
    unittest.main()
