# -*- coding: utf-8 -*-
"""Random tool tests for cos_comparison.interface.tools.random_tool.

Covers the abstract RandomToolBase, the default RandomTool (random module),
the SecureRandomTool (secrets), generation / shuffle / choice / sample /
weighted choices, error handling, seeding, and custom-backend extension.
"""

import unittest


def make_random():
    from cos_comparison.interface.tools.random_tool import RandomTool
    return RandomTool()


def make_secure():
    from cos_comparison.interface.tools.random_tool import SecureRandomTool
    return SecureRandomTool()


class TestRandomToolBase(unittest.TestCase):
    def test_abstract_cannot_instantiate(self):
        from cos_comparison.interface.tools.random_tool import RandomToolBase
        with self.assertRaises(TypeError):
            RandomToolBase()

    def test_minimal_backend_full_api(self):
        """A backend implementing only random() still gains the full API."""
        from cos_comparison.interface.tools.random_tool import RandomToolBase

        class MinimalTool(RandomToolBase):
            __slots__ = ()

            def random(self):
                return 0.5

        t = MinimalTool()
        self.assertEqual(t.random(), 0.5)
        self.assertEqual(t.uniform(0.0, 1.0), 0.5)
        self.assertEqual(t.randint(3, 3), 3)
        self.assertEqual(t.randrange(10), 5)
        self.assertEqual(t.choice(["x", "y"]), "y")
        seq = [1, 2, 3]
        t.shuffle(seq)
        self.assertEqual(sorted(seq), [1, 2, 3])
        self.assertEqual(t.sample([1, 2, 3], 2), [3, 2])
        self.assertEqual(t.choices(["a", "b"], weights=[5, 1], k=3), ["a", "a", "a"])
        self.assertEqual(t._weighted_index([1, 0]), 0)
        self.assertIn(t._randbelow(7), range(7))
        with self.assertRaises(ValueError):
            t._randbelow(0)
        with self.assertRaises(ValueError):
            t._weighted_index([-1, 2])
        with self.assertRaises(NotImplementedError):
            t.seed()

    def test_concrete_backends_implement_all_primitives(self):
        from cos_comparison.interface.tools.random_tool import (RandomTool,
                                                                SecureRandomTool)
        for t in (RandomTool(), SecureRandomTool()):
            self.assertTrue(callable(t.random))
            self.assertTrue(callable(t._randbelow))
            self.assertTrue(callable(t._weighted_index))
            self.assertIn(t._randbelow(100), range(100))
            self.assertIn(t._weighted_index([3, 1, 1]), range(3))

    def test_repr(self):
        from cos_comparison.interface.tools.random_tool import RandomTool
        self.assertIn("RandomTool", repr(RandomTool()))


class TestRandomGeneration(unittest.TestCase):
    def test_random_float_in_unit_interval(self):
        for t in (make_random(), make_secure()):
            values = [t.random() for _ in range(1000)]
            for v in values:
                self.assertGreaterEqual(v, 0.0)
                self.assertLess(v, 1.0)
            self.assertGreater(len(set(values)), 900)

    def test_uniform_bounds(self):
        for t in (make_random(), make_secure()):
            for _ in range(200):
                v = t.uniform(2.0, 3.0)
                self.assertGreaterEqual(v, 2.0)
                self.assertLessEqual(v, 3.0)

    def test_randint_bounds(self):
        for t in (make_random(), make_secure()):
            for _ in range(200):
                v = t.randint(-5, 5)
                self.assertGreaterEqual(v, -5)
                self.assertLessEqual(v, 5)
            self.assertEqual(t.randint(7, 7), 7)
            with self.assertRaises(ValueError):
                t.randint(5, 4)

    def test_randrange(self):
        for t in (make_random(), make_secure()):
            for _ in range(100):
                self.assertIn(t.randrange(5), range(5))
            for _ in range(100):
                self.assertIn(t.randrange(1, 9), range(1, 9))
            for _ in range(100):
                self.assertIn(t.randrange(0, 10, 3), {0, 3, 6, 9})
            for _ in range(100):
                self.assertIn(t.randrange(5, 0, -2), {5, 3, 1})
            with self.assertRaises(ValueError):
                t.randrange(0, 10, 0)
            with self.assertRaises(ValueError):
                t.randrange(5, 5)
            with self.assertRaises(ValueError):
                t.randrange(1, 0)


class TestRandomSelection(unittest.TestCase):
    def test_choice(self):
        for t in (make_random(), make_secure()):
            self.assertEqual(t.choice(["only"]), "only")
            for _ in range(100):
                self.assertIn(t.choice("abc"), "abc")
            with self.assertRaises(IndexError):
                t.choice([])

    def test_shuffle_keeps_multiset(self):
        for t in (make_random(), make_secure()):
            for _ in range(20):
                seq = list(range(50))
                t.shuffle(seq)
                self.assertEqual(sorted(seq), list(range(50)))
            single = [42]
            t.shuffle(single)
            self.assertEqual(single, [42])

    def test_sample(self):
        for t in (make_random(), make_secure()):
            pop = list(range(30))
            s = t.sample(pop, 10)
            self.assertEqual(len(s), 10)
            self.assertEqual(len(set(s)), 10)
            self.assertEqual(t.sample(pop, 0), [])
            with self.assertRaises(ValueError):
                t.sample(pop, 31)
            with self.assertRaises(ValueError):
                t.sample(pop, -1)
            with self.assertRaises(TypeError):
                t.sample(pop, 1.5)

    def test_sample_full_returns_shuffled_population(self):
        for t in (make_random(), make_secure()):
            pop = list(range(100))
            s = t.sample(pop, 100)
            self.assertEqual(sorted(s), pop)

    def test_choices_unweighted(self):
        for t in (make_random(), make_secure()):
            self.assertEqual(t.choices(["a"], k=5), ["a", "a", "a", "a", "a"])
            res = t.choices([1, 2, 3], k=50)
            self.assertEqual(len(res), 50)
            for v in res:
                self.assertIn(v, [1, 2, 3])
            with self.assertRaises(IndexError):
                t.choices([], k=1)


class TestWeightedRandom(unittest.TestCase):
    def test_weights_deterministic_extremes(self):
        for t in (make_random(), make_secure()):
            res = t.choices(["a", "b"], weights=[1, 0], k=300)
            self.assertEqual(res, ["a"] * 300)
            res = t.choices(["a", "b"], weights=[0, 1], k=300)
            self.assertEqual(res, ["b"] * 300)
            res = t.choices(["a", "b"], weights=[100, 0], k=300)
            self.assertEqual(res, ["a"] * 300)

    def test_weights_rough_distribution(self):
        t = make_random()
        counts = {0: 0, 1: 0}
        for _ in range(10000):
            idx = t._weighted_index([90, 10])
            counts[idx] += 1
        self.assertGreater(counts[0], counts[1] * 5)
        self.assertGreater(counts[1], 0)

    def test_weights_int_and_float(self):
        for t in (make_random(), make_secure()):
            for w in ([3, 1], [0.5, 0.5], [0.2, 0.8]):
                idx = t._weighted_index(w)
                self.assertIn(idx, range(len(w)))

    def test_weights_errors(self):
        for t in (make_random(), make_secure()):
            with self.assertRaises(ValueError):
                t._weighted_index([-1, 2])
            with self.assertRaises(ValueError):
                t._weighted_index([0, 0])
            with self.assertRaises(ValueError):
                t.choices(["a", "b"], weights=[1], k=1)
            with self.assertRaises(ValueError):
                t.choices([], weights=[], k=1)
            with self.assertRaises(ValueError):
                t.choices(["a", "b"], weights=[1, 0], k=-1)
            with self.assertRaises(TypeError):
                t.choices(["a", "b"], k=1.5)


class TestSeeding(unittest.TestCase):
    def test_random_tool_reproducible(self):
        from cos_comparison.interface.tools.random_tool import RandomTool
        t1, t2 = RandomTool(), RandomTool()
        t1.seed(42)
        t2.seed(42)
        self.assertEqual(t1.random(), t2.random())
        self.assertEqual(t1.randint(0, 1000), t2.randint(0, 1000))
        self.assertEqual(t1.choice("abcdefgh"), t2.choice("abcdefgh"))
        seq1 = list("abcdef")
        seq2 = list("abcdef")
        t1.shuffle(seq1)
        t2.shuffle(seq2)
        self.assertEqual(seq1, seq2)

    def test_secure_tool_rejects_seed(self):
        t = make_secure()
        with self.assertRaises(TypeError):
            t.seed(42)
        with self.assertRaises(TypeError):
            t.seed()

    def test_injectable_generator(self):
        from cos_comparison.interface.tools.random_tool import RandomTool
        import random
        r = random.Random(7)
        t = RandomTool(rand=r)
        t.seed(7)
        self.assertIs(t._random, r)
        self.assertEqual(t.random(), random.Random(7).random())


class TestDelegatedRandomTool(unittest.TestCase):
    def test_concrete_and_same_level_as_others(self):
        from cos_comparison.interface.tools.random_tool import (DelegatedRandomTool,
                                                                RandomTool,
                                                                SecureRandomTool,
                                                                RandomToolBase)
        t = DelegatedRandomTool()
        self.assertIsInstance(t, RandomToolBase)
        self.assertIsInstance(RandomTool(), RandomToolBase)
        self.assertIsInstance(SecureRandomTool(), RandomToolBase)
        self.assertIs(type(t).__mro__[1], RandomToolBase)

    def test_inject_functions(self):
        from cos_comparison.interface.tools.random_tool import DelegatedRandomTool
        import random

        calls = {"random": 0, "below": 0, "weighted": 0, "seed": 0}
        r = random.Random(3)

        def rnd():
            calls["random"] += 1
            return r.random()

        def below(n):
            calls["below"] += 1
            return r.randrange(n)

        def weighted(weights):
            calls["weighted"] += 1
            return r.choices(range(len(weights)), weights=weights, k=1)[0]

        def seed(a):
            calls["seed"] += 1
            r.seed(a)

        t = DelegatedRandomTool(random_func=rnd, randbelow_func=below,
                                weighted_func=weighted, seed_func=seed)
        for _ in range(20):
            v = t.random()
            self.assertGreaterEqual(v, 0.0)
            self.assertLess(v, 1.0)
        self.assertIn(t.randint(0, 100), range(0, 101))
        self.assertIn(t.choice("abcdef"), "abcdef")
        self.assertEqual(len(t.sample(range(20), 5)), 5)
        self.assertEqual(len(t.choices(["a", "b"], weights=[1, 1], k=10)), 10)
        t.seed(42)
        self.assertGreater(calls["random"], 0)
        self.assertGreater(calls["below"], 0)
        self.assertGreater(calls["weighted"], 0)
        self.assertEqual(calls["seed"], 1)

    def test_inject_random_module_instance(self):
        from cos_comparison.interface.tools.random_tool import DelegatedRandomTool
        import random

        t = DelegatedRandomTool(random.Random(42))
        for _ in range(50):
            v = t.random()
            self.assertGreaterEqual(v, 0.0)
            self.assertLess(v, 1.0)
        self.assertIn(t._randbelow(100), range(100))
        t.seed(7)
        self.assertIn(t.randint(0, 100), range(0, 101))
        # random.Random exposes no _weighted_index - adapting choices would
        # be high-difficulty delegation, so the slot is left unresolved.
        with self.assertRaises(NotImplementedError):
            t._weighted_index([3, 1, 1])
        with self.assertRaises(NotImplementedError):
            t.choices(["a", "b"], weights=[1, 1], k=1)

    def test_inject_secrets_module(self):
        from cos_comparison.interface.tools.random_tool import DelegatedRandomTool
        import secrets

        t = DelegatedRandomTool(secrets)
        for _ in range(50):
            self.assertIn(t._randbelow(100), range(100))
        self.assertIn(t.choice("abcdef"), "abcdef")
        with self.assertRaises(NotImplementedError):
            t.seed(1)

    def test_inject_random_tool_instance(self):
        from cos_comparison.interface.tools.random_tool import (DelegatedRandomTool,
                                                                RandomTool)
        inner = RandomTool()
        inner.seed(11)
        t = DelegatedRandomTool(inner)
        self.assertIn(t._weighted_index([1, 2, 3]), range(3))
        self.assertIn(t.randint(0, 100), range(0, 101))

    def test_inject_minimal_object(self):
        from cos_comparison.interface.tools.random_tool import DelegatedRandomTool

        class OnlyRandom:
            def random(self):
                return 0.5

        t = DelegatedRandomTool(OnlyRandom())
        self.assertEqual(t.random(), 0.5)
        with self.assertRaises(NotImplementedError):
            t._randbelow(10)
        with self.assertRaises(NotImplementedError):
            t._weighted_index([1, 1])
        with self.assertRaises(NotImplementedError):
            t.seed()

    def test_missing_slots_raise(self):
        from cos_comparison.interface.tools.random_tool import DelegatedRandomTool
        t = DelegatedRandomTool()
        with self.assertRaises(NotImplementedError):
            t.random()
        with self.assertRaises(NotImplementedError):
            t._randbelow(10)
        with self.assertRaises(NotImplementedError):
            t._weighted_index([1, 1])
        with self.assertRaises(NotImplementedError):
            t.seed()

    def test_repr(self):
        from cos_comparison.interface.tools.random_tool import DelegatedRandomTool
        self.assertIn("DelegatedRandomTool", repr(DelegatedRandomTool()))


class TestSingletons(unittest.TestCase):
    def test_default_and_secure_singletons(self):
        from cos_comparison.interface.tools.random_tool import (RandomTool,
                                                                SecureRandomTool,
                                                                default, secure)
        self.assertIsInstance(default, RandomTool)
        self.assertIsInstance(secure, SecureRandomTool)

    def test_no_recursion_on_large_inputs(self):
        import sys
        from cos_comparison.interface.tools.random_tool import RandomTool
        t = RandomTool()
        big = list(range(5000))
        t.shuffle(big)
        self.assertEqual(len(set(big)), 5000)
        s = t.sample(big, 3000)
        self.assertEqual(len(set(s)), 3000)
        self.assertIsNotNone(sys.getrecursionlimit())


if __name__ == "__main__":
    unittest.main()