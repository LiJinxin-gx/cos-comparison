"""SuperParallel tests: grid-style layered parallel framework and the
default process-thread model (scale/call decoupling, getIdx hierarchy,
delegating slots, thread + process paths)."""

import unittest

from cos_comparison.interface.api.parallel_api import (
    DefaultParallel,
    Layer,
    SuperParallel,
)


def _proc_kernel(q):
    """Pickleable process-path kernel (top-level)."""
    L = SuperParallel.getIdx()
    q.put((L[0].index, L[1].index))


class TestSuperParallelFramework(unittest.TestCase):
    def test_scale_setup_and_grid(self):
        sp = SuperParallel()
        sp[2, 3]
        coords = list(sp.grid(2))
        self.assertEqual(coords, [(0, 0), (0, 1), (0, 2),
                                  (1, 0), (1, 1), (1, 2)])
        self.assertEqual(next(sp.grid(1)), (0,))
        self.assertEqual(len(list(sp.grid(1))), 6)
        with self.assertRaises(ValueError):
            sp.grid(3)  # dim >= 2 needs matching setup dimension

    def test_getidx_default_empty(self):
        sp = SuperParallel()
        self.assertEqual(sp.getIdx(), ())

    def test_sequential_batch_and_getidx(self):
        seen = []

        def k():
            L = SuperParallel.getIdx()
            seen.append(L[0].index)
        sp = SuperParallel()
        sp[3].set_kernel(k)
        n = sp()
        self.assertEqual(n, 3)
        self.assertEqual(seen, [0, 1, 2])
        self.assertEqual(sp.getIdx(), ())

    def test_kernel_required(self):
        sp = SuperParallel()
        with self.assertRaises(RuntimeError):
            sp()

    def test_delegation_replacement(self):
        calls = []

        def my_batch(par, args, kwargs):
            calls.append((args, kwargs))
            return 7
        sp = SuperParallel(batch=my_batch)
        sp[10].set_kernel(lambda: None)
        self.assertEqual(sp("a", x=1), 7)
        self.assertEqual(calls[0][0], ("a",))
        self.assertEqual(calls[0][1], {"x": 1})
        sp.batch = lambda p, a, k: 1  # direct assignment
        self.assertEqual(sp(), 1)

    def test_layer_object(self):
        layer = Layer("thread", 3, 8)
        self.assertEqual(layer.name, "thread")
        self.assertEqual(layer.index, 3)
        self.assertEqual(layer.size, 8)


class TestDefaultParallel(unittest.TestCase):
    def test_thread_path_two_dim(self):
        got = []

        def k():
            L = SuperParallel.getIdx()
            got.append((L[0].index, L[1].index))
        sp = DefaultParallel()
        sp[1, 4].set_kernel(k)
        n = sp()
        self.assertEqual(n, 4)
        self.assertEqual(sorted(got), [(0, i) for i in range(4)])

    def test_one_dim_auto_expansion(self):
        # one-dimensional setups expand into processes (kernel must be
        # pickleable) - collect the (process, thread) coordinates
        import multiprocessing as mp
        mgr = mp.Manager()
        q = mgr.Queue()
        sp = DefaultParallel()
        sp[5].set_kernel(_proc_kernel)
        sp(q)
        got = []
        for _ in range(5):
            got.append(q.get(timeout=30))
        self.assertEqual(len(got), 5)
        # every unit reports its own process layer (t is 0 when the
        # expansion fits one thread unit per process)
        self.assertTrue(all(t == 0 for _p, t in got) or
                        len({p for p, _ in got}) > 1)


class TestDefaultParallelProcess(unittest.TestCase):
    def test_process_thread_hierarchy(self):
        import multiprocessing as mp
        mgr = mp.Manager()
        q = mgr.Queue()
        sp = DefaultParallel()
        sp[2, 2].set_kernel(_proc_kernel)
        sp(q)
        got = []
        for _ in range(4):
            got.append(q.get(timeout=30))
        self.assertEqual(sorted(got), [(0, 0), (0, 1), (1, 0), (1, 1)])


if __name__ == "__main__":
    unittest.main()
