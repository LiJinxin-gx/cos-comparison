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
        import os
        procs = min(os.cpu_count() or 1, 5)
        per = -(-5 // procs)  # ceil(5 / procs)
        expected = sorted(
            (p, t) for p in range(procs)
            for t in range(min(per, max(0, 5 - p * per))))
        with mp.Manager() as mgr:
            q = mgr.Queue()
            sp = DefaultParallel()
            sp[5].set_kernel(_proc_kernel)
            self.assertEqual(sp(q), 5)
            got = [q.get(timeout=30) for _ in range(5)]
        self.assertEqual(sorted(got), expected)

    def test_one_dim_tail_trimmed_to_n(self):
        # emulated cpu_count=2 with n=5 -> (2, 3) allocation grid; the
        # tail beyond N must not execute (exactly 5 units)
        from cos_comparison.interface.api.parallel_api import (
            _default_worker_run)
        got = []

        def k():
            L = SuperParallel.getIdx()
            got.append((L[0].index, L[1].index))

        for p in range(2):
            _default_worker_run((p, k, (), {}, 2, 3, 5))
        self.assertEqual(sorted(got), [(0, 0), (0, 1), (0, 2),
                                       (1, 0), (1, 1)])

    def test_zero_scale_runs_nothing(self):
        sp = DefaultParallel()
        sp[0].set_kernel(lambda: None)
        self.assertEqual(sp(), 0)

    def test_load_array_accepts_sets(self):
        from cos_comparison.interface.api import load_array
        arr = load_array("d", {1.0, 2.0, 3.0})
        self.assertEqual(sorted(arr), [1.0, 2.0, 3.0])


class TestDefaultParallelProcess(unittest.TestCase):
    def test_process_thread_hierarchy(self):
        import multiprocessing as mp
        with mp.Manager() as mgr:
            q = mgr.Queue()
            sp = DefaultParallel()
            sp[2, 2].set_kernel(_proc_kernel)
            sp(q)
            got = [q.get(timeout=30) for _ in range(4)]
        self.assertEqual(sorted(got), [(0, 0), (0, 1), (1, 0), (1, 1)])


if __name__ == "__main__":
    unittest.main()
