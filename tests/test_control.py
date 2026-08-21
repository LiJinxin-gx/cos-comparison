"""
Tests for brain_layer.control: flat control-flow containers, iterative
flattening (ControlFlatten), and the composable ControlFlowDriver.
Non-GUI, dependency-light (brain layer + stdlib).
"""

import unittest

from cos_comparison.brain_layer.control import (
    Control, ControlFlowDriver, ControlFlatten, default_flattener,
    Sequence, Branch, Loop)


def _run(flow):
    """Execute a flattened flow via ControlFlatten (caller-execution contract)."""
    calls = []
    for func in default_flattener(flow):
        calls.append(func())
    return calls


class TestSequence(unittest.TestCase):
    def test_sequence_protocol_maps_to_func_chain(self):
        f1 = lambda: 1
        f2 = lambda: 2
        seq = Sequence([f1, f2])
        self.assertEqual(len(seq), 2)
        self.assertIs(seq[0], f1)
        self.assertIs(seq[1], f2)
        seq[1] = f1
        self.assertIs(seq[1], f1)
        self.assertEqual(list(seq), [f1, f1])

    def test_flat_iteration(self):
        calls = []
        seq = Sequence([lambda: calls.append("a"), lambda: calls.append("b")])
        for func in seq:
            func()
        self.assertEqual(calls, ["a", "b"])


class TestBranch(unittest.TestCase):
    def test_branch_mapping_keys(self):
        t = lambda: True
        f = lambda: "hit"
        b = Branch([(t, f)], else_func=lambda: "miss")
        self.assertEqual(b["if_chain"], [(t, f)])
        self.assertEqual(b["else_func"](), "miss")
        b["else_func"] = lambda: "other"
        self.assertEqual(b["else_func"](), "other")
        with self.assertRaises(KeyError):
            b["nope"]

    def test_branch_selection(self):
        calls = _run(Branch([(lambda: False, lambda: "a")], else_func=lambda: "b"))
        self.assertEqual(calls, ["b"])

    def test_branch_trigger_error_skipped(self):
        def bad_trigger():
            raise RuntimeError("boom")

        calls = _run(Branch([(bad_trigger, lambda: "a")], else_func=lambda: "b"))
        self.assertEqual(calls, ["b"])


class TestLoop(unittest.TestCase):
    def test_loop_mapping_keys(self):
        do = lambda: False
        loop = Loop(do, lambda: "x")
        self.assertIs(loop["do_func"], do)
        loop["exec_func"] = lambda: "y"
        self.assertEqual(loop["exec_func"](), "y")
        with self.assertRaises(KeyError):
            loop["nope"]

    def test_loop_expansion(self):
        state = [0]

        def do():
            return state[0] < 3

        calls = _run(Loop(do, lambda: state.__setitem__(0, state[0] + 1) or state[0]))
        self.assertEqual(calls, [1, 2, 3])


class TestControlPlaceholder(unittest.TestCase):
    def test_control_is_pure_marker(self):
        self.assertEqual(Control.__slots__, ())
        self.assertFalse(hasattr(Control, "__iter__"))
        self.assertFalse(hasattr(Control, "__getitem__"))
        self.assertFalse(hasattr(Control, "keys"))


class TestControlFlatten(unittest.TestCase):
    def test_nested_sequence(self):
        flow = Sequence([Sequence([lambda: 1, lambda: 2]), lambda: 3])
        self.assertEqual(_run(flow), [1, 2, 3])

    def test_branch_inside_sequence(self):
        flow = Sequence([lambda: 1, Branch([(lambda: True, lambda: 2)], else_func=lambda: 9), lambda: 3])
        self.assertEqual(_run(flow), [1, 2, 3])

    def test_loop_inside_sequence(self):
        state = [0]

        def do():
            return state[0] < 2

        flow = Sequence([lambda: 1, Loop(do, lambda: state.__setitem__(0, state[0] + 1) or state[0]), lambda: 3])
        self.assertEqual(_run(flow), [1, 1, 2, 3])

    def test_nested_loop_rechecks_inner_condition(self):
        outer = [0]
        inner_n = [0]

        def outer_do():
            return outer[0] < 2

        def inner_do():
            return inner_n[0] < 2

        def inner_exec():
            inner_n[0] += 1
            return "i"

        def outer_update():
            outer[0] += 1
            inner_n[0] = 0          # reset inner state per outer iteration

        flow = Loop(outer_do, Loop(inner_do, inner_exec), update_func=outer_update)
        calls = [c for c in _run(flow) if c is not None]
        self.assertEqual(calls, ["i", "i", "i", "i"])   # 2 outer x 2 inner

    def test_only_yields_never_calls(self):
        called = []

        def f():
            called.append(1)

        flow = Sequence([f])
        iterator = default_flattener(flow)
        next(iterator)                    # yield only
        self.assertEqual(called, [])

    def test_custom_flattener_subclass(self):
        counter = [0]

        class CountingFlatten(ControlFlatten):
            def flatten(self, control):
                counter[0] += 1
                return super().flatten(control)

        driver = ControlFlowDriver([Sequence([lambda: 1])], flattener=CountingFlatten())
        calls = [f() for f in driver]
        self.assertEqual(calls, [1])
        self.assertGreater(counter[0], 0)


class TestControlFlowDriver(unittest.TestCase):
    def test_init_to_iteration_adjustment(self):
        d = ControlFlowDriver()
        self.assertEqual(_run(d), [])
        d.append(Sequence([lambda: 1]))
        d.append(Branch([(lambda: True, lambda: 2)], else_func=lambda: 0))
        self.assertEqual(_run(d), [1, 2])

    def test_sequence_assignment_and_mapping_keys(self):
        d = ControlFlowDriver([Sequence([lambda: 1]), None])
        b = Branch([(lambda: True, lambda: 2)])
        d[1] = b
        d[1]["if_chain"] = [(lambda: True, lambda: 22)]
        self.assertEqual(_run(d), [1, 22])

    def test_multi_index_access(self):
        d = ControlFlowDriver([Sequence([lambda: 1, lambda: 2])])
        self.assertIsNotNone(d[0, 1])
        self.assertEqual(d[0][1](), 2)

    def test_multi_index_assignment(self):
        d = ControlFlowDriver([Sequence([lambda: 1, lambda: 2])])
        d[0, 1] = lambda: 99
        self.assertEqual(_run(d), [1, 99])

    def test_nested_driver(self):
        inner = ControlFlowDriver([Sequence([lambda: 1])])
        d = ControlFlowDriver([inner, lambda: 2])
        self.assertEqual(_run(d), [1, 2])

    def test_iteration_is_flattened_and_non_callable(self):
        d = ControlFlowDriver([Sequence([lambda: 5])])
        funcs = [f for f in d]
        self.assertEqual(len(funcs), 1)
        self.assertEqual(funcs[0](), 5)


if __name__ == "__main__":
    unittest.main()
