"""
Tests for interface.tools.func_tool: ComposalFunction (stack-shared
composition), FuncHelper (stdlib helper slots), FuncWrap.
Stdlib only, non-GUI.
"""

import functools
import unittest

from cos_comparison.interface.tools.func_tool import (
    ComposalFunction, ComposalFunctionManage, FuncHelper, FuncWrap)


class TestComposalFunction(unittest.TestCase):
    def test_chain_via_stack(self):
        first = lambda a, b: a + b            # -> slot 0
        step1 = (lambda x: x * 2, (0,), {})   # reads slot 0 -> slot 1
        step2 = (lambda x, y: x + y, (0, 1), {})  # reads 0,1 -> slot 2
        last = lambda z: z + 1, (2,), {}      # reads slot 2

        cf = ComposalFunction(first, last[0], steps=[step1, step2],
                              last_args_index=last[1])
        self.assertEqual(cf(2, 3), (2 + 3) * 2 + (2 + 3) + 1)

    def test_direct_first_last(self):
        cf = ComposalFunction(lambda a: a + 1, lambda x: x * 10,
                              last_args_index=(0,))
        self.assertEqual(cf(4), 50)

    def test_kwargs_index(self):
        first = lambda a: a * 2                 # -> slot 0
        last = (lambda v: v - 1), (), {"v": 0}  # keyword from slot 0
        cf = ComposalFunction(first, last[0], last_kwargs_index=last[2])
        self.assertEqual(cf(5), 9)

    def test_steps_res_start(self):
        first = lambda: 7                        # -> slot 5
        step = (lambda x: x + 1, (5,), {})       # reads 5 -> slot 2
        last = lambda y: y, (2,), {}
        cf = ComposalFunction(first, last[0], steps=[step],
                              first_res_index=5, steps_res_start=2,
                              last_args_index=(2,))
        self.assertEqual(cf(), 8)

    def test_shared_stack_injected(self):
        stack = [None] * 4
        first = lambda: 1
        last = lambda x: x, (0,), {}
        cf = ComposalFunction(first, last[0], stack=stack, last_args_index=(0,))
        self.assertEqual(cf(), 1)
        self.assertEqual(stack[0], 1)

    def test_owned_stack_is_fresh_per_call(self):
        cf = ComposalFunction(lambda n: n, lambda x: x, last_args_index=(0,))
        self.assertEqual(cf(1), 1)
        self.assertEqual(cf(2), 2)              # no residue from previous call

    def test_long_chain_without_recursion(self):
        first = lambda: 0
        steps = [(lambda x: x + 1, (i,), {}) for i in range(100)]
        last = lambda x: x, (100,), {}
        cf = ComposalFunction(first, last[0], steps=steps, last_args_index=(100,))
        self.assertEqual(cf(), 100)

    def test_delegated_manager(self):
        events = []

        class CustomManage(ComposalFunctionManage):
            def place_result(self, stack, index, value):
                events.append((index, value))
                return super().place_result(stack, index, value)

        cf = ComposalFunction(lambda: 1, lambda x: x, last_args_index=(0,),
                              manager=CustomManage())
        self.assertEqual(cf(), 1)
        self.assertEqual(events, [(0, 1)])


class TestFuncHelper(unittest.TestCase):
    def test_defaults_from_stdlib(self):
        helper = FuncHelper()
        p = helper.partial_func(lambda a, b: a + b, 10)
        self.assertEqual(p(5), 15)
        self.assertEqual(helper.reduce_func(lambda a, b: a + b, [1, 2, 3]), 6)
        c = helper.compose_func(lambda x: x + 1, lambda x: x * 2)
        self.assertEqual(c(5), 11)              # f(g(x)): (5*2)+1
        self.assertEqual(helper.itemgetter_func(1)([9, 8, 7]), 8)
        self.assertEqual(helper.attrgetter_func("real")(1j), 0.0)

    def test_wrap(self):
        helper = FuncHelper()

        def wrapped():
            """docstring kept"""

        @helper.wrap_func(wrapped)
        def f():
            pass
        self.assertEqual(f.__doc__, "docstring kept")

    def test_injected_replacement(self):
        def my_compose(*funcs):
            def composed(arg):
                value = arg
                for func in reversed(funcs):
                    value = func(value)
                return value
            return composed

        helper = FuncHelper(compose_func=my_compose)
        c = helper.compose_func(lambda x: x * 10)
        self.assertEqual(c(3), 30)

    def test_compose_requires_functions(self):
        helper = FuncHelper()
        with self.assertRaises(ValueError):
            helper.compose_func()


class TestFuncWrap(unittest.TestCase):
    def test_first_argument_discarded(self):
        wrap = FuncWrap(lambda x, y: x + y)
        self.assertEqual(wrap("ignored", 2, 3), 5)


if __name__ == "__main__":
    unittest.main()
