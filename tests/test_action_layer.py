# -*- coding: utf-8 -*-
"""action_layer.executer: Action executor with delegated async execution."""

import unittest

from cos_comparison.action_layer.executer import (
    ActionResult,
    ExecuterDriver,
)


class TestActionResult(unittest.TestCase):
    """ActionResult data class."""

    def test_default_values(self):
        result = ActionResult()
        self.assertIsNone(result.out)
        self.assertIsNone(result.err)

    def test_with_values(self):
        result = ActionResult(out=42, err=None)
        self.assertEqual(result.out, 42)
        self.assertIsNone(result.err)

    def test_with_error(self):
        err = ValueError("test error")
        result = ActionResult(out=None, err=err)
        self.assertIsNone(result.out)
        self.assertIs(result.err, err)


class TestExecuterDriverBasic(unittest.TestCase):
    """ExecuterDriver basic functionality."""

    def test_creation(self):
        executor = ExecuterDriver()
        self.assertIsNotNone(executor)

    def test_run_sync_function(self):
        executor = ExecuterDriver([lambda a, b: a + b])
        self.assertEqual(executor.call(0, (3, 4)), 7)

    def test_run_function_with_error(self):
        def failing():
            raise ValueError("boom")

        executor = ExecuterDriver([failing])
        with self.assertRaises(ValueError):
            executor.call(0)
        self.assertEqual(executor.call_list[0], failing)


class TestExecuterDriverBatch(unittest.TestCase):
    """ExecuterDriver batch calls: ordered background execution."""

    def test_call_all_captures_results(self):
        executor = ExecuterDriver([lambda x: x * 2, lambda x: x * 2,
                                   lambda x: x * 2])
        executor.call_all(args=(2,))
        self.assertTrue(executor.wait(timeout=5))
        self.assertTrue(executor.is_done())
        self.assertEqual(len(executor.results), 3)
        self.assertEqual([executor.out(i) for i in range(3)], [4, 4, 4])
        self.assertIsNone(executor.err(0))

    def test_call_all_error_is_captured(self):
        def fail():
            raise ValueError("boom")

        executor = ExecuterDriver([fail, lambda: 1])
        executor.call_all()
        self.assertTrue(executor.wait(timeout=5))
        self.assertIsInstance(executor.err(0), ValueError)
        self.assertIsNone(executor.out(0))
        self.assertEqual(executor.out(1), 1)

    def test_call_all_kwargs_forwarded(self):
        executor = ExecuterDriver([lambda a, b=0: a + b])
        executor.call_all(args=(1,), kwargs={"b": 41})
        self.assertTrue(executor.wait(timeout=5))
        self.assertEqual(executor.out(0), 42)

    def test_clear(self):
        executor = ExecuterDriver([lambda: 5])
        executor.call_all()
        self.assertTrue(executor.wait(timeout=5))
        self.assertEqual(executor.out(0), 5)
        executor.clear()
        self.assertIsNone(executor.out(0))


if __name__ == "__main__":
    unittest.main()
