# -*- coding: utf-8 -*-
"""action_layer.executer: Action executor with delegated async execution."""

import time
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
        executor = ExecuterDriver()

        def add(a, b):
            return a + b

        result = executor.call(add, 3, 4)
        self.assertEqual(result.out, 7)
        self.assertIsNone(result.err)

    def test_run_function_with_error(self):
        executor = ExecuterDriver()

        def failing_func():
            raise ValueError("boom")

        result = executor.call(failing_func)
        self.assertIsNone(result.out)
        self.assertIsInstance(result.err, ValueError)


class TestExecuterDriverBatch(unittest.TestCase):
    """ExecuterDriver batch calls."""

    def test_call_all(self):
        executor = ExecuterDriver()

        def double(x):
            return x * 2

        calls = [
            (double, (1,), {}),
            (double, (2,), {}),
            (double, (3,), {}),
        ]

        results = executor.call_all(calls)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].out, 2)
        self.assertEqual(results[1].out, 4)
        self.assertEqual(results[2].out, 6)


if __name__ == "__main__":
    unittest.main()
