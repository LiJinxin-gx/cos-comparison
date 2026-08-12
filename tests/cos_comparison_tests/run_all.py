# -*- coding: utf-8 -*-
"""Runner for the cos_comparison test suite.

Usage (from anywhere, with the venv_test interpreter):
    python -E tests\\cos_comparison_tests\\run_all.py
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = os.path.dirname(HERE)
# expose the `tests` parent on sys.path so the package context of
# cos_comparison_tests is preserved during discovery; the project root
# is intentionally NOT on sys.path (it would shadow the installed
# package with the source tree).
sys.path.insert(0, TESTS_DIR)


def main():
    loader = unittest.TestLoader()
    suite = loader.discover(TESTS_DIR, pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
