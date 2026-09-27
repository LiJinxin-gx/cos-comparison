"""Core func_namespace enhancement tests: mapping protocol (**s unpacking)
and namespace creation control (use_namespace / namespace_hook) across
the backends."""

import unittest

from cos_comparison import core
from cos_comparison.core import cos_comparison as _py

BACKENDS = [".cos_comparison"]
try:
    import cos_comparison.core.cos_comparison_pydll  # noqa: F401
    BACKENDS.append(".cos_comparison_pydll")
except Exception:  # pragma: no cover - C extension unavailable
    pass


def _algo(seen):
    def algo(main, other, mu, name):
        seen.append(name)
        return mu
    return algo


class TestNamespaceControl(unittest.TestCase):
    def _run_passive(self, mod, **kw):
        seen = []
        mod.cos_comparison_passive(
            [1.0, 2.0, 3.0], algorithm=_algo(seen), **kw)
        return seen

    def test_mapping_protocol(self):
        ns = _py.func_name_space(a=1)
        ns.c = 3
        self.assertEqual(dict(ns), {"a": 1, "c": 3})
        self.assertEqual({**ns}, {"a": 1, "c": 3})
        self.assertEqual(sorted(ns), ["a", "c"])
        self.assertEqual(len(ns), 2)
        with self.assertRaises(KeyError):
            ns["missing"]

    def test_default_namespace_per_backend(self):
        for b in BACKENDS:
            with self.subTest(backend=b):
                core.set_mode([b])
                import importlib
                m = importlib.import_module(
                    "cos_comparison.core" + b)
                seen = self._run_passive(m)
                self.assertTrue(seen)
                self.assertTrue(all(n is not None for n in seen))

    def test_use_namespace_false_passes_none(self):
        for b in BACKENDS:
            with self.subTest(backend=b):
                import importlib
                m = importlib.import_module(
                    "cos_comparison.core" + b)
                seen = self._run_passive(m, use_namespace=False)
                self.assertTrue(seen)
                self.assertTrue(all(n is None for n in seen))

    def test_custom_hook(self):
        for b in BACKENDS:
            with self.subTest(backend=b):
                import importlib
                m = importlib.import_module(
                    "cos_comparison.core" + b)
                calls = []

                def hook(**kw):
                    calls.append(kw)
                    return m.func_name_space(**kw)
                m.cos_comparison_passive(
                    [1.0, 2.0], output=[0.0, 0.0],
                    algorithm=_algo([]), namespace_hook=hook)
                self.assertEqual(len(calls), 1)
                self.assertIn("output", calls[0])

    def test_active_kernel_and_disable(self):
        for b in BACKENDS:
            with self.subTest(backend=b):
                import importlib
                m = importlib.import_module(
                    "cos_comparison.core" + b)
                seen = []
                m.cos_comparison_active(
                    [1.0, 2.0, 3.0], kernel=[0.5, 0.5],
                    algorithm=_algo(seen), use_namespace=False)
                self.assertTrue(seen)
                self.assertTrue(all(n is None for n in seen))


if __name__ == "__main__":
    unittest.main()
