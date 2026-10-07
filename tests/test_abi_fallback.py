"""S0 ABI-migration checks: active-backend reporting and fallback guards."""

import os
import unittest

import cos_comparison.core as core


def _active_module():
    name = core.get_active_backend()
    backend = getattr(core, "_BACKENDS", {})
    if name and name in backend:
        return backend[name].get("module")
    return None


class TestActiveBackend(unittest.TestCase):

    def test_active_backend_reported_and_configured(self):
        name = core.get_active_backend()
        self.assertIn(name, core.get_available_backends())

    def test_set_mode_py_and_restore(self):
        old = core.get_active_backend()
        self.assertIsNotNone(old)
        core.set_mode("py")
        try:
            self.assertEqual(core.get_active_backend(), "py")
            self.assertTrue(callable(core.no_done))
        finally:
            core.set_mode(old)

    def test_strict_mode_requires_c(self):
        if _active_module() == ".cos_comparison":
            self.skipTest("pure Python only environment")
        old_env = os.environ.get("COS_COMPARISON_REQUIRE_C")
        old = core.get_active_backend()
        os.environ["COS_COMPARISON_REQUIRE_C"] = "1"
        try:
            with self.assertRaises(ImportError):
                core.set_mode("py")
        finally:
            if old_env is None:
                os.environ.pop("COS_COMPARISON_REQUIRE_C", None)
            else:
                os.environ["COS_COMPARISON_REQUIRE_C"] = old_env
            core.set_mode(old)
        self.assertEqual(core.get_active_backend(), old)


if __name__ == "__main__":
    unittest.main()
