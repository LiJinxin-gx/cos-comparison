# -*- coding: utf-8 -*-
"""Guard: the package and this test suite must never touch GUI frameworks
(tkinter / Qt / wx / ...).  Acceptance must not pop windows on shared
devices.

The legacy GUI drivers (test_image_gui.pyw and the GUI shell error
tests) were intentionally removed from the suite for this reason; GUI
behaviour is out of scope for automated verification.
"""

import importlib
import os
import sys
import unittest

MODULES = [
    "cos_comparison",
    "cos_comparison.core",
    "cos_comparison.interface",
    "cos_comparison.interface.api",
    "cos_comparison.interface.tools",
    "cos_comparison.interface.api.system_api",
    "cos_comparison.interface.api.call_api",
    "cos_comparison.interface.api.communicate_api",
    "cos_comparison.interface.api.parallel_api",
    "cos_comparison.interface.api.async_api",
    "cos_comparison.interface.api.database_api",
    "cos_comparison.interface.tools.context_tool",
    "cos_comparison.interface.tools.random_tool",
    "cos_comparison.interface.tools.math_tool",
    "cos_comparison.data",
    "cos_comparison.data.tensor",
    "cos_comparison.sense_layer",
    "cos_comparison.memory_layer",
    "cos_comparison.memory_layer.memory",
    "cos_comparison.brain_layer",
    "cos_comparison.brain_layer.logic",
    "cos_comparison.brain_layer.reflex",
    "cos_comparison.brain_layer.mapper",
    "cos_comparison.brain_layer.control",
    "cos_comparison.interface.tools.func_tool",
    "cos_comparison.action_layer",
    "cos_comparison.generate_layer",
    "cos_comparison.extension_layer",
    "cos_comparison.test_tool",
    "cos_comparison.app",
]

GUI_FRAMEWORKS = (
    "tkinter", "customtkinter", "PyQt5", "PyQt6", "PySide2", "PySide6",
    "wx", "PySimpleGUI", "pygame", "easygui",
)


class TestNoGUI(unittest.TestCase):
    def test_package_imports_no_gui_framework(self):
        for mod in MODULES:
            importlib.import_module(mod)
        for gui in GUI_FRAMEWORKS:
            self.assertNotIn(gui, sys.modules,
                             "%s was imported by the package" % gui)

    def test_suite_sources_reference_no_gui(self):
        import pathlib
        here = pathlib.Path(__file__).parent
        offenders = []
        for f in sorted(here.glob("test_*.py")):
            if f.name == os.path.basename(__file__):
                continue  # this file enumerates the framework names itself
            src = f.read_text(encoding="utf-8")
            if any(g in src for g in GUI_FRAMEWORKS):
                offenders.append(f.name)
        self.assertEqual(offenders, [],
                         "GUI references in test sources: %r" % offenders)

    def test_suite_never_references_legacy_gui_driver(self):
        # the legacy test_image_gui.pyw may still sit in tests/ (it is a
        # user-owned driver); the suite must never import or call it
        import pathlib
        here = pathlib.Path(__file__).parent
        offenders = []
        for f in sorted(here.glob("test_*.py")):
            if f.name == os.path.basename(__file__):
                continue  # this file enumerates the framework names itself
            src = f.read_text(encoding="utf-8")
            if "test_image_gui" in src or "image_gui" in src:
                offenders.append(f.name)
        self.assertEqual(offenders, [],
                         "suite files referencing the GUI driver: %r"
                         % offenders)


if __name__ == "__main__":
    unittest.main()
