# -*- coding: utf-8 -*-
"""Every subpackage imports cleanly and exports its documented API.
Non-GUI: importing modules must never create windows."""
import importlib
import unittest

import cos_comparison
from cos_comparison import core

import testutil

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
    "cos_comparison.interface.tools.math_tool.topology",
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


class TestImports(unittest.TestCase):
    def setUp(self):
        self.pkg_path = testutil.check_local_env()

    def test_all_modules_import(self):
        failed = []
        for mod in MODULES:
            try:
                importlib.import_module(mod)
            except Exception as exc:  # pragma: no cover - diagnostic
                failed.append("%s: %s: %s" % (mod, type(exc).__name__, exc))
        self.assertEqual(failed, [], "import failures: %r" % failed)

    def test_version_consistency(self):
        v = cos_comparison.__version__.strip()
        self.assertRegex(v, r"^\d+\.\d+\.\d+$", v)
        self.assertTrue(cos_comparison.version_tuple[:3],
                        cos_comparison.version_tuple)

    def test_core_exports(self):
        for name in ("create_void_list", "load_as_default_data",
                     "infer_shape", "vector_map_as_tensor",
                     "cos_comparison_passive", "cos_comparison_active",
                     "cos", "mean_local", "local_variance",
                     "multiple_chain", "add_chain", "get_item",
                     "set_item", "default_contain", "func_name_space",
                     "get_mode", "set_mode", "get_available_backends"):
            self.assertTrue(callable(getattr(core, name, None)), name)

    def test_interface_exports(self):
        from cos_comparison.interface import api
        for name in ("EventLoop", "AsyncRunner", "run_in_thread",
                     "thread_lock", "process_lock", "parallel_lock",
                     "make_parallel_lock", "share_array", "load_array",
                     "Communicate", "SocketCommunicate",
                     "Module_CallContain", "DatabaseToolWrap"):
            self.assertTrue(hasattr(api, name), name)

    def test_data_exports(self):
        from cos_comparison.data import DataWrap
        from cos_comparison.data.tensor import (BaseTensor, Tensor,
                                                SafeTensor, ParallelTensor)
        self.assertTrue(callable(DataWrap))
        self.assertTrue(callable(Tensor))
        self.assertTrue(callable(SafeTensor))
        self.assertTrue(callable(ParallelTensor))
        self.assertTrue(callable(BaseTensor))

    def test_sense_layer_exports(self):
        from cos_comparison.sense_layer import Receptor, TensorReceptor
        self.assertTrue(callable(Receptor))
        self.assertTrue(callable(TensorReceptor))

    def test_memory_layer_exports(self):
        from cos_comparison.memory_layer.memory import (Memory, BaseMemory,
                                                        MapMemory,
                                                        TableMemory,
                                                        DatabaseMemory,
                                                        MemoryWrap,
                                                        MemoryWrapPool,
                                                        MemoryWrapMap)
        for cls in (Memory, BaseMemory, MapMemory, TableMemory,
                    DatabaseMemory, MemoryWrap, MemoryWrapPool,
                    MemoryWrapMap):
            self.assertTrue(callable(cls), cls)

    def test_brain_layer_exports(self):
        from cos_comparison.brain_layer import logic
        for name in ("Variable", "Atomic_proposition", "Logic_bind",
                     "Logic_context", "UnionEvent", "IntersectionEvent",
                     "GlobalEvent", "event_bind", "event_context"):
            self.assertTrue(hasattr(logic, name), name)
        from cos_comparison.brain_layer import reflex
        for name in ("Monitor", "Trigger", "default_maintainer"):
            self.assertTrue(hasattr(reflex, name), name)
        from cos_comparison.brain_layer import mapper
        for name in ("BaseMap", "Map", "FuncWrap"):
            self.assertTrue(hasattr(mapper, name), name)

    def test_action_generate_testtool_exports(self):
        from cos_comparison.action_layer import ExecuterDriver
        from cos_comparison.generate_layer import Generator, TensorGenerator
        from cos_comparison.test_tool import Timer
        for cls in (ExecuterDriver, Generator, TensorGenerator, Timer):
            self.assertTrue(callable(cls), cls)


if __name__ == "__main__":
    unittest.main()
