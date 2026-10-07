# -*- coding: utf-8 -*-
"""extension_layer.plugin: PluginPool batch aggregation with keyed pools."""

import unittest

from cos_comparison.extension_layer.plugin import PluginPool


class TestPluginPoolBasics(unittest.TestCase):
    """Basic PluginPool creation and default pools."""

    def test_default_pools_are_lists(self):
        pool = PluginPool()
        self.assertIsInstance(pool.resources, list)
        self.assertIsInstance(pool.plugins, list)
        self.assertIsInstance(pool.func_pool, list)
        self.assertEqual(len(pool.resources), 0)
        self.assertEqual(len(pool.plugins), 0)
        self.assertEqual(len(pool.func_pool), 0)

    def test_custom_pools(self):
        custom_resources = {}
        custom_plugins = {}
        custom_funcs = {}
        pool = PluginPool(resources=custom_resources, plugins=custom_plugins,
                           func_pool=custom_funcs)
        self.assertIs(pool.resources, custom_resources)
        self.assertIs(pool.plugins, custom_plugins)
        self.assertIs(pool.func_pool, custom_funcs)


class TestResourcePool(unittest.TestCase):
    """Resource pool add/get operations."""

    def test_add_list_resource_by_index(self):
        pool = PluginPool()
        pool.add_resource(0, "data1")
        pool.add_resource(1, "data2")
        self.assertEqual(pool.get_resource(0), "data1")
        self.assertEqual(pool.get_resource(1), "data2")

    def test_add_dict_resource_by_key(self):
        pool = PluginPool(resources={})
        pool.add_resource("config", {"key": "value"})
        pool.add_resource("path", "/tmp/data")
        self.assertEqual(pool.get_resource("config"), {"key": "value"})
        self.assertEqual(pool.get_resource("path"), "/tmp/data")

    def test_duck_keyed_container(self):
        class DuckPool:
            def __init__(self):
                self.store = {}
            def __setitem__(self, key, value):
                self.store[key] = value
            def __getitem__(self, key):
                return self.store[key]

        pool = PluginPool(resources=DuckPool())
        pool.add_resource("cfg", 1)
        self.assertEqual(pool.get_resource("cfg"), 1)


class TestPluginPool(unittest.TestCase):
    """Plugin pool hosting and method calls."""

    def test_host_plugin_and_call_method(self):
        pool = PluginPool()

        class DummyPlugin:
            def __init__(self, name):
                self.name = name
                self.calls = 0

            def greet(self, greeting="Hello"):
                self.calls += 1
                return f"{greeting}, {self.name}"

            def get_name(self):
                return self.name

        plugin = DummyPlugin("TestBot")
        pool.add_plugin(0, plugin)

        result = pool.call_plugin(0, "greet")
        self.assertEqual(result, "Hello, TestBot")

        result2 = pool.call_plugin(0, "greet", "Hi")
        self.assertEqual(result2, "Hi, TestBot")
        self.assertEqual(plugin.calls, 2)

    def test_get_plugin_attr(self):
        pool = PluginPool()

        class DummyPlugin:
            version = "1.0.0"

        pool.add_plugin(0, DummyPlugin())
        self.assertEqual(pool.get_plugin_attr(0, "version"), "1.0.0")
        self.assertIs(pool.get_plugin(0), pool.plugins[0])


class TestFuncPool(unittest.TestCase):
    """Function pool direct callables."""

    def test_add_and_call_func(self):
        pool = PluginPool()

        def add(a, b):
            return a + b

        def multiply(a, b):
            return a * b

        pool.add_func(0, add)
        pool.add_func(1, multiply)

        self.assertEqual(pool.call_func(0, 3, 4), 7)
        self.assertEqual(pool.call_func(1, 3, 4), 12)

    def test_call_func_with_kwargs(self):
        pool = PluginPool()

        def greet(name, greeting="Hello"):
            return f"{greeting}, {name}"

        pool.add_func(0, greet)
        self.assertEqual(pool.call_func(0, "World"), "Hello, World")
        self.assertEqual(pool.call_func(0, "World", greeting="Hi"), "Hi, World")


class TestMixedPools(unittest.TestCase):
    """Mixed usage across all three pools."""

    def test_all_pools_together(self):
        pool = PluginPool()

        # Resources
        pool.add_resource(0, {"dataset": "mnist"})

        # Plugins
        class Recognizer:
            def predict(self, data):
                return "digit"

        pool.add_plugin(0, Recognizer())

        # Functions
        def preprocess(img):
            return img / 255.0

        pool.add_func(0, preprocess)

        # Use them together
        config = pool.get_resource(0)
        self.assertEqual(config["dataset"], "mnist")

        result = pool.call_plugin(0, "predict", [1, 2, 3])
        self.assertEqual(result, "digit")

        processed = pool.call_func(0, 255.0)
        self.assertEqual(processed, 1.0)


if __name__ == "__main__":
    unittest.main()
