"""
test_behavior_agent.py — Behavior-composition memory Agent framework tests.

Offline tests: external behaviors use an injected inline mock module,
validating the von Neumann closed loop (behavior-as-data) / dynamic
configuration / composition / parallelism / reflective loading.
"""
import os
import sys
import tempfile
import types
import unittest

from cos_comparison.brain_layer.control import ControlFlowDriver, Sequence

# Register explore/ on the path (portable, no cwd dependency)
_EXPLORE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "explore")
if _EXPLORE_DIR not in sys.path:
    sys.path.insert(0, _EXPLORE_DIR)

from behavior_agent import (
    Agent,
    BehaviorComposer,
    BehaviorExecutor,
    BehaviorMemory,
    run_parallel,
)

# --- Inline mock behavior module (replaces deleted sample_behaviors.py) ---
_sample = types.ModuleType("tests.sample_behaviors")

def _add(a=0, b=0):
    return a + b

def _one():
    return 1

def _two():
    return 2

def _pipe(x, k=1):
    return x * k

_sample.add = _add
_sample.one = _one
_sample.two = _two
_sample.pipe = _pipe
sys.modules["tests.sample_behaviors"] = _sample


def _tmp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.remove(path)
    return path


class TestBehaviorMemory(unittest.TestCase):
    """Memory layer: behavior-as-data (von Neumann stored-program)."""

    def setUp(self):
        self.mem = BehaviorMemory(":memory:")

    def test_store_and_get(self):
        self.mem.store("add", "primitive", "tests.sample_behaviors.add",
                       {"a": "num", "b": "num"}, {"b": 0}, ["math"])
        b = self.mem.get("add")
        self.assertEqual(b["kind"], "primitive")
        self.assertEqual(b["target"], "tests.sample_behaviors.add")

    def test_configure(self):
        self.mem.store("f", "primitive", "tests.sample_behaviors.one",
                       {}, {}, [])
        self.mem.configure("f", "target", "tests.sample_behaviors.two")
        b = self.mem.get("f")
        self.assertEqual(b["target"], "tests.sample_behaviors.two")

    def test_list_behaviors(self):
        for name, target in [("one", "tests.sample_behaviors.one"),
                             ("two", "tests.sample_behaviors.two")]:
            self.mem.store(name, "primitive", target, {}, {}, [])
        rows = self.mem.list()
        names = [r[0] for r in rows]
        self.assertIn("one", names)
        self.assertIn("two", names)


class TestBehaviorExecutor(unittest.TestCase):
    """Executor: reflection-based loading + execution."""

    def setUp(self):
        self.mem = BehaviorMemory(":memory:")
        self.executor = BehaviorExecutor(self.mem)

    def test_execute_primitive(self):
        self.mem.store("add", "primitive", "tests.sample_behaviors.add",
                       {"a": "num", "b": "num"}, {"b": 1}, ["math"])
        self.executor.register_all()
        result = self.executor.execute("add", {"a": 10, "b": 3})
        self.assertEqual(result, 13)

    def test_execute_with_defaults(self):
        self.mem.store("add", "primitive", "tests.sample_behaviors.add",
                       {"a": "num", "b": "num"}, {"b": 5}, ["math"])
        self.executor.register_all()
        result = self.executor.execute("add", {"a": 7})
        self.assertEqual(result, 12)

    def test_custom_loader(self):
        called = []
        def custom_loader(target):
            called.append(target)
            return _sample.add
        self.mem.store("add", "primitive", "custom.add", {}, {}, [])
        executor = BehaviorExecutor(self.mem, loader_func=custom_loader)
        executor.register_all()
        result = executor.execute("add", {"a": 1, "b": 2})
        self.assertEqual(result, 3)
        self.assertIn("custom.add", called)


class TestBehaviorComposer(unittest.TestCase):
    """Composer: composite behavior = sub-behavior sequence."""

    def setUp(self):
        self.mem = BehaviorMemory(":memory:")
        self.composer = BehaviorComposer(
            self.mem,
            stage_behaviors={
                "fetch": ["one", "two"],
                "process": ["add"],
            },
            stage_keywords={
                "fetch": ("search", "find", "lookup"),
                "process": ("compute", "calculate", "sum"),
            },
        )

    def test_compose(self):
        self.composer.compose("pipeline", ["one", "two", "add"])
        b = self.mem.get("pipeline")
        self.assertEqual(b["kind"], "composite")
        self.assertEqual(b["defaults"]["steps"], ["one", "two", "add"])

    def test_prompt_plan(self):
        result = self.composer.prompt_plan(
            "search and compute the result", "task1")
        self.assertEqual(result["name"], "task1")
        self.assertIn("fetch", result["stages"])
        self.assertIn("process", result["stages"])

    def test_load_config(self):
        config = {
            "behaviors": [
                {"name": "x", "target": "tests.sample_behaviors.one"},
                {"name": "y", "target": "tests.sample_behaviors.two"},
            ]
        }
        n = self.composer.load_config(config)
        self.assertEqual(n, 2)


class TestAgent(unittest.TestCase):
    """High-level Agent integration."""

    def test_register_and_execute(self):
        agent = Agent(":memory:")
        agent.register_behavior("add", "tests.sample_behaviors.add",
                                {"a": "num", "b": "num"}, {"b": 0})
        agent.executor.register_all()
        result = agent.executor.execute("add", {"a": 5, "b": 7})
        self.assertEqual(result, 12)

    def test_composite_execution(self):
        agent = Agent(":memory:",
                      stage_behaviors={"calc": ["one", "add"]},
                      stage_keywords={"calc": ("sum",)})
        agent.register_behavior("one", "tests.sample_behaviors.one", {}, {})
        agent.register_behavior("add", "tests.sample_behaviors.add",
                                {"a": "num", "b": "num"}, {"b": 1})
        agent.executor.register_all()
        agent.composer.prompt_plan("sum", "composite1",
                                   sub_names=["one", "add"])
        result = agent.executor.execute("composite1", {})
        self.assertIsNotNone(result)


class TestRunParallel(unittest.TestCase):
    """Parallel execution helper."""

    def test_parallel_basic(self):
        mem = BehaviorMemory(":memory:")
        executor = BehaviorExecutor(mem)
        funcs = [lambda: 1, lambda: 2, lambda: 3]
        results = run_parallel(executor, funcs)
        self.assertEqual(sorted(results), [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
