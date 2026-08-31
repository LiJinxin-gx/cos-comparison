"""
test_behavior_agent.py — Behavior-composition memory Agent framework tests.

Offline tests: external behaviors use an injected mock module
(tests.sample_behaviors), validating the von Neumann closed loop
(behavior-as-data) / dynamic configuration / composition / parallelism /
reflective loading.
"""
import os
import sys
import tempfile
import unittest

from cos_comparison.brain_layer.control import ControlFlowDriver, Sequence

# the agent reflects on "tests.sample_behaviors"; the sample module also
# lives in explore/ - register the alias (portable, no cwd dependency)
import importlib as _importlib

from behavior_agent import (
    Agent,
    BehaviorComposer,
    BehaviorExecutor,
    BehaviorMemory,
    run_parallel,
)

sys.modules["tests.sample_behaviors"] = _importlib.import_module(
    "sample_behaviors")


def _tmp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.remove(path)
    return path


class TestBehaviorMemory(unittest.TestCase):
    """Memory layer: behavior-as-data (von Neumann stored-program)."""

    def test_store_and_get(self):
        mem = BehaviorMemory(":memory:")
        mem.store("add", "primitive", "tests.sample_behaviors.add",
                  {"a": "num", "b": "num"}, {"b": 1}, ["math"])
        b = mem.get("add")
        self.assertEqual(b["kind"], "primitive")
        self.assertEqual(b["target"], "tests.sample_behaviors.add")
        self.assertEqual(b["defaults"], {"b": 1})
        self.assertTrue(b["enabled"])
        mem.close()

    def test_dynamic_configure(self):
        """Dynamic reconfiguration: change DB config only, no code change."""
        mem = BehaviorMemory(":memory:")
        mem.store("f", "primitive", "tests.sample_behaviors.one",
                  {}, {}, [])
        self.assertTrue(mem.get("f")["enabled"])
        mem.configure("f", "enabled", 0)
        self.assertFalse(mem.get("f")["enabled"])
        mem.configure("f", "target", "tests.sample_behaviors.two")
        self.assertEqual(mem.get("f")["target"],
                         "tests.sample_behaviors.two")
        mem.configure("f", "defaults", {"k": 9})
        self.assertEqual(mem.get("f")["defaults"], {"k": 9})
        mem.close()

    def test_recall_by_tags(self):
        mem = BehaviorMemory(":memory:")
        mem.store("a", "primitive", "x.y", {}, {}, ["search", "web"])
        mem.store("b", "primitive", "x.z", {}, {}, ["store"])
        found = dict(mem.recall(tags=["search"]))
        self.assertIn("a", found)
        self.assertNotIn("b", found)
        mem.close()


class TestBehaviorExecutor(unittest.TestCase):
    """Execution layer: fetch/decode/reflective-execute/write-back."""

    def setUp(self):
        self.path = _tmp_db()
        self.mem = BehaviorMemory(self.path)

    def tearDown(self):
        self.mem.close()
        if os.path.exists(self.path):
            os.remove(self.path)

    def test_execute_primitive(self):
        """Reflectively load external behavior module + execute + write
        history back."""
        self.mem.store("add", "primitive", "tests.sample_behaviors.add",
                       {"a": "num", "b": "num"}, {"b": 1}, ["math"])
        ex = BehaviorExecutor(self.mem)
        self.assertTrue(ex.register("add"))
        self.assertIn("add", ex.dispatch.dict)
        r = ex.execute("add", {"a": 2})
        self.assertEqual(r, 3)
        stats, (n_runs, n_ok) = self.mem.stats()
        self.assertEqual(n_runs, 1)
        self.assertEqual(n_ok, 1)
        usage = {r[0]: r[2] for r in stats}["add"]
        self.assertEqual(usage, 1)

    def test_execute_unknown_and_disabled(self):
        ex = BehaviorExecutor(self.mem)
        self.assertIn("error", ex.execute("nope"))
        self.mem.store("f", "primitive", "tests.sample_behaviors.one",
                       {}, {}, [], enabled=0)
        ex.register("f")
        r = ex.execute("f")
        self.assertEqual(r, {"skipped": "f", "reason": "disabled"})

    def test_dynamic_reroute_target(self):
        """Changing target swaps implementation (no code change): 1 -> 2."""
        self.mem.store("f", "primitive", "tests.sample_behaviors.one",
                       {}, {}, [])
        ex = BehaviorExecutor(self.mem)
        ex.register("f")
        self.assertEqual(ex.execute("f"), 1)
        self.mem.configure("f", "target", "tests.sample_behaviors.two")
        ex.register("f")  # re-register via reflection
        self.assertEqual(ex.execute("f"), 2)

    def test_composite_sequence(self):
        """Composite behavior: execute sub-behaviors in DB step order."""
        for name, target in [("one", "tests.sample_behaviors.one"),
                             ("two", "tests.sample_behaviors.two")]:
            self.mem.store(name, "primitive", target, {}, {}, [])
        self.mem.store("seq", "composite", "", {},
                       {"steps": ["one", "two"]}, ["composite"])
        ex = BehaviorExecutor(self.mem)
        ex.register_all()
        r = ex.execute("seq", {})
        self.assertEqual(r, {"one": 1, "two": 2})


class TestAgent(unittest.TestCase):
    """High-level Agent: assembly + control flow + parallelism + monitoring."""

    def test_agent_lifecycle(self):
        path = _tmp_db()
        agent = Agent(path)
        agent.register_behavior("add", "tests.sample_behaviors.add",
                                {"a": "num", "b": "num"}, {"b": 1}, ["math"])
        self.assertEqual(agent.executor.execute("add", {"a": 4}), 5)
        agent.set_steps("pipe", ["add"])
        self.assertIn(("pipe", "composite"),
                      agent.memory.recall(tags=["composite"]))
        agent.close()
        os.remove(path)

    def test_parallel_runner(self):
        """Parallel execution (ExecuterDriver delegates to workers)."""
        path = _tmp_db()
        mem = BehaviorMemory(path)
        mem.store("add", "primitive", "tests.sample_behaviors.add",
                  {"a": "num", "b": "num"}, {"b": 1}, [])
        ex = BehaviorExecutor(mem)
        ex.register("add")
        funcs = [lambda n=n: ex.execute("add", {"a": n})
                 for n in (1, 2, 3)]
        results = run_parallel(ex, funcs)
        self.assertEqual(results, [2, 3, 4])
        mem.close()
        os.remove(path)

    def test_control_flow_driver(self):
        flow = ControlFlowDriver()
        flow.append(Sequence([lambda: "s1", lambda: "s2"]))
        calls = [f() for f in flow]
        self.assertEqual(calls, ["s1", "s2"])

    def test_import_scope(self):
        import tests.sample_behaviors as SB
        self.assertEqual(SB.one(), 1)


class TestPromptDriven(unittest.TestCase):
    """Generalization evolution: prompt-generated behaviors / external
    config / ordinary + specified tasks."""

    def setUp(self):
        self.mem = BehaviorMemory(":memory:")
        self.composer = BehaviorComposer(self.mem)

    def tearDown(self):
        self.mem.close()

    def test_prompt_plan_maps_stages(self):
        """External prompt -> behavior sequence (generate logic, no code
        change)."""
        plan = self.composer.prompt_plan(
            "搜索收集数据并推理分析观点, 最后生成论文", "report_task")
        self.assertEqual(plan["stages"], ["collect", "reason", "report"])
        self.assertEqual(plan["steps"],
                         ["web_search", "fetch_resource", "extract_fields",
                          "aggregate_data", "generate_report"])
        b = self.mem.get("report_task")
        self.assertEqual(b["kind"], "composite")

    def test_prompt_plan_plain_task(self):
        """Ordinary task: collect and summarize only (no report stage)."""
        plan = self.composer.prompt_plan("数据收集 汇总", "plain_task")
        self.assertEqual(plan["stages"], ["collect"])
        self.assertEqual(plan["steps"],
                         ["web_search", "fetch_resource", "extract_fields"])

    def test_load_config(self):
        """External config file -> behavior definitions stored (no code
        change)."""
        n = self.composer.load_config({
            "behaviors": [
                {"name": "web_search", "target":
                 "tests.sample_behaviors.add", "defaults": {"b": 1},
                 "tags": ["search"]},
                {"name": "fetch_resource", "target":
                 "tests.sample_behaviors.two", "tags": ["fetch"]},
            ]})
        self.assertEqual(n, 2)
        self.assertEqual(self.mem.get("web_search")["defaults"], {"b": 1})
        self.assertTrue(self.mem.get("fetch_resource")["enabled"])

    def test_prompt_then_execute(self):
        """Prompt-generated behaviors are executable (fixed code runs
        diverse logic)."""
        self.composer.load_config({"behaviors": [
            {"name": "web_search", "target": "tests.sample_behaviors.add",
             "defaults": {"b": 1}},
            {"name": "fetch_resource", "target": "tests.sample_behaviors.two"},
            {"name": "extract_fields",
             "target": "tests.sample_behaviors.pipe", "defaults": {"k": 2}},
            {"name": "aggregate_data", "target": "tests.sample_behaviors.one"},
            {"name": "generate_report", "target": "tests.sample_behaviors.two"},
        ]})
        self.composer.prompt_plan("搜索 分析 生成论文", "task_a")
        ex = BehaviorExecutor(self.mem)
        ex.register_all()
        r = ex.execute("task_a", {"web_search": {"a": 2},
                                  "extract_fields": {"x": 0}})
        self.assertEqual(r["web_search"], 3)
        self.assertEqual(r["fetch_resource"], 2)
        self.assertEqual(r["extract_fields"], 0)

    def test_dual_task_same_code(self):
        """Same fixed code executes ordinary task + specified task (only the
        prompt differs)."""
        config = {"behaviors": [
            {"name": "web_search", "target": "tests.sample_behaviors.add",
             "defaults": {"b": 1}},
            {"name": "fetch_resource", "target": "tests.sample_behaviors.two"},
            {"name": "extract_fields",
             "target": "tests.sample_behaviors.pipe", "defaults": {"k": 2}},
            {"name": "aggregate_data", "target": "tests.sample_behaviors.one"},
            {"name": "generate_report", "target": "tests.sample_behaviors.two"},
        ]}
        self.composer.load_config(config)
        self.composer.prompt_plan("数据收集", "normal_task")
        self.composer.prompt_plan("数据收集 推理分析 生成论文", "special_task")
        ex = BehaviorExecutor(self.mem)
        ex.register_all()
        normal = ex.execute("normal_task", {})
        special = ex.execute("special_task", {})
        self.assertEqual(list(normal.keys()),
                         ["web_search", "fetch_resource", "extract_fields"])
        self.assertEqual(list(special.keys()),
                         ["web_search", "fetch_resource", "extract_fields",
                          "aggregate_data", "generate_report"])


class TestGenericProgram(unittest.TestCase):
    """Generic executor: atomic instructions (func, args, kwargs) stored in
    DB and executed."""

    def setUp(self):
        self.mem = BehaviorMemory(":memory:")
        self.ex = BehaviorExecutor(self.mem)

    def tearDown(self):
        self.mem.close()

    def test_program_atomic_instructions(self):
        """Triple instruction sequence: positional args + keyword args +
        result references."""
        self.mem.store_program("arith", [
            ("tests.sample_behaviors.add", (10, 3), {}),
            ("tests.sample_behaviors.add", (), {"a": "$0", "b": 5}),
            ("tests.sample_behaviors.pipe", ("$1",), {"k": 10}),
        ])
        regs = self.ex.run_program("arith")
        self.assertEqual(regs[0], 13)
        self.assertEqual(regs[1], 18)
        self.assertEqual(regs[2], 180)     # pipe(18, k=10)
        self.assertEqual(regs["$last"], 180)

    def test_program_composite_embed(self):
        """Composite embedding: program:<sub> instruction recursively
        executes (diverse embedding)."""
        self.mem.store_program("double_inc", [
            ("tests.sample_behaviors.add", ("$in", 1), {}),
            ("tests.sample_behaviors.add", ("$0", 1), {}),
        ])
        self.mem.store_program("outer", [
            ("program:double_inc", (), {"input": "$in"}),
        ])
        regs = self.ex.run_program("outer", {"in": 5})
        self.assertEqual(regs[0][0], 6)
        self.assertEqual(regs[0][1], 7)

    def test_program_plugin_style(self):
        """Plugin-style call: any module function (func reflection) + error
        isolation."""
        self.mem.store_program("plug", [
            ("tests.sample_behaviors.one", (), {}),
            ("tests.no_such_module.func", (), {}),
        ])
        regs = self.ex.run_program("plug")
        self.assertEqual(regs[0], 1)
        self.assertIn("error", regs[1])
        self.assertIn("ModuleNotFoundError", regs[1]["error"])


if __name__ == "__main__":
    unittest.main()
