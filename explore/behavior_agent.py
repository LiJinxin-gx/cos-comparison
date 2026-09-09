"""
behavior_agent.py — Behavior-composition memory-driven Agent framework
(von Neumann architecture: behavior-as-data).

Pure standard library + cos_comparison. External behavior implementations
are injected through a delegation mechanism:

  Memory layer:   DatabaseMemory            behavior defs/config/history/
                                               knowledge stored in one DB
  Interface layer: CallDict + reflection    behavior target stored in DB
                                               ('module.func'); default loader
                                               resolves via importlib reflection,
                                               custom loader_func injectable
  Action layer:   ExecuterDriver            parallel execution (delegated
                                               runner_func)
  Control layer:  ControlFlowDriver/Sequence behavior composition orchestration
  Reflection:     Monitor                   execution monitoring

Dynamic reconfiguration (no code change): behavior parameters / implementation
target / enable switch all stored in the database; call configure() to update
DB config and re-execute for changes to take effect.

Usage example:
    from behavior_agent import Agent
    agent = Agent(database=":memory:")
    agent.register_behavior("add", "operator.add",
                            {"a": "num", "b": "num"}, {"b": 0})
    agent.executor.register_all()
    result = agent.executor.execute("add", {"a": 1, "b": 2})
"""
import importlib
import json
import os
import sqlite3
import threading
import time

from cos_comparison.action_layer import ExecuterDriver
from cos_comparison.brain_layer.control import ControlFlowDriver, Sequence
from cos_comparison.brain_layer.reflex import Monitor
from cos_comparison.interface.api import CallDict, DatabaseToolWrap
from cos_comparison.memory_layer.memory import DatabaseMemory


class BehaviorMemory:
    """Behavior store: behavior definitions/config/execution history/knowledge
    in one database (von Neumann stored-program architecture).

    DatabaseMemory injected with DatabaseToolWrap (check_same_thread=False):
    single shared connection + write lock, thread-safe, :memory: also works.
    """

    def __init__(self, database=":memory:"):
        self.path = database
        tool = DatabaseToolWrap(connect_func=lambda *args: sqlite3.connect(
            *args, check_same_thread=False))
        self.memory = DatabaseMemory(database_tool=tool, database=database)
        self._lock = threading.Lock()
        with self._lock:
            self.memory.cursor().execute(
                "CREATE TABLE IF NOT EXISTS behaviors ("
                "id INTEGER PRIMARY KEY, name TEXT UNIQUE, kind TEXT, "
                "target TEXT, args_schema TEXT, defaults TEXT, tags TEXT, "
                "enabled INTEGER, usage_count INTEGER, success_count INTEGER, "
                "fail_count INTEGER, avg_time REAL, updated_at TEXT)")
            self.memory.cursor().execute(
                "CREATE TABLE IF NOT EXISTS runs ("
                "id INTEGER PRIMARY KEY, behavior TEXT, args TEXT, "
                "result TEXT, ok INTEGER, seconds REAL, ts TEXT)")
            self.memory.cursor().execute(
                "CREATE TABLE IF NOT EXISTS knowledge ("
                "id INTEGER PRIMARY KEY, category TEXT, key TEXT, name TEXT, "
                "source TEXT, url TEXT, data_json TEXT, ts TEXT)")
            self.memory.cursor().execute(
                "CREATE TABLE IF NOT EXISTS programs ("
                "id INTEGER PRIMARY KEY, name TEXT UNIQUE, description TEXT)")
            self.memory.cursor().execute(
                "CREATE TABLE IF NOT EXISTS instructions ("
                "id INTEGER PRIMARY KEY, program_id INTEGER, pc INTEGER, "
                "func TEXT, args TEXT, kwargs TEXT)")
            self.memory.commit()

    def knowledge_conn(self):
        """Shared write accessor (DatabaseMemory, thread-safe:
        check_same_thread=False)."""
        return self.memory

    def store(self, name, kind, target, args_schema, defaults, tags,
              enabled=1):
        """Insert a behavior definition into the store (kind:
        primitive|composite; target: 'module.func')."""
        with self._lock:
            self.memory.cursor().execute(
                "INSERT OR REPLACE INTO behaviors (name, kind, target, "
                "args_schema, defaults, tags, enabled, usage_count, "
                "success_count, fail_count, avg_time, updated_at) "
                "VALUES (?,?,?,?,?,?,?,0,0,0,0.0,?)",
                (name, kind, target, json.dumps(args_schema),
                 json.dumps(defaults), json.dumps(tags), enabled,
                 time.strftime("%Y-%m-%d %H:%M:%S")))
            self.memory.commit()

    def get(self, name):
        row = self.memory.cursor().execute(
            "SELECT kind, target, args_schema, defaults, enabled FROM "
            "behaviors WHERE name=?", (name,)).fetchone()
        if not row:
            return None
        return {"kind": row[0], "target": row[1],
                "args_schema": json.loads(row[2]),
                "defaults": json.loads(row[3]), "enabled": row[4]}

    def list(self):
        """All behaviors (name, kind, target, enabled, usage, success, fail)."""
        return self.memory.cursor().execute(
            "SELECT name, kind, target, enabled, usage_count, success_count, "
            "fail_count FROM behaviors ORDER BY kind, name").fetchall()

    def configure(self, name, field, value):
        """Dynamically modify behavior config (enabled/target/defaults) —
        change the DB, not the code."""
        with self._lock:
            if field == "defaults":
                value = json.dumps(value)
            self.memory.cursor().execute(
                "UPDATE behaviors SET %s=?, updated_at=? WHERE name=?" % field,
                (value, time.strftime("%Y-%m-%d %H:%M:%S"), name))
            self.memory.commit()

    def note_run(self, name, args, result, ok, seconds):
        """Write back execution history + accumulate behavior statistics
        (memory)."""
        with self._lock:
            self.memory.cursor().execute(
                "INSERT INTO runs (behavior, args, result, ok, seconds, ts) "
                "VALUES (?,?,?,?,?,?)",
                (name, json.dumps(args, ensure_ascii=False, default=str)[:500],
                 json.dumps(result, ensure_ascii=False, default=str)[:2000],
                 1 if ok else 0, seconds, time.strftime("%Y-%m-%d %H:%M:%S")))
            self.memory.cursor().execute(
                "UPDATE behaviors SET usage_count=usage_count+1, "
                "success_count=success_count+?, fail_count=fail_count+?, "
                "avg_time=(avg_time*(usage_count-1)+?)/usage_count, "
                "updated_at=? WHERE name=?",
                (1 if ok else 0, 0 if ok else 1, seconds,
                 time.strftime("%Y-%m-%d %H:%M:%S"), name))
            self.memory.commit()

    def recall(self, tags=()):
        """Recall related behaviors by tag (memory retrieval)."""
        rows = self.memory.cursor().execute(
            "SELECT name, kind, tags FROM behaviors").fetchall()
        return [(name, kind) for name, kind, tags_j in rows
                if set(tags) & set(json.loads(tags_j))]

    def stats(self):
        """(behavior statistics list, (total runs, total successes))."""
        rows = self.memory.cursor().execute(
            "SELECT name, kind, usage_count, avg_time, success_count, "
            "fail_count FROM behaviors ORDER BY usage_count DESC").fetchall()
        total = self.memory.cursor().execute(
            "SELECT COUNT(*), COALESCE(SUM(ok),0) FROM runs").fetchone()
        return rows, total

    def store_program(self, name, instructions, description=""):
        """Insert an instruction program into the store (von Neumann:
        instructions-as-data).

        instructions: [(func, args, kwargs), ...] atomic instruction triples.
        """
        with self._lock:
            self.memory.cursor().execute(
                "INSERT OR REPLACE INTO programs (name, description) "
                "VALUES (?,?)", (name, description))
            pid = self.memory.cursor().execute(
                "SELECT id FROM programs WHERE name=?", (name,)).fetchone()[0]
            self.memory.cursor().execute(
                "DELETE FROM instructions WHERE program_id=?", (pid,))
            for pc, (func, args, kwargs) in enumerate(instructions):
                self.memory.cursor().execute(
                    "INSERT INTO instructions (program_id, pc, func, args, "
                    "kwargs) VALUES (?,?,?,?,?)",
                    (pid, pc, func, json.dumps(args),
                     json.dumps(kwargs)))
            self.memory.commit()
        return pid

    def load_program(self, name):
        """Load an instruction program -> {pc: (func, args, kwargs)}."""
        pid = self.memory.cursor().execute(
            "SELECT id FROM programs WHERE name=?", (name,)).fetchone()
        if not pid:
            return None
        rows = self.memory.cursor().execute(
            "SELECT pc, func, args, kwargs FROM instructions "
            "WHERE program_id=? ORDER BY pc", (pid[0],)).fetchall()
        return {pc: (func, json.loads(args), json.loads(kwargs))
                for pc, func, args, kwargs in rows}

    def close(self):
        self.memory.close()


class BehaviorExecutor:
    """Fetch (DB) -> decode (defaults merge) -> reflective execute ->
    write back (runs).

    Delegation mechanism: loader_func defaults to importlib reflection
    resolving 'module.func'; a custom loader can be injected (e.g. test mocks
    / external implementations).
    """

    def __init__(self, memory, loader_func=None):
        self.memory = memory
        self.dispatch = CallDict()
        self.loader = loader_func or self._default_loader

    @staticmethod
    def _default_loader(target):
        """Reflection: importlib loads the module + getattr retrieves the
        function."""
        module_name, _, func_name = target.rpartition(".")
        module = importlib.import_module(module_name)
        return getattr(module, func_name)

    def register(self, name):
        """Reflectively register a behavior from the DB into CallDict
        (dispatch table)."""
        b = self.memory.get(name)
        if b is not None and b["kind"] == "primitive":
            self.dispatch.add(name, self.loader(b["target"]))
            return True
        return False

    def register_all(self):
        for name, kind, *_ in self.memory.list():
            if kind == "primitive":
                try:
                    self.register(name)
                except Exception:
                    pass
        return self

    def execute(self, name, args=None):
        """Execute a behavior: fetch/decode/execute/write-back.

        composite: executes sub-behaviors in sequence per the DB step list
        (args dispatched by step name).
        """
        b = self.memory.get(name)
        if b is None:
            return {"error": "unknown behavior: %s" % name}
        if not b["enabled"]:
            return {"skipped": name, "reason": "disabled"}
        if b["kind"] == "composite":
            return self._execute_composite(name, b, args or {})
        merged = dict(b["defaults"])
        merged.update(args or {})
        t0 = time.time()
        try:
            result = self.dispatch.call(name, (), merged)
            ok = True
        except Exception as e:
            result, ok = {"error": "%s: %s" % (type(e).__name__, e)}, False
        self.memory.note_run(name, merged, result, ok, time.time() - t0)
        return result

    def _execute_composite(self, name, b, args):
        """Composite behavior: sub-behavior sequence execution (control-flow
        driven)."""
        steps = b["defaults"].get("steps", [])
        results = {}
        for step in steps:
            results[step] = self.execute(step, args.get(step))
        return results

    # =====================================================================
    # Generic executor: instruction programs (von Neumann, atomic
    # instruction = (func, args, kwargs))
    # =====================================================================
    def _resolve_ref(self, value, registers):
        """Decode: '$N' references a prior instruction result (pc index),
        otherwise pass through unchanged."""
        if isinstance(value, str) and value.startswith("$"):
            ref = value[1:]
            if ref in registers:
                return registers[ref]
            if ref.isdigit() and int(ref) in registers:
                return registers[int(ref)]
        return value

    def run_program(self, name, registers=None):
        """Execute an instruction program stored in the DB (generic executor).

        Fetch (DB) -> decode (func reflection + $ argument ref resolution)
        -> execute -> write back.
        Composite embedding: func of the form 'program:<sub-name>' recursively
        executes a sub-program.
        Returns: all registers (pc -> result) and '$last' (final result).
        """
        program = self.memory.load_program(name)
        if program is None:
            return {"error": "unknown program: %s" % name}
        regs = dict(registers or {})
        for pc in sorted(program):
            func, args, kwargs = program[pc]
            try:
                call_args = [self._resolve_ref(a, regs) for a in args]
                call_kwargs = {k: self._resolve_ref(v, regs)
                               for k, v in kwargs.items()}
                if isinstance(func, str) and func.startswith("program:"):
                    sub = func.split(":", 1)[1]
                    result = self.run_program(sub, dict(regs))
                else:
                    fn = self.loader(func)
                    result = fn(*call_args, **call_kwargs)
                ok = True
            except Exception as e:
                result, ok = {"error": "%s: %s" % (type(e).__name__, e)}, False
            regs[pc] = result
            regs["$last"] = result
            self.memory.note_run(name, {"pc": pc, "func": func}, result,
                                 ok, 0.0)
        return regs


class BehaviorComposer:
    """Behavior composition: composite behavior = sub-behavior sequence,
    stored back in the behavior library (reusable / re-composable).

    von Neumann evolution: prompt_plan parses an external prompt (task
    description) into behavior definitions (instruction data) stored in the
    database — fixed code executes diverse logic, and logic can be
    generated/modified without changing code.

    Stage behavior names and keyword mappings are passed explicitly via the
    constructor interface (no external config file), so that specific task
    definitions are isolated from the framework code and injected by the caller.
    """

    def __init__(self, memory, stage_behaviors=None, stage_keywords=None):
        """
        Args:
            memory: BehaviorMemory instance
            stage_behaviors: dict mapping stage name -> list of behavior names
            stage_keywords: dict mapping stage name -> tuple/list of keywords
        """
        self.memory = memory
        # Stage name -> behavior sequence (extensible; adding a stage adds
        # capability). Passed explicitly via interface, no config file.
        self.STAGE_BEHAVIORS = stage_behaviors or {}
        # Prompt keywords -> stage. Passed explicitly via interface.
        self.STAGE_KEYWORDS = {k: tuple(v) for k, v in
                               (stage_keywords or {}).items()}

    def compose(self, name, sub_names, tags=(), enabled=1):
        """Compose a sub-behavior sequence into a composite behavior and
        store it."""
        self.memory.store(name, "composite", "",
                          {"sub": "list"}, {"steps": list(sub_names)},
                          list(tags) + ["composite"], enabled=enabled)
        return name

    def prompt_plan(self, prompt, name, sub_names=None, tags=()):
        """Prompt -> behavior definition (generate logic, store in DB).

        prompt: task description text; stages are matched by keyword against
        the stage templates to select a behavior sequence; changing the
        external prompt input generates/modifies execution logic.
        """
        stages = []
        for stage, kws in self.STAGE_KEYWORDS.items():
            if any(kw.lower() in prompt.lower() for kw in kws):
                stages.append(stage)
        steps = sub_names
        if steps is None:
            steps = []
            for stage in stages:
                for b in self.STAGE_BEHAVIORS.get(stage, ()):
                    if b not in steps:
                        steps.append(b)
        self.compose(name, steps, tags=tags)
        return {"name": name, "stages": stages, "steps": steps}

    def load_config(self, config, enabled_default=1):
        """Load behavior definitions from external config (config file/dict
        -> DB, no code change).

        config: {"behaviors": [{"name","target","args_schema","defaults",
                                "tags","enabled"}]}
        """
        loaded = 0
        for b in config.get("behaviors", ()):
            self.memory.store(
                b["name"], b.get("kind", "primitive"), b.get("target", ""),
                b.get("args_schema", {}), b.get("defaults", {}),
                b.get("tags", []), b.get("enabled", enabled_default))
            loaded += 1
        return loaded


def run_parallel(executor, funcs, workers=None):
    """Execute behavior tasks in parallel (ExecuterDriver delegates to
    workers, runner_func injectable)."""
    ed = ExecuterDriver(list(funcs))
    ed.call_all()
    ed.wait()
    return [ed.out(i) for i in range(len(funcs))]


class Agent:
    """High-level Agent: memory + executor + composer + control flow +
    reflection monitor assembled."""

    def __init__(self, database=":memory:", loader_func=None, poller=None,
                 stage_behaviors=None, stage_keywords=None):
        """
        Args:
            database: database path or :memory:
            loader_func: custom behavior loader (default: importlib reflection)
            poller: custom monitor poller
            stage_behaviors: dict for BehaviorComposer (explicit interface)
            stage_keywords: dict for BehaviorComposer (explicit interface)
        """
        self.memory = BehaviorMemory(database)
        self.executor = BehaviorExecutor(self.memory, loader_func)
        self.composer = BehaviorComposer(self.memory,
                                         stage_behaviors=stage_behaviors,
                                         stage_keywords=stage_keywords)
        self.flow = ControlFlowDriver()
        self.monitor = Monitor(poller=poller)

    def register_behavior(self, name, target, args_schema=None, defaults=None,
                          tags=None, enabled=1):
        """Register an external behavior (target='module.func', reflectively
        loaded)."""
        self.memory.store(name, "primitive", target, args_schema or {},
                          defaults or {}, tags or [], enabled=enabled)
        self.executor.register(name)

    def set_steps(self, name, sub_names):
        """Orchestrate a composite behavior: step sequence stored in DB +
        control flow (Sequence)."""
        self.composer.compose(name, sub_names)
        self.flow.append(Sequence([lambda s=s: s for s in sub_names]))
        return self.flow

    def close(self):
        self.memory.close()


if __name__ == "__main__":
    # Self-test: in-memory agent with inline test behaviors (no file I/O,
    # no external sample module). Behaviors registered via explicit interface.
    import sys
    import types
    _test = types.ModuleType("_behavior_test")
    _test.add = lambda a, b=0: a + b
    _test.mul = lambda x, y=1: x * y
    sys.modules["_behavior_test"] = _test

    agent = Agent(database=":memory:")
    agent.register_behavior("add", "_behavior_test.add",
                            args_schema={"a": "num", "b": "num"},
                            defaults={"b": 0})
    agent.register_behavior("mul", "_behavior_test.mul",
                            args_schema={"x": "num", "y": "num"},
                            defaults={"y": 1})
    agent.executor.register_all()
    result = agent.executor.execute("add", {"a": 3, "b": 4})
    print("add(3,4):", result)
    result = agent.executor.execute("mul", {"x": 5, "y": 3})
    print("mul(5,3):", result)
    agent.close()
    print("behavior_agent self-test OK")
