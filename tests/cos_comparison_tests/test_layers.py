# -*- coding: utf-8 -*-
"""Upper layers (everything outside core): data / sense / memory / brain /
interface / action / generate / test_tool behaviour.

Known upper-layer defects are pinned down in TestKnownUpperBugs with
assertions on the *current* (broken) behaviour, each carrying a short
explanation - the same documentation-as-test convention used for
TestKnownDivergences in test_core_backends.py.
"""

import sys
import time
import unittest

from cos_comparison import core

from . import common


def make(shape, default=0.0):
    return core.create_void_list(shape, default=default)


class TestDataLayer(unittest.TestCase):
    def test_datawrap_rw(self):
        from cos_comparison.data import DataWrap
        v = make((2, 2), default=1.0)
        dw = DataWrap(v)
        self.assertEqual(dw[(0, 0)], 1.0)
        dw[(1, 1)] = 7.0
        self.assertEqual(v[1, 1], 7.0)
        self.assertEqual(dw[1, 1], 7.0)

    def test_datawrap_call_getattr(self):
        from cos_comparison.data import DataWrap
        dw = DataWrap(make((3,), default=0.5))
        self.assertEqual(dw.call("__len__"), 3)
        self.assertEqual(dw.getattr("tensor_size"), (3,))
        self.assertEqual(dw.getattr("dimension"), 1)

    def test_tensor_subclass(self):
        from cos_comparison.data.tensor import Tensor
        t = Tensor(data=[[1, 2], [3, 4]])
        self.assertEqual(t.shape, (2, 2))
        self.assertEqual(t[0, 1], 2.0)
        self.assertEqual(t[1, 0], 3.0)

    def test_safe_tensor(self):
        from cos_comparison.data.tensor import SafeTensor
        t = SafeTensor(data=[[1.0, 2.0], [3.0, 4.0]])
        t[0, 0] = 9.0
        self.assertEqual(t[0, 0], 9.0)
        self.assertEqual(t.shape, (2, 2))

    def test_parallel_tensor(self):
        from cos_comparison.data.tensor import ParallelTensor
        t = ParallelTensor(data=[[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(t.shape, (2, 2))
        t[1, 1] = 8.0
        self.assertEqual(t[1, 1], 8.0)

    def test_parallel_tensor_shape_mismatch_rejected(self):
        from cos_comparison.data.tensor import ParallelTensor
        with self.assertRaises(ValueError):
            ParallelTensor(data=[[1.0, 2.0], [3.0, 4.0]], shape=(3, 3))

    def test_parallel_tensor_requires_shape_without_data(self):
        from cos_comparison.data.tensor import ParallelTensor
        with self.assertRaises(TypeError):
            ParallelTensor()


class TestSenseLayer(unittest.TestCase):
    def test_receptor_initialize(self):
        from cos_comparison.sense_layer import Receptor
        r = Receptor(make((2, 2), default=2.0))
        # len() on a 2x2 tensor is the first dimension
        self.assertEqual(r.initialize(len), 2)

    def test_tensor_receptor_comparisons(self):
        from cos_comparison.sense_layer import TensorReceptor
        data = make((2, 2), default=1.0)
        tr = TensorReceptor(data)
        self.assertIs(tr.data, data)   # no read-only memoryview conversion
        ref = make((2, 2), default=1.0)
        out = tr.comparison_passive(output=ref)
        self.assertEqual(float(out[0, 0]), 1.0)
        k = make((2, 2), default=1.0)
        act = tr.comparison_active(kernel=k)
        self.assertEqual(act.shape, (1, 1))


class TestMemoryLayer(unittest.TestCase):
    def test_map_memory_save_commit(self):
        from cos_comparison.memory_layer.memory import MapMemory
        store = {}
        m = MapMemory(store)
        m.save("k", 42)
        self.assertNotIn("k", store)  # deferred until commit
        m.commit()
        self.assertEqual(store["k"], 42)

    def test_map_memory_rollback(self):
        from cos_comparison.memory_layer.memory import MapMemory
        m = MapMemory({})
        m.save("k", 1)
        m.rollback()
        m.commit()
        self.assertEqual(m.memory, {})

    def test_map_memory_nested_auto_create(self):
        from cos_comparison.memory_layer.memory import MapMemory
        m = MapMemory({})
        m.save(("a", "b"), 7, nesting=True)
        m.commit()
        self.assertEqual(m.refer(("a", "b"), nesting=True), 7)

    def test_table_memory(self):
        from cos_comparison.memory_layer.memory import TableMemory
        m = TableMemory({})
        m.save(("r", "c"), 5)
        m.commit()
        self.assertEqual(m.refer(("r", "c")), 5)

    def test_memory_wrap(self):
        from cos_comparison.memory_layer.memory import MemoryWrap
        class Box:
            pass
        box = Box()
        w = MemoryWrap(box, level=1)
        w.set("x", 3.0)
        self.assertEqual(w.get("x"), 3.0)
        self.assertEqual(w.level, 1)
        self.assertEqual(w.process(lambda obj, mul: obj.x * mul, (2,)), 6.0)

    def test_database_memory_basic(self):
        from cos_comparison.memory_layer.memory import DatabaseMemory
        db = DatabaseMemory()
        self.assertIsNotNone(db.cursor)
        db.execute("CREATE TABLE t (a INTEGER)")
        db.commit()
        self.assertEqual(db.cursor.execute("SELECT * FROM t").fetchall(), [])
        db.close()


class TestBrainLogic(unittest.TestCase):
    def test_event_bind(self):
        from cos_comparison.brain_layer.logic import event_bind
        eb = event_bind("e1")
        self.assertEqual(eb.bind("A", 0.5), 0)
        self.assertTrue(eb.bind_exist("A"))
        self.assertEqual(eb.get_bind("A"), 0.5)
        self.assertEqual(eb.unbind("A"), 0)
        self.assertFalse(eb.bind_exist("A"))

    def test_event_bind_rejects_bad_probability(self):
        from cos_comparison.brain_layer.logic import event_bind
        with self.assertRaises(ValueError):
            event_bind().bind("A", 1.5)

    def test_event_context(self):
        from cos_comparison.brain_layer.logic import event_context
        ec = event_context()
        ec.add_bind([("A", "B", 0.3)])
        self.assertEqual(ec.bind_probability("A", "B"), 0.3)
        # unknown pair falls back to the default function -> 0
        self.assertEqual(ec.bind_probability("X"), 0)

    def test_logic_bind(self):
        from cos_comparison.brain_layer.logic import Logic_bind, Logic
        lb = Logic_bind("r", "s", status=Logic.TRUE)
        self.assertTrue(bool(lb))
        self.assertEqual(len(lb), 4)


class TestBrainReflex(unittest.TestCase):
    def test_trigger_with_stack(self):
        # the shared stack holds the trigger result; args_index selects
        # stack entries fed into the callback
        from cos_comparison.brain_layer.reflex import Trigger
        def add(a, b):
            return a + b
        tr = Trigger(lambda: 3, add, args_index=(0, 1), b_res_index=2,
                     stack=[3, 5, None])
        self.assertEqual(tr.exec(), 0)
        self.assertEqual(tr.stack[2], 8)

    def test_trigger_target(self):
        from cos_comparison.brain_layer.reflex import Trigger
        def inc(x):
            return x + 1
        tr = Trigger(lambda v: v * 2, inc, args_index=(0,),
                     a_res_index=0, b_res_index=1, stack=[None, None])
        self.assertEqual(tr.target(4), 0)
        self.assertEqual(tr.stack[0], 8)
        self.assertEqual(tr.stack[1], 9)

    def test_monitor_construct_and_record(self):
        from cos_comparison.brain_layer.reflex import Monitor
        m = Monitor()
        state = []
        handle = m.add_event(lambda: state.append("p") or True, times=-1,
                             interval=0.002)
        self.assertIsNotNone(handle)
        m.run(timeout=0.03)
        self.assertGreaterEqual(m.hits(handle), 1)
        self.assertEqual(m.errors(handle), [])

    def test_monitor_times_limit(self):
        from cos_comparison.brain_layer.reflex import Monitor
        m = Monitor()
        state = []
        handle = m.add_event(lambda: state.append("p") or True,
                             times=1, interval=0.002)
        self.assertIsNotNone(handle)
        m.run(timeout=0.05)
        self.assertEqual(m.hits(handle), 1)


class TestActionLayer(unittest.TestCase):
    def test_executer_driver(self):
        from cos_comparison.action_layer import ExecuterDriver
        ed = ExecuterDriver([lambda a, b: a + b, lambda: 42])
        self.assertEqual(ed.call(0, (1, 2)), 3)
        self.assertEqual(ed.call(1), 42)


class TestInterfaceTools(unittest.TestCase):
    def test_void_context(self):
        from cos_comparison.interface.tools import VoidContext
        with VoidContext() as c:
            pass

    def test_integrate_context_clean_exit(self):
        from cos_comparison.interface.tools import IntegrateContext
        entered = []
        class C:
            def __init__(self, n):
                self.n = n
            def __enter__(self):
                entered.append(self.n)
                return self
            def __exit__(self, *a):
                entered.append(-self.n)
                return False
        with IntegrateContext(C(1), C(2)) as ic:
            self.assertEqual(entered, [1, 2])
        self.assertEqual(entered, [1, 2, -2, -1])

    def test_no_done(self):
        from cos_comparison.interface.tools import no_done
        self.assertIsNone(no_done(1, 2, k=3))


class TestInterfaceApi(unittest.TestCase):
    def test_run_in_thread(self):
        from cos_comparison.interface.api import run_in_thread
        box = []
        run_in_thread(lambda: box.append(1), is_join=True)
        self.assertEqual(box, [1])

    def test_share_load_array(self):
        from cos_comparison.interface.api import share_array, load_array
        a = load_array("d", [1.0, 2.0, 3.0])
        self.assertEqual(list(a), [1.0, 2.0, 3.0])
        b = share_array("d", 2)
        b[0] = 5.0
        self.assertEqual(b[0], 5.0)

    def test_call_dict(self):
        from cos_comparison.interface.api import CallDict
        cd = CallDict()
        cd.add("sq", lambda x: x * x)
        self.assertEqual(cd.call("sq", (5,)), 25)
        self.assertEqual(cd.call("sq", (), kwargs={"x": 6}), 36)

    def test_module_call_container(self):
        from cos_comparison.interface.api import Module_CallContain
        m = Module_CallContain("math")
        self.assertEqual(m.call("sqrt", (16.0,)), 4.0)

    def test_async_runner_short_run(self):
        from cos_comparison.interface.api import AsyncRunner
        import asyncio
        ar = AsyncRunner()
        ran = []
        async def ev():
            ran.append("x")
        h = ar.add_event(ev())
        ar.run(timeout=0.05)
        self.assertEqual(ran, ["x"])
        self.assertTrue(ar.done(h))

    def test_command_quick_exit(self):
        from cos_comparison.interface.api import command
        out, err, code = command([sys.executable, "-c", "print('ok')"])
        self.assertEqual(code, 0)
        self.assertIn(b"ok", out)

    def test_command_no_deadlock_on_stdin_reader(self):
        # a child waiting on stdin must finish promptly: communicate
        # closes stdin (EOF), so input() returns immediately without
        # hanging the caller; EOFError -> exit code 1
        from cos_comparison.interface.api import command
        out, err, code = command(
            [sys.executable, "-c", "input()"], timeout=5)
        self.assertEqual(code, 1)
        self.assertIn(b"EOFError", err)

    def test_command_timeout_escalates(self):
        # timeout= protects against long-running children
        import subprocess
        from cos_comparison.interface.api import command
        with self.assertRaises(subprocess.TimeoutExpired):
            command([sys.executable, "-c",
                     "import time; time.sleep(30)"], timeout=1)

    def test_command_writes_stdin(self):
        from cos_comparison.interface.api import command
        out, err, code = command(
            [sys.executable, "-c",
             "import sys; sys.stdout.write(sys.stdin.readline())"],
            input=b"hello\n")
        self.assertEqual(code, 0)
        self.assertIn(b"hello", out)

    def test_process_write_and_read(self):
        from cos_comparison.interface.api import Process
        p = Process(sys.executable, ["-c",
                     "import sys; sys.stdout.write('hello')"])
        time.sleep(0.3)
        self.assertIn(b"hello", p.get_stdout())
        p.stop()
        p.terminate()


class TestGenerateTestTool(unittest.TestCase):
    def test_generator_fix(self):
        from cos_comparison.generate_layer import Generator
        g = Generator([1, 2, 3])
        self.assertEqual(g.fix(len), 3)
        self.assertEqual(g.fix(list.append, (4,)), None)

    def test_timer(self):
        from cos_comparison.test_tool import Timer
        t = Timer()
        t.mark()
        time.sleep(0.01)
        self.assertGreaterEqual(t.get_time(), 0.0)
        t.reset()
        self.assertEqual(t.total_time, 0.0)


class TestAbstractBases(unittest.TestCase):
    """Every ABC enforces abstract implementations (v0.4.0)."""

    def test_base_memory_abstract(self):
        from cos_comparison.memory_layer.memory import (BaseMemory, Memory,
                                                        MapMemory)
        with self.assertRaises(TypeError):
            BaseMemory()
        self.assertIsInstance(Memory({}), Memory)
        self.assertIsInstance(MapMemory({}), MapMemory)

    def test_base_communicate_abstract(self):
        from cos_comparison.interface.api import (BaseCommunicate,
                                                  Communicate,
                                                  PIPECommunicate)
        with self.assertRaises(TypeError):
            BaseCommunicate()
        self.assertIsInstance(Communicate(), Communicate)
        self.assertIsInstance(PIPECommunicate(), PIPECommunicate)

    def test_base_map_abstract(self):
        from cos_comparison.brain_layer.mapper import BaseMap, Map
        with self.assertRaises(TypeError):
            BaseMap()
        self.assertIsInstance(Map(), Map)

    def test_base_docker_abstract(self):
        from cos_comparison.app import BaseDocker
        with self.assertRaises(TypeError):
            BaseDocker()


class TestNoDoneSource(unittest.TestCase):
    """no_done is imported from the core module everywhere (v0.4.0)."""

    def test_no_done_single_source(self):
        from cos_comparison import core
        from cos_comparison.interface.api import communicate_api
        from cos_comparison.memory_layer.memory import basememory
        from cos_comparison.brain_layer.logic import symbol_logic
        from cos_comparison.brain_layer.mapper import base_map
        self.assertIs(communicate_api.no_done, core.no_done)
        self.assertIs(basememory.no_done, core.no_done)
        self.assertIs(symbol_logic.no_done, core.no_done)
        self.assertIs(base_map.no_done, core.no_done)


class TestKnownUpperBugs(unittest.TestCase):
    """Upper-layer behaviour tests (0.4.0).

    All known upper-layer defects are fixed; every test below asserts the
    fixed/correct behaviour.
    """

    def test_variable_eq(self):
        from cos_comparison.brain_layer.logic import Variable
        self.assertEqual(Variable("x", 1.0), Variable("x", 1.0))
        self.assertNotEqual(Variable("x", 1.0), Variable("x", 2.0))
        v = Variable("x", 1.0)
        self.assertEqual(v, v)

    def test_atomic_proposition_arg_names(self):
        from cos_comparison.brain_layer.logic import (Atomic_proposition,
                                                      UnsupportedError)
        a = Atomic_proposition(1, 2, arg_names=("a", "b"))
        self.assertEqual(a.keys(), ("a", "b"))
        self.assertEqual(a["a"], 1)
        a["a"] = 9
        self.assertEqual(a["a"], 9)
        with self.assertRaises(UnsupportedError):
            Atomic_proposition(1, 2)["x"]

    def test_logic_context_defaults(self):
        from cos_comparison.brain_layer.logic import Logic_context
        ctx = Logic_context()
        self.assertIsNone(ctx.logic_judge(1, 2))   # no_done fallback
        self.assertIsNone(ctx.initialize())

    def test_map_accessors(self):
        from cos_comparison.brain_layer.mapper import Map
        m = Map()
        self.assertIsNone(m[1])          # no_done fallback, no NameError
        self.assertFalse(1 in m)
        m2 = Map(map_obj={1: "a"}, map_func=lambda obj, k: obj.get(k))
        self.assertEqual(m2[1], "a")

    def test_funcwrap(self):
        from cos_comparison.brain_layer.mapper import FuncWrap
        self.assertEqual(FuncWrap(len)("ignored", [1, 2]), 2)

    def test_receptor_point(self):
        from cos_comparison.sense_layer import Receptor
        v = make((2, 2), default=1.0)
        r = Receptor(v)
        self.assertEqual(r.point((1, 1)), 1.0)   # tuple -> 2D index
        v1 = make((3,), default=4.0)
        self.assertEqual(Receptor(v1).point(1), 4.0)  # scalar -> 1D index

    def test_database_memory_refer_default(self):
        from cos_comparison.memory_layer.memory import DatabaseMemory
        db = DatabaseMemory()
        try:
            self.assertIsNone(db.refer())  # no_done fallback, no TypeError
        finally:
            db.close()

    def test_integrate_context_rollback_only_entered(self):
        # only contexts that successfully entered get __exit__ on rollback
        from cos_comparison.interface.tools import IntegrateContext
        calls = []
        class C:
            def __init__(self, n):
                self.n = n
            def __enter__(self):
                calls.append(("enter", self.n))
                return self
            def __exit__(self, *a):
                calls.append(("exit", self.n))
                return False
        class Boom(C):
            def __enter__(self):
                calls.append(("enter", self.n))
                raise ValueError("boom")
        with self.assertRaises(ValueError):
            IntegrateContext(C(1), Boom(2), C(3)).__enter__()
        self.assertEqual(calls, [("enter", 1), ("enter", 2),
                                 ("exit", 1)])

    def test_tensor_generator_set_point(self):
        from cos_comparison.generate_layer import TensorGenerator
        tg = TensorGenerator(make((2, 2), default=0.0))
        tg.set_point((1, 1), 9.0)
        self.assertEqual(tg.data[1, 1], 9.0)

    def test_api_process_not_shadowed(self):
        from cos_comparison.interface import api
        from cos_comparison.interface.api import system_api
        self.assertIs(api.Process, system_api.Process)


class TestFix20260812(unittest.TestCase):
    """Regression tests for the 2026-08-12 bugfix round: C backend no_done
    parity, async context rollback, logic hash/ne contracts, memory id
    slots, tensor shape passthrough."""

    def test_no_done_accepts_kwargs_on_all_backends(self):
        from cos_comparison import core
        initial = core.get_mode()
        try:
            for b in ("cos_comparison_pydll", "cos_comparison_c",
                      "cos_comparison"):
                core.set_mode(b)
                self.assertIsNone(core.no_done(a=3, b=2))
                self.assertIsNone(core.no_done(1, 2))
        finally:
            for name in initial:
                try:
                    core.set_mode(name)
                    break
                except Exception:
                    pass

    def test_variable_hash_matches_equality(self):
        from cos_comparison.brain_layer.logic import Variable
        v1 = Variable("a", 5)
        v2 = Variable("b", 5)
        self.assertEqual(v1, v2)
        self.assertEqual(hash(v1), hash(v2))
        self.assertNotEqual(hash(Variable("x", 1)), hash(Variable("x", 2)))

    def test_event_ne_complementary(self):
        from cos_comparison.brain_layer.logic import (UnionEvent,
                                                      IntersectionEvent)
        u = UnionEvent(1, 2, 3)
        i = IntersectionEvent(1, 2, 3)
        self.assertFalse(u == i)
        self.assertTrue(u != i)
        self.assertTrue(u == u)
        self.assertFalse(u != u)
        self.assertNotEqual(u, i)

    def test_memory_wrap_id_slot(self):
        from cos_comparison.memory_layer.memory import (MemoryWrap,
                                                        MemoryWrapPool)
        m = MemoryWrap(memory_body={}, name="n", level=1)
        self.assertIsNone(m.id)
        p = MemoryWrapPool(name="p", level=0)
        self.assertIsNone(p.id)

    def test_async_integrate_context_rollback_only_entered(self):
        import asyncio
        from cos_comparison.interface.tools import AsyncIntegrateContext
        calls = []
        class C:
            def __init__(self, n):
                self.n = n
            async def __aenter__(self):
                calls.append(("enter", self.n))
                return self
            async def __aexit__(self, *a):
                calls.append(("exit", self.n))
                return False
        class Boom(C):
            async def __aenter__(self):
                calls.append(("enter", self.n))
                raise ValueError("boom")
        async def main():
            with self.assertRaises(ValueError):
                async with AsyncIntegrateContext(C(1), Boom(2), C(3)):
                    pass
        asyncio.run(main())
        self.assertEqual(calls, [("enter", 1), ("enter", 2), ("exit", 1)])

    def test_tensor_explicit_shape_reshapes(self):
        from cos_comparison.data.tensor import Tensor, SafeTensor
        t = Tensor(data=[1.0, 2.0, 3.0, 4.0], shape=(2, 2))
        self.assertEqual(t.shape, (2, 2))
        self.assertEqual(t[0, 1], 2.0)
        s = SafeTensor(data=[1.0, 2.0, 3.0, 4.0], shape=(2, 2))
        self.assertEqual(s.shape, (2, 2))
        with self.assertRaises(ValueError):
            Tensor(data=[1.0, 2.0, 3.0], shape=(2, 2))
        base = Tensor(data=[[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(base.shape, (2, 2))   # no-shape path unchanged


class TestFix20260812b(unittest.TestCase):
    """Regression tests for the second 2026-08-12 round: PIPECommunicate
    auto-mode fd handling and dead-code cleanup (no behavioural change)."""

    def test_pipe_communicate_auto_creates_pipe(self):
        import os
        from cos_comparison.interface.api import PIPECommunicate
        pc = PIPECommunicate(auto=True)
        self.assertIsInstance(pc.obj, int)      # real fd
        self.assertIsInstance(pc.target, int)
        self.assertGreaterEqual(pc.obj, 0)
        self.assertGreaterEqual(pc.target, 0)
        os.write(pc.target, b"x")
        self.assertEqual(os.read(pc.obj, 1), b"x")
        os.close(pc.obj)
        os.close(pc.target)

    def test_pipe_communicate_explicit_fd_zero(self):
        # fd 0 is a valid descriptor; auto mode must not treat it as missing
        from cos_comparison.interface.api import PIPECommunicate
        pc = PIPECommunicate(read_fd=0, auto=True)
        self.assertEqual(pc.obj, 0)
        self.assertIsNotNone(pc.target)

    def test_flat_to_window_cleanup_no_regression(self):
        # dead locals removed; behaviour of the window builder is unchanged
        from cos_comparison.core import cos_comparison as cc
        self.assertEqual(cc._flat_to_window([1, 2, 3, 4], (2, 2)),
                         [[1, 2], [3, 4]])
        self.assertEqual(cc._flat_to_window([1, 2, 3, 4], (4,)),
                         [1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
