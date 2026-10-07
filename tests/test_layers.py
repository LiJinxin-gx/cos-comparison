# -*- coding: utf-8 -*-
"""Upper layers (everything outside core): data / sense / memory / brain /
interface / action / generate / test_tool behaviour.  Non-GUI."""

import sys
import time
import unittest

from cos_comparison import core


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

    def test_datawrap_protocol_access(self):
        from cos_comparison.data import DataWrap
        dw = DataWrap([10, 20])
        self.assertEqual(dw.__get_item__(1), 20)  # plain indexing fallback

        class ProtocolOnly:
            def __init__(self):
                self.store = {}
            def __get_item__(self, *index):
                return self.store[index]
            def __set_item__(self, index, value):
                self.store[index] = value

        dw2 = DataWrap(ProtocolOnly())
        dw2.__set_item__((0,), 5)
        self.assertEqual(core.get_item(dw2, (0,)), 5)

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

    def test_receptor_point(self):
        from cos_comparison.sense_layer import TensorReceptor
        v = make((2, 2), default=1.0)
        r = TensorReceptor(v)
        self.assertEqual(r.point((1, 1)), 1.0)   # tuple -> 2D index (core.get_item)
        v1 = make((3,), default=4.0)
        self.assertEqual(TensorReceptor(v1).point(1), 4.0)  # scalar -> 1D index

    def test_receptor_threshold_map(self):
        from cos_comparison.sense_layer import TensorReceptor
        from cos_comparison import core
        tr = TensorReceptor([1.0, 5.0, 9.0])
        out = core.create_void_list((3,), 0.0)
        judge = core.threshold_judge(low=2.0, high=7.0)
        status = tr.threshold_map([(judge, 1.0)], default_value=0.0,
                                  output=out)
        self.assertEqual(status, 0)
        self.assertEqual(list(out), [0.0, 1.0, 0.0])

    def test_receptor_threshold_map_no_output(self):
        from cos_comparison.sense_layer import TensorReceptor
        tr = TensorReceptor([1.0])
        self.assertEqual(tr.threshold_map([], output=None), 2)

    def test_receptor_threshold_map_region(self):
        from cos_comparison.sense_layer import TensorReceptor
        from cos_comparison import core
        tr = TensorReceptor([1.0, 5.0, 9.0, 2.0])
        out = core.create_void_list((2,), 0.0)
        judge = core.threshold_judge(low=2.0, high=7.0)
        status = tr.threshold_map([(judge, 1.0)], default_value=0.0,
                                  output=out, start=(1,), shape=(2,),
                                  step=(1,))
        self.assertEqual(status, 0)
        self.assertEqual(list(out), [1.0, 0.0])

    def test_receptor_threshold_match(self):
        from cos_comparison.sense_layer import TensorReceptor
        tr = TensorReceptor([1.0, 5.0, 9.0, 3.0])
        positions = list(tr.threshold_match(low=2.0, high=7.0))
        self.assertEqual(positions, [(1,), (3,)])

    def test_receptor_threshold_match_region(self):
        from cos_comparison.sense_layer import TensorReceptor
        tr = TensorReceptor([1.0, 5.0, 9.0, 3.0, 4.0])
        positions = list(tr.threshold_match(low=2.0, high=7.0,
                                            start=(1,), shape=(2,),
                                            step=(1,)))
        self.assertEqual(positions, [(1,)])

    def test_elementwise_extract_full(self):
        from cos_comparison.sense_layer import elementwise_extract
        out = [0.0, 0.0]
        status = elementwise_extract([1.0, 2.0], [3.0, 4.0],
                                     func=lambda a, b: a + b, output=out)
        self.assertEqual(status, 0)
        self.assertEqual(out, [4.0, 6.0])

    def test_elementwise_extract_region(self):
        from cos_comparison.sense_layer import elementwise_extract
        out = [0.0, 0.0]  # read region sized (2)
        status = elementwise_extract(
            [1.0, 2.0, 3.0, 4.0], func=lambda v: v * 2,
            output=out, start=(1,), shape=(2,), step=(1,))
        self.assertEqual(status, 0)
        self.assertEqual(out, [4.0, 6.0])

    def test_elementwise_extract_no_tensors(self):
        from cos_comparison.sense_layer import elementwise_extract
        # no tensors: the underlying elementwise decides the status
        self.assertEqual(elementwise_extract(func=lambda: 0), 1)

    def test_elementwise_extract_write_region(self):
        from cos_comparison.sense_layer import elementwise_extract
        from cos_comparison import core
        out = core.create_void_list((4,), 0.0)
        status = elementwise_extract([1.0, 2.0], func=lambda v: v * 10,
                                     output=out, out_start=(1,),
                                     shape=(2,), out_step=(1,))
        self.assertEqual(status, 0)
        self.assertEqual(list(out), [0.0, 10.0, 20.0, 0.0])

    def test_elementwise_extract_shape_mismatch(self):
        from cos_comparison.sense_layer import elementwise_extract
        # output smaller than the read region -> status code, no raise
        status = elementwise_extract([1.0, 2.0, 3.0],
                                     func=lambda v: v,
                                     output=[0.0, 0.0])
        self.assertEqual(status, 1)


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

    def test_memory_falsy_callable_rules_kept(self):
        from cos_comparison.memory_layer.memory import Memory

        class FalsyRule:
            def __bool__(self):
                return False
            def __call__(self, memory, *args, **kwargs):
                return "called"

        m = Memory({}, init_func=FalsyRule())
        self.assertEqual(m.initialize(), "called")
        self.assertEqual(m.process(FalsyRule()), "called")

    def test_transaction_hashable_with_unhashable_value(self):
        from cos_comparison.memory_layer.memory import Transaction
        record = Transaction("k", [1, 2])
        self.assertIsInstance(hash(record), int)
        self.assertEqual(list(record), ["k", [1, 2], True, True])

    def test_transaction_subclass_fields_iterated(self):
        from cos_comparison.memory_layer.memory import Transaction

        class SubTransaction(Transaction):
            __slots__ = ("extra",)
            def __init__(self, key, value):
                super().__init__(key, value)
                self.extra = "e"

        self.assertEqual(list(SubTransaction("k", 1)),
                         ["k", 1, True, True, "e"])

    def test_memory_wrap(self):
        from cos_comparison.memory_layer.memory import MemoryWrap
        body = {"x": 3.0}
        w = MemoryWrap(body, level=1)
        self.assertEqual(w.get("x"), 3.0)
        w.set("y", 6.0)
        self.assertEqual(w.get("y"), 6.0)
        self.assertEqual(w.level, 1)
        self.assertEqual(w.process(lambda d, mul: d["x"] * mul, (2,)), 6.0)

    def test_database_memory_basic(self):
        from cos_comparison.memory_layer.memory import DatabaseMemory
        db = DatabaseMemory()
        self.assertIsNotNone(db.cursor())
        db.execute("CREATE TABLE t (a INTEGER)")
        db.commit()
        self.assertEqual(db.cursor().execute("SELECT * FROM t").fetchall(), [])
        db.close()

    def test_database_memory_refer_default(self):
        from cos_comparison.memory_layer.memory import DatabaseMemory
        db = DatabaseMemory()
        try:
            self.assertIsNone(db.refer())  # no_done fallback, no TypeError
        finally:
            db.close()


class TestIOStreamMemory(unittest.TestCase):
    def setUp(self):
        import os
        import tempfile
        fd, self.tmp_path = tempfile.mkstemp(suffix=".txt")
        os.close(fd)

    def tearDown(self):
        import os
        try:
            os.unlink(self.tmp_path)
        except OSError:
            pass

    def _text_mem(self):
        from cos_comparison.interface.api.io_api import IOMemory
        from cos_comparison.memory_layer.memory import IOStreamMemory
        return IOStreamMemory(IOMemory("", binary=False))

    def test_io_stream_memory_key_value_basic(self):
        m = self._text_mem()
        m.save("name", "cos")
        m.save("version", 0.4)
        with self.assertRaises(KeyError):
            m.refer("name")  # uncommitted is invisible
        m.commit()
        self.assertEqual(m.refer("name"), "cos")
        self.assertEqual(m.refer("version"), 0.4)

    def test_io_re_save_overwrites(self):
        m = self._text_mem()
        m.save("k", 1)
        m.save("k", 2)
        m.commit()
        self.assertEqual(m.refer("k"), 2)  # dict overwrite semantics
        self.assertEqual(len(m), 1)

    def test_io_rollback(self):
        m = self._text_mem()
        m.save("a", 1)
        m.rollback()
        m.commit()
        self.assertEqual(len(m), 0)
        with self.assertRaises(KeyError):
            m.refer("a")

    def test_io_arbitrary_content(self):
        m = self._text_mem()
        m.save("para", "line1\nline2\n\nChinese content")
        m.commit()
        self.assertEqual(m.refer("para"), "line1\nline2\n\nChinese content")

    def test_io_tuple_key_and_lazy_restore(self):
        from cos_comparison.interface.api.io_api import IOMemory
        from cos_comparison.memory_layer.memory import IOStreamMemory
        body = IOMemory("", binary=False)
        m = IOStreamMemory(body)
        m.save(("session", "start"), "old-record")
        m.commit()
        m2 = IOStreamMemory(body)  # fresh instance over the same stream
        self.assertEqual(m2.refer(("session", "start")), "old-record")
        m2.save(("session", "end"), "next")
        m2.commit()
        self.assertEqual(m2.refer(("session", "end")), "next")
        self.assertEqual(len(m2), 2)

    def test_io_key_must_be_hashable(self):
        m = self._text_mem()
        with self.assertRaises(TypeError):
            m.save([1, 2], "no")

    def test_io_file_persistence(self):
        from cos_comparison.interface.api.io_api import IOFile
        from cos_comparison.memory_layer.memory import IOStreamMemory
        m = IOStreamMemory(IOFile(self.tmp_path, "a+"))
        m.save("persisted", 42)
        m.commit()
        m.close()
        m2 = IOStreamMemory(IOFile(self.tmp_path, "r"))
        try:
            self.assertEqual(m2.refer("persisted"), 42)
            m2.save("blocked", 1)
            with self.assertRaises(RuntimeError):
                m2.commit()  # read-only carrier rejects writes
        finally:
            m2.close()

    def test_io_writeonly_refer_fails(self):
        from cos_comparison.interface.api.io_api import IOFile
        from cos_comparison.memory_layer.memory import IOStreamMemory
        m = IOStreamMemory(IOFile(self.tmp_path, "a"))
        m.save("append-only", 1)
        m.commit()
        with self.assertRaises(RuntimeError):
            m.refer("append-only")  # not readable
        m.close()

    def test_io_binary(self):
        from cos_comparison.interface.api.io_api import IOMemory
        from cos_comparison.memory_layer.memory import IOStreamMemory
        payload = b"\x00\x01\xffline\nnext"
        m = IOStreamMemory(IOMemory(b"", binary=True), binary=True)
        m.save("blob", payload)
        m.commit()
        self.assertEqual(m.refer("blob"), payload)

    def test_io_duck_stream_carrier(self):
        import io
        from cos_comparison.memory_layer.memory import IOStreamMemory
        m = IOStreamMemory(io.StringIO())
        m.save("duck", "value")
        m.commit()
        self.assertEqual(m.refer("duck"), "value")
        self.assertIn("duck", m)
        self.assertEqual(len(m), 1)
        m.close()

    def test_io_mapping_protocol_without_carrier(self):
        from cos_comparison.interface.api.io_api import IOMemory
        from cos_comparison.memory_layer.memory import IOStreamMemory
        open_carrier = IOStreamMemory(IOMemory("", binary=False))
        open_carrier.close()
        for m in (IOStreamMemory(None), open_carrier):
            with self.assertRaises(RuntimeError):
                len(m)
            with self.assertRaises(RuntimeError):
                list(m.keys())
            with self.assertRaises(RuntimeError):
                m.__contains__("k")

    def test_io_len_keys_contains(self):
        m = self._text_mem()
        self.assertEqual(len(m), 0)
        m.save("a", 1)
        m.save("b", 2)
        m.commit()
        self.assertEqual(len(m), 2)
        self.assertEqual(sorted(m.keys()), ["a", "b"])
        self.assertIn("a", m)
        self.assertNotIn("c", m)

    def test_io_initialize(self):
        m = self._text_mem()
        m.save("x", "y")
        m.commit()
        n = m.initialize()
        self.assertEqual(n, 1)
        self.assertEqual(m.refer("x"), "y")

    def test_io_close_commit(self):
        from cos_comparison.interface.api.io_api import IOMemory
        from cos_comparison.memory_layer.memory import IOStreamMemory
        body = IOMemory("", binary=False)
        m = IOStreamMemory(body, close_commit=True)
        m.save("auto", 1)
        m.close()
        m2 = IOStreamMemory(body)
        self.assertEqual(m2.refer("auto"), 1)

    def test_io_carrier_type_check(self):
        from cos_comparison.memory_layer.memory import IOStreamMemory
        with self.assertRaises(TypeError):
            IOStreamMemory({})

    def test_io_custom_encode_decode(self):
        # record-format extension point: keys stored raw on their own line;
        # records are NOT readable by the working-default decoder
        from cos_comparison.interface.api.io_api import IOMemory
        from cos_comparison.memory_layer.memory import IOStreamMemory

        def enc(mem, key, content):
            key_line = key + "\n"
            return key_line + f"{len(content)}\n" + content

        def dec(mem):
            stream = mem.memory
            key_line = mem.readline_func(stream)
            if not key_line:
                return None
            header = mem.readline_func(stream)
            try:
                length = int(header.rstrip("\r\n"))
            except (ValueError, AttributeError):
                return None
            content = mem.read_func(stream, length)
            if len(content) != length:
                return None
            return (key_line.rstrip("\r\n"), content)

        body = IOMemory("", binary=False)
        m = IOStreamMemory(body, encode_func=enc, decode_func=dec)
        m.save("raw", "custom-format")
        m.commit()
        m2 = IOStreamMemory(
            IOMemory(body.getvalue(), binary=False),
            encode_func=enc, decode_func=dec)
        self.assertEqual(m2.refer("raw"), "custom-format")
        # the working-default decoder must NOT read the custom format
        m3 = IOStreamMemory(IOMemory(body.getvalue(), binary=False))
        self.assertEqual(m3.initialize(), 0)

    def test_io_custom_memory_slot(self):
        # memory-level slot injection: a save_func of the caller's own
        # signature (content-only append with auto keys)
        from cos_comparison.interface.api.io_api import IOMemory
        from cos_comparison.memory_layer.memory import IOStreamMemory
        from cos_comparison.memory_layer.memory.inner_memory import (
            Transaction)
        calls = []

        def save_func(mem, content):
            calls.append(content)
            mem.cache.append(Transaction(len(mem.cache), content))

        body = IOMemory("", binary=False)
        m = IOStreamMemory(body, save_func=save_func)
        m.save("entry-a")
        m.save("entry-b")
        m.commit()
        self.assertEqual(calls, ["entry-a", "entry-b"])
        m2 = IOStreamMemory(body)
        self.assertEqual(m2.refer(0), "entry-a")
        self.assertEqual(m2.refer(1), "entry-b")

    def test_io_custom_operation_slot(self):
        # operation-slot injection: intercept every write through the slot
        from cos_comparison.interface.api.io_api import IOMemory
        from cos_comparison.memory_layer.memory import IOStreamMemory
        seen = []
        m = IOStreamMemory(IOMemory("", binary=False),
                           write_func=lambda s, d: seen.append(d)
                           or s.write(d))
        m.save("k", "slot")
        m.commit()
        self.assertTrue(any("slot" in d for d in seen))
        m2 = IOStreamMemory(IOMemory(m.memory.getvalue(), binary=False))
        self.assertEqual(m2.refer("k"), "slot")

    def test_io_custom_capability(self):
        # capability extension point: force write rejection
        from cos_comparison.interface.api.io_api import IOMemory
        from cos_comparison.memory_layer.memory import IOStreamMemory
        m = IOStreamMemory(
            IOMemory("", binary=False),
            capability_func=lambda mem, kind: kind == "read")
        m.save("blocked", 1)
        with self.assertRaises(RuntimeError):
            m.commit()
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

    def test_variable_eq_hash(self):
        from cos_comparison.brain_layer.logic import Variable
        v1 = Variable("a", 5)
        v2 = Variable("b", 5)
        self.assertEqual(v1, v2)
        self.assertEqual(hash(v1), hash(v2))
        self.assertNotEqual(hash(Variable("x", 1)), hash(Variable("x", 2)))

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
        from cos_comparison.brain_layer.logic import (Logic, Logic_context)
        ctx = Logic_context()
        # default judge is rigorous: an empty rule set determines non-truth
        # -> Logic.SURE (2, determined not-true), not a guessed False
        self.assertEqual(ctx.logic_judge(1, 2), Logic.SURE)
        self.assertIsNone(ctx.initialize())

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


class TestBrainMapper(unittest.TestCase):
    def test_map_accessors(self):
        from cos_comparison.brain_layer.mapper import Map
        m = Map()
        # default dict-protocol: missing map_obj fails loudly (no silent None)
        with self.assertRaises(TypeError):
            m[1]
        with self.assertRaises(TypeError):
            1 in m
        m2 = Map(map_obj={1: "a"}, map_func=lambda obj, k: obj.get(k))
        self.assertEqual(m2[1], "a")

    def test_funcwrap(self):
        from cos_comparison.brain_layer.mapper import FuncWrap
        self.assertEqual(FuncWrap(len)("ignored", [1, 2]), 2)

    def test_map_non_callable_keys_attr(self):
        from cos_comparison.brain_layer.mapper import Map

        class Carrier:
            def __init__(self):
                self.keys = ("a", "b")
            def __getitem__(self, key):
                return 1

        self.assertEqual(list(Map(Carrier()).keys()), [])

    def test_trigger_keyboard_interrupt_propagates(self):
        from cos_comparison.brain_layer.reflex import Trigger

        def kb():
            raise KeyboardInterrupt()

        tr = Trigger(kb, lambda: None, a_res_index=0, b_res_index=1,
                     stack=[None, None])
        with self.assertRaises(KeyboardInterrupt):
            tr.target()


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

    def test_monitor_maintain_background_carrying(self):
        from cos_comparison.brain_layer.reflex import Monitor
        m = Monitor()
        handle = m.add_event(lambda: True, times=-1, interval=0.002)
        m.maintain()  # background carrying (run_in_thread delegation)
        try:
            time.sleep(0.05)
            self.assertGreaterEqual(m.hits(handle), 1)
        finally:
            m.stop()


class TestFeedback(unittest.TestCase):
    """Trigger-style reflex hub: paired feedback objects/functions,
    non-blocking receive trigger, hub dispatch."""

    def _make(self):
        from cos_comparison.brain_layer.reflex import Feedback
        return Feedback(), {"value": 0}

    def test_register_and_receive(self):
        h, o = self._make()
        h.register(o, lambda obj: obj.__setitem__("value", 9))
        h.receive().join(timeout=5)  # non-blocking by default
        self.assertEqual(o["value"], 9)
        self.assertEqual(len(h), 1)
        self.assertTrue(h.has(o))

    def test_each_function_gets_its_paired_object(self):
        h, o1 = self._make()
        o2 = {"value": 0}
        got = []
        h.register(o1, lambda obj: got.append(id(obj)))
        h.register(o2, lambda obj: got.append(id(obj)) or obj.update(value=2))
        h.receive().join(timeout=5)
        self.assertEqual(got, [id(o1), id(o2)])
        self.assertEqual(o2["value"], 2)

    def test_unhashable_object_pairs(self):
        h, _ = self._make()
        box = []
        h.register(box, lambda obj: obj.append("hit"))
        h.receive().join(timeout=5)
        self.assertEqual(box, ["hit"])

    def test_error_isolation(self):
        h, o1 = self._make()
        o2 = {"value": 0}
        h.register(o1, lambda obj: (_ for _ in ()).throw(ValueError("x")))
        h.register(o2, lambda obj: obj.__setitem__("value", 5))
        self.assertEqual(h.dispatch(), 1)  # one failed, one succeeded
        self.assertIsInstance(h.last_error, ValueError)
        self.assertEqual(o2["value"], 5)

    def test_dispatch_keyboard_interrupt_propagates(self):
        h, o = self._make()

        def kb(obj):
            raise KeyboardInterrupt()

        h.register(o, kb)
        with self.assertRaises(KeyboardInterrupt):
            h.dispatch()

    def test_remove(self):
        h, o = self._make()
        h.register(o, lambda obj: None)
        self.assertEqual(h.remove(o), 0)
        self.assertFalse(h.has(o))
        self.assertEqual(h.remove(o), 1)

    def test_synchronous_injected_receive(self):
        # an injected hub-receive function may trigger synchronously
        h, o = self._make()
        h.register(o, lambda obj: obj.__setitem__("value", 3))
        status = []
        h.receive_func = lambda hub: status.append(hub.dispatch())
        h.receive()
        self.assertEqual(status, [0])
        self.assertEqual(o["value"], 3)

    def test_injected_dispatch(self):
        # an injected hub-feedback function replaces the delivery loop
        h, o = self._make()
        order = []
        h.register(o, lambda obj: order.append("a"))
        h.dispatch_func = lambda hub: order.append("dispatch")
        h.receive().join(timeout=5)
        self.assertEqual(order, ["dispatch"])


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
            self.assertIsInstance(c, VoidContext)

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

    def test_no_done(self):
        # no_done is only importable from core (single source); the interface
        # layer must NOT re-export it
        from cos_comparison import core
        from cos_comparison.interface import api
        self.assertIsNone(core.no_done(1, 2, k=3))
        self.assertFalse(hasattr(api, "no_done"))


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

    def test_call_container_falsy_init_func_kept(self):
        from cos_comparison.interface.api import BaseCallContainer

        class FalsyInit:
            def __bool__(self):
                return False
            def __call__(self, target):
                return "wrapped:" + target

        class Source:
            name = "raw"

        c = BaseCallContainer(Source())
        self.assertEqual(c.get_call("name", init_func=FalsyInit()),
                         "wrapped:raw")

    def test_communicate_falsy_reader_kept(self):
        from cos_comparison.interface.api import FdCommunicate

        class FalsyReader:
            def __bool__(self):
                return False
            def __call__(self, fd, size):
                return b"kept"

        comm = FdCommunicate(reader=FalsyReader())
        self.assertEqual(comm.recv(4), b"kept")

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
        out, err, code = command([sys.executable, "-c", "print('ok')"],
                                 timeout=10)
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
            input=b"hello\n", timeout=10)
        self.assertEqual(code, 0)
        self.assertIn(b"hello", out)

    def test_process_write_and_read(self):
        from cos_comparison.interface.api import Process
        p = Process(sys.executable, ["-c",
                     "import sys; sys.stdout.write('hello')"])
        try:
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                if b"hello" in p.get_stdout():
                    break
                time.sleep(0.01)
            self.assertIn(b"hello", p.get_stdout())
        finally:
            p.stop()

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
        import os
        from cos_comparison.interface.api import PIPECommunicate
        pc = PIPECommunicate(read_fd=0, auto=True)
        try:
            self.assertEqual(pc.obj, 0)
            self.assertIsInstance(pc.target, int)
        finally:
            try:
                os.close(pc.target)
            except OSError:
                pass

    def test_api_process_not_shadowed(self):
        from cos_comparison.interface import api
        from cos_comparison.interface.api import system_api
        self.assertIs(api.Process, system_api.Process)


class TestExtensionLayer(unittest.TestCase):
    """extension_layer.plugin: proactive PluginPool aggregation."""

    def test_default_pools_are_lists(self):
        from cos_comparison.extension_layer.plugin import PluginPool
        p = PluginPool()
        self.assertEqual(p.resources, [])
        self.assertEqual(p.plugins, [])
        self.assertEqual(p.func_pool, [])

    def test_add_and_get_by_index(self):
        from cos_comparison.extension_layer.plugin import PluginPool
        p = PluginPool()
        p.add_resource(0, "data")
        p.add_plugin(0, object())
        p.add_func(0, lambda x: x * 2)
        self.assertEqual(p.get_resource(0), "data")
        self.assertIsNotNone(p.get_plugin(0))
        self.assertEqual(p.call_func(0, 5), 10)

    def test_mapping_pools(self):
        from cos_comparison.extension_layer.plugin import PluginPool
        p = PluginPool(resources={}, plugins={}, func_pool={})
        p.add_resource("cfg", {"a": 1})
        p.add_func("add", lambda a, b: a + b)
        self.assertEqual(p.get_resource("cfg"), {"a": 1})
        self.assertEqual(p.call_func("add", 2, 3), 5)

    def test_call_plugin_method(self):
        from cos_comparison.extension_layer.plugin import PluginPool
        class Calc:
            def add(self, a, b):
                return a + b
            def mul(self, a, b):
                return a * b
        p = PluginPool()
        p.add_plugin(0, Calc())
        self.assertEqual(p.call_plugin(0, "add", 2, 3), 5)
        self.assertEqual(p.call_plugin(0, "mul", 4, 5), 20)
        self.assertEqual(p.get_plugin_attr(0, "mul")(4, 5), 20)

    def test_external_plugin_unchanged(self):
        from cos_comparison.extension_layer.plugin import PluginPool

        def external_tool(x, factor=2):
            return x * factor

        p = PluginPool()
        p.add_func(0, external_tool)
        self.assertEqual(p.call_func(0, 4), 8)
        self.assertEqual(p.call_func(0, 4, factor=3), 12)


class TestGenerateTestTool(unittest.TestCase):
    def test_generator_fix(self):
        from cos_comparison.generate_layer import Generator
        g = Generator([1, 2, 3])
        self.assertEqual(g.fix(len), 3)
        self.assertEqual(g.fix(list.append, (4,)), None)

    def test_tensor_generator_set_point(self):
        from cos_comparison.generate_layer import TensorGenerator
        tg = TensorGenerator(make((2, 2), default=0.0))
        tg.set_point((1, 1), 9.0)
        self.assertEqual(tg.data[1, 1], 9.0)

    def test_transform_self_full(self):
        from cos_comparison.generate_layer import TensorGenerator
        tg = TensorGenerator([1.0, 2.0, 3.0])
        status = tg.transform_self(lambda v: v * 2)
        self.assertEqual(status, 0)
        self.assertEqual(list(tg.data), [2.0, 4.0, 6.0])

    def test_transform_self_with_others(self):
        from cos_comparison.generate_layer import TensorGenerator
        tg = TensorGenerator([1.0, 2.0])
        status = tg.transform_self(lambda a, b: a + b, [10.0, 20.0])
        self.assertEqual(status, 0)
        self.assertEqual(list(tg.data), [11.0, 22.0])

    def test_transform_self_region(self):
        from cos_comparison.generate_layer import TensorGenerator
        tg = TensorGenerator([1.0, 2.0, 3.0, 4.0])
        status = tg.transform_self(lambda v: v * 10, start=(1,), shape=(2,),
                                   step=(1,))
        self.assertEqual(status, 0)
        self.assertEqual(list(tg.data), [1.0, 20.0, 30.0, 4.0])

    def test_transform_self_write_region(self):
        from cos_comparison.generate_layer import TensorGenerator
        from cos_comparison import core
        tg = TensorGenerator(core.create_void_list((4,), 1.0))
        status = tg.transform_self(lambda v: 9.0, start=(1,), shape=(2,),
                                   out_start=(1,), out_step=(1,))
        self.assertEqual(status, 0)
        self.assertEqual(list(tg.data), [1.0, 9.0, 9.0, 1.0])

    def test_timer(self):
        from cos_comparison.test_tool import Timer
        t = Timer()
        t.mark()
        time.sleep(0.01)
        self.assertGreater(t.get_time(), 0.0)
        t.reset()
        self.assertEqual(t.total_time, 0.0)


class TestAbstractBases(unittest.TestCase):
    """Every ABC enforces abstract implementations."""

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
    """no_done is imported from the core module everywhere."""

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

    def test_no_done_accepts_kwargs_on_all_backends(self):
        from cos_comparison import core
        old = core.get_active_backend()
        if old is not None:
            self.addCleanup(core.set_mode, old)
        for b in core.get_available_backends():
            try:
                core.set_mode(b)
            except ImportError:
                continue  # optional compiled backend not built
            self.assertIsNone(core.no_done(a=3, b=2))
            self.assertIsNone(core.no_done(1, 2))


class TestLayerFixes(unittest.TestCase):
    """Regression tests for known upper-layer defects (all fixed)."""

    def test_variable_eq(self):
        from cos_comparison.brain_layer.logic import Variable
        self.assertEqual(Variable("x", 1.0), Variable("x", 1.0))
        self.assertNotEqual(Variable("x", 1.0), Variable("x", 2.0))
        v = Variable("x", 1.0)
        self.assertEqual(v, v)

    def test_memory_wrap_id_slot(self):
        from cos_comparison.memory_layer.memory import (MemoryWrap,
                                                        MemoryWrapPool)
        m = MemoryWrap(memory_body={}, name="n", level=1)
        self.assertIsNone(m.id)
        p = MemoryWrapPool(name="p", level=0)
        self.assertIsNone(p.id)

    def test_memory_wrap_non_callable_keys_attr(self):
        from cos_comparison.memory_layer.memory import MemoryWrap

        class Carrier:
            def __init__(self):
                self.keys = ("a", "b")
            def __getitem__(self, key):
                return 1

        self.assertEqual(list(MemoryWrap(Carrier()).keys()), [])

    def test_memory_wrap_map_contains_list_pool(self):
        from cos_comparison.memory_layer.memory import (MemoryWrap,
                                                        MemoryWrapMap)
        first = MemoryWrap({}, name="first")
        pool = MemoryWrapMap([first])
        self.assertIn(first, pool)
        self.assertNotIn(MemoryWrap({}, name="other"), pool)

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

    def test_flat_to_window_cleanup_no_regression(self):
        # dead locals removed; behaviour of the window builder is unchanged
        from cos_comparison.core import cos_comparison as cc
        self.assertEqual(cc._flat_to_window([1, 2, 3, 4], (2, 2)),
                         [[1, 2], [3, 4]])
        self.assertEqual(cc._flat_to_window([1, 2, 3, 4], (4,)),
                         [1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
