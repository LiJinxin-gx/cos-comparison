"""app module tests: protocol-style Docker with default runnability.

The default protocol provides shell-format reading with data<N> ->
data_pool index refs (like C compilation), decoupled submit
interfaces, brain-layer control-flow driving, non-blocking
action-layer operate-flow driving (ExecuterDriver), and manage /
store_error interfaces.
"""

import time
import unittest

from cos_comparison.app import Docker, DockerProtocol
from cos_comparison.app.protocol import DataRef
from cos_comparison.brain_layer.control import Branch, Loop, Sequence


def _input(x):
    return x


def _add(a, b):
    return a + b


def _mul(a, b):
    return a * b


class TestDefaultRunnable(unittest.TestCase):
    def test_direct_operate_pool(self):
        d = Docker(operate_pool=[(_input, (5,), {}, 0),
                                 (_add, (5, 2), {}, 1),
                                 (_mul, (DataRef(1), 3), {}, 2)])
        driver = d.run()
        self.assertTrue(driver.wait(timeout=5))
        self.assertEqual(d.data_pool, {0: 5, 1: 7, 2: 21})

    def test_four_tuple_result_pos(self):
        d = Docker(operate_pool=[(_add, (2, 3), {}, 10),
                                 (_mul, (4, 5), {}, 20)])
        driver = d.run()
        driver.wait(timeout=5)
        self.assertEqual(d.data_pool, {10: 5, 20: 20})

    def test_data_injection_decided_by_function(self):
        def with_data(data, x):
            return data[0] + x
        d = Docker(operate_pool=[(_input, (1,), {}),
                                 (with_data, (10,), {})])
        driver = d.run()
        driver.wait(timeout=5)
        self.assertEqual(d.data_pool, {0: 1, 1: 11})

    def test_manage_default_noop(self):
        d = Docker()
        self.assertIsNone(d.manage("terminate"))
        self.assertIsNone(d.manage("send", 1))

    def test_store_error(self):
        d = Docker()
        exc = ValueError("boom")
        d.store_error(exc)
        self.assertIs(d.error, exc)

    def test_terminate_flag_stops_run(self):
        import threading
        entered = threading.Event()
        release = threading.Event()

        def first():
            entered.set()
            release.wait(timeout=5)
            return 1

        def second():
            return 2

        d = Docker(operate_pool=[(first, (), {}, 0), (second, (), {}, 1)])
        handle = d.start()
        try:
            self.assertTrue(entered.wait(timeout=5))
            d.terminated = True
            release.set()
            handle.join(timeout=5)
        finally:
            release.set()
        # the in-flight step completes on the action-layer worker; poll
        deadline = time.monotonic() + 5
        while d.data_pool.get(0) != 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        # the next boundary saw the flag: exactly the first step ran
        self.assertEqual(d.data_pool, {0: 1})


class TestShellReader(unittest.TestCase):
    """Default reader: imperative shell format, data<N> -> index refs."""

    def test_read_flow_from_text(self):
        def add(data, x):
            return data[0] + x
        d = Docker(namespace={"input": _input, "add": add})
        flow = d.read_operate_flow(
            "input 5 -> 0\n"
            "add data[0] 2 -> 1\n")
        self.assertEqual(len(flow), 2)
        fn, args, _kwargs, pos = flow[1]
        self.assertIs(fn, add)
        # values are compiled (ValueCode); the data reference executes
        # against the pool at run time
        self.assertIsInstance(args[0].code[0][1], DataRef)
        self.assertEqual(args[0].code[0][1].index, 0)
        self.assertEqual(pos, 1)

    def test_run_from_shell_text(self):
        d = Docker(namespace={"input": _input, "add": _add, "mul": _mul})
        d.operate_pool = d.read_operate_flow(
            "input 5 -> 0\n"
            "add data[0] 2 -> 1\n"
            "mul data[1] 3 -> 2\n")
        driver = d.run()
        self.assertTrue(driver.wait(timeout=5))
        self.assertEqual(d.data_pool, {0: 5, 1: 7, 2: 21})

    def test_keyword_and_bare(self):
        def f(x, *, k):
            return x + k
        d = Docker(namespace={"f": f})
        d.operate_pool = d.read_operate_flow("f 1 -k 2 -> 0")
        driver = d.run()
        self.assertTrue(driver.wait(timeout=5))
        self.assertEqual(d.data_pool, {0: 3})

    def test_comment_and_blank_lines(self):
        d = Docker(namespace={"input": _input})
        flow = d.read_operate_flow("# comment\n\ninput 9 -> 3\n")
        self.assertEqual(len(flow), 1)
        self.assertEqual(flow[0][3], 3)

    def test_read_from_file(self):
        import io
        d = Docker(namespace={"input": _input})
        src = io.StringIO("input 7 -> 5\n")
        flow = d.read_operate_flow(src)
        self.assertEqual(flow[0][3], 5)

    def test_unknown_function(self):
        d = Docker(namespace={})
        with self.assertRaises(ValueError):
            d.read_operate_flow("nope 1\n")


class TestDefaultExecutorNamespace(unittest.TestCase):
    """Default executor namespace: project package modules (dotted names),
    dot attribute/method extraction, and the injected namespace
    management functions (import_module / list_modules)."""

    def test_project_module_dotted_call(self):
        d = Docker()
        d.operate_pool = d.read_operate_flow(
            "core.add_chain (1, 2, 3) -> 0")
        driver = d.run()
        self.assertTrue(driver.wait(timeout=5))
        self.assertEqual(d.data_pool, {0: 6})

    def test_core_reflection_dotted(self):
        d = Docker()
        d.operate_pool = d.read_operate_flow(
            "core.multiple_chain (1, 2, 3) -> 0")
        driver = d.run()
        self.assertTrue(driver.wait(timeout=5))
        self.assertEqual(d.data_pool, {0: 6})

    def test_dotted_attribute_method(self):
        from cos_comparison.shell_tool import shell as SH
        SH.ns_set("s", "hello")
        try:
            d = Docker()
            d.operate_pool = d.read_operate_flow("s.upper -> 0")
            driver = d.run()
            self.assertTrue(driver.wait(timeout=5))
            self.assertEqual(d.data_pool, {0: "HELLO"})
        finally:
            SH.ns_delete(["s"])

    def test_management_function_import_module(self):
        d = Docker()
        d.operate_pool = d.read_operate_flow(
            "import_module operator -> 0")
        driver = d.run()
        self.assertTrue(driver.wait(timeout=5))
        self.assertIn("imported operator", d.data_pool.get(0, ""))

    def test_management_function_list_modules(self):
        d = Docker()
        d.operate_pool = d.read_operate_flow(
            "list_modules -> 0")
        driver = d.run()
        self.assertTrue(driver.wait(timeout=5))
        self.assertIn("core", d.data_pool.get(0, ""))

    def test_import_module_alias(self):
        d = Docker()
        d.operate_pool = d.read_operate_flow(
            "import_module operator -name op -> 0")
        driver = d.run()
        self.assertTrue(driver.wait(timeout=5))
        self.assertIn("imported op", d.data_pool.get(0, ""))
        d.operate_pool = d.read_operate_flow("op.add 1 2 -> 1")
        driver = d.run()
        self.assertTrue(driver.wait(timeout=5))
        self.assertEqual(d.data_pool.get(1), 3)

    def test_var_index_pointer_forms(self):
        from cos_comparison.shell_tool import shell as SH
        SH.ns_import("operator")
        SH.ns_set("x", "5")
        try:
            d = Docker()
            d.operate_pool = d.read_operate_flow(
                "operator.add *&x 0 -> 0\n"
                "operator.mul *&x 2 -> 1")
            driver = d.run()
            self.assertTrue(driver.wait(timeout=5))
            self.assertEqual(d.data_pool.get(0), 5)
            self.assertEqual(d.data_pool.get(1), 10)
        finally:
            SH.ns_delete(["x"])


class TestControlBlocks(unittest.TestCase):
    """Default reader wraps the brain-layer Control containers (Branch /
    Loop / Sequence): IF/ELSE/END and WHILE/END blocks, nested blocks;
    reading stays decoupled from execution."""

    def _run_flow(self, text):
        d = Docker()
        d.operate_pool = d.read_operate_flow(text)
        driver = d.run()
        self.assertTrue(driver.wait(timeout=5))
        return d

    def test_if_true_branch(self):
        d = self._run_flow(
            "IF core.add_chain (1, 1)\n"
            "    core.add_chain (1, 2) -> 0\n"
            "ELSE\n"
            "    core.add_chain (9, 9) -> 0\n"
            "END\n")
        self.assertEqual(d.data_pool, {0: 3})

    def test_if_false_branch(self):
        d = self._run_flow(
            "IF builtins.bool\n"
            "    core.add_chain (1, 2) -> 0\n"
            "ELSE\n"
            "    core.add_chain (9, 9) -> 0\n"
            "END\n")
        self.assertEqual(d.data_pool, {0: 18})

    def test_if_without_else(self):
        d = self._run_flow(
            "IF core.add_chain (1, 1)\n"
            "    core.add_chain (1, 2) -> 0\n"
            "END\n")
        self.assertEqual(d.data_pool, {0: 3})

    def test_nested_if(self):
        d = self._run_flow(
            "IF core.add_chain (1, 1)\n"
            "    IF core.add_chain (1, 1)\n"
            "        core.add_chain (1, 2) -> 0\n"
            "    ELSE\n"
            "        core.add_chain (1, 7) -> 0\n"
            "    END\n"
            "ELSE\n"
            "    core.add_chain (1, 9) -> 0\n"
            "END\n")
        self.assertEqual(d.data_pool, {0: 3})

    def test_while_block(self):
        from cos_comparison.shell_tool import shell as SH
        SH.register_callable("cond_lt3",
                             lambda data: data[0] < 3)

        def bump(data):
            data[0] = data[0] + 1
            return data[0]

        SH.register_callable("bump", bump)
        try:
            d = self._run_flow(
                "core.add_chain (0,) -> 0\n"
                "WHILE cond_lt3\n"
                "    bump -> 0\n"
                "END\n")
            self.assertEqual(d.data_pool[0], 3)
        finally:
            SH.ns_delete(["cond_lt3", "bump"])

    def test_unterminated_block(self):
        d = Docker()
        with self.assertRaises(ValueError):
            d.read_operate_flow("IF core.add_chain (1, 1)\n"
                                "    core.add_chain (1, 2) -> 0\n")

    def test_shell_and_app_control_consistent(self):
        """Different implementations (shell: instruction-jump functions
        injected into the namespace; app: reader wrapping brain-layer
        Control containers), same observable behaviour."""
        from cos_comparison.shell_tool import shell as SH
        SH.register_callable("fiv", lambda: 5)
        SH.register_callable("nine", lambda: 9)
        try:
            shell_true = SH.execute_call(
                *SH.parse_call("IF 1 fiv nine"))
            shell_false = SH.execute_call(
                *SH.parse_call("IF 0 fiv nine"))
        finally:
            SH.ns_delete(["fiv", "nine"])
        d_true = self._run_flow(
            "IF core.add_chain (1, 1)\n"
            "    core.add_chain (1, 4) -> 0\n"
            "ELSE\n"
            "    core.add_chain (1, 8) -> 0\n"
            "END\n")
        d_false = self._run_flow(
            "IF builtins.bool\n"
            "    core.add_chain (1, 4) -> 0\n"
            "ELSE\n"
            "    core.add_chain (1, 8) -> 0\n"
            "END\n")
        self.assertEqual(shell_true, 5)
        self.assertEqual(shell_false, 9)
        self.assertEqual(d_true.data_pool, {0: 5})
        self.assertEqual(d_false.data_pool, {0: 9})


class TestFunctionPool(unittest.TestCase):
    """Docker background function pool: duck-typed storage (dict default),
    every invocation receives self, keyed operations."""

    def test_default_dict_and_auto_keys(self):
        d = Docker()
        k0 = d.submit_function(lambda self: 1)
        k1 = d.submit_function(lambda self: 2)
        self.assertEqual((k0, k1), (0, 1))
        self.assertEqual(sorted(d.function_pool), [0, 1])
        self.assertEqual(d.call_function(0), 1)
        self.assertEqual(d.call_function(1), 2)

    def test_explicit_key(self):
        d = Docker()
        key = d.submit_function(lambda self: 42, key="named")
        self.assertEqual(key, "named")
        self.assertEqual(d.call_function("named"), 42)

    def test_self_injected_first(self):
        d = Docker()
        seen = []

        def probe(self, x):
            seen.append((self, x))
            return x * 2

        d.submit_function(probe)
        self.assertEqual(d.call_function(0, 5), 10)
        self.assertIs(seen[0][0], d)

    def test_remove_function(self):
        d = Docker()
        d.submit_function(lambda self: 1)
        d.remove_function(0)
        self.assertEqual(d.function_pool, {})
        with self.assertRaises(KeyError):
            d.call_function(0)

    def test_start_functions_threads(self):
        import threading
        d = Docker()
        done = threading.Event()

        def bg(self):
            done.set()

        d.submit_function(bg)
        self.assertEqual(d.start_functions(), 1)
        self.assertTrue(done.wait(timeout=3))

    def test_stop_functions_clears(self):
        d = Docker()
        d.submit_function(lambda self: 1)
        d.submit_function(lambda self: 2)
        d.stop_functions()
        self.assertEqual(d.function_pool, {})

    def test_sequence_pool_duck(self):
        d = Docker(function_pool=[])
        k0 = d.submit_function(lambda self: 7)
        k1 = d.submit_function(lambda self: 8)
        self.assertEqual((k0, k1), (0, 1))
        self.assertEqual(d.call_function(0), 7)
        self.assertEqual(d.call_function(1), 8)
        d.remove_function(0)
        self.assertEqual(d.call_function(0), 8)  # list shifts


class TestSubDocker(unittest.TestCase):
    """External delegation: a Docker creates a child Docker and delegates
    a task to it (module function subdocker)."""

    def test_delegate_text_flow(self):
        from cos_comparison.app.protocol import subdocker
        d = Docker()
        d.operate_pool = d.read_operate_flow(
            "core.add_chain (1, 2) -> 0")
        result = subdocker(d, "core.add_chain (3, 4) -> 0")
        self.assertEqual(result, {0: 7})

    def test_delegate_copies_data_pool(self):
        from cos_comparison.app.protocol import subdocker
        d = Docker()
        d.operate_pool = d.read_operate_flow(
            "core.add_chain (1, 2) -> 0")
        d.run().wait(timeout=5)
        result = subdocker(d, "core.add_chain (1, 1) -> 1")
        self.assertEqual(result, {0: 3, 1: 2})  # child copies parent data_pool
        self.assertEqual(d.data_pool, {0: 3})   # parent unaffected by child

    def test_delegate_data_isolation(self):
        from cos_comparison.app.protocol import subdocker
        d = Docker()
        d.operate_pool = d.read_operate_flow(
            "core.add_chain (1, 2) -> 0")
        d.run().wait(timeout=5)
        subdocker(d, "core.add_chain (9, 9) -> 0")
        self.assertEqual(d.data_pool, {0: 3})  # child writes do not pollute parent

    def test_delegate_explicit_data(self):
        from cos_comparison.app.protocol import subdocker
        from cos_comparison.shell_tool import shell as SH
        SH.register_callable("add2", lambda a, b: a + b)
        try:
            d = Docker()
            result = subdocker(d, "add2 data[0] 5 -> 1",
                               data={0: 10})
            self.assertEqual(result, {0: 10, 1: 15})
        finally:
            SH.ns_delete(["add2"])

    def test_delegate_item_list(self):
        from cos_comparison.app.protocol import subdocker
        d = Docker()
        items = [(_add, (2, 3), {}, 0)]
        result = subdocker(d, items)
        self.assertEqual(result, {0: 5})

    def test_delegate_child_error_raises(self):
        from cos_comparison.app.protocol import subdocker
        d = Docker()

        def boom(data):
            raise ValueError("child boom")

        from cos_comparison.shell_tool import shell as SH
        SH.register_callable("boom", boom)
        try:
            with self.assertRaises(ValueError):
                subdocker(d, "boom -> 0")
        finally:
            SH.ns_delete(["boom"])

    def test_delegate_nested(self):
        from cos_comparison.app.protocol import subdocker
        from cos_comparison.shell_tool import shell as SH

        def outer_flow(docker):
            inner = subdocker(docker, "core.add_chain (1, 2) -> 0")
            return inner[0] + 1

        SH.register_callable("outer_flow", outer_flow)
        try:
            d = Docker()
            result = subdocker(
                d,
                "core.add_chain (1, 1) -> 0\n"
                "outer_flow -> 5\n")
            self.assertEqual(result, {0: 2, 5: 4})
        finally:
            SH.ns_delete(["outer_flow"])


class TestSubmitDecoupling(unittest.TestCase):
    """Reading and running are decoupled: reading logic commits the
    arrangement through the submit interfaces."""

    def test_submit_operate_flow(self):
        d = Docker()
        d.submit_operate_flow([(_input, (4,), {}, 0),
                               (_add, (4, 1), {}, 1)])
        driver = d.run()
        driver.wait(timeout=5)
        self.assertEqual(d.data_pool, {0: 4, 1: 5})

    def test_submit_control_flow(self):
        def f1(data):
            return data[0] + 1
        d = Docker()
        d.submit_operate_flow([(_input, (1,), {}, 0)])
        d.submit_control_flow(Sequence([f1]))
        driver = d.run()
        driver.wait(timeout=5)
        self.assertEqual(d.data_pool[1], 2)

    def test_custom_reader_submits(self):
        # a custom reader parses a functional-imperative file and commits
        # the arrangement through the submit interface
        lines = [
            "input 4 -> 0",
            "add data[0] 1 -> 10",
        ]
        def file_reader(docker, *a, **k):
            items = docker.read_operate_flow("\n".join(lines))
            docker.submit_operate_flow(items)
        d = Docker(namespace={"input": _input, "add": _add})
        file_reader(d)
        driver = d.run()
        driver.wait(timeout=5)
        self.assertEqual(d.data_pool, {0: 4, 10: 5})


class TestControlFlow(unittest.TestCase):
    """brain-layer control flow containers run as expanded steps."""

    def test_sequence_expanded(self):
        def f1(data):
            return data[0] * 2
        d = Docker(operate_pool=[(_input, (3,), {}, 0),
                                 Sequence([f1])])
        driver = d.run()
        driver.wait(timeout=5)
        self.assertEqual(d.data_pool, {0: 3, 1: 6})

    def test_sequence_four_tuple_items(self):
        # control-flow items may be operate 4-tuples (result_pos honoured)
        d = Docker(operate_pool=[(_input, (3,), {}, 0),
                                 Sequence([(_add, (2, 3), {}, 10),
                                           (_mul, (4, 5), {}, 20)])])
        driver = d.run()
        driver.wait(timeout=5)
        self.assertEqual(d.data_pool, {0: 3, 10: 5, 20: 20})

    def test_branch_four_tuple_else(self):
        # branch yields a 4-tuple as the else item
        def yes(data):
            return data[0] + 100
        d = Docker(operate_pool=[
            (_input, (5,), {}, 0),
            Branch([(lambda: False, yes)],
                   else_func=(_mul, (2, 3), {}, 30))])
        driver = d.run()
        driver.wait(timeout=5)
        self.assertEqual(d.data_pool[30], 6)

    def test_loop_four_tuple_exec(self):
        # loop exec item as a 4-tuple (fixed two iterations)
        counter = [0]
        def do():
            counter[0] += 1
            return counter[0] <= 2
        d = Docker(operate_pool=[
            Loop(do, exec_func=(_add, (1, 1), {}, 40))])
        driver = d.run()
        driver.wait(timeout=5)
        self.assertEqual(d.data_pool[40], 2)

    def test_mixed_callables_and_tuples(self):
        # callables and 4-tuples mixed inside one sequence; callables take
        # the automatic step index as their result position
        def bump(data):
            return data[10] + 1
        d = Docker(operate_pool=[
            Sequence([(_input, (7,), {}, 10), bump, (_mul, (2, 3), {}, 20)])])
        driver = d.run()
        driver.wait(timeout=5)
        self.assertEqual(d.data_pool, {10: 7, 1: 8, 20: 6})


class TestStart(unittest.TestCase):
    def test_start_non_blocking(self):
        def slow(x):
            time.sleep(0.2)
            return x * 2
        d = Docker(operate_pool=[(slow, (21,), {}, 0)])
        handle = d.start()
        self.assertTrue(handle.is_alive())
        handle.join(timeout=5)
        # the run thread launches the action-layer worker thread; poll until
        # the result lands
        deadline = time.monotonic() + 3
        while 0 not in d.data_pool and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(d.data_pool[0], 42)


class TestInterfaces(unittest.TestCase):
    def test_read_control_flow_default(self):
        seq = Sequence([_input])
        d = Docker(operate_pool=[(_input, (1,), {}), seq])
        self.assertIn(seq, d.read_control_flow())

    def test_custom_reader_extension(self):
        def custom_read(docker, *a, **k):
            return {"custom": True}
        d = Docker()
        d.manager.read_operate_flow = custom_read
        self.assertEqual(d.read_operate_flow(), {"custom": True})

    def test_manager_instance_extension(self):
        class ExtProtocol(DockerProtocol):
            def read_operate_flow(self, docker, *a, **k):
                return "extended"
        d = Docker(manager=ExtProtocol())
        self.assertEqual(d.read_operate_flow(), "extended")

    def test_data_ref_resolve(self):
        self.assertEqual(DataRef(3).resolve({3: 9}), 9)

    def test_exception_stores_and_terminates(self):
        def boom(x):
            raise ValueError("boom")
        d = Docker(operate_pool=[(boom, (1,), {}, 0),
                                 (_input, (2,), {}, 1)])
        driver = d.run()
        driver.wait(timeout=5)
        self.assertIsInstance(d.error, ValueError)
        self.assertNotIn(1, d.data_pool)  # later steps terminated

    def test_maintainer_decides(self):
        # a custom maintainer fully controls execution (here: run each step
        # synchronously and swallow errors)
        def maintainer(docker, *a, **k):
            docker.terminated = False
            driver = docker.manager.run(docker, *a, **k)
            driver.wait(timeout=5)
            return driver
        d = Docker(operate_pool=[(_input, (1,), {}, 0)],
                   maintainer=maintainer)
        driver = d.run()
        self.assertIsNotNone(driver)


class TestExtensionScenarios(unittest.TestCase):
    """Extensibility: instructions stored in a database, and non-default
    formats - both plug in through a custom reader (protocol slot)."""

    def test_database_stored_instructions(self):
        # instructions live in a database (DatabaseMemory); a custom reader
        # queries them and hands the arrangement to the docker
        from cos_comparison.memory_layer.memory.database_memory import DatabaseMemory
        db = DatabaseMemory()
        db.execute("CREATE TABLE IF NOT EXISTS steps "
                   "(seq INTEGER, line TEXT)")
        db.executemany(
            "INSERT INTO steps (seq, line) VALUES (?, ?)",
            [(0, "input 4 -> 0"),
             (1, "add data[0] 1 -> 10")])
        db.commit()

        def db_reader(docker, *a, **k):
            lines = []
            for row in db.execute("SELECT line FROM steps ORDER BY seq"):
                lines.append(row[0])
            # compose: query the storage, then parse with the default
            # shell-format parser (protocol default, not the replaced slot)
            items = DockerProtocol().read_operate_flow(
                docker, "\n".join(lines))
            docker.submit_operate_flow(items)
            return len(items)

        try:
            d = Docker(namespace={"input": _input, "add": _add})
            d.manager.read_operate_flow = db_reader
            self.assertEqual(d.read_operate_flow(), 2)  # 2 steps submitted
            driver = d.run()
            driver.wait(timeout=5)
            self.assertEqual(d.data_pool, {0: 4, 10: 5})
        finally:
            db.close()

    def test_non_default_format_reader(self):
        # a JSON arrangement (non-default format) parsed by a custom reader
        import json
        payload = json.dumps([
            {"fn": "add", "args": [2, 3], "kwargs": {}, "result_pos": 10},
            {"fn": "mul", "args": [4, 5], "kwargs": {}, "result_pos": 20},
        ])

        def json_reader(docker, *a, **k):
            items = []
            for entry in json.loads(payload):
                fn = docker.namespace[entry["fn"]]
                items.append((fn, tuple(entry["args"]),
                              dict(entry["kwargs"]), entry["result_pos"]))
            docker.submit_operate_flow(items)
            return len(items)

        d = Docker(namespace={"add": _add, "mul": _mul})
        d.manager.read_operate_flow = json_reader
        self.assertEqual(d.read_operate_flow(), 2)
        driver = d.run()
        driver.wait(timeout=5)
        self.assertEqual(d.data_pool, {10: 5, 20: 20})

class TestSuspend(unittest.TestCase):
    """Docker suspend/resume: cooperative stop at step boundaries, snapshot
    export, continuation from the recorded cursor without re-running."""

    def _flow(self, n, calls, delay=0.02):
        def step(i):
            def run(x):
                calls.append(i)
                time.sleep(delay)
                return x
            return run
        return Docker(operate_pool=[(step(i), (i,), {}, i)
                                    for i in range(n)])

    def test_suspend_returns_snapshot(self):
        calls = []
        d = self._flow(8, calls, delay=0.05)
        t = d.start()
        deadline = time.monotonic() + 5
        while d.done_steps == 0 and time.monotonic() < deadline:
            time.sleep(0.005)
        snap = d.suspend(timeout=5)
        t.join(timeout=5)
        self.assertTrue(d.suspended)
        self.assertGreaterEqual(snap.done_steps, 0)
        self.assertLess(snap.done_steps, 8)  # stopped mid-run
        self.assertEqual(snap.data_pool, d.data_pool)
        self.assertIsNot(snap.data_pool, d.data_pool)  # shallow copy
        self.assertIs(snap.manager, d.manager)
        self.assertLessEqual(len(calls), 8)

    def test_resume_continues_exactly_once(self):
        calls = []
        d = self._flow(10, calls, delay=0.05)
        t = d.start()
        deadline = time.monotonic() + 5
        while d.done_steps == 0 and time.monotonic() < deadline:
            time.sleep(0.005)
        snap = d.suspend(timeout=5)
        t.join(timeout=5)
        done = snap.done_steps
        self.assertGreater(done, 0)
        d2 = Docker()
        d2.resume(snap)
        self.assertFalse(d2.suspended)
        self.assertEqual(d2.done_steps, done)
        self.assertEqual(d2.data_pool, snap.data_pool)
        d2.start()
        deadline = time.monotonic() + 5
        while d2.data_pool.get(9) != 9 and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertEqual(d2.data_pool, {i: i for i in range(10)})
        self.assertEqual(len(calls), 10)  # each step ran exactly once

    def test_resume_keeps_existing_config(self):
        # resume loads state only; manager/namespace stay the instance's own
        calls = []
        d = self._flow(6, calls, delay=0.01)
        t = d.start()
        time.sleep(0.05)
        snap = d.suspend(timeout=5)
        t.join(timeout=5)
        d2 = Docker(manager=d.manager, namespace=d.namespace)
        d2.resume(snap)
        d2.start()
        deadline = time.monotonic() + 5
        while d2.data_pool.get(5) != 5 and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertEqual(len(calls), 6)

    def test_plain_run_resets_cursor(self):
        calls = []
        d = self._flow(3, calls, delay=0.0)
        d.run().wait(timeout=5)
        first = len(calls)
        d.run().wait(timeout=5)  # ordinary run: full re-run, no skip
        self.assertEqual(len(calls), first * 2)
        self.assertEqual(d.data_pool, {0: 0, 1: 1, 2: 2})

    def test_suspend_without_run(self):
        d = Docker(operate_pool=[(_input, (1,), {}, 0)])
        snap = d.suspend()
        self.assertEqual(snap.done_steps, 0)
        self.assertTrue(d.suspended)
        d2 = Docker()
        d2.resume(snap)
        d2.run().wait(timeout=5)
        self.assertEqual(d2.data_pool, {0: 1})  # nothing skipped

    def test_presuspended_run_skips_all(self):
        d = Docker(operate_pool=[(_input, (1,), {}, 0),
                                 (_add, (1, 1), {}, 1)])
        d.suspended = True
        driver = d.run()
        driver.wait(timeout=5)
        self.assertEqual(d.data_pool, {})  # stopped at the first boundary


if __name__ == "__main__":
    unittest.main()



