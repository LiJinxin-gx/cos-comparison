"""shell_tool tests: runpy directory plugin execution, imperative shell."""
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import cos_comparison.__main__ as CLI
from cos_comparison.shell_tool import shell as SH

import testutil


def _snapshot_shared_state():
    ns = {key: (dict(value) if isinstance(value, dict)
                else list(value) if isinstance(value, list) else value)
          for key, value in SH._ns.items()}
    return ns, dict(SH._interrupt), dict(CLI._NAMESPACE)


def _restore_shared_state(snapshot):
    ns, interrupt, cli_ns = snapshot
    for key in list(SH._ns):
        if key not in ns:
            del SH._ns[key]
    for key, value in ns.items():
        SH._ns[key] = value
    SH._interrupt.clear()
    SH._interrupt.update(interrupt)
    CLI._NAMESPACE.clear()
    CLI._NAMESPACE.update(cli_ns)


class ShellTestCase(unittest.TestCase):
    """Snapshot/restore the process-global shell and CLI namespace state
    so tests never leak variables, callables or interrupt counters."""

    def setUp(self):
        self._shared_snapshot = _snapshot_shared_state()

    def tearDown(self):
        _restore_shared_state(self._shared_snapshot)


class TestPluginSearch(ShellTestCase):
    """Directory code search: file name == command name (no registry)."""

    def test_plugin_path_exists(self):
        self.assertIsNotNone(CLI.plugin_path("version"))
        self.assertIsNotNone(CLI.plugin_path("shell"))

    def test_plugin_path_unknown(self):
        self.assertIsNone(CLI.plugin_path("no_such_command"))

    def test_list_commands(self):
        cmds = CLI.list_commands()
        self.assertIn("version", cmds)
        self.assertIn("shell", cmds)


def _value(text, data=None, env=None):
    """Compile + execute a single value word (tests the unified literal
    grammar end to end)."""
    from cos_comparison.shell_tool.value import compile_value, ValueEnv
    if env is None:
        env = ValueEnv(data=data, vars={}, resolve=SH.resolve_callable,
                       var_index={})
    return compile_value(text).execute(env)


def _call_values(call_text, data=None):
    """Positional/keyword values of a parsed call, executed (so a test
    sees the real values, not the compiled ValueCode objects)."""
    _f, args, kwargs = SH.parse_call(call_text)
    env = SH.ValueEnv(data=data, vars={}, resolve=SH.resolve_callable,
                      var_index={})
    out_args = tuple(
        a.execute(env) if isinstance(a, SH.ValueCode) else a for a in args)
    out_kwargs = {k: (v.execute(env) if isinstance(v, SH.ValueCode) else v)
                  for k, v in kwargs.items()}
    return out_args, out_kwargs


class TestParseCall(ShellTestCase):
    def test_positional(self):
        func, args, kwargs = SH.parse_call("math.sqrt 16.0")
        self.assertEqual(func, "math.sqrt")
        self.assertEqual(_call_values("math.sqrt 16.0")[0], (16.0,))
        self.assertEqual(kwargs, {})

    def test_keyword(self):
        _func, _args, kwargs = SH.parse_call("math.pow 2 -exp 3")
        self.assertEqual(_call_values("math.pow 2 -exp 3")[0], (2,))
        self.assertEqual(_call_values("math.pow 2 -exp 3")[1],
                         {"exp": 3})
        self.assertEqual(tuple(kwargs), ("exp",))

    def test_nested_group(self):
        # a multi-element group without commas is a NESTED CALL; tuples
        # are Python-style, comma-separated
        func, args, _kwargs = SH.parse_call("f (1, 2, 3) (4, 5)")
        self.assertEqual(func, "f")
        self.assertEqual(_call_values("f (1, 2, 3) (4, 5)")[0],
                         ((1, 2, 3), (4, 5)))

    def test_quoted(self):
        _func, _args, _kw = SH.parse_call('f "hello world"')
        self.assertEqual(_call_values('f "hello world"')[0],
                         ("hello world",))

    def test_numbers(self):
        _func, _args, _kw = SH.parse_call("f 1 2.5 x")
        self.assertEqual(_call_values("f 1 2.5 x")[0], (1, 2.5, "x"))


class TestLiterals(ShellTestCase):
    """Python-convention literals: tuples with commas, lists, booleans,
    None, single/double quoted strings, nesting (spaces still accepted)."""

    def test_tuple_comma(self):
        self.assertEqual(SH._atom("(1, 0)"), (1, 0))
        self.assertEqual(SH._atom("(1,)"), (1,))
        self.assertEqual(SH._atom("()"), ())
        self.assertEqual(SH._atom("(1, 0)"), (1, 0))

    def test_list(self):
        self.assertEqual(SH._atom("[1, 2]"), [1, 2])
        self.assertEqual(SH._atom("[1 2]"), [1, 2])
        self.assertEqual(SH._atom("[]"), [])

    def test_constants(self):
        self.assertIs(SH._atom("True"), True)
        self.assertIs(SH._atom("False"), False)
        self.assertIs(SH._atom("None"), None)

    def test_nested(self):
        self.assertEqual(SH._atom("((1, 0), (2, 3))"), ((1, 0), (2, 3)))
        self.assertEqual(SH._atom("[[1, 2], [3, 4]]"), [[1, 2], [3, 4]])
        self.assertEqual(SH._atom("(1, (2, 3))"), (1, (2, 3)))

    def test_quoted_in_group(self):
        self.assertEqual(SH._atom("('a', 'b')"), ("a", "b"))
        self.assertEqual(SH._atom('("x", "y")'), ("x", "y"))

    def test_call_with_python_literals(self):
        self.assertEqual(SH.execute_call(
            *SH.parse_call("cos (1, 0) (1, 0)")), 1.0)
        self.assertEqual(SH.execute_call(
            *SH.parse_call("cos [1, 0] [1, 0]")), 1.0)

    def test_quoted_comma_verbatim(self):
        self.assertEqual(SH._atom('("a,b", 1)'), ("a,b", 1))

    def test_deep_nesting_iterative(self):
        # 2000 levels of nesting: iterative parser, no RecursionError
        deep = "(" * 2000 + "1" + ")" * 2000
        v = SH._atom(deep)
        self.assertIsInstance(v, tuple)
        for _ in range(1999):
            v = v[0]
        self.assertEqual(v, (1,))

    def test_unbalanced_raises(self):
        with self.assertRaises(ValueError):
            SH._atom("(1, 2")


class TestExecute(ShellTestCase):
    def test_core_call(self):
        self.assertEqual(SH.execute_call(*SH.parse_call("cos (1, 0) (1, 0)")),
                         1.0)

    def test_error_intercepted(self):
        r = SH.execute_call(*SH.parse_call("no.such.func 1"))
        self.assertIn("error", r)

    def test_kb_propagates(self):
        def kb(*_a, **_k):
            raise KeyboardInterrupt()

        SH.register_callable("kb_f", kb)
        with self.assertRaises(KeyboardInterrupt):
            SH.execute_call(*SH.parse_call("kb_f"))


class TestNamespaceOps(ShellTestCase):
    def test_let_get_delete(self):
        self.assertEqual(SH.ns_set("x", "5"), "5")
        self.assertEqual(SH.ns_get("x"), "5")
        self.assertEqual(SH.ns_delete(["x"]), "deleted: x")
        self.assertEqual(SH.ns_get("x"), "undefined")

    def test_get_none_value_not_undefined(self):
        SH.ns_set("n", "None")
        try:
            self.assertEqual(SH.ns_get("n"), "None")
        finally:
            SH.ns_delete(["n"])

    def test_negative_literal_is_not_a_keyword(self):
        _func, _args, kwargs = SH.parse_call("math.pow -2 3")
        self.assertEqual(kwargs, {})
        self.assertEqual(_call_values("math.pow -2 3")[0], (-2, 3))
        # keyword form still works
        _func, _args, kwargs = SH.parse_call("math.pow 2 -exp 3")
        self.assertEqual(tuple(kwargs), ("exp",))
        self.assertEqual(_call_values("math.pow 2 -exp 3")[1], {"exp": 3})

    def test_import_module(self):
        result = SH.ns_import("operator")
        self.assertIn("imported operator", result)
        r = SH.execute_call(*SH.parse_call("operator.add 1 2"))
        self.assertEqual(r, 3)


class TestProjectNamespace(ShellTestCase):
    """Default namespace: project package modules, dotted attribute/method
    extraction, and the injected namespace management functions."""

    def test_dotted_module_function(self):
        r = SH.execute_call(*SH.parse_call("core.add_chain (1, 2, 3)"))
        self.assertEqual(r, 6)

    def test_dotted_core_reflection(self):
        r = SH.execute_call(
            *SH.parse_call("core.cos_comparison_passive 4"))
        self.assertIsNotNone(r)

    def test_dotted_nested_module(self):
        obj = SH._extract_dotted("core.cos_comparison.add_chain")
        self.assertTrue(callable(obj))

    def test_dotted_attribute_method(self):
        SH.ns_set("s", "hello")
        r = SH.execute_call(*SH.parse_call("s.upper"))
        self.assertEqual(r, "HELLO")
        SH.ns_delete(["s"])

    def test_dotted_value_extraction(self):
        SH.ns_import("math")
        self.assertEqual(SH.ns_get("math.pi"), str(math.pi))

    def test_import_module_project_path(self):
        result = SH.ns_import("core")
        self.assertIn("imported core", result)
        self.assertIn("core", SH._ns["modules"])
        r = SH.execute_call(*SH.parse_call("core.add_chain (1, 2)"))
        self.assertEqual(r, 3)

    def test_list_modules(self):
        result = SH.ns_list_modules()
        self.assertIn("core", result)
        self.assertIn("shell_tool", result)

    def test_management_functions_injected(self):
        ns = SH.namespace()
        self.assertIs(ns["import_module"], SH.ns_import)
        self.assertIs(ns["list_modules"], SH.ns_list_modules)

    def test_import_module_name_alias(self):
        result = SH.run_shell(
            ["import_module", "operator", "-name", "op"])
        self.assertIn("imported op", result)
        r = SH.execute_call(*SH.parse_call("op.add 1 2"))
        self.assertEqual(r, 3)

    def test_import_module_name_kwarg(self):
        result = SH.ns_import("operator", name="o2")
        self.assertIn("imported o2", result)
        r = SH.execute_call(*SH.parse_call("o2.mul 3 4"))
        self.assertEqual(r, 12)

    def test_var_index_ref(self):
        SH.ns_set("x", "5")
        idx = SH._atom("&x")
        self.assertIsInstance(idx, int)
        SH.ns_set("p", "&x")
        self.assertEqual(SH._atom("*p"), 5)
        SH.ns_delete(["x", "p"])

    def test_deref_direct_and_nested(self):
        SH.ns_set("x", "5")
        r = SH.execute_call(*SH.parse_call("operator.add *&x 0"))
        self.assertEqual(r, 5)
        SH.ns_delete(["x"])

    def test_deref_by_index(self):
        SH.ns_set("y", "7")
        idx = SH._atom("&y")
        self.assertEqual(SH._atom("*" + str(idx)), 7)
        SH.ns_delete(["y"])

    def test_var_index_unknown(self):
        with self.assertRaises(ValueError):
            SH._atom("&zz")

    def test_deref_invalid(self):
        SH.ns_set("s", "hello")
        with self.assertRaises(ValueError):
            SH._atom("*s")
        SH.ns_delete(["s"])

    def test_quoted_ampersand_star_verbatim(self):
        """& and * inside quotes are plain text (no accidental parsing)."""
        self.assertEqual(SH._atom('"&x"'), "&x")
        self.assertEqual(SH._atom('"*3"'), "*3")
        self.assertEqual(SH._atom("'&x'"), "&x")
        func, _args, kwargs = SH.parse_call(
            'operator.add "&x" "*3" -k "*&y"')
        self.assertEqual(func, "operator.add")
        self.assertEqual(tuple(kwargs), ("k",))
        self.assertEqual(_call_values('operator.add "&x" "*3" -k "*&y"')[0],
                         ("&x", "*3"))
        self.assertEqual(_call_values('operator.add "&x" "*3" -k "*&y"')[1],
                         {"k": "*&y"})

    def test_quoted_inside_group_verbatim(self):
        self.assertEqual(SH._atom('("a&b" "*c")'), ("a&b", "*c"))


class TestBuiltinsPreimport(ShellTestCase):
    """CLI pre-imports built-in functions (functions/types from the __builtins__ module) — directly usable without overriding project-registered names."""

    def test_builtin_function_callable(self):
        r = SH.execute_call(*SH.parse_call("len (1, 2, 3)"))
        self.assertEqual(r, 3)
        r2 = SH.execute_call(*SH.parse_call("sum (1, 2, 3)"))
        self.assertEqual(r2, 6)

    def test_builtin_type_callable(self):
        r = SH.execute_call(*SH.parse_call("int 5"))
        self.assertEqual(r, 5)
        r2 = SH.execute_call(*SH.parse_call("str 7"))
        self.assertEqual(r2, "7")

    def test_builtins_registered_in_namespace(self):
        ns = SH.namespace()
        self.assertIn("len", ns)
        self.assertIn("sum", ns)
        self.assertIn("int", ns)

    def test_builtins_do_not_override_project(self):
        # project-registered names take priority (setdefault does not overwrite)
        ns = SH.namespace()
        self.assertIs(ns["IF"], SH.IF)
        self.assertIsNotNone(ns.get("cos"))

    def test_builtins_batch_usage(self):
        from cos_comparison.shell_tool.batch import run_batch
        data, _ = run_batch([
            "core.add_chain (1, 2) -> 0",
            "str data[0] -> 1",
        ])
        self.assertEqual(data, {0: 3, 1: "3"})


class TestIndexSyntax(ShellTestCase):
    """data[index] indexing syntax (replaces the data<N> name-collision form), % / %% unpack operations, and let functional assignment."""

    def test_data_index_syntax(self):
        from cos_comparison.shell_tool.batch import run_batch
        data, _ = run_batch([
            "import_module math",
            "math.sqrt 9.0 -> 0",
            "math.sqrt data[0] -> 1",
        ])
        self.assertAlmostEqual(data[0], 3.0)
        self.assertAlmostEqual(data[1], 3.0 ** 0.5)

    def test_old_data_legacy_compatible(self):
        from cos_comparison.shell_tool.batch import run_batch
        data, _ = run_batch([
            "import_module math",
            "math.sqrt 9.0 -> 0",
            "math.sqrt data0 -> 1",
        ])
        self.assertAlmostEqual(data[1], 3.0 ** 0.5)

    def test_percent_sequence_unpack(self):
        r = SH.execute_call(*SH.parse_call("max %(1, 2, 3)"))
        self.assertEqual(r, 3)
        r2 = SH.execute_call(*SH.parse_call("min %(1, 2, 3)"))
        self.assertEqual(r2, 1)

    def test_percent_unpack_data_ref(self):
        from cos_comparison.shell_tool.batch import run_batch
        data, _ = run_batch([
            "list (1, 2) -> 0",
            "max %data[0] -> 1",
        ])
        self.assertEqual(data, {0: [1, 2], 1: 2})

    def test_percent_percent_mapping_unpack(self):
        from cos_comparison.shell_tool import shell as SH

        def key_join(**kw):
            return kw["a"] + kw["b"]

        SH.register_callable("key_join", key_join)
        try:
            from cos_comparison.shell_tool.batch import run_batch
            data, _ = run_batch([
                "{a: 1, b: 2} -> 0",
                "key_join %%data[0] -> 1",
            ])
            self.assertEqual(data, {0: {"a": 1, "b": 2}, 1: 3})
        finally:
            SH.ns_delete(["key_join"])

    def test_let_functional_assignment(self):
        from cos_comparison.shell_tool.batch import run_batch
        _data, stats = run_batch([
            "let x 5",
            "let y (1, 2)",
        ])
        self.assertEqual(stats["run"], 2)
        self.assertEqual(SH.ns_get("x"), "5")
        self.assertEqual(SH.ns_get("y"), "(1, 2)")

    def test_unpack_in_app_reader(self):
        from cos_comparison.app import Docker
        d = Docker()
        d.operate_pool = d.read_operate_flow(
            "max %(1, 2, 3) -> 0\n"
            "list (1, 2) -> 1\n"
            "max %data[1] -> 2\n")
        drv = d.run()
        self.assertTrue(drv.wait(timeout=5))
        self.assertEqual(d.data_pool, {0: 3, 1: [1, 2], 2: 2})


class TestControlFunctions(ShellTestCase):
    """Instruction-style control flow (assembly-like jumps, function
    unit) with name-mapping targets, injected into the namespace like
    import_module."""

    def _register_pair(self):
        SH.register_callable("fiv", lambda: 5)
        SH.register_callable("nine", lambda: 9)

    def _clean_pair(self):
        SH.ns_delete(["fiv", "nine"])

    def test_control_functions_injected(self):
        ns = SH.namespace()
        self.assertIs(ns["IF"], SH.IF)
        self.assertIs(ns["WHILE"], SH.WHILE)

    def test_if_truthy_jumps_true_target(self):
        self._register_pair()
        try:
            r = SH.execute_call(*SH.parse_call("IF 1 fiv nine"))
            self.assertEqual(r, 5)
        finally:
            self._clean_pair()

    def test_if_falsy_jumps_false_target(self):
        self._register_pair()
        try:
            r = SH.execute_call(*SH.parse_call("IF 0 fiv nine"))
            self.assertEqual(r, 9)
        finally:
            self._clean_pair()

    def test_if_without_false_target(self):
        self._register_pair()
        try:
            self.assertIsNone(
                SH.execute_call(*SH.parse_call("IF 0 fiv")))
        finally:
            self._clean_pair()

    def test_if_callable_condition(self):
        SH.register_callable("yes", lambda: True)
        SH.register_callable("fiv", lambda: 5)
        try:
            r = SH.execute_call(*SH.parse_call("IF yes fiv"))
            self.assertEqual(r, 5)
        finally:
            SH.ns_delete(["yes", "fiv"])

    def test_while_jumps_body(self):
        n = {"v": 0}
        SH.register_callable("lt3", lambda: n["v"] < 3)
        SH.register_callable("inc", lambda: n.__setitem__("v", n["v"] + 1))
        try:
            r = SH.execute_call(*SH.parse_call("WHILE lt3 inc"))
            self.assertEqual(r, 3)
            self.assertEqual(n["v"], 3)
        finally:
            SH.ns_delete(["lt3", "inc"])

    def test_while_false_condition_zero(self):
        SH.register_callable("no", lambda: False)
        SH.register_callable("fiv", lambda: 5)
        try:
            r = SH.execute_call(*SH.parse_call("WHILE no fiv"))
            self.assertEqual(r, 0)
        finally:
            SH.ns_delete(["no", "fiv"])

    def test_run_shell_import_module(self):
        result = SH.run_shell(["import_module", "operator"])
        self.assertIn("imported operator", result)


class TestImportAllModule(ShellTestCase):
    """import_all_module: from module import * effect — all public objects are attached to the specified namespace (keyword namespace, defaults to the current shell namespace)."""

    def test_import_all_default_namespace(self):
        result = SH.import_all_module("operator")
        self.assertIn("imported operator", result)
        # all public objects (functions and data) enter the current funcs namespace
        ns = SH.namespace()
        self.assertIn("add", ns)
        self.assertIn("truth", ns)
        r = SH.execute_call(*SH.parse_call("add 1 2"))
        self.assertEqual(r, 3)

    def test_import_all_into_custom_namespace(self):
        target = {}
        result = SH.import_all_module("math", namespace=target)
        self.assertIn("imported math", result)
        self.assertIn("sqrt", target)
        self.assertIn("pi", target)
        self.assertEqual(target["sqrt"](16.0), 4.0)

    def test_import_all_unknown_module(self):
        result = SH.import_all_module("no.such.module.xyz")
        self.assertTrue(result.startswith("cannot import"))

    def test_import_all_via_command_line(self):
        result = SH.run_shell(["import_all_module", "operator"])
        self.assertIn("imported operator", result)
        r = SH.execute_call(*SH.parse_call("add 1 2"))
        self.assertEqual(r, 3)

    def test_import_all_in_batch(self):
        from cos_comparison.shell_tool.batch import run_batch
        data, _ = run_batch([
            "import_all_module operator",
            "add 1 2 -> 0",
            "mul 3 4 -> 1",
        ])
        self.assertEqual(data[0], 3)
        self.assertEqual(data[1], 12)

    def test_import_all_in_app_reader(self):
        from cos_comparison.app import Docker
        d = Docker()
        d.operate_pool = d.read_operate_flow(
            "import_all_module operator -> 0\n"
            "add 1 2 -> 1\n")
        drv = d.run()
        self.assertTrue(drv.wait(timeout=5))
        self.assertEqual(d.data_pool[1], 3)

    def test_linear_algebra_not_pre_registered(self):
        # undo hard registration: la function no longer appears directly in namespace (import_all_module on demand)
        ns = SH.namespace()
        self.assertNotIn("dot", ns)
        result = SH.import_all_module(
            "cos_comparison.interface.tools.math_tool.linear_algebra")
        self.assertIn("imported", result)
        self.assertIn("dot", SH.namespace())


class TestMainCLI(ShellTestCase):
    """python -m cos_comparison <command> — runpy plugin execution."""

    def _run(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "cos_comparison", *args],
            capture_output=True, text=True, env=testutil.clean_env(),
            cwd=os.path.dirname(os.path.abspath(__file__)),
            timeout=60, check=False)

    def test_version_cli(self):
        p = self._run("version")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertRegex(p.stdout.strip(), r"\d+\.\d+")

    def test_shell_cli(self):
        p = self._run("shell", "cos", "(1, 0)", "(1, 0)")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(p.stdout.strip(), "1.0")

    def test_shell_kwargs_cli(self):
        p = self._run("shell", "cos_comparison_passive",
                      "((1, 2, 3, 4), (5, 6, 7, 8), (9, 1, 2, 3), (4, 5, 6, 7))",
                      "-window_size", "(3, 3)")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertNotIn("error", p.stdout)

    def test_unknown_command_cli(self):
        p = self._run("bogus_cmd")
        self.assertEqual(p.returncode, 1)
        self.assertIn("unknown command", p.stdout)

    def test_error_intercepted_cli(self):
        p = self._run("shell", "no.such.func", "1")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("error:", p.stdout)


class TestMainMonitor(ShellTestCase):
    """__main__ monitors KeyboardInterrupt and intercepts errors."""

    def _make_plugin(self, code, name="kb"):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        path = os.path.join(tmp, name + ".py")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(code)
        return path

    def test_kb_returns_130(self):
        path = self._make_plugin(
            "import sys\n"
            "raise KeyboardInterrupt()\n")
        with mock.patch.object(CLI, "plugin_path",
                               return_value=path):
            rc = CLI.main(["kb"])
        self.assertEqual(rc, 130)

    def test_error_intercepted(self):
        path = self._make_plugin(
            "import sys\n"
            "raise ValueError('boom')\n")
        with mock.patch.object(CLI, "plugin_path",
                               return_value=path):
            rc = CLI.main(["kb"])
        self.assertEqual(rc, 1)

    def test_unknown_command(self):
        with mock.patch.object(CLI, "plugin_path", return_value=None):
            rc = CLI.main(["nope"])
        self.assertEqual(rc, 1)

    def test_namespace_injected(self):
        """__ns__ mapping is injected into the plugin globals."""
        path = self._make_plugin(
            "import sys\n"
            "if '__ns__' in globals():\n"
            "    globals()['__ns__']['marker'] = 42\n",
            name="nscheck")
        with mock.patch.object(CLI, "plugin_path", return_value=path):
            rc = CLI.main(["nscheck"])
        self.assertEqual(rc, 0)
        self.assertEqual(CLI._NAMESPACE.get("marker"), 42)

    def test_shared_ns_across_plugins(self):
        """Shared mapping keeps values across plugin runs (same process)."""
        import contextlib
        import io
        put = self._make_plugin("globals()['__ns__']['v'] = 'shared'\n",
                                name="put")
        get = self._make_plugin(
            "import sys\n"
            "sys.stdout.write(str(globals()['__ns__'].get('v')))\n",
            name="get")
        with mock.patch.object(CLI, "plugin_path",
                               return_value=put):
            rc = CLI.main(["put"])
        self.assertEqual(rc, 0)
        buf = io.StringIO()
        with mock.patch.object(CLI, "plugin_path",
                               return_value=get), \
                contextlib.redirect_stdout(buf):
            rc = CLI.main(["get"])
        self.assertEqual(rc, 0)
        self.assertEqual(buf.getvalue(), "shared")


class TestPlugins(ShellTestCase):
    """shell_tool/__init__: lazy import + file-scan plugin listing."""

    def test_list_plugins_file_scan(self):
        from cos_comparison.shell_tool import list_plugins
        plugins = list_plugins()
        for name in ("shell", "version", "batch", "helps"):
            self.assertIn(name, plugins)
        self.assertNotIn("value", plugins)  # helper module, not a command

    def test_load_plugin_lazy(self):
        from cos_comparison.shell_tool import load_plugin
        mod = load_plugin("batch")
        self.assertEqual(mod.__name__.rsplit(".", 1)[-1], "batch")

    def test_is_command_rejects_undecodable_file(self):
        from cos_comparison.shell_tool import _is_command
        fd, path = tempfile.mkstemp(suffix=".py")
        os.write(fd, b"\xff\xfe\x00if __name__ == '__main__':")
        os.close(fd)
        try:
            self.assertFalse(_is_command(path))
        finally:
            os.unlink(path)


class TestHelps(ShellTestCase):
    """helps plugin: pydoc documentation viewer (extensible/plugin)."""

    def test_helps_module(self):
        from cos_comparison.shell_tool.helps import run
        out = run(["math"])
        self.assertIn("math", out)
        self.assertNotIn("error:", out)

    def test_helps_unknown(self):
        from cos_comparison.shell_tool.helps import run
        out = run(["no.such.module.xyz"])
        self.assertTrue(out.startswith("error:"))

    def test_helps_usage(self):
        from cos_comparison.shell_tool.helps import run
        self.assertIn("usage", run([]))

    def test_helps_member_name(self):
        from cos_comparison.shell_tool.helps import run
        out = run(["math", "sqrt"])
        self.assertIn("sqrt", out)

    def test_helps_shell_namespace_alias(self):
        # a module registered in the shell namespace renders through the
        # injected __ns__ mapping (reachability documented for helps)
        import contextlib
        import io
        import math as _math
        from cos_comparison import __main__ as CLI
        ns = CLI._NAMESPACE
        ns.setdefault("modules", {})["helps_alias"] = _math
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                rc = CLI.run_command("helps", ["helps_alias"])
        finally:
            ns.get("modules", {}).pop("helps_alias", None)
            CLI._NAMESPACE.clear()
        self.assertEqual(rc, 0)
        self.assertIn("math", buf.getvalue())


class TestBatch(ShellTestCase):
    """batch plugin: independent batch execution (shell principle, own
    data region, no shared stack, no app dependency)."""

    def test_batch_execute(self):
        from cos_comparison.shell_tool.batch import run_batch
        data, stats = run_batch([
            "import_module math",
            "math.sqrt 16.0 -> 0",
            "let x 5",
            "get x",
        ])
        # instruction results without -> are automatically stored in sequential positions (unified instruction protocol semantics)
        self.assertEqual(data, {0: 4.0, 2: "5", 3: "5"})
        self.assertEqual(stats["run"], 4)

    def test_batch_variable_ref(self):
        from cos_comparison.shell_tool.batch import run_batch
        data, _stats = run_batch([
            "import_module math",
            "math.sqrt 9.0 -> 0",
            "math.sqrt data[0] -> 1",
        ])
        self.assertAlmostEqual(data[0], 3.0)
        self.assertAlmostEqual(data[1], 3.0 ** 0.5)

    def test_batch_result_position(self):
        from cos_comparison.shell_tool.batch import run_batch
        data, _stats = run_batch([
            "cos (1, 0) (1, 0) -> 7",
        ])
        self.assertIn(7, data)
        self.assertEqual(data[7], 1.0)

    def test_batch_comments_blank(self):
        from cos_comparison.shell_tool.batch import run_batch
        data, stats = run_batch([
            "# comment",
            "",
            "import_module math",
            "math.sqrt 4.0 -> 0",
        ])
        self.assertEqual(stats["run"], 2)
        self.assertEqual(data, {0: 2.0})

    def test_arrow_inside_quotes_is_verbatim(self):
        from cos_comparison.shell_tool.batch import run_batch
        data, _stats = run_batch(['let x "a->b"', "str data[0] -> 0"])
        self.assertEqual(data[0], "a->b")

    def test_batch_first_interrupt_continues(self):
        # first interrupt does not terminate the subprogram (it may have its own interrupt handler):
        # record the interrupt and continue executing the remaining commands
        from cos_comparison.shell_tool import shell as SH
        from cos_comparison.shell_tool.batch import run_batch
        def kb_cmd(*_a, **_k):
            raise KeyboardInterrupt()
        SH.register_callable("kb_cmd", kb_cmd)
        _data, stats = run_batch(["kb_cmd", "import_module math"])
        self.assertTrue(stats["interrupt"])
        self.assertEqual(stats["run"], 1)  # math command continues executing

    def test_batch_second_consecutive_forced(self):
        # second consecutive interrupt (within window) -> forced exit
        from cos_comparison.shell_tool import shell as SH
        from cos_comparison.shell_tool.batch import run_batch
        def kb_cmd(*_a, **_k):
            raise KeyboardInterrupt()
        SH.register_callable("kb_cmd", kb_cmd)
        with self.assertRaises(KeyboardInterrupt):
            run_batch(["kb_cmd", "kb_cmd"])

    def test_batch_loop_interrupt_finalizes_result(self):
        # a force-unwound loop still completes its result position / stats
        # (like a normal loop completion), per the documented policy
        from cos_comparison.shell_tool import shell as SH
        from cos_comparison.shell_tool.batch import run_batch

        def kb_body():
            raise KeyboardInterrupt()

        SH.register_callable("always_true", lambda: True)
        SH.register_callable("kb_body", kb_body)
        try:
            data, stats = run_batch(["WHILE always_true", "kb_body", "END"])
        finally:
            SH.ns_delete(["always_true", "kb_body"])
        self.assertEqual(data[0], 0)          # completed iterations
        self.assertTrue(stats["interrupt"])

    def test_batch_interrupt_not_accumulated(self):
        # interrupts do not accumulate persistently: outside the window (interrupt_window=0.0) they reset —
        # each interrupt never counts as a "second consecutive", so forced exit never occurs
        from cos_comparison.shell_tool import shell as SH
        from cos_comparison.shell_tool.batch import run_batch
        def kb_cmd(*_a, **_k):
            raise KeyboardInterrupt()
        SH.register_callable("kb_cmd", kb_cmd)
        _data, stats = run_batch(["kb_cmd", "kb_cmd"],
                                 interrupt_window=0.0)
        self.assertTrue(stats["interrupt"])
        self.assertEqual(stats["run"], 0)  # both interrupted, but not forced

    def test_batch_idle_interrupt_two_exits(self):
        # no subprogram running (line-read interrupt): two interrupts cause exit
        from cos_comparison.shell_tool.batch import run_batch
        class IdleKb:
            def __init__(self):
                self.count = 0
            def __iter__(self):
                return self
            def __next__(self):
                self.count += 1
                if self.count <= 2:
                    raise KeyboardInterrupt()
                raise StopIteration
        with self.assertRaises(KeyboardInterrupt):
            run_batch(IdleKb())


class TestUnifiedInstructions(ShellTestCase):
    """Unified instruction-file protocol: app / shell(run) / batch read the same format and uniformly use
    functional imperative control (IF/WHILE instruction functions + operation-flow function calls) —
    behaviour is consistent."""

    TEXT = (
        "core.add_chain (0,) -> 0\n"
        "WHILE lt3 data[0]\n"
        "    bump data[0] -> 0\n"
        "END\n"
        "add2 data[0] 10 -> 1\n"
    )

    def _register(self):
        SH.register_callable("lt3", lambda x: x < 3)
        SH.register_callable("bump", lambda x: x + 1)
        SH.register_callable("add2", lambda a, b: a + b)

    def _clean(self):
        SH.ns_delete(["lt3", "bump", "add2"])

    def test_consistent_across_executors(self):
        self._register()
        try:
            from cos_comparison.app import Docker
            d = Docker()
            d.operate_pool = d.read_operate_flow(self.TEXT)
            drv = d.run()
            self.assertTrue(drv.wait(timeout=5))

            from cos_comparison.shell_tool.batch import run_batch
            bdata, _ = run_batch(self.TEXT.splitlines())

            fd, path = tempfile.mkstemp(suffix=".txt")
            os.close(fd)
            try:
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(self.TEXT)
                sdata, _ = SH.run_file(path)
            finally:
                os.unlink(path)

            expected = {0: 3, 1: 13}
            self.assertEqual(d.data_pool, expected)
            self.assertEqual(bdata, expected)
            self.assertEqual(sdata, expected)
        finally:
            self._clean()

    def test_batch_control_flow_blocks(self):
        from cos_comparison.shell_tool.batch import run_batch
        data, _ = run_batch([
            "core.add_chain (1, 1) -> 0",
            "IF core.add_chain (1, 1)",
            "    core.add_chain (1, 2) -> 1",
            "ELSE",
            "    core.add_chain (9, 9) -> 1",
            "END",
        ])
        self.assertEqual(data, {0: 2, 1: 3})

    def test_run_shell_run_command(self):
        fd, path = tempfile.mkstemp(suffix=".txt")
        os.close(fd)
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("core.add_chain (1, 2) -> 0\n")
            result = SH.run_shell(["run", path])
            self.assertIn("1 instructions", result)
        finally:
            os.unlink(path)

    def test_while_loop_interrupt(self):
        # blocking and interrupt at loop end: first interrupt yields (subprogram may catch),
        # second consecutive within window forces exit — does not wrongly kill the subprogram
        n = {"v": 0}

        def kb_body():
            n["v"] += 1
            raise KeyboardInterrupt()

        SH.register_callable("always_true", lambda: True)
        SH.register_callable("kb_body", kb_body)
        try:
            with self.assertRaises(KeyboardInterrupt):
                SH.execute_call(*SH.parse_call("WHILE always_true kb_body"))
            self.assertEqual(n["v"], 2)  # first yields (continues); second forces
        finally:
            SH.ns_delete(["always_true", "kb_body"])

    def test_batch_cli_file(self):
        import subprocess
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        path = os.path.join(tmp, "flow.txt")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("import_module math\n"
                     "math.sqrt 25.0 -> 0\n")
        p = subprocess.run(
            [sys.executable, "-m", "cos_comparison", "batch", path],
            capture_output=True, text=True, env=testutil.clean_env(),
            cwd=os.path.dirname(os.path.abspath(__file__)),
            timeout=60, check=False)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("imported math", p.stdout)


class TestNoHardcodedPackageName(ShellTestCase):
    """shell_tool scripts must not hard-code the package name."""

    def test_shell_tool_no_package_name(self):
        import re

        from cos_comparison import shell_tool
        here = os.path.dirname(os.path.abspath(shell_tool.__file__))
        # package-name references: imports / dotted module paths (core API
        # function names like cos_comparison_passive are NOT package refs)
        pattern = re.compile(
            r"(import\s+cos_comparison|from\s+cos_comparison"
            r"|cos_comparison\.\w)")
        for fname in os.listdir(here):
            if not fname.endswith(".py"):
                continue
            with open(os.path.join(here, fname), encoding="utf-8") as fh:
                text = fh.read()
            self.assertNotRegex(
                text, pattern,
                f"{fname} hard-codes the package name")


if __name__ == "__main__":
    unittest.main()
