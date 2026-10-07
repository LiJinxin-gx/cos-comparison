"""__main__ CLI tests: directory-searched plugins via runpy, help,
namespace injection, KeyboardInterrupt monitoring, error interception."""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import cos_comparison.__main__ as CLI

import testutil


class TestUsage(unittest.TestCase):
    def _help_output(self, argv):
        with mock.patch("sys.stdout") as fake_out:
            rc = CLI.main(argv)
        text = "".join(str(call.args[0])
                       for call in fake_out.write.call_args_list)
        return rc, text

    def test_usage_help(self):
        rc, text = self._help_output(["-h"])
        self.assertEqual(rc, 0)
        self.assertIn("usage", text)

    def test_usage_no_args(self):
        rc, text = self._help_output([])
        self.assertEqual(rc, 0)
        self.assertIn("call page", text)

    def test_usage_long_help(self):
        rc, text = self._help_output(["--help"])
        self.assertEqual(rc, 0)
        self.assertIn("usage", text)

    def test_usage_help_word(self):
        rc, text = self._help_output(["help"])
        self.assertEqual(rc, 0)
        self.assertIn("usage", text)


class TestPluginSearch(unittest.TestCase):
    def test_plugin_path_exists(self):
        self.assertIsNotNone(CLI.plugin_path("version"))
        self.assertIsNotNone(CLI.plugin_path("shell"))

    def test_plugin_path_unknown(self):
        self.assertIsNone(CLI.plugin_path("no_such_command"))

    def test_plugin_path_rejects_path_components(self):
        self.assertIsNone(CLI.plugin_path("../__init__"))
        self.assertIsNone(CLI.plugin_path("sub/version"))
        self.assertIsNone(CLI.plugin_path(".hidden"))
        self.assertIsNone(CLI.plugin_path(""))

    def test_value_is_not_a_command(self):
        # value.py is a shared helper module (no __main__ guard)
        self.assertIsNone(CLI.plugin_path("value"))
        self.assertNotIn("value", CLI.list_commands())

    def test_list_commands(self):
        cmds = CLI.list_commands()
        self.assertIn("version", cmds)
        self.assertIn("shell", cmds)


class TestRunCommand(unittest.TestCase):
    def setUp(self):
        self.addCleanup(CLI._NAMESPACE.clear)

    def _write_plugin(self, name, source):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        path = os.path.join(tmp, name + ".py")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(source)
        return path

    def test_unknown_command(self):
        with mock.patch("sys.stdout") as fake_out:
            rc = CLI.run_command("nope")
        self.assertEqual(rc, 1)
        text = "".join(str(call.args[0])
                       for call in fake_out.write.call_args_list)
        self.assertIn("unknown command", text)

    def test_namespace_injected(self):
        path = self._write_plugin("nscheck",
                                  "globals()['__ns__']['marker'] = 42\n")
        with mock.patch.object(CLI, "plugin_path", return_value=path):
            rc = CLI.run_command("nscheck")
        self.assertEqual(rc, 0)
        self.assertEqual(CLI._NAMESPACE.get("marker"), 42)

    def test_kb_returns_130(self):
        path = self._write_plugin("kb", "raise KeyboardInterrupt()\n")
        with mock.patch.object(CLI, "plugin_path", return_value=path):
            rc = CLI.run_command("kb")
        self.assertEqual(rc, 130)

    def test_error_intercepted(self):
        path = self._write_plugin("boom", "raise ValueError('boom')\n")
        with mock.patch.object(CLI, "plugin_path", return_value=path):
            rc = CLI.run_command("boom")
        self.assertEqual(rc, 1)

    def test_success_returns_0(self):
        path = self._write_plugin("ok", "pass\n")
        with mock.patch.object(CLI, "plugin_path", return_value=path):
            rc = CLI.run_command("ok")
        self.assertEqual(rc, 0)

    def test_sys_exit_int_code(self):
        path = self._write_plugin("code", "import sys\nsys.exit(7)\n")
        with mock.patch.object(CLI, "plugin_path", return_value=path):
            rc = CLI.run_command("code")
        self.assertEqual(rc, 7)

    def test_sys_exit_message_code(self):
        # Python semantics: a string exit code prints the message and
        # exits 1 (never 0)
        import contextlib
        import io
        path = self._write_plugin(
            "msg", "import sys\nsys.exit('boom-message')\n")
        err = io.StringIO()
        with mock.patch.object(CLI, "plugin_path", return_value=path), \
                contextlib.redirect_stderr(err):
            rc = CLI.run_command("msg")
        self.assertEqual(rc, 1)
        self.assertIn("boom-message", err.getvalue())

    def test_shell_runs_share_namespace(self):
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc1 = CLI.run_command("shell", ["let", "cli_shared", "7"])
            rc2 = CLI.run_command("shell", ["get", "cli_shared"])
        self.assertEqual((rc1, rc2), (0, 0))
        self.assertIn("7", buf.getvalue())
        self.assertNotIn("undefined", buf.getvalue())


class TestMainCLI(unittest.TestCase):
    def _run(self, *args, input=None):
        return subprocess.run(
            [sys.executable, "-m", "cos_comparison", *args],
            capture_output=True, text=True, env=testutil.clean_env(),
            cwd=os.path.dirname(os.path.abspath(__file__)),
            timeout=60, check=False, input=input)

    def test_version_cli(self):
        p = self._run("version")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertRegex(p.stdout.strip(), r"\d+\.\d+")

    def test_shell_cli(self):
        p = self._run("shell", "cos", "(1, 0)", "(1, 0)")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(p.stdout.strip(), "1.0")

    def test_shell_cli_bare_enters_command_line(self):
        # no arguments: the shell command line (EOF ends it), matching
        # `python -m cos_comparison sh` behaviour
        p = self._run("shell", input="")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("shell command line", p.stdout)

    def test_help_cli(self):
        p = self._run("-h")
        self.assertEqual(p.returncode, 0)
        self.assertIn("usage", p.stdout)

    def test_no_args_cli(self):
        # no arguments: enter the interactive call page (EOF ends it)
        p = self._run(input="")
        self.assertEqual(p.returncode, 0)
        self.assertIn("call page", p.stdout)
        self.assertIn(">>> ", p.stdout)

    def test_unknown_command_cli(self):
        p = self._run("bogus_cmd")
        self.assertEqual(p.returncode, 1)
        self.assertIn("unknown command", p.stdout)

    def test_error_intercepted_cli(self):
        p = self._run("shell", "no.such.func", "1")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("error:", p.stdout)


class TestInteractivePage(unittest.TestCase):
    """python -m cos_comparison call page: >>> commands, ... nested."""

    def _repl(self, script):
        return subprocess.run(
            [sys.executable, "-m", "cos_comparison"],
            input=script, capture_output=True, text=True,
            env=testutil.clean_env(),
            cwd=os.path.dirname(os.path.abspath(__file__)),
            timeout=60, check=False)

    def test_enter_call_page(self):
        p = self._repl("exit\n")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("call page", p.stdout)
        self.assertIn(">>> ", p.stdout)

    def test_help_command(self):
        p = self._repl("help\nexit\n")
        self.assertIn("interactive call page", p.stdout)
        self.assertIn(">>> command", p.stdout)

    def test_command_execution(self):
        p = self._repl("version\nexit\n")
        self.assertRegex(p.stdout, r">>> \d+\.\d+")

    def test_command_with_args(self):
        p = self._repl("shell cos (1, 0) (1, 0)\nexit\n")
        self.assertRegex(p.stdout, r">>> 1\.0")

    def test_shell_nested_prompt(self):
        """shell falls into its own command line operation shown with ..."""
        p = self._repl("shell\ncos (1, 0) (1, 0)\nexit\nexit\n")
        self.assertIn("shell command line", p.stdout)
        self.assertIn("... 1.0", p.stdout)

    def test_shell_nested_namespace(self):
        p = self._repl("shell\nlet x 5\nget x\nexit\nexit\n")
        self.assertIn("... 5", p.stdout)
        self.assertNotIn("undefined", p.stdout)

    def test_shell_namespace_shared_with_page_commands(self):
        # a shell run with arguments and the nested shell command line
        # share the same namespace (the injected __ns__ mapping)
        p = self._repl("shell let shared 9\nshell\nget shared\nexit\nexit\n")
        self.assertIn("... 9", p.stdout)

    def test_nested_returns_to_outer(self):
        p = self._repl("shell\nexit\nversion\nexit\n")
        self.assertRegex(p.stdout, r">>> \d+\.\d+")

    def test_unknown_command_in_page(self):
        p = self._repl("bogus\nexit\n")
        self.assertIn("unknown command", p.stdout)

    def test_exit(self):
        p = self._repl("exit\n")
        self.assertEqual(p.returncode, 0, p.stderr)


if __name__ == "__main__":
    unittest.main()
