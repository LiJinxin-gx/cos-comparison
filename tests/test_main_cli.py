"""__main__ CLI tests: directory-searched plugins via runpy, help,
namespace injection, KeyboardInterrupt monitoring, error interception."""
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import cos_comparison.__main__ as CLI


class TestUsage(unittest.TestCase):
    def test_usage_help(self):
        with mock.patch("sys.stdout") as fake_out:
            rc = CLI.main(["-h"])
        self.assertEqual(rc, 0)
        fake_out.write.assert_called()

    def test_usage_no_args(self):
        with mock.patch("sys.stdout") as fake_out:
            rc = CLI.main([])
        self.assertEqual(rc, 0)
        fake_out.write.assert_called()

    def test_usage_long_help(self):
        with mock.patch("sys.stdout") as fake_out:
            rc = CLI.main(["--help"])
        self.assertEqual(rc, 0)
        fake_out.write.assert_called()

    def test_usage_help_word(self):
        with mock.patch("sys.stdout") as fake_out:
            rc = CLI.main(["help"])
        self.assertEqual(rc, 0)
        fake_out.write.assert_called()


class TestPluginSearch(unittest.TestCase):
    def test_plugin_path_exists(self):
        self.assertIsNotNone(CLI.plugin_path("version"))
        self.assertIsNotNone(CLI.plugin_path("shell"))

    def test_plugin_path_unknown(self):
        self.assertIsNone(CLI.plugin_path("no_such_command"))

    def test_list_commands(self):
        cmds = CLI.list_commands()
        self.assertIn("version", cmds)
        self.assertIn("shell", cmds)


class TestRunCommand(unittest.TestCase):
    def test_unknown_command(self):
        with mock.patch("sys.stdout") as fake_out:
            rc = CLI.run_command("nope")
        self.assertEqual(rc, 1)
        fake_out.write.assert_called()

    def test_namespace_injected(self):
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "nscheck.py")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("globals()['__ns__']['marker'] = 42\n")
        with mock.patch.object(CLI, "plugin_path", return_value=path):
            rc = CLI.run_command("nscheck")
        self.assertEqual(rc, 0)
        self.assertEqual(CLI._NAMESPACE.get("marker"), 42)
        CLI._NAMESPACE.clear()

    def test_kb_returns_130(self):
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "kb.py")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("raise KeyboardInterrupt()\n")
        with mock.patch.object(CLI, "plugin_path", return_value=path):
            rc = CLI.run_command("kb")
        self.assertEqual(rc, 130)

    def test_error_intercepted(self):
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "boom.py")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("raise ValueError('boom')\n")
        with mock.patch.object(CLI, "plugin_path", return_value=path):
            rc = CLI.run_command("boom")
        self.assertEqual(rc, 1)

    def test_success_returns_0(self):
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "ok.py")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("pass\n")
        with mock.patch.object(CLI, "plugin_path", return_value=path):
            rc = CLI.run_command("ok")
        self.assertEqual(rc, 0)


class TestMainCLI(unittest.TestCase):
    def _run(self, *args):
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [os.path.dirname(os.path.abspath(__file__)),
             env.get("PYTHONPATH", "")])
        return subprocess.run(
            [sys.executable, "-m", "cos_comparison", *args],
            capture_output=True, text=True, env=env, timeout=60,
            check=False)

    def test_version_cli(self):
        p = self._run("version")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertRegex(p.stdout.strip(), r"\d+\.\d+")

    def test_shell_cli(self):
        p = self._run("shell", "cos", "(1, 0)", "(1, 0)")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(p.stdout.strip(), "1.0")

    def test_help_cli(self):
        p = self._run("-h")
        self.assertEqual(p.returncode, 0)
        self.assertIn("usage", p.stdout)

    def test_no_args_cli(self):
        # no arguments: enter the interactive call page (EOF ends it)
        p = self._run()
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
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [os.path.dirname(os.path.abspath(__file__)),
             env.get("PYTHONPATH", "")])
        return subprocess.run(
            [sys.executable, "-m", "cos_comparison"],
            input=script, capture_output=True, text=True, env=env,
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
        self.assertIn("... 5", p.stdout)

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
