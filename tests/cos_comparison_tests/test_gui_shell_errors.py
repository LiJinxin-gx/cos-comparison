# -*- coding: utf-8 -*-
"""
Regression tests for the GUI shell error reporting (test_image_gui.pyw).

Covers the two reported bugs in the Python Shell tab (not the Script tab):
  * an error message must appear exactly ONCE in the transcript - the old
    code wrote it twice (once from exec_code's stderr stream, once from
    the _submit re-render of the captured stderr)
  * the error output must contain a full Python traceback (the old code
    only printed ``TypeName: message``)

Drives the platform exclusively through its public interfaces
(create_platform / exec_code / run_complete_source / _submit), matching
how external drivers use the module.
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = os.path.dirname(HERE)
sys.path.insert(0, TESTS_DIR)

import test_image_gui as tp  # noqa: E402


class ShellErrorReportingTest(unittest.TestCase):
    """Shell / shared-namespace error output: single report + traceback."""

    def setUp(self):
        self.p = tp.create_platform()
        self.app = self.p.app
        self.p.pump()

    def tearDown(self):
        try:
            self.p.close()
        except Exception:
            pass

    def transcript(self):
        return self.app.shell.text.get('1.0', 'end-1c')

    def test_exec_code_runtime_error_has_traceback(self):
        res = self.p.exec_code("def boom():\n    return 1 / 0\nboom()")
        self.assertFalse(res['ok'])
        stderr = res['stderr']
        self.assertIn('Traceback (most recent call last)', stderr)
        self.assertTrue(stderr.rstrip().endswith(
            'ZeroDivisionError: division by zero'), stderr)

    def test_exec_code_error_reported_once_in_transcript(self):
        self.p.exec_code("def boom():\n    return 1 / 0\nboom()")
        self.p.pump()
        self.assertEqual(self.transcript().count('ZeroDivisionError'), 1)

    def test_run_complete_source_returns_traceback(self):
        res = self.app.shell.run_complete_source("raise ValueError('boom2')")
        self.assertFalse(res['ok'])
        self.assertIn('ValueError: boom2', res['stderr'])

    def test_shell_submit_syntax_error_not_duplicated(self):
        # keyboard-Enter path: _submit() must show the error only once
        self.app.shell._submit("x = = 1")
        self.p.pump()
        self.assertEqual(self.transcript().count('SyntaxError'), 1)

    def test_shell_submit_runtime_error_once_with_traceback(self):
        self.app.shell._submit("def f():\n    raise RuntimeError('kboom')")
        self.app.shell._submit("f()")
        self.p.pump()
        text = self.transcript()
        self.assertEqual(text.count('RuntimeError'), 1)
        self.assertIn('File "<injected>"', text)

    def test_script_tab_error_reported_once(self):
        # Script tab must keep working (single report + traceback)
        self.app.script_editor.delete('1.0', 'end')
        self.app.script_editor.insert('1.0', "print('script-out')\n1/0")
        self.app.run_script()
        self.p.pump()
        text = self.transcript()
        self.assertIn('script-out', text)
        self.assertEqual(text.count('ZeroDivisionError'), 1)

    def test_success_path_unaffected(self):
        res = self.p.exec_code("print('hello'); 40 + 2")
        self.assertTrue(res['ok'])
        self.assertEqual(res['stdout'].strip(), 'hello')
        self.assertEqual(res['stderr'].strip(), '')

    def test_user_stderr_not_duplicated(self):
        self.p.exec_code("import sys; print('my-err', file=sys.stderr)")
        self.p.pump()
        self.assertEqual(self.transcript().count('my-err'), 1)


if __name__ == '__main__':
    unittest.main()