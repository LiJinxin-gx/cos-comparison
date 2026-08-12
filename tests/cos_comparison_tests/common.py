# -*- coding: utf-8 -*-
"""
Shared helpers for the cos_comparison test suite.

* Subprocess runner: each backend is exercised in a *fresh* interpreter
  because switching backends inside one process leaves objects of the
  previous backend behind, which the new backend's dispatch then rejects
  (TypeError: takes at most 19 arguments (24 given)).
* The suite must be run with ``python -E`` so a user-level PYTHONPATH
  (e.g. D:\\pysite with cos-comparison 0.3.10) does not shadow the
  installed package.
"""

import json
import os
import subprocess
import sys

BACKENDS = (".cos_comparison_pydll", ".cos_comparison_c", ".cos_comparison")

PREFIX = (
    "import json,sys\n"
    "from cos_comparison import core\n"
)


def run_backend(backend, body, timeout=120):
    """Run *body* (python source) in a fresh interpreter pinned to
    *backend*.  Returns (exit_code, stdout, stderr).  `core.set_mode`
    is done before anything else; the script body can rely on
    ``core`` being imported and pinned.
    """
    code = PREFIX + "core.set_mode(%r)\n" % backend + body
    proc = subprocess.run(
        [sys.executable, "-E", "-c", code],
        capture_output=True, text=True, timeout=timeout,
    )
    return proc.returncode, proc.stdout, proc.stderr


def run_backend_result(backend, expr, timeout=120):
    """Run *expr* on *backend*, expecting it to print a JSON line with
    the result under key 'ok' (plus optional 'detail').  Raises
    AssertionError on failure and returns the parsed dict."""
    body = (
        "try:\n"
        "    result = %s\n"
        "except Exception as e:\n"
        "    print(json.dumps({'ok': False, 'err': type(e).__name__,\n"
        "                      'msg': str(e)}))\n"
        "else:\n"
        "    print(json.dumps({'ok': True, 'r': result}))\n"
    ) % expr
    code, out, err = run_backend(backend, body, timeout)
    if code != 0:
        raise AssertionError(
            "subprocess crashed (code %d): %s" % (code, err.strip()[-800:]))
    for line in out.splitlines():
        if line.startswith("{"):
            return json.loads(line)
    raise AssertionError("no JSON result line; stdout=%r stderr=%r"
                         % (out[-500:], err[-500:]))


def subprocess_ok(body, timeout=120):
    """Run *body* (no forced backend) and return exit code."""
    proc = subprocess.run(
        [sys.executable, "-E", "-c", body],
        capture_output=True, text=True, timeout=timeout,
    )
    return proc.returncode, proc.stdout, proc.stderr


def check_local_env():
    """Sanity guard: fail loudly when the running interpreter is not the
    expected clean environment (source tree shadowing or PYTHONPATH
    pollution would make the whole suite meaningless)."""
    import cos_comparison
    path = os.path.dirname(os.path.abspath(cos_comparison.__file__))
    if "site-packages" not in path:
        raise AssertionError(
            "cos_comparison imported from %r - run with the venv_test "
            "interpreter and `-E`, away from the source tree" % path)
    return path
