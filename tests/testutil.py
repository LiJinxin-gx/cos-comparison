# -*- coding: utf-8 -*-
"""Shared helpers for the flat non-GUI cos_comparison test suite.

Environment hygiene (learned the hard way, see the PYTHONPATH history):
  * subprocess probes run with ``-E`` and an explicitly sanitised
    environment (no PYTHON* variables) so a user-level PYTHONPATH
    (e.g. an old cos_comparison on another drive) can never shadow the
    installed package;
  * subprocesses run with ``cwd`` set to a neutral scratch directory so
    the implicit ``''`` entry on ``sys.path`` can never shadow the
    installed package with the source tree.
"""
import json
import os
import subprocess
import sys
import tempfile

BACKENDS = (".cos_comparison_pydll", ".cos_comparison_c", ".cos_comparison")

PREFIX = (
    "import json,sys\n"
    "from cos_comparison import core\n"
)

_NEUTRAL_CWD = tempfile.mkdtemp(prefix="cc_test_cwd_")


def _clean_env():
    env = {}
    for key, value in os.environ.items():
        if key.startswith("PYTHON"):
            continue
        env[key] = value
    return env


def run_backend(backend, body, timeout=120):
    """Run *body* (python source) in a fresh interpreter pinned to
    *backend*.  Returns (exit_code, stdout, stderr)."""
    code = PREFIX + "core.set_mode(%r)\n" % backend + body
    proc = subprocess.run(
        [sys.executable, "-E", "-c", code],
        capture_output=True, text=True, timeout=timeout,
        env=_clean_env(), cwd=_NEUTRAL_CWD,
    )
    return proc.returncode, proc.stdout, proc.stderr


def run_probe(backend_import, code, timeout=60):
    """Run *code* against one backend module imported directly (no
    core.set_mode round-trip).  Used by the empty/edge-input tables."""
    full = "%s; %s" % (backend_import, code)
    proc = subprocess.run(
        [sys.executable, "-E", "-c", full],
        capture_output=True, text=True, timeout=timeout,
        env=_clean_env(), cwd=_NEUTRAL_CWD,
    )
    return proc.returncode, proc.stdout, proc.stderr


def json_result(out):
    for line in out.splitlines():
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def check_local_env():
    """Fail loudly when the installed package is not the one under test."""
    import cos_comparison
    path = os.path.dirname(os.path.abspath(cos_comparison.__file__))
    if "site-packages" not in path:
        raise AssertionError(
            "cos_comparison imported from %r - run with the venv_test "
            "interpreter and `-E`, away from the source tree" % path)
    return path
