# -*- coding: utf-8 -*-
"""Shared helpers for the flat non-GUI cos_comparison test suite.

Environment hygiene (see the PYTHONPATH history): probes run with
``-E`` and a sanitised environment (no PYTHON* variables), so a
user-level PYTHONPATH can never shadow the installed package;
subprocesses also use a neutral scratch ``cwd`` so the implicit ``''``
entry on ``sys.path`` cannot shadow it with the source tree.
"""
import atexit
import json
import os
import shutil
import subprocess
import sys
import tempfile

BACKENDS = (".cos_comparison_pydll", ".cos_comparison")

PREFIX = (
    "import json,sys\n"
    "from cos_comparison import core\n"
)

_NEUTRAL_CWD = tempfile.mkdtemp(prefix="cc_test_cwd_")
atexit.register(shutil.rmtree, _NEUTRAL_CWD, ignore_errors=True)


def clean_env():
    """Environment for child probes: no PYTHON*/COS_COMPARISON_* entries,
    so a user-level PYTHONPATH or strict-mode flag cannot leak in."""
    env = {}
    for key, value in os.environ.items():
        if key.startswith("PYTHON") or key.startswith("COS_COMPARISON_"):
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
        env=clean_env(), cwd=_NEUTRAL_CWD,
    )
    return proc.returncode, proc.stdout, proc.stderr


def run_probe(backend_import, code, timeout=60):
    """Run *code* against one backend module imported directly (no
    core.set_mode round-trip).  Used by the empty/edge-input tables."""
    full = "%s; %s" % (backend_import, code)
    proc = subprocess.run(
        [sys.executable, "-E", "-c", full],
        capture_output=True, text=True, timeout=timeout,
        env=clean_env(), cwd=_NEUTRAL_CWD,
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
    """Fail loudly when the package under test is not the installed one.

    Accepts both regular installs (imported from site-packages) and pip
    editable installs (the installed package *is* the source tree; the
    ``__editable__*cos_comparison*`` marker in site-packages proves it)."""
    import cos_comparison
    path = os.path.dirname(os.path.abspath(cos_comparison.__file__))
    if "site-packages" in path:
        return path
    import glob
    import site
    for sp in site.getsitepackages():
        if glob.glob(os.path.join(sp, "__editable__*cos_comparison*")):
            return path
    raise AssertionError(
        "cos_comparison imported from %r - run with the venv_test "
        "interpreter and `-E`, away from the source tree" % path)
