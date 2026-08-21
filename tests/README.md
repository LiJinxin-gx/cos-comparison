# cos-comparison Test Suite

## Overview

The test suite covers all three backends (pure Python, ctypes C, C extension) and verifies behavioral parity. Tests use only `unittest` from the standard library — no pytest required.

## Test Files

| File | Scope | How to Run |
|------|-------|-----------|
| `test_core_algorithms.py` | Passive/active mode, statistics, callbacks, arithmetic, chain compute | in-process |
| `test_core_data.py` | `infer_shape`, `load_as_default_data`, `load_data`, buffer protocol | in-process |
| `test_core_tensor.py` | Tensor creation, indexing, slicing, iteration, views, serialization | in-process |
| `test_core_backends.py` | Cross-backend parity and documented divergences (subprocess isolation) | in-process* |
| `test_empty_edge.py` | Empty inputs, zero dimensions, invalid args across all backends | subprocess |
| `test_normal_consistency.py` | Random-data three-backend result comparison | subprocess |
| `test_layers.py` | Upper layers: sense, memory, brain, action, generate, interface, data | in-process |
| `test_topology.py` | Directed graph (topology): DirectedGraph counts, degrees, weak/strong connectivity, topological sort, reachability, Eulerian tests, recursion-free | in-process |
| `test_random_tool.py` | Random tools: abstract base, default random backend, secure secrets backend, generation/shuffle/choice/sample/weighted choices, seeding | in-process |
| `test_imports.py` | All subpackages import cleanly; public API exports verified | neutral dir |
| `test_no_gui.py` | No GUI framework is imported by the package or test suite | in-process |
| `test_image_gui.pyw` | Manual GUI visual test (excluded from automated runs) | manual |
| `testutil.py` | Shared helpers: subprocess isolation, backend switching, JSON parsing | library |

\* `test_core_backends.py` needs `tests/` on `sys.path` (it imports `testutil`).

## Running Tests

From the project root, using the venv_test interpreter:

```powershell
$py = Join-Path $PWD "venv_test\Scripts\python.exe"
$testsDir = Join-Path $PWD "tests"

# In-process tests (fast)
& $py -c "import unittest, sys; loader=unittest.TestLoader(); suite=unittest.TestSuite();
[suite.addTests(loader.loadTestsFromName(f'tests.{m}')) for m in
 ['test_core_algorithms','test_core_data','test_core_tensor','test_layers']];
sys.exit(0 if unittest.TextTestRunner(verbosity=1).run(suite).wasSuccessful() else 1)"

# Backend parity (needs tests/ on path)
& $py -c "import sys, unittest; sys.path.insert(0,'tests');
sys.exit(0 if unittest.TextTestRunner(verbosity=1).run(
unittest.TestLoader().loadTestsFromName('test_core_backends')).wasSuccessful() else 1)"

# Subprocess-isolated edge cases
& $py tests/test_empty_edge.py
& $py tests/test_normal_consistency.py

# Import tests (run from a neutral directory so site-packages is used)
Push-Location $env:TEMP
& $py -E -c "import sys, unittest; sys.path.insert(0, r'$testsDir');
sys.exit(0 if unittest.TextTestRunner(verbosity=1).run(
unittest.TestLoader().loadTestsFromName('test_imports')).wasSuccessful() else 1)"
Pop-Location
```

## Test Design Principles

1. **Backend parity**: all three backends must produce identical results for valid inputs. `test_core_backends.py` and `test_normal_consistency.py` enforce this.
2. **Subprocess isolation**: tests that could crash (segfaults, heap corruption) run each backend in a separate subprocess via `testutil.run_backend()`.
3. **No external dependencies**: tests use only stdlib. The optional `pillow` dependency is only needed for image/GUI tests.
4. **Documented divergences**: known backend differences are explicitly documented in `TestKnownDivergences`, not silently ignored.
5. **No GUI in automated runs**: `test_image_gui.pyw` is a manual visual test; `test_no_gui.py` verifies no GUI framework is imported.
