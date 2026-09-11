# Building cos-comparison

Build instructions for the C extension backend, dual-compiler strict-mode
checking, and the free-threaded (cp314t) build.

## Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Build the C Backend](#build-the-c-backend)
- [Install](#install)
- [Dual-Compiler Strict-Mode Checking](#dual-compiler-strict-mode-checking)
- [Free-Threaded (cp314t) Build](#free-threaded-cp314t-build)
- [Version Management](#version-management)
- [Tests](#tests)
- [Troubleshooting](#troubleshooting)

---

## Overview

The core ships two interchangeable backends (auto-fallback in priority
order):

| Backend | Implementation | Notes |
|---------|----------------|-------|
| **C extension** | `cos_comparison_pydll`, built by `setup.py` | Fastest; declares `Py_MOD_GIL_NOT_USED` (free-threaded compatible) |
| **pure Python** | `cos_comparison/core/cos_comparison.py` | Zero dependencies; always the final fallback |

`setup.py` compiles the C extension (and the `math_tool` C extensions:
`_topology` / `_fourier` / `_linear_algebra` / `_unit_map`).
If compilation fails, the package gracefully falls back to pure Python.

---

## Prerequisites

- Python 3.8+ (CPython). A free-threaded build (3.13+/3.14t) additionally
  requires the free-threaded interpreter (see below).
- A C compiler:
  - Windows: MSVC (Visual Studio Build Tools) or MinGW-w64 (e.g. the
    toolchain shipped with RedPanda C++);
  - Linux/macOS: `gcc` / `clang`.
- `setuptools` (comes with most distributions; install with
  `python -m pip install setuptools` if missing).

---

## Build the C Backend

In-place build (writes the extension into the source tree, e.g. for
development):

```bash
python setup.py build_ext -f --inplace
```

Expected artifact:

```
cos_comparison/core/cos_comparison_pydll.cp3XX-*.pyd   # C extension backend
```

---

## Install

Build and install into the current environment:

```bash
python -m pip install .
```

Install from PyPI (the wheel compiles the C backends, or falls back to
pure Python):

```bash
python -m pip install cos-comparison
```

Verify the active backends:

```python
from cos_comparison import core as cc
print(cc.get_mode())          # enabled backends in priority order
print(cc.get_available_backends())
```

---

## Dual-Compiler Strict-Mode Checking

The C sources are checked with three compilers: MSVC (`/Wall /WX`), a
strict-mode MinGW-w64 GCC (e.g. the toolchain from RedPanda C++) and a
strict-mode Linux GCC (e.g. WSL Ubuntu).  The GCC checks below use
`-std=c99 -Wall -Wextra -Wpedantic -Werror`.

### C extensions (syntax check; needs the CPython headers)

```bash
gcc -std=c99 -Wall -Wextra -Wpedantic -Werror \
    -Wno-unused-parameter -Wno-cast-function-type \
    -Wno-missing-field-initializers -Wno-error=pedantic \
    -fsyntax-only \
    -I <python-include-dir> \
    -I cos_comparison/core/include \
    cos_comparison/core/include/cos_comparison_pydll.c
# math_tool extensions: same flags with
#   -I cos_comparison/interface/tools/math_tool/include
```

Notes:

- `-Wno-unused-parameter` / `-Wno-cast-function-type` suppress CPython
  extension idioms (getter signatures, `(PyCFunction)` method-table
  casts).
- Two CPython API patterns are inherently non-ISO and are downgraded:
  the `PyModuleDef_Slot` function-pointer cast (`(void*)module_exec`,
  a PEP 573 requirement — `-Wno-error=pedantic`) and partial
  `PyTypeObject` initializers (`-Wno-missing-field-initializers`).
  Every other warning is an error.
- The MSVC build is the primary one on Windows; all three compilers
  must pass.

---

## Free-Threaded (cp314t) Build

All C extensions (the core backend and the math_tool extensions) declare
`Py_MOD_GIL_NOT_USED` under `#if PY_VERSION_HEX >= 0x030D0000`: they run
without the GIL on free-threaded builds, while older interpreters
(< 3.13) simply skip the slot (reliable fallback).  Only the build tag
changes:

```bash
<python3.14t> -m pip install setuptools        # if missing
<python3.14t> setup.py build_ext -f --inplace
```

Expected artifacts:

```
cos_comparison/core/cos_comparison_pydll.cp314t-*.pyd
cos_comparison/interface/tools/math_tool/*.cp314t-*.pyd
```

> A regular (`cp314`) extension cannot be loaded by the free-threaded
> interpreter; the package then falls back to pure Python.

---

## Version Management

`pyproject.toml` is the single source of truth for the version.
`setup.py` extracts it (regex, Python 3.8+ compatible, no `tomllib`)
and writes `cos_comparison/VERSION.txt`, which the package reads at
runtime:

```python
import cos_comparison
print(cos_comparison.__version__)      # e.g. 0.5.0
print(cos_comparison.version_tuple)    # (0, 5, 0)
```

---

## Tests

Run the full suite from the source tree:

```bash
python -m unittest discover -s tests
```

Notes:

- Tests that probe backends in fresh subprocesses
  (`test_empty_edge`, `test_core_backends`, ...) use the *installed*
  package (site-packages). After changing C code, reinstall first:

  ```bash
  python -m pip install --force-reinstall --no-deps .
  ```

- `test_imports` is designed to run against the installed package from a
  neutral working directory (away from the source tree):

  ```bash
  cd <neutral-dir>
  python -m unittest discover -s <repo>/tests -p "test_imports.py"
  ```

- Run a single module: `python -m unittest tests.test_core_algorithms`.

---

## Troubleshooting

| Symptom | Cause / Fix |
|---------|-------------|
| `ImportError: no backend available` | Installation failed; `python -m pip install .` again, or rely on pure Python |
| `get_mode()` shows only the pure Python fallback | Compilation failed and the package fell back; check the compiler output |
| Free-threaded interpreter cannot load the C extension | The regular (`cp314`) tag is not loadable; build the cp314t variant |
| `test_imports` fails with "run with the venv_test interpreter and `-E`, away from the source tree" | Run it from a neutral directory against the installed package |
| Subprocess tests behave like an old build | Reinstall with `--force-reinstall --no-deps .` |

---

**Need help?** [Open an issue](https://github.com/LiJinxin-gx/cos-comparison/issues)
