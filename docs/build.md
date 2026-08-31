# Building cos-comparison

Build instructions for the three backends, dual-compiler strict-mode
checking, and the free-threaded (cp314t) build.

## Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Build the C Backends](#build-the-c-backends)
- [Install](#install)
- [Dual-Compiler Strict-Mode Checking](#dual-compiler-strict-mode-checking)
- [Free-Threaded (cp314t) Build](#free-threaded-cp314t-build)
- [Version Management](#version-management)
- [Tests](#tests)
- [Troubleshooting](#troubleshooting)

---

## Overview

The core ships three interchangeable backends (auto-fallback in priority
order):

| Backend | Implementation | Notes |
|---------|----------------|-------|
| **pydll** | C extension (`cos_comparison_pydll`, built by `setup.py`) | Fastest; declares `Py_MOD_GIL_NOT_USED` (free-threaded compatible) |
| **ctypes** | Pure C shared library (`core.dll` / `core.so` / `core.dylib`) | C99, no Python C API; loadable by any interpreter |
| **pure Python** | `cos_comparison/core/cos_comparison.py` | Zero dependencies; always the final fallback |

`setup.py` compiles the pydll extension and the ctypes shared library.
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

## Build the C Backends

In-place build (writes the extension and the shared library into the
source tree, e.g. for development):

```bash
python setup.py build_ext -f --inplace
```

Expected artifacts:

```
cos_comparison/core/cos_comparison_pydll.cp3XX-*.pyd   # pydll backend
cos_comparison/core/cos_comparison_c/core.dll           # ctypes backend
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

The C sources are checked with both MSVC (via `setup.py build_ext`) and
a strict-mode GCC (e.g. the MinGW-w64 toolchain from RedPanda C++).
The GCC checks below use `-std=c99 -Wall -Wextra -pedantic`.

### pydll extension (syntax check; needs the CPython headers)

```bash
gcc -std=c99 -Wall -Wextra -pedantic \
    -Wno-unused-parameter -Wno-cast-function-type \
    -fsyntax-only \
    -I <python-include-dir> \
    -I cos_comparison/core/include \
    cos_comparison/core/include/cos_comparison_pydll.c
```

### ctypes shared library (full compile)

```bash
gcc -std=c99 -Wall -Wextra -pedantic \
    -Wno-unused-parameter -Wno-cast-function-type \
    -shared -O2 \
    -I cos_comparison/core/cos_comparison_c/include \
    cos_comparison/core/cos_comparison_c/include/core.c \
    -o core_check.dll
```

Notes:

- `-Wno-unused-parameter` / `-Wno-cast-function-type` suppress CPython
  extension idioms (getter signatures, `(PyCFunction)` method-table
  casts); with them, strict mode reports zero warnings.
- The `PyModuleDef_Slot` function-pointer cast is a CPython API
  requirement and cannot be removed.
- The MSVC build is the primary one on Windows; both compilers must pass.

---

## Free-Threaded (cp314t) Build

The free-threaded interpreter (no GIL, e.g. Python 3.14t) requires the
extension to carry the `t` ABI tag (`cp314t`). Build with the
free-threaded interpreter itself — its `Python.h` defines
`Py_GIL_DISABLED` automatically:

```bash
<python3.14t> -m pip install setuptools        # if missing
<python3.14t> setup.py build_ext -f --inplace
```

Expected artifact:

```
cos_comparison/core/cos_comparison_pydll.cp314t-*.pyd
```

The pydll backend declares `Py_MOD_GIL_NOT_USED`, so the code is
free-threaded compatible; only the build tag changes. The ctypes shared
library has no ABI tag (pure C) and is loadable by any interpreter.

> A regular (`cp314`) pydll cannot be loaded by the free-threaded
> interpreter; the core then falls back to ctypes / pure Python.

---

## Version Management

`pyproject.toml` is the single source of truth for the version.
`setup.py` extracts it (regex, Python 3.8+ compatible, no `tomllib`)
and writes `cos_comparison/VERSION.txt`, which the package reads at
runtime:

```python
import cos_comparison
print(cos_comparison.__version__)      # e.g. 0.4.4
print(cos_comparison.version_tuple)    # (0, 4, 4)
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
| `get_mode()` shows only `.cos_comparison` | Compilation failed and the package fell back; check the compiler output |
| Free-threaded interpreter cannot load pydll | The regular (`cp314`) tag is not loadable; build the cp314t variant |
| `test_imports` fails with "run with the venv_test interpreter and `-E`, away from the source tree" | Run it from a neutral directory against the installed package |
| Subprocess tests behave like an old build | Reinstall with `--force-reinstall --no-deps .` |

---

**Need help?** [Open an issue](https://github.com/LiJinxin-gx/cos-comparison/issues)
