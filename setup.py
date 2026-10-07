"""
setup.py – Build and install cos_comparison with optional C acceleration.

This script compiles:
1. The Python C extension (cos_comparison_pydll) – placed inside cos_comparison/core/
2. The math_tool C extensions (_topology / _fourier / _linear_algebra / _unit_map)

Robust fallbacks keep the installation working even when compilation or
version injection fails: every extension is built in isolation (one
failure skips that extension only), an extension that rejects the
compiler standard flags is retried once without them (low-version
toolchains), and a version-file injection failure leaves the package
installable (the runtime then falls back to a default version).  The
ctypes backend was removed in v0.5.0: the compiled extension IS the C
backend (call name "c" in core/config.json).
"""

import os
import sys
import platform
import shutil

import re
import setuptools
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

# Change to the directory of this script so that all relative paths work correctly
setup_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(setup_dir)

# ----------------------------------------------------------------------
#  Version management: extract version from pyproject.toml (regex,
#  Python 3.8+ compatible, no tomllib) and write cos_comparison/VERSION
#  so the package can expose the single source-of-truth version at runtime.
#  Failure is never fatal: the runtime falls back to a default version.
# ----------------------------------------------------------------------
def _write_version_file():
    pyproject_path = os.path.join(setup_dir, "pyproject.toml")
    version_path = os.path.join(setup_dir, "cos_comparison", "VERSION.txt")
    try:
        # utf-8-sig tolerates a BOM in pyproject.toml (a BOM would otherwise
        # break the ^version regex anchor).
        with open(pyproject_path, encoding="utf-8-sig") as text_file:
            text = text_file.read()
        match = re.search(r"^version\s*=\s*[\"']([^\"']+)[\"']", text, re.MULTILINE)
        if not match or not match.group(1):
            print("Warning: no version found in pyproject.toml; "
                  "the package will fall back to the runtime default.")
            return None
        version_str = match.group(1).strip()
        # write the pure version string: no trailing newline, no BOM
        with open(version_path, "w", encoding="utf-8", newline="") as version_file:
            version_file.write(version_str)
        print("VERSION file written: {0}".format(version_str))
        return version_str
    except Exception as exc:  # noqa: BLE001 - never break the build
        print("Warning: skipping VERSION file generation ({0}); "
              "the package will fall back to the runtime default.".format(exc))
        return None

try:
    _write_version_file()
except Exception as e:  # noqa: BLE001 - never break the build
    print(f"Warning: version injection failed, continuing without it ({e})")

# Platform-specific compile arguments
is_windows = platform.system() == "Windows"
if is_windows:
    compile_args = ["/O2", "/std:c11"]
    math_libs = []  # Math functions are in msvcrt.dll on Windows, no separate lib needed
else:
    compile_args = ["-O2", "-std=c99"]
    math_libs = ["m"]  # Explicitly link libm on Linux/macOS for math functions

# ----------------------------------------------------------------------
#  Extension factory with construction-failure fallback: a bad extension
#  definition is skipped (never breaks the installation).
# ----------------------------------------------------------------------
def _make_extension(name, source, include_dir, libs=None):
    try:
        return Extension(
            name=name,
            sources=[source],
            include_dirs=[include_dir],
            extra_compile_args=compile_args,
            libraries=libs if libs is not None else math_libs,
        )
    except Exception as exc:  # noqa: BLE001 - construction must not break setup
        print("Warning: extension {0} could not be configured ({1}); "
              "skipping it.".format(name, exc))
        return None


ext_modules = []

# ----------------------------------------------------------------------
#  Python C extension (the C backend: cos_comparison_pydll)
# ----------------------------------------------------------------------
c_source_abs = os.path.abspath("cos_comparison/core/include/cos_comparison_pydll.c")
c_source_rel = os.path.relpath(c_source_abs, setup_dir)
c_include_dir = os.path.relpath(os.path.abspath("cos_comparison/core/include"), setup_dir)

if os.path.isfile(c_source_rel):
    ext = _make_extension("cos_comparison.core.cos_comparison_pydll",
                          c_source_rel, c_include_dir)
    if ext is not None:
        ext_modules.append(ext)
        print("Python C extension (cos_comparison_pydll) configured.")
else:
    print("Warning: C backend source not found, skipping.")

# ----------------------------------------------------------------------
#  math_tool C extensions (_topology / _fourier / _linear_algebra / _unit_map)
# ----------------------------------------------------------------------
math_inc_dir = os.path.relpath(
    os.path.abspath(
        "cos_comparison/interface/tools/math_tool/include"), setup_dir)

math_sources = [
    ("cos_comparison.interface.tools.math_tool._topology",
     "cos_comparison/interface/tools/math_tool/include/_topology.c", None),
    ("cos_comparison.interface.tools.math_tool._fourier",
     "cos_comparison/interface/tools/math_tool/include/_fourier.c",
     math_libs),
    ("cos_comparison.interface.tools.math_tool._linear_algebra",
     "cos_comparison/interface/tools/math_tool/include/_linear_algebra.c",
     math_libs),
    ("cos_comparison.interface.tools.math_tool._unit_map",
     "cos_comparison/interface/tools/math_tool/include/_unit_map.c",
     math_libs),
]
for name, rel_path, libs in math_sources:
    abs_path = os.path.abspath(rel_path)
    rel = os.path.relpath(abs_path, setup_dir)
    if os.path.isfile(rel):
        ext = _make_extension(name, rel, math_inc_dir, libs)
        if ext is not None:
            ext_modules.append(ext)
            print("Python C extension ({0}) configured.".format(name))
    else:
        print("Warning: {0} source not found, skipping.".format(rel_path))

# ----------------------------------------------------------------------
#  Custom build_ext: per-extension failure isolation
# ----------------------------------------------------------------------
def _strip_std_flags(ext):
    """Drop compiler standard flags (/std:... or -std=...) from an
    extension in place; True when something was removed (the old-toolchain
    retry), False when there is nothing to reduce."""
    args = list(getattr(ext, "extra_compile_args", ()) or ())
    reduced = [a for a in args
               if not (isinstance(a, str)
                       and (a.startswith("/std:") or a.startswith("-std=")))]
    if len(reduced) == len(args):
        return False
    ext.extra_compile_args = reduced
    return True


class SafeBuildExt(build_ext):
    def build_extension(self, ext):
        """Build one extension; a failure is retried once without the
        compiler standard flags (low-version toolchains), then skips only
        that extension so the remaining ones (and the pure-Python install)
        keep working."""
        try:
            super().build_extension(ext)
            return
        except Exception as e:  # noqa: BLE001 - per-extension fallback
            if _strip_std_flags(ext):
                print(f"\n*** extension {ext.name} failed with the configured "
                      f"compiler standard flags; retrying without them "
                      f"(low-version toolchain). ***")
                self.force = True   # recompile after the partial failure
                try:
                    super().build_extension(ext)
                    return
                except Exception as retry_exc:  # noqa: BLE001
                    e = retry_exc
            print(f"\n*** extension {ext.name} compilation failed: {e} ***")
            print("*** That extension will be skipped; the package stays "
                  "installable (pure Python fallback). ***")
            self.extensions = [x for x in self.extensions if x is not ext]

    def run(self):
        # Build the Python C extensions; per-extension failures are handled
        # inside build_extension.  A global failure (e.g. no compiler) clears
        # the extensions so the pure-Python install proceeds.
        try:
            super().run()
        except Exception as e:  # noqa: BLE001 - global fallback
            print(f"\n*** C extension build failed globally: {e} ***")
            print("*** The package will be installed without C acceleration "
                  "(pure Python fallback). ***")
            self.extensions = []

# ----------------------------------------------------------------------
#  License metadata shim for setuptools < 77
#  pyproject.toml carries the modern SPDX form (`license = "MIT"` +
#  `license-files`), understood only by setuptools>=77 - but the Python
#  3.8 floor allows setuptools 61.  On older setuptools the build is
#  pointed at a temporary legacy-compatible sibling copy
#  (`license = {text = "MIT"}`, no `license-files` key) instead of the
#  real file, which is never modified: sdists and the working tree always
#  keep the modern metadata.
# ----------------------------------------------------------------------
def _setuptools_before_77():
    try:
        parts = (setuptools.__version__ or "").split(".")
        major = int(re.match(r"\d+", parts[0]).group(0))
        minor = int(re.match(r"\d+", parts[1]).group(0)) if len(parts) > 1 else 0
    except Exception:  # noqa: BLE001 - version parsing must never break setup
        return False
    return (major, minor) < (77, 0)


def _install_license_shim():
    """Teach pre-77 setuptools to read a legacy license table for this
    build without touching pyproject.toml on disk."""
    if not _setuptools_before_77():
        return
    try:
        import setuptools.config.pyprojecttoml as pyprojecttoml
    except Exception:  # noqa: BLE001 - unknown layout: keep the old behaviour
        return

    original_apply = pyprojecttoml.apply_configuration

    def apply_configuration(dist, filename, ignore_option_errors=False):
        try:
            with open(filename, encoding="utf-8") as handle:
                text = handle.read()
        except OSError:
            return original_apply(dist, filename, ignore_option_errors)
        legacy = re.sub(
            r'^([ \t]*)license[ \t]*=[ \t]*"([^"]*)"',
            lambda m: '%slicense = {text = "%s"}' % (m.group(1), m.group(2)),
            text, count=1, flags=re.MULTILINE)
        legacy = re.sub(r'^[ \t]*license-files[ \t]*=.*(?:\n|$)', '',
                        legacy, flags=re.MULTILINE)
        if legacy == text:
            return original_apply(dist, filename, ignore_option_errors)
        legacy_path = os.path.join(
            os.path.dirname(os.path.abspath(filename)),
            "_cc_legacy_pyproject.toml")
        try:
            with open(legacy_path, "w", encoding="utf-8", newline="") as handle:
                handle.write(legacy)
            print("License shim: setuptools<77 detected; using a legacy "
                  "license table for this build (pyproject.toml untouched).")
            return original_apply(dist, legacy_path, ignore_option_errors)
        finally:
            try:
                os.remove(legacy_path)
            except OSError:
                pass

    pyprojecttoml.apply_configuration = apply_configuration


_install_license_shim()

# ----------------------------------------------------------------------
#  Final setup
# ----------------------------------------------------------------------
try:
    setup(
        cmdclass={'build_ext': SafeBuildExt},
        ext_modules=ext_modules,
        include_package_data=True,   # Use package_data from pyproject.toml
        # packages / package_data are defined in pyproject.toml (single
        # source of truth) – no need to duplicate them here
    )
except SystemExit:
    raise
except Exception as e:  # noqa: BLE001 - last-resort fallback: install pure Python
    print(f"\n*** setup() failed ({e}); retrying without C extensions. ***")
    setup(
        cmdclass={'build_ext': SafeBuildExt},
        ext_modules=[],
        include_package_data=True,
    )
