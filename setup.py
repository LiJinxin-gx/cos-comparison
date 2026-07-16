"""
setup.py – Build and install cos_comparison with optional C acceleration.

This script compiles:
1. The Python C extension (pydll) – placed inside cos_comparison/core/
2. The ctypes shared library (core.dll / .so / .dylib) – placed inside cos_comparison/core/cos_comparison_c/

If compilation fails, the package gracefully falls back to pure Python.
"""

import os
import sys
import platform
import shutil
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

# Change to the directory of this script so that all relative paths work correctly
setup_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(setup_dir)

# Platform-specific compile arguments
is_windows = platform.system() == "Windows"
if is_windows:
    compile_args = ["/O2", "/std:c11"]
    math_libs = []  # Math functions are in msvcrt.dll on Windows, no separate lib needed
else:
    compile_args = ["-O2", "-std=c99"]
    math_libs = ["m"]  # Explicitly link libm on Linux/macOS for math functions

# ----------------------------------------------------------------------
#  Python C extension (pydll)
#  Source: cos_comparison/core/include/cos_comparison_pydll.c
#  Output: cos_comparison.core.cos_comparison_pydll (-> .pyd/.so)
# ----------------------------------------------------------------------
c_extension_name = "cos_comparison.core.cos_comparison_pydll"
c_source_abs = os.path.abspath("cos_comparison/core/include/cos_comparison_pydll.c")
c_source_rel = os.path.relpath(c_source_abs, setup_dir)
c_include_dir = os.path.relpath(os.path.abspath("cos_comparison/core/include"), setup_dir)

ext_modules = []

if os.path.isfile(c_source_rel):
    ext1 = Extension(
        name=c_extension_name,
        sources=[c_source_rel],
        include_dirs=[c_include_dir],
        extra_compile_args=compile_args,
        libraries=math_libs,
    )
    ext_modules.append(ext1)
    print("Python C extension (cos_comparison_pydll) configured.")
else:
    print("Warning: pydll source not found, skipping.")

# ----------------------------------------------------------------------
#  Custom build_ext: build both the Python extension and the ctypes shared library
# ----------------------------------------------------------------------
class SafeBuildExt(build_ext):
    def build_ctypes_backend(self):
        """Build the ctypes C backend shared library and place it in build/lib."""
        ctypes_src_dir = os.path.abspath("cos_comparison/core/cos_comparison_c/include")
        ctypes_out_dir = os.path.abspath("cos_comparison/core/cos_comparison_c")
        core_src = os.path.join(ctypes_src_dir, "core.c")
        
        if not os.path.isfile(core_src):
            print("Info: ctypes backend source not found, skipping.")
            return
        
        try:
            compiler = self.compiler
            os.makedirs(self.build_temp, exist_ok=True)
            
            # Compile core.c to object file
            print("Building ctypes C backend...")
            objects = compiler.compile(
                [core_src],
                output_dir=self.build_temp,
                include_dirs=[ctypes_src_dir],
                extra_preargs=compile_args,
                macros=[],
            )
            
            # Determine output library name per platform
            if is_windows:
                lib_name = "core"
                lib_ext = ".dll"
            elif sys.platform == "darwin":
                lib_name = "core"
                lib_ext = ".dylib"
            else:
                lib_name = "core"
                lib_ext = ".so"
            
            # --- IMPORTANT: output directly to build/lib ---
            build_lib = self.build_lib
            target_dir = os.path.join(build_lib, "cos_comparison", "core", "cos_comparison_c")
            os.makedirs(target_dir, exist_ok=True)
            output_lib = os.path.join(target_dir, lib_name + lib_ext)
            
            link_args = []
            if is_windows:
                link_args = ['/DLL']
            
            # Use link_shared_object for precise control over output path
            compiler.link_shared_object(
                objects,
                output_lib,
                libraries=math_libs,
                library_dirs=[],
                runtime_library_dirs=[],
                extra_preargs=link_args
            )
            
            print(f"ctypes backend built successfully: {output_lib}")
            
            # (Optional) Copy to source directory for in-place development
            source_lib = os.path.join(ctypes_out_dir, lib_name + lib_ext)
            if source_lib != output_lib:
                shutil.copy2(output_lib, source_lib)
                print(f"Copied to source directory: {source_lib}")
            
            # Enable ctypes backend in config.json
            config_path = os.path.abspath("cos_comparison/core/config.json")
            if os.path.isfile(config_path):
                import json
                with open(config_path, "r") as f:
                    config = json.load(f)
                for entry in config:
                    if entry.get("name") == ".cos_comparison_c":
                        entry["enabled"] = True
                with open(config_path, "w") as f:
                    json.dump(config, f, indent=4)
                print("ctypes backend enabled by default in config.json")
                
        except Exception as e:
            print(f"\n*** ctypes backend compilation failed: {e} ***")
            print("*** ctypes acceleration will not be available. ***")
    
    def run(self):
        # First, build the Python C extension
        try:
            super().run()
        except Exception as e:
            print(f"\n*** Python C extension compilation failed: {e} ***")
            print("*** The package will be installed without Python C extension acceleration. ***")
            # Clear extensions so that installation proceeds without them
            self.extensions = []
        
        # Build ctypes backend regardless of Python extension status
        self.build_ctypes_backend()

# ----------------------------------------------------------------------
#  Final setup
# ----------------------------------------------------------------------
setup(
    cmdclass={'build_ext': SafeBuildExt} if ext_modules else {'build_ext': SafeBuildExt},
    packages=find_packages(where=".", include=["cos_comparison", "cos_comparison.*"]),
    ext_modules=ext_modules,
    include_package_data=True,   # Use package_data from pyproject.toml
    # package_data is defined in pyproject.toml – no need to duplicate here
)
