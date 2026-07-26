# Cos Comparison

[!\[PyPI version](https://badge.fury.io/py/cos-comparison.svg)](https://pypi.org/project/cos-comparison/)
[!\[Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[!\[License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**An AGI-oriented project based on local similarity comparison for feature extraction – biologically inspired, zero-training edge / pattern detection.**

\---

## Core Idea

Information is produced by **local comparison** in raw data.  
This module implements the **centre-surround antagonism** mechanism from neuroscience, extracting edges, textures, and keypoints using only sliding-window similarity.

The core formula (cosine-modulated similarity, recommended default):

$$
\\text{cosmod} = \\frac{2(A \\cdot B)}{|A|^2 + |B|^2}
$$

* **A step toward biologically plausible AGI**
* **No training, no labels, no backpropagation**
* Works on **1D, 2D, 3D, 4D** data (audio, images, video, volumetric data)
* Supports **passive** (reflexive boundary detection) and **active** (template matching) modes
* Three high-performance backends with automatic fallback: Python C extension, ctypes pure C, pure Python
* Cross-platform support for Windows, Linux, and macOS
* **Zero external dependencies** for core functionality

\---

## Seven-Layer Cognitive Architecture

This project follows a biologically inspired seven-layer cognitive architecture, mimicking the structure of the mammalian brain. Currently only the core brainstem/cerebellum layer is production-ready; all other layers are in early evolutionary stage with skeleton implementations, and will be gradually improved in future versions:

|Layer|Directory|Corresponding Brain Structure|Maturity|Core Function|
|-|-|-|-|-|
|1|`core`|Brainstem / Cerebellum|✅ Production|Low-level local comparison calculation, three-backend acceleration, free-thread support|
|2|`sense\_layer`|Sensory Cortex|🟡 Early Development|Receive external stimuli, extract raw features (Data/Auto\_Data classes available)|
|3|`memory\_layer`|Hippocampus / Cerebral Cortex|🔴 Skeleton|Short-term and long-term memory storage|
|4|`brain\_layer`|Prefrontal Cortex|🟡 Early Development|High-level cognition, logical reasoning (symbolic logic system implemented)|
|5|`action\_layer`|Motor Cortex|🔴 Skeleton|Control action output, interact with environment|
|6|`generate\_layer`|Broca's / Wernicke's Area|🔴 Skeleton|Generate language, images and other high-level outputs|
|7|`extension\_layer`|Association Cortex|🔴 Skeleton|Extended functions and special capabilities|

> \*\*Note\*\*: Non-core layers are currently in early development and do not affect the stability of the core feature extraction API. The core `cos\_comparison.core` module is fully production-ready and follows semantic versioning guarantees. All non-core modules now import without fatal errors as of v0.3.6.

\---

## 🚀 What's New in Version 0.3.7

Version 0.3.7 is a performance **with an urgent fix for issues in version 0.3.6**, stability and portability release:

* **Full Python GC support**: Added proper traverse/clear functions for garbage collection, fixes memory leaks and circular reference issues
* **Portability improvements**: Replaced non-standard `alloca()` with standard C `malloc`/`free`, conforms to C11 standard, works with all C compilers
* **Subclass support**: All operations and slicing return the correct subclass type, enabling proper inheritance
* **Numerically stable statistics**: `mean()` and `variance()` use Welford's online algorithm across all backends, eliminates large number overflow and precision loss
* **Iteration support**: C extension now supports default Python iteration (`for row in tensor`, `list(tensor)`) matching pure Python behavior
* **Performance optimizations**: Added SIMD auto-vectorization hints for all linear loops, 50-100% speedup on element-wise operations
* **Enhanced buffer support**: Leaf-dimension slice assignment works with all buffer-protocol objects (array.array, memoryview, numpy arrays, byte buffers)
* **New operator**: Added `\*\*` (pow/ipow) operator for element-wise exponentiation across all backends
* **Critical bug fixes**: Fixed slice length bug, new\_cache calculation bug, buffer cast bug, set\_item typo, and compiler warnings
* **Zero dependencies**: Core package remains 100% dependency-free, no numpy or other third-party requirements
* **Comprehensively tested**: All functionality tested in clean virtual environment, 100% behavioral parity across all three backends
* **Dual Python 3.14 support**: Precompiled binaries for both standard GIL and free-threaded (no-GIL) Python 3.14, zero compiler warnings

\---

## Installation

```bash
pip install cos-comparison
```

The installer will automatically attempt to compile both C backends during installation. If a C compiler is not available on your system, installation will complete successfully with the pure Python backend only.

To install with optional test dependencies:

```bash
pip install cos-comparison\[test]
```

### Manual Compilation

If you need to recompile the C backends after modifying source code:

```bash
# From the project root directory
python setup.py build\_ext --inplace
```

This will automatically build both the Python C extension and the ctypes shared library.

\---

## Performance Benchmarks

Benchmark results for all three backends using a **322×424×3 RGB test image** with 3×3 window.  
Test environment: Windows 11 x64, 18-thread CPU, Python 3.14.6, JIT enabled, MSVC -O2 optimization.

|Backend|Execution Time|Speedup vs Pure Python|Peak Memory|Status|Free-thread Support|
|-|-|-|-|-|-|
|Python C Extension|0.004s|\~130×|\~5–8 MB|✅ Stable|✅ Full (no GIL)|
|ctypes C Backend|0.007s|\~70×|\~8–12 MB|✅ Stable|✅ Full|
|Pure Python|0.52s|1×|\~22 MB|✅ Stable|✅ Full|

> All three backends produce numerically identical output values within floating-point precision. C backends automatically fall back to pure Python if compilation or loading fails.

\---

## Quick Start

### 2D edge detection (passive mode)

```python
from cos\_comparison import core

# Create test data
data = core.create\_void\_list((5,5))
for i in range(5):
    for j in range(5):
        data\[i,j] = 1.0 if i < 2 and j < 2 else 0.0

# Detect vertical edges
edges = core.cos\_comparison\_passive(data, window\_size=3, d=(0,1))
print(edges\[1,1])
```

### Multi-index assignment

```python
# Fast tuple assignment (new in 0.3.6)
v = core.create\_void\_list((3,3))
v\[1,1] = 123.0  # Direct flat-offset assignment, no intermediate objects
assert v\[1]\[1] == 123.0
```

### Switching backends

```python
# Check current backend
print(core.get\_mode())

# Force pure Python mode for debugging
core.set\_mode("cos\_comparison")
```

\---

## Author

I was born on May 31, 2008, and I feel fortunate to grow up in an era of rapid progress in artificial intelligence.

I have run extensive tests and observed many surprising emergent properties in the outputs. Earlier versions of this project contained numerous issues, as examination-oriented education left me limited time for thorough testing. I now have the opportunity to properly test, refine, and polish this work.

There remains a long road ahead to achieve true general artificial intelligence. I may be forced to set aside this research due to personal circumstances, but I do not want these ideas to fade away unnoticed. The purpose of open-sourcing this project is to share my thoughts, in the hope that others may build upon them and continue this line of inquiry.

\---

## Contact \& Feedback

* **Bug Reports \& Issues**: Please submit issues on [GitHub Issues](https://github.com/LiJinxin-gx/cos-comparison/issues)
* **Email Contact**: lijinxin\_gx@sina.cn

\---

## License

MIT © 2026 Li Jinxin. See [LICENSE](LICENSE.txt) for details.

