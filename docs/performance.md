# Performance & Resource Audit

> Measured figures for **cos-comparison v0.5.2** on a single reference machine.
> Every number below was produced by the project's in-house audit harness kept
> **outside the distributed package** (it is not a public benchmark suite and
> ships with nothing); the methodology, environment and limitations are
> documented so the numbers can be reproduced or discounted honestly.

---

## Contents

| Section | What it covers |
|---------|----------------|
| [Test Environment](#test-environment) | Hardware, OS, interpreter, compilers, measurement tools |
| [Methodology](#methodology) | How each number was taken, and its caveats |
| [Core Module](#core-module-pure-python-vs-c-extension) | passive / active / statistics / pooling / element-wise / similarity / access |
| [math_tool](#math_tool-python-reference-vs-c-extension) | linear algebra / topology / unit map / Fourier |
| [Brain-Layer Logic](#brain-layer-logic) | probability engines and the rule-chain judge |
| [Memory & Value Layers](#memory--value-layers) | MapMemory round-trip, value grammar compile/execute |
| [GPU / Hardware Acceleration](#gpu--hardware-acceleration) | Intel Arc OpenCL end-to-end vs CPU baselines |
| [Super-Parallel Scaling](#super-parallel-scaling-v052) | CPU thread sharding (GIL vs free-threaded) and GPU pooling via `iterate=` |
| [Resource Measurements](#resource-measurements) | Cold import, memory footprints, artifact sizes |
| [Findings & Known Issues](#findings--known-issues) | Defects and asymmetries surfaced by the audit |

---

## Test Environment

| Item | Value |
|------|-------|
| CPU | Intel Core Ultra 5 125H (14 cores / 18 threads; Model 170 Stepping 4) |
| GPU | Intel Arc Graphics — 112 compute units, driver 31.0.101.5382, 14.7 GB shared memory |
| OS | Windows 11 x64 (build 10.0.26200) |
| Interpreter | CPython 3.14.6 (traditional, with-GIL build) |
| C compiler | MSVC 19.x (`/O2 /std:c11`, built with `/Wall /WX`, warning-clean) |
| Measurement tools (not runtime deps) | numpy 2.5.2, pyopencl 2026.1.4 |

The package itself stayed **stdlib-only** during the audit; numpy and pyopencl
were used only to create measurement containers / drive the GPU device.

## Methodology

- **Harness**: in-house scripts (`bench_all`, `bench_rest`, `bench_gpu`,
  `bench_resources`) kept in a system temp directory, never packaged.
- **Data**: deterministic nested Python float grids (LCG-generated, no
  third-party dependency) for CPU backends; numpy float32 arrays for the GPU
  comparisons (the device requires contiguous buffers).
- **Timing**: `time.perf_counter()`, `gc.collect()` before each timed run,
  warmup runs excluded, then repeats; the table value is the **median** with
  min/max noted in the raw JSON. One run per measurement unless `n=` says
  otherwise.
- **Batch effect**: this is a laptop — sustained runs show thermal throttling
  (observed up to ±2.5× between batches on the heavy pure-Python rows). Ratios
  taken **within one batch** are stable; treat large-size absolute numbers as
  indicative. Rows were measured in one session across several harness
  batches; the fill-in rows (large pure-Python sizes, graph algebra, helper
  APIs) name their own context, and cross-batch ratios are indicative only.
- **GPU figures are end-to-end**: OpenCL program build excluded, but buffer
  creation, host→device upload, kernel execution and device→host read-back all
  included (queue finished inside the timed region).
- **Resource figures**: `tracemalloc` for Python-level allocations, process
  working set (RSS) for actual memory use, cold-import wall time = process
  start + import, best of 3 subprocess runs.

---

## Core Module (pure Python vs C extension)

Data: nested Python float grids. `passive`/`active` use the default `cosmod`
algorithm. Times in **ms**.

### passive 2D (3×3 window, step (1,1), d (1,1))

| Size | Pure Python | C extension | py / C |
|------|------------:|------------:|-------:|
| 64² | 52.6 | 0.46 | 115× |
| 256² | 817 | 10.3 | 79× |
| 512² | 3838 | 26.7 | 144× |
| 1024² | 13 662 | 102.7 | 133× |
| 2048² | 55 573 | 417.8 | 133× |
| 64² ×100 calls | 5233 | 38.7 | 135× |

The 1024²/2048² pure-Python rows were measured single-run in a follow-up
batch (the C rows came from the main batch; cross-batch ratios indicative).

### passive 1D / 3D and active 2D

| Case | Pure Python | C extension | py / C |
|------|------------:|------------:|-------:|
| passive 1D, 200k, window 33 | 12374 | 80.9 | 153× |
| passive 3D, 64³, 3³ window | 22885 | 241.5 | 95× |
| active 2D, 256², 7×7 kernel | 12711 | 94.4 | 135× |

### Local statistics (512², local_size (3,3), step (1,1))

| Function | Pure Python | C extension | py / C |
|----------|------------:|------------:|-------:|
| `mean_local` | 10640 | 41.9 | 254× |
| `local_variance` | 4687 | 24.0 | 195× |

### Pooling (3×3 window, step (1,1), numpy float32 input)

Sliding windows: step 1 over a 3×3 window means heavily overlapping
windows; the classic non-overlapping case is `step == window_size`, and
larger steps leave gaps.  All three are the same code path - `step`
alone decides.

| Case | Pure Python | C extension | py / C |
|------|------------:|------------:|-------:|
| `pool` 512², mean (`func=None`) | 2158 | 166 | 13.0× |
| `pool` 512², `max` | 2803 | 243 | 11.5× |
| `pool` 512², `sum` as `lambda *vs: sum(vs)` | 2839 | 299 | 9.5× |
| `pool` 1024², mean (`func=None`) | 10709 | 1018 | 10.5× |
| `pool` 1024², `max` | 10697 | 1224 | 8.7× |

The C fast path converts the input data once to its flat double form and
reads windows natively; the per-window Python `func` callback (when given)
and the output writes stay duck-dispatch bound, which is where the
remaining C cost lives.  A C `create_void_list` output is ~20% faster than
a numpy output at the same size (138.9 ms vs 166.8 ms for 512² mean).
Earlier pre-test prototype (shifted views forwarded into `elementwise`,
kept outside the package): 3.4-3.9 s pure Python and 3.7-4.3 s "C" at
512² - the direct skeleton is ~15-25× that composition.

### Element-wise family (512², per-element Python callback)

| Function | Pure Python | C extension | note |
|----------|------------:|------------:|------|
| `data_mapping` (v·2+1) | 681 | 208 | C moves the loop, callback stays Python |
| `data_filter` (v > 0.5, materialized) | 247 | 114 | |
| `position_map` (logical coords) | 292 | 117 | |
| `elementwise` (a+b, two tensors) | 297 | 308 | callback-bound, parity |
| `threshold_map` ((func,value) pairs) | 758 | 1238 | callback-bound |
| `threshold_filter` (low=0.5, materialized) | 686 | 671 | parity |

### Additional core measurements

| Case | Pure Python | C extension | py / C |
|------|------------:|------------:|-------:|
| passive 2D 256², `algorithm=_cos` | 793 | 14.4 | 55× |
| passive 2D 256², `algorithm=_mod` | 794 | 15.4 | 52× |
| `elementwise_position` 512² (two tensors) | 562 | 314 | 1.8× |
| `load_data` 1024² self-copy | 1320 | 930 | 1.4× |
| `infer_shape` 64² ×1000 | 0.48 | 0.23 | 2.1× |
| `create_void_list` 256² ×100 | 83.3 | 2.35 | 35× |
| tensor view slice (1024², ×1000) | 4.27 | 15.9 | 0.27× |
| passive 2D 128² via injected engine (`iterate=`) | 201 | 200 | 1.0× |
| `cos` 512² ×4 concurrent: 1 thread / 4 threads | 751 / 756 | 35.2 / 35.1 | — |

The `iterate=` row is the key contract check: with an external element engine
injected, the C extension delegates to the Python reference path, so both
column values coincide (±1%). View slicing is another pure-Python win (views
are stride+offset; the C call pays dispatch cost per slice). Concurrency: at
this task size (4 × 512² whole-tensor comparisons) neither backend gains from
4 threads — each C call is fast but the per-call work is too small to scale.

### Whole-tensor similarity and point access

| Case | Pure Python | C extension | py / C |
|------|------------:|------------:|-------:|
| `cos` over 1e6 floats | 1372 | 42.6 | 32× |
| `get_item` ×100 000 (256² grid) | 25.4 | 76.3 | 0.33× |
| `set_item` ×100 000 (256² grid) | 35.1 | 83.4 | 0.42× |

Single-element access is the one place the C extension **loses** to pure
Python: the per-call dynamic dispatch outweighs the indexing cost. Bulk
paths are the intended use.

---

## math_tool (Python reference vs C extension)

Both implementations are importable side by side (`math_tool.<name>` and
`math_tool._<name>`); per the module contract every linear-algebra call
writes through `output=` and returns an integer status.

> Note: `dot`/`norm`/... without `output=` return status `2` (NO_OUTPUT)
> immediately by design — always measure with a real output container.

> Update (v0.5.2 optimization pass): the *Python reference* implementations
> were optimized without changing any arithmetic (linear algebra: one-walk
> flatten+convert with type fast paths and streamed element-wise writes;
> topology: one shared union-find base).  On the audit workloads the
> reference got ~1.3–4.3× faster with lower peak memory (e.g. `dot` 40k:
> 9.4 → 4.8 ms, 2.83 → 2.50 MB; `normalize` 40k: 15.7 → 3.6 ms,
> 1.59 → 0.34 MB; 20k-chain SCC: 12.0 → 9.4 ms), so the Python-reference
> columns and py/C ratios below are the pre-optimization audit record.
>
> The later duck-protocol round touched the C side of `_linear_algebra`
> and `_topology` only (numpy containers/scalars, numeric-protocol
> subclass conversion, one-shot-iterable Euler cells).  The fixed C Euler
> scan measures **0.88 ms** for 100k cells (the audit row below, 2.83 ms,
> was slowed by the overflow-guard defect that forced the big-int
> fallback on every multi-cell scan); the linear-algebra C figures are
> unaffected by the exact-type fast paths.

### Linear algebra over 1e6 floats

| Operation | Python reference | C extension | py / C |
|-----------|-----------------:|------------:|-------:|
| `dot` | 860 | 468 | 1.8× |
| `norm` | 406 | 236 | 1.7× |
| `normalize` | 1055 | 285 | 3.7× |
| `scale` | 1152 | 281 | 4.1× |
| `add` | 2702 | 1208 | 2.2× |
| `multiply` | 2647 | 1253 | 2.1× |
| `power` | 2407 | 707 | 3.4× |
| `clip` | 2222 | 674 | 3.3× |
| `tensor_sum` | 752 | 532 | 1.4× |
| `tensor_mean` | 712 | 548 | 1.3× |
| `flatten` (512² → 1D) | 51.8 | 34.5 | 1.5× |

### Topology

Directed-graph figures use a deterministic DAG (2 000 vertices / 10 000 arcs) —
the largest size verified **stable** on the C implementation during the audit
(see [Findings](#findings--known-issues)). Undirected `Graph` and the Euler
cell scan: Python reference only.

| Case | Python reference | C extension | py / C |
|------|-----------------:|------------:|-------:|
| `DirectedGraph.add_edge` build, 2k/10k | 41.2 | 23.5 | 1.8× |
| `DirectedGraph.add_edge` build, 5k/20k (isolated process) | — | 14.7–15.5 | — |
| `strong_components`, 2k/10k | 7.19 | 4.47 | 1.6× |
| `strong_components`, 5k/20k (isolated) | — | 3.07 | — |
| `topological_sort`, 2k/10k | 5.91 | 3.57 | 1.7× |
| `topological_sort`, 5k/20k (isolated) | — | 1.68 | — |
| `reachable` ×1000 (2k DAG) | 57.4 | 29.9 | 1.9× |
| `shortest_path` ×1000 (2k DAG) | 67.7 | 35.2 | 1.9× |
| `shortest_path_between` ×1000 (2k DAG) | 100.6 | 47.7 | 2.1× |
| `has_eulerian_path` ×1000 (2k DAG) | 452 | crash (fixed in v0.5.1; see [Findings](#findings--known-issues)) | — |
| `Graph` euler / connected / cycle_rank ×1000 (2k) | 0.18 / 0.15 / 0.15 | crash (fixed in v0.5.1; see [Findings](#findings--known-issues)) | — |
| `Euler_characteristic_compute_by_cell`, 100k cells | 11.4 | 2.83 | 4.0× |

The isolated-process rows ran in fresh interpreter processes (cold); the 2k
rows ran inside the mixed batch (warm, more variable) — compare absolutely
only within a row's own batch.

### Unit map (200 000 units, 50 distinct values)

| Case | Python reference | C extension | py / C |
|------|-----------------:|------------:|-------:|
| fold + `put` + `count_vector` | 53.4 | 8.32 | 6.4× |
| `count_vector` only | 31.0 | 6.05 | 5.1× |
| `window_units` 64², local (8,8) | 1.12 | 0.064 | 17.6× |
| `map_data` 64², local (8,8) | 25.0 | 6.04 | 4.1× |

### Fourier (256 samples)

| Case | Python reference | C extension | py / C |
|------|-----------------:|------------:|-------:|
| `dft` | 13.0 | 0.85 | 15.3× |
| `power_spectrum` | 13.0 | 0.91 | 14.3× |
| `dft_kernel_real` (256, freq [3]) | 0.15 | 0.028 | 5.4× |
| `dft_kernel_imag` (256, freq [3]) | 0.15 | 0.031 | 4.9× |
| `idft` | 26.1 | — | see [Findings](#findings--known-issues) |

---

## Brain-Layer Logic

Probability engine (`EventBinds`, cached dependency graph) and rule-chain
judge, on 20 000-binding / 2 000-rule fixtures:

| Case | Time | Per operation |
|------|-----:|--------------:|
| add 20 000 bindings (incl. graph cache warm) | 5.1 ms | 0.25 µs/bind |
| `resolve` direct hit ×50 000 | 54.2 ms | 1.1 µs |
| `resolve` 100-arc chain ×1 000 (cached graph) | 32.8 ms | 33 µs |
| strict exact chain, 200 arcs ×500 | 15.1 ms | 30 µs |
| non-strict (Markov) 200-arc chain ×500 | 155–173 ms | ~0.3 ms |
| judge 500 queries over 2 000 rules (rebuilds graph per call) | 1.38–1.54 s | ~2.8 ms |
| `union_probability` ×50 000 (3 resolves each) | 485 ms | 9.7 µs |
| `strict_probability_func` ×50 000 (direct hit) | 11.8 ms | 0.24 µs |
| `EventContextProtocol` engine, direct hit ×50 000 | 49.6 ms | 0.99 µs |

Two observables: the strict chain **beats** the Markov fallback on long
dependency chains (it walks bindings directly, while the fallback rebuilds
the graph per call), and the judge is the one hot spot — it rebuilds the rule
graph on every call by design (the probability engines cache theirs).

## Memory & Value Layers

| Case | Time |
|------|-----:|
| `MapMemory`: save + commit + refer, 20 000 entries | 20.6–22.3 ms |
| `TableMemory`: save + commit + refer, 20 000 entries | 36.4 ms |
| `MemoryWrap`: get + set ×100 000 | 9.5–10.0 ms |
| `compile_value` nested expression ×5 000 | 52–60 ms (≈11 µs) |
| `compile_value` container literal ×5 000 | 15.6 ms |
| execute compiled code ×5 000 | 7.5 ms (≈1.5 µs) |
| `data[0]+1` compiled execute ×5 000 | 2.3 ms |

---

## GPU / Hardware Acceleration

The same 3×3 window aggregation used by core passive mode (cosmod,
d=(1,1)), an element-wise map (`v·2+1`) and saxpy — all measured
**end-to-end** on the Intel Arc iGPU through OpenCL, against a
single-thread numpy CPU baseline and the package backends measured above.

### passive 3×3 window aggregation (ms)

| Size | numpy (CPU) | OpenCL (GPU, e2e) | GPU vs numpy | package C | package py | GPU vs C | GPU vs py |
|------|------------:|------------------:|-------------:|----------:|-----------:|---------:|----------:|
| 512² | 7.78 | 0.51 | 15.2× | 26.7 | 3838 | 52× | 7 510× |
| 1024² | 68.5 | 2.13 | 32.1× | 102.7 | 13 662 | 48× | 6 414× |
| 2048² | 283 | 6.96 | 40.7× | 417.8 | 55 573 | 60× | 7 985× |

### element-wise map `v·2+1` (ms)

| Size | numpy (CPU) | OpenCL (GPU, e2e) | GPU vs numpy | package C | package py | GPU vs C |
|------|------------:|------------------:|-------------:|----------:|-----------:|---------:|
| 512² | 1.25 | 0.56 | 2.2× | 208 | 681 | 372× |
| 1024² | 3.13 | 1.77 | 1.8× | 866 | 2535 | 489× |
| 2048² | 11.4 | 5.67 | 2.0× | 3515 | 10 369 | 620× |

(The package columns are the per-element-callback `data_mapping` path: the C
extension still crosses into Python per element, which is where the GPU gap
comes from.)

### saxpy (`2.5x + 3.0y`) (ms)

| Length | numpy (CPU) | OpenCL (GPU, e2e) | GPU vs numpy |
|-------:|------------:|------------------:|-------------:|
| 1e6 | 3.90 | 2.00 | 2.0× |
| 1e7 | 41.7 | 19.8 | 2.1× |

Reading: the GPU wins big where arithmetic intensity is high (window
aggregation: 15–41× over CPU SIMD, up to ~7 500× over the pure Python
reference); memory-bound element-wise work is transfer-limited and lands at
~2× end-to-end. The package's `iterate=` slot is what lets an external engine
of this kind drive the core without touching the package.

## Super-Parallel Scaling (v0.5.2)

Pooling (`core.pool`) is measured along both super-parallel axes: CPU
multi-thread sharding under the free-threaded interpreter and GPU
execution through the `iterate=` engine slot.

**CPU — row-shard threads** (2048² float32, 3×3 window, C backend, wall
ms; each thread runs one `core.pool` over a row slice with a 2-row halo;
the GIL build 3.14.6 versus the free-threaded build 3.14.6t):

| Shards | GIL mean | GIL max | free-threaded mean | free-threaded max |
|-------:|---------:|--------:|-------------------:|------------------:|
| 1 | 2797.0 | 3464.6 | 3266.1 | 3994.5 |
| 2 | 2803.3 | 3478.3 | 2467.6 | 3404.0 |
| 4 | 2798.5 | 3447.3 | 1778.9 | 1636.2 |
| 7 | 2780.7 | 3460.2 | 1893.3 | 1905.8 |
| 14 | 2780.2 | 3441.4 | 1269.5 | 1369.9 |
| **14-shard speedup** | **1.0×** | **1.0×** | **2.6×** | **2.9×** |

**GPU — OpenCL engine via `iterate=`** (3×3 window, step 1, end-to-end
including host↔device transfers and buffer setup; the C column is the
single-thread C fast path):

| Size | case | C (ms) | GPU (ms) | GPU vs C |
|------|------|-------:|---------:|---------:|
| 512² | mean | 168.3 | 1.31 | 128× |
| 512² | max | 202.6 | 1.05 | 193× |
| 512² | sum | 231.2 | 1.02 | 227× |
| 1024² | mean | 1075.0 | 3.49 | 308× |
| 1024² | max | 968.8 | 2.66 | 364× |
| 1024² | sum | 1171.2 | 2.75 | 426× |
| 2048² | mean | 3083.2 | 8.77 | 352× |
| 2048² | max | 4084.2 | 15.16 | 269× |
| 2048² | sum | 4692.2 | 8.33 | 563× |
| 4096² | mean | 13451.2 | 32.52 | 414× |

Correctness: the device results match the C inline path (max/min
bit-exact; mean and sum within float32 rounding, max |diff| ≤ 9.6e-7).
The test engine recognises the canonical reducers — `None` (mean), the
builtin `max` / `min` / `sum` — and falls back to a per-index host loop
otherwise; the C backend hands it a mode-5 C kernel, which the device fast
path bypasses.  Reading: the GPU is ~130–560× the C fast path end-to-end
at every size (the working set still fits device memory, so transfers
amortise); CPU sharding scales only under the free-threaded interpreter
(2.6–2.9× at 14 shards — bounded by memory bandwidth and the per-output
duck writes, not linear), while the GIL build is flat at 1.0× by
construction (the C pool call holds the GIL for its whole duration).

## Resource Measurements

### Cold import (process start + import, best of 3)

| Import | Time |
|--------|-----:|
| `python -c pass` (baseline) | 54.6 ms |
| `import cos_comparison` | 54.2 ms |
| `import cos_comparison.core` (C backend load) | 76.2 ms |
| `import cos_comparison.interface.tools.math_tool` (4 C extensions) | 171.0 ms |
| `import cos_comparison.brain_layer.logic` | 173.5 ms |
| `import cos_comparison.memory_layer.memory` | 171.1 ms |
| `import cos_comparison.shell_tool.value` | 54.1 ms |

### Module load memory (RSS delta, isolated process per import)

| Import | RSS delta |
|--------|----------:|
| `import cos_comparison` (top level only) | +0.03 MB |
| `import cos_comparison.core` (loads the C backend) | +1.16 MB |
| `import cos_comparison.interface.tools.math_tool` (incl. core + 4 C extensions) | +10.3 MB |
| `import cos_comparison.brain_layer.logic` (incl. math_tool) | +11.0 MB |
| `import cos_comparison.memory_layer.memory` (incl. core) | +10.3 MB |
| `import cos_comparison.shell_tool.value` (stdlib-only module) | +0.13 MB |

### Data structure footprints (net RSS + tracemalloc)

| Structure | net RSS | tracemalloc |
|-----------|--------:|------------:|
| nested grid 512² (262k floats) | +11.0 MB | 8.5 MB |
| nested grid 1024² (1.05M floats) | +42.0 MB | 34.3 MB |
| nested grid 2048² (4.19M floats) | +174.8 MB | 138.0 MB |
| numpy float32 2048² (reference container) | +8.0 MB | native array (outside the Python allocator) |
| `create_void_list` 512² | +2.2 MB | 0.007 MB |
| `vector_map_as_tensor` 1e6 floats | +8.0 MB | 8.0 MB |

A nested Python grid costs ~42 bytes per element against **4 bytes** for a
float32 buffer — at scale, use the buffer / zero-copy paths.  (The sampled
RSS peak during nested-2048² generation reads 642 MB with tracemalloc
enabled: that includes tracemalloc's own tracing tables for millions of
allocations; the net figure is the right column.)

### Runtime peaks (sampled RSS, 5 ms interval)

| Case | Sampled RSS peak delta |
|------|-----------------------:|
| passive 512² (C, 510² nested output) | +4.0 MB |
| passive 2048² (C, 2046² nested output) | +66.7 MB |
| `EventBinds` 100k / 500k bindings | +69.2 MB / +337.3 MB |
| judge over 10k rules (graph build + BFS) | +11.2 MB |

Allocation counting on the pure-Python path: a passive 512² call allocates
**+259 144 blocks (~8.3 MB)** — that is the nested output structure, not the
computation; the C backend allocates its output on the native side
(`sys.getallocatedblocks` delta **+7**).  `tracemalloc` sees 0.0 MB for the
C call by design.

### Memory-layer carriers (sampled RSS)

| Carrier | RSS delta | note |
|---------|----------:|------|
| `MapMemory` 100k entries | +16.3 MB | flat dict |
| `TableMemory` 100k entries | +41.1 MB | nested dicts |
| `IOStreamMemory` (in-memory) 20k entries | +1.4 MB | string-stream records |
| `IOStreamMemory` (file-backed) 20k entries | +0.2 MB | 310 KB on disk |
| `DatabaseMemory` (sqlite `:memory:`) 20k entries | +0.2 MB | SQLite allocates outside the Python allocator |

### Disk footprint

| Item | Size |
|------|-----:|
| `cos_comparison` package total (source + built extensions) | 2.07 MB |
| — `core` (0.32 MB C backend pyd + 0.34 MB C sources + Python) | 0.87 MB |
| — `interface` (math_tool: 4 C extensions + Python references) | 0.66 MB |
| — `shell_tool` | 0.17 MB |
| — `brain_layer` | 0.13 MB |
| — all other subpackages + top-level files | 0.24 MB |

### Compiled artifact sizes

| Artifact | Size |
|----------|-----:|
| `cos_comparison_pydll.cp314-win_amd64.pyd` (core C backend) | 316 KB |
| `_fourier` | 25.0 KB |
| `_linear_algebra` | 20.0 KB |
| `_topology` | 36.0 KB |
| `_unit_map` | 37.5 KB |

---

## Findings & Known Issues

Surfaced during this audit (all reproducible; the C issues are reported for
follow-up and are **not** silent-corruption cases in normal use):

1. **C topology graph types: intermittent access violation. — FIXED in
   v0.5.1.** Reproduced in every subprocess-isolated attempt of the
   undirected `Graph` type (2 000 vertices / 10 000 edges) and in the
   directed `has_eulerian_path` sweep (2 000-vertex DAG ×1000 calls);
   repeated directed builds crashed in some runs and passed in others
   (heap-state dependent). The Python reference was stable at all sizes.
   Root causes (fixed in the v0.5.1 math_tool hardening round): the
   union-find path compression dropped its borrowed parent chain,
   `dg_non_isolated_weakly_connected` NULL-dereferenced its saved root
   after the first comparison, and the C graph internals leaked
   references (`PyDict_SetItem` temporaries, borrowed `PyDict_GetItem`
   values, the adjacency-dict ownership in `dg_add_edge`).  After the
   rewrite the former crash scenarios pass 50 heavy rounds and a 1000×
   eulerian sweep; the C columns above document the pre-fix state for
   the sizes that were measurable then.
2. **C `_fourier.idft` rejected Python-complex inputs. — FIXED in
   v0.5.1.** The Python reference accepts them (`_split` falls back to the
   complex protocol); the C `scale` leaf path only spoke float.  The C
   extension now accepts complex inputs and returns identical values
   (verified: complex idft, dft→idft round trips, 2-D complex matrices).
   The audit lists no C idft figure; dft/power_spectrum were
   bit-identical throughout.
3. **Callback-bound element-wise paths narrow the C advantage.** The C
   extension still calls the Python callback per element
   (`data_mapping` 208 ms vs 681 ms on 512² — ~3× — and outright parity for
   `elementwise`/`threshold_map`/`threshold_filter`), which is exactly the
   gap the external `iterate=` engines are designed to eliminate (see the GPU
   table: 372× over `data_mapping` at 512², 620× at 2048²).
4. **Single-element access has negative C ROI** (`get_item`/`set_item`
   ~0.4–0.8 µs/call vs 0.25–0.35 µs pure Python); documented above so users
   do not micro-optimize the wrong layer.
5. **Thermal variance** on the reference laptop reaches ±2.5× on sustained
   pure-Python rows; within-batch ratios are the meaningful quantity.

## Reproduction

The harness lives in the experiment area (not distributed). To reproduce the
shape of these numbers:

```python
# CPU backends, one case
from cos_comparison import core
core.set_mode(['py'])   # or ['c']
import time
t = time.perf_counter()
core.cos_comparison_passive(grid, window_size=(3, 3), step=(1, 1), d=(1, 1))
print((time.perf_counter() - t) * 1000, 'ms')
```

GPU harness: requires `pyopencl` + an OpenCL device; it compiles the inline
OpenCL C kernels and measures end-to-end as described in Methodology.

---

*Audit performed 2026-09-26 for v0.5.1. Environment: see
[Test Environment](#test-environment).*
