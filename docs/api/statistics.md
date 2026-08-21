# Statistics Functions

Sliding-window local mean and variance, built on active mode infrastructure.

## Contents

- [mean_local](#mean_local)
- [local_variance](#local_variance)
- [Dimension Aliases](#dimension-aliases)
- [Tensor Statistics](#tensor-statistics)
- [Notes](#notes)

---

## mean_local

Local mean (average pooling) within a sliding window.

```python
mean_local(data, *, local_size=None, step=None, weight=None,
           output=None, output_start=None, output_step=None, **kwarg)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data` | required | Input; 1D–ND |
| `local_size` | `(1,1,...)` | Window size (tuple/int/`default_contain`) |
| `step` | `(1,1,...)` | Sliding step |
| `weight` | `None` | Custom kernel replacing uniform ones (should match `local_size`) |
| `output` / `output_start` / `output_step` | `None` / 0 / 1 | Pre-allocated output and placement |
| `**kwarg` | — | Forwarded to `cos_comparison_active` (e.g. `start`, `end`, callbacks) |

**Output size:** `((data_size - local_size) // step) + 1`. Raises `ValueError("effectless args.")` if window doesn't fit.

**Implementation:** active mode with all-ones kernel and `algorithm = lambda a, b, ab, name: ab / N` (with ones kernel, `ab` = window sum, `N` = element count).

```python
cc.mean_local_1d([1, 2, 3, 4, 5, 6], local_size=(3,), step=(1,))
# [2.0, 3.0, 4.0, 5.0]

cc.mean_local_2d(image, local_size=(2, 2), step=(2, 2))  # 2×2 average pooling
```

---

## local_variance

Local variance within a sliding window. Same parameters as `mean_local`.

**Formula:** `variance = E[X²] − E[X]²`

**Implementation:** `algorithm = lambda a, b, ab, name: a/N − (ab/N)²` (with ones kernel, `a` = sum of squares, `ab` = sum).

```python
cc.local_variance_1d([1, 2, 3, 4, 5], local_size=(3,), step=(1,))
# Window [1,2,3]: var ≈ 0.667

variance_map = cc.local_variance_2d(image, local_size=(5, 5))  # texture strength
```

---

## Dimension Aliases

```python
mean_local_1d/2d/3d/4d(...)        local_variance_1d/2d/3d/4d(...)
```

---

## Tensor Statistics

`vector_map_as_tensor` provides global statistics on whole tensors or views:

```python
t = cc.load_as_default_data([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
t.mean()      # 5.0 (Welford's online algorithm)
t.variance()  # 6.6667
t[1:3].mean() # 6.5 (on a view)
```

---

## Notes

- Statistics inherit active mode's performance, backend acceleration, and callbacks
- Only uniform windows via `local_size`; pass `weight` for custom kernels, or use `cos_comparison_active` directly
- `mean_local` = average pooling; no max pooling currently
- The `E[X²]−E[X]²` formula may suffer catastrophic cancellation for tiny variances; tensor-level `variance()` uses numerically stable Welford's algorithm
- Standard deviation: `math.sqrt(v)` per element

---

**See also:** [Active Mode](active-mode.md) · [Core Module](core.md)
