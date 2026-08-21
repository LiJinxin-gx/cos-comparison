# Active Mode API

Template matching: slides an external kernel across data, computing similarity between each local window and the kernel.

→ [Dual Modes](../principles/dual-mode.md) for conceptual background.

## Contents

- [Signature](#signature)
- [Parameters](#parameters)
- [Returns](#returns)
- [Aliases](#aliases)
- [Examples](#examples)
- [Tips](#tips)

---

## Signature

```python
cos_comparison_active(data, *, kernel=None,
                      w1=1, w2=1, b1=0, b2=0,
                      start=None, end=None, step=None,
                      algorithm=_default_algorithm,
                      output=None, output_start=None, output_step=None,
                      start_callback=None, end_callback=None,
                      global_error_callback=None, local_error_callback=None,
                      return_callback=lambda output, name: output, **kwargs)
```

---

## Parameters

Shared parameters (`data`, `w1`/`w2`/`b1`/`b2`, `start`/`end`/`step`, `algorithm`, `output*`, callbacks) are documented in [Common Parameters](core.md#common-parameters).

**Mode-specific:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `kernel` | nested list / tensor | **Required** (`ValueError` if `None`). Template to search for; same dimensionality as data; must be regular; window size set from kernel shape |

- `data` duck typing: if `data` exposes `__cos_comparison_active__`, the call is delegated to it

---

## Returns

`vector_map_as_tensor` (or `return_callback` result). Output size per dimension:

```
num[i] = ((end[i] - start[i] - kernel_size[i]) // step[i]) + 1
```

High values = good match. Raises `ValueError("effectless args.")` if kernel doesn't fit.

---

## Aliases

```python
cos_comparison_active_1d/2d/3d/4d(data, kernel=..., ...)
```

`kernel` must be a keyword argument.

---

## Examples

### 1D Pattern Matching

```python
data = [1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0]
result = cc.cos_comparison_active_1d(data, kernel=[3, 4, 5], step=(1,))
# Peak indicates where the pattern occurs
```

### 2D Template Matching

```python
result = cc.cos_comparison_active_2d(image, kernel=[[1, 1], [1, 1]], step=(1, 1))
# High values at positions where the 2×2 square pattern is found
```

### Directional Kernel (Horizontal Edges)

```python
hk = [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]
edges = cc.cos_comparison_active_2d(image, kernel=hk, algorithm=cc._cos)
```

### Pre-allocated Output

```python
output = cc.create_void_list((h, w))
cc.cos_comparison_active_2d(image, kernel=kernel, output=output)
```

---

## Tips

- **Normalize kernel** to unit length for cosine similarity
- **Small kernels** = fast, fine-grained; **large kernels** = slower, noise-robust
- **Larger `step`** = fewer positions = faster
- Avoid `local_error_callback` in performance-critical code
- Similar to convolution/correlation but uses normalized similarity (robust to intensity variation)
- All algorithms handle zero vectors gracefully (division-by-zero protected)

---

**See also:** [Passive Mode](passive-mode.md) · [Similarity Measures](../principles/similarity-measures.md) · [Statistics](statistics.md)
