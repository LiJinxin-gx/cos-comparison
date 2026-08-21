# Passive Mode API

Self-similarity: compares two offset sliding windows within data, detecting edges, boundaries, and texture variations without an external template.

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
cos_comparison_passive(data, *, window_size=None,
                       w1=1, w2=1, b1=0, b2=0,
                       start=None, end=None, step=None, d=None,
                       algorithm=_default_algorithm,
                       output=None, output_start=None, output_step=None,
                       start_callback=None, end_callback=None,
                       global_error_callback=None, local_error_callback=None,
                       return_callback=lambda output, name: output, **kwargs)
```

---

## Parameters

Shared parameters (`data`, `window_size`, `w1`/`w2`/`b1`/`b2`, `start`/`end`/`step`, `algorithm`, `output*`, callbacks) are documented in [Common Parameters](core.md#common-parameters).

**Mode-specific:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `d` | tuple of ints | `(1, 0, 0, ...)` | Displacement vector between the two comparison windows; length must match data dimension |

**`d` examples (2D):**

| `d` | Comparison | Detects |
|-----|------------|---------|
| `(1, 0)` | Bottom neighbor | Horizontal edges |
| `(0, 1)` | Right neighbor | Vertical edges |
| `(1, 1)` | Diagonal neighbor | Diagonal edges |

- `data` duck typing: if `data` exposes `__cos_comparison_passive__`, the call is delegated to it

---

## Returns

`vector_map_as_tensor` (or `return_callback` result). Output size per dimension:

```
num[i] = ((end[i] - start[i] - window_size[i] - d[i]) // step[i]) + 1
```

Raises `ValueError("effectless args.")` if window + displacement doesn't fit.

---

## Aliases

```python
cos_comparison_passive_1d/2d/3d/4d(data, ...)
```

---

## Examples

### 1D Edge Detection

```python
data = [1.0, 1.0, 1.0, 5.0, 5.0, 5.0]
result = cc.cos_comparison_passive_1d(data, window_size=(2,), step=(1,), d=(1,))
# Low similarity at the 1.0→5.0 transition
```

### 2D Horizontal / Vertical Edges

```python
# Horizontal edges (compare with bottom neighbor)
h = cc.cos_comparison_passive_2d(image, window_size=(3, 3), d=(1, 0))
# Vertical edges (compare with right neighbor)
v = cc.cos_comparison_passive_2d(image, window_size=(3, 3), d=(0, 1))
```

### Multi-Scale

```python
for ws in [(3, 3), (5, 5), (7, 7)]:
    features = cc.cos_comparison_passive_2d(image, window_size=ws, d=(1, 0))
```

### Pre-allocated Output

```python
output = cc.create_void_list((4, 4))
cc.cos_comparison_passive_2d(image, window_size=(3, 3), d=(1, 0), output=output)
```

---

## Tips

- Larger window/step trade off: more computation per position vs fewer positions
- Avoid `local_error_callback` in performance-critical code
- Both windows must fit entirely; no padding; invalid params raise `ValueError`
- All algorithms handle zero vectors gracefully (division-by-zero protected)

---

**See also:** [Active Mode](active-mode.md) · [Similarity Measures](../principles/similarity-measures.md) · [Core Module](core.md)
