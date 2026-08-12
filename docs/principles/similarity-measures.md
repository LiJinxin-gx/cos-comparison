# Similarity Measures

Cos-comparison provides three complementary similarity algorithms, each capturing different aspects of the relationship between two vectors. All three follow a **universal form**: given the accumulated sums `a = Σ A²` (sum of squares of vector 1), `b = Σ B²` (sum of squares of vector 2) and `ab = Σ A·B` (dot product), the similarity is `f(a, b, ab)`.

## Overview

| Algorithm | Formula (given a, b, ab) | Measures | Default |
|-----------|--------------------------|----------|---------|
| `_cos` | `ab / sqrt(a * b)` | Angular similarity (direction) | |
| `_mod` | `2 * sqrt(a * b) / (a + b)` | Magnitude similarity (amplitude) | |
| `_cosmod` | `2 * ab / (a + b)` | Combined direction + magnitude | ✅ Yes |

## Mathematical Definitions

### 1. Cosine Similarity (`_cos`)

```
cos(A, B) = (A · B) / (‖A‖ · ‖B‖)
```

**Geometric interpretation**:
- Measures the cosine of the angle between two vectors
- Ranges from -1 to 1 (for non-negative data: 0 to 1)
- 1 = identical direction, 0 = orthogonal, -1 = opposite direction

**Use cases**:
- When only direction matters, not magnitude
- Pattern recognition where intensity variations are irrelevant
- Normalized feature comparison

**Properties**:
- Scale-invariant: multiplying one vector by a scalar does not change cosine similarity
- Sensitive to direction changes
- Insensitive to absolute magnitude

### 2. Magnitude Similarity (`_mod`)

```
mod(A, B) = 2 · ‖A‖ · ‖B‖ / (‖A‖² + ‖B‖²)
```

**Geometric interpretation**:
- Measures how close the magnitudes of two vectors are
- Based on the ratio of geometric mean to arithmetic mean of the squared norms
- Ranges from 0 to 1
- 1 = identical magnitude, approaches 0 as magnitudes diverge

**Use cases**:
- When only magnitude matters, not direction
- Energy/intensity comparison
- Contrast detection

**Properties**:
- Direction-invariant: rotating a vector does not change magnitude similarity
- Sensitive to magnitude differences
- Equals 1 when ‖A‖ = ‖B‖

### 3. Cosine-Modulated Similarity (`_cosmod`)

```
cosmod(A, B) = 2 · (A · B) / (‖A‖² + ‖B‖²)
```

**Geometric interpretation**:
- Combines both direction and magnitude information
- Equivalent to cosine similarity modulated by magnitude similarity
- Ranges from 0 to 1 for non-negative vectors
- 1 = identical vectors (same direction AND same magnitude)

**Derivation**:

```
cosmod = 2 · (A·B) / (‖A‖² + ‖B‖²)
       = [ (A·B) / (‖A‖·‖B‖) ] · [ 2 · ‖A‖·‖B‖ / (‖A‖² + ‖B‖²) ]
       = cos(A, B) · mod(A, B)
```

The combined measure is exactly the product of the cosine and magnitude similarities — both pattern shape and intensity must match for a high score.

**Use cases**:
- General-purpose similarity measure
- Edge detection (both direction and magnitude changes matter)
- Template matching (both pattern and intensity must match)

**Properties**:
- Sensitive to both direction and magnitude
- More strict than cosine alone
- Default algorithm in cos-comparison
- Numerically efficient: no square root needed

## Edge Case Handling

All three algorithms handle zero vectors gracefully:

```python
# If a * b == 0 (at least one vector is zero):
if a == b:
    return 1.0  # Both zero → identical
else:
    return 0.0  # One zero, one non-zero → completely different
```

This ensures numerical stability and meaningful results even when local regions are uniform or empty.

## Choosing the Right Algorithm

| Scenario | Recommended Algorithm | Reason |
|----------|----------------------|--------|
| Edge detection | `_cosmod` (default) | Captures both intensity and direction changes |
| Texture analysis | `_cos` | Directional patterns matter more than magnitude |
| Contrast detection | `_mod` | Magnitude differences indicate contrast |
| Template matching | `_cosmod` | Both pattern shape and intensity must match |
| Normalized comparison | `_cos` | Scale-invariant comparison |

## Implementation Notes

- All algorithms are implemented as lightweight functions with the signature `func(a, b, ab, name)`
- Accept pre-computed `a` (sum of squares of window 1), `b` (sum of squares of window 2), and `ab` (dot product)
- This design avoids redundant computation in sliding window scenarios
- The `name` parameter is a `func_name_space` object (may be `None`), reserved for future extensibility
- The `private_dict` container maps each algorithm name to its function for dynamic lookup
- The `_default_algorithm` symbol points to `_cosmod` and is the default of all core comparison functions

---

**Next**: [Dual Working Modes](dual-mode.md) — Passive and Active modes