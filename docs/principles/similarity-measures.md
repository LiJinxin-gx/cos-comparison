# Similarity Measures

Cos-comparison provides three complementary similarity algorithms, each capturing different aspects of the relationship between two vectors.

## Overview

| Algorithm | Formula | Measures | Default |
|-----------|---------|----------|---------|
| `_cos` | `ab / sqrt(a * b)` | Angular similarity (direction) | |
| `_mod` | `2 * sqrt(a * b) / (a + b)` | Magnitude similarity (amplitude) | |
| `_cosmod` | `2 * ab / (a + b)` | Combined direction + magnitude | ✅ Yes |

## Mathematical Definitions

### 1. Cosine Similarity (`_cos`)

```
cos(a, b) = (a · b) / (||a|| * ||b||)
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
mod(a, b) = 2 * sqrt(||a|| * ||b||) / (||a|| + ||b||)
```

**Geometric interpretation**:
- Measures how close the magnitudes of two vectors are
- Based on the ratio of geometric mean to arithmetic mean
- Ranges from 0 to 1
- 1 = identical magnitude, approaches 0 as magnitudes diverge

**Use cases**:
- When only magnitude matters, not direction
- Energy/intensity comparison
- Contrast detection

**Properties**:
- Direction-invariant: rotating a vector does not change magnitude similarity
- Sensitive to magnitude differences
- Equals 1 when ||a|| = ||b||

### 3. Cosine-Modulated Similarity (`_cosmod`)

```
cosmod(a, b) = 2 * (a · b) / (||a||² + ||b||²)
```

**Geometric interpretation**:
- Combines both direction and magnitude information
- Equivalent to cosine similarity modulated by magnitude similarity
- Ranges from 0 to 1 for non-negative vectors
- 1 = identical vectors (same direction AND same magnitude)

**Derivation**:
```
cosmod = 2 * (a · b) / (||a||² + ||b||²)
       = [2 * sqrt(||a|| * ||b||) / (||a|| + ||b||)] * [(a · b) / (||a|| * ||b||)] * [2 * ||a|| * ||b|| / (||a||² + ||b||)]
       ≈ mod * cos * correction_factor
```

**Use cases**:
- General-purpose similarity measure
- Edge detection (both direction and magnitude changes matter)
- Template matching (both pattern and intensity must match)

**Properties**:
- Sensitive to both direction and magnitude
- More strict than cosine alone
- Default algorithm in cos-comparison

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

- All algorithms are implemented as lightweight lambda functions
- Accept pre-computed `a` (sum of squares of vector 1), `b` (sum of squares of vector 2), and `ab` (dot product)
- This design avoids redundant computation in sliding window scenarios
- The `name` parameter is reserved for future extensibility

---

**Next**: [Dual Working Modes](dual-mode.md) — Passive and Active modes
