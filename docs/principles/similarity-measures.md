# Similarity Measures

Three complementary algorithms sharing a universal form: given `a = ΣA²`, `b = ΣB²`, `ab = ΣA·B`, the similarity is `f(a, b, ab)`.

## Contents

- [Overview](#overview)
- [Cosine (`_cos`)](#cosine-_cos)
- [Magnitude (`_mod`)](#magnitude-_mod)
- [Cosine-Modulated (`_cosmod`)](#cosine-modulated-_cosmod)
- [Edge Case Handling](#edge-case-handling)
- [Choosing an Algorithm](#choosing-an-algorithm)
- [Implementation](#implementation)

---

## Overview

| Algorithm | Formula | Measures | Default |
|-----------|---------|----------|---------|
| `_cos` | `ab / sqrt(a·b)` | Angular similarity (direction) | |
| `_mod` | `2·sqrt(a·b) / (a+b)` | Magnitude similarity (amplitude) | |
| `_cosmod` | `2·ab / (a+b)` | Combined direction + magnitude | ✅ |

---

## Cosine (`_cos`)

`cos(A,B) = (A·B) / (‖A‖·‖B‖)` — cosine of the angle between vectors.

| Property | Value |
|----------|-------|
| Range | −1 to 1 (0 to 1 for non-negative data) |
| Scale-invariant | Yes |
| Best for | Direction matters, intensity doesn't (texture, normalized features) |

---

## Magnitude (`_mod`)

`mod(A,B) = 2·‖A‖·‖B‖ / (‖A‖²+‖B‖²)` — ratio of geometric to arithmetic mean of squared norms.

| Property | Value |
|----------|-------|
| Range | 0 to 1 |
| Direction-invariant | Yes |
| Equals 1 when | ‖A‖ = ‖B‖ |
| Best for | Energy/intensity comparison, contrast detection |

---

## Cosine-Modulated (`_cosmod`)

`cosmod(A,B) = 2·(A·B) / (‖A‖²+‖B‖²) = cos(A,B) · mod(A,B)` — product of cosine and magnitude.

| Property | Value |
|----------|-------|
| Range | 0 to 1 (non-negative vectors) |
| Sensitive to | Both direction AND magnitude |
| Efficient | No square root needed |
| Best for | General-purpose edge detection, template matching (default) |

Both shape and intensity must match for a high score.

---

## Edge Case Handling

If `a·b == 0` (at least one zero vector):

| Condition | Result |
|-----------|--------|
| Both zero | `1.0` (identical) |
| One zero, one non-zero | `0.0` (different) |

Division-by-zero is always avoided.

---

## Choosing an Algorithm

| Scenario | Algorithm |
|----------|-----------|
| Edge detection, template matching | `_cosmod` (default) |
| Texture, directional patterns, scale-invariant comparison | `_cos` |
| Contrast, magnitude/energy comparison | `_mod` |

---

## Implementation

All algorithms are `func(a, b, ab, name)` where `name` is a `func_name_space` (may be `None`, reserved for extensibility). Pre-computed sums avoid redundant work in sliding windows.

- `private_dict` maps names to functions
- `_default_algorithm` is `_cosmod`

---

**Next:** [Dual Modes](dual-mode.md) · [First Principle](first-principle.md)
