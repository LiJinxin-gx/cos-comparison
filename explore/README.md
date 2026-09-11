# Explore: Group-Theoretic Continuous Mapping Hierarchical Isolation

Algorithm exploration code for the cos-comparison project.
All code is zero-dependency (stdlib + cos-comparison core only), each file runs standalone.

## Core Principles

### Transformation Group Structure

Continuous mappings between levels form a transformation group:

```
Level spaces: {L_0, L_1, ..., L_n}
Mappings:     M = {f_{i,j}: L_i -> L_j | f is continuous}
```

Group axioms:

| Axiom | Verification | Note |
|-------|-------------|------|
| Closure | f_{i,j} o f_{j,k} = f_{i,k} | Composition of continuous maps is continuous |
| Associativity | (f o g) o h = f o (g o h) | Function composition is associative by definition |
| Identity | f_{i,i} = id | Identity map |
| Inverse | f_{j,i} = f_{i,j}^{-1} | Structural inverse (not exact, info loss) |

### Bidirectional Mapping A <-> B

- Down-sampling f_{i,i+1}: L_i -> L_{i+1} (coarsening)
- Up-sampling f_{i+1,i}: L_{i+1} -> L_i (refinement)
- These are **structural inverses**: shape preserved, but not exact inverses
  (information is lost during coarsening).

### Primitive Isolation

- **Active mode**: extract raw values (features of the data itself)
- **Passive mode**: extract local differences (gaps between elements)
- **Critical**: these are different data types, NEVER averaged together
- Decision-layer fusion: take the max of both channels (not weighted sum)

### Level Isolation

- Each level stores features independently
- No values flow across levels; only structural correspondence
- No cross-level weighted propagation

## Mathematical Rigor Analysis

### 1. Why a transformation group?

Mappings between levels must satisfy algebraic closure. If f_{i,j} and f_{j,k}
are continuous, their composition f_{i,k} is also continuous. This guarantees
that a unique mapping exists between any pair of levels.

Operations that break closure (e.g. cross-primitive weighted fusion) are not
group elements and corrupt the algebraic structure.

### 2. Why structural inverse, not exact inverse?

Down-sampling is many-to-one; up-sampling is one-to-many.
f_{i+1,i} o f_{i,i+1} != id — information is lost in coarsening.
The up-sample restores shape, not content. Generation tasks must accept
this information-theoretic loss.

### 3. Why primitives must be isolated?

Active features live in the data space R^n (element values).
Passive features live in the interstitial space R^{n-1} (gaps).
Different norms, different scales, different physical meaning.
Averaging them is like adding apples and oranges.

Experimental evidence (v11.46 ablation):
- Active only: 95.00%
- Active + passive average: 93.86% (-1.14%)
Low-variance passive features dilute high-variance active signals.

### 4. Why no cross-level weighted propagation?

L_0, L_1, L_2 have different dimensionality. Weighted propagation
(H' = H + w*V^T) assumes feature-space isomorphism — a deep-learning
residual trick that is not valid here. Each level is an independent
feature space; only structural correspondence is shared.

Experimental evidence:
- Strict top-down (v11.38): L1 dropped 42%
- All levels simultaneously constrained (v11.40): L1 improved 7.3%

### 5. Why K=1, no voting?

Voting introduces a human prior ("majority is correct") that is not
in the data. K=1 nearest neighbor directly compares the query to its
closest prototype. No artificial thresholds, no confidence gates.

### 6. Why pooling provides tolerance?

Average pooling is a low-pass filter. High-frequency handwriting
deformations (small stroke shifts) are smoothed out; low-frequency
structure (overall digit shape) is preserved.

Experimental evidence:
- L0 only: 92.60%
- L0+L1: 93.86% (+1.26%)
- L0+L1+L2: 95.08% (+1.22%)

## Files

| File | Function | Principle |
|------|----------|-----------|
| `group_hierarchical.py` | Core classifier | Transformation group + bidirectional mapping + primitive isolation |
| `demo_image.py` | Image classification demo | 2D tensor -> 3 levels -> cosine K=1 |
| `demo_text.py` | Text classification demo | Frequency encoding -> 1D levels -> cosine K=1 |
| `demo_speech.py` | Speech classification demo | WAV -> spectrogram -> 3 levels -> cosine K=1 |

## Validated Configuration (MNIST)

- Pool factors: [1, 2, 4]
- Active mode only (passive averaging degrades signal by -1.14%)
- Similarity: cosine K=1 nearest neighbor
- Prototypes: 500 per class
- Accuracy: 95.92% (single process) / 97.08% (full group architecture)

## Directions Verified to Fail

- Active + passive average fusion (dilutes signal)
- Cross-primitive weighted operations
- Transposed propagation (deep-learning trick)
- Strict top-down propagation
- Staircase / asymmetric pooling
- Patch statistical features (loses spatial structure)

## Usage

```bash
# Image classification (synthetic demo)
python demo_image.py

# Image classification (real data)
python demo_image.py <data_dir> [size] [n_per_class]

# Text classification
python demo_text.py

# Speech classification
python demo_speech.py
```

## Data Format

All processing code accepts plain tensors (list of lists), independent of
file format:

- Image: 2D list [[gray, ...], ...]
- Text: frequency-encoded to 1D list
- Speech: WAV -> spectrogram as 2D list
