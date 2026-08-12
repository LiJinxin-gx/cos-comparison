# First Principle: Information Arises from Local Comparison

## Core Proposition

**Information is produced by local comparison in raw data.**

This is the foundational philosophical basis of cos-comparison, distinguishing it from the mainstream deep learning paradigm of "statistical fitting".

## Philosophical Foundation

### From Absolute to Relative

Traditional information processing paradigms focus on **absolute values** of data (pixel intensity, signal strength, numerical magnitude). Cos-comparison posits that **information resides not in absolute values, but in differences and relationships**.

> The gray value of a single pixel is meaningless. Its difference from surrounding pixels constitutes edges, textures, shapes, and other information.

### Neuroscience Basis

This principle directly derives from the **centre-surround antagonism** mechanism in neuroscience:

- **Retinal ganglion cells**: Centre excitation with surround inhibition (or vice versa)
- **Receptive field structure**: Sensitive to local differences, insensitive to uniform regions
- **Information compression**: Removes redundant uniform information, preserves meaningful differences

Biological visual systems, from the lowest level, do not record absolute brightness but extract local contrast.

## Mathematical Formulation

For two local regions A and B in data, information is extracted by comparing their similarity:

```
Information = f(Similarity(A, B))
```

Where A, B are two local windows in the data, f is a transformation function, and lower similarity indicates greater difference and higher information content.

Local comparison is performed across the entire data using a **sliding window** approach: define `window_size` (size of comparison windows) and displacement `d` (offset between the two windows), then slide across the data, computing similarity at each position to form a feature map/sequence. See [Dual Working Modes](dual-mode.md) for the concrete mechanism.

## Why This Matters

1. **Generality**: Local comparison applies to arbitrary dimensions — 1D (audio, time series), 2D (images), 3D (video, volumetric data), 4D (spatiotemporal data), and beyond
2. **Data efficiency**: Zero-shot (no training data), unsupervised (no labels), ready to apply to any data
3. **Interpretability**: Every output has clear geometric meaning — high similarity = local uniformity (low information), low similarity = large local difference (high information)
4. **Computational efficiency**: Linear time complexity O(N), naturally parallelizable, low memory footprint

## Comparison with Mainstream Paradigms

| Dimension | Local Comparison Paradigm | Statistical Fitting Paradigm |
|-----------|--------------------------|-----------------------------|
| Core Idea | Information from differences | Information from statistical patterns |
| Data Requirement | Zero-shot | Massive data |
| Learning Method | No learning needed | Batch training |
| Interpretability | 100% interpretable | Black box |
| Generalization | Principle-based | Interpolation-based |
| Complexity | Linear | Quadratic / Cubic |

## Broader Implications

Local comparison is not merely an algorithm, but a **worldview**:

1. **Perception level**: All sensation essentially detects changes and differences
2. **Cognition level**: Knowledge originates from comparison and classification
3. **Intelligence level**: The foundation of general intelligence may be this fundamental comparison mechanism

From this perspective, cos-comparison is more than a feature extraction library — it explores a path to building general intelligence from first principles.

---

**Next**: [Similarity Measures](similarity-measures.md) — Three specific similarity metrics