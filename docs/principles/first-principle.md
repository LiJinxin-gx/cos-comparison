# First Principle: Information Arises from Local Comparison

**Information is produced by local comparison in raw data.**

This is the foundational principle of cos-comparison, distinguishing it from deep learning's "statistical fitting" paradigm.

## Contents

- [From Absolute to Relative](#from-absolute-to-relative)
- [Mathematical Formulation](#mathematical-formulation)
- [Why This Matters](#why-this-matters)
- [vs Statistical Fitting](#vs-statistical-fitting)
- [Broader Implications](#broader-implications)

---

## From Absolute to Relative

Information resides not in absolute values but in differences and relationships. A single pixel's gray value is meaningless; its difference from surrounding pixels constitutes edges, textures, and shapes.

This derives from the neuroscience **centre-surround antagonism** mechanism: retinal ganglion cells are excited by their centre and inhibited by their surround (or vice versa), making them sensitive to local differences and insensitive to uniform regions — compressing redundancy while preserving meaningful contrast.

---

## Mathematical Formulation

For two local regions A and B:

```
Information = f(Similarity(A, B))
```

Lower similarity = greater difference = higher information content.

A sliding window with `window_size` and displacement `d` sweeps the data, computing similarity at each position.

→ [Dual Modes](dual-mode.md) for the concrete mechanism.

---

## Why This Matters

| Property | Description |
|----------|-------------|
| **Generality** | Applies to any dimension (1D audio, 2D images, 3D video, 4D spatiotemporal) |
| **Data efficiency** | Zero-shot, unsupervised, no training or labels |
| **Interpretability** | Every output has clear geometric meaning |
| **Computational efficiency** | Linear O(N), naturally parallelizable, low memory |

---

## vs Statistical Fitting

| Dimension | Local Comparison | Statistical Fitting |
|-----------|-----------------|---------------------|
| Core idea | Information from differences | Statistical patterns |
| Data | Zero-shot | Massive |
| Learning | None | Batch training |
| Interpretability | Fully interpretable | Black box |
| Complexity | Linear | Quadratic/cubic |

---

## Broader Implications

Local comparison is a worldview:

1. **Perception** — all sensation essentially detects changes and differences
2. **Cognition** — knowledge originates from comparison and classification
3. **Intelligence** — the foundation of general intelligence may be this fundamental comparison mechanism

From this perspective, cos-comparison is more than a feature extraction library — it explores a path to building general intelligence from first principles.

---

**Next:** [Similarity Measures](similarity-measures.md) · [Dual Modes](dual-mode.md)
