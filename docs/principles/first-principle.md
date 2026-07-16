# First Principle: Information Arises from Local Comparison

## Core Proposition

**Information is produced by local comparison in raw data.**

This is the foundational philosophical basis of cos-comparison, distinguishing it from the mainstream deep learning paradigm of "statistical fitting".

## Philosophical Foundation

### From Absolute to Relative

Traditional information processing paradigms focus on **absolute values** of data:
- Pixel intensity
- Signal strength
- Numerical magnitude

Cos-comparison posits that **information resides not in absolute values, but in differences and relationships**.

> The gray value of a single pixel is meaningless. Its difference from surrounding pixels constitutes edges, textures, shapes, and other information.

### Neuroscience Basis

This principle directly derives from the **centre-surround antagonism** mechanism in neuroscience:

- **Retinal ganglion cells**: Centre excitation with surround inhibition (or vice versa)
- **Receptive field structure**: Sensitive to local differences, insensitive to uniform regions
- **Information compression**: Removes redundant uniform information, preserves meaningful differences

Biological visual systems, from the lowest level, do not record absolute brightness but extract local contrast.

## Mathematical Formulation

### Basic Form

For two local regions A and B in data, information is extracted by comparing their similarity:

```
Information = f(Similarity(A, B))
```

Where:
- A, B are two local windows in the data
- f is a transformation function
- Lower similarity indicates greater difference and higher information content

### Sliding Window Implementation

Local comparison is performed across the entire data using a sliding window approach:

1. Define `window_size` — the size of local comparison windows
2. Define displacement vector `d` — offset between the two windows
3. Slide across the data, computing similarity at each position
4. Similarity values form a feature map/sequence

## Why This Matters

### 1. Generality

Local comparison is an extremely fundamental operation applicable to:
- 1D: Audio, time series, 1D signals
- 2D: Images, 2D matrices
- 3D: Video, volumetric data, 3D tensors
- 4D: Spatiotemporal volumetric data
- ...arbitrary dimensions

### 2. Data Efficiency

- **Zero-shot**: No training data required
- **Unsupervised**: No labels needed
- **Ready-to-use**: Directly applicable to any data

### 3. Interpretability

Every output value has clear geometric meaning:
- High similarity → Local uniformity, low information
- Low similarity → Large local difference, high information

No black box problems.

### 4. Computational Efficiency

- Linear time complexity O(N)
- Naturally parallelizable
- Low memory footprint

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
