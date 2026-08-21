# Dual Working Modes

Two complementary modes: **Passive** (self-similarity) and **Active** (template matching). This duality extends across all architecture layers.

## Contents

- [Overview](#overview)
- [Passive Mode](#passive-mode)
- [Active Mode](#active-mode)
- [Relationship](#relationship)
- [Selection Guide](#selection-guide)

---

## Overview

| Mode | Core Idea | Analogy | Key Parameter |
|------|-----------|---------|---------------|
| Passive | Self-similarity within data | Reflex / bottom-up | Displacement `d` |
| Active | Matching against external template | Attention / top-down | `kernel` |

---

## Passive Mode

Compares two offset windows within the same data:

```
result[x] = similarity(data[x:x+window], data[x+d:x+d+window])
```

**Characteristics:**
- Template-free, unsupervised, reflex-like
- Naturally detects edges and boundaries
- Varying window size (scale), displacement magnitude (comparison distance), and direction (oriented features) enables multi-scale, multi-directional extraction

**Use cases:** edge detection, texture analysis, motion detection, anomaly/keypoint detection.

---

## Active Mode

Slides an external kernel across data:

```
result[x] = similarity(data[x:x+window], kernel)
```

**Characteristics:**
- Template-driven, goal-oriented
- Finds occurrences of known patterns
- Kernel design: size (feature scale), shape (pattern), values (weighting), normalization

**Use cases:** template matching, feature detection, pattern recognition, convolution-like operations.

---

## Relationship

- **Passive** = what data tells about itself; **Active** = what we look for in data
- Mathematically, passive is active with a data-derived kernel (shifted window)
- Biologically: passive ≈ feedforward sensory pathways; active ≈ top-down attention

---

## Selection Guide

| Scenario | Mode |
|----------|------|
| Unknown data exploration, edge detection, unsupervised features, texture | Passive |
| Finding specific patterns, target detection | Active |
| Richer representations | Combine both (passive candidates → active matching) |

---

**Next:** [Seven-Layer Architecture](../architecture/seven-layer.md) · [Similarity Measures](similarity-measures.md)
