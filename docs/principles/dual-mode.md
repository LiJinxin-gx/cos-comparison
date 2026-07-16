# Dual Working Modes

Cos-comparison features two complementary working modes: **Passive Mode** and **Active Mode**. This dual-mode philosophy is a core design principle that extends across all layers of the architecture.

## Overview

| Mode | Core Idea | Analogy | Key Parameter |
|------|-----------|---------|---------------|
| Passive | Self-similarity within data | Reflex / Bottom-up | Displacement vector `d` |
| Active | Matching against external template | Attention / Top-down | Kernel `kernel` |

## Passive Mode

### Concept

**Passive mode** compares two offset windows within the same data, computing self-similarity across the dataset.

```
Data: [ 1, 2, 3, 4, 5, 6, 7, 8 ]
        ↑     ↑
      Window A  Window B
         <── d ──>
```

The displacement vector `d` controls both the direction and distance between the two comparison windows.

### Key Characteristics

1. **Template-free**: No external template required
2. **Unsupervised**: Works on any data without prior knowledge
3. **Reflex-like**: Analogous to automatic sensory processing
4. **Boundary detection**: Naturally detects edges, transitions, and boundaries

### Mathematical Formulation

For each position `x` in the data:

```
result[x] = similarity(data[x : x+window], data[x+d : x+d+window])
```

Where:
- `window` is the window size
- `d` is the displacement vector
- `similarity` is the chosen similarity algorithm

### Use Cases

- **Edge detection**: Displacement perpendicular to edge direction
- **Texture analysis**: Multiple displacement directions capture texture anisotropy
- **Motion detection**: Temporal displacement in video data
- **Anomaly detection**: Regions with unusual self-similarity patterns
- **Keypoint detection**: Points with high directional derivative diversity

### Multi-Scale and Multi-Direction

By varying:
- **Window size**: Captures features at different scales
- **Displacement magnitude**: Controls the scale of comparison
- **Displacement direction**: Captures oriented features

Passive mode naturally supports multi-scale, multi-directional feature extraction.

## Active Mode

### Concept

**Active mode** slides an external kernel (template) across the data, computing similarity between each local window and the kernel.

```
Data:   [ 1, 2, 3, 4, 5, 6, 7, 8 ]
          ↑  ↑  ↑
        Window A
Kernel: [ 1, 0, 1 ]
          ↑  ↑  ↑
```

The kernel serves as a "search template" that the algorithm actively looks for in the data.

### Key Characteristics

1. **Template-driven**: Requires an external kernel/template
2. **Goal-oriented**: Actively searches for specific patterns
3. **Attention-like**: Analogous to top-down attention
4. **Pattern matching**: Finds occurrences of known patterns

### Mathematical Formulation

For each position `x` in the data:

```
result[x] = similarity(data[x : x+window], kernel)
```

Where:
- `kernel` is the template to match
- `window` is determined by the kernel size
- `similarity` is the chosen similarity algorithm

### Use Cases

- **Template matching**: Finding specific patterns in data
- **Object detection**: Sliding window object detection
- **Feature detection**: Matching predefined feature templates
- **Pattern recognition**: Identifying known patterns
- **Convolution-like operations**: With similarity instead of dot product

### Kernel Design

The kernel is the "query" in active mode. Design considerations:

- **Size**: Determines the scale of features to detect
- **Shape**: Defines the pattern to match
- **Values**: Weighting of different positions within the kernel
- **Normalization**: Whether to normalize the kernel values

## Relationship Between Modes

### Complementary, Not Opposite

Passive and Active modes are complementary perspectives:

- **Passive** = what the data tells us about itself
- **Active** = what we look for in the data

### Mathematical Connection

Active mode can be seen as a generalization:
- When kernel = shifted version of data → becomes passive mode
- Passive mode is active mode with a data-derived kernel

### Cognitive Analogy

| Mode | Cognitive Analog | Neural Correlate |
|------|-----------------|------------------|
| Passive | Bottom-up processing | Feedforward sensory pathways |
| Active | Top-down attention | Feedback from higher brain areas |

Both modes work together in biological perception: passive processing extracts basic features, active attention selects relevant patterns.

## Mode Selection Guide

| Scenario | Recommended Mode | Reason |
|----------|-----------------|--------|
| Unknown data exploration | Passive | No prior assumptions needed |
| Edge detection | Passive | Naturally emerges from self-comparison |
| Finding specific patterns | Active | Template defines what to look for |
| Unsupervised feature extraction | Passive | Works without prior knowledge |
| Target detection | Active | Explicit target template |
| Texture analysis | Passive + multiple directions | Captures intrinsic texture properties |

## Advanced: Combining Modes

For more sophisticated processing, both modes can be combined:

1. **Passive → Active pipeline**: Use passive mode to extract candidate regions, then active mode for precise matching
2. **Multi-mode features**: Concatenate features from both modes for richer representations
3. **Feedback loops**: Active mode results modulate passive mode parameters (attention mechanism)

This layered, multi-mode approach mirrors how biological vision systems work.

---

**Next**: [Seven-Layer Cognitive Architecture](../architecture/seven-layer.md) — Brain-inspired layered design
