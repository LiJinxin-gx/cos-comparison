# Atomic Contrast Point Matching - Exploration Code

## Overview

This directory contains independent, self-contained exploration modules for
structure-first pattern matching using contrast point sets. Based on the
PCML v10 series research (v10.37-v10.42).

**Key result**: Unsupervised atomic contrast point matching achieves **47.1%**
on 30-class speech recognition, exceeding supervised methods (22.6%).

## Core Principles

1. **Continuous mapping level isolation** - hierarchical coarse-to-fine matching
2. **Passive boundary extraction** - multi-scale element-gap detection
3. **Active matching masking** - coarse level narrows candidate set
4. **Structure-first** - no label prior in feature extraction
5. **Local aggregation** - atomic samples, no global features
6. **Selective combination** - KNN selection, no weighted voting
7. **Contrast point set intersection** - Jaccard similarity on binary sets
8. **Asymmetric storage** - different thresholds per feature group

## Modules

### Core Modules (independent, import only cos_comparison)

| Module | Purpose | Input | Output |
|--------|---------|-------|--------|
| `atomic_feature.py` | Feature extraction | 2D tensor | feature vector / grouped vectors |
| `contrast_match.py` | Jaccard KNN matching | binary feature matrix | predictions |
| `hierarchical_match.py` | Coarse-to-fine matching | grouped binary features | predictions |
| `text_encoder.py` | Text frequency encoding | text strings | frequency tensor (UnitMap) |
| `behavior_agent.py` | Behavior-composition Agent | explicit behavior registrations | execution results |

### Agent Framework

`behavior_agent.py` implements a von Neumann architecture (behavior-as-data):
behaviors are stored as data in the database, executed by fixed code.
All configuration is passed via explicit interface (no config files):

```python
from behavior_agent import Agent
agent = Agent(database=":memory:",
              stage_behaviors={"collect": ["search", "fetch"]},
              stage_keywords={"collect": ("search", "collect")})
agent.register_behavior("add", "mymodule.add", {"a": "num"}, {"a": 0})
result = agent.executor.execute("add", {"a": 3, "b": 4})
```

### Demo Scripts (self-contained, runnable)

| Script | Task | Data |
|--------|------|------|
| `demo_speech.py` | Speech recognition | WAV files in class directories |
| `demo_text.py` | Text classification | Text strings |
| `demo_image.py` | Image classification | Images in class directories |

## Quick Start

```python
# 1. Extract features from a 2D tensor
from atomic_feature import extract, binarize
feature = extract(spectrogram)  # 2D tensor -> 1D feature vector
binary = binarize(feature, top_percent=10)  # -> contrast point set

# 2. Match using Jaccard similarity
from contrast_match import ContrastMatcher
matcher = ContrastMatcher()
matcher.fit(train_binary, train_labels)
predictions = matcher.predict(test_binary, k=1)

# 3. Hierarchical matching (coarse boundary -> fine combined)
from hierarchical_match import HierarchicalMatcher
from atomic_feature import extract_grouped
train_b, train_r = extract_grouped(train_spec)  # separate groups
test_b, test_r = extract_grouped(test_spec)
hm = HierarchicalMatcher()
hm.fit(binarize(train_b), binarize(train_r), train_labels)
preds = hm.predict(binarize(test_b), binarize(test_r), coarse_n=10)

# 4. Text encoding using UnitMap
from text_encoder import encode_texts
tensor, encoder = encode_texts(texts, mode='character')
```

## Design Principles

- **Data loading separated from training**: demos include loaders, core
  modules only receive tensors or directly-readable data
- **Zero deep learning**: no gradients, no neural networks, no backpropagation
- **No weighted voting**: KNN uses majority vote only
- **No global features**: all analysis is local (atomic samples + pooling)
- **Asymmetric storage**: boundary features (strict threshold) vs raw features
  (looser threshold)
- **Independent modules**: each file can run standalone with `python file.py`

## Performance Reference (Speech Commands, 30 classes)

| Method | Train/class | Accuracy |
|--------|------------|----------|
| Supervised Fisher | 100 | 22.6% |
| K-means unsupervised | 200 | 7.2% |
| Atomic flat matching | 500 | 42.9% |
| Atomic flat matching | 1000 | 46.0% |
| **Hierarchical matching** | **500** | **47.1%** |

## Dependencies

- `cos_comparison` (core project)
- Python standard library only (no numpy, scipy, Pillow, or other third-party deps)

All modules use pure Python standard library. Image loading in demos uses
tkinter PhotoImage (GIF/PNG/PGM/PPM). Audio loading uses the standard `wave`
module.
