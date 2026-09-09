# Atomic Contrast Point Matching (PCML v10.37-v10.42)

Exploration of unsupervised structure-first pattern matching using contrast
point sets (Jaccard similarity) instead of global feature vectors.

## Core Idea

Traditional matching uses global feature vectors and cosine similarity. This
exploration uses **binary contrast point sets** (top X% response dimensions)
and **Jaccard similarity** (set intersection / union), combined with
hierarchical coarse-to-fine matching.

**Key principle**: information arises from differences; only the most
distinctive contrast points matter for matching.

## Results Summary

### Speech Recognition (Google Speech Commands, 30 classes)

| Version | Method | Train/class | Threshold | Accuracy | Notes |
|---------|--------|------------|-----------|----------|-------|
| v10.36 | Supervised Fisher + 2-stage | 100 | — | 22.6% | Baseline |
| v10.37 | Unsupervised K-means | 80 | — | 7.2% | 0 pure clusters; labels are fuzzy |
| v10.38 | Atomic Jaccard KNN | 50 | top30% | 16.2% | Beats K-means 2.25x |
| v10.39 | Atomic Jaccard KNN | 200 | top20% | **31.6%** | First unsupervised > supervised |
| v10.40 | Atomic Jaccard KNN | 500 | top10% | **42.9%** | Stricter = better |
| v10.41 | Atomic Jaccard KNN | 1000 | top10% | 46.0% | Diminishing returns |
| v10.42 | Multi-scale + Hierarchical | 500 | b10%/r10% | **47.1%** | Algorithm > data scaling |

### Scaling Trend (v10.38-v10.41, flat matching, top10%)

| Train/class | Accuracy | Gain per +100/class |
|-------------|----------|---------------------|
| 50 | 16.2% | — |
| 200 | 31.6% | +10.3% |
| 500 | 42.9% | +3.8% |
| 1000 | 46.0% | +0.6% |

Conclusion: marginal gains diminish; algorithmic improvement (v10.42
hierarchical) is more effective than further data scaling.

### Threshold Sweep (v10.40, 500 train/class, K=1)

| Threshold | Active points | Accuracy |
|-----------|--------------|----------|
| top 10% | ~1404 | **42.9%** |
| top 15% | ~2104 | 40.2% |
| top 20% | ~2805 | 38.3% |
| top 25% | ~3505 | 31.4% |

Conclusion: stricter binarization preserves only the most distinctive
contrast points; including weaker points adds noise.

### K Value Sweep (v10.39, 200 train/class, top20%)

| K | Accuracy |
|---|----------|
| 1 | **31.6%** |
| 3 | 25.9% |
| 5 | 29.9% |

Conclusion: K=1 (nearest neighbor) is best; majority voting introduces noise
from wrong neighbors. Aligns with "selective combination, not voting"
principle.

### Hierarchical Matching (v10.42, 500 train/class)

| Config | Accuracy |
|--------|----------|
| Baseline (flat, top10%) | 42.9% |
| Multi-scale flat (b5%/r15%) | 45.7% |
| Hierarchical b5/r10/n10 | 46.3% |
| **Hierarchical b10/r10/n10** | **47.1%** |
| Hierarchical b10/r20/n10 | 46.8% |

Conclusion: two-level matching (coarse boundary -> fine combined) improves
over flat matching by active candidate masking.

## Method Pipeline

```
2D tensor (spectrogram / image / reshaped text)
    |
    v
Passive boundary extraction (multi-scale, element gaps)
    |
    +---> Group A: boundary features (strict threshold, e.g. top10%)
    |
    +---> Group B: raw pooling features (looser threshold, e.g. top10-15%)
    |
    v
Binarize -> contrast point sets (uint8)
    |
    v
Level 1 (coarse): Jaccard match on boundary features -> top-N candidate classes
    |
    v
Level 2 (fine): Jaccard match on combined features within candidates only
    |
    v
Prediction (nearest neighbor, K=1)
```

## Key Findings

1. **Contrast point sets beat global vectors**: Jaccard on binary sets
   outperforms cosine on full vectors for speech (47.1% vs 22.6%).

2. **Unsupervised can exceed supervised**: when structure discovery is right
   (contrast points), no label prior is needed.

3. **Stricter is better**: top 10% binarization works best; only the most
   distinctive dimensions matter.

4. **K=1 is optimal**: majority voting (K>1) hurts; nearest neighbor is
   cleanest selective combination.

5. **Algorithm > data**: hierarchical matching (500/class, 47.1%) beats flat
   matching (1000/class, 46.0%).

6. **Labels are fuzzy**: K-means clustering produced 0 pure clusters,
   confirming speech-label correspondence is not one-to-one.

## Principle Alignment (8/8)

| Principle | Implementation |
|-----------|---------------|
| Continuous mapping level isolation | 2-level hierarchical matching |
| Passive boundary | Multi-scale element-gap extraction |
| Active matching masking | Coarse level masks candidate set |
| Structure-first | Unsupervised, no label prior |
| Local aggregation | Atomic samples, no global features |
| Selective combination | K=1 nearest neighbor, no voting |
| Contrast point set intersection | Jaccard similarity on binary sets |
| Asymmetric storage | Different thresholds per feature group |

## Code Location

Exploration modules in `explore/` directory:

- `atomic_feature.py` — feature extraction (passive boundary + pooling)
- `contrast_match.py` — Jaccard KNN matching
- `hierarchical_match.py` — coarse-to-fine hierarchical matching
- `text_encoder.py` — text frequency encoding (UnitMap)
- `demo_speech.py` — speech recognition demo
- `demo_text.py` — text classification demo
- `demo_image.py` — image classification demo

## Datasets

- **Google Speech Commands v0.01**: 64,721 files, 30 word classes, 16kHz,
  ~1s duration, multiple speakers (downloaded to local testdata)
- **FSDD**: cleaned to 1,602 high-quality files from 4 speakers
  (removed 1,398 low-quality: low volume, excessive silence, clipping)

## References

- Internal record: `PCML_V10_SERIES_RECORD.md` (v10.33-v10.42)
- Math analysis: `PCML_MATH_RIGOR_ANALYSIS.md` (30 properties P1-P30)
- Principle mapping: `CORE_PRINCIPLES_MATH_MAPPING.md`
