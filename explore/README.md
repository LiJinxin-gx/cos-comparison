# Explore: Core Principle Implementations

Algorithm exploration code for the cos-comparison project.
**Zero dependencies**: pure Python standard library only.
Each file runs standalone. Data loading is separated from processing.

## Core Principles

### Transformation Group Structure

Continuous mappings between levels form a transformation group:

```
Level spaces: {L_0, L_1, ..., L_n}
Mappings:     M = {f_{i,j}: L_i → L_j | f is continuous}
```

Group axioms:

| Axiom | Verification | Note |
|-------|-------------|------|
| Closure | f_{i,j} ∘ f_{j,k} = f_{i,k} | Composition of continuous maps is continuous |
| Associativity | (f ∘ g) ∘ h = f ∘ (g ∘ h) | Function composition is associative |
| Identity | f_{i,i} = id | Identity map |
| Inverse | f_{j,i} = f_{i,j}⁻¹ | Structural inverse (not exact, info loss) |

### Bidirectional Mapping A ↔ B

- **Down-sampling** f_{i,i+1}: L_i → L_{i+1} (coarsening)
- **Up-sampling** f_{i+1,i}: L_{i+1} → L_i (refinement)
- These are **structural inverses**: shape preserved, not exact inverses (information is lost).

### Primitive Isolation

- **Active mode**: raw values (element space, features of the data itself)
- **Passive mode**: local differences (gap space, interstitial between elements)
- **Critical**: these are different data types, NEVER averaged together
- Decision fusion: take the max of both channels (not weighted sum)

### Level Isolation

- Each level stores features independently
- No values flow across levels; only structural correspondence
- No cross-level weighted propagation

## File Organization

### Core Algorithm Modules

| File | Function | Key Principles |
|------|----------|----------------|
| `minimal_principle.py` | Minimal working example | Down/up mapping + passive/active + max fusion |
| `group_hierarchical.py` | Full group-theoretic classifier | Transformation group + bidirectional mapping + primitive isolation |
| `atomic_contrast.py` | Atomic contrast point matching | Local extrema + Jaccard set similarity + multi-scale |
| `local_knn.py` | Local k-NN learner | Independent prototypes + zero-forgetting + local mean prediction |
| `memory.py` | Open memory system | Protocol-based pluggable backends + group theory structure |

### Domain Demos

| File | Domain | Input Type |
|------|--------|------------|
| `demo_image.py` | Image classification | 2D grayscale tensor (tkinter load) |
| `demo_text.py` | Text classification | 1D sequence tensor |
| `demo_speech.py` | Speech classification | WAV -> spectrogram 2D tensor |
| `demo_generation.py` | Image generation | Structural inverse propagation |
| `demo_reasoning.py` | Reasoning engine | Top-down hypothesis testing |

### Test & Analysis

| File | Purpose |
|------|---------|
| `test_multi_domain.py` | Universality test across 4 domains (image/text/timeseries/spectrogram) |
| `test_mnist.py` | MNIST training test (synthetic or real data) |
| `heatmap_analysis.py` | Visualize and analyze feature maps at each level |

### Training Examples

| File | Purpose |
|------|---------|
| `train_real.py` | Small synthetic digit training |
| `train_llm.py` | Text sequence training example |

## Usage

```bash
# Run any demo
python demo_image.py      # Image classification
python demo_text.py       # Text classification
python demo_speech.py     # Speech classification
python demo_generation.py # Image generation (structural inverse)
python demo_reasoning.py  # Reasoning engine demo

# Run tests
python test_multi_domain.py   # Multi-domain universality test
python test_mnist.py          # MNIST test
python heatmap_analysis.py   # Heatmap analysis

# Run core principle examples
python minimal_principle.py   # Minimal working example
python group_hierarchical.py  # Full classifier demo
python atomic_contrast.py     # Atomic contrast matching
python local_knn.py           # Local k-NN learner
python memory.py              # Open memory system demo
```

## Data Format

All processing code accepts plain tensors (list of lists), independent of file format:

- **Image**: 2D list [[gray, ...], ...]
- **Text**: 1D sequence list [float, ...]
- **Speech**: WAV -> spectrogram as 2D list

Data loading functions are separate from processing logic.

## Validated Configuration (MNIST)

- Pool factors: [1, 2, 4]
- Active + passive channels, max fusion at decision
- Similarity: cosine K=1 nearest neighbor
- Accuracy: ~95% (synthetic), ~97% (real MNIST)

## Directions Verified to Fail

- Active + passive average fusion (dilutes signal)
- Cross-primitive weighted operations
- Transposed propagation (deep-learning residual trick)
- Strict top-down propagation
- Global statistics / global thresholds
- Hardcoded level semantics
