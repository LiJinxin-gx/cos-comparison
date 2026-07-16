# Seven-Layer Cognitive Architecture

Cos-comparison adopts a **seven-layer brain-inspired cognitive architecture**, where each layer corresponds to different levels of biological cognitive processing. This layered design enables modular development and clear separation of concerns.

## Overview

| Layer | Directory | Brain Analogue | Maturity | Core Function |
|-------|-----------|----------------|----------|---------------|
| 1 | `core` | Brainstem / Cerebellum | ✅ Mature | Low-level local comparison computation |
| 2 | `sense_layer` | Sensory Cortex | 🔴 Skeleton | Receive external stimuli, extract raw features |
| 3 | `memory_layer` | Hippocampus / Cortex | 🔴 Skeleton | Store short-term and long-term memory |
| 4 | `brain_layer` | Prefrontal Cortex | 🟡 Partial | Higher cognition, logic, decision making |
| 5 | `action_layer` | Motor Cortex | 🔴 Skeleton | Control action output, interact with environment |
| 6 | `generate_layer` | Broca's / Wernicke's Area | 🔴 Skeleton | Generate language, images, high-level output |
| 7 | `expend_layer` | Association Areas | 🔴 Skeleton | Extended functions and special capabilities |

## Design Principles

### 1. Bottom-Up Dependency

Each layer only depends on layers below it, never on layers above.

```
Layer 7 (Extension)
    ↑ depends on
Layer 6 (Generation)
    ↑ depends on
Layer 5 (Action)
    ↑ depends on
Layer 4 (Brain)
    ↑ depends on
Layer 3 (Memory)
    ↑ depends on
Layer 2 (Sense)
    ↑ depends on
Layer 1 (Core)
```

This ensures:
- Clean separation of concerns
- Each layer can be tested independently
- Lower layers can be optimized without affecting higher layers
- The system remains functional even if higher layers are incomplete

### 2. Neuroscience Homology

Each layer has a clear correspondence with biological brain structures:

- **Core (Layer 1)**: Analogous to brainstem and cerebellum — low-level, fast, automatic processing
- **Sense (Layer 2)**: Analogous to sensory cortex — initial feature extraction from raw input
- **Memory (Layer 3)**: Analogous to hippocampus and association cortex — storage and retrieval of information
- **Brain (Layer 4)**: Analogous to prefrontal cortex — reasoning, planning, decision making
- **Action (Layer 5)**: Analogous to motor cortex — execution of actions
- **Generate (Layer 6)**: Analogous to language areas — generation of complex outputs
- **Expend (Layer 7)**: Analogous to association areas — integration and special functions

### 3. Dual-Mode Consistency

The passive/active dual-mode philosophy extends vertically through all layers:

| Layer | Passive Mode | Active Mode |
|-------|-------------|-------------|
| Core | Self-similarity comparison | Template matching |
| Memory | Content-based retrieval | Pattern-based search |
| Brain | Bottom-up reasoning | Top-down planning |
| ... | ... | ... |

This consistent design philosophy creates architectural coherence.

## Layer Details

### Layer 1: Core

**Status**: ✅ Mature

**Responsibilities**:
- Fundamental local comparison computation
- Passive mode: sliding window self-similarity
- Active mode: template matching
- Statistical operations: mean, variance
- Multiple backend implementations

**Key Features**:
- Dimension-agnostic (1D, 2D, 3D, 4D, ...)
- Multiple similarity algorithms (cos, mod, cosmod)
- Zero-dependency pure Python reference
- Three acceleration backends with automatic fallback: Python C extension, ctypes pure C, pure Python
- Full free-threaded Python (CPython 3.13+) support, runs without GIL for true parallelism
- Duck-typing support for NumPy/PyTorch arrays without dedicated backend
- Callback system for extensibility

### Layer 2: Sense Layer

**Status**: 🔴 Skeleton (planned)

**Responsibilities**:
- Input data normalization and preprocessing
- Multi-sensory integration
- Basic feature extraction pipelines
- Sensory adaptation mechanisms

**Design Ideas**:
- Standardized input interfaces for different data types
- Automatic dimension detection and handling
- Sensory attention mechanisms (spatial/temporal)
- Adaptation to input statistics

### Layer 3: Memory Layer

**Status**: 🔴 Skeleton (planned)

**Responsibilities**:
- Short-term memory (working memory)
- Long-term memory storage
- Memory consolidation
- Pattern-based memory retrieval
- Dual-mode database (passive + active)

**Design Ideas**:
- Key-value store abstraction
- Vector similarity search
- Memory decay and reinforcement mechanisms
- Episodic vs semantic memory distinction
- Database backend pluggability

### Layer 4: Brain Layer

**Status**: 🟡 Partial (symbol_logic implemented)

**Responsibilities**:
- Symbolic logic and reasoning
- Decision making
- Goal management
- Planning and problem solving
- Metacognition

**Current Implementation**:
- `Logic` flag enum: three-valued logic system (TRUE / SURE / uncertain) supporting confidence levels
- `Atomic_proposition`: structured atomic proposition representation (subject-verb-object structure with modifiers and limits)
- `Logic_bind`: implication/binding between propositions, supporting conditional reasoning
- `Logic_context`: knowledge context management, supporting proposition addition and retrieval
- Foundation for symbolic knowledge representation and rule-based reasoning

**Design Ideas**:
- Three-valued logic system (addressing hallucination problem)
- Rule-based reasoning engine
- Goal hierarchy management
- Planning with subgoals
- Confidence estimation

### Layer 5: Action Layer

**Status**: 🔴 Skeleton (planned)

**Responsibilities**:
- Action selection and execution
- Motor control
- Environment interaction
- Action outcome evaluation
- Reflex arcs

**Design Ideas**:
- Action primitive library
- Action sequencing
- Feedback-based action adjustment
- Exploration vs exploitation balance
- Innate vs learned reflexes

### Layer 6: Generate Layer

**Status**: 🔴 Skeleton (planned)

**Responsibilities**:
- Language generation
- Image/signal generation
- Output formatting
- Creative generation
- Expression of internal states

**Design Ideas**:
- Template-based generation
- Compositional generation
- Style control
- Multi-modal output generation

### Layer 7: Expend Layer

**Status**: 🔴 Skeleton (planned)

**Responsibilities**:
- Specialized capabilities
- Cross-modal integration
- Advanced cognitive functions
- Plugin system for extensions

**Design Ideas**:
- Plugin architecture
- Cross-modal association
- Meta-learning capabilities
- Social cognition modules

## Current State Assessment

### Strengths
- Clear architectural vision with strong neuroscience inspiration
- Well-designed core layer with solid implementation
- Clean separation of concerns and dependency direction
- Dual-mode philosophy provides conceptual coherence

### Gaps
- Only core layer is truly functional
- Brain layer has minimal implementation
- All other layers are empty skeletons
- No clear roadmap for layer-by-layer development

### Recommendations
1. **Prioritize memory layer** as the next step — it's the bridge from perception to cognition
2. **Flesh out brain layer** with more reasoning capabilities
3. **Define clear interfaces** between layers before extensive implementation
4. **Build vertical demos** showing multiple layers working together

---

**Next**: [Backend Management System](backend-system.md) — Multi-backend dynamic loading
