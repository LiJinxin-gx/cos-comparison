# Continuous Mapping Hierarchical Isolation — Principle Reflection & Design Notes

> Records the iterative reflection and final design of tensor-domain hierarchical isolation for the "structure-first general model", to prevent future deviation from core principles.

## 1. Core Principles (Reference Baseline)

1. **Continuous mapping hierarchical isolation A↔B**: layers are continuous mapping transformations (group-theoretic structure), not limited to pooling
2. **Information arises from local comparison**: passive/active compare within local windows
3. **Structure-first**: structure before detail
4. **Unsupervised / no priors / no deep learning**: no weights, no gradients, no probabilities, no hardcoded thresholds
5. **Absolute locality**: local independence, local → larger local aggregation
6. **Discrete symbols → tensor domain → unified general method**: symbols fold into tensors, passive/active is consistent across all modalities
7. **Inference chain as data**: emerges from data, propagates between layers
8. **No recursion**

## 2. Repeated Deviation Errors (Confirmed)

| Error | Symptom |
|---|---|
| Counting mapping maps as scalars | Passive outputs similarity maps, but they were counted as "number of edges" |
| Operating on flag index streams | UnitMap position indices have no structural meaning |
| Text n-gram masquerading as general method | Bigram/trigram frequency statistics repackaged as "hierarchical mapping" |
| Hardcoded rules | <0.75 threshold, AGG=4, low-density repeat detection |
| Global pooling | Cross-segment Counter aggregation |

**Root cause of repeated deviation**: text n-gram is simple, runs, and produces output; tensor-domain mapping hierarchy is complex. Each iteration chose the simple option to deliver something, rebranded, and ran again — essentially all n-gram lookup tables.

## 3. Final Understanding: Hierarchical Isolation = Different Primitives per Layer Extract Different Information

"Isolation" does not mean each layer does the same operation — it means **each layer uses a different primitive to extract a different kind of information**:

| Layer | Primitive | Information Extracted |
|---|---|---|
| L0 | Raw tensor | — |
| L1 | **Passive mode** | **Differences / edges** |
| L2 | **Active mode** | **Same points / templates** |

- L1 passively extracts **differences** (edge map, position-wise similarity)
- L2 actively matches **same points** (repeated templates) on the L1 edge map
- **Never reduce mapping maps to scalars**
- **No cross-primitive numerical operations** (do not weight passive and active results together)

## 4. v90/v91 Validation

- v90: all layers use passive → higher layers saturate to 1 (step-signal same-value cos ≈ 1), information lost
- v91: L1 passive (differences) + L2 active (same points) → hierarchical isolation holds

## 5. Next Directions

1. L2 kernels should select **repeated fragments from L1** (not fragments from L1 itself)
2. Inference engine = L1↔L2 mapping-guided composition to generate new data
3. Validate on real continuous data (image gradients / large corpora), not 0/1 step signals
