# Structural-Inverse Perception-Generation Loop (PCML v40-v50)

An end-to-end loop built **only from the project-native components**
(`TensorReceptor`, `Memory`, `TensorGenerator`, `UnitMap`): perception ->
hierarchical fold -> asymmetric memory -> structural-inverse generation.
Validated on English literary text, classical Chinese text and modern Chinese
prose (tokenized as one unit per symbol for Chinese; no external preprocessing).

## Enforced core principles

- Continuous-mapping hierarchical isolation: layer relation is a two-way
  mapping A<->B (group closure, identity, structural inverse - not exact inverse).
- Information arises from **local comparison**; local-first, a global is just a
  special local.
- Active / passive primitives kept strictly isolated (no cross-primitive
  weighting or blending).
- Unsupervised, no priors, no hard-coded layer count, no deep-learning
  mechanisms (no weights, gradient, residual, argmax, global temperature,
  probability, or numeric fingerprint).
- Deterministic and iterative (no recursion).

## Native pipeline

```
data -> UnitMap flags (symbol -> tensor domain)
     -> TensorReceptor.passive        (local adjacent comparison)
     -> threshold_map                 (continuous -> interval symbols)
     -> run-fold                      (repeated runs -> atoms, (symbol,length))
     -> asymmetric Memory             (low: run details; high: symbol sequence)
     -> TensorGenerator structural inverse (B -> A)
```

## Results

| Stage | Version | Result |
|-------|---------|--------|
| Native hierarchy | v40 | passive distribution healthy (min 0.004 / p50 0.45 / p95 0.83); run-fold compresses 3991 -> 193 (~25.7x) |
| Training loop | v41 | asymmetric memory stored; **exact structural inverse OK** (rebuilt == original) |
| Local identify | v43 | local-template slide: 30 hits with positions (cos 0.83-0.95) |
| Structural-inverse generate | v44 | skeleton (X rhythm) + memory flesh (blocks from other segments) |
| Token emergence | v47 | high-frequency repeated char-groups auto-fold into tokens |
| Reasoning engine | v50 | f<->inverse exact; perturb one high-level run -> low-level cells respond exactly |

## Token emergence (original-layer frequency fold)

`UnitMap.window_units(data, local_size=2)` turns the stream into bigram window
units; `most_common()` ranks high-frequency windows. A repeatedly appearing
char-group (e.g. a real word) emerges as ONE high-frequency unit. This is
**native UnitMap frequency folding at the original layer**, not BPE greedy merge.

```
count=3   'sound'  'new_'
count=2   'cross'  'ground'  'distant'  'behind'
count=1   'dawn'   'street'            <- low-frequency noise
```

## Key findings

- The structural inverse is **exact and deterministic**: run-fold and expand
  are mutual inverses (f^-1(f(S)) == S).
- Identification is a **local template slide** (query's local rhythm fragment
  slides over targets and reports hit positions), not a global cosine on a
  flattened vector.
- Frequency folding belongs at the **original data layer** (window units +
  count/most_common), NOT on the passive-symbol layer (passive = difference,
  frequency = repetition; the two primitives must not be mixed).
- A high-level perturbation drives low-level output exactly - the high<->low
  closed loop is structural, not statistical.

## Empirical boundaries

- A single-layer run rhythm is a **length rhythm**, not semantics; one-shot
  structural-inverse text is not yet coherent.
- Small corpora leave some rhythm keys without instances (filler markers).
- Cross-segment generation coherence needs larger corpus coverage.

## Code

Exploration scripts live alongside the project exploration directory; the loop
is reproducible from the native components listed above (no external learner).
